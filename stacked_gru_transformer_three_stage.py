from __future__ import annotations

import argparse
import csv
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from classifykat import EarlyStopping, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level
from stacked_gru_transformer import (
    GRUAttentionEncoder,
    MultiEdgeGATFusion,
    batch_edge_index,
    build_classifier,
    classification_metrics_per_channel,
)


class StackedGRUThreeStagePredictor(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        gru_dim: int = 64,
        gru_layers: int = 2,
        gru_heads: int = 4,
        gat_hidden: int = 64,
        gat_heads: int = 2,
        classifier_name: str = "TSiTPlus",
        dropout: float = 0.15,
        chunk_size: int = 200,
    ) -> None:
        super().__init__()
        self.c_out = c_out
        self.chunk_size = chunk_size
        self.feature_dim = gru_dim + gat_hidden

        self.encoder = GRUAttentionEncoder(
            c_in=c_in,
            gru_dim=gru_dim,
            n_layers=gru_layers,
            n_heads=gru_heads,
            dropout=dropout,
        )
        self.gat = MultiEdgeGATFusion(
            in_channels=gru_dim,
            hidden_channels=gat_hidden,
            heads=gat_heads,
        )
        self.classifier = build_classifier(
            classifier_name,
            c_in=self.feature_dim,
            c_out=c_out,
            seq_len=seq_len,
        )

        self.regressor_delayed = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, c_out),
        )
        self.regressor_nondelayed = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, c_out),
        )

    def _extract_pooled_features(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_nodes, c_in, seq_len = x.shape
        total = bsz * n_nodes

        x_flat = x.view(total, c_in, seq_len)
        gru_seq = self.encoder(x_flat)
        gru_pooled = gru_seq.mean(dim=2)

        gat_out = self.gat(gru_pooled, edge_index_adj, edge_index_od, edge_index_od_t)
        gat_broadcast = gat_out.unsqueeze(2).expand(-1, -1, seq_len)
        enriched = torch.cat([gru_seq, gat_broadcast], dim=1)
        pooled = enriched.mean(dim=2)
        return pooled.view(bsz, n_nodes, self.feature_dim)

    def forward_classifier(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_nodes, _, _ = x.shape
        total = bsz * n_nodes

        x_flat = x.view(total, x.size(2), x.size(3))
        gru_seq = self.encoder(x_flat)
        gru_pooled = gru_seq.mean(dim=2)
        gat_out = self.gat(gru_pooled, edge_index_adj, edge_index_od, edge_index_od_t)
        gat_broadcast = gat_out.unsqueeze(2).expand(-1, -1, x.size(3))
        enriched = torch.cat([gru_seq, gat_broadcast], dim=1)

        logits = self.classifier(enriched)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        if logits.dim() > 2:
            logits = logits.flatten(1)
        return logits.view(bsz, n_nodes, self.c_out)

    def forward_regressor(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
        *,
        which: str = "delayed",
    ) -> torch.Tensor:
        pooled = self._extract_pooled_features(x, edge_index_adj, edge_index_od, edge_index_od_t)
        if which == "delayed":
            return self.regressor_delayed(pooled)
        return self.regressor_nondelayed(pooled)


def reshape_for_graph(
    x: torch.Tensor,
    y_reg: torch.Tensor,
    y_cls: torch.Tensor,
    seq_len: int,
    feature_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, n_nodes, _ = x.shape
    xg = x.view(n, n_nodes, seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    return xg, y_reg, y_cls


def _set_requires_grad(module: nn.Module, value: bool) -> None:
    for p in module.parameters():
        p.requires_grad = value


def _channel_stats(mask: torch.Tensor) -> float:
    return float(mask.float().mean().item()) if mask.numel() > 0 else 0.0


def train_stage1(
    model: StackedGRUThreeStagePredictor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    class_threshold: float,
) -> Tuple[List[Dict], float]:
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STAGE 1: CLASSIFIER (STACKED GRU + GAT + TRANSFORMER)")
    print("=" * 80)

    _set_requires_grad(model.encoder, True)
    _set_requires_grad(model.gat, True)
    _set_requires_grad(model.classifier, True)
    _set_requires_grad(model.regressor_delayed, False)
    _set_requires_grad(model.regressor_nondelayed, False)

    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = 1e-4

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    early_stopping = EarlyStopping(patience=patience, mode="max")

    history: List[Dict] = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_losses: List[float] = []

        for bx, by_reg, by_cls in train_loader:
            bx = bx.to(device, non_blocking=True)
            by_cls = by_cls.to(device, non_blocking=True)
            bsz = bx.size(0)

            bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
            bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
            bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_classifier(bx, bei_adj, bei_od, bei_od_t)
            loss = loss_fn(logits, by_cls)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for bx, _, by_cls in val_loader:
                bx = bx.to(device, non_blocking=True)
                bsz = bx.size(0)
                bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
                bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
                bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)
                logits = model.forward_classifier(bx, bei_adj, bei_od, bei_od_t)
                val_probs.append(torch.sigmoid(logits).cpu())
                val_targets.append(by_cls)

        vm = classification_metrics_per_channel(
            torch.cat(val_probs).numpy(),
            torch.cat(val_targets).numpy(),
            threshold=class_threshold,
        )

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()},
                "gat": {k: v.cpu().clone() for k, v in model.gat.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
            }

        ep_sec = time.time() - ep_t0
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history.append(
            {
                "stage": 1,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": vm["f1"],
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
                "val_accuracy": vm["accuracy"],
                "epoch_time_seconds": ep_sec,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} | val_f1={vm['f1']:.4f} "
            f"(arr={vm['f1_arrival']:.4f}, dep={vm['f1_departure']:.4f}) | sec={ep_sec:.1f}"
        )

        if early_stopping(vm["f1"], epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.encoder.load_state_dict(best_state["encoder"])
        model.gat.load_state_dict(best_state["gat"])
        model.classifier.load_state_dict(best_state["classifier"])

    _set_requires_grad(model.regressor_delayed, True)
    _set_requires_grad(model.regressor_nondelayed, True)

    return history, time.time() - t0


def train_stage2(
    model: StackedGRUThreeStagePredictor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    delay_threshold: float,
    patience: int,
) -> Tuple[List[Dict], float]:
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STAGE 2: DELAYED REGRESSOR")
    print("=" * 80)

    _set_requires_grad(model.encoder, False)
    _set_requires_grad(model.gat, False)
    _set_requires_grad(model.classifier, False)
    _set_requires_grad(model.regressor_delayed, True)
    _set_requires_grad(model.regressor_nondelayed, False)

    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = 1e-4

    huber = nn.HuberLoss(reduction="none", delta=2.0)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32), device=device)
        std_t = torch.tensor(np.array(scaler.std, dtype=np.float32), device=device)
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
        thr_scaled = (torch.full_like(mean_t, float(delay_threshold)) - mean_t) / std_t
    else:
        thr_scaled = torch.tensor([float(delay_threshold)], device=device)

    def masked_loss(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        thr = thr_scaled
        if thr.numel() == 1 and targets.shape[-1] > 1:
            thr = thr.expand(targets.shape[-1])
        mask = (targets > thr).float()
        per = huber(preds, targets) * mask
        denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
        loss_ch = per.sum(dim=(0, 1)) / denom
        return loss_ch.mean(), mask

    history: List[Dict] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_masks: List[float] = []

        for bx, by_reg, _ in train_loader:
            bx = bx.to(device, non_blocking=True)
            by_reg = by_reg.to(device, non_blocking=True)
            bsz = bx.size(0)

            bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
            bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
            bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)

            optimizer.zero_grad(set_to_none=True)
            preds = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="delayed")
            loss, mask = masked_loss(preds, by_reg)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(float(loss.item()))
            tr_masks.append(_channel_stats(mask))

        model.eval()
        va_losses: List[float] = []
        va_masks: List[float] = []
        with torch.no_grad():
            for bx, by_reg, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                by_reg = by_reg.to(device, non_blocking=True)
                bsz = bx.size(0)
                bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
                bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
                bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)
                preds = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="delayed")
                loss, mask = masked_loss(preds, by_reg)
                va_losses.append(float(loss.item()))
                va_masks.append(_channel_stats(mask))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        tr_mask = float(np.mean(tr_masks)) if tr_masks else 0.0
        va_mask = float(np.mean(va_masks)) if va_masks else 0.0

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.regressor_delayed.state_dict().items()}

        ep_sec = time.time() - ep_t0
        history.append(
            {
                "stage": 2,
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_mask_ratio": tr_mask,
                "val_loss": va_loss,
                "val_mask_ratio": va_mask,
                "epoch_time_seconds": ep_sec,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"mask={tr_mask*100:.1f}% (val {va_mask*100:.1f}%) | sec={ep_sec:.1f}"
        )

        if early_stopping(va_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.regressor_delayed.load_state_dict(best_state)

    for p in model.parameters():
        p.requires_grad = True

    return history, time.time() - t0


def train_stage3(
    model: StackedGRUThreeStagePredictor,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    delay_threshold: float,
    patience: int,
) -> Tuple[List[Dict], float]:
    t0 = time.time()
    print("\n" + "=" * 80)
    print("STAGE 3: NON-DELAYED REGRESSOR")
    print("=" * 80)

    _set_requires_grad(model.encoder, False)
    _set_requires_grad(model.gat, False)
    _set_requires_grad(model.classifier, False)
    _set_requires_grad(model.regressor_delayed, False)
    _set_requires_grad(model.regressor_nondelayed, True)

    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = 1e-5

    huber = nn.HuberLoss(reduction="none", delta=1.0)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32), device=device)
        std_t = torch.tensor(np.array(scaler.std, dtype=np.float32), device=device)
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
    else:
        mean_t = None
        std_t = None

    def nondelayed_mask(targets_scaled: torch.Tensor) -> torch.Tensor:
        if mean_t is None or std_t is None:
            targets_denorm = targets_scaled
        else:
            targets_denorm = targets_scaled * std_t + mean_t
        return (targets_denorm.abs() < float(delay_threshold)).float()

    history: List[Dict] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_masks: List[float] = []

        for bx, by_reg, _ in train_loader:
            bx = bx.to(device, non_blocking=True)
            by_reg = by_reg.to(device, non_blocking=True)
            bsz = bx.size(0)

            mask = nondelayed_mask(by_reg)
            if float(mask.sum().item()) <= 0.0:
                continue

            bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
            bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
            bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)

            optimizer.zero_grad(set_to_none=True)
            preds = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="nondelayed")
            per = huber(preds, by_reg) * mask
            denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
            loss_ch = per.sum(dim=(0, 1)) / denom
            loss = loss_ch.mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            tr_losses.append(float(loss.item()))
            tr_masks.append(_channel_stats(mask))

        model.eval()
        va_losses: List[float] = []
        va_masks: List[float] = []
        with torch.no_grad():
            for bx, by_reg, _ in val_loader:
                bx = bx.to(device, non_blocking=True)
                by_reg = by_reg.to(device, non_blocking=True)
                bsz = bx.size(0)
                mask = nondelayed_mask(by_reg)
                if float(mask.sum().item()) <= 0.0:
                    continue
                bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
                bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
                bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)
                preds = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="nondelayed")
                per = huber(preds, by_reg) * mask
                denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
                loss_ch = per.sum(dim=(0, 1)) / denom
                loss = loss_ch.mean()
                va_losses.append(float(loss.item()))
                va_masks.append(_channel_stats(mask))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        tr_mask = float(np.mean(tr_masks)) if tr_masks else 0.0
        va_mask = float(np.mean(va_masks)) if va_masks else 0.0

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.regressor_nondelayed.state_dict().items()}

        ep_sec = time.time() - ep_t0
        history.append(
            {
                "stage": 3,
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_mask_ratio": tr_mask,
                "val_loss": va_loss,
                "val_mask_ratio": va_mask,
                "epoch_time_seconds": ep_sec,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"mask={tr_mask*100:.1f}% (val {va_mask*100:.1f}%) | sec={ep_sec:.1f}"
        )

        if early_stopping(va_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.regressor_nondelayed.load_state_dict(best_state)

    for p in model.parameters():
        p.requires_grad = True

    return history, time.time() - t0


def final_evaluation(
    model: StackedGRUThreeStagePredictor,
    test_loader: DataLoader,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    scaler,
    class_threshold: float,
    delay_threshold: float,
    out_dir: str,
    history: List[Dict],
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
) -> None:
    model.eval()
    cls_probs, cls_targets = [], []
    reg_preds_delayed, reg_preds_nondelayed, reg_targets = [], [], []

    with torch.no_grad():
        for bx, by_reg, by_cls in test_loader:
            bx = bx.to(device, non_blocking=True)
            by_reg = by_reg.to(device, non_blocking=True)
            bsz = bx.size(0)
            bei_adj = batch_edge_index(edge_index_adj, n_nodes, bsz)
            bei_od = batch_edge_index(edge_index_od, n_nodes, bsz)
            bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, bsz)

            logits = model.forward_classifier(bx, bei_adj, bei_od, bei_od_t)
            pred_delayed = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="delayed")
            pred_nondelayed = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="nondelayed")

            cls_probs.append(torch.sigmoid(logits).cpu())
            cls_targets.append(by_cls)
            reg_preds_delayed.append(pred_delayed.cpu())
            reg_preds_nondelayed.append(pred_nondelayed.cpu())
            reg_targets.append(by_reg.cpu())

    cls_probs_np = torch.cat(cls_probs).numpy()
    cls_targets_np = torch.cat(cls_targets).numpy()
    test_cls = classification_metrics_per_channel(cls_probs_np, cls_targets_np, threshold=class_threshold)

    pred_delayed_np = torch.cat(reg_preds_delayed).numpy()
    pred_nondelayed_np = torch.cat(reg_preds_nondelayed).numpy()
    reg_targets_np = torch.cat(reg_targets).numpy()

    route_mask = cls_probs_np >= class_threshold
    reg_pred_np = np.where(route_mask, pred_delayed_np, pred_nondelayed_np)

    if scaler is not None and hasattr(scaler, "inverse_transform"):
        preds_2d = reg_pred_np.reshape(-1, reg_pred_np.shape[-1])
        tars_2d = reg_targets_np.reshape(-1, reg_targets_np.shape[-1])
        preds_denorm = scaler.inverse_transform(preds_2d).reshape(reg_pred_np.shape)
        tars_denorm = scaler.inverse_transform(tars_2d).reshape(reg_targets_np.shape)
    else:
        preds_denorm = reg_pred_np
        tars_denorm = reg_targets_np

    preds_denorm = np.maximum(0.0, preds_denorm)
    tars_denorm = np.maximum(0.0, tars_denorm)

    pred_flat = preds_denorm.reshape(-1)
    tar_flat = tars_denorm.reshape(-1)

    delayed_mask = tar_flat > delay_threshold
    nondelayed_mask = ~delayed_mask

    def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        if y_true.size == 0:
            return 0.0, 0.0
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        return mae, rmse

    mae_delayed, rmse_delayed = _mae_rmse(tar_flat[delayed_mask], pred_flat[delayed_mask])
    mae_nd, rmse_nd = _mae_rmse(tar_flat[nondelayed_mask], pred_flat[nondelayed_mask])
    mae_all, rmse_all = _mae_rmse(tar_flat, pred_flat)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    print(f"Classification F1: {test_cls['f1']:.4f} (arr={test_cls['f1_arrival']:.4f}, dep={test_cls['f1_departure']:.4f})")
    print(f"Regression delayed>thr: MAE={mae_delayed:.4f}, RMSE={rmse_delayed:.4f}")
    print(f"Regression non-delayed: MAE={mae_nd:.4f}, RMSE={rmse_nd:.4f}")
    print(f"Regression overall: MAE={mae_all:.4f}, RMSE={rmse_all:.4f}")

    model_path = os.path.join(out_dir, "stacked_gru_three_stage_best.pth")
    torch.save(model.state_dict(), model_path)

    hist_path = os.path.join(out_dir, "three_stage_history.csv")
    with open(hist_path, "w", newline="", encoding="utf-8") as f:
        if history:
            keys = sorted({k for row in history for k in row.keys()})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(history)

    metrics_path = os.path.join(out_dir, "three_stage_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["classification_f1_macro", f"{test_cls['f1']:.6f}"])
        w.writerow(["classification_f1_arrival", f"{test_cls['f1_arrival']:.6f}"])
        w.writerow(["classification_f1_departure", f"{test_cls['f1_departure']:.6f}"])
        w.writerow(["regression_mae_delayed", f"{mae_delayed:.6f}"])
        w.writerow(["regression_rmse_delayed", f"{rmse_delayed:.6f}"])
        w.writerow(["regression_mae_nondelayed", f"{mae_nd:.6f}"])
        w.writerow(["regression_rmse_nondelayed", f"{rmse_nd:.6f}"])
        w.writerow(["regression_mae_overall", f"{mae_all:.6f}"])
        w.writerow(["regression_rmse_overall", f"{rmse_all:.6f}"])
        w.writerow(["stage1_time_seconds", f"{stage1_time:.2f}"])
        w.writerow(["stage2_time_seconds", f"{stage2_time:.2f}"])
        w.writerow(["stage3_time_seconds", f"{stage3_time:.2f}"])
        w.writerow(["total_time_seconds", f"{(stage1_time + stage2_time + stage3_time):.2f}"])

    print(f"Saved: {model_path}")
    print(f"Saved: {hist_path}")
    print(f"Saved: {metrics_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Three-stage stacked GRU+GAT+Transformer predictor")
    p.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    p.add_argument("--weather_file", type=str, default="weather_cn.npy")
    p.add_argument("--period_hours", type=int, default=24)

    p.add_argument("--seq_len", type=int, default=18)
    p.add_argument("--horizons", type=int, nargs=1, default=[12], choices=[3, 6, 12, 24])
    p.add_argument("--delay_threshold", type=float, default=5.0)
    p.add_argument("--class_threshold", type=float, default=0.5)

    p.add_argument("--gru_dim", type=int, default=64)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--gru_heads", type=int, default=4)
    p.add_argument("--gat_hidden", type=int, default=64)
    p.add_argument("--gat_heads", type=int, default=2)
    p.add_argument("--classifier", type=str, default="TSiTPlus", choices=["TSiTPlus", "ConvTranPlus"])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--chunk_size", type=int, default=200)

    p.add_argument("--stage1_epochs", type=int, default=10)
    p.add_argument("--stage2_epochs", type=int, default=10)
    p.add_argument("--stage3_epochs", type=int, default=14)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--stage1_lr", type=float, default=None)
    p.add_argument("--stage2_lr", type=float, default=None)
    p.add_argument("--stage3_lr", type=float, default=None)
    p.add_argument("--patience", type=int, default=5)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=-1, help="-1 auto (Windows=0, else min(8,cpu_count))")
    p.add_argument("--output_dir", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    print("\n" + "=" * 80)
    print("THREE-STAGE STACKED GRU + TRANSFORMER TRAINING")
    print("=" * 80)
    print(f"Device: {device}")

    (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_inputs,
        val_inputs,
        test_inputs,
        train_delay_scaled,
        val_delay_scaled,
        test_delay_scaled,
        train_raw,
        val_raw,
        test_raw,
        scaler,
        _,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

    train_inputs = train_inputs[:, :, :-2]
    val_inputs = val_inputs[:, :, :-2]
    test_inputs = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    horizon = int(args.horizons[0])

    train_x, train_y_reg, train_y_cls = build_sequences_node_level(
        train_inputs,
        train_delay_scaled,
        train_raw,
        args.seq_len,
        horizon,
        args.delay_threshold,
        args.horizons,
    )
    val_x, val_y_reg, val_y_cls = build_sequences_node_level(
        val_inputs,
        val_delay_scaled,
        val_raw,
        args.seq_len,
        horizon,
        args.delay_threshold,
        args.horizons,
    )
    test_x, test_y_reg, test_y_cls = build_sequences_node_level(
        test_inputs,
        test_delay_scaled,
        test_raw,
        args.seq_len,
        horizon,
        args.delay_threshold,
        args.horizons,
    )

    trX, trY_reg, trY_cls = reshape_for_graph(train_x, train_y_reg, train_y_cls, args.seq_len, feature_dim)
    vaX, vaY_reg, vaY_cls = reshape_for_graph(val_x, val_y_reg, val_y_cls, args.seq_len, feature_dim)
    teX, teY_reg, teY_cls = reshape_for_graph(test_x, test_y_reg, test_y_cls, args.seq_len, feature_dim)

    trY_reg = trY_reg.float()
    vaY_reg = vaY_reg.float()
    teY_reg = teY_reg.float()
    trY_cls = trY_cls.float()
    vaY_cls = vaY_cls.float()
    teY_cls = teY_cls.float()

    n_nodes = trX.shape[1]

    if args.num_workers >= 0:
        worker_count = int(args.num_workers)
    else:
        worker_count = 0 if os.name == "nt" else min(8, os.cpu_count() or 0)

    loader_kwargs = dict(
        num_workers=worker_count,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(worker_count > 0),
    )
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        TensorDataset(trX, trY_reg, trY_cls),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TensorDataset(vaX, vaY_reg, vaY_cls),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TensorDataset(teX, teY_reg, teY_cls),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    pos_rate = trY_cls.reshape(-1, delay_dim).mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    model = StackedGRUThreeStagePredictor(
        c_in=feature_dim,
        c_out=delay_dim,
        seq_len=args.seq_len,
        gru_dim=args.gru_dim,
        gru_layers=args.gru_layers,
        gru_heads=args.gru_heads,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        classifier_name=args.classifier,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    s1_lr = args.stage1_lr if args.stage1_lr is not None else args.lr
    s2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
    s3_lr = args.stage3_lr if args.stage3_lr is not None else (args.lr * 0.2)

    ei_adj = edge_index_adj.to(device)
    ei_od = edge_index_od.to(device)
    ei_od_t = edge_index_od_t.to(device)

    h1, t1 = train_stage1(
        model,
        optimizer,
        train_loader,
        val_loader,
        ei_adj,
        ei_od,
        ei_od_t,
        n_nodes,
        device,
        args.stage1_epochs,
        s1_lr,
        pos_weight,
        args.patience,
        args.class_threshold,
    )

    h2, t2 = train_stage2(
        model,
        optimizer,
        train_loader,
        val_loader,
        ei_adj,
        ei_od,
        ei_od_t,
        n_nodes,
        device,
        args.stage2_epochs,
        s2_lr,
        scaler,
        args.delay_threshold,
        args.patience,
    )

    h3, t3 = train_stage3(
        model,
        optimizer,
        train_loader,
        val_loader,
        ei_adj,
        ei_od,
        ei_od_t,
        n_nodes,
        device,
        args.stage3_epochs,
        s3_lr,
        scaler,
        args.delay_threshold,
        args.patience,
    )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir if args.output_dir != "auto" else f"stacked_gru_three_stage_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    final_evaluation(
        model,
        test_loader,
        ei_adj,
        ei_od,
        ei_od_t,
        n_nodes,
        device,
        scaler,
        args.class_threshold,
        args.delay_threshold,
        out_dir,
        h1 + h2 + h3,
        t1,
        t2,
        t3,
    )


if __name__ == "__main__":
    main()
