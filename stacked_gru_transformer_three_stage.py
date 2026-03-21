from __future__ import annotations

import argparse
import csv
import math
import os
import time
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast

from classifykat import EarlyStopping, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level
from stacked_gru_transformer import (
    GRUAttentionEncoder,
    MultiEdgeGATFusion,
    batch_edge_index,
    build_classifier,
    classification_metrics_per_channel,
)


class ResidualRegressor(nn.Module):
    def __init__(self, dim: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.fc3 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_norm(x)
        r = h
        h = self.drop(self.act(self.fc1(h)))
        h = self.drop(self.act(self.fc2(h)))
        h = h + r
        h = self.drop(self.act(self.fc3(h)))
        return self.out(h)


class GRURegressionHead(nn.Module):
    def __init__(self, c_in: int, c_out: int, dropout: float = 0.15) -> None:
        super().__init__()
        self.encoder = GRUAttentionEncoder(
            c_in=c_in,
            gru_dim=c_in,
            n_layers=2,
            n_heads=4,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_in, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq = self.encoder(x)
        pooled = seq.mean(dim=2)
        return self.head(pooled)


def build_regressor_head(
    name: str,
    feature_dim: int,
    out_dim: int,
    seq_len: int,
    dropout: float = 0.2,
) -> Tuple[nn.Module, bool]:
    if name == "mlp":
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, out_dim),
        ), False
    if name == "deep_mlp":
        hidden = feature_dim
        return nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_dim),
        ), False
    if name == "residual_mlp":
        return ResidualRegressor(feature_dim, out_dim, dropout=dropout), False
    if name == "gru":
        return GRURegressionHead(feature_dim, out_dim, dropout=dropout), True
    if name == "tsit":
        return build_classifier("TSiTPlus", c_in=feature_dim, c_out=out_dim, seq_len=seq_len), True
    if name == "convtran":
        return build_classifier("ConvTranPlus", c_in=feature_dim, c_out=out_dim, seq_len=seq_len), True
    raise ValueError(f"Unknown regressor type: {name}")


def epsilon_upper_bound_approx(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    if noise_multiplier <= 0:
        return float("inf")
    if sample_rate <= 0 or steps <= 0 or not (0.0 < delta < 1.0):
        return float("inf")

    term1 = sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier
    term2 = steps * (sample_rate ** 2) / (noise_multiplier ** 2)
    return float(term1 + term2)


def solve_noise_multiplier_for_epsilon(
    *,
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    tol: float = 1e-3,
    max_iter: int = 80,
) -> float:
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")

    lo, hi = 1e-4, 1.0
    eps_hi = epsilon_upper_bound_approx(
        noise_multiplier=hi,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )

    expand_guard = 0
    while eps_hi > target_epsilon and expand_guard < 60:
        hi *= 2.0
        eps_hi = epsilon_upper_bound_approx(
            noise_multiplier=hi,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        expand_guard += 1

    if eps_hi > target_epsilon:
        raise RuntimeError(
            "Could not find finite noise multiplier for requested epsilon. "
            "Try a larger epsilon, larger batch size, fewer epochs, or larger delta."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        eps_mid = epsilon_upper_bound_approx(
            noise_multiplier=mid,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        if eps_mid <= target_epsilon:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= tol:
            break

    return float(hi)


def _dp_noise_and_step(
    params: List[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    accum_grads: List[torch.Tensor],
    batch_size: int,
    noise_multiplier: float,
    max_grad_norm: float,
) -> None:
    optimizer.zero_grad(set_to_none=True)
    noise_std = noise_multiplier * max_grad_norm
    for j, p in enumerate(params):
        noise = torch.randn_like(accum_grads[j]) * noise_std
        p.grad = (accum_grads[j] + noise) / float(batch_size)
    optimizer.step()


def _get_batched_edges(
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if batch_size not in cache:
        cache[batch_size] = (
            batch_edge_index(edge_index_adj, n_nodes, batch_size),
            batch_edge_index(edge_index_od, n_nodes, batch_size),
            batch_edge_index(edge_index_od_t, n_nodes, batch_size),
        )
    return cache[batch_size]


def _build_regression_feature_dataset(
    model: StackedGRUThreeStagePredictor,
    source_loader: Iterable,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    use_amp: bool,
) -> TensorDataset:
    model.eval()
    features: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    with torch.no_grad():
        for bx, by_reg, _ in source_loader:
            bx = bx.to(device, non_blocking=True)
            bsz = bx.size(0)
            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                edge_cache,
                edge_index_adj,
                edge_index_od,
                edge_index_od_t,
                n_nodes,
                bsz,
            )
            with autocast(enabled=use_amp):
                pooled = model._extract_pooled_features(bx, bei_adj, bei_od, bei_od_t)
            features.append(pooled.float().cpu())
            targets.append(by_reg.float().cpu())

    return TensorDataset(torch.cat(features, dim=0), torch.cat(targets, dim=0))


def _dp_stage1_step(
    model: StackedGRUThreeStagePredictor,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    bx: torch.Tensor,
    by_cls: torch.Tensor,
    ei_adj: torch.Tensor,
    ei_od: torch.Tensor,
    ei_od_t: torch.Tensor,
    n_nodes: int,
    max_grad_norm: float,
    noise_multiplier: float,
    device: torch.device,
) -> float:
    params = [p for p in model.parameters() if p.requires_grad]
    accum_grads = [torch.zeros_like(p, device=device) for p in params]

    bx = bx.to(device, non_blocking=True)
    by_cls = by_cls.to(device, non_blocking=True)
    bsz = bx.size(0)
    total_loss = 0.0

    bei_adj_1 = batch_edge_index(ei_adj, n_nodes, 1)
    bei_od_1 = batch_edge_index(ei_od, n_nodes, 1)
    bei_od_t_1 = batch_edge_index(ei_od_t, n_nodes, 1)

    for i in range(bsz):
        optimizer.zero_grad(set_to_none=True)
        logits = model.forward_classifier(
            bx[i : i + 1],
            bei_adj_1,
            bei_od_1,
            bei_od_t_1,
        )
        loss = loss_fn(logits, by_cls[i : i + 1])
        total_loss += float(loss.item())
        loss.backward()

        sq_norm = torch.zeros((), device=device)
        for p in params:
            if p.grad is not None:
                sq_norm += p.grad.detach().pow(2).sum()
        grad_norm = torch.sqrt(sq_norm + 1e-12)
        clip_coef = (max_grad_norm / (grad_norm + 1e-12)).clamp(max=1.0)

        for j, p in enumerate(params):
            if p.grad is not None:
                accum_grads[j].add_(p.grad.detach() * clip_coef)

    _dp_noise_and_step(params, optimizer, accum_grads, bsz, noise_multiplier, max_grad_norm)
    return total_loss / float(bsz)


def _dp_stage_reg_step(
    regressor: nn.Module,
    optimizer: torch.optim.Optimizer,
    feat_batch: torch.Tensor,
    by_reg: torch.Tensor,
    max_grad_norm: float,
    noise_multiplier: float,
    device: torch.device,
    mask_fn,
    huber_delta: float,
) -> Tuple[float, float]:
    params = [p for p in regressor.parameters() if p.requires_grad]
    accum_grads = [torch.zeros_like(p, device=device) for p in params]

    feat_batch = feat_batch.to(device, non_blocking=True)
    by_reg = by_reg.to(device, non_blocking=True)
    bsz = feat_batch.size(0)
    total_loss = 0.0
    total_mask = 0.0

    huber = nn.HuberLoss(reduction="none", delta=huber_delta)

    for i in range(bsz):
        optimizer.zero_grad(set_to_none=True)
        yi = by_reg[i : i + 1]
        mask = mask_fn(yi)
        mask_sum = float(mask.sum().item())
        total_mask += mask_sum / float(mask.numel())
        if mask_sum <= 0.0:
            continue

        preds = regressor(feat_batch[i : i + 1])
        per = huber(preds, yi) * mask
        denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
        loss = (per.sum(dim=(0, 1)) / denom).mean()
        total_loss += float(loss.item())
        loss.backward()

        sq_norm = torch.zeros((), device=device)
        for p in params:
            if p.grad is not None:
                sq_norm += p.grad.detach().pow(2).sum()
        grad_norm = torch.sqrt(sq_norm + 1e-12)
        clip_coef = (max_grad_norm / (grad_norm + 1e-12)).clamp(max=1.0)

        for j, p in enumerate(params):
            if p.grad is not None:
                accum_grads[j].add_(p.grad.detach() * clip_coef)

    _dp_noise_and_step(params, optimizer, accum_grads, bsz, noise_multiplier, max_grad_norm)
    return total_loss / float(max(1, bsz)), total_mask / float(max(1, bsz))


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
        regressor_name: str = "mlp",
        dropout: float = 0.15,
        chunk_size: int = 200,
    ) -> None:
        super().__init__()
        self.c_out = c_out
        self.chunk_size = chunk_size
        self.feature_dim = gru_dim + gat_hidden
        self.seq_len = seq_len

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

        self.regressor_name = regressor_name
        self.regressor_delayed, self.regressor_expects_sequence = build_regressor_head(
            regressor_name,
            self.feature_dim,
            c_out,
            seq_len=seq_len,
            dropout=0.2,
        )
        self.regressor_nondelayed, _ = build_regressor_head(
            regressor_name,
            self.feature_dim,
            c_out,
            seq_len=seq_len,
            dropout=0.2,
        )

    def _extract_enriched_sequence(
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
        return enriched.view(bsz, n_nodes, self.feature_dim, seq_len)

    def _extract_pooled_features(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        enriched = self._extract_enriched_sequence(x, edge_index_adj, edge_index_od, edge_index_od_t)
        return enriched.mean(dim=3)

    def forward_classifier(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_nodes, _, seq_len = x.shape
        total = bsz * n_nodes
        enriched = self._extract_enriched_sequence(x, edge_index_adj, edge_index_od, edge_index_od_t)
        enriched_flat = enriched.view(total, self.feature_dim, seq_len)

        logits = self.classifier(enriched_flat)
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
        if self.regressor_expects_sequence:
            enriched = self._extract_enriched_sequence(x, edge_index_adj, edge_index_od, edge_index_od_t)
            bsz, n_nodes, feature_dim, seq_len = enriched.shape
            seq_flat = enriched.view(bsz * n_nodes, feature_dim, seq_len)
            if which == "delayed":
                pred = self.regressor_delayed(seq_flat)
            else:
                pred = self.regressor_nondelayed(seq_flat)
            if isinstance(pred, (tuple, list)):
                pred = pred[0]
            if pred.dim() > 2:
                pred = pred.flatten(1)
            return pred.view(bsz, n_nodes, self.c_out)

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


def _iter_limited(loader: DataLoader, max_batches: int):
    if max_batches is None or max_batches <= 0:
        yield from loader
        return
    for idx, batch in enumerate(loader):
        if idx >= max_batches:
            break
        yield batch


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
    dp_enabled: bool,
    noise_multiplier: float,
    max_grad_norm: float,
    target_delta: float,
    sample_rate: float,
    privacy_steps_offset: int,
    use_amp: bool,
    max_train_batches: int,
    max_val_batches: int,
    effective_train_steps: int,
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
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scaler = GradScaler(enabled=(use_amp and not dp_enabled))

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_losses: List[float] = []

        for bx, by_reg, by_cls in _iter_limited(train_loader, max_train_batches):
            bsz = bx.size(0)
            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                edge_cache,
                edge_index_adj,
                edge_index_od,
                edge_index_od_t,
                n_nodes,
                bsz,
            )

            if dp_enabled:
                loss = _dp_stage1_step(
                    model,
                    optimizer,
                    loss_fn,
                    bx,
                    by_cls,
                    edge_index_adj,
                    edge_index_od,
                    edge_index_od_t,
                    n_nodes,
                    max_grad_norm,
                    noise_multiplier,
                    device,
                )
            else:
                bx = bx.to(device, non_blocking=True)
                by_cls = by_cls.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    logits = model.forward_classifier(bx, bei_adj, bei_od, bei_od_t)
                    loss_t = loss_fn(logits, by_cls)
                scaler.scale(loss_t).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                loss = float(loss_t.item())

            train_losses.append(float(loss))

        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for bx, _, by_cls in _iter_limited(val_loader, max_val_batches):
                bx = bx.to(device, non_blocking=True)
                bsz = bx.size(0)
                bei_adj, bei_od, bei_od_t = _get_batched_edges(
                    edge_cache,
                    edge_index_adj,
                    edge_index_od,
                    edge_index_od_t,
                    n_nodes,
                    bsz,
                )
                with autocast(enabled=use_amp):
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
                "epsilon_approx": epsilon_upper_bound_approx(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                    steps=privacy_steps_offset + epoch * effective_train_steps,
                    delta=target_delta,
                ) if dp_enabled else 0.0,
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
    dp_enabled: bool,
    noise_multiplier: float,
    max_grad_norm: float,
    target_delta: float,
    sample_rate: float,
    privacy_steps_offset: int,
    use_amp: bool,
    cache_stage_features: bool,
    base_num_workers: int,
    max_train_batches: int,
    max_val_batches: int,
    effective_train_steps: int,
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
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scaler_amp = GradScaler(enabled=(use_amp and not dp_enabled))

    feature_train_loader = None
    feature_val_loader = None
    if cache_stage_features:
        print("  [perf] caching frozen stage-2 features...")
        train_feat_ds = _build_regression_feature_dataset(
            model,
            _iter_limited(train_loader, max_train_batches),
            edge_index_adj,
            edge_index_od,
            edge_index_od_t,
            n_nodes,
            device,
            use_amp,
        )
        val_feat_ds = _build_regression_feature_dataset(
            model,
            _iter_limited(val_loader, max_val_batches),
            edge_index_adj,
            edge_index_od,
            edge_index_od_t,
            n_nodes,
            device,
            use_amp,
        )
        worker_count = min(2, max(0, base_num_workers))
        loader_kwargs = dict(
            num_workers=worker_count,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(worker_count > 0),
        )
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = 2
        feature_train_loader = DataLoader(train_feat_ds, batch_size=train_loader.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
        feature_val_loader = DataLoader(val_feat_ds, batch_size=val_loader.batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_masks: List[float] = []

        if feature_train_loader is not None:
            iter_train = ((bf, by_reg, None) for bf, by_reg in feature_train_loader)
        else:
            iter_train = _iter_limited(train_loader, max_train_batches)

        for bx, by_reg, _ in iter_train:
            bsz = bx.size(0)

            if dp_enabled:
                def _mask_delayed(yi: torch.Tensor) -> torch.Tensor:
                    thr = thr_scaled
                    if thr.numel() == 1 and yi.shape[-1] > 1:
                        thr = thr.expand(yi.shape[-1])
                    return (yi > thr).float()

                if feature_train_loader is not None:
                    feat_batch = bx
                else:
                    bx_dev = bx.to(device, non_blocking=True)
                    with torch.no_grad():
                        if model.regressor_expects_sequence:
                            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                                edge_cache,
                                edge_index_adj,
                                edge_index_od,
                                edge_index_od_t,
                                n_nodes,
                                bsz,
                            )
                            feat_batch = model._extract_enriched_sequence(bx_dev, bei_adj, bei_od, bei_od_t).detach()
                        else:
                            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                                edge_cache,
                                edge_index_adj,
                                edge_index_od,
                                edge_index_od_t,
                                n_nodes,
                                bsz,
                            )
                            feat_batch = model._extract_pooled_features(bx_dev, bei_adj, bei_od, bei_od_t).detach()

                loss_val, mask_ratio = _dp_stage_reg_step(
                    model.regressor_delayed,
                    optimizer,
                    feat_batch,
                    by_reg,
                    max_grad_norm,
                    noise_multiplier,
                    device,
                    mask_fn=_mask_delayed,
                    huber_delta=2.0,
                )
                tr_losses.append(float(loss_val))
                tr_masks.append(float(mask_ratio))
            else:
                by_reg = by_reg.to(device, non_blocking=True)
                if feature_train_loader is not None:
                    feat_batch = bx.to(device, non_blocking=True)
                else:
                    bx = bx.to(device, non_blocking=True)
                    bei_adj, bei_od, bei_od_t = _get_batched_edges(
                        edge_cache,
                        edge_index_adj,
                        edge_index_od,
                        edge_index_od_t,
                        n_nodes,
                        bsz,
                    )
                    with torch.no_grad():
                        if model.regressor_expects_sequence:
                            feat_batch = model._extract_enriched_sequence(bx, bei_adj, bei_od, bei_od_t)
                        else:
                            feat_batch = model._extract_pooled_features(bx, bei_adj, bei_od, bei_od_t)
                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    preds = model.regressor_delayed(feat_batch)
                    loss, mask = masked_loss(preds, by_reg)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                tr_losses.append(float(loss.item()))
                tr_masks.append(_channel_stats(mask))

        model.eval()
        va_losses: List[float] = []
        va_masks: List[float] = []
        if feature_val_loader is not None:
            iter_val = ((bf, by_reg, None) for bf, by_reg in feature_val_loader)
        else:
            iter_val = _iter_limited(val_loader, max_val_batches)

        with torch.no_grad():
            for bx, by_reg, _ in iter_val:
                by_reg = by_reg.to(device, non_blocking=True)
                bsz = bx.size(0)
                if feature_val_loader is not None:
                    feat_batch = bx.to(device, non_blocking=True)
                else:
                    bx = bx.to(device, non_blocking=True)
                    bei_adj, bei_od, bei_od_t = _get_batched_edges(
                        edge_cache,
                        edge_index_adj,
                        edge_index_od,
                        edge_index_od_t,
                        n_nodes,
                        bsz,
                    )
                    with autocast(enabled=use_amp):
                        if model.regressor_expects_sequence:
                            feat_batch = model._extract_enriched_sequence(bx, bei_adj, bei_od, bei_od_t)
                        else:
                            feat_batch = model._extract_pooled_features(bx, bei_adj, bei_od, bei_od_t)
                with autocast(enabled=use_amp):
                    preds = model.regressor_delayed(feat_batch)
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
                "epsilon_approx": epsilon_upper_bound_approx(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                    steps=privacy_steps_offset + epoch * effective_train_steps,
                    delta=target_delta,
                ) if dp_enabled else 0.0,
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
    dp_enabled: bool,
    noise_multiplier: float,
    max_grad_norm: float,
    target_delta: float,
    sample_rate: float,
    privacy_steps_offset: int,
    use_amp: bool,
    cache_stage_features: bool,
    base_num_workers: int,
    max_train_batches: int,
    max_val_batches: int,
    effective_train_steps: int,
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
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scaler_amp = GradScaler(enabled=(use_amp and not dp_enabled))

    feature_train_loader = None
    feature_val_loader = None
    if cache_stage_features:
        print("  [perf] caching frozen stage-3 features...")
        train_feat_ds = _build_regression_feature_dataset(
            model,
            _iter_limited(train_loader, max_train_batches),
            edge_index_adj,
            edge_index_od,
            edge_index_od_t,
            n_nodes,
            device,
            use_amp,
        )
        val_feat_ds = _build_regression_feature_dataset(
            model,
            _iter_limited(val_loader, max_val_batches),
            edge_index_adj,
            edge_index_od,
            edge_index_od_t,
            n_nodes,
            device,
            use_amp,
        )
        worker_count = min(2, max(0, base_num_workers))
        loader_kwargs = dict(
            num_workers=worker_count,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(worker_count > 0),
        )
        if worker_count > 0:
            loader_kwargs["prefetch_factor"] = 2
        feature_train_loader = DataLoader(train_feat_ds, batch_size=train_loader.batch_size, shuffle=True, drop_last=True, **loader_kwargs)
        feature_val_loader = DataLoader(val_feat_ds, batch_size=val_loader.batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_masks: List[float] = []

        if feature_train_loader is not None:
            iter_train = ((bf, by_reg, None) for bf, by_reg in feature_train_loader)
        else:
            iter_train = _iter_limited(train_loader, max_train_batches)

        for bx, by_reg, _ in iter_train:
            bsz = bx.size(0)

            if dp_enabled:
                def _mask_nondelayed(yi: torch.Tensor) -> torch.Tensor:
                    if mean_t is None or std_t is None:
                        yi_den = yi
                    else:
                        yi_den = yi * std_t + mean_t
                    return (yi_den.abs() < float(delay_threshold)).float()

                if feature_train_loader is not None:
                    feat_batch = bx
                else:
                    bx_dev = bx.to(device, non_blocking=True)
                    with torch.no_grad():
                        if model.regressor_expects_sequence:
                            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                                edge_cache,
                                edge_index_adj,
                                edge_index_od,
                                edge_index_od_t,
                                n_nodes,
                                bsz,
                            )
                            feat_batch = model._extract_enriched_sequence(bx_dev, bei_adj, bei_od, bei_od_t).detach()
                        else:
                            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                                edge_cache,
                                edge_index_adj,
                                edge_index_od,
                                edge_index_od_t,
                                n_nodes,
                                bsz,
                            )
                            feat_batch = model._extract_pooled_features(bx_dev, bei_adj, bei_od, bei_od_t).detach()

                loss_val, mask_ratio = _dp_stage_reg_step(
                    model.regressor_nondelayed,
                    optimizer,
                    feat_batch,
                    by_reg,
                    max_grad_norm,
                    noise_multiplier,
                    device,
                    mask_fn=_mask_nondelayed,
                    huber_delta=1.0,
                )
                tr_losses.append(float(loss_val))
                tr_masks.append(float(mask_ratio))
            else:
                by_reg = by_reg.to(device, non_blocking=True)
                if feature_train_loader is not None:
                    feat_batch = bx.to(device, non_blocking=True)
                else:
                    bx = bx.to(device, non_blocking=True)
                    bei_adj, bei_od, bei_od_t = _get_batched_edges(
                        edge_cache,
                        edge_index_adj,
                        edge_index_od,
                        edge_index_od_t,
                        n_nodes,
                        bsz,
                    )
                    with torch.no_grad():
                        if model.regressor_expects_sequence:
                            feat_batch = model._extract_enriched_sequence(bx, bei_adj, bei_od, bei_od_t)
                        else:
                            feat_batch = model._extract_pooled_features(bx, bei_adj, bei_od, bei_od_t)

                mask = nondelayed_mask(by_reg)
                if float(mask.sum().item()) <= 0.0:
                    continue

                optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=use_amp):
                    preds = model.regressor_nondelayed(feat_batch)
                    per = huber(preds, by_reg) * mask
                    denom = mask.sum(dim=(0, 1)).clamp_min(1.0)
                    loss_ch = per.sum(dim=(0, 1)) / denom
                    loss = loss_ch.mean()
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                tr_losses.append(float(loss.item()))
                tr_masks.append(_channel_stats(mask))

        model.eval()
        va_losses: List[float] = []
        va_masks: List[float] = []
        if feature_val_loader is not None:
            iter_val = ((bf, by_reg, None) for bf, by_reg in feature_val_loader)
        else:
            iter_val = _iter_limited(val_loader, max_val_batches)

        with torch.no_grad():
            for bx, by_reg, _ in iter_val:
                by_reg = by_reg.to(device, non_blocking=True)
                bsz = bx.size(0)
                if feature_val_loader is not None:
                    feat_batch = bx.to(device, non_blocking=True)
                else:
                    bx = bx.to(device, non_blocking=True)
                    bei_adj, bei_od, bei_od_t = _get_batched_edges(
                        edge_cache,
                        edge_index_adj,
                        edge_index_od,
                        edge_index_od_t,
                        n_nodes,
                        bsz,
                    )
                    with autocast(enabled=use_amp):
                        if model.regressor_expects_sequence:
                            feat_batch = model._extract_enriched_sequence(bx, bei_adj, bei_od, bei_od_t)
                        else:
                            feat_batch = model._extract_pooled_features(bx, bei_adj, bei_od, bei_od_t)
                mask = nondelayed_mask(by_reg)
                if float(mask.sum().item()) <= 0.0:
                    continue
                with autocast(enabled=use_amp):
                    preds = model.regressor_nondelayed(feat_batch)
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
                "epsilon_approx": epsilon_upper_bound_approx(
                    noise_multiplier=noise_multiplier,
                    sample_rate=sample_rate,
                    steps=privacy_steps_offset + epoch * effective_train_steps,
                    delta=target_delta,
                ) if dp_enabled else 0.0,
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
    dp_enabled: bool,
    epsilon_target: float,
    epsilon_approx_final: float,
    delta: float,
    noise_multiplier: float,
    max_grad_norm: float,
    use_amp: bool,
    max_test_batches: int,
) -> None:
    model.eval()
    cls_probs, cls_targets = [], []
    reg_preds_delayed, reg_preds_nondelayed, reg_targets = [], [], []

    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    with torch.no_grad():
        for bx, by_reg, by_cls in _iter_limited(test_loader, max_test_batches):
            bx = bx.to(device, non_blocking=True)
            by_reg = by_reg.to(device, non_blocking=True)
            bsz = bx.size(0)
            bei_adj, bei_od, bei_od_t = _get_batched_edges(
                edge_cache,
                edge_index_adj,
                edge_index_od,
                edge_index_od_t,
                n_nodes,
                bsz,
            )

            with autocast(enabled=use_amp):
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
        w.writerow(["dp_enabled", int(dp_enabled)])
        w.writerow(["epsilon_target", f"{epsilon_target:.6f}"])
        w.writerow(["epsilon_approx_final", f"{epsilon_approx_final:.6f}"])
        w.writerow(["delta", f"{delta:.12f}"])
        w.writerow(["noise_multiplier", f"{noise_multiplier:.6f}"])
        w.writerow(["max_grad_norm", f"{max_grad_norm:.6f}"])

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
    p.add_argument(
        "--regressor",
        type=str,
        default="mlp",
        choices=["mlp", "deep_mlp", "residual_mlp", "gru", "tsit", "convtran"],
        help="Regressor head used for delayed/non-delayed stages",
    )
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

    p.add_argument("--dp", action="store_true", help="Enable manual DP-SGD for stage training")
    p.add_argument("--epsilon", type=float, default=7.5)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Manual sigma override; if unset and --dp is on, solved from epsilon",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num_workers", type=int, default=-1, help="-1 auto (Windows=0, else min(8,cpu_count))")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision AMP on CUDA")
    p.add_argument(
        "--no_cache_stage23_features",
        action="store_true",
        help="Disable feature caching for stage 2/3 (slower, but lower host RAM usage)",
    )
    p.add_argument("--max_train_batches", type=int, default=0, help="Limit train batches per epoch for quick benchmarks (0=all)")
    p.add_argument("--max_val_batches", type=int, default=0, help="Limit val batches per epoch for quick benchmarks (0=all)")
    p.add_argument("--max_test_batches", type=int, default=0, help="Limit test batches for quick benchmarks (0=all)")
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

    use_amp = (device.type == "cuda") and (not args.no_amp)
    cache_stage23_features = not args.no_cache_stage23_features

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

    effective_train_steps = args.max_train_batches if args.max_train_batches > 0 else len(train_loader)

    pos_rate = trY_cls.reshape(-1, delay_dim).mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    steps_per_stage = effective_train_steps
    total_steps = (args.stage1_epochs + args.stage2_epochs + args.stage3_epochs) * steps_per_stage
    sample_rate = args.batch_size / max(1, len(train_loader.dataset))

    if args.dp:
        if args.noise_multiplier is None:
            sigma = solve_noise_multiplier_for_epsilon(
                target_epsilon=args.epsilon,
                delta=args.delta,
                sample_rate=sample_rate,
                steps=total_steps,
            )
        else:
            sigma = float(args.noise_multiplier)

        epsilon_final = epsilon_upper_bound_approx(
            noise_multiplier=sigma,
            sample_rate=sample_rate,
            steps=total_steps,
            delta=args.delta,
        )
        print("[DP] Enabled")
        print(f"  target epsilon={args.epsilon:.4f}, delta={args.delta:.1e}")
        print(f"  sample_rate={sample_rate:.6f}, total_steps={total_steps}")
        print(f"  noise_multiplier(sigma)={sigma:.6f}")
        print(f"  approx epsilon upper bound={epsilon_final:.4f}")
    else:
        sigma = 0.0
        epsilon_final = 0.0
        print("[DP] Disabled")

    print(f"[PERF] AMP={'on' if use_amp else 'off'} | cache_stage23_features={'on' if cache_stage23_features else 'off'}")

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
        regressor_name=args.regressor,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
    ).to(device)

    if model.regressor_expects_sequence and cache_stage23_features:
        cache_stage23_features = False
        print("[PERF] stage2/3 feature cache auto-disabled for sequence regressor")

    print(f"[MODEL] classifier={args.classifier} | regressor={args.regressor}")

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
        args.dp,
        sigma,
        args.max_grad_norm,
        args.delta,
        sample_rate,
        0,
        use_amp,
        args.max_train_batches,
        args.max_val_batches,
        effective_train_steps,
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
        args.dp,
        sigma,
        args.max_grad_norm,
        args.delta,
        sample_rate,
        args.stage1_epochs * steps_per_stage,
        use_amp,
        cache_stage23_features,
        worker_count,
        args.max_train_batches,
        args.max_val_batches,
        effective_train_steps,
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
        args.dp,
        sigma,
        args.max_grad_norm,
        args.delta,
        sample_rate,
        (args.stage1_epochs + args.stage2_epochs) * steps_per_stage,
        use_amp,
        cache_stage23_features,
        worker_count,
        args.max_train_batches,
        args.max_val_batches,
        effective_train_steps,
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
        args.dp,
        float(args.epsilon),
        float(epsilon_final),
        float(args.delta),
        float(sigma),
        float(args.max_grad_norm),
        use_amp,
        args.max_test_batches,
    )


if __name__ == "__main__":
    main()
