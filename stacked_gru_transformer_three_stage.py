# # 3) Resume after timeout/disconnect
# !python stacked_gru_transformer_three_stage.py \
#   --resume_checkpoint "/content/drive/MyDrive/stpn_ckpts/latest_checkpoint.pt" \
#   --checkpoint_dir "/content/drive/MyDrive/stpn_ckpts" \
#   --checkpoint_every 1


from __future__ import annotations

import argparse
import csv
import math
import os
import time
import json
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

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


def _choose_num_heads(d_model: int, preferred: Tuple[int, ...] = (8, 4, 2, 1)) -> int:
    for h in preferred:
        if h <= d_model and d_model % h == 0:
            return h
    return 1


class _SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, ff_mult: int = 4) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + a)
        x = self.ln2(x + self.ff(x))
        return x


class TFTRegressionHead(nn.Module):
    """Lightweight TFT-like temporal head over enriched sequence.

    Input:  (B, F, T) channel-first
    Output: (B, c_out)
    """

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.15, n_blocks: int = 2) -> None:
        super().__init__()
        n_heads = _choose_num_heads(c_in)
        self.in_norm = nn.LayerNorm(c_in)
        self.blocks = nn.Sequential(*[_SelfAttentionBlock(c_in, n_heads, dropout=dropout) for _ in range(max(1, n_blocks))])
        self.head = nn.Sequential(
            nn.LayerNorm(c_in),
            nn.Linear(c_in, c_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(c_in, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, F, T) -> (B, T, F)
        xt = x.permute(0, 2, 1).contiguous()
        xt = self.in_norm(xt)
        xt = self.blocks(xt)
        pooled = xt.mean(dim=1)
        return self.head(pooled)


class NBeatsRegressionHead(nn.Module):
    """Simple N-BEATS-like residual MLP over flattened sequence."""

    def __init__(
        self,
        c_in: int,
        c_out: int,
        seq_len: int,
        dropout: float = 0.15,
        hidden_mult: int = 4,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        flat_dim = c_in * seq_len
        hidden = max(64, c_in * hidden_mult)
        self.in_norm = nn.LayerNorm(flat_dim)

        blocks: List[nn.Module] = []
        for _ in range(max(1, n_blocks)):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(flat_dim),
                    nn.Linear(flat_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, flat_dim),
                    nn.Dropout(dropout),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.out = nn.Sequential(
            nn.LayerNorm(flat_dim),
            nn.Linear(flat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F, T)
        bsz, feat_dim, seq_len = x.shape
        flat = x.reshape(bsz, feat_dim * seq_len)
        h = self.in_norm(flat)
        for block in self.blocks:
            h = h + block(h)
        return self.out(h)


class NodeTransformerRegressionHead(nn.Module):
    """Cross-node self-attention head on pooled per-node features.

    Input:  (B, N, F)
    Output: (B, N, c_out)
    """

    def __init__(self, feature_dim: int, out_dim: int, dropout: float = 0.15, n_blocks: int = 2) -> None:
        super().__init__()
        n_heads = _choose_num_heads(feature_dim)
        self.in_norm = nn.LayerNorm(feature_dim)
        self.blocks = nn.Sequential(*[_SelfAttentionBlock(feature_dim, n_heads, dropout=dropout) for _ in range(max(1, n_blocks))])
        self.proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, F)
        h = self.in_norm(x)
        h = self.blocks(h)
        return self.proj(h)


class GraphAwareRegressorHead(nn.Module):
    """Graph-aware head that applies an extra GAT fusion on pooled features.

    Requires batched edge indices for the current batch size.
    """

    requires_edges: bool = True

    def __init__(self, feature_dim: int, out_dim: int, dropout: float = 0.15, heads: int = 2) -> None:
        super().__init__()
        self.gat = MultiEdgeGATFusion(in_channels=feature_dim, hidden_channels=feature_dim, heads=heads)
        self.head = nn.Sequential(
            nn.LayerNorm(feature_dim * 2),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, out_dim),
        )

    def forward_with_edges(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        # x: (B, N, F)
        bsz, n_nodes, feat_dim = x.shape
        flat = x.view(bsz * n_nodes, feat_dim)
        ctx = self.gat(flat, edge_index_adj, edge_index_od, edge_index_od_t)
        out = self.head(torch.cat([flat, ctx], dim=1))
        return out.view(bsz, n_nodes, -1)


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
    if name == "tft":
        return TFTRegressionHead(feature_dim, out_dim, dropout=dropout), True
    if name == "nbeats":
        return NBeatsRegressionHead(feature_dim, out_dim, seq_len=seq_len, dropout=dropout), True
    if name == "node_transformer":
        return NodeTransformerRegressionHead(feature_dim, out_dim, dropout=dropout), False
    if name == "graph_gat":
        return GraphAwareRegressorHead(feature_dim, out_dim, dropout=dropout), False
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


def _privacy_epsilon_for_steps(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    return epsilon_upper_bound_approx(
        noise_multiplier=noise_multiplier,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )


def _history_privacy_steps(
    history: List[Dict],
    *,
    stage: int,
    default_steps_per_epoch: int,
) -> int:
    total_steps = 0
    for row in history:
        if int(row.get("stage", stage)) != int(stage):
            continue
        steps = row.get("privacy_steps_in_epoch")
        if steps is None:
            steps = row.get("privacy_steps_epoch")
        if steps is None:
            steps = row.get("privacy_steps")
        if steps is None:
            steps = default_steps_per_epoch
        total_steps += int(steps)
    return int(total_steps)


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
    edge_indices_1: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
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

        preds = _apply_regressor(regressor, feat_batch[i : i + 1], edge_indices=edge_indices_1)
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

        self.regressor_requires_edges = bool(
            getattr(self.regressor_delayed, "requires_edges", False)
            or getattr(self.regressor_nondelayed, "requires_edges", False)
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
            if bool(getattr(self.regressor_delayed, "requires_edges", False)):
                return self.regressor_delayed.forward_with_edges(pooled, edge_index_adj, edge_index_od, edge_index_od_t)
            return self.regressor_delayed(pooled)
        if bool(getattr(self.regressor_nondelayed, "requires_edges", False)):
            return self.regressor_nondelayed.forward_with_edges(pooled, edge_index_adj, edge_index_od, edge_index_od_t)
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


def _apply_regressor(
    regressor: nn.Module,
    features: torch.Tensor,
    *,
    edge_indices: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
) -> torch.Tensor:
    """Apply regressor to pooled [B,N,F] or sequence [B,N,F,T] features.

    For edge-aware regressors, provide `edge_indices` for the current batch size.
    """
    requires_edges = bool(getattr(regressor, "requires_edges", False))
    if requires_edges and edge_indices is None:
        raise ValueError("Regressor requires edges but edge_indices was not provided")

    if features.dim() == 4:
        bsz, n_nodes, feat_dim, seq_len = features.shape
        flat = features.view(bsz * n_nodes, feat_dim, seq_len)
        out = regressor(flat)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:
            out = out.flatten(1)
        return out.view(bsz, n_nodes, -1)

    if requires_edges:
        edge_index_adj, edge_index_od, edge_index_od_t = edge_indices  # type: ignore[misc]
        out = regressor.forward_with_edges(features, edge_index_adj, edge_index_od, edge_index_od_t)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out
    return regressor(features)


def save_training_checkpoint(
    checkpoint_dir: str,
    stage: int,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    history_stage1: List[Dict],
    history_stage2: List[Dict],
    history_stage3: List[Dict],
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    sigma: float,
    epsilon_final: float,
    privacy_steps: int = 0,
    save_every: int = 1,
    is_best: bool = False,
    best_metric_name: str = "",
    best_metric_value: float = 0.0,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "stage": int(stage),
        "epoch": int(epoch),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "history_stage1": history_stage1,
        "history_stage2": history_stage2,
        "history_stage3": history_stage3,
        "stage1_time": float(stage1_time),
        "stage2_time": float(stage2_time),
        "stage3_time": float(stage3_time),
        "sigma": float(sigma),
        "epsilon_final": float(epsilon_final),
        "privacy_steps": int(privacy_steps),
        "is_best": bool(is_best),
        "best_metric_name": str(best_metric_name),
        "best_metric_value": float(best_metric_value),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }

    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(payload, latest_path)

    if save_every > 0 and (epoch % save_every == 0):
        snap_path = os.path.join(checkpoint_dir, f"checkpoint_stage{stage}_epoch{epoch}.pt")
        torch.save(payload, snap_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, f"best_stage{stage}.pt")
        torch.save(payload, best_path)


def load_training_checkpoint(path: str, device: torch.device) -> Dict:
    return torch.load(path, map_location=device)


def _load_compatible_model_state(model: nn.Module, checkpoint_state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    model_state = model.state_dict()
    compatible_state: Dict[str, torch.Tensor] = {}

    for key, value in checkpoint_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            compatible_state[key] = value

    model.load_state_dict(compatible_state, strict=False)
    skipped = len(checkpoint_state) - len(compatible_state)
    return len(compatible_state), skipped


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
    start_epoch: int = 1,
    history_init: Optional[List[Dict]] = None,
    stage_time_offset: float = 0.0,
    checkpoint_callback: Optional[Callable[[int, List[Dict], float, bool], None]] = None,
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

    history: List[Dict] = list(history_init) if history_init is not None else []
    best_f1 = -1.0
    best_state = None
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scaler = GradScaler(enabled=(use_amp and not dp_enabled))
    completed_privacy_steps = _history_privacy_steps(
        history,
        stage=1,
        default_steps_per_epoch=effective_train_steps,
    )
    running_privacy_steps = int(completed_privacy_steps)

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.time()
        model.train()
        train_losses: List[float] = []
        epoch_privacy_steps = 0

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
            if dp_enabled:
                epoch_privacy_steps += 1

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

        is_best = False
        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {
                "encoder": {k: v.cpu().clone() for k, v in model.encoder.state_dict().items()},
                "gat": {k: v.cpu().clone() for k, v in model.gat.state_dict().items()},
                "classifier": {k: v.cpu().clone() for k, v in model.classifier.state_dict().items()},
            }
            is_best = True

        ep_sec = time.time() - ep_t0
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        if dp_enabled:
            running_privacy_steps += int(epoch_privacy_steps)
            privacy_steps_total = int(privacy_steps_offset + running_privacy_steps)
        else:
            privacy_steps_total = 0
        privacy_epsilon = _privacy_epsilon_for_steps(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=privacy_steps_total,
            delta=target_delta,
        ) if dp_enabled else 0.0
        history.append(
            {
                "stage": 1,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_f1": vm["f1"],
                "val_precision": vm["precision"],
                "val_recall": vm["recall"],
                "val_accuracy": vm["accuracy"],
                "is_best": int(is_best),
                "epoch_time_seconds": ep_sec,
                "privacy_steps_in_epoch": int(epoch_privacy_steps),
                "privacy_steps_total": int(privacy_steps_total),
                "epsilon_approx": privacy_epsilon,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | loss={train_loss:.4f} | val_f1={vm['f1']:.4f} "
            f"(arr={vm['f1_arrival']:.4f}, dep={vm['f1_departure']:.4f}) | sec={ep_sec:.1f}"
            + (
                f" | dp_steps={privacy_steps_total} | epsilon={privacy_epsilon:.4f}"
                if dp_enabled
                else ""
            ),
            flush=True,
        )

        if checkpoint_callback is not None:
            checkpoint_callback(epoch, list(history), stage_time_offset + (time.time() - t0), bool(is_best))

        if early_stopping(vm["f1"], epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.encoder.load_state_dict(best_state["encoder"])
        model.gat.load_state_dict(best_state["gat"])
        model.classifier.load_state_dict(best_state["classifier"])

    _set_requires_grad(model.regressor_delayed, True)
    _set_requires_grad(model.regressor_nondelayed, True)

    return history, stage_time_offset + (time.time() - t0)


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
    start_epoch: int = 1,
    history_init: Optional[List[Dict]] = None,
    stage_time_offset: float = 0.0,
    checkpoint_callback: Optional[Callable[[int, List[Dict], float, bool], None]] = None,
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

    history: List[Dict] = list(history_init) if history_init is not None else []
    best_val = float("inf")
    best_state = None
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scaler_amp = GradScaler(enabled=(use_amp and not dp_enabled))
    completed_privacy_steps = _history_privacy_steps(
        history,
        stage=2,
        default_steps_per_epoch=effective_train_steps,
    )
    running_privacy_steps = int(completed_privacy_steps)

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

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_masks: List[float] = []
        epoch_privacy_steps = 0

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
                    edge_indices_1=(
                        _get_batched_edges(edge_cache, edge_index_adj, edge_index_od, edge_index_od_t, n_nodes, 1)
                        if model.regressor_requires_edges and feature_train_loader is None
                        else None
                    ),
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
                    preds = _apply_regressor(
                        model.regressor_delayed,
                        feat_batch,
                        edge_indices=(bei_adj, bei_od, bei_od_t)
                        if model.regressor_requires_edges and feature_train_loader is None
                        else None,
                    )
                    loss, mask = masked_loss(preds, by_reg)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
                tr_losses.append(float(loss.item()))
                tr_masks.append(_channel_stats(mask))

            if dp_enabled:
                epoch_privacy_steps += 1

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
                    preds = _apply_regressor(
                        model.regressor_delayed,
                        feat_batch,
                        edge_indices=(bei_adj, bei_od, bei_od_t)
                        if model.regressor_requires_edges and feature_val_loader is None
                        else None,
                    )
                    loss, mask = masked_loss(preds, by_reg)
                va_losses.append(float(loss.item()))
                va_masks.append(_channel_stats(mask))

        tr_loss = float(np.mean(tr_losses)) if tr_losses else 0.0
        va_loss = float(np.mean(va_losses)) if va_losses else 0.0
        tr_mask = float(np.mean(tr_masks)) if tr_masks else 0.0
        va_mask = float(np.mean(va_masks)) if va_masks else 0.0

        is_best = False
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.regressor_delayed.state_dict().items()}
            is_best = True

        ep_sec = time.time() - ep_t0
        if dp_enabled:
            running_privacy_steps += int(epoch_privacy_steps)
            privacy_steps_total = int(privacy_steps_offset + running_privacy_steps)
        else:
            privacy_steps_total = 0
        privacy_epsilon = _privacy_epsilon_for_steps(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=privacy_steps_total,
            delta=target_delta,
        ) if dp_enabled else 0.0
        history.append(
            {
                "stage": 2,
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_mask_ratio": tr_mask,
                "val_loss": va_loss,
                "val_mask_ratio": va_mask,
                "is_best": int(is_best),
                "epoch_time_seconds": ep_sec,
                "privacy_steps_in_epoch": int(epoch_privacy_steps),
                "privacy_steps_total": int(privacy_steps_total),
                "epsilon_approx": privacy_epsilon,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"mask={tr_mask*100:.1f}% (val {va_mask*100:.1f}%) | sec={ep_sec:.1f}"
            + (
                f" | dp_steps={privacy_steps_total} | epsilon={privacy_epsilon:.4f}"
                if dp_enabled
                else ""
            ),
            flush=True,
        )

        if checkpoint_callback is not None:
            checkpoint_callback(epoch, list(history), stage_time_offset + (time.time() - t0), bool(is_best))

        if early_stopping(va_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.regressor_delayed.load_state_dict(best_state)

    for p in model.parameters():
        p.requires_grad = True

    return history, stage_time_offset + (time.time() - t0)


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
    start_epoch: int = 1,
    history_init: Optional[List[Dict]] = None,
    stage_time_offset: float = 0.0,
    checkpoint_callback: Optional[Callable[[int, List[Dict], float, bool], None]] = None,
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

    history: List[Dict] = list(history_init) if history_init is not None else []
    best_val = float("inf")
    best_state = None
    edge_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    scaler_amp = GradScaler(enabled=(use_amp and not dp_enabled))
    completed_privacy_steps = _history_privacy_steps(
        history,
        stage=3,
        default_steps_per_epoch=effective_train_steps,
    )
    running_privacy_steps = int(completed_privacy_steps)

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

    for epoch in range(start_epoch, epochs + 1):
        ep_t0 = time.time()
        model.train()
        tr_losses: List[float] = []
        tr_masks: List[float] = []
        epoch_privacy_steps = 0

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
                    edge_indices_1=(
                        _get_batched_edges(edge_cache, edge_index_adj, edge_index_od, edge_index_od_t, n_nodes, 1)
                        if model.regressor_requires_edges and feature_train_loader is None
                        else None
                    ),
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
                    preds = _apply_regressor(
                        model.regressor_nondelayed,
                        feat_batch,
                        edge_indices=(bei_adj, bei_od, bei_od_t)
                        if model.regressor_requires_edges and feature_train_loader is None
                        else None,
                    )
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

            if dp_enabled:
                epoch_privacy_steps += 1

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
                    preds = _apply_regressor(
                        model.regressor_nondelayed,
                        feat_batch,
                        edge_indices=(bei_adj, bei_od, bei_od_t)
                        if model.regressor_requires_edges and feature_val_loader is None
                        else None,
                    )
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

        is_best = False
        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.cpu().clone() for k, v in model.regressor_nondelayed.state_dict().items()}
            is_best = True

        ep_sec = time.time() - ep_t0
        if dp_enabled:
            running_privacy_steps += int(epoch_privacy_steps)
            privacy_steps_total = int(privacy_steps_offset + running_privacy_steps)
        else:
            privacy_steps_total = 0
        privacy_epsilon = _privacy_epsilon_for_steps(
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            steps=privacy_steps_total,
            delta=target_delta,
        ) if dp_enabled else 0.0
        history.append(
            {
                "stage": 3,
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_mask_ratio": tr_mask,
                "val_loss": va_loss,
                "val_mask_ratio": va_mask,
                "is_best": int(is_best),
                "epoch_time_seconds": ep_sec,
                "privacy_steps_in_epoch": int(epoch_privacy_steps),
                "privacy_steps_total": int(privacy_steps_total),
                "epsilon_approx": privacy_epsilon,
            }
        )
        print(
            f"Epoch {epoch}/{epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | "
            f"mask={tr_mask*100:.1f}% (val {va_mask*100:.1f}%) | sec={ep_sec:.1f}"
            + (
                f" | dp_steps={privacy_steps_total} | epsilon={privacy_epsilon:.4f}"
                if dp_enabled
                else ""
            ),
            flush=True,
        )

        if checkpoint_callback is not None:
            checkpoint_callback(epoch, list(history), stage_time_offset + (time.time() - t0), bool(is_best))

        if early_stopping(va_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.regressor_nondelayed.load_state_dict(best_state)

    for p in model.parameters():
        p.requires_grad = True

    return history, stage_time_offset + (time.time() - t0)


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
    gating_k: float,
    delay_threshold: float,
    out_dir: str,
    history: List[Dict],
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    dp_enabled: bool,
    epsilon_target: float,
    epsilon_approx_final: float,
    privacy_steps_final: int,
    delta: float,
    noise_multiplier: float,
    max_grad_norm: float,
    use_amp: bool,
    max_test_batches: int,
    cli_args: Optional[Dict[str, Any]] = None,
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

    # Soft gating blend
    weight = 1.0 / (1.0 + np.exp(-gating_k * (cls_probs_np - class_threshold)))
    if weight.ndim < pred_delayed_np.ndim:
        weight = np.expand_dims(weight, axis=-1)  # broadcast over output features only if dimensions mismatch
    reg_pred_np = weight * pred_delayed_np + (1.0 - weight) * pred_nondelayed_np
    reg_pred_np = np.maximum(0.0, reg_pred_np) # truncate negatives

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

    preds_arr = preds_denorm[..., 0].reshape(-1) if preds_denorm.shape[-1] >= 2 else pred_flat
    tars_arr = tars_denorm[..., 0].reshape(-1) if tars_denorm.shape[-1] >= 2 else tar_flat
    preds_dep = preds_denorm[..., 1].reshape(-1) if preds_denorm.shape[-1] >= 2 else pred_flat
    tars_dep = tars_denorm[..., 1].reshape(-1) if tars_denorm.shape[-1] >= 2 else tar_flat

    del_mask_arr = tars_arr > delay_threshold
    ndel_mask_arr = ~del_mask_arr
    del_mask_dep = tars_dep > delay_threshold
    ndel_mask_dep = ~del_mask_dep

    mae_del_arr, rmse_del_arr = _mae_rmse(tars_arr[del_mask_arr], preds_arr[del_mask_arr])
    mae_nd_arr, rmse_nd_arr = _mae_rmse(tars_arr[ndel_mask_arr], preds_arr[ndel_mask_arr])
    mae_del_dep, rmse_del_dep = _mae_rmse(tars_dep[del_mask_dep], preds_dep[del_mask_dep])
    mae_nd_dep, rmse_nd_dep = _mae_rmse(tars_dep[ndel_mask_dep], preds_dep[ndel_mask_dep])

    mae_arr, rmse_arr = _mae_rmse(tars_arr, preds_arr)
    mae_dep, rmse_dep = _mae_rmse(tars_dep, preds_dep)

    print("\n" + "=" * 80)
    print("FINAL EVALUATION")
    print("=" * 80)
    print(f"Classification F1: {test_cls['f1']:.4f} (arr={test_cls['f1_arrival']:.4f}, dep={test_cls['f1_departure']:.4f})")
    print(f"Regression delayed>thr: MAE={mae_delayed:.4f}, RMSE={rmse_delayed:.4f}")
    print(f"  --> Arrival delayed : MAE={mae_del_arr:.4f}, RMSE={rmse_del_arr:.4f}")
    print(f"  --> Departure delayed : MAE={mae_del_dep:.4f}, RMSE={rmse_del_dep:.4f}")
    print(f"Regression non-delayed: MAE={mae_nd:.4f}, RMSE={rmse_nd:.4f}")
    print(f"  --> Arrival non-delayed: MAE={mae_nd_arr:.4f}, RMSE={rmse_nd_arr:.4f}")
    print(f"  --> Departure non-delayed: MAE={mae_nd_dep:.4f}, RMSE={rmse_nd_dep:.4f}")
    print(f"Regression overall: MAE={mae_all:.4f}, RMSE={rmse_all:.4f}")
    print(f"  --> Arrival overall   : MAE={mae_arr:.4f}, RMSE={rmse_arr:.4f}")
    print(f"  --> Departure overall : MAE={mae_dep:.4f}, RMSE={rmse_dep:.4f}")
    if dp_enabled:
        print(f"Privacy: steps={privacy_steps_final} | target_epsilon={epsilon_target:.4f} | epsilon={epsilon_approx_final:.4f} | delta={delta:.1e}")

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
        w.writerow(["regression_mae_delayed_arrival", f"{mae_del_arr:.6f}"])
        w.writerow(["regression_rmse_delayed_arrival", f"{rmse_del_arr:.6f}"])
        w.writerow(["regression_mae_delayed_departure", f"{mae_del_dep:.6f}"])
        w.writerow(["regression_rmse_delayed_departure", f"{rmse_del_dep:.6f}"])
        w.writerow(["regression_mae_nondelayed", f"{mae_nd:.6f}"])
        w.writerow(["regression_rmse_nondelayed", f"{rmse_nd:.6f}"])
        w.writerow(["regression_mae_nondelayed_arrival", f"{mae_nd_arr:.6f}"])
        w.writerow(["regression_rmse_nondelayed_arrival", f"{rmse_nd_arr:.6f}"])
        w.writerow(["regression_mae_nondelayed_departure", f"{mae_nd_dep:.6f}"])
        w.writerow(["regression_rmse_nondelayed_departure", f"{rmse_nd_dep:.6f}"])
        w.writerow(["regression_mae_overall", f"{mae_all:.6f}"])
        w.writerow(["regression_rmse_overall", f"{rmse_all:.6f}"])
        w.writerow(["regression_mae_overall_arrival", f"{mae_arr:.6f}"])
        w.writerow(["regression_rmse_overall_arrival", f"{rmse_arr:.6f}"])
        w.writerow(["regression_mae_overall_departure", f"{mae_dep:.6f}"])
        w.writerow(["regression_rmse_overall_departure", f"{rmse_dep:.6f}"])
        w.writerow(["stage1_time_seconds", f"{stage1_time:.2f}"])
        w.writerow(["stage2_time_seconds", f"{stage2_time:.2f}"])
        w.writerow(["stage3_time_seconds", f"{stage3_time:.2f}"])
        w.writerow(["total_time_seconds", f"{(stage1_time + stage2_time + stage3_time):.2f}"])
        w.writerow(["dp_enabled", int(dp_enabled)])
        w.writerow(["privacy_steps_total", int(privacy_steps_final)])
        w.writerow(["epsilon_target", f"{epsilon_target:.6f}"])
        w.writerow(["epsilon_approx_final", f"{epsilon_approx_final:.6f}"])
        w.writerow(["delta", f"{delta:.12f}"])
        w.writerow(["noise_multiplier", f"{noise_multiplier:.6f}"])
        w.writerow(["max_grad_norm", f"{max_grad_norm:.6f}"])

        if cli_args:
            w.writerow(["__cli_args__", ""])
            for key in sorted(cli_args.keys()):
                value = cli_args.get(key)
                if isinstance(value, (dict, list, tuple)):
                    try:
                        value_str = json.dumps(value)
                    except TypeError:
                        value_str = str(value)
                else:
                    value_str = str(value)
                w.writerow([f"arg.{key}", value_str])

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
    p.add_argument("--gating_k", type=float, default=5.0, help="Gating steepness for soft gating")

    p.add_argument("--gru_dim", type=int, default=64)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--gru_heads", type=int, default=4)
    p.add_argument("--gat_hidden", type=int, default=64)
    p.add_argument("--gat_heads", type=int, default=2)
    p.add_argument("--classifier", type=str, default="TSiTPlus", choices=["TSiTPlus", "ConvTranPlus"])
    p.add_argument(
        "--regressor",
        type=str,
        default="nbeats",
        choices=[
            # "mlp",
            "deep_mlp",
            "residual_mlp",
            "gru",
            "tsit",
            "convtran",
            "tft",
            "nbeats",
            "node_transformer",
            "graph_gat",
        ],
        help="Regressor head used for delayed/non-delayed stages",
    )
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--chunk_size", type=int, default=200)

    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=20)
    p.add_argument("--stage3_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--stage1_lr", type=float, default=1e-4)
    p.add_argument("--stage2_lr", type=float, default=1e-4)
    p.add_argument("--stage3_lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=6)

    p.add_argument("--dp", action="store_true", default=True, help="Enable manual DP-SGD for stage training")
    p.add_argument("--epsilon", type=float, default=15)
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
    p.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (PyTorch 2.x)")
    p.add_argument(
        "--compile_mode",
        type=str,
        default="reduce-overhead",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="torch.compile mode (ignored if --compile is off)",
    )
    p.add_argument("--fused_adamw", action="store_true", help="Use fused AdamW optimizer on CUDA when available")
    p.add_argument(
        "--no_cache_stage23_features",
        default=True,
        action="store_true",
        help="Disable feature caching for stage 2/3 (slower, but lower host RAM usage)",
    )
    p.add_argument("--max_train_batches", type=int, default=0, help="Limit train batches per epoch for quick benchmarks (0=all)")
    p.add_argument("--max_val_batches", type=int, default=0, help="Limit val batches per epoch for quick benchmarks (0=all)")
    p.add_argument("--max_test_batches", type=int, default=0, help="Limit test batches for quick benchmarks (0=all)")
    p.add_argument(
        "--checkpoint_dir",
        type=str,
        default="auto",
        help="Directory for progress checkpoints ('auto' creates ./checkpoints/<run_name>)",
    )
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        # default="D:\\flight delay\\stpn paper\\STPN-main\\checkpoints\\stacked_gru_three_stage_20260412_002258\\checkpoint_stage3_epoch3.pt",
        help=(
            "Path to a saved checkpoint .pt file OR a directory containing checkpoints to resume training. "
            "If a directory is provided, the script will try best_stage{start_stage}.pt then latest_checkpoint.pt."
        ),
    )
    p.add_argument(
        "--checkpoint_every",
        type=int,
        default=1,
        help="Save a numbered checkpoint every N epochs (latest is always updated). Use 0 to disable numbered checkpoints.",
    )
    p.add_argument("--output_dir", type=str, default="auto")

    p.add_argument(
        "--start_stage",
        type=int,
        default=1,
        choices=[1, 2, 3],
        help="Force starting from this stage (skips earlier stages even if incomplete). Useful with --resume_checkpoint.",
    )
    p.add_argument(
        "--eval_only_model",
        type=str,
        default="",
        help="Path to a single model file to load. Will skip all training and only run `final_evaluation`.",
    )
    return p.parse_args()


def _resolve_resume_checkpoint(path_or_dir: str, *, start_stage: int) -> str:
    """Resolve a resume checkpoint.

    Accepts either:
    - a direct path to a .pt file, or
    - a directory that contains checkpoint files (e.g., a "checkpoints" folder).

    For directories, prefers stage-appropriate best checkpoints.
    """
    p = os.path.abspath(str(path_or_dir))
    if os.path.isfile(p):
        return p

    if not os.path.isdir(p):
        raise FileNotFoundError(f"Checkpoint not found: {p}")

    preferred: List[str] = []
    if int(start_stage) >= 3:
        preferred.append(os.path.join(p, "best_stage3.pt"))
    if int(start_stage) >= 2:
        preferred.append(os.path.join(p, "best_stage2.pt"))
    preferred.extend(
        [
            os.path.join(p, "best_stage1.pt"),
            os.path.join(p, "latest_checkpoint.pt"),
        ]
    )

    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate

    # Fall back to the newest .pt file in the directory.
    try:
        pt_files = [
            os.path.join(p, f)
            for f in os.listdir(p)
            if f.lower().endswith(".pt") and os.path.isfile(os.path.join(p, f))
        ]
        if pt_files:
            pt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return pt_files[0]
    except Exception:
        pass

    raise FileNotFoundError(
        "No checkpoint .pt files found under directory: " + p + "\n"
        "Expected one of: best_stage3.pt, best_stage2.pt, best_stage1.pt, latest_checkpoint.pt"
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    is_colab = False
    try:
        import google.colab
        is_colab = True
    except ImportError:
        pass

    if is_colab:
        # from google.colab import drive
        # drive.mount('/content/drive')
        print("[COLAB] Running in Google Colab, default paths adjusted to Drive.")

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

    print(
        "[DATA] "
        f"train_samples={len(train_loader.dataset)} | val_samples={len(val_loader.dataset)} | test_samples={len(test_loader.dataset)} | "
        f"batch_size={args.batch_size} | train_batches={len(train_loader)} | val_batches={len(val_loader)} | test_batches={len(test_loader)} | "
        f"max_train_batches={args.max_train_batches} | max_val_batches={args.max_val_batches} | effective_train_steps/epoch={effective_train_steps}",
        flush=True,
    )

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

    print(
        f"[PERF] AMP={'on' if use_amp else 'off'} | "
        f"cache_stage23_features={'on' if cache_stage23_features else 'off'} | "
        f"compile={'on' if args.compile else 'off'} | fused_adamw={'on' if args.fused_adamw else 'off'}"
    )

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

    if model.regressor_requires_edges and cache_stage23_features:
        cache_stage23_features = False
        print("[PERF] stage2/3 feature cache auto-disabled for edge-aware regressor")

    print(f"[MODEL] classifier={args.classifier} | regressor={args.regressor}")

    if args.compile:
        try:
            model = torch.compile(model, mode=str(args.compile_mode))
            print(f"[PERF] torch.compile enabled (mode={args.compile_mode})")
        except Exception as e:
            print(f"[PERF] torch.compile requested but unavailable/failed: {e}")

    optimizer_kwargs = dict(lr=args.lr, weight_decay=1e-4)
    if args.fused_adamw and device.type == "cuda":
        # Fused optimizers can be faster but require supported PyTorch/CUDA builds.
        try:
            optimizer = torch.optim.AdamW(model.parameters(), fused=True, **optimizer_kwargs)
            print("[PERF] Using fused AdamW")
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            print("[PERF] Fused AdamW not supported in this PyTorch build; using standard AdamW")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)

    history_state = {
        "h1": [],
        "h2": [],
        "h3": [],
        "t1": 0.0,
        "t2": 0.0,
        "t3": 0.0,
    }
    resume_stage = 1
    resume_epoch = 0
    loaded_stage = 1
    loaded_epoch = 0

    resume_checkpoint_path = ""
    if args.resume_checkpoint:
        resume_checkpoint_path = _resolve_resume_checkpoint(str(args.resume_checkpoint), start_stage=int(args.start_stage))
        if os.path.abspath(str(args.resume_checkpoint)) != os.path.abspath(resume_checkpoint_path):
            print(f"[RESUME] Resolved resume checkpoint to: {resume_checkpoint_path}")
        ckpt = load_training_checkpoint(resume_checkpoint_path, device)
        resume_stage = int(ckpt.get("stage", 1))
        resume_epoch = int(ckpt.get("epoch", 0))
        loaded_stage, loaded_epoch = resume_stage, resume_epoch
        history_state["h1"] = list(ckpt.get("history_stage1", []))
        history_state["h2"] = list(ckpt.get("history_stage2", []))
        history_state["h3"] = list(ckpt.get("history_stage3", []))
        history_state["t1"] = float(ckpt.get("stage1_time", 0.0))
        history_state["t2"] = float(ckpt.get("stage2_time", 0.0))
        history_state["t3"] = float(ckpt.get("stage3_time", 0.0))
        try:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            print(f"[RESUME] Loaded full checkpoint state: stage={resume_stage}, epoch={resume_epoch}")
        except RuntimeError:
            loaded, skipped = _load_compatible_model_state(model, ckpt["model_state"])
            optimizer = torch.optim.AdamW(model.parameters(), **optimizer_kwargs)
            print(
                f"[RESUME] Loaded partial checkpoint state: stage={resume_stage}, epoch={resume_epoch}, "
                f"matched_tensors={loaded}, skipped_tensors={skipped}"
            )
            print("[RESUME] Reinitialized optimizer because checkpoint architecture does not fully match current model.")
        if args.dp:
            sigma = float(ckpt.get("sigma", sigma))
            epsilon_final = float(ckpt.get("epsilon_final", epsilon_final))
        print(f"[RESUME] Checkpoint metadata: stage={resume_stage}, epoch={resume_epoch}")

    # If user requests skipping earlier stages, bump resume_stage accordingly.
    # We keep the loaded model weights, but reset epoch counters for the forced stage.
    if int(args.start_stage) > int(resume_stage):
        print(f"[RESUME] Forcing start at stage={args.start_stage} (skipping stages < {args.start_stage}).")
        resume_stage = int(args.start_stage)
        resume_epoch = 0

    def _max_epoch(hist: List[Dict]) -> int:
        if not hist:
            return 0
        return int(max(int(r.get("epoch", 0)) for r in hist))

    completed_s1_epochs = _max_epoch(history_state["h1"])
    completed_s2_epochs = _max_epoch(history_state["h2"])
    completed_s3_epochs = _max_epoch(history_state["h3"])

    # If the loaded checkpoint is mid-stage and history is empty, fall back to its epoch.
    if completed_s1_epochs == 0 and loaded_stage == 1:
        completed_s1_epochs = int(loaded_epoch)
    if completed_s2_epochs == 0 and loaded_stage == 2:
        completed_s2_epochs = int(loaded_epoch)
    if completed_s3_epochs == 0 and loaded_stage == 3:
        completed_s3_epochs = int(loaded_epoch)

    checkpoint_dir_arg = args.checkpoint_dir.strip()
    checkpoint_dir = ""
    if checkpoint_dir_arg.lower() == "auto" or not checkpoint_dir_arg:
        if resume_checkpoint_path:
            checkpoint_dir = os.path.dirname(os.path.abspath(resume_checkpoint_path))
        else:
            ckpt_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if is_colab:
                checkpoint_dir = f"/content/drive/MyDrive/stpn_ckpts/stacked_gru_three_stage_{ckpt_ts}"
            else:
                checkpoint_dir = os.path.join(os.getcwd(), "checkpoints", f"stacked_gru_three_stage_{ckpt_ts}")
    else:
        checkpoint_dir = checkpoint_dir_arg

    if checkpoint_dir:
        checkpoint_dir = os.path.abspath(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"[CHECKPOINT] Saving to: {checkpoint_dir}")

    def _current_privacy_steps() -> int:
        return (
            _history_privacy_steps(history_state["h1"], stage=1, default_steps_per_epoch=effective_train_steps)
            + _history_privacy_steps(history_state["h2"], stage=2, default_steps_per_epoch=effective_train_steps)
            + _history_privacy_steps(history_state["h3"], stage=3, default_steps_per_epoch=effective_train_steps)
        )

    def _write_checkpoint(stage: int, epoch: int, *, is_best: bool = False, best_metric_name: str = "", best_metric_value: float = 0.0) -> None:
        if not checkpoint_dir:
            return
        current_privacy_steps = _current_privacy_steps() if args.dp else 0
        current_epsilon = _privacy_epsilon_for_steps(
            noise_multiplier=sigma,
            sample_rate=sample_rate,
            steps=current_privacy_steps,
            delta=args.delta,
        ) if args.dp else 0.0
        save_training_checkpoint(
            checkpoint_dir=checkpoint_dir,
            stage=stage,
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            history_stage1=history_state["h1"],
            history_stage2=history_state["h2"],
            history_stage3=history_state["h3"],
            stage1_time=history_state["t1"],
            stage2_time=history_state["t2"],
            stage3_time=history_state["t3"],
            sigma=sigma,
            epsilon_final=current_epsilon,
            privacy_steps=current_privacy_steps,
            save_every=int(args.checkpoint_every),
            is_best=bool(is_best),
            best_metric_name=str(best_metric_name),
            best_metric_value=float(best_metric_value),
        )

    s1_lr = args.stage1_lr if args.stage1_lr is not None else args.lr
    s2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
    s3_lr = args.stage3_lr if args.stage3_lr is not None else (args.lr * 0.2)

    ei_adj = edge_index_adj.to(device)
    ei_od = edge_index_od.to(device)
    ei_od_t = edge_index_od_t.to(device)

    if getattr(args, "eval_only_model", ""):
        print(f"[EVAL ONLY] Loading model from: {args.eval_only_model}")
        ckpt = load_training_checkpoint(args.eval_only_model, device)
        state_dict = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
        compat_loaded, skipped = _load_compatible_model_state(model, state_dict)
        print(f"Loaded {compat_loaded} parameters, skipped {skipped}.")
        
        base_dir = os.path.dirname(args.eval_only_model) or "."
        model_name = os.path.basename(args.eval_only_model)
        out_dir = os.path.join(base_dir, f"eval_results_{model_name}")
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
            args.gating_k,
            args.delay_threshold,
            out_dir,
            [],
            0.0,
            0.0,
            0.0,
            args.dp,
            args.epsilon,
            0.0,
            0,
            args.delta,
            sigma,
            args.max_grad_norm,
            use_amp,
            args.max_test_batches,
            vars(args)
        )
        return

    stage1_done = (resume_stage > 1) or (resume_stage == 1 and resume_epoch >= args.stage1_epochs)
    if stage1_done:
        h1, t1 = history_state["h1"], history_state["t1"]
        print("[RESUME] Stage 1 already complete; skipping.")
    else:
        s1_start = (resume_epoch + 1) if resume_stage == 1 else 1

        def _s1_ckpt(ep: int, hist: List[Dict], elapsed: float, is_best: bool) -> None:
            history_state["h1"] = hist
            history_state["t1"] = elapsed
            best_metric_value = 0.0
            if hist:
                try:
                    best_metric_value = float(hist[-1].get("val_f1", 0.0))
                except Exception:
                    best_metric_value = 0.0
            _write_checkpoint(1, ep, is_best=bool(is_best), best_metric_name="val_f1", best_metric_value=best_metric_value)

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
            start_epoch=s1_start,
            history_init=history_state["h1"],
            stage_time_offset=history_state["t1"],
            checkpoint_callback=_s1_ckpt,
        )
        history_state["h1"], history_state["t1"] = h1, t1

    completed_s1_privacy_steps = _history_privacy_steps(history_state["h1"], stage=1, default_steps_per_epoch=effective_train_steps)
    if completed_s1_privacy_steps == 0 and loaded_stage == 1:
        completed_s1_privacy_steps = int(loaded_epoch) * int(effective_train_steps)

    stage2_done = (resume_stage > 2) or (resume_stage == 2 and resume_epoch >= args.stage2_epochs)
    if stage2_done:
        h2, t2 = history_state["h2"], history_state["t2"]
        print("[RESUME] Stage 2 already complete; skipping.")
    else:
        s2_start = (resume_epoch + 1) if resume_stage == 2 else 1

        def _s2_ckpt(ep: int, hist: List[Dict], elapsed: float, is_best: bool) -> None:
            history_state["h2"] = hist
            history_state["t2"] = elapsed
            best_metric_value = 0.0
            if hist:
                try:
                    best_metric_value = float(hist[-1].get("val_loss", 0.0))
                except Exception:
                    best_metric_value = 0.0
            _write_checkpoint(2, ep, is_best=bool(is_best), best_metric_name="val_loss", best_metric_value=best_metric_value)

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
            int(completed_s1_privacy_steps),
            use_amp,
            cache_stage23_features,
            worker_count,
            args.max_train_batches,
            args.max_val_batches,
            effective_train_steps,
            start_epoch=s2_start,
            history_init=history_state["h2"],
            stage_time_offset=history_state["t2"],
            checkpoint_callback=_s2_ckpt,
        )
        history_state["h2"], history_state["t2"] = h2, t2

    completed_s2_privacy_steps = _history_privacy_steps(history_state["h2"], stage=2, default_steps_per_epoch=effective_train_steps)
    if completed_s2_privacy_steps == 0 and loaded_stage == 2:
        completed_s2_privacy_steps = int(loaded_epoch) * int(effective_train_steps)

    stage3_done = (resume_stage > 3) or (resume_stage == 3 and resume_epoch >= args.stage3_epochs)
    if stage3_done:
        h3, t3 = history_state["h3"], history_state["t3"]
        print("[RESUME] Stage 3 already complete; skipping.")
    else:
        s3_start = (resume_epoch + 1) if resume_stage == 3 else 1

        def _s3_ckpt(ep: int, hist: List[Dict], elapsed: float, is_best: bool) -> None:
            history_state["h3"] = hist
            history_state["t3"] = elapsed
            best_metric_value = 0.0
            if hist:
                try:
                    best_metric_value = float(hist[-1].get("val_loss", 0.0))
                except Exception:
                    best_metric_value = 0.0
            _write_checkpoint(3, ep, is_best=bool(is_best), best_metric_name="val_loss", best_metric_value=best_metric_value)

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
            int(completed_s1_privacy_steps) + int(completed_s2_privacy_steps),
            use_amp,
            cache_stage23_features,
            worker_count,
            args.max_train_batches,
            args.max_val_batches,
            effective_train_steps,
            start_epoch=s3_start,
            history_init=history_state["h3"],
            stage_time_offset=history_state["t3"],
            checkpoint_callback=_s3_ckpt,
        )
        history_state["h3"], history_state["t3"] = h3, t3

    completed_s3_privacy_steps = _history_privacy_steps(history_state["h3"], stage=3, default_steps_per_epoch=effective_train_steps)
    if completed_s3_privacy_steps == 0 and loaded_stage == 3:
        completed_s3_privacy_steps = int(loaded_epoch) * int(effective_train_steps)

    final_privacy_steps = int(completed_s1_privacy_steps + completed_s2_privacy_steps + completed_s3_privacy_steps)
    epsilon_final_actual = _privacy_epsilon_for_steps(
        noise_multiplier=sigma,
        sample_rate=sample_rate,
        steps=final_privacy_steps,
        delta=args.delta,
    ) if args.dp else 0.0

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir != "auto":
        out_dir = args.output_dir
    else:
        if is_colab:
            out_dir = f"/content/drive/MyDrive/stpn_results/stacked_gru_three_stage_{ts}"
        else:
            out_dir = f"stacked_gru_three_stage_{ts}"
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
        args.gating_k,
        args.delay_threshold,
        out_dir,
        h1 + h2 + h3,
        t1,
        t2,
        t3,
        args.dp,
        float(args.epsilon),
        float(epsilon_final_actual),
        int(final_privacy_steps),
        float(args.delta),
        float(sigma),
        float(args.max_grad_norm),
        use_amp,
        args.max_test_batches,
        cli_args=dict(vars(args)),
    )


if __name__ == "__main__":
    main()
