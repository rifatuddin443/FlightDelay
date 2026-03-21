"""
Stacked GRUAttention Encoder → TSiTPlus / ConvTranPlus Classifier
==================================================================
Three-stage spatiotemporal architecture for flight delay classification:

  Stage 1  GRUAttention Encoder
           Bidirectional GRU + multi-head self-attention → outputs a full
           hidden-state *sequence* per node: (B*N, gru_dim, T)

  Stage 2  Multi-Edge GAT  (adj / OD / OD_t)
           Operates on mean-pooled GRU embeddings per node.  GAT output
           is broadcast back to each timestep and concatenated with the
           GRU sequence → graph-enriched sequence (B*N, gru_dim + gat_dim, T)

  Stage 3  Transformer Classifier  (TSiTPlus or ConvTranPlus)
           Classifies from the enriched sequence → (B*N, c_out)

Data flow (per batch of B graph snapshots, N=50 nodes):
  (B, N, c_in, T) → flatten to (B*N, c_in, T)
       ↓ GRU encoder
  (B*N, gru_dim, T)               ← temporal features per timestep
       ↓ mean-pool over T → (B*N, gru_dim) → GAT
  (B*N, gat_dim)                   ← graph context per node
       ↓ broadcast to T and concat with GRU seq
  (B*N, gru_dim + gat_dim, T)     ← graph-enriched sequence
       ↓ TSiTPlus or ConvTranPlus
  (B*N, c_out)                     ← node-level binary logits
       ↓ reshape
  (B, N, c_out)

Usage:
    python stacked_gru_transformer.py --classifier TSiTPlus --epochs 50
    python stacked_gru_transformer.py --classifier ConvTranPlus --epochs 50
    python stacked_gru_transformer.py --classifier both --epochs 50
"""
from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
import sys
import time
import traceback
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GATConv

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import EarlyStopping, load_flight_data, set_seed

try:
    from classifykat import KAN as _KAN  # type: ignore
    KAN_AVAILABLE = True
except ImportError:
    try:
        from efficient_kan import KAN as _KAN  # type: ignore
        KAN_AVAILABLE = True
    except ImportError:
        KAN_AVAILABLE = False

from classifykat_balanced import build_sequences_node_level


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — GRUAttention Encoder (sequence output)
# ═══════════════════════════════════════════════════════════════════════════════

class GRUAttentionEncoder(nn.Module):
    """
    Bidirectional GRU + multi-head self-attention that *preserves the time
    dimension*.  Output shape: (B, gru_dim, T) — a per-timestep feature map.

    Unlike tsai's GRUAttention which collapses to a single vector, this
    keeps the full sequence so a downstream transformer classifier can
    operate on it.
    """

    def __init__(
        self,
        c_in: int,
        gru_dim: int = 64,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.15,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.c_in = c_in
        self.gru_dim = gru_dim
        self.bidirectional = bidirectional

        # Input projection (channel mixing before GRU)
        self.input_proj = nn.Sequential(
            nn.Linear(c_in, gru_dim),
            nn.LayerNorm(gru_dim),
            nn.GELU(),
        )

        # Bidirectional GRU
        rnn_hidden = gru_dim // 2 if bidirectional else gru_dim
        self.gru = nn.GRU(
            input_size=gru_dim,
            hidden_size=rnn_hidden,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Multi-head self-attention over time
        self.attn = nn.MultiheadAttention(
            embed_dim=gru_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(gru_dim)

        # Feed-forward block (applied per timestep)
        self.ffn = nn.Sequential(
            nn.Linear(gru_dim, gru_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_dim * 2, gru_dim),
        )
        self.ffn_norm = nn.LayerNorm(gru_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, c_in, T)  — standard tsai channel-first format
        Returns:
            (B, gru_dim, T)  — per-timestep hidden features
        """
        B, C, T = x.shape

        # (B, C, T) → (B, T, C) → project → (B, T, gru_dim)
        x_t = x.permute(0, 2, 1)
        h = self.input_proj(x_t)  # (B, T, gru_dim)

        # GRU sequence → (B, T, gru_dim)
        h, _ = self.gru(h)  # bidir concat → (B, T, gru_dim)

        # Self-attention over time (with residual)
        attn_out, _ = self.attn(h, h, h)
        h = self.attn_norm(h + self.dropout(attn_out))

        # Feed-forward (with residual)
        h = self.ffn_norm(h + self.dropout(self.ffn(h)))

        # (B, T, gru_dim) → (B, gru_dim, T)  — channel-first for downstream
        return h.permute(0, 2, 1).contiguous()


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — Multi-Edge GAT Fusion (same as hybrid_graph_tsai.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """KAN head if available, else 2-layer MLP."""
    if KAN_AVAILABLE:
        return _KAN(
            layers_hidden=[in_dim, hidden_dim, out_dim],
            grid_size=3, spline_order=2,
        )
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.ReLU(),
        nn.Dropout(0.2), nn.Linear(hidden_dim, out_dim),
    )


class MultiEdgeGATFusion(nn.Module):
    """Three parallel GATConv (adj / OD / OD_t) + learned α-weighting + KAN fusion."""

    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 2):
        super().__init__()
        self.alpha_adj  = nn.Parameter(torch.tensor(1.0))
        self.alpha_od   = nn.Parameter(torch.tensor(1.0))
        self.alpha_od_t = nn.Parameter(torch.tensor(1.0))

        self.gat_adj  = GATConv(in_channels, hidden_channels, heads=heads,
                                concat=False, dropout=0.1)
        self.gat_od   = GATConv(in_channels, hidden_channels, heads=heads,
                                concat=False, dropout=0.1)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads,
                                concat=False, dropout=0.1)

        fusion_in = hidden_channels * 3 + 3
        self.fusion = _make_head(fusion_in, hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,              # (total_nodes, in_channels)
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        w = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
        w_adj, w_od, w_od_t = w

        x_adj  = self.gat_adj (x, edge_index_adj)
        x_od   = self.gat_od  (x, edge_index_od)
        x_od_t = self.gat_od_t(x, edge_index_od_t)

        n = x_adj.size(0)
        scalars = torch.stack([
            w_adj.expand(n), w_od.expand(n), w_od_t.expand(n),
        ], dim=1)

        fused = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
        return self.dropout(F.relu(self.fusion(fused)))


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — Transformer Classifier (TSiTPlus or ConvTranPlus)
# ═══════════════════════════════════════════════════════════════════════════════

class _FlattenHead(nn.Module):
    """Flatten non-(B, D) tsai outputs and project to target dim."""
    def __init__(self, backbone: nn.Module, flat_dim: int, out_dim: int):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(nn.LayerNorm(flat_dim), nn.Linear(flat_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return self.proj(out.flatten(1))


def build_classifier(
    name: str,
    c_in: int,
    c_out: int,
    seq_len: int,
) -> nn.Module:
    """Build TSiTPlus or ConvTranPlus as the final classifier stage."""
    registry = {
        "TSiTPlus":     ("tsai.models.TSiTPlus",     "TSiTPlus"),
        "ConvTranPlus": ("tsai.models.ConvTranPlus",  "ConvTranPlus"),
    }
    if name not in registry:
        raise ValueError(f"Classifier must be TSiTPlus or ConvTranPlus, got {name}")

    mod_path, cls_name = registry[name]
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    model = cls(c_in=c_in, c_out=c_out, seq_len=seq_len)

    # Probe output shape and wrap if needed
    dummy = torch.randn(2, c_in, seq_len)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]

    expected = torch.Size([2, c_out])
    if out.shape != expected:
        flat_dim = int(out.flatten(1).shape[1])
        print(f"    [classifier-wrap] {name}: {tuple(out.shape)} "
              f"→ flatten({flat_dim}) → Linear({flat_dim}, {c_out})")
        model = _FlattenHead(model, flat_dim, c_out)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# FULL STACKED MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class StackedGRUTransformer(nn.Module):
    """
    GRUAttention encoder → Multi-edge GAT → TSiTPlus/ConvTranPlus classifier.

    Data flow:
      (B, N, c_in, T) → reshape (B*N, c_in, T)
           ↓ GRUAttentionEncoder
      (B*N, gru_dim, T)                    ← temporal features per timestep
           ↓ mean-pool over T
      (B*N, gru_dim) → reshape (B, N, gru_dim)
           ↓ batched multi-edge GAT
      (B*N, gat_dim)                       ← graph-aware node embeddings
           ↓ broadcast to T, concat with GRU sequence
      (B*N, gru_dim + gat_dim, T)          ← graph-enriched temporal features
           ↓ Transformer classifier (TSiTPlus or ConvTranPlus)
      (B*N, c_out) → reshape (B, N, c_out) ← node-level logits
    """

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
    ):
        super().__init__()
        self.c_out = c_out
        self.gru_dim = gru_dim
        self.gat_hidden = gat_hidden
        self.chunk_size = chunk_size  # max nodes processed at once through encoder/classifier

        # Stage 1: GRU encoder (sequence output)
        self.encoder = GRUAttentionEncoder(
            c_in=c_in, gru_dim=gru_dim, n_layers=gru_layers,
            n_heads=gru_heads, dropout=dropout,
        )

        # Stage 2: Multi-edge GAT
        self.gat = MultiEdgeGATFusion(
            in_channels=gru_dim, hidden_channels=gat_hidden, heads=gat_heads,
        )

        # Stage 3: Transformer classifier on enriched sequence
        enriched_dim = gru_dim + gat_hidden  # concat GRU seq + GAT broadcast
        self.classifier = build_classifier(
            classifier_name, c_in=enriched_dim, c_out=c_out, seq_len=seq_len,
        )

        self.classifier_name = classifier_name

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Run GRU encoder on a small chunk (gradient-checkpointed)."""
        if self.training:
            return grad_checkpoint(self.encoder, chunk, use_reentrant=False)
        return self.encoder(chunk)

    def _classify_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        """Run transformer classifier on a small chunk (gradient-checkpointed)."""
        if self.training:
            out = grad_checkpoint(self.classifier, chunk, use_reentrant=False)
        else:
            out = self.classifier(chunk)
        if isinstance(out, (tuple, list)):
            out = out[0]
        if out.dim() > 2:
            out = out.flatten(1)
        return out

    def forward(
        self,
        x: torch.Tensor,                  # (B, N, c_in, T)
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C, T = x.shape
        total = B * N
        cs = self.chunk_size

        # ── Stage 1: GRU encoder in chunks (saves ~60% peak memory) ──────────
        x_flat = x.view(total, C, T)
        gru_chunks = []
        for i in range(0, total, cs):
            gru_chunks.append(self._encode_chunk(x_flat[i : i + cs]))
        gru_seq = torch.cat(gru_chunks, dim=0)           # (B*N, gru_dim, T)
        del gru_chunks, x_flat

        # ── Stage 2: GAT on mean-pooled GRU output (small, no chunking) ──────
        gru_pooled = gru_seq.mean(dim=2)                  # (B*N, gru_dim)
        gat_out = self.gat(
            gru_pooled, edge_index_adj, edge_index_od, edge_index_od_t,
        )                                                 # (B*N, gat_hidden)
        del gru_pooled

        # ── Enrich: broadcast GAT output back to each timestep ───────────────
        gat_broadcast = gat_out.unsqueeze(2).expand(-1, -1, T)  # (B*N, gat_hidden, T)
        enriched = torch.cat([gru_seq, gat_broadcast], dim=1)   # (B*N, gru_dim+gat_hidden, T)
        del gru_seq, gat_broadcast, gat_out

        # ── Stage 3: Transformer classifier in chunks ────────────────────────
        logit_chunks = []
        for i in range(0, total, cs):
            logit_chunks.append(self._classify_chunk(enriched[i : i + cs]))
        logits = torch.cat(logit_chunks, dim=0)           # (B*N, c_out)
        del logit_chunks, enriched

        return logits.view(B, N, self.c_out)              # (B, N, c_out)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS (shared with hybrid_graph_tsai.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph_tensors(
    train_x, train_y_cls, val_x, val_y_cls,
    test_x, test_y_cls, seq_len: int, feature_dim: int,
) -> Tuple:
    """Reshape (n, n_nodes, seq_len*f) → (n, n_nodes, f, seq_len)."""
    def _reshape(X, Y):
        n, n_nodes, _ = X.shape
        Xg = X.view(n, n_nodes, seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
        return Xg, Y
    trX, trY = _reshape(train_x, train_y_cls)
    vaX, vaY = _reshape(val_x,   val_y_cls)
    teX, teY = _reshape(test_x,  test_y_cls)
    return trX, trY, vaX, vaY, teX, teY


def batch_edge_index(edge_index: torch.Tensor, n_nodes: int, batch_size: int) -> torch.Tensor:
    """Repeat edge_index B times with node offsets."""
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes
    ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
    ei = ei + offsets.view(batch_size, 1, 1)
    return ei.reshape(2, -1)


def classification_metrics_per_channel(
    preds: np.ndarray, targets: np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-channel binary metrics + macro averages."""
    p2 = preds.reshape(-1, preds.shape[-1])
    t2 = targets.reshape(-1, targets.shape[-1])
    metrics: Dict[str, float] = {}
    precs, recs, f1s, accs = [], [], [], []
    for c in range(p2.shape[1]):
        pb = p2[:, c] >= threshold
        tb = t2[:, c] >= 0.5
        tp = int(np.logical_and( pb,  tb).sum())
        fp = int(np.logical_and( pb, ~tb).sum())
        fn = int(np.logical_and(~pb,  tb).sum())
        tn = int(np.logical_and(~pb, ~tb).sum())
        pr = tp / (tp + fp + 1e-8)
        re = tp / (tp + fn + 1e-8)
        f1 = 2 * pr * re / (pr + re + 1e-8)
        ac = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        nm = channel_names[c] if c < len(channel_names) else f"ch{c}"
        metrics[f"precision_{nm}"] = pr
        metrics[f"recall_{nm}"]    = re
        metrics[f"f1_{nm}"]        = f1
        metrics[f"accuracy_{nm}"]  = ac
        precs.append(pr); recs.append(re); f1s.append(f1); accs.append(ac)
    metrics["precision"] = float(np.mean(precs))
    metrics["recall"]    = float(np.mean(recs))
    metrics["f1"]        = float(np.mean(f1s))
    metrics["accuracy"]  = float(np.mean(accs))
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model: StackedGRUTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    class_threshold: float = 0.5,
    save_path: Optional[str] = None,
    accumulation_steps: int = 4,
) -> Tuple[Dict[str, float], float]:
    """Train with BCEWithLogitsLoss, warmup + cosine LR, gradient accumulation.

    Three param groups with different learning rates:
      • GRU encoder:      lr       (already learned reasonable features)
      • GAT graph layers: lr * 3   (new graph structure to learn)
      • Classifier:       lr * 2   (transformer classifier fine-tuning)
    """
    ei_adj  = edge_index_adj.to(device)
    ei_od   = edge_index_od.to(device)
    ei_od_t = edge_index_od_t.to(device)

    # Three param groups with staged learning rates
    encoder_params    = list(model.encoder.parameters())
    gat_params        = list(model.gat.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params,    "lr": lr,       "weight_decay": 1e-4},
        {"params": gat_params,        "lr": lr * 3,   "weight_decay": 1e-5},
        {"params": classifier_params, "lr": lr * 2,   "weight_decay": 1e-4},
    ])

    # Warmup + cosine decay
    warmup_epochs = min(5, max(1, epochs // 6))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    es        = EarlyStopping(patience=patience, mode="max")

    best_f1, best_state = -1.0, None
    t0 = time.time()
    acc_steps = max(1, accumulation_steps)

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        ep_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        for step, (bx, by) in enumerate(train_loader, 1):
            B = bx.size(0)
            bx = bx.to(device)
            by = by.to(device)

            bei_adj  = batch_edge_index(ei_adj,  n_nodes, B)
            bei_od   = batch_edge_index(ei_od,   n_nodes, B)
            bei_od_t = batch_edge_index(ei_od_t, n_nodes, B)

            logits = model(bx, bei_adj, bei_od, bei_od_t)
            loss   = loss_fn(logits, by) / acc_steps
            loss.backward()
            ep_losses.append(loss.item() * acc_steps)

            if step % acc_steps == 0 or step == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                B = bx.size(0)
                bx = bx.to(device)
                bei_adj  = batch_edge_index(ei_adj,  n_nodes, B)
                bei_od   = batch_edge_index(ei_od,   n_nodes, B)
                bei_od_t = batch_edge_index(ei_od_t, n_nodes, B)
                logits = model(bx, bei_adj, bei_od, bei_od_t)
                vp.append(torch.sigmoid(logits).cpu())
                vt.append(by)

        vm = classification_metrics_per_channel(
            torch.cat(vp).numpy(), torch.cat(vt).numpy(), threshold=class_threshold,
        )

        if vm["f1"] > best_f1:
            best_f1    = vm["f1"]
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save({
                    "epoch":           epoch,
                    "model_state":     best_state,
                    "val_f1":          best_f1,
                    "classifier_name": model.classifier_name,
                    "gru_dim":         model.gru_dim,
                    "gat_hidden":      model.gat_hidden,
                }, save_path)

        if epoch % 5 == 0 or epoch == epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"loss={np.mean(ep_losses):.4f}  val_f1={vm['f1']:.4f}  "
                  f"lr={cur_lr:.2e}")

        if es(vm["f1"], epoch):
            print(f"    early stop @ epoch {epoch}  best_val_f1={best_f1:.4f}")
            break

    train_sec = time.time() - t0

    # ── save checkpoint ───────────────────────────────────────────────────────
    if save_path is not None and best_state is not None:
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["train_sec"] = train_sec
        torch.save(ckpt, save_path)
        print(f"  Saved best model -> {save_path}")

    # ── test ──────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    tp_list, tt_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            B = bx.size(0)
            bx = bx.to(device)
            bei_adj  = batch_edge_index(ei_adj,  n_nodes, B)
            bei_od   = batch_edge_index(ei_od,   n_nodes, B)
            bei_od_t = batch_edge_index(ei_od_t, n_nodes, B)
            logits = model(bx, bei_adj, bei_od, bei_od_t)
            tp_list.append(torch.sigmoid(logits).cpu())
            tt_list.append(by)

    return (
        classification_metrics_per_channel(
            torch.cat(tp_list).numpy(), torch.cat(tt_list).numpy(),
            threshold=class_threshold,
        ),
        train_sec,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stacked GRUAttention → TSiTPlus/ConvTranPlus classifier"
    )
    p.add_argument("--data_source",        default="cdata", choices=["cdata", "udata"])
    p.add_argument("--seq_len",            type=int,   default=18)
    p.add_argument("--horizons",           type=int,   nargs="+", default=[12])
    p.add_argument("--delay_threshold",    type=float, default=5.0)
    p.add_argument("--class_threshold",    type=float, default=0.5)
    # GRU encoder
    p.add_argument("--gru_dim",            type=int,   default=64,
                   help="GRU encoder hidden dim (output features per timestep)")
    p.add_argument("--gru_layers",         type=int,   default=2)
    p.add_argument("--gru_heads",          type=int,   default=4,
                   help="Number of self-attention heads in the GRU encoder")
    # GAT
    p.add_argument("--gat_hidden",         type=int,   default=64)
    p.add_argument("--gat_heads",          type=int,   default=2)
    # Classifier
    p.add_argument("--classifier",         type=str,   default="TSiTPlus",
                   choices=["TSiTPlus", "ConvTranPlus", "both"],
                   help="Which transformer to use as the classifier head")
    # Training
    p.add_argument("--weather_file",       type=str,   default="weather_cn.npy")
    p.add_argument("--period_hours",       type=int,   default=24)
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--batch_size",         type=int,   default=16,
                   help="Graph snapshots per batch (lower to save memory)")
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--patience",           type=int,   default=12)
    p.add_argument("--accumulation_steps", type=int,   default=16,
                   help="Grad accum steps (effective batch = batch_size * this = 256)")
    p.add_argument("--chunk_size",         type=int,   default=200,
                   help="Max nodes per forward chunk through encoder/classifier")
    p.add_argument("--dropout",            type=float, default=0.15)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--output_dir",         type=str,   default="auto")
    p.add_argument("--device",             type=str,   default="auto")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    classifiers = (
        ["TSiTPlus", "ConvTranPlus"] if args.classifier == "both"
        else [args.classifier]
    )

    print(f"\n{'='*65}")
    print(f"  Stacked GRUAttention → Transformer Classifier  |  device={device}")
    print(f"  Architecture: GRU encoder → multi-edge GAT → transformer head")
    print(f"  KAN available: {KAN_AVAILABLE}")
    print(f"  Classifiers: {', '.join(classifiers)}")
    print(f"{'='*65}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    print("[1/4] Loading data ...")
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

    # Remove time features (last 2 columns)
    train_inputs = train_inputs[:, :, :-2]
    val_inputs   = val_inputs[:, :, :-2]
    test_inputs  = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim   = train_delay_scaled.shape[2]
    max_horizon = sorted(set(args.horizons))[0]

    print(f"  Nodes: {num_nodes}  feature_dim: {feature_dim}  "
          f"delay_dim: {delay_dim}  seq_len: {args.seq_len}")

    # ── 2. Build graph-level sequences ────────────────────────────────────────
    print("[2/4] Building sequences ...")
    train_x, _, train_y_cls = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    val_x, _, val_y_cls = build_sequences_node_level(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    test_x, _, test_y_cls = build_sequences_node_level(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )

    trX, trY, vaX, vaY, teX, teY = build_graph_tensors(
        train_x, train_y_cls, val_x, val_y_cls,
        test_x, test_y_cls, args.seq_len, feature_dim,
    )
    n_nodes = trX.shape[1]

    print(f"  Train {tuple(trX.shape)}  Val {tuple(vaX.shape)}  Test {tuple(teX.shape)}")
    print(f"  c_in={feature_dim}  c_out={delay_dim}  n_nodes={n_nodes}")

    delayed_frac = trY.mean().item()
    print(f"  Class balance (train): {delayed_frac:.2%} delayed")

    pos_rate   = trY.reshape(-1, delay_dim).mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    trY = trY.float()
    vaY = vaY.float()
    teY = teY.float()

    train_loader = DataLoader(TensorDataset(trX, trY),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(TensorDataset(vaX, vaY),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(teX, teY),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── 3. Output directory ───────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir if args.output_dir != "auto"
           else f"stacked_gru_transformer_{ts}")
    os.makedirs(out, exist_ok=True)

    # ── 4. Train each classifier variant ──────────────────────────────────────
    print(f"\n[3/4] Training {len(classifiers)} stacked model(s) ...\n")

    results:  Dict[str, Dict[str, float]] = {}
    n_params: Dict[str, int]              = {}
    timings:  Dict[str, float]            = {}
    errors:   Dict[str, str]              = {}

    for idx, clf_name in enumerate(classifiers, 1):
        label = f"GRUAttn->{clf_name}"
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{len(classifiers)}] {label}")
        print(f"  GRU: dim={args.gru_dim}, layers={args.gru_layers}, "
              f"heads={args.gru_heads}")
        print(f"  GAT: hidden={args.gat_hidden}, heads={args.gat_heads}")
        print(f"  Classifier input: {args.gru_dim + args.gat_hidden} channels × "
              f"{args.seq_len} timesteps")
        print(f"{'─'*60}")

        try:
            model = StackedGRUTransformer(
                c_in=feature_dim,
                c_out=delay_dim,
                seq_len=args.seq_len,
                gru_dim=args.gru_dim,
                gru_layers=args.gru_layers,
                gru_heads=args.gru_heads,
                gat_hidden=args.gat_hidden,
                gat_heads=args.gat_heads,
                classifier_name=clf_name,
                dropout=args.dropout,
                chunk_size=args.chunk_size,
            ).to(device)

            np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_params[label] = np_

            enc_p = sum(p.numel() for p in model.encoder.parameters()    if p.requires_grad)
            gat_p = sum(p.numel() for p in model.gat.parameters()        if p.requires_grad)
            clf_p = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
            print(f"  params: {np_:,}  "
                  f"(encoder={enc_p:,}  GAT={gat_p:,}  classifier={clf_p:,})")

            save_path = os.path.join(out, f"Stacked_GRUAttn_{clf_name}_best.pth")

            metrics, t_sec = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                edge_index_adj, edge_index_od, edge_index_od_t,
                n_nodes=n_nodes, device=device,
                epochs=args.epochs, lr=args.lr,
                pos_weight=pos_weight, patience=args.patience,
                class_threshold=args.class_threshold,
                save_path=save_path,
                accumulation_steps=args.accumulation_steps,
            )
            results[label] = metrics
            timings[label] = t_sec

            # Per-model CSV
            csv_path = os.path.join(out, f"Stacked_GRUAttn_{clf_name}_metrics.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in metrics.items():
                    w.writerow([k, f"{v:.6f}"])

            f1_arr = metrics.get("f1_arrival",   metrics.get("f1_ch0", 0))
            f1_dep = metrics.get("f1_departure", metrics.get("f1_ch1", 0))
            print(f"\n  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}"
                  f"  |  F1_arr={f1_arr:.4f}  F1_dep={f1_dep:.4f}"
                  f"  ({t_sec:.1f}s)")

        except Exception:
            tb = traceback.format_exc()
            errors[label] = tb
            print(f"  FAILED\n{tb.splitlines()[-1]}")
            with open(os.path.join(out, f"Stacked_GRUAttn_{clf_name}_error.txt"), "w") as ef:
                ef.write(tb)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[4/4] Writing summary -> {out}")
    print(f"{'='*60}\n")

    channel_names: Tuple[str, ...] = (
        ("arrival",) if delay_dim == 1
        else ("arrival", "departure") if delay_dim == 2
        else tuple(f"ch{i}" for i in range(delay_dim))
    )

    base_fields = ["model", "n_params", "train_sec",
                   "f1", "accuracy", "precision", "recall"]
    ch_fields: List[str] = []
    for ch in channel_names:
        ch_fields += [f"f1_{ch}", f"accuracy_{ch}", f"precision_{ch}", f"recall_{ch}"]

    summary_path = os.path.join(out, "stacked_gru_transformer_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + ch_fields)
        writer.writeheader()
        for mname, m in results.items():
            row: Dict[str, str] = {
                "model":     mname,
                "n_params":  str(n_params.get(mname, "")),
                "train_sec": f"{timings.get(mname, 0):.1f}",
                "f1":        f"{m.get('f1',        0):.4f}",
                "accuracy":  f"{m.get('accuracy',  0):.4f}",
                "precision": f"{m.get('precision', 0):.4f}",
                "recall":    f"{m.get('recall',    0):.4f}",
            }
            for ch in channel_names:
                row[f"f1_{ch}"]        = f"{m.get(f'f1_{ch}',        0):.4f}"
                row[f"accuracy_{ch}"]  = f"{m.get(f'accuracy_{ch}',  0):.4f}"
                row[f"precision_{ch}"] = f"{m.get(f'precision_{ch}', 0):.4f}"
                row[f"recall_{ch}"]    = f"{m.get(f'recall_{ch}',    0):.4f}"
            writer.writerow(row)

    if errors:
        err_path = os.path.join(out, "failed_models.csv")
        with open(err_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "error"])
            for en, emsg in errors.items():
                w.writerow([en, emsg.splitlines()[-1]])

    # ── Console leaderboard ───────────────────────────────────────────────────
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
        col_w  = max(len(n) for n, _ in ranked) + 2

        hdr = f"{'Rank':<5} {'Model':<{col_w}} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
        for ch in channel_names:
            hdr += f"  {'F1_'+ch[:3]:>7}"
        hdr += f"  {'Params':>11}  {'sec':>7}"
        print(hdr)
        print("─" * len(hdr))

        for rank, (mname, m) in enumerate(ranked, 1):
            row_str = (f"{rank:<5} {mname:<{col_w}} "
                       f"{m['f1']:7.4f} {m['accuracy']:7.4f} "
                       f"{m['precision']:7.4f} {m['recall']:7.4f}")
            for ch in channel_names:
                row_str += f"  {m.get(f'f1_{ch}', 0):7.4f}"
            row_str += (f"  {n_params.get(mname, 0):>11,}"
                        f"  {timings.get(mname, 0):>7.1f}")
            print(row_str)

    if errors:
        print(f"\n[ERRORS] {len(errors)} model(s) failed: "
              + ", ".join(errors.keys()))

    print(f"\nSummary CSV : {summary_path}")
    print(f"Per-model   : {out}/")


if __name__ == "__main__":
    main()
