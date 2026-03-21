"""
Hybrid Graph-TSAI Classification Benchmark
===========================================
Combines the KAN-GAT graph architecture from threestagev4noise.py with the
top tsai temporal backbones to build spatiotemporal hybrid classifiers.

Architecture (per graph snapshot):
  Input : (n_nodes, c_in, seq_len)
       ↓  shared tsai backbone (treated as a node-level temporal encoder)
  (n_nodes, embed_dim)
       ↓  multi-edge GAT — adj + OD + OD_t with learned α weights + KAN fusion
            (identical design to LightweightGATEncoder in classifykat.py)
  (n_nodes, gat_hidden)
       ↓  KAN / MLP classifier head
  (n_nodes, c_out)          ← node-level binary labels (arrival, departure)

Training data format:
  (n_samples, n_nodes, seq_len * feature_dim)  ← one sample = one time snapshot
  (n_samples, n_nodes, c_out)                  ← node-level binary targets

The whole batch of snapshots is handled efficiently:
  • tsai backbone:  one batched call (B*n_nodes, c_in, seq_len)
  • GAT:            one batched call using offset edge indices
  • No per-sample Python loops in the inner training loop

Models evaluated (top-7 from tsai benchmark + GRU as lightweight reference):
  TSiTPlus · TSSequencerPlus · gMLP · GRUAttention · LSTMAttentionPlus
  GRU · ConvTranPlus

Usage:
    python hybrid_graph_tsai.py [options]

    --data_source       cdata|udata   (default: cdata)
    --seq_len           int           (default: 18)
    --horizons          int           (default: 12)
    --delay_threshold   float         (default: 5.0)
    --class_threshold   float         (default: 0.5)
    --embed_dim         int           (default: 64)   tsai encoder output dim
    --gat_hidden        int           (default: 64)   GAT hidden dim
    --gat_heads         int           (default: 2)
    --epochs            int           (default: 30)
    --batch_size        int           (default: 32)   graph-level batches
    --lr                float         (default: 3e-4)
    --patience          int           (default: 8)
    --seed              int           (default: 42)
    --output_dir        str           (default: auto-timestamped)
    --models            name [name…]  run only listed models
    --skip_models       name [name…]  skip listed models
    --device            cpu|cuda      (default: auto)
"""
from __future__ import annotations

import argparse
import csv
import importlib
import os
import sys
import time
import traceback
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

# Try to import KAN for the classifier head; fall back to MLP if unavailable
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
# TOP MODEL REGISTRY
# (display_name, tsai_module, class_name, needs_seq_len)
# ═══════════════════════════════════════════════════════════════════════════════
TOP_MODELS: List[Tuple[str, str, str, bool]] = [
    ("TSiTPlus",         "tsai.models.TSiTPlus",         "TSiTPlus",         True),
    ("TSSequencerPlus",  "tsai.models.TSSequencerPlus",  "TSSequencerPlus",  True),
    ("gMLP",             "tsai.models.gMLP",             "gMLP",             True),
    ("GRUAttention",     "tsai.models.RNNAttention",     "GRUAttention",     True),
    ("LSTMAttentionPlus","tsai.models.RNNAttentionPlus", "LSTMAttentionPlus",True),
    ("GRU",              "tsai.models.RNN",              "GRU",              False),
    ("ConvTranPlus",     "tsai.models.ConvTranPlus",     "ConvTranPlus",     True),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TSAI BACKBONE BUILDER
# Builds any tsai model as a *temporal encoder* (outputs embed_dim, not c_out).
# Handles non-standard output shapes (e.g. PatchTST → (B, C, T)) via flatten+linear.
# ═══════════════════════════════════════════════════════════════════════════════

class _FlattenHead(nn.Module):
    """Flatten non-(B, D) tsai outputs and project to embed_dim."""
    def __init__(self, backbone: nn.Module, flat_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Sequential(nn.LayerNorm(flat_dim), nn.Linear(flat_dim, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return self.proj(out.flatten(1))


def build_tsai_encoder(
    display_name: str, mod_path: str, cls_name: str,
    needs_seq_len: bool, c_in: int, embed_dim: int, seq_len: int,
) -> nn.Module:
    """Instantiate a tsai model configured as a node-level temporal encoder."""
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    kw: Dict = dict(c_in=c_in, c_out=embed_dim)
    if needs_seq_len:
        kw["seq_len"] = seq_len

    model = cls(**kw)

    # Probe output shape
    dummy = torch.randn(2, c_in, seq_len)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]

    expected = torch.Size([2, embed_dim])
    if out.shape != expected:
        flat_dim = int(out.flatten(1).shape[1])
        print(f"    [encoder-wrap] {display_name}: {tuple(out.shape)} "
              f"→ flatten({flat_dim}) → Linear({flat_dim}, {embed_dim})")
        model = _FlattenHead(model, flat_dim, embed_dim)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-EDGE GAT FUSION MODULE
# Mirrors LightweightGATEncoder from classifykat.py but takes a plain tensor
# (n_nodes, embed_dim) instead of a PyG Data object, so it can be used easily
# inside a batched training loop.
# ═══════════════════════════════════════════════════════════════════════════════

def _make_classifier_head(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Module:
    """KAN head if available, otherwise a 2-layer MLP."""
    if KAN_AVAILABLE:
        return _KAN(
            layers_hidden=[in_dim, hidden_dim, out_dim],
            grid_size=3,
            spline_order=2,
        )
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, out_dim),
    )


class MultiEdgeGATFusion(nn.Module):
    """
    Three parallel GATConv layers (adj / OD / OD_t) with learned α weights,
    followed by a KAN/MLP fusion layer — identical design to LightweightGATEncoder.
    Accepts plain node feature tensors (not Data objects).
    """

    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 2):
        super().__init__()
        self.alpha_adj   = nn.Parameter(torch.tensor(1.0))
        self.alpha_od    = nn.Parameter(torch.tensor(1.0))
        self.alpha_od_t  = nn.Parameter(torch.tensor(1.0))

        self.gat_adj   = GATConv(in_channels, hidden_channels, heads=heads,
                                  concat=False, dropout=0.1)
        self.gat_od    = GATConv(in_channels, hidden_channels, heads=heads,
                                  concat=False, dropout=0.1)
        self.gat_od_t  = GATConv(in_channels, hidden_channels, heads=heads,
                                  concat=False, dropout=0.1)

        # fusion_input_dim = hidden * 3 + 3 (learned scalar weights appended)
        fusion_in = hidden_channels * 3 + 3
        self.fusion = _make_classifier_head(fusion_in, hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,                  # (total_nodes, in_channels)
        edge_index_adj: torch.Tensor,     # (2, E_adj)
        edge_index_od: torch.Tensor,      # (2, E_od)
        edge_index_od_t: torch.Tensor,    # (2, E_od_t)
    ) -> torch.Tensor:                    # (total_nodes, hidden_channels)
        w = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
        w_adj, w_od, w_od_t = w

        x_adj   = self.gat_adj  (x, edge_index_adj)
        x_od    = self.gat_od   (x, edge_index_od)
        x_od_t  = self.gat_od_t (x, edge_index_od_t)

        n = x_adj.size(0)
        scalars = torch.stack([
            w_adj  .expand(n),
            w_od   .expand(n),
            w_od_t .expand(n),
        ], dim=1)  # (n, 3)

        fused = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
        out   = F.relu(self.fusion(fused))
        return self.dropout(out)          # (n, hidden_channels)


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HybridGraphTSAI(nn.Module):
    """
    Spatiotemporal hybrid: tsai temporal backbone + multi-edge GAT + KAN head.

    Forward signature:
        x               : (B, n_nodes, c_in, seq_len)  →  batched node features
        edge_index_adj  : (2, B * E_adj)   →  batched edge index (pre-offset)
        edge_index_od   : (2, B * E_od)
        edge_index_od_t : (2, B * E_od_t)

    Returns:
        logits          : (B, n_nodes, c_out)
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        gat_hidden: int,
        c_out: int,
        gat_heads: int = 2,
    ):
        super().__init__()
        self.backbone  = backbone                                    # tsai encoder
        self.gat       = MultiEdgeGATFusion(embed_dim, gat_hidden, gat_heads)
        self.classifier = _make_classifier_head(gat_hidden, gat_hidden // 2, c_out)

    def forward(
        self,
        x: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> torch.Tensor:
        B, N, C, T = x.shape

        # ── temporal encoding (all nodes in all graphs at once) ───────────────
        x_flat = x.view(B * N, C, T)              # (B*N, c_in, seq_len)
        h = self.backbone(x_flat)                  # (B*N, embed_dim)
        if isinstance(h, (tuple, list)):
            h = h[0]
        if h.dim() > 2:
            h = h.flatten(1)

        # ── graph message passing ─────────────────────────────────────────────
        g = self.gat(h, edge_index_adj, edge_index_od, edge_index_od_t)  # (B*N, gat_hidden)

        # ── classification head ───────────────────────────────────────────────
        logits = self.classifier(g)                # (B*N, c_out)
        return logits.view(B, N, -1)               # (B, N, c_out)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_graph_tensors(
    train_x, train_y_cls,
    val_x,   val_y_cls,
    test_x,  test_y_cls,
    seq_len: int,
    feature_dim: int,
) -> Tuple:
    """
    Reshape node-level tensors from (n, n_nodes, seq_len*f) → (n, n_nodes, f, seq_len).
    Targets stay at (n, n_nodes, c_out).
    Returns: trX, trY, vaX, vaY, teX, teY   — all torch.Tensor
    """
    def _reshape(X, Y):
        n, n_nodes, _ = X.shape
        # (n, n_nodes, seq_len*f) → (n, n_nodes, f, seq_len)
        Xg = X.view(n, n_nodes, seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
        return Xg, Y  # Y already (n, n_nodes, c_out)

    trX, trY = _reshape(train_x, train_y_cls)
    vaX, vaY = _reshape(val_x,   val_y_cls)
    teX, teY = _reshape(test_x,  test_y_cls)
    return trX, trY, vaX, vaY, teX, teY


def batch_edge_index(edge_index: torch.Tensor, n_nodes: int, batch_size: int) -> torch.Tensor:
    """Repeat edge_index B times, offsetting node indices by b * n_nodes."""
    offsets = torch.arange(batch_size, device=edge_index.device) * n_nodes  # (B,)
    # edge_index: (2, E) → repeat B times with offset
    ei = edge_index.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 2, E)
    ei = ei + offsets.view(batch_size, 1, 1)                  # broadcast offset
    return ei.reshape(2, -1)                                   # (2, B*E)


def classification_metrics_per_channel(
    preds: np.ndarray,
    targets: np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Per-channel binary metrics + macro averages.  Inputs: any shape ending in C."""
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
# TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model: HybridGraphTSAI,
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
    """Train HybridGraphTSAI with BCEWithLogitsLoss + warmup + cosine LR.

    Key design choices for graph-level training:
      • Separate param groups: backbone LR=lr, graph layers LR=lr*3
      • Linear warmup for first 5 epochs → cosine decay
      • Gradient accumulation to simulate larger effective batch
    """

    # Pre-move fixed edge indices to device
    ei_adj   = edge_index_adj.to(device)
    ei_od    = edge_index_od.to(device)
    ei_od_t  = edge_index_od_t.to(device)

    # ── separate param groups: backbone (pretrained-style) vs graph (new) ────
    backbone_params = list(model.backbone.parameters())
    graph_params    = (list(model.gat.parameters()) +
                       list(model.classifier.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr,      "weight_decay": 1e-4},
        {"params": graph_params,    "lr": lr * 3,  "weight_decay": 1e-5},
    ])

    # ── warmup + cosine schedule ──────────────────────────────────────────────
    warmup_epochs = min(5, max(1, epochs // 6))
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # linear warmup
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.05 + 0.95 * 0.5 * (1 + np.cos(np.pi * progress))  # cosine → 5%

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    es      = EarlyStopping(patience=patience, mode="max")

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
            bx = bx.to(device)     # (B, n_nodes, c_in, seq_len)
            by = by.to(device)     # (B, n_nodes, c_out)  float

            # Build batched edge indices (same graph topology, B copies)
            bei_adj   = batch_edge_index(ei_adj,   n_nodes, B)
            bei_od    = batch_edge_index(ei_od,    n_nodes, B)
            bei_od_t  = batch_edge_index(ei_od_t,  n_nodes, B)

            logits = model(bx, bei_adj, bei_od, bei_od_t)  # (B, N, c_out)
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
                    "epoch":       epoch,
                    "model_state": best_state,
                    "val_f1":      best_f1,
                    "embed_dim":   model.gat.gat_adj.in_channels,
                    "gat_hidden":  model.gat.gat_adj.out_channels,
                }, save_path)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"loss={np.mean(ep_losses):.4f}  val_f1={vm['f1']:.4f}")

        if es(vm["f1"], epoch):
            print(f"    early stop @ epoch {epoch}  best_val_f1={best_f1:.4f}")
            break

    train_sec = time.time() - t0

    # ── save final best checkpoint ────────────────────────────────────────────
    if save_path is not None and best_state is not None:
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["train_sec"] = train_sec
        torch.save(ckpt, save_path)
        print(f"  💾 Saved best model → {save_path}")

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
        description="Hybrid Graph-TSAI classification benchmark"
    )
    p.add_argument("--data_source",       default="cdata", choices=["cdata", "udata"])
    p.add_argument("--seq_len",           type=int,   default=18)
    p.add_argument("--horizons",          type=int,   nargs="+", default=[12])
    p.add_argument("--delay_threshold",   type=float, default=5.0)
    p.add_argument("--class_threshold",   type=float, default=0.5)
    p.add_argument("--embed_dim",         type=int,   default=64,
                   help="tsai backbone output / GAT input dimension")
    p.add_argument("--gat_hidden",        type=int,   default=64,
                   help="GAT hidden dimension (output of multi-edge fusion)")
    p.add_argument("--gat_heads",         type=int,   default=2)
    p.add_argument("--weather_file",      type=str,   default="weather_cn.npy")
    p.add_argument("--period_hours",      type=int,   default=24)
    p.add_argument("--epochs",            type=int,   default=50)
    p.add_argument("--batch_size",        type=int,   default=64,
                   help="Number of graph snapshots per batch")
    p.add_argument("--lr",                type=float, default=1e-4)
    p.add_argument("--patience",          type=int,   default=12)
    p.add_argument("--accumulation_steps", type=int,  default=4,
                   help="Gradient accumulation steps (effective batch = batch_size * this)")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--output_dir",        type=str,   default="auto")
    p.add_argument("--device",            type=str,   default="auto")
    p.add_argument("--models",            nargs="+",  default=None,
                   help="Whitelist of model names (default: all 7)")
    p.add_argument("--skip_models",       nargs="+",  default=[])
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

    print(f"\n{'='*65}")
    print(f"  Hybrid Graph-TSAI Classification Benchmark  |  device={device}")
    print(f"  Architecture: tsai backbone + multi-edge GAT + KAN head")
    print(f"  KAN available: {KAN_AVAILABLE}")
    print(f"  {len(TOP_MODELS)} top-7 tsai backbones")
    print(f"{'='*65}\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    print("[1/4] Loading data …")
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

    # Remove time features (last 2 columns) to match the tsai benchmark
    train_inputs = train_inputs[:, :, :-2]
    val_inputs   = val_inputs[:, :, :-2]
    test_inputs  = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim   = train_delay_scaled.shape[2]
    max_horizon = sorted(set(args.horizons))[0]

    print(f"  Nodes: {num_nodes}  feature_dim: {feature_dim}  "
          f"delay_dim: {delay_dim}  seq_len: {args.seq_len}")

    # ── 2. Build graph-level sequences ────────────────────────────────────────
    print("[2/4] Building sequences …")
    #  build_sequences_node_level returns:
    #    X   : (n_samples, n_nodes, seq_len * feature_dim)
    #    Yreg: (n_samples, n_nodes, delay_dim)
    #    Ycls: (n_samples, n_nodes, delay_dim)
    train_x, _, train_y_cls = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    val_x,   _, val_y_cls   = build_sequences_node_level(
        val_inputs,   val_delay_scaled,   val_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    test_x,  _, test_y_cls  = build_sequences_node_level(
        test_inputs,  test_delay_scaled,  test_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )

    # Reshape to graph-level tensors (n, n_nodes, c_in, seq_len)
    trX, trY, vaX, vaY, teX, teY = build_graph_tensors(
        train_x, train_y_cls,
        val_x,   val_y_cls,
        test_x,  test_y_cls,
        args.seq_len, feature_dim,
    )
    n_nodes = trX.shape[1]

    print(f"  Train {tuple(trX.shape)}  Val {tuple(vaX.shape)}  Test {tuple(teX.shape)}")
    print(f"  c_in={feature_dim}  c_out={delay_dim}  n_nodes={n_nodes}")

    delayed_frac = trY.mean().item()
    print(f"  Class balance (train): {delayed_frac:.2%} delayed")

    # pos_weight per channel: (1-p)/p averaged over nodes, then applied globally
    pos_rate   = trY.reshape(-1, delay_dim).mean(dim=0)          # (c_out,)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)    # (c_out,)

    # Cast targets to float
    trY = trY.float()
    vaY = vaY.float()
    teY = teY.float()

    train_loader = DataLoader(TensorDataset(trX, trY),
                              batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(vaX, vaY),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(teX, teY),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── 3. Output directory ───────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir if args.output_dir != "auto"
           else f"hybrid_graph_tsai_{ts}")
    os.makedirs(out, exist_ok=True)

    # ── 4. Apply whitelist / skip filters ────────────────────────────────────
    skip_set = set(args.skip_models or [])
    only_set = set(args.models) if args.models else None

    active = [
        row for row in TOP_MODELS
        if row[0] not in skip_set and (only_set is None or row[0] in only_set)
    ]

    print(f"\n[3/4] Training {len(active)} hybrid model(s) …\n")

    results:  Dict[str, Dict[str, float]] = {}
    n_params: Dict[str, int]              = {}
    timings:  Dict[str, float]            = {}
    errors:   Dict[str, str]              = {}

    for idx, (display_name, mod_path, cls_name, needs_seq) in enumerate(active, 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{len(active)}] Hybrid-{display_name}")
        print(f"{'─'*60}")

        try:
            # Build tsai encoder (c_out = embed_dim, so output is (n, embed_dim))
            backbone = build_tsai_encoder(
                display_name, mod_path, cls_name, needs_seq,
                c_in=feature_dim, embed_dim=args.embed_dim, seq_len=args.seq_len,
            )

            model = HybridGraphTSAI(
                backbone  = backbone,
                embed_dim = args.embed_dim,
                gat_hidden= args.gat_hidden,
                c_out     = delay_dim,
                gat_heads = args.gat_heads,
            ).to(device)

            np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_params[display_name] = np_
            print(f"  params: {np_:,}  "
                  f"(backbone + GAT + head)")

            save_path = os.path.join(out, f"Hybrid_{display_name}_best.pth")
            metrics, t_sec = train_and_evaluate(
                model,
                train_loader, val_loader, test_loader,
                edge_index_adj, edge_index_od, edge_index_od_t,
                n_nodes  = n_nodes,
                device   = device,
                epochs   = args.epochs,
                lr       = args.lr,
                pos_weight = pos_weight,
                patience   = args.patience,
                class_threshold = args.class_threshold,
                save_path = save_path,
                accumulation_steps = args.accumulation_steps,
            )
            results[display_name] = metrics
            timings[display_name] = t_sec

            # per-model CSV
            csv_path = os.path.join(out, f"Hybrid_{display_name}_metrics.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in metrics.items():
                    w.writerow([k, f"{v:.6f}"])

            f1_arr = metrics.get("f1_arrival",   metrics.get("f1_ch0", 0))
            f1_dep = metrics.get("f1_departure", metrics.get("f1_ch1", 0))
            print(f"\n  ✓  F1={metrics['f1']:.4f}  Acc={metrics['accuracy']:.4f}"
                  f"  |  F1_arr={f1_arr:.4f}  F1_dep={f1_dep:.4f}"
                  f"  ({t_sec:.1f}s)")

        except Exception:
            tb = traceback.format_exc()
            errors[display_name] = tb
            print(f"  ✗  FAILED\n{tb.splitlines()[-1]}")
            with open(os.path.join(out, f"Hybrid_{display_name}_error.txt"), "w") as ef:
                ef.write(tb)

    # ── 5. Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[4/4] Writing summary → {out}")
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

    summary_path = os.path.join(out, "hybrid_graph_tsai_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=base_fields + ch_fields)
        writer.writeheader()
        for mname, m in results.items():
            row: Dict[str, str] = {
                "model":     f"Hybrid-{mname}",
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
        col_w  = max(len(f"Hybrid-{n}") for n, _ in ranked) + 2

        hdr = f"{'Rank':<5} {'Model':<{col_w}} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
        for ch in channel_names:
            hdr += f"  {'F1_'+ch[:3]:>7}"
        hdr += f"  {'Params':>11}  {'sec':>7}"
        print(hdr)
        print("─" * len(hdr))

        for rank, (mname, m) in enumerate(ranked, 1):
            label = f"Hybrid-{mname}"
            row_str = (f"{rank:<5} {label:<{col_w}} "
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

    print(f"\n✓ Summary CSV : {summary_path}")
    print(f"✓ Per-model   : {out}/")


if __name__ == "__main__":
    main()
