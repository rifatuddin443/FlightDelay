"""
tsai Models Classification Benchmark
=====================================
Evaluates all 43 tsai time-series classification models on the
flight-delay dataset. Reports per-channel (arrival / departure) metrics
for every model and writes comparison CSVs.

Data format for tsai: (n_samples, c_in, seq_len)
  → obtained by permuting our (n_samples, seq_len, feature_dim) tensors.

Usage:
    python tsai_classification_benchmark.py [options]

    --data_source       cdata|udata   (default: cdata)
    --seq_len           int           (default: 18)
    --horizons          int           (default: 12)
    --delay_threshold   float         (default: 5.0)
    --class_threshold   float         (default: 0.5)
    --epochs            int           (default: 30)
    --batch_size        int           (default: 128)
    --lr                float         (default: 1e-3)
    --patience          int           (default: 7)
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
from torch.utils.data import DataLoader, TensorDataset

# ─── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import EarlyStopping, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level, build_sequences


# ═══════════════════════════════════════════════════════════════════════════════
# TSAI MODEL REGISTRY
# Fields: (display_name, tsai_module, class_name, needs_seq_len)
# needs_seq_len=True  → pass seq_len=N to constructor
# needs_seq_len=False → constructor only takes (c_in, c_out)
# ═══════════════════════════════════════════════════════════════════════════════

TSAI_MODEL_REGISTRY: List[Tuple[str, str, str, bool]] = [
    # ── Classic CNNs ──────────────────────────────────────────────────────────
    ("FCN",              "tsai.models.FCN",              "FCN",              False),
    ("FCNPlus",          "tsai.models.FCNPlus",          "FCNPlus",          False),
    ("ResNet",           "tsai.models.ResNet",           "ResNet",           False),
    ("ResNetPlus",       "tsai.models.ResNetPlus",       "ResNetPlus",       False),
    ("ResCNN",           "tsai.models.ResCNN",           "ResCNN",           False),
    ("OmniScaleCNN",     "tsai.models.OmniScaleCNN",     "OmniScaleCNN",     True),
    ("XceptionTime",     "tsai.models.XceptionTime",     "XceptionTime",     False),
    ("XceptionTimePlus", "tsai.models.XceptionTimePlus", "XceptionTimePlus", False),
    ("XResNet1d",        "tsai.models.XResNet1d",        "xresnet1d34",      False),
    ("XResNet1dPlus",    "tsai.models.XResNet1dPlus",    "xresnet1d34plus",  False),

    # ── Inception family ─────────────────────────────────────────────────────
    ("InceptionTime",     "tsai.models.InceptionTime",     "InceptionTime",     False),
    ("InceptionTimePlus", "tsai.models.InceptionTimePlus", "InceptionTimePlus", False),

    # ── RNNs ─────────────────────────────────────────────────────────────────
    ("LSTM",    "tsai.models.RNN",    "LSTM",    False),
    ("GRU",     "tsai.models.RNN",    "GRU",     False),
    ("RNN",     "tsai.models.RNN",    "RNN",     False),
    ("LSTMPlus","tsai.models.RNNPlus","LSTMPlus",False),
    ("GRUPlus", "tsai.models.RNNPlus","GRUPlus", False),
    ("RNNPlus", "tsai.models.RNNPlus","RNNPlus", False),

    # ── RNN + Attention ──────────────────────────────────────────────────────
    ("LSTMAttention",     "tsai.models.RNNAttention",     "LSTMAttention",     True),
    ("GRUAttention",      "tsai.models.RNNAttention",     "GRUAttention",      True),
    ("LSTMAttentionPlus", "tsai.models.RNNAttentionPlus", "LSTMAttentionPlus", True),
    ("GRUAttentionPlus",  "tsai.models.RNNAttentionPlus", "GRUAttentionPlus",  True),

    # ── RNN + FCN hybrids ────────────────────────────────────────────────────
    ("LSTM_FCN",      "tsai.models.RNN_FCN",     "LSTM_FCN",      True),
    ("GRU_FCN",       "tsai.models.RNN_FCN",     "GRU_FCN",       True),
    ("MLSTM_FCN",     "tsai.models.RNN_FCN",     "MLSTM_FCN",     True),
    ("MGRU_FCN",      "tsai.models.RNN_FCN",     "MGRU_FCN",      True),
    ("LSTM_FCNPlus",  "tsai.models.RNN_FCNPlus", "LSTM_FCNPlus",  True),
    ("GRU_FCNPlus",   "tsai.models.RNN_FCNPlus", "GRU_FCNPlus",   True),
    ("MLSTM_FCNPlus", "tsai.models.RNN_FCNPlus", "MLSTM_FCNPlus", True),

    # ── Temporal CNN ─────────────────────────────────────────────────────────
    ("TCN", "tsai.models.TCN", "TCN", False),

    # ── MLP family ───────────────────────────────────────────────────────────
    ("MLP",  "tsai.models.MLP",  "MLP",  True),
    ("gMLP", "tsai.models.gMLP", "gMLP", True),

    # ── Wavelet / mWDN ───────────────────────────────────────────────────────
    ("mWDN", "tsai.models.mWDN", "mWDN", True),

    # ── Transformer family ───────────────────────────────────────────────────
    ("TST",              "tsai.models.TST",             "TST",              True),
    ("TSTPlus",          "tsai.models.TSTPlus",          "TSTPlus",          True),
    ("TransformerModel", "tsai.models.TransformerModel", "TransformerModel", False),
    # PatchTST's output is auto-wrapped into a classification head (see _build_model)
    ("PatchTST",         "tsai.models.PatchTST",         "PatchTST",         True),
    ("TSiTPlus",         "tsai.models.TSiTPlus",         "TSiTPlus",         True),
    ("TSSequencerPlus",  "tsai.models.TSSequencerPlus",  "TSSequencerPlus",  True),

    # ── ConvTran ─────────────────────────────────────────────────────────────
    ("ConvTranPlus", "tsai.models.ConvTranPlus", "ConvTranPlus", True),

    # ── XCM (Explainable CNN) ────────────────────────────────────────────────
    ("XCM",     "tsai.models.XCM",     "XCM",     True),
    ("XCMPlus", "tsai.models.XCMPlus", "XCMPlus", True),

    # ── MiniRocket (PyTorch) ─────────────────────────────────────────────────
    ("MiniRocketPlus", "tsai.models.MINIROCKETPlus_Pytorch", "MiniRocketPlus", True),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SHAPE-FIX WRAPPER
# PatchTST (and any encoder-only model) outputs (B, C, T) instead of (B, c_out).
# This wrapper detects the mismatch and adds a flatten → linear classification head.
# ═══════════════════════════════════════════════════════════════════════════════

class ClassificationHeadWrapper(nn.Module):
    """Append a linear classification head to an encoder with non-standard output shape."""
    def __init__(self, backbone: nn.Module, in_features: int, c_out: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, c_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return self.head(out.flatten(1))   # (B, in_features) → (B, c_out)


def _build_model(
    display_name: str, mod_path: str, cls_name: str,
    needs_seq_len: bool, c_in: int, c_out: int, seq_len: int,
) -> nn.Module:
    """Instantiate a tsai model and wrap it if its output shape is non-standard."""
    mod = importlib.import_module(mod_path)
    cls = getattr(mod, cls_name)

    kw: Dict = dict(c_in=c_in, c_out=c_out)
    if needs_seq_len:
        kw["seq_len"] = seq_len

    model = cls(**kw)

    # Verify output shape with a dummy forward pass
    dummy = torch.randn(2, c_in, seq_len)
    with torch.no_grad():
        out = model(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]

    expected = torch.Size([2, c_out])
    if out.shape != expected:
        # Flatten everything after the batch dim and attach a linear head
        in_features = int(out.flatten(1).shape[1])
        print(f"  [wrap] {display_name} output {tuple(out.shape)} "
              f"→ adding head({in_features}→{c_out})")
        model = ClassificationHeadWrapper(model, in_features, c_out)

    return model


# ═══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def build_node_level_tensors(
    train_x, train_y_cls, val_x, val_y_cls, test_x, test_y_cls,
    seq_len: int, feature_dim: int,
) -> Tuple:
    """Flatten graph-level tensors to (n*nodes, seq_len, feature_dim)."""
    def _reshape(X, Y):
        n, n_nodes, _ = X.shape
        Xn = X.view(n, n_nodes, seq_len, feature_dim).reshape(n * n_nodes, seq_len, feature_dim)
        Yn = Y.reshape(n * n_nodes, -1)
        return Xn, Yn
    trX, trY = _reshape(train_x, train_y_cls)
    vaX, vaY = _reshape(val_x, val_y_cls)
    teX, teY = _reshape(test_x, test_y_cls)
    return trX, trY, vaX, vaY, teX, teY


def to_tsai(x: torch.Tensor) -> torch.Tensor:
    """(n, seq_len, c) → (n, c, seq_len) as required by tsai models."""
    return x.permute(0, 2, 1).contiguous()


def classification_metrics_per_channel(
    preds: np.ndarray,
    targets: np.ndarray,
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
# TRAIN / EVALUATE
# ═══════════════════════════════════════════════════════════════════════════════

def train_and_evaluate(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    model_name: str,
    class_threshold: float = 0.5,
) -> Tuple[Dict[str, float], float]:
    """Train with BCEWithLogitsLoss + cosine LR, return test metrics of best val checkpoint."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    es        = EarlyStopping(patience=patience, mode="max")

    best_f1, best_state = -1.0, None
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        ep_losses = []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(bx)
            if isinstance(out, (tuple, list)):
                out = out[0]
            loss = loss_fn(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_losses.append(loss.item())
        scheduler.step()

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx.to(device))
                if isinstance(out, (tuple, list)):
                    out = out[0]
                vp.append(torch.sigmoid(out).cpu())
                vt.append(by)
        vm = classification_metrics_per_channel(
            torch.cat(vp).numpy(), torch.cat(vt).numpy(), threshold=class_threshold,
        )
        if vm["f1"] > best_f1:
            best_f1   = vm["f1"]
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"    epoch {epoch:3d}/{epochs}  "
                  f"loss={np.mean(ep_losses):.4f}  val_f1={vm['f1']:.4f}")

        if es(vm["f1"], epoch):
            print(f"    early stop @ epoch {epoch}  best_val_f1={best_f1:.4f}")
            break

    train_sec = time.time() - t0

    # ── test ──────────────────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    tp_list, tt_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            out = model(bx.to(device))
            if isinstance(out, (tuple, list)):
                out = out[0]
            tp_list.append(torch.sigmoid(out).cpu())
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
    p = argparse.ArgumentParser(description="tsai classification benchmark — all models")
    p.add_argument("--data_source", default="cdata", choices=["cdata", "udata"])
    p.add_argument("--seq_len", type=int, default=18)
    p.add_argument("--horizons", type=int, nargs="+", default=[12])
    p.add_argument("--delay_threshold", type=float, default=5.0)
    p.add_argument("--class_threshold", type=float, default=0.5)
    p.add_argument("--use_node_level", action="store_true", default=True)
    p.add_argument("--exclude_time_features", action="store_true", default=True)
    p.add_argument("--weather_file", type=str, default="weather_cn.npy")
    p.add_argument("--period_hours", type=int, default=24)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="auto")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--models", nargs="+", default=None,
                   help="Whitelist of model display names (default: all 43)")
    p.add_argument("--skip_models", nargs="+", default=[],
                   help="Models to skip")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    print(f"\n{'='*65}")
    print(f"  tsai Classification Benchmark  |  device={device}")
    print(f"  {len(TSAI_MODEL_REGISTRY)} models registered")
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

    if args.exclude_time_features:
        train_inputs = train_inputs[:, :, :-2]
        val_inputs   = val_inputs[:, :, :-2]
        test_inputs  = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]        # becomes c_in for tsai
    delay_dim   = train_delay_scaled.shape[2]  # becomes c_out (2: arr + dep)
    max_horizon = sorted(set(args.horizons))[0]

    # ── 2. Build sequences ────────────────────────────────────────────────────
    print("[2/4] Building sequences …")
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences

    train_x, _, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons)
    val_x,   _, val_y_cls   = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons)
    test_x,  _, test_y_cls  = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons)

    trX, trY, vaX, vaY, teX, teY = build_node_level_tensors(
        train_x, train_y_cls, val_x, val_y_cls, test_x, test_y_cls,
        args.seq_len, feature_dim,
    )

    # Convert (n, seq_len, c_in) → (n, c_in, seq_len) for tsai
    trX_t = to_tsai(trX)
    vaX_t = to_tsai(vaX)
    teX_t = to_tsai(teX)

    print(f"  Train {tuple(trX_t.shape)}  Val {tuple(vaX_t.shape)}  Test {tuple(teX_t.shape)}")
    print(f"  c_in={feature_dim}  c_out={delay_dim}  seq_len={args.seq_len}")

    pos_rate   = trY.mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    train_loader = DataLoader(TensorDataset(trX_t, trY),
                              batch_size=args.batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(TensorDataset(vaX_t, vaY),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(TensorDataset(teX_t, teY),
                              batch_size=args.batch_size, shuffle=False, drop_last=False)

    # ── 3. Output directory ───────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir if args.output_dir != "auto" else f"tsai_benchmark_{ts}"
    os.makedirs(out, exist_ok=True)

    # ── 4. Apply whitelist / skip filters ────────────────────────────────────
    skip_set = set(args.skip_models or [])
    only_set = set(args.models) if args.models else None

    active = [
        row for row in TSAI_MODEL_REGISTRY
        if row[0] not in skip_set and (only_set is None or row[0] in only_set)
    ]

    print(f"\n[3/4] Training {len(active)} model(s) …\n")

    results:  Dict[str, Dict[str, float]] = {}
    n_params: Dict[str, int]              = {}
    timings:  Dict[str, float]            = {}
    errors:   Dict[str, str]              = {}

    for idx, (display_name, mod_path, cls_name, needs_seq) in enumerate(active, 1):
        print(f"\n{'─'*60}")
        print(f"  [{idx}/{len(active)}] {display_name}")
        print(f"{'─'*60}")

        try:
            model = _build_model(
                display_name, mod_path, cls_name, needs_seq,
                c_in=feature_dim, c_out=delay_dim, seq_len=args.seq_len,
            ).to(device)

            np_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_params[display_name] = np_
            print(f"  params: {np_:,}")

            metrics, t_sec = train_and_evaluate(
                model, train_loader, val_loader, test_loader,
                device=device, epochs=args.epochs, lr=args.lr,
                pos_weight=pos_weight, patience=args.patience,
                model_name=display_name, class_threshold=args.class_threshold,
            )
            results[display_name] = metrics
            timings[display_name] = t_sec

            # per-model CSV
            with open(os.path.join(out, f"{display_name}_metrics.csv"), "w", newline="") as f:
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
            with open(os.path.join(out, f"{display_name}_error.txt"), "w") as ef:
                ef.write(tb)

    # ── 5. Summary CSV ────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[4/4] Writing summary → {out}")
    print(f"{'='*60}\n")

    # Determine channel names from delay_dim
    if delay_dim == 1:
        channel_names: Tuple[str, ...] = ("arrival",)
    elif delay_dim == 2:
        channel_names = ("arrival", "departure")
    else:
        channel_names = tuple(f"ch{i}" for i in range(delay_dim))

    base_fields = ["model", "n_params", "train_sec",
                   "f1", "accuracy", "precision", "recall"]
    ch_fields: List[str] = []
    for ch in channel_names:
        ch_fields += [f"f1_{ch}", f"accuracy_{ch}", f"precision_{ch}", f"recall_{ch}"]

    summary_path = os.path.join(out, "tsai_classification_summary.csv")
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

    # ── 6. Failed models CSV ──────────────────────────────────────────────────
    if errors:
        err_path = os.path.join(out, "failed_models.csv")
        with open(err_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "error"])
            for en, emsg in errors.items():
                w.writerow([en, emsg.splitlines()[-1]])

    # ── 7. Console leaderboard (sorted by macro F1) ───────────────────────────
    if results:
        ranked = sorted(results.items(), key=lambda x: x[1].get("f1", 0), reverse=True)
        col_w  = max(len(n) for n, _ in ranked) + 2

        hdr  = f"{'Rank':<5} {'Model':<{col_w}} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}"
        for ch in channel_names:
            hdr += f"  {'F1_'+ch[:3]:>7}"
        hdr += f"  {'Params':>10}  {'sec':>6}"
        print(hdr)
        print("─" * len(hdr))

        for rank, (mname, m) in enumerate(ranked, 1):
            row_str = (f"{rank:<5} {mname:<{col_w}} "
                       f"{m['f1']:7.4f} {m['accuracy']:7.4f} "
                       f"{m['precision']:7.4f} {m['recall']:7.4f}")
            for ch in channel_names:
                row_str += f"  {m.get(f'f1_{ch}', 0):7.4f}"
            row_str += (f"  {n_params.get(mname, 0):>10,}"
                        f"  {timings.get(mname, 0):>6.1f}")
            print(row_str)

    if errors:
        print(f"\n[ERRORS] {len(errors)} model(s) failed: "
              + ", ".join(errors.keys()))

    print(f"\n✓ Summary CSV  : {summary_path}")
    print(f"✓ Per-model CSVs: {out}/")


if __name__ == "__main__":
    main()
