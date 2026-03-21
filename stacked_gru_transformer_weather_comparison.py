"""
Weather Impact Comparison for Stacked GRUAttention → Transformer Classifier
=============================================================================

This script trains the same architecture twice:
  1. WITH weather data (baseline)
  2. WITHOUT weather data (to isolate weather contribution)

Data flow comparison:

WITH WEATHER:
  (B, N, c_in, T) includes [delay_features, weather_features]
       ↓ GRU encoder → GAT → Transformer
  Metrics recorded

WITHOUT WEATHER:
  (B, N, c_in_no_weather, T) includes [delay_features only]
       ↓ GRU encoder → GAT → Transformer  
  Metrics recorded

COMPARISON ANALYSIS:
  - F1 score improvement with weather
  - Per-channel performance delta
  - Memory usage comparison
  - Training time comparison

Usage:
    python stacked_gru_transformer_weather_comparison.py --epochs 50
    python stacked_gru_transformer_weather_comparison.py --classifier both --epochs 50
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
    from classifykat import KAN as _KAN
    KAN_AVAILABLE = True
except ImportError:
    try:
        from efficient_kan import KAN as _KAN
        KAN_AVAILABLE = True
    except ImportError:
        KAN_AVAILABLE = False

from classifykat_balanced import build_sequences_node_level
from stacked_gru_transformer import (
    GRUAttentionEncoder,
    MultiEdgeGATFusion,
    build_classifier,
    StackedGRUTransformer,
    batch_edge_index,
    classification_metrics_per_channel,
    build_graph_tensors,
)


# ═══════════════════════════════════════════════════════════════════════════════
# WEATHER DATA UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def extract_feature_components(X: np.ndarray, delay_dim: int, weather_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split input features into delay and weather components.
    
    Assumed structure (from load_flight_data):
      - Columns 0:delay_dim                     → delay features
      - Columns delay_dim:delay_dim+weather_dim → weather features
    
    Returns:
        (delay_features, weather_features)
    """
    delay_features = X[:, :, :delay_dim]
    weather_features = X[:, :, delay_dim:delay_dim + weather_dim]
    return delay_features, weather_features


def remove_weather_from_inputs(
    train_x: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
    delay_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Remove weather data from inputs (keep only delay features).
    
    Args:
        train_x, val_x, test_x: Input arrays with shape (n, n_nodes, features)
        delay_dim: Number of delay feature columns
    
    Returns:
        (train_x_no_weather, val_x_no_weather, test_x_no_weather, new_feature_dim)
    """
    train_no_w = train_x[:, :, :delay_dim]
    val_no_w = val_x[:, :, :delay_dim]
    test_no_w = test_x[:, :, :delay_dim]
    return train_no_w, val_no_w, test_no_w, delay_dim


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING WRAPPER (reuse from stacked_gru_transformer.py)
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
    """Train with BCEWithLogitsLoss, warmup + cosine LR, gradient accumulation."""
    ei_adj  = edge_index_adj.to(device)
    ei_od   = edge_index_od.to(device)
    ei_od_t = edge_index_od_t.to(device)

    encoder_params    = list(model.encoder.parameters())
    gat_params        = list(model.gat.parameters())
    classifier_params = list(model.classifier.parameters())

    optimizer = torch.optim.AdamW([
        {"params": encoder_params,    "lr": lr,       "weight_decay": 1e-4},
        {"params": gat_params,        "lr": lr * 3,   "weight_decay": 1e-5},
        {"params": classifier_params, "lr": lr * 2,   "weight_decay": 1e-4},
    ])

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
                }, save_path)

        if epoch % 5 == 0 or epoch == epochs:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"      epoch {epoch:3d}/{epochs}  "
                  f"loss={np.mean(ep_losses):.4f}  val_f1={vm['f1']:.4f}  "
                  f"lr={cur_lr:.2e}")

        if es(vm["f1"], epoch):
            print(f"      early stop @ epoch {epoch}  best_val_f1={best_f1:.4f}")
            break

    train_sec = time.time() - t0

    if save_path is not None and best_state is not None:
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["train_sec"] = train_sec
        torch.save(ckpt, save_path)

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
# MAIN COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Weather Impact Comparison")
    p.add_argument("--data_source",        default="cdata", choices=["cdata", "udata"])
    p.add_argument("--seq_len",            type=int,   default=18)
    p.add_argument("--horizons",           type=int,   nargs="+", default=[12])
    p.add_argument("--delay_threshold",    type=float, default=5.0)
    p.add_argument("--class_threshold",    type=float, default=0.5)
    p.add_argument("--gru_dim",            type=int,   default=64)
    p.add_argument("--gru_layers",         type=int,   default=2)
    p.add_argument("--gru_heads",          type=int,   default=4)
    p.add_argument("--gat_hidden",         type=int,   default=64)
    p.add_argument("--gat_heads",          type=int,   default=2)
    p.add_argument("--classifier",         type=str,   default="TSiTPlus",
                   choices=["TSiTPlus", "ConvTranPlus", "both"])
    p.add_argument("--weather_file",       type=str,   default="weather_cn.npy")
    p.add_argument("--period_hours",       type=int,   default=24)
    p.add_argument("--epochs",             type=int,   default=50)
    p.add_argument("--batch_size",         type=int,   default=16)
    p.add_argument("--lr",                 type=float, default=1e-4)
    p.add_argument("--patience",           type=int,   default=12)
    p.add_argument("--accumulation_steps", type=int,   default=16)
    p.add_argument("--chunk_size",         type=int,   default=200)
    p.add_argument("--dropout",            type=float, default=0.15)
    p.add_argument("--seed",               type=int,   default=42)
    p.add_argument("--output_dir",         type=str,   default="auto")
    p.add_argument("--device",             type=str,   default="auto")
    return p.parse_args()


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

    print(f"\n{'='*70}")
    print(f"  WEATHER IMPACT COMPARISON - Stacked GRUAttention → Transformer")
    print(f"  Device: {device}  |  KAN available: {KAN_AVAILABLE}")
    print(f"  Classifiers: {', '.join(classifiers)}")
    print(f"{'='*70}\n")

    # ── 1. Load data WITH weather (standard pipeline) ──────────────────────────
    print("[1/5] Loading data WITH weather ...")
    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

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
    weather_dim = feature_dim - delay_dim
    max_horizon = sorted(set(args.horizons))[0]

    print(f"  Nodes: {num_nodes}  total_features: {feature_dim}  "
          f"  delay_features: {delay_dim}  weather_features: {weather_dim}  "
          f"  seq_len: {args.seq_len}")

    # ── 2. Build sequences (WITH weather) ─────────────────────────────────────
    print("[2/5] Building sequences (WITH weather) ...")
    train_x_with_w, _, train_y_cls = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    val_x_with_w, _, val_y_cls = build_sequences_node_level(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    test_x_with_w, _, test_y_cls = build_sequences_node_level(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )

    trX_w, trY, vaX_w, vaY, teX_w, teY = build_graph_tensors(
        train_x_with_w, train_y_cls, val_x_with_w, val_y_cls,
        test_x_with_w, test_y_cls, args.seq_len, feature_dim,
    )

    print(f"  WITH weather:  Train {tuple(trX_w.shape)}  Val {tuple(vaX_w.shape)}  Test {tuple(teX_w.shape)}")

    # ── 3. Build sequences (WITHOUT weather) ──────────────────────────────────
    print("[3/5] Building sequences (WITHOUT weather) ...")
    
    # Reconstruct inputs without weather
    train_inputs_no_w = train_inputs[:, :, :delay_dim]
    val_inputs_no_w = val_inputs[:, :, :delay_dim]
    test_inputs_no_w = test_inputs[:, :, :delay_dim]

    train_x_no_w, _, _ = build_sequences_node_level(
        train_inputs_no_w, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    val_x_no_w, _, _ = build_sequences_node_level(
        val_inputs_no_w, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    test_x_no_w, _, _ = build_sequences_node_level(
        test_inputs_no_w, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )

    trX_no_w, _, vaX_no_w, _, teX_no_w, _ = build_graph_tensors(
        train_x_no_w, train_y_cls, val_x_no_w, val_y_cls,
        test_x_no_w, test_y_cls, args.seq_len, delay_dim,
    )

    print(f"  WITHOUT weather: Train {tuple(trX_no_w.shape)}  Val {tuple(vaX_no_w.shape)}  Test {tuple(teX_no_w.shape)}")

    # ── 4. Prepare data loaders ────────────────────────────────────────────────
    print("[4/5] Preparing data loaders ...")
    
    trY = trY.float()
    vaY = vaY.float()
    teY = teY.float()

    train_loader_w = DataLoader(TensorDataset(trX_w, trY),
                                batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader_w   = DataLoader(TensorDataset(vaX_w, vaY),
                                batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader_w  = DataLoader(TensorDataset(teX_w, teY),
                                batch_size=args.batch_size, shuffle=False, drop_last=False)

    train_loader_no_w = DataLoader(TensorDataset(trX_no_w, trY),
                                   batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader_no_w   = DataLoader(TensorDataset(vaX_no_w, vaY),
                                   batch_size=args.batch_size, shuffle=False, drop_last=False)
    test_loader_no_w  = DataLoader(TensorDataset(teX_no_w, teY),
                                   batch_size=args.batch_size, shuffle=False, drop_last=False)

    pos_rate   = trY.reshape(-1, delay_dim).mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    # ── 5. Output directory ───────────────────────────────────────────────────
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (args.output_dir if args.output_dir != "auto"
           else f"weather_comparison_{ts}")
    os.makedirs(out, exist_ok=True)

    # ── 6. Train both variants ────────────────────────────────────────────────
    print(f"\n[5/5] Training {len(classifiers)} classifiers (WITH & WITHOUT weather) ...\n")

    results: Dict[str, Dict[str, float]] = {}
    timings: Dict[str, float] = {}
    n_params: Dict[str, int] = {}
    comparison_summary: List[Dict[str, str]] = []

    channel_names: Tuple[str, ...] = (
        ("arrival", "departure") if delay_dim == 2
        else ("arrival",) if delay_dim == 1
        else tuple(f"ch{i}" for i in range(delay_dim))
    )

    for clf_idx, clf_name in enumerate(classifiers, 1):
        print(f"\n{'─'*70}")
        print(f"[{clf_idx}/{len(classifiers)}] Classifier: {clf_name}")
        print(f"{'─'*70}\n")

        # ── Train WITH weather ────
        label_with = f"{clf_name}_WITH_weather"
        print(f"  Training WITH weather (input_dim={feature_dim}) ...")
        
        try:
            model_w = StackedGRUTransformer(
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

            np_w = sum(p.numel() for p in model_w.parameters() if p.requires_grad)
            n_params[label_with] = np_w
            print(f"    params: {np_w:,}")

            save_path_w = os.path.join(out, f"{clf_name}_WITH_weather_best.pth")
            metrics_w, time_w = train_and_evaluate(
                model_w, train_loader_w, val_loader_w, test_loader_w,
                edge_index_adj, edge_index_od, edge_index_od_t,
                n_nodes=num_nodes, device=device,
                epochs=args.epochs, lr=args.lr,
                pos_weight=pos_weight, patience=args.patience,
                class_threshold=args.class_threshold,
                save_path=save_path_w,
                accumulation_steps=args.accumulation_steps,
            )
            results[label_with] = metrics_w
            timings[label_with] = time_w
            print(f"    F1={metrics_w['f1']:.4f}  Acc={metrics_w['accuracy']:.4f}  ({time_w:.1f}s)")

        except Exception as e:
            print(f"    FAILED: {str(e).splitlines()[-1]}")
            results[label_with] = {}
            timings[label_with] = 0
            continue

        # ── Train WITHOUT weather ─
        label_no_w = f"{clf_name}_NO_weather"
        print(f"\n  Training WITHOUT weather (input_dim={delay_dim}) ...")
        
        try:
            model_no_w = StackedGRUTransformer(
                c_in=delay_dim,
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

            np_no_w = sum(p.numel() for p in model_no_w.parameters() if p.requires_grad)
            n_params[label_no_w] = np_no_w
            print(f"    params: {np_no_w:,}")

            save_path_no_w = os.path.join(out, f"{clf_name}_NO_weather_best.pth")
            metrics_no_w, time_no_w = train_and_evaluate(
                model_no_w, train_loader_no_w, val_loader_no_w, test_loader_no_w,
                edge_index_adj, edge_index_od, edge_index_od_t,
                n_nodes=num_nodes, device=device,
                epochs=args.epochs, lr=args.lr,
                pos_weight=pos_weight, patience=args.patience,
                class_threshold=args.class_threshold,
                save_path=save_path_no_w,
                accumulation_steps=args.accumulation_steps,
            )
            results[label_no_w] = metrics_no_w
            timings[label_no_w] = time_no_w
            print(f"    F1={metrics_no_w['f1']:.4f}  Acc={metrics_no_w['accuracy']:.4f}  ({time_no_w:.1f}s)")

        except Exception as e:
            print(f"    FAILED: {str(e).splitlines()[-1]}")
            results[label_no_w] = {}
            timings[label_no_w] = 0
            continue

        # ── Comparison ────────────
        if results.get(label_with) and results.get(label_no_w):
            m_w = results[label_with]
            m_no_w = results[label_no_w]
            f1_improvement = (m_w['f1'] - m_no_w['f1']) / (m_no_w['f1'] + 1e-6) * 100
            acc_improvement = (m_w['accuracy'] - m_no_w['accuracy']) / (m_no_w['accuracy'] + 1e-6) * 100

            print(f"\n  WEATHER IMPACT ({clf_name}):")
            print(f"    F1:  {m_no_w['f1']:.4f} → {m_w['f1']:.4f}  ({f1_improvement:+.2f}%)")
            print(f"    Acc: {m_no_w['accuracy']:.4f} → {m_w['accuracy']:.4f}  ({acc_improvement:+.2f}%)")
            print(f"    Params:  {np_no_w:,} → {np_w:,}")
            print(f"    Time:    {time_no_w:.1f}s → {time_w:.1f}s")

            # Store per-channel improvements
            for ch in channel_names:
                f1_ch_w = m_w.get(f'f1_{ch}', 0)
                f1_ch_no_w = m_no_w.get(f'f1_{ch}', 0)
                if f1_ch_no_w > 0:
                    ch_improvement = (f1_ch_w - f1_ch_no_w) / f1_ch_no_w * 100
                    print(f"    F1_{ch:3s}: {f1_ch_no_w:.4f} → {f1_ch_w:.4f}  ({ch_improvement:+.2f}%)")

            comparison_summary.append({
                "classifier": clf_name,
                "f1_with_weather": f"{m_w['f1']:.6f}",
                "f1_no_weather": f"{m_no_w['f1']:.6f}",
                "f1_improvement": f"{f1_improvement:+.2f}%",
                "accuracy_with_weather": f"{m_w['accuracy']:.6f}",
                "accuracy_no_weather": f"{m_no_w['accuracy']:.6f}",
                "accuracy_improvement": f"{acc_improvement:+.2f}%",
                "params_with": str(np_w),
                "params_without": str(np_no_w),
                "time_with": f"{time_w:.1f}s",
                "time_without": f"{time_no_w:.1f}s",
            })

    # ── 7. Write comparison summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Writing results to {out}/")
    print(f"{'='*70}\n")

    # Master comparison CSV
    comparison_path = os.path.join(out, "WEATHER_COMPARISON_SUMMARY.csv")
    if comparison_summary:
        with open(comparison_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=comparison_summary[0].keys())
            w.writeheader()
            w.writerows(comparison_summary)
        print(f"  Comparison summary: {comparison_path}")

    # Per-model detailed metrics
    for label, metrics in results.items():
        if metrics:
            csv_path = os.path.join(out, f"{label}_metrics.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["metric", "value"])
                for k, v in metrics.items():
                    w.writerow([k, f"{v:.6f}"])

    # Configuration log
    config_path = os.path.join(out, "CONFIG.txt")
    with open(config_path, "w") as f:
        f.write("WEATHER IMPACT COMPARISON - Configuration\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Data Source:        {args.data_source}\n")
        f.write(f"Weather File:       {args.weather_file}\n")
        f.write(f"Num Nodes:          {num_nodes}\n")
        f.write(f"Features WITH weather:    {feature_dim}\n")
        f.write(f"Features NO weather:      {delay_dim}\n")
        f.write(f"Seq Len:            {args.seq_len}\n")
        f.write(f"\nArchitecture:\n")
        f.write(f"  GRU: dim={args.gru_dim}, layers={args.gru_layers}, heads={args.gru_heads}\n")
        f.write(f"  GAT: hidden={args.gat_hidden}, heads={args.gat_heads}\n")
        f.write(f"  Classifiers: {', '.join(classifiers)}\n")
        f.write(f"\nTraining:\n")
        f.write(f"  Epochs:     {args.epochs}\n")
        f.write(f"  LR:         {args.lr}\n")
        f.write(f"  Batch Size: {args.batch_size}\n")
        f.write(f"  Device:     {device}\n")

    print(f"\nComparison output directory: {out}/\n")


if __name__ == "__main__":
    main()
