"""
Threshold Search for Stacked GRU Three-Stage Predictor (Comprehensive)
======================================================================
Load the three-stage checkpoint, run inference on val + test sets, sweep thresholds,
and evaluate regression performance.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from classifykat import load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level
from stacked_gru_transformer import (
    batch_edge_index,
)
from stacked_gru_transformer_three_stage import (
    StackedGRUThreeStagePredictor,
    reshape_for_graph,
    NBeatsRegressionHead,
)

def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold_arr: float | np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
) -> Dict[str, float]:
    """Compute per-channel + macro metrics at given threshold(s)."""
    p2 = probs.reshape(-1, probs.shape[-1])
    t2 = targets.reshape(-1, targets.shape[-1])
    th = np.atleast_1d(np.asarray(threshold_arr, dtype=np.float64))
    if th.shape[0] == 1:
        th = np.broadcast_to(th, p2.shape[1])

    metrics: Dict[str, float] = {}
    precs, recs, f1s, accs = [], [], [], []
    for c in range(p2.shape[1]):
        pb = p2[:, c] >= th[c]
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
        metrics[f"threshold_{nm}"] = float(th[c])
        precs.append(pr); recs.append(re); f1s.append(f1); accs.append(ac)

    metrics["precision"] = float(np.mean(precs))
    metrics["recall"]    = float(np.mean(recs))
    metrics["f1"]        = float(np.mean(f1s))
    metrics["accuracy"]  = float(np.mean(accs))
    return metrics

@torch.no_grad()
def collect_predictions(
    model: StackedGRUThreeStagePredictor,
    loader: DataLoader,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_nodes: int,
    device: torch.device,
    scaler,
) -> Dict[str, np.ndarray]:
    """Run classifier and regressor parts, return probs, targets, and regression preds."""
    ei_adj, ei_od, ei_od_t = edge_indices
    model.eval()
    all_probs, all_targets = [], []
    all_reg_preds = []
    all_reg_delayed = []
    all_reg_nondelayed = []
    
    for bx, by in loader:
        bx = bx.to(device)
        # 1. Classification
        logits = model.forward_classifier(bx, ei_adj, ei_od, ei_od_t)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(by.numpy())

        # 2. Regression combined
        reg_delayed = model.forward_regressor(bx, ei_adj, ei_od, ei_od_t, which="delayed")
        reg_nondelayed = model.forward_regressor(bx, ei_adj, ei_od, ei_od_t, which="nondelayed")
        
        combined_reg = probs * reg_delayed + (1.0 - probs) * reg_nondelayed
        
        # Inverse transform
        def inv(x_torch):
            x_np = x_torch.cpu().numpy()
            shape = x_np.shape
            return scaler.inverse_transform(x_np.reshape(-1, shape[-1])).reshape(shape)

        all_reg_preds.append(inv(combined_reg))
        all_reg_delayed.append(inv(reg_delayed))
        all_reg_nondelayed.append(inv(reg_nondelayed))

    return {
        "probs": np.concatenate(all_probs),
        "targets": np.concatenate(all_targets),
        "reg_preds": np.concatenate(all_reg_preds),
        "reg_delayed": np.concatenate(all_reg_delayed),
        "reg_nondelayed": np.concatenate(all_reg_nondelayed)
    }

def compute_regression_metrics(preds: np.ndarray, targets_raw: np.ndarray):
    """Compute MAE, RMSE on raw (minutes) values."""
    mae = np.mean(np.abs(preds - targets_raw))
    rmse = np.sqrt(np.mean((preds - targets_raw)**2))
    return mae, rmse

def main():
    parser = argparse.ArgumentParser(description="Threshold search for stacked three-stage model")
    parser.add_argument("--checkpoint", type=str,
                        default="stacked_gru_three_stage_20260412_115836/stacked_gru_three_stage_best.pth")
    parser.add_argument("--data_source", default="cdata")
    parser.add_argument("--seq_len", type=int, default=18)
    parser.add_argument("--horizons", type=int, nargs="+", default=[12])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--gru_dim", type=int, default=64)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_heads", type=int, default=4)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--classifier_name", type=str, default="TSiTPlus")
    parser.add_argument("--regressor_name", type=str, default="nbeats")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    print(f"\n{'='*75}")
    print(f"  Comprehensive Threshold Search & Regression Eval |  device={device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*75}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        return
        
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  Checkpoint loaded.")

    # ── Load data ─────────────────────────────────────────────────────────────
    weather_file = ("weather2016_2021.npy" if args.data_source == "udata"
                    else "weather_cn.npy")
    print("\n[1/3] Loading data ...")
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source, weather_file=weather_file,
        period_hours=24, data_source=args.data_source,
    )

    if train_inputs.shape[2] == 5:
        print("  Stripping weather features (found 5, expected 3 based on checkpoint)")
        train_inputs = train_inputs[:, :, :-2]
        val_inputs   = val_inputs[:, :, :-2]
        test_inputs  = test_inputs[:, :, :-2]
    
    feature_dim = train_inputs.shape[2]
    delay_dim   = train_delay_scaled.shape[2]
    max_horizon = sorted(set(args.horizons))[0]

    # ── Build sequences ───────────────────────────────────────────────────────
    print("[2/3] Building sequences ...")
    # For three-stage, load_flight_data returns pre-split raw_data
    # and build_sequences_node_level applies sliding window correctly.
    # To get matching regression targets, we can simply build them from the same internal logic.
    
    val_x, val_y_reg, val_y_cls = build_sequences_node_level(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )
    test_x, test_y_reg, test_y_cls = build_sequences_node_level(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, args.horizons,
    )

    # To get raw values (minutes) for regression, we inverse transform the scaled targets
    # val_y_reg: (N_samples, N_nodes, C) or (N_samples * N_nodes, C)
    # build_sequences_node_level returns (N_samples, N_nodes, C) based on previous context.
    
    def get_real_y(y_scaled, scaler):
        n_samples, n_nodes, c = y_scaled.shape
        y_flat = y_scaled.view(-1, c).numpy()
        y_real = scaler.inverse_transform(y_flat)
        return y_real.reshape(n_samples, n_nodes, c)

    val_targets_raw = get_real_y(val_y_reg, scaler)
    test_targets_raw = get_real_y(test_y_reg, scaler)

    vaX = val_x.view(val_x.shape[0], num_nodes, args.seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    vaY = val_y_cls.view(val_y_cls.shape[0], num_nodes, -1).float()
    
    teX = test_x.view(test_x.shape[0], num_nodes, args.seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    teY = test_y_cls.view(test_y_cls.shape[0], num_nodes, -1).float()

    val_loader  = DataLoader(TensorDataset(vaX, vaY), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(teX, teY), batch_size=args.batch_size, shuffle=False)

    # ── Rebuild model & load weight ──────────────────────────────────────────
    print("[3/3] Loading model ...")
    model = StackedGRUThreeStagePredictor(
        c_in=feature_dim, c_out=delay_dim, seq_len=args.seq_len,
        gru_dim=args.gru_dim, gru_layers=args.gru_layers,
        gru_heads=args.gru_heads, gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads, classifier_name=args.classifier_name,
        regressor_name=args.regressor_name,
        dropout=0.15,
    ).to(device)

    reg_feature_dim = args.gru_dim + args.gat_hidden
    model.regressor_delayed = NBeatsRegressionHead(reg_feature_dim, delay_dim, args.seq_len).to(device)
    model.regressor_nondelayed = NBeatsRegressionHead(reg_feature_dim, delay_dim, args.seq_len).to(device)
    
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    edge_indices = (edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device))

    # ── Collect predictions ───────────────────────────────────────────────────
    print("\n  Running full inference (Classification + Regression) ...")
    t0 = time.time()
    val_out = collect_predictions(model, val_loader, edge_indices, num_nodes, device, scaler)
    test_out = collect_predictions(model, test_loader, edge_indices, num_nodes, device, scaler)
    print(f"  Inference done in {time.time()-t0:.1f}s")

    val_probs, val_targets = val_out["probs"], val_out["targets"]
    test_probs, test_targets = test_out["probs"], test_out["targets"]
    test_reg_preds = test_out["reg_preds"]

    channel_names = ("arrival", "departure") if delay_dim == 2 else ("arrival",)

    # Global Sweep
    print(f"\n{'='*75}\n  Overall F1 Threshold Sweep (val set)\n{'='*75}")
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_global_f1, best_global_th = -1.0, 0.5
    for th in thresholds:
        m = compute_metrics(val_probs, val_targets, th, channel_names)
        if m["f1"] > best_global_f1:
            best_global_f1, best_global_th = m["f1"], th
    print(f"  >>> Best Global Threshold: {best_global_th:.2f} (val F1={best_global_f1:.4f})")

    # Per-channel Sweep
    best_per_ch = np.full(delay_dim, 0.5)
    for c in range(delay_dim):
        best_ch_f1 = -1.0
        p2, t2 = val_probs.reshape(-1, delay_dim)[:, c], val_targets.reshape(-1, delay_dim)[:, c]
        for th in thresholds:
            pb, tb = p2 >= th, t2 >= 0.5
            tp, fp, fn = int((pb & tb).sum()), int((pb & ~tb).sum()), int((~pb & tb).sum())
            pr, re = tp/(tp+fp+1e-8), tp/(tp+fn+1e-8)
            f1 = 2*pr*re/(pr+re+1e-8)
            if f1 > best_ch_f1: best_ch_f1, best_per_ch[c] = f1, th

    # Final Test Performance
    test_def = compute_metrics(test_probs, test_targets, 0.5, channel_names)
    test_glo = compute_metrics(test_probs, test_targets, best_global_th, channel_names)
    test_pch = compute_metrics(test_probs, test_targets, best_per_ch, channel_names)

    print(f"\n{'='*75}\n  FINAL PERFORMANCE (TEST SET)\n{'='*75}")
    print(f"  Classification F1:  Default={test_def['f1']:.4f}, Global={test_glo['f1']:.4f}, Per-Ch={test_pch['f1']:.4f}")
    mae, rmse = compute_regression_metrics(test_reg_preds, test_targets_raw)
    print(f"  Regression (min):   MAE={mae:6.2f}, RMSE={rmse:6.2f}")

if __name__ == "__main__":
    main()
