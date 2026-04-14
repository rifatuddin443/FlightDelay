"""
Threshold Search for Stacked GRU Three-Stage Predictor (Comprehensive) - Enhanced
================================================================================
Load the three-stage checkpoint, run inference, sweep thresholds with regression gating,
report delayed/non-delayed MAE, and per-channel details.
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
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    scaler,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_probs, all_targets = [], []
    all_reg_preds = []
    all_reg_delayed = []
    all_reg_nondelayed = []
    
    for bx, by in loader:
        B = bx.size(0)
        bx = bx.to(device)
        
        # In stacked_gru_transformer.py, edge indices are batched with offsets
        bei_adj = batch_edge_index(edge_index_adj, n_nodes, B)
        bei_od = batch_edge_index(edge_index_od, n_nodes, B)
        bei_od_t = batch_edge_index(edge_index_od_t, n_nodes, B)
        
        logits = model.forward_classifier(bx, bei_adj, bei_od, bei_od_t)
        probs = torch.sigmoid(logits)
        all_probs.append(probs.cpu().numpy())
        all_targets.append(by.numpy())

        reg_delayed = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="delayed")
        reg_nondelayed = model.forward_regressor(bx, bei_adj, bei_od, bei_od_t, which="nondelayed")
        
        combined_reg = probs * reg_delayed + (1.0 - probs) * reg_nondelayed
        
        def inv(x_torch):
            x_np = x_torch.cpu().numpy()
            s = x_np.shape
            return scaler.inverse_transform(x_np.reshape(-1, s[-1])).reshape(s)

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
    mae = np.mean(np.abs(preds - targets_raw))
    rmse = np.sqrt(np.mean((preds - targets_raw)**2))
    return mae, rmse

def main():
    parser = argparse.ArgumentParser(description="Threshold search for stacked three-stage model")
    parser.add_argument("--checkpoint", type=str, default="stacked_gru_three_stage_20260412_115836/stacked_gru_three_stage_best.pth")
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
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device))

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(args.data_source, weather_file=("weather2016_2021.npy" if args.data_source == "udata" else "weather_cn.npy"), period_hours=24, data_source=args.data_source)

    if train_inputs.shape[2] == 5:
        train_inputs, val_inputs, test_inputs = train_inputs[:, :, :-2], val_inputs[:, :, :-2], test_inputs[:, :, :-2]
    
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    max_horizon = sorted(set(args.horizons))[0]

    val_x, val_y_reg, val_y_cls = build_sequences_node_level(val_inputs, val_delay_scaled, val_raw, args.seq_len, max_horizon, args.delay_threshold, args.horizons)
    test_x, test_y_reg, test_y_cls = build_sequences_node_level(test_inputs, test_delay_scaled, test_raw, args.seq_len, max_horizon, args.delay_threshold, args.horizons)

    def get_real_y(y_scaled, scaler):
        n, nodes, c = y_scaled.shape
        return scaler.inverse_transform(y_scaled.view(-1, c).numpy()).reshape(n, nodes, c)

    val_targets_raw = get_real_y(val_y_reg, scaler)
    test_targets_raw = get_real_y(test_y_reg, scaler)

    vaX = val_x.view(val_x.shape[0], num_nodes, args.seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    vaY = val_y_cls.view(val_y_cls.shape[0], num_nodes, -1).float()
    teX = test_x.view(test_x.shape[0], num_nodes, args.seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    teY = test_y_cls.view(test_y_cls.shape[0], num_nodes, -1).float()

    val_loader = DataLoader(TensorDataset(vaX, vaY), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(teX, teY), batch_size=args.batch_size, shuffle=False)

    model = StackedGRUThreeStagePredictor(feature_dim, delay_dim, args.seq_len, args.gru_dim, args.gru_layers, args.gru_heads, args.gat_hidden, args.gat_heads, args.classifier_name, args.regressor_name, 0.15).to(device)
    reg_feature_dim = args.gru_dim + args.gat_hidden
    model.regressor_delayed = NBeatsRegressionHead(reg_feature_dim, delay_dim, args.seq_len).to(device)
    model.regressor_nondelayed = NBeatsRegressionHead(reg_feature_dim, delay_dim, args.seq_len).to(device)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()

    edge_indices = (edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device))
    ei_adj, ei_od, ei_od_t = edge_indices
    val_out = collect_predictions(model, val_loader, ei_adj, ei_od, ei_od_t, num_nodes, device, scaler)
    test_out = collect_predictions(model, test_loader, ei_adj, ei_od, ei_od_t, num_nodes, device, scaler)

    channel_names = ("arrival", "departure") if delay_dim == 2 else ("arrival",)
    
    print(f"\n{'='*95}")
    print(f"  Overall Gated Sweep (val set)")
    print(f"{'='*95}")
    print(f"  {'Thresh':>7} {'F1':>7} | {'MAE_hard':>9} {'RMSE_hard':>10}")
    print("  " + "─" * 45)

    thresholds = [1, 2, 5, 10, 20, 50, 100, 200, 500] # k values for softness
    best_mae, best_k = float('inf'), 100
    
    th = 0.5 # fixed threshold Let's evaluate at threshold
    
    # Pre-calculated delayed/non-delayed values for threshold sweep (hard gating)
    val_probs = val_out["probs"]
    val_targets = val_out["targets"]
    val_reg_delayed = val_out["reg_delayed"]
    val_reg_nondelayed = val_out["reg_nondelayed"]
    
    print(f"\n{'='*95}")
    print(f"  Overall Gated Sweep (Softness `k` at th=0.5)")
    print(f"{'='*95}")
    print(f"  {'k (Soft)':>10} | {'MAE_ovr':>9} {'RMSE_ovr':>10} | {'MAE_del':>9} {'MAE_nd':>9}")
    print("  " + "─" * 60)

    for k in thresholds:
        # Soft gating weight
        # logit mapping around th:
        # p_scaled = log(probs / (1 - probs)) - log(th / (1 - th))
        # weight = sigmoid(k * p_scaled) ?
        # Or simple: weight = 1 / (1 + exp(-k * (probs - th)))
        weight = 1.0 / (1.0 + np.exp(-k * (val_probs - th)))

        soft_preds = weight * val_reg_delayed + (1.0 - weight) * val_reg_nondelayed
        v_mae, v_rmse = compute_regression_metrics(soft_preds, val_targets_raw)
        
        # also compute delayed / non-delayed
        delayed_mask = (val_targets == 1)
        nondelayed_mask = (val_targets == 0)
        
        v_mae_del, _ = compute_regression_metrics(soft_preds[delayed_mask], val_targets_raw[delayed_mask])
        v_mae_nd, _ = compute_regression_metrics(soft_preds[nondelayed_mask], val_targets_raw[nondelayed_mask])

        # Track best MAE
        if v_mae < best_mae: 
            best_mae, best_k = v_mae, k
            
        print(f"  {k:10.1f} | {v_mae:9.2f} {v_rmse:10.2f} | {v_mae_del:9.2f} {v_mae_nd:9.2f}")

    print(f"\n  >>> Best Validation MAE: {best_mae:.2f} min at softness k = {best_k:.1f}")

    # Now run the BEST softness on the TEST set to get the final metrics
    test_probs = test_out["probs"]
    test_targets = test_out["targets"]
    test_reg_delayed = test_out["reg_delayed"]
    test_reg_nondelayed = test_out["reg_nondelayed"]
    
    test_weight = 1.0 / (1.0 + np.exp(-best_k * (test_probs - th)))
    hard_best_test = test_weight * test_reg_delayed + (1.0 - test_weight) * test_reg_nondelayed
    test_mae_best, test_rmse_best = compute_regression_metrics(hard_best_test, test_targets_raw)
    test_metrics_best = compute_metrics(test_probs, test_targets, th, channel_names)
    
    delayed_mask = (test_targets == 1)
    nondelayed_mask = (test_targets == 0)
    
    test_mae_del, test_rmse_del = compute_regression_metrics(hard_best_test[delayed_mask], test_targets_raw[delayed_mask])
    test_mae_nd, test_rmse_nd = compute_regression_metrics(hard_best_test[nondelayed_mask], test_targets_raw[nondelayed_mask])

    print(f"\n{'='*95}\n  FINAL PERFORMANCE (TEST SET) at Best Softness k={best_k:.1f} (th={th})\n{'='*95}")
    print(f"Classification F1: {test_metrics_best['f1']:.4f} (arr={test_metrics_best.get('f1_arrival', 0):.4f}, dep={test_metrics_best.get('f1_departure', 0):.4f})")
    print(f"Regression delayed>thr: MAE={test_mae_del:.4f}, RMSE={test_rmse_del:.4f}")
    print(f"Regression non-delayed: MAE={test_mae_nd:.4f}, RMSE={test_rmse_nd:.4f}")
    print(f"Regression overall: MAE={test_mae_best:.4f}, RMSE={test_rmse_best:.4f}")
    
    print("\n  Per-Channel Regression Breakdown:")
    for c in range(delay_dim):
        c_preds = hard_best_test[..., c]
        c_targs = test_targets_raw[..., c]
        c_mask_del = (test_targets[..., c] == 1)
        c_mask_nd  = (test_targets[..., c] == 0)

        cm, cr = compute_regression_metrics(c_preds, c_targs)
        cm_del, cr_del = compute_regression_metrics(c_preds[c_mask_del], c_targs[c_mask_del])
        cm_nd, cr_nd = compute_regression_metrics(c_preds[c_mask_nd], c_targs[c_mask_nd])

        print(f"    {channel_names[c]:<15}:")
        print(f"      Delayed      -> MAE={cm_del:6.2f}, RMSE={cr_del:6.2f}")
        print(f"      Non-Delayed  -> MAE={cm_nd:6.2f}, RMSE={cr_nd:6.2f}")
        print(f"      Overall      -> MAE={cm:6.2f}, RMSE={cr:6.2f}")

if __name__ == "__main__":
    main()
