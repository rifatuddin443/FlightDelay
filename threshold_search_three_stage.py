"""
Threshold Search for Stacked GRU Three-Stage Predictor
======================================================
Load the three-stage checkpoint, run inference on val + test sets, sweep thresholds
on the *validation* set to pick optimal per-channel thresholds for the classifier stage,
then report final metrics.
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Run classifier part of the model, return (probs, targets) as numpy."""
    ei_adj, ei_od, ei_od_t = edge_indices
    model.eval()
    all_probs, all_targets = [], []
    for bx, by in loader:
        B = bx.size(0)
        bx = bx.to(device)
        # Handle batched edge indices if necessary, but forward_classifier handles raw ones typically or uses a cache.
        # Actually in stacked_gru_transformer_three_stage.py, it expects the base edge indices.
        logits = model.forward_classifier(bx, ei_adj, ei_od, ei_od_t)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(by.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)

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
    parser.add_argument("--regressor_name", type=str, default="mlp")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    print(f"\n{'='*65}")
    print(f"  Threshold Search — Stacked Three-Stage  |  device={device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*65}\n")

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

    # Detect feature dimension from checkpoint if possible, or strip manually 
    # based on the error we saw (expected 3, got 5).
    # The 5 probably includes weather (2 features).
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
    _, _, train_y_cls = build_sequences_node_level(
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

    # Reshape for graph
    # We only need val and test for threshold search
    # (n, n_nodes, seq_len, feature_dim) -> (n, n_nodes, feature_dim, seq_len)
    vaX = val_x.view(val_x.shape[0], num_nodes, args.seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    vaY = val_y_cls.view(val_y_cls.shape[0], num_nodes, -1).float()
    
    teX = test_x.view(test_x.shape[0], num_nodes, args.seq_len, feature_dim).permute(0, 1, 3, 2).contiguous()
    teY = test_y_cls.view(test_y_cls.shape[0], num_nodes, -1).float()

    n_nodes = vaX.shape[1]

    val_loader  = DataLoader(TensorDataset(vaX, vaY),
                             batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(teX, teY),
                             batch_size=args.batch_size, shuffle=False)

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

    # Use NBeatsRegressionHead for state dict loading based on user request
    from stacked_gru_transformer_three_stage import NBeatsRegressionHead
    # The feature dim for the regressor is gru_dim + gat_hidden
    # According to the __init__ in stacked_gru_transformer_three_stage.py
    reg_feature_dim = args.gru_dim + args.gat_hidden
    model.regressor_delayed = NBeatsRegressionHead(reg_feature_dim, delay_dim, args.seq_len).to(device)
    model.regressor_nondelayed = NBeatsRegressionHead(reg_feature_dim, delay_dim, args.seq_len).to(device)
    
    # Try loading directly if it's a state dict, or use ckpt["model_state"] if it's a dict
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print(f"  Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )

    # ── Collect predictions ───────────────────────────────────────────────────
    print("\n  Running inference on val + test sets ...")
    t0 = time.time()
    val_probs,  val_targets  = collect_predictions(model, val_loader,  edge_indices, n_nodes, device)
    test_probs, test_targets = collect_predictions(model, test_loader, edge_indices, n_nodes, device)
    print(f"  Inference done in {time.time()-t0:.1f}s")

    channel_names = ("arrival", "departure") if delay_dim == 2 else ("arrival",)

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY: Fine-grained per-channel
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  Strategy: Per-channel threshold sweep (val set)")
    print(f"{'='*65}")

    thresholds = np.arange(0.10, 0.91, 0.01)
    best_per_ch = np.full(delay_dim, 0.5)
    
    test_default = compute_metrics(test_probs, test_targets, 0.5, channel_names)

    for c in range(delay_dim):
        ch_name = channel_names[c] if c < len(channel_names) else f"ch{c}"
        best_ch_f1 = -1.0

        p2 = val_probs.reshape(-1, val_probs.shape[-1])[:, c]
        t2 = val_targets.reshape(-1, val_targets.shape[-1])[:, c]

        print(f"\n  Channel: {ch_name}")
        print(f"  {'Thresh':>7} {'F1':>7} {'Prec':>7} {'Rec':>7} {'Acc':>7}")
        print("  " + "─" * 40)

        for th in thresholds:
            pb = p2 >= th
            tb = t2 >= 0.5
            tp = int(np.logical_and( pb,  tb).sum())
            fp = int(np.logical_and( pb, ~tb).sum())
            fn = int(np.logical_and(~pb,  tb).sum())
            tn = int(np.logical_and(~pb, ~tb).sum())
            pr = tp / (tp + fp + 1e-8)
            re = tp / (tp + fn + 1e-8)
            f1 = 2 * pr * re / (pr + re + 1e-8)
            ac = (tp + tn) / (tp + tn + fp + fn + 1e-8)

            if f1 > best_ch_f1:
                best_ch_f1 = f1
                best_per_ch[c] = th

            if abs(th % 0.10) < 0.005 or abs(th - 0.50) < 0.005:
                print(f"  {th:7.2f} {f1:7.4f} {pr:7.4f} {re:7.4f} {ac:7.4f}")

        print(f"  >>> Best {ch_name} threshold: {best_per_ch[c]:.2f}  "
              f"(val F1_{ch_name}={best_ch_f1:.4f})")

    # Apply per-channel thresholds to TEST set
    test_perch = compute_metrics(test_probs, test_targets, best_per_ch, channel_names)

    print(f"\n{'─'*65}")
    print(f"  Per-channel thresholds: "
          + ", ".join(f"{channel_names[c]}={best_per_ch[c]:.2f}" for c in range(delay_dim)))
    print(f"\n  TEST @ default (0.50, 0.50):  "
          f"F1={test_default['f1']:.4f}  Acc={test_default['accuracy']:.4f}  "
          f"Prec={test_default['precision']:.4f}  Rec={test_default['recall']:.4f}")
    print(f"  TEST @ per-channel:           "
          f"F1={test_perch['f1']:.4f}  Acc={test_perch['accuracy']:.4f}  "
          f"Prec={test_perch['precision']:.4f}  Rec={test_perch['recall']:.4f}")
    delta_f1_pc = test_perch["f1"] - test_default["f1"]
    print(f"  F1 change: {delta_f1_pc:+.4f}")

if __name__ == "__main__":
    main()
