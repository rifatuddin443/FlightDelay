"""
Threshold Search for Stacked GRUAttn→TSiTPlus
==============================================
Load the saved checkpoint, run inference on val + test sets, sweep thresholds
on the *validation* set to pick optimal per-channel thresholds, then report
final test metrics at those thresholds.

Usage:
    python threshold_search_stacked.py
    python threshold_search_stacked.py --checkpoint path/to/.pth
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from classifykat import load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level
from stacked_gru_transformer import (
    StackedGRUTransformer,
    batch_edge_index,
    build_graph_tensors,
)


def compute_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold_arr: float | np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
) -> Dict[str, float]:
    """Compute per-channel + macro metrics at given threshold(s).
    threshold_arr can be a scalar or (c_out,) array for per-channel thresholds."""
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
    model: nn.Module,
    loader: DataLoader,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    n_nodes: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run model on a full data loader, return (probs, targets) as numpy."""
    ei_adj, ei_od, ei_od_t = edge_indices
    model.eval()
    all_probs, all_targets = [], []
    for bx, by in loader:
        B = bx.size(0)
        bx = bx.to(device)
        bei_adj  = batch_edge_index(ei_adj,  n_nodes, B)
        bei_od   = batch_edge_index(ei_od,   n_nodes, B)
        bei_od_t = batch_edge_index(ei_od_t, n_nodes, B)
        logits = model(bx, bei_adj, bei_od, bei_od_t)
        all_probs.append(torch.sigmoid(logits).cpu().numpy())
        all_targets.append(by.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser(description="Threshold search for stacked model")
    parser.add_argument("--checkpoint", type=str,
                        default="stacked_gru_transformer_20260301_194354/Stacked_GRUAttn_TSiTPlus_best.pth")
    parser.add_argument("--data_source", default="cdata")
    parser.add_argument("--seq_len", type=int, default=18)
    parser.add_argument("--horizons", type=int, nargs="+", default=[12])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--gru_dim", type=int, default=64)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_heads", type=int, default=4)
    parser.add_argument("--gat_hidden", type=int, default=64)
    parser.add_argument("--gat_heads", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = (torch.device("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else torch.device(args.device))

    print(f"\n{'='*65}")
    print(f"  Threshold Search — Stacked GRUAttn→TSiTPlus  |  device={device}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"{'='*65}\n")

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  Checkpoint epoch: {ckpt.get('epoch', '?')}  "
          f"val_f1: {ckpt.get('val_f1', '?'):.4f}  "
          f"train_sec: {ckpt.get('train_sec', '?'):.0f}")

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

    train_inputs = train_inputs[:, :, :-2]
    val_inputs   = val_inputs[:, :, :-2]
    test_inputs  = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim   = train_delay_scaled.shape[2]
    max_horizon = sorted(set(args.horizons))[0]

    # ── Build sequences ───────────────────────────────────────────────────────
    print("[2/3] Building sequences ...")
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
    vaY = vaY.float()
    teY = teY.float()

    val_loader  = DataLoader(TensorDataset(vaX, vaY),
                             batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(teX, teY),
                             batch_size=args.batch_size, shuffle=False)

    # ── Rebuild model & load weights ──────────────────────────────────────────
    print("[3/3] Loading model ...")
    model = StackedGRUTransformer(
        c_in=feature_dim, c_out=delay_dim, seq_len=args.seq_len,
        gru_dim=args.gru_dim, gru_layers=args.gru_layers,
        gru_heads=args.gru_heads, gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads, classifier_name="TSiTPlus",
        dropout=0.15, chunk_size=args.chunk_size,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
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
    # STRATEGY 1: Single global threshold (same for both channels)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  Strategy 1: Global threshold sweep (val set)")
    print(f"{'='*65}")

    thresholds = np.arange(0.20, 0.81, 0.01)
    best_global_f1  = -1.0
    best_global_th  = 0.5

    print(f"\n  {'Thresh':>7} {'F1':>7} {'Acc':>7} {'Prec':>7} {'Rec':>7}", end="")
    for ch in channel_names:
        print(f" {'F1_'+ch[:3]:>7}", end="")
    print()
    print("  " + "─" * 55)

    for th in thresholds:
        m = compute_metrics(val_probs, val_targets, th, channel_names)
        if m["f1"] > best_global_f1:
            best_global_f1 = m["f1"]
            best_global_th = th
        # Print every 0.05 step and the default 0.50
        if abs(th % 0.05) < 0.005 or abs(th - 0.50) < 0.005:
            row = f"  {th:7.2f} {m['f1']:7.4f} {m['accuracy']:7.4f} " \
                  f"{m['precision']:7.4f} {m['recall']:7.4f}"
            for ch in channel_names:
                row += f" {m.get(f'f1_{ch}', 0):7.4f}"
            print(row)

    print(f"\n  >>> Best global threshold: {best_global_th:.2f}  "
          f"(val F1={best_global_f1:.4f})")

    # Apply best global threshold to TEST set
    test_global = compute_metrics(test_probs, test_targets, best_global_th, channel_names)
    test_default = compute_metrics(test_probs, test_targets, 0.5, channel_names)

    print(f"\n  TEST @ default 0.50:  "
          f"F1={test_default['f1']:.4f}  Acc={test_default['accuracy']:.4f}  "
          f"Prec={test_default['precision']:.4f}  Rec={test_default['recall']:.4f}")
    print(f"  TEST @ best    {best_global_th:.2f}:  "
          f"F1={test_global['f1']:.4f}  Acc={test_global['accuracy']:.4f}  "
          f"Prec={test_global['precision']:.4f}  Rec={test_global['recall']:.4f}")
    delta_f1 = test_global["f1"] - test_default["f1"]
    print(f"  F1 change: {delta_f1:+.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 2: Per-channel threshold (independent optimization)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  Strategy 2: Per-channel threshold sweep (val set)")
    print(f"{'='*65}")

    best_per_ch = np.full(delay_dim, 0.5)

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

            if abs(th % 0.05) < 0.005 or abs(th - 0.50) < 0.005:
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

    # ═══════════════════════════════════════════════════════════════════════════
    # STRATEGY 3: Fine-grained grid around best per-channel (0.001 steps)
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  Strategy 3: Fine-grained per-channel (0.001 steps around best)")
    print(f"{'='*65}")

    fine_per_ch = best_per_ch.copy()

    for c in range(delay_dim):
        ch_name = channel_names[c] if c < len(channel_names) else f"ch{c}"
        center = best_per_ch[c]
        fine_th = np.arange(max(0.05, center - 0.10), min(0.95, center + 0.10), 0.001)
        best_fine_f1 = -1.0

        p2 = val_probs.reshape(-1, val_probs.shape[-1])[:, c]
        t2 = val_targets.reshape(-1, val_targets.shape[-1])[:, c]

        for th in fine_th:
            pb = p2 >= th
            tb = t2 >= 0.5
            tp = int(np.logical_and( pb,  tb).sum())
            fp = int(np.logical_and( pb, ~tb).sum())
            fn = int(np.logical_and(~pb,  tb).sum())
            tn = int(np.logical_and(~pb, ~tb).sum())
            pr = tp / (tp + fp + 1e-8)
            re = tp / (tp + fn + 1e-8)
            f1 = 2 * pr * re / (pr + re + 1e-8)

            if f1 > best_fine_f1:
                best_fine_f1 = f1
                fine_per_ch[c] = th

        print(f"  {ch_name}: {fine_per_ch[c]:.3f}  (val F1={best_fine_f1:.4f})")

    # Apply fine-grained thresholds to TEST set
    test_fine = compute_metrics(test_probs, test_targets, fine_per_ch, channel_names)

    print(f"\n  Fine-grained thresholds: "
          + ", ".join(f"{channel_names[c]}={fine_per_ch[c]:.3f}" for c in range(delay_dim)))
    print(f"\n  TEST @ fine-grained:  "
          f"F1={test_fine['f1']:.4f}  Acc={test_fine['accuracy']:.4f}  "
          f"Prec={test_fine['precision']:.4f}  Rec={test_fine['recall']:.4f}")
    delta_f1_fine = test_fine["f1"] - test_default["f1"]
    print(f"  F1 change vs default: {delta_f1_fine:+.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("  FINAL COMPARISON (all on TEST set)")
    print(f"{'='*65}\n")

    strategies = [
        ("Default (0.50)",       test_default, "0.50"),
        (f"Global ({best_global_th:.2f})", test_global, f"{best_global_th:.2f}"),
        ("Per-channel",          test_perch,
         "+".join(f"{best_per_ch[c]:.2f}" for c in range(delay_dim))),
        ("Fine-grained",         test_fine,
         "+".join(f"{fine_per_ch[c]:.3f}" for c in range(delay_dim))),
    ]

    print(f"  {'Strategy':<22} {'Thresh':<15} {'F1':>7} {'Acc':>7} "
          f"{'Prec':>7} {'Rec':>7}", end="")
    for ch in channel_names:
        print(f" {'F1_'+ch[:3]:>7}", end="")
    print(f" {'dF1':>7}")
    print("  " + "─" * (74 + 8 * len(channel_names)))

    for name, m, th_str in strategies:
        delta = m["f1"] - test_default["f1"]
        row = (f"  {name:<22} {th_str:<15} {m['f1']:7.4f} {m['accuracy']:7.4f} "
               f"{m['precision']:7.4f} {m['recall']:7.4f}")
        for ch in channel_names:
            row += f" {m.get(f'f1_{ch}', 0):7.4f}"
        row += f" {delta:+7.4f}"
        print(row)

    # Best overall
    best_strat = max(strategies, key=lambda x: x[1]["f1"])
    print(f"\n  >>> BEST: {best_strat[0]} — F1={best_strat[1]['f1']:.4f} "
          f"(threshold={best_strat[2]})")

    # Per-channel detail for best
    bm = best_strat[1]
    for ch in channel_names:
        print(f"      {ch}: F1={bm.get(f'f1_{ch}',0):.4f}  "
              f"Prec={bm.get(f'precision_{ch}',0):.4f}  "
              f"Rec={bm.get(f'recall_{ch}',0):.4f}  "
              f"Acc={bm.get(f'accuracy_{ch}',0):.4f}")


if __name__ == "__main__":
    main()
