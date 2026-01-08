"""Grid-search classifier gating threshold based on validation overall MAE.

This script is meant for the 3-stage pipeline:
- Stage 1: classifier produces delay probability p (per node, per channel)
- Stage 2: delayed regressor
- Stage 3: non-delayed regressor

We sweep a threshold t and evaluate the final routed prediction:
  pred = reg_delayed if p>=t else reg_nondelayed

Objective: minimize overall MAE in *minutes* on the validation set.

Usage (example):
  python grid_search_threshold_overall_mae.py --model_path auto --horizons 12 --use_node_level

Notes:
- Works with both legacy single-regressor checkpoints and newer dual-regressor checkpoints.
- Uses the same data loading + scaling as the training/evaluation scripts.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Local imports
import evaluate_regression_v4 as eval_v4
from classifykat import load_flight_data
from classifykat_balanced import build_sequences_node_level
from classifykat import build_sequences


class GraphSequenceData(Data):
    """PyG Data that correctly offsets our custom edge indices when batching."""

    def __inc__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key in {"edge_index_adj", "edge_index_od", "edge_index_od_t"}:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


def _inverse_transform_np(scaler, x: np.ndarray) -> np.ndarray:
    """Inverse-transform with StandardScaler-like object (mean/std attributes)."""
    if scaler is None:
        return x
    mean = getattr(scaler, "mean", None)
    std = getattr(scaler, "std", None)
    if mean is None or std is None:
        # Fall back to method if present
        if hasattr(scaler, "inverse_transform"):
            return scaler.inverse_transform(x)
        raise ValueError("Scaler has no mean/std and no inverse_transform")
    return x * std + mean


def _f1_score_binary(preds: np.ndarray, targets: np.ndarray) -> float:
    """Binary F1 for 0/1 arrays."""
    preds_b = preds.astype(bool)
    targets_b = targets.astype(bool)
    tp = np.logical_and(preds_b, targets_b).sum()
    fp = np.logical_and(preds_b, ~targets_b).sum()
    fn = np.logical_and(~preds_b, targets_b).sum()
    precision = float(tp / (tp + fp + 1e-12))
    recall = float(tp / (tp + fn + 1e-12))
    return float(2.0 * precision * recall / (precision + recall + 1e-12))


def _classification_f1_macro(
    probs: np.ndarray,
    targets_cls: np.ndarray,
    threshold: float,
) -> Tuple[float, float, float]:
    """Compute (macro_f1, f1_arrival, f1_departure) from probs/targets [N, C]."""
    if probs.ndim != 2:
        raise ValueError(f"Expected probs as [N, C], got shape {probs.shape}")
    if targets_cls.shape != probs.shape:
        raise ValueError(f"targets_cls shape {targets_cls.shape} must match probs shape {probs.shape}")

    preds = (probs >= threshold).astype(np.int32)
    targets = (targets_cls >= 0.5).astype(np.int32)

    if probs.shape[1] == 1:
        f1 = _f1_score_binary(preds[:, 0], targets[:, 0])
        return f1, f1, f1

    f1_arr = _f1_score_binary(preds[:, 0], targets[:, 0])
    f1_dep = _f1_score_binary(preds[:, 1], targets[:, 1])
    return float((f1_arr + f1_dep) / 2.0), float(f1_arr), float(f1_dep)


@torch.no_grad()
def _collect_val_outputs(
    model,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    batch_size: int,
    max_samples: int | None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Run the model once on the validation set and collect arrays.

    Returns:
      probs: [N, C]
      targets: [N, C]
      reg_delayed: [N, C]
      reg_nondelayed: [N, C] or None if not available
            cls_targets: [N, C]

    Here N = num_samples * num_nodes, C = channels (arrival/departure).
    """
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices

    has_dual = hasattr(model, "regressor_delayed") and hasattr(model, "regressor_nondelayed")

    n = int(val_x.shape[0])
    if max_samples is not None:
        n = min(n, int(max_samples))

    dataset: List[GraphSequenceData] = []
    for i in range(n):
        d = GraphSequenceData(
            x=val_x[i],
            y_reg=val_y_reg[i],
            y_cls=val_y_cls[i],
            edge_index_adj=edge_index_adj,
            edge_index_od=edge_index_od,
            edge_index_od_t=edge_index_od_t,
        )
        d.num_nodes = val_x[i].shape[0]
        dataset.append(d)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs_list: List[np.ndarray] = []
    targets_list: List[np.ndarray] = []
    cls_targets_list: List[np.ndarray] = []
    reg_delayed_list: List[np.ndarray] = []
    reg_nondelayed_list: List[np.ndarray] = []

    model.eval()
    seen = 0
    for batch in loader:
        batch = batch.to(device)

        hidden, logits = model.forward_classifier(batch)
        probs = torch.sigmoid(logits)
        hidden_dropped = model.dropout_reg(hidden)

        if has_dual:
            reg_delayed = model.regressor_delayed(hidden_dropped)
            reg_nondelayed = model.regressor_nondelayed(hidden_dropped)
        else:
            reg_delayed = model.forward_regressor(hidden)
            reg_nondelayed = None

        probs_list.append(probs.detach().cpu().numpy())
        targets_list.append(batch.y_reg.detach().cpu().numpy())
        cls_targets_list.append(batch.y_cls.detach().cpu().numpy())
        reg_delayed_list.append(reg_delayed.detach().cpu().numpy())
        if reg_nondelayed is not None:
            reg_nondelayed_list.append(reg_nondelayed.detach().cpu().numpy())

        seen += int(batch.num_graphs)
        if seen % 500 == 0 or seen >= n:
            print(f"  Collected {min(seen, n)}/{n} samples...")

    probs_all = np.concatenate(probs_list, axis=0)
    targets_all = np.concatenate(targets_list, axis=0)
    cls_targets_all = np.concatenate(cls_targets_list, axis=0)
    reg_delayed_all = np.concatenate(reg_delayed_list, axis=0)
    reg_nondelayed_all = (
        np.concatenate(reg_nondelayed_list, axis=0) if reg_nondelayed_list else None
    )

    return probs_all, targets_all, reg_delayed_all, reg_nondelayed_all, cls_targets_all


def _overall_mae_minutes(
    probs: np.ndarray,
    targets_scaled: np.ndarray,
    reg_delayed_scaled: np.ndarray,
    reg_nondelayed_scaled: np.ndarray | None,
    threshold: float,
    scaler,
) -> float:
    """Compute overall MAE in minutes for a given gating threshold."""
    if reg_nondelayed_scaled is None:
        mask = (probs >= threshold).astype(reg_delayed_scaled.dtype)
        preds_scaled = reg_delayed_scaled * mask
    else:
        mask = (probs >= threshold).astype(reg_delayed_scaled.dtype)
        preds_scaled = reg_delayed_scaled * mask + reg_nondelayed_scaled * (1.0 - mask)

    preds_denorm = _inverse_transform_np(scaler, preds_scaled)
    targets_denorm = _inverse_transform_np(scaler, targets_scaled)

    # Treat negative delays as on-time (0 min), consistent with evaluator
    preds_denorm = np.maximum(0.0, preds_denorm)
    targets_denorm = np.maximum(0.0, targets_denorm)

    return float(np.mean(np.abs(preds_denorm - targets_denorm)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid-search gating threshold on a chosen split (val/test)")
    p.add_argument("--model_path", type=str, default="auto", help="Path to checkpoint or 'auto' for latest")
    p.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument("--horizons", type=int, nargs=1, default=[12], choices=[3, 6, 12, 24])
    p.add_argument("--delay_threshold", type=float, default=5.0)
    p.add_argument("--use_node_level", action="store_true", default=True)
    p.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Which data split to run the sweep on (default: val).",
    )
    p.add_argument("--weather_file", type=str, default="weather_cn.npy")
    p.add_argument("--period_hours", type=int, default=24)
    p.add_argument("--hidden_channels", type=int, default=64)

    p.add_argument("--batch_size", type=int, default=32, help="PyG DataLoader batch size (graphs per batch)")
    p.add_argument("--max_samples", type=int, default=None, help="Optional cap on number of samples")

    p.add_argument("--t_min", type=float, default=0.05)
    p.add_argument("--t_max", type=float, default=0.95)
    p.add_argument("--t_steps", type=int, default=19)
    p.add_argument(
        "--metric",
        type=str,
        default="cls_f1",
        choices=["overall_mae", "cls_f1"],
        help="Grid-search objective: overall MAE (minutes) or classification macro-F1.",
    )
    p.add_argument("--save_csv", action="store_true", default=True)
    return p.parse_args()

def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    horizons = sorted({h for h in args.horizons if h > 0})
    if len(horizons) != 1:
        raise ValueError(f"Pass exactly one horizon via --horizons. Got: {args.horizons}")

    # Load data (same as training)
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
        num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

    max_horizon = horizons[0]
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = delay_dim

    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    print("[INFO] Using NODE-LEVEL labels" if args.use_node_level else "[INFO] Using GRAPH-LEVEL labels")

    if args.split == "test":
        split_name = "test"
        split_inputs = test_inputs
        split_delay_scaled = test_delay_scaled
        split_raw = test_raw
    else:
        split_name = "validation"
        split_inputs = val_inputs
        split_delay_scaled = val_delay_scaled
        split_raw = val_raw

    # Build sequences for selected split
    split_x, split_y_reg, split_y_cls = build_fn(
        split_inputs, split_delay_scaled, split_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons,
    )

    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )

    # Resolve model path
    model_path = args.model_path
    if model_path == "auto":
        model_path = eval_v4.find_latest_model()
        print(f"Auto-detected latest model: {model_path}")

    # Load model (reuse evaluator loader for architecture auto-detect + dual-regressor support)
    model, final_eps, final_delta, target_eps, epsilon_exceeded = eval_v4._load_three_stage_model(
        model_path=model_path,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        device=device,
    )

    print(f"\nLoaded checkpoint: {os.path.basename(model_path)}")
    print(f"  final_eps={final_eps:.3f}, delta={final_delta:.2e}")
    if target_eps is not None:
        print(f"  target_eps={target_eps:.3f}, exceeded={epsilon_exceeded}")

    # Collect outputs once
    print(f"\n[COLLECT] Running model on {split_name} set once...")
    probs, targets_scaled, reg_delayed_scaled, reg_nondelayed_scaled, cls_targets = _collect_val_outputs(
        model=model,
        edge_indices=edge_indices,
        device=device,
        val_x=split_x,
        val_y_reg=split_y_reg,
        val_y_cls=split_y_cls,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    print(f"Collected arrays: probs={probs.shape}, targets={targets_scaled.shape}, cls_targets={cls_targets.shape}")

    thresholds = np.linspace(args.t_min, args.t_max, args.t_steps)

    if args.metric == "cls_f1":
        best_t = None
        best_score = -1.0
        rows_f1: List[Tuple[float, float, float, float]] = []

        print("\n[GRID SEARCH] threshold -> classification macro-F1")
        for t in thresholds:
            f1_macro, f1_arr, f1_dep = _classification_f1_macro(
                probs=probs,
                targets_cls=cls_targets,
                threshold=float(t),
            )
            rows_f1.append((float(t), float(f1_macro), float(f1_arr), float(f1_dep)))
            print(f"  t={t:.3f} -> F1(macro)={f1_macro:.4f} | arrival={f1_arr:.4f} | departure={f1_dep:.4f}")
            if f1_macro > best_score:
                best_score = float(f1_macro)
                best_t = float(t)

        print("\n[BEST]")
        print(f"  threshold={best_t:.3f}  cls_F1_macro={best_score:.4f}")

        if args.save_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = f"threshold_gridsearch_cls_f1_{ts}.csv"
            with open(out_csv, "w", newline="") as f:
                f.write("threshold,f1_macro,f1_arrival,f1_departure\n")
                for t, f1m, f1a, f1d in rows_f1:
                    f.write(f"{t:.6f},{f1m:.6f},{f1a:.6f},{f1d:.6f}\n")
            print(f"\nSaved: {out_csv}")
    else:
        best_t = None
        best_mae = float("inf")
        rows_mae: List[Tuple[float, float]] = []

        print("\n[GRID SEARCH] threshold -> overall MAE (minutes)")
        for t in thresholds:
            mae = _overall_mae_minutes(
                probs=probs,
                targets_scaled=targets_scaled,
                reg_delayed_scaled=reg_delayed_scaled,
                reg_nondelayed_scaled=reg_nondelayed_scaled,
                threshold=float(t),
                scaler=scaler,
            )
            rows_mae.append((float(t), float(mae)))
            print(f"  t={t:.3f} -> MAE={mae:.4f}")
            if mae < best_mae:
                best_mae = float(mae)
                best_t = float(t)

        print("\n[BEST]")
        print(f"  threshold={best_t:.3f}  overall_MAE={best_mae:.4f} min")

        if args.save_csv:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_csv = f"threshold_gridsearch_overall_mae_{ts}.csv"
            with open(out_csv, "w", newline="") as f:
                f.write("threshold,overall_mae_minutes\n")
                for t, mae in rows_mae:
                    f.write(f"{t:.6f},{mae:.6f}\n")
            print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
