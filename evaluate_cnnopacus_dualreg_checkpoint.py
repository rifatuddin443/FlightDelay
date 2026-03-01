"""Evaluate a saved cnnopacus dual-regressor checkpoint (no training).

This script loads a checkpoint produced by `cnnopacus - deepDualReg - kan.py` and
recomputes regression metrics on the test split. It also reports diagnostics to
separate:
- routing/gating quality (soft/hard gating)
- non-delayed head quality (head-only)

Example:
  python evaluate_cnnopacus_dualreg_checkpoint.py \
    --checkpoint "checkpoints\\...\\cnnopacus_-_deepDualReg_-_kan_...model...pth" \
    --data_source cdata --seq_len 18 --horizon 12 --delay_threshold 5 --class_threshold 0.5

Notes:
- Uses the same gating rule as the training script by default:
    gate = sigmoid((probs - class_threshold) * gate_temperature)
    pred = delayed*gate + nondelayed*(1-gate)
- By default clamps negative denormalized delays to 0, matching the training script.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch


def _load_training_module(script_path: str) -> Any:
    """Load the training script as a module (filename contains spaces/hyphens)."""
    script_path = str(Path(script_path).resolve())
    spec = importlib.util.spec_from_file_location("cnnopacus_dualreg_train", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _infer_hidden_channels(checkpoint: Dict[str, Any]) -> int:
    """Infer hidden_channels from the classifier weights (robust across formats)."""
    clf = checkpoint.get("classifier")
    if not isinstance(clf, dict):
        raise ValueError("Checkpoint missing 'classifier' state_dict; can't infer hidden size.")

    # Typical nn.Sequential keys: '0.weight' (hidden->hidden), ..., last linear -> out
    for key, tensor in clf.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if key.endswith("weight") and tensor.ndim == 2:
            # Prefer square layers if present.
            if tensor.shape[0] == tensor.shape[1] and tensor.shape[0] >= 8:
                return int(tensor.shape[0])

    # Fallback: take the largest 2D weight's input dimension.
    best = None
    for key, tensor in clf.items():
        if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
            if best is None or int(tensor.shape[1]) > best:
                best = int(tensor.shape[1])
    if best is None:
        raise ValueError("Could not infer hidden size from classifier state_dict.")
    return best


def _detect_stage3_regressor(checkpoint: Dict[str, Any]) -> str:
    """Best-effort detect whether Stage-3 regressor is KAN or MLP."""
    reg3 = checkpoint.get("regressor_nondelayed")
    if not isinstance(reg3, dict):
        return "mlp"

    keys = list(reg3.keys())
    # Heuristics: KAN implementations often contain 'grid'/'spline'/'coef'/'bases'.
    kan_markers = ("spline", "grid", "coef", "basis", "knots")
    if any(any(m in k.lower() for m in kan_markers) for k in keys):
        return "kan"

    # If it looks like a standard Sequential MLP
    if any(k.endswith("0.weight") for k in keys) or any("weight" in k for k in keys):
        return "mlp"

    return "mlp"


def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    mae = float(np.mean(np.abs(y_pred - y_true)))
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    return mae, rmse


def _evaluate_modes(
    *,
    preds_soft: np.ndarray,
    preds_hard: np.ndarray,
    preds_oracle: np.ndarray,
    preds_delayed_head: np.ndarray,
    preds_nondelayed_head: np.ndarray,
    targets: np.ndarray,
    delay_threshold: float,
) -> Dict[str, Dict[str, float]]:
    """Compute delayed/non-delayed/overall metrics for multiple prediction modes."""
    targets_flat = targets.flatten()

    delayed_mask = targets_flat > float(delay_threshold)
    nondelayed_mask = targets_flat <= float(delay_threshold)

    out: Dict[str, Dict[str, float]] = {}
    for name, preds in {
        "soft_mix": preds_soft,
        "hard_gate": preds_hard,
        "oracle_gate": preds_oracle,
        "delayed_head_only": preds_delayed_head,
        "nondelayed_head_only": preds_nondelayed_head,
    }.items():
        preds_flat = preds.flatten()

        mae_all, rmse_all = _mae_rmse(targets_flat, preds_flat)

        if delayed_mask.any():
            mae_d, rmse_d = _mae_rmse(targets_flat[delayed_mask], preds_flat[delayed_mask])
        else:
            mae_d, rmse_d = 0.0, 0.0

        if nondelayed_mask.any():
            mae_nd, rmse_nd = _mae_rmse(targets_flat[nondelayed_mask], preds_flat[nondelayed_mask])
        else:
            mae_nd, rmse_nd = 0.0, 0.0

        out[name] = {
            "mae_overall": mae_all,
            "rmse_overall": rmse_all,
            "mae_delayed": mae_d,
            "rmse_delayed": rmse_d,
            "mae_nondelayed": mae_nd,
            "rmse_nondelayed": rmse_nd,
            "n_delayed": int(delayed_mask.sum()),
            "n_nondelayed": int(nondelayed_mask.sum()),
        }

    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate cnnopacus dual-regressor checkpoint (no training)")
    p.add_argument(
        "--checkpoint",
        default=(
            "D:\\flight delay\\stpn paper\\STPN-main\\checkpoints\\cnnopacus_-_deepDualReg_-_kan_-_time_enable_stage3_cdata_sigma0_44_dualreg_model_eps7.31_20260129_042038\\cnnopacus_-_deepDualReg_-_kan_-_time_enable_stage3_cdata_sigma0_44_dualreg_model_eps7.31_20260129_042038.pth"
        ),
        type=str,
        help="Path to .pth checkpoint",
    )
    p.add_argument("--train_script", default="cnnopacus - deepDualReg - kan.py", type=str, help="Training script to load model class from")

    p.add_argument("--data_source", default="cdata", choices=["cdata", "udata"], type=str)
    p.add_argument("--seq_len", default=18, type=int)
    p.add_argument("--horizon", default=12, type=int, choices=[3, 6, 12, 24])

    p.add_argument("--delay_threshold", default=5.0, type=float)
    p.add_argument("--class_threshold", default=0.5, type=float, help="Used when --no_sweep")
    p.add_argument("--gate_temperature", default=10.0, type=float, help="Used when --no_sweep")

    # Sweep options (defaults are chosen to require zero CLI inputs).
    p.add_argument(
        "--no_sweep",
        action="store_true",
        help="Disable sweep; evaluate a single (class_threshold, gate_temperature) pair.",
    )
    p.add_argument("--sweep_threshold_start", default=0.30, type=float)
    p.add_argument("--sweep_threshold_stop", default=0.90, type=float)
    p.add_argument("--sweep_threshold_step", default=0.05, type=float)
    p.add_argument("--sweep_temps", default="5,10,20,30,50,80", type=str)
    p.add_argument(
        "--top_k",
        default=100,
        type=int,
        help="Print top-K configs.",
    )

    p.add_argument(
        "--sweep_sort",
        default="overall",
        choices=["overall", "delayed", "nondelayed", "weighted"],
        type=str,
        help="How to rank sweep configs (default: overall).",
    )
    p.add_argument("--w_overall", default=1.0, type=float, help="Used when --sweep_sort weighted")
    p.add_argument("--w_delayed", default=1.0, type=float, help="Used when --sweep_sort weighted")
    p.add_argument("--w_nondelayed", default=1.0, type=float, help="Used when --sweep_sort weighted")

    # Optional sweep constraints to enforce a routing quality level.
    p.add_argument("--min_tpr", default=None, type=float, help="Only keep rows with tpr(D->D) >= this")
    p.add_argument("--max_fpr", default=None, type=float, help="Only keep rows with fpr(ND->D) <= this")
    p.add_argument("--pareto_k", default=25, type=int, help="Print up to K Pareto-efficient rows")

    p.add_argument("--exclude_time_features", action="store_true", default=True, help="Exclude last 2 features (hour, day_of_week)")
    p.add_argument("--include_time_features", dest="exclude_time_features", action="store_false", help="Include time features")

    p.add_argument("--weather_file", default="weather_cn.npy", type=str)
    p.add_argument("--period_hours", default=24, type=int)

    p.add_argument("--hidden_channels", default=None, type=int, help="Override hidden_channels (else inferred from checkpoint)")
    p.add_argument("--stage3_regressor", default=None, choices=["mlp", "kan"], type=str, help="Override Stage-3 regressor type (else inferred)")

    p.add_argument("--no_clamp_negative", action="store_true", help="Do not clamp negative denorm values to 0")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(__file__).resolve().parent
    train_script_path = (root / args.train_script).resolve() if not os.path.isabs(args.train_script) else Path(args.train_script).resolve()
    if not train_script_path.exists():
        raise FileNotFoundError(f"Training script not found: {train_script_path}")

    ckpt_path = Path(args.checkpoint).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_mod = _load_training_module(str(train_script_path))

    checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected checkpoint to be a dict.")

    # Load data
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
    ) = train_mod.load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

    if args.exclude_time_features:
        train_inputs = train_inputs[:, :, :-2]
        val_inputs = val_inputs[:, :, :-2]
        test_inputs = test_inputs[:, :, :-2]

    feature_dim = int(train_inputs.shape[2])
    delay_dim = int(train_delay_scaled.shape[2])
    in_channels = int(args.seq_len * feature_dim)
    out_channels = int(delay_dim)

    hidden_channels = int(args.hidden_channels) if args.hidden_channels is not None else _infer_hidden_channels(checkpoint)
    stage3_regressor = str(args.stage3_regressor) if args.stage3_regressor is not None else _detect_stage3_regressor(checkpoint)

    print(f"Model config: in_channels={in_channels}, out_channels={out_channels}, hidden_channels={hidden_channels}, stage3_regressor={stage3_regressor}")

    model = train_mod.SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
        regressor_extra_layer=True,
        seq_len=int(args.seq_len),
        stage3_regressor=stage3_regressor,
    ).to(device)

    # Load weights (new-format dict expected)
    if "encoder" in checkpoint and "classifier" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        if "regressor_delayed" in checkpoint and "regressor_nondelayed" in checkpoint:
            model.regressor_delayed.load_state_dict(checkpoint["regressor_delayed"])
            model.regressor_nondelayed.load_state_dict(checkpoint["regressor_nondelayed"])
        elif "regressor" in checkpoint:
            model.regressor_delayed.load_state_dict(checkpoint["regressor"])
            model.regressor_nondelayed.load_state_dict(checkpoint["regressor"])
        else:
            raise ValueError("Checkpoint missing regressor weights.")
    else:
        raise ValueError("Checkpoint does not look like a cnnopacus dual-regressor checkpoint (missing encoder/classifier).")

    model.eval()

    build_fn = train_mod.build_sequences_node_level
    horizons = [int(args.horizon)]
    test_x, test_y_reg, test_y_cls = build_fn(
        test_inputs,
        test_delay_scaled,
        test_raw,
        int(args.seq_len),
        int(args.horizon),
        float(args.delay_threshold),
        horizons,
    )

    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )

    probs_list = []
    soft_list = []
    hard_list = []
    oracle_list = []
    delayed_list = []
    nondelayed_list = []
    targets_list = []

    with torch.no_grad():
        for i in range(len(test_x)):
            data = train_mod.Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )

            hidden, node_logits = model.forward_classifier(data)
            probs = torch.sigmoid(node_logits)

            hidden_dropped = model.dropout_reg(hidden)
            reg_delayed = model.regressor_delayed(hidden_dropped)
            reg_nondelayed = model.regressor_nondelayed(hidden_dropped)

            gate_soft = torch.sigmoid((probs - float(args.class_threshold)) * float(args.gate_temperature))
            gate_hard = (probs >= float(args.class_threshold)).float()

            pred_soft = reg_delayed * gate_soft + reg_nondelayed * (1.0 - gate_soft)
            pred_hard = reg_delayed * gate_hard + reg_nondelayed * (1.0 - gate_hard)

            probs_list.append(probs.cpu().numpy())
            soft_list.append(pred_soft.cpu().numpy())
            hard_list.append(pred_hard.cpu().numpy())
            delayed_list.append(reg_delayed.cpu().numpy())
            nondelayed_list.append(reg_nondelayed.cpu().numpy())
            targets_list.append(test_y_reg[i].cpu().numpy())

    probs_np = np.concatenate(probs_list, axis=0)
    soft_np = np.concatenate(soft_list, axis=0)
    hard_np = np.concatenate(hard_list, axis=0)
    delayed_np = np.concatenate(delayed_list, axis=0)
    nondelayed_np = np.concatenate(nondelayed_list, axis=0)
    targets_np = np.concatenate(targets_list, axis=0)

    # Denormalize
    if scaler is not None:
        soft_denorm = scaler.inverse_transform(soft_np)
        hard_denorm = scaler.inverse_transform(hard_np)
        delayed_denorm = scaler.inverse_transform(delayed_np)
        nondelayed_denorm = scaler.inverse_transform(nondelayed_np)
        targets_denorm = scaler.inverse_transform(targets_np)
    else:
        soft_denorm, hard_denorm = soft_np, hard_np
        delayed_denorm, nondelayed_denorm = delayed_np, nondelayed_np
        targets_denorm = targets_np

    # Oracle routing based on true targets in denorm space
    oracle_gate = (targets_denorm > float(args.delay_threshold)).astype(np.float32)
    oracle_denorm = delayed_denorm * oracle_gate + nondelayed_denorm * (1.0 - oracle_gate)

    if not args.no_clamp_negative:
        soft_denorm = np.maximum(0, soft_denorm)
        hard_denorm = np.maximum(0, hard_denorm)
        oracle_denorm = np.maximum(0, oracle_denorm)
        delayed_denorm = np.maximum(0, delayed_denorm)
        nondelayed_denorm = np.maximum(0, nondelayed_denorm)
        targets_denorm = np.maximum(0, targets_denorm)

    # Precompute masks once.
    targets_flat = targets_denorm.flatten()
    delayed_mask = targets_flat > float(args.delay_threshold)
    nondelayed_mask = targets_flat <= float(args.delay_threshold)

    # Gate diagnostics on true non-delayed set
    probs_flat = probs_np.flatten()
    if nondelayed_mask.any():
        nd_probs = probs_flat[nondelayed_mask]
        print("\n[GATE DIAGNOSTICS] On true non-delayed elements:")
        print(f"  mean prob(delayed): {nd_probs.mean():.4f} | p50: {np.median(nd_probs):.4f} | p90: {np.quantile(nd_probs, 0.9):.4f}")
        print(f"  fraction probs >= class_threshold: {(nd_probs >= float(args.class_threshold)).mean():.3f}")

    if args.no_sweep:
        metrics = _evaluate_modes(
            preds_soft=soft_denorm,
            preds_hard=hard_denorm,
            preds_oracle=oracle_denorm,
            preds_delayed_head=delayed_denorm,
            preds_nondelayed_head=nondelayed_denorm,
            targets=targets_denorm,
            delay_threshold=float(args.delay_threshold),
        )

        print("\n=== Regression metrics (minutes) ===")
        for mode in ["soft_mix", "hard_gate", "oracle_gate", "nondelayed_head_only", "delayed_head_only"]:
            m = metrics[mode]
            print(f"\n[{mode}]")
            print(f"  Overall:      MAE {m['mae_overall']:.4f} | RMSE {m['rmse_overall']:.4f}")
            print(f"  Delayed:      MAE {m['mae_delayed']:.4f} | RMSE {m['rmse_delayed']:.4f} (n={m['n_delayed']})")
            print(f"  Non-delayed:  MAE {m['mae_nondelayed']:.4f} | RMSE {m['rmse_nondelayed']:.4f} (n={m['n_nondelayed']})")

        print("\nInterpretation:")
        print("- Compare soft_mix vs hard_gate: if hard_gate improves non-delayed MAE, routing/soft mixing is hurting you.")
        print("- Compare nondelayed_head_only vs soft_mix on non-delayed: if head-only is far better, your non-delayed head is good; gating is the issue.")
        print("- oracle_gate is an upper bound assuming perfect routing by ground truth.")
        return

    # Sweep mode (default): avoid re-running model inference. Only recompute gating math.
    temps = [float(x.strip()) for x in str(args.sweep_temps).split(",") if x.strip()]
    if not temps:
        temps = [10.0]

    thr_values = np.arange(
        float(args.sweep_threshold_start),
        float(args.sweep_threshold_stop) + 1e-9,
        float(args.sweep_threshold_step),
        dtype=np.float64,
    )
    thr_values = [float(x) for x in thr_values]

    # Flatten arrays for fast metrics.
    delayed_flat = delayed_denorm.flatten()
    nondelayed_flat = nondelayed_denorm.flatten()

    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    results = []
    for thr in thr_values:
        # NOTE: probs_np can be 2D; we use flattened values to match element-wise targets.
        for temp in temps:
            gate_soft = _sigmoid_np((probs_flat - thr) * temp)
            preds_soft_flat = delayed_flat * gate_soft + nondelayed_flat * (1.0 - gate_soft)

            gate_hard = (probs_flat >= thr).astype(np.float32)
            preds_hard_flat = delayed_flat * gate_hard + nondelayed_flat * (1.0 - gate_hard)

            # Routing diagnostics (using hard decision). For soft gate, treat >0.5 as "delayed".
            if nondelayed_mask.any():
                fpr_hard = float(np.mean(gate_hard[nondelayed_mask]))
                fpr_soft = float(np.mean((gate_soft[nondelayed_mask] > 0.5).astype(np.float32)))
            else:
                fpr_hard = float("nan")
                fpr_soft = float("nan")

            if delayed_mask.any():
                tpr_hard = float(np.mean(gate_hard[delayed_mask]))
                tpr_soft = float(np.mean((gate_soft[delayed_mask] > 0.5).astype(np.float32)))
            else:
                tpr_hard = float("nan")
                tpr_soft = float("nan")

            # Non-delayed MAE is primary objective.
            if nondelayed_mask.any():
                mae_nd_soft = float(np.mean(np.abs(preds_soft_flat[nondelayed_mask] - targets_flat[nondelayed_mask])))
                mae_nd_hard = float(np.mean(np.abs(preds_hard_flat[nondelayed_mask] - targets_flat[nondelayed_mask])))
            else:
                mae_nd_soft = float("nan")
                mae_nd_hard = float("nan")

            if delayed_mask.any():
                mae_d_soft = float(np.mean(np.abs(preds_soft_flat[delayed_mask] - targets_flat[delayed_mask])))
                mae_d_hard = float(np.mean(np.abs(preds_hard_flat[delayed_mask] - targets_flat[delayed_mask])))
            else:
                mae_d_soft = float("nan")
                mae_d_hard = float("nan")

            # Also track overall MAE for tie-breaking.
            mae_all_soft = float(np.mean(np.abs(preds_soft_flat - targets_flat)))
            mae_all_hard = float(np.mean(np.abs(preds_hard_flat - targets_flat)))

            # Store both modes; user can pick what they care about.
            results.append(("soft_mix", thr, temp, mae_nd_soft, mae_d_soft, mae_all_soft, fpr_soft, tpr_soft))
            results.append(("hard_gate", thr, temp, mae_nd_hard, mae_d_hard, mae_all_hard, fpr_hard, tpr_hard))

    def _sort_key(row: tuple) -> tuple:
        # row = (mode, thr, temp, mae_nd, mae_d, mae_all, fpr, tpr)
        _, _, _, mae_nd, mae_d, mae_all, _, _ = row
        if args.sweep_sort == "overall":
            return (mae_all, mae_d, mae_nd)
        if args.sweep_sort == "delayed":
            return (mae_d, mae_all, mae_nd)
        if args.sweep_sort == "nondelayed":
            return (mae_nd, mae_all, mae_d)
        # weighted
        score = float(args.w_overall) * float(mae_all) + float(args.w_delayed) * float(mae_d) + float(args.w_nondelayed) * float(mae_nd)
        return (score, mae_all, mae_d, mae_nd)

    def _print_table(title: str, rows: list[tuple]) -> None:
        print(f"\n=== {title} ===")
        print("mode\tthr\ttemp\tmae_nondelayed\tmae_delayed\tmae_overall\tfpr(ND->D)\ttpr(D->D)")
        for mode, thr, temp, mae_nd, mae_d, mae_all, fpr, tpr in rows:
            print(f"{mode}\t{thr:.2f}\t{temp:.1f}\t{mae_nd:.4f}\t{mae_d:.4f}\t{mae_all:.4f}\t{fpr:.3f}\t{tpr:.3f}")

    def _passes_constraints(row: tuple) -> bool:
        # row = (mode, thr, temp, mae_nd, mae_d, mae_all, fpr, tpr)
        fpr = float(row[6])
        tpr = float(row[7])
        if args.min_tpr is not None and not (tpr >= float(args.min_tpr)):
            return False
        if args.max_fpr is not None and not (fpr <= float(args.max_fpr)):
            return False
        return True

    def _pareto_front(rows: list[tuple]) -> list[tuple]:
        """Return Pareto-efficient rows minimizing (overall, delayed, nondelayed)."""
        front: list[tuple] = []
        for r in rows:
            _, _, _, mae_nd, mae_d, mae_all, _, _ = r
            dominated = False
            for q in rows:
                if q is r:
                    continue
                _, _, _, q_nd, q_d, q_all, _, _ = q
                if (q_all <= mae_all and q_d <= mae_d and q_nd <= mae_nd) and (q_all < mae_all or q_d < mae_d or q_nd < mae_nd):
                    dominated = True
                    break
            if not dominated:
                front.append(r)
        # Present nicely sorted by overall first.
        return sorted(front, key=lambda r: (r[5], r[4], r[3]))

    # 1) Apply optional constraints, then print top-K by chosen objective.
    results_filtered = [r for r in results if _passes_constraints(r)]
    if not results_filtered:
        print("\n[WARN] No sweep rows passed constraints; printing unconstrained results.")
        results_filtered = results

    results_sorted = sorted(results_filtered, key=_sort_key)
    _print_table(f"Sweep results (sorted by {args.sweep_sort})", results_sorted[: max(1, int(args.top_k))])

    # 2) Always show the single best config for each key objective (within filtered set).
    best_overall = min(results_filtered, key=lambda r: (r[5], r[4], r[3]))
    best_delayed = min(results_filtered, key=lambda r: (r[4], r[5], r[3]))
    best_nondelayed = min(results_filtered, key=lambda r: (r[3], r[5], r[4]))

    _print_table("Best by overall MAE", [best_overall])
    _print_table("Best by delayed MAE", [best_delayed])
    _print_table("Best by non-delayed MAE", [best_nondelayed])

    # 3) Pareto front (tradeoff curve).
    pareto = _pareto_front(results_filtered)
    _print_table("Pareto-efficient (overall/delayed/nondelayed)", pareto[: max(1, int(args.pareto_k))])


if __name__ == "__main__":
    main()
