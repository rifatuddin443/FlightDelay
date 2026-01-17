import argparse
import os
import re
from datetime import datetime

import numpy as np
import torch

from cnnopacus_simple import SequentialRegressor
from classifykat import build_sequences, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved CNN/Opacus SIMPLE checkpoint on the test split")
    p.add_argument(
        "--model_path",
        type=str,
        default="cnnopacus_simple_udata_sigma0_50_model_20260112_080706.pth",
        help=(
            "Path to a saved .pth checkpoint produced by cnnopacus_simple.py. "
            "Use 'latest' to auto-pick the most recent checkpoint in ./checkpoints (and current folder)."
        ),
    )
    p.add_argument("--data_source", type=str, default="udata", choices=["cdata", "udata"])
    p.add_argument("--seq_len", type=int, default=8)
    p.add_argument(
        "--horizons",
        type=int,
        nargs=1,
        default=[12],
        choices=[3, 6, 12, 24],
        help="Test ONLY this horizon (choose one). Example: --horizons 24",
    )
    p.add_argument("--delay_threshold", type=float, default=5.0)
    p.add_argument("--use_node_level", action="store_true", default=True)
    p.add_argument("--weather_file", type=str, default="weather_cn.npy")
    p.add_argument("--period_hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--save_csv",
        action="store_true",
        default=False,
        help="Save a small CSV with overall/delayed/nondelayed MAE/RMSE",
    )
    return p.parse_args()


def _find_latest_checkpoint(search_dirs: list[str]) -> str | None:
    candidates: list[str] = []
    for d in search_dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fn in files:
                lower = fn.lower()
                if lower.endswith(".pth") or lower.endswith(".pt"):
                    candidates.append(os.path.join(root, fn))
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _extract_sigma_from_path(model_path: str) -> float | None:
    # Matches sigma2_00 or sigma0_50 etc
    m = re.search(r"sigma(?P<num>\d+(?:_\d+)?)", os.path.basename(model_path))
    if not m:
        return None
    try:
        return float(m.group("num").replace("_", "."))
    except ValueError:
        return None


def _np_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    if yt.size == 0:
        return 0.0, 0.0
    mae = float(np.mean(np.abs(yp - yt)))
    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    return mae, rmse


def main() -> None:
    args = parse_args()

    if args.model_path is None or str(args.model_path).strip().lower() == "latest":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        latest = _find_latest_checkpoint([
            os.path.join(script_dir, "checkpoints"),
            script_dir,
        ])
        if latest is None:
            raise FileNotFoundError(
                "Could not find any .pth/.pt checkpoints under ./checkpoints or the project folder. "
                "Pass --model_path explicitly."
            )
        args.model_path = latest
        print(f"[INFO] Auto-selected latest checkpoint: {args.model_path}")

    if args.data_source == "udata" and args.weather_file == "weather_cn.npy":
        args.weather_file = "weather2016_2021.npy"

    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)

    # Metadata fallbacks (old checkpoints might not store these)
    if isinstance(checkpoint, dict):
        noise_multiplier = float(checkpoint.get("noise_multiplier", 0.0))
        dp_enabled = bool(checkpoint.get("dp_enabled", noise_multiplier > 0.0))
        target_epsilon = float(checkpoint.get("target_epsilon", float("nan")))
        final_epsilon = float(checkpoint.get("final_epsilon", float("nan")))
        final_delta = float(checkpoint.get("final_delta", 0.0))
    else:
        noise_multiplier = 0.0
        dp_enabled = False
        target_epsilon = float("nan")
        final_epsilon = float("nan")
        final_delta = 0.0

    if (not noise_multiplier) or (noise_multiplier == 0.0):
        sigma_from_name = _extract_sigma_from_path(str(args.model_path))
        if sigma_from_name is not None:
            noise_multiplier = float(sigma_from_name)
            dp_enabled = bool(noise_multiplier > 0.0)

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

    horizons = sorted({h for h in args.horizons if h > 0})
    if len(horizons) != 1:
        raise ValueError(f"Pass exactly one horizon via --horizons. Got: {args.horizons}")

    max_horizon = horizons[0]
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = delay_dim

    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    test_x, test_y_reg, _test_y_cls = build_fn(
        test_inputs,
        test_delay_scaled,
        test_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        horizons,
    )

    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )

    model = SequentialRegressor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=args.seq_len,
    ).to(device)

    # Load weights.
    if isinstance(checkpoint, dict) and "encoder" in checkpoint and "regressor" in checkpoint:
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.regressor.load_state_dict(checkpoint["regressor"])
    elif isinstance(checkpoint, dict) and ("state_dict" in checkpoint or "model_state_dict" in checkpoint):
        sd = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint has state_dict but it is not a dict")
        model.load_state_dict(sd, strict=False)
    elif isinstance(checkpoint, dict) and checkpoint and all(isinstance(k, str) for k in checkpoint.keys()):
        # Plain state_dict
        model.load_state_dict(checkpoint, strict=False)
    else:
        raise ValueError("Unrecognized checkpoint format")

    model.eval()

    preds_list: list[np.ndarray] = []
    targets_list: list[np.ndarray] = []

    print("[EVALUATION] Processing test samples...")
    with torch.no_grad():
        for i in range(len(test_x)):
            data = torch_geometric_data(
                x=test_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )
            node_pred = model(data)
            preds_list.append(node_pred.cpu().numpy())
            targets_list.append(test_y_reg[i].cpu().numpy())

            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")

    preds_scaled = np.concatenate(preds_list, axis=0)
    targets_scaled = np.concatenate(targets_list, axis=0)

    if scaler is not None:
        preds_denorm = scaler.inverse_transform(preds_scaled)
        targets_denorm = scaler.inverse_transform(targets_scaled)
    else:
        preds_denorm = preds_scaled
        targets_denorm = targets_scaled

    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)

    preds_flat = preds_denorm.reshape(-1)
    targets_flat = targets_denorm.reshape(-1)

    delayed_mask = targets_flat > float(args.delay_threshold)
    nondelayed_mask = ~delayed_mask

    mae_overall, rmse_overall = _np_mae_rmse(targets_denorm, preds_denorm)
    mae_delayed, rmse_delayed = _np_mae_rmse(targets_flat[delayed_mask], preds_flat[delayed_mask])
    mae_nondelayed, rmse_nondelayed = _np_mae_rmse(targets_flat[nondelayed_mask], preds_flat[nondelayed_mask])

    print("\nREGRESSION (overall):")
    print(f"  MAE: {mae_overall:.4f} min | RMSE: {rmse_overall:.4f} min")
    print(f"\nREGRESSION (delayed flights > {args.delay_threshold} min):")
    print(f"  MAE: {mae_delayed:.4f} min | RMSE: {rmse_delayed:.4f} min | N={int(delayed_mask.sum())}")
    print(f"\nREGRESSION (non-delayed flights <= {args.delay_threshold} min):")
    print(f"  MAE: {mae_nondelayed:.4f} min | RMSE: {rmse_nondelayed:.4f} min | N={int(nondelayed_mask.sum())}")

    print("\nPRIVACY (from checkpoint metadata):")
    print(f"  DP enabled: {dp_enabled}")
    print(f"  Noise multiplier (sigma): {noise_multiplier}")
    print(f"  Target ε: {target_epsilon}")
    print(f"  Final ε: {final_epsilon}")
    print(f"  Final δ: {final_delta}")

    if args.save_csv:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = os.path.join(os.path.dirname(args.model_path), f"test_cnnopacus_simple_summary_{ts}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            f.write("metric,value\n")
            f.write(f"mae_overall,{mae_overall}\n")
            f.write(f"rmse_overall,{rmse_overall}\n")
            f.write(f"mae_delayed,{mae_delayed}\n")
            f.write(f"rmse_delayed,{rmse_delayed}\n")
            f.write(f"mae_nondelayed,{mae_nondelayed}\n")
            f.write(f"rmse_nondelayed,{rmse_nondelayed}\n")
            f.write(f"num_delayed,{int(delayed_mask.sum())}\n")
            f.write(f"num_nondelayed,{int(nondelayed_mask.sum())}\n")
        print(f"\n✓ Wrote: {out_csv}")


def torch_geometric_data(**kwargs):
    """Local import helper so this script doesn't import torch_geometric unless needed."""
    from torch_geometric.data import Data

    return Data(**kwargs)


if __name__ == "__main__":
    main()
