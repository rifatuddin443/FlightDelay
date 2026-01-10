import argparse
import os
import re
from datetime import datetime

import torch

from cnnopacus import (
    SequentialTwoStagePredictor,
    _run_tag,
    final_evaluation,
    load_flight_data,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level
from classifykat import build_sequences


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved CNN/Opacus checkpoint on the test split")
    p.add_argument(
        "--model_path",
        type=str,
        default="D:\\flight delay\\stpn paper\\STPN-main\\cnn_dp_three_stage_sigma1_00_20260110_154518.pth",
        help=(
            "Path to a saved .pth checkpoint. "
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
    p.add_argument("--class_threshold", type=float, default=0.5)
    p.add_argument("--use_node_level", action="store_true", default=True)
    p.add_argument("--weather_file", type=str, default="weather_cn.npy")
    p.add_argument("--period_hours", type=int, default=24)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--train_script",
        type=str,
        default="cnnopacus.py",
        help="Training script name to embed in filenames if checkpoint lacks run_tag",
    )
    return p.parse_args()


def _find_latest_checkpoint(search_dirs: list[str]) -> str | None:
    candidates: list[str] = []
    for d in search_dirs:
        if not d:
            continue
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            for fn in files:
                lower = fn.lower()
                if lower.endswith(".pth") or lower.endswith(".pt"):
                    candidates.append(os.path.join(root, fn))
    if not candidates:
        return None
    return max(candidates, key=lambda p: os.path.getmtime(p))


def _is_state_dict(obj: object) -> bool:
    if not isinstance(obj, dict):
        return False
    if not obj:
        return False
    return all(isinstance(k, str) for k in obj.keys()) and all(isinstance(v, torch.Tensor) for v in obj.values())


def _extract_sigma_from_path(model_path: str) -> float | None:
    # Matches sigma2_00 or sigma0_50 etc
    m = re.search(r"sigma(?P<num>\d+(?:_\d+)?)", os.path.basename(model_path))
    if not m:
        return None
    try:
        return float(m.group("num").replace("_", "."))
    except ValueError:
        return None


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
        run_tag = checkpoint.get("run_tag")
    else:
        noise_multiplier = 0.0
        dp_enabled = False
        target_epsilon = float("nan")
        final_epsilon = float("nan")
        final_delta = 0.0
        run_tag = None

    if (not noise_multiplier) or (noise_multiplier == 0.0):
        sigma_from_name = _extract_sigma_from_path(str(args.model_path))
        if sigma_from_name is not None:
            noise_multiplier = float(sigma_from_name)
            dp_enabled = bool(noise_multiplier > 0.0)

    if not run_tag:
        run_tag = _run_tag(
            train_script=str(args.train_script),
            data_source=str(args.data_source),
            noise_multiplier=float(noise_multiplier),
            dp_enabled=dp_enabled,
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

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

    test_x, test_y_reg, test_y_cls = build_fn(
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

    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=args.seq_len,
    ).to(device)

    # Note: We don't validate checkpoint keys here; `final_evaluation()` will load
    # compatible formats (submodule dicts or full state_dict) and raise a clear error otherwise.

    out_dir = os.path.dirname(args.model_path)
    print(f"Outputs will be saved next to checkpoint: {out_dir or '.'}")

    # Call shared evaluation/export; do NOT overwrite the checkpoint
    final_evaluation(
        model=model,
        edge_indices=edge_indices,
        device=device,
        scaler=scaler,
        horizons=horizons,
        delay_dim=delay_dim,
        num_nodes=num_nodes,
        test_x=test_x,
        test_y_reg=test_y_reg,
        test_y_cls=test_y_cls,
        class_threshold=float(args.class_threshold),
        delay_threshold=float(args.delay_threshold),
        model_path=str(args.model_path),
        run_tag=str(run_tag),
        timestamp=str(timestamp),
        histories=[],
        final_epsilon=final_epsilon,
        final_delta=final_delta,
        stage1_time=0.0,
        stage2_time=0.0,
        stage3_time=0.0,
        train_samples=0,
        val_samples=0,
        dp_enabled=dp_enabled,
        target_epsilon=target_epsilon,
        noise_multiplier=noise_multiplier,
        save_model=False,
        artifact_prefix="test",
    )


if __name__ == "__main__":
    main()
