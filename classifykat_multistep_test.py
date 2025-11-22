"""Standalone evaluator for 3/6/12-step delay predictions.

Loads a trained SequentialTwoStagePredictor checkpoint, reconstructs the
standard dataset splits, and reports per-horizon arrival/departure metrics
alongside classification performance. Designed for quick verification of
multi-step forecasts without retraining.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Reuse shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (  # noqa: E402
    SequentialTwoStagePredictor,
    build_sequences,
    classification_metrics,
    load_flight_data,
    regression_metrics,
    set_seed,
)
from baseline_methods import test_error  # noqa: E402


def _evaluate_multistep(
    model: SequentialTwoStagePredictor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    scaler,
    horizons: List[int],
    delay_dim: int,
    num_nodes: int,
    test_x: torch.Tensor,
    test_y_reg: torch.Tensor,
    test_y_cls: torch.Tensor,
    class_threshold: float,
) -> Tuple[
    Dict[str, float],
    Dict[int, Dict[str, float]],
    Dict[str, float],
    np.ndarray,
    np.ndarray,
]:
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    model.eval()

    logits_list, reg_list = [], []
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            logits, reg = model(data)
            logits_list.append(torch.sigmoid(logits).cpu().numpy())
            reg_list.append(reg.cpu().numpy())

    test_probs = np.array(logits_list)
    test_reg_preds = np.array(reg_list)

    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_y_cls.cpu().numpy().reshape(-1, 1),
    )

    test_mask = test_probs >= class_threshold
    gated_preds = test_reg_preds * test_mask

    num_forecast_steps = len(horizons)
    preds_flat = gated_preds.reshape(-1, num_forecast_steps * delay_dim)
    targets_flat = test_y_reg.cpu().numpy().reshape(-1, num_forecast_steps * delay_dim)

    preds_denorm = scaler.inverse_transform(preds_flat).reshape(gated_preds.shape)
    targets_denorm = scaler.inverse_transform(targets_flat).reshape(test_y_reg.shape)

    preds_h = preds_denorm.reshape(-1, num_nodes, num_forecast_steps, delay_dim)
    targets_h = targets_denorm.reshape(-1, num_nodes, num_forecast_steps, delay_dim)

    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, horizon in enumerate(horizons):
        arrival_preds = preds_h[:, :, idx, 0]
        arrival_targets = targets_h[:, :, idx, 0]
        dep_preds = preds_h[:, :, idx, 1]
        dep_targets = targets_h[:, :, idx, 1]

        arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
        dep_mae, dep_rmse, dep_r2 = test_error(dep_preds, dep_targets)
        per_horizon_metrics[horizon] = {
            "arrival_mae": arr_mae,
            "arrival_rmse": arr_rmse,
            "arrival_r2": arr_r2,
            "departure_mae": dep_mae,
            "departure_rmse": dep_rmse,
            "departure_r2": dep_r2,
        }

    targets_cls = test_y_cls.cpu().numpy()
    delayed_mask = np.broadcast_to(targets_cls.astype(bool), test_reg_preds.shape)
    reg_metrics = regression_metrics(preds_denorm, targets_denorm, delayed_mask)

    return test_cls_metrics, per_horizon_metrics, reg_metrics, preds_h, targets_h


def _load_model(
    model_path: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    device: torch.device,
) -> SequentialTwoStagePredictor:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if all(key in checkpoint for key in ("encoder", "classifier", "regressor")):
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.regressor.load_state_dict(checkpoint["regressor"])
    else:
        model.load_state_dict(checkpoint)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved KAN-GAT checkpoints on 3/6/12-step horizons.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="kan_gat_dp_multisteptest.pth",
        help="Path to the trained checkpoint (defaults to kan_gat_sequential_dp.pth)",
    )
    parser.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--summary_csv", type=str, default="kan_gat_multistep_test_summary.csv")
    parser.add_argument("--history_csv", type=str, default="kan_gat_multistep_test_predictions.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    horizons = sorted({h for h in args.horizons if h > 0})
    if not horizons:
        raise ValueError("Please provide at least one positive horizon.")
    max_horizon = max(horizons)

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]

    in_channels = args.seq_len * feature_dim
    out_channels = len(horizons) * delay_dim

    test_x, test_y_reg, test_y_cls = build_sequences(
        test_inputs,
        test_delay_scaled,
        test_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        target_horizons=horizons,
    )

    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )

    model = _load_model(
        args.model_path,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        device=device,
    )

    (
        cls_metrics,
        per_horizon_metrics,
        reg_metrics,
        preds_h,
        targets_h,
    ) = _evaluate_multistep(
        model,
        edge_indices,
        device,
        scaler,
        horizons,
        delay_dim,
        num_nodes,
        test_x,
        test_y_reg,
        test_y_cls,
        args.class_threshold,
    )

    print("\nCLASSIFICATION METRICS:")
    print(
        f"  Precision: {cls_metrics['precision']:.4f} | "
        f"Recall: {cls_metrics['recall']:.4f} | "
        f"F1: {cls_metrics['f1']:.4f} | Accuracy: {cls_metrics['accuracy']:.4f}"
    )

    print("\nREGRESSION (Delayed Nodes):")
    print(f"  MAE: {reg_metrics['mae']:.4f} min | RMSE: {reg_metrics['rmse']:.4f} min")

    print("\nPER-HORIZON RESULTS:")
    for horizon in horizons:
        metrics = per_horizon_metrics[horizon]
        print(
            f"  {horizon}-step Arrival  -> MAE: {metrics['arrival_mae']:.4f}, "
            f"RMSE: {metrics['arrival_rmse']:.4f}, R²: {metrics['arrival_r2']:.4f}"
        )
        print(
            f"  {horizon}-step Departure-> MAE: {metrics['departure_mae']:.4f}, "
            f"RMSE: {metrics['departure_rmse']:.4f}, R²: {metrics['departure_r2']:.4f}"
        )

    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for key, value in cls_metrics.items():
                writer.writerow([f"classification_{key}", value])
            writer.writerow(["regression_mae", reg_metrics["mae"]])
            writer.writerow(["regression_rmse", reg_metrics["rmse"]])
            for horizon, metrics in per_horizon_metrics.items():
                for metric_name, metric_value in metrics.items():
                    writer.writerow([f"{metric_name}_h{horizon}", metric_value])

    if args.history_csv:
        with open(args.history_csv, "w", newline="") as f:
            fieldnames = [
                "sample_index",
                "horizon",
                "arrival_pred",
                "arrival_target",
                "departure_pred",
                "departure_target",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            num_samples = preds_h.shape[0]
            for idx in range(num_samples):
                for h_idx, horizon in enumerate(horizons):
                    arrival_pred = preds_h[idx, :, h_idx, 0].mean()
                    arrival_target = targets_h[idx, :, h_idx, 0].mean()
                    departure_pred = preds_h[idx, :, h_idx, 1].mean()
                    departure_target = targets_h[idx, :, h_idx, 1].mean()
                    writer.writerow(
                        {
                            "sample_index": idx,
                            "horizon": horizon,
                            "arrival_pred": arrival_pred,
                            "arrival_target": arrival_target,
                            "departure_pred": departure_pred,
                            "departure_target": departure_target,
                        }
                    )

if __name__ == "__main__":
    main()
