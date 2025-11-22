"""Test script for classifykatdpnew.py DP models with 3/6/12-step evaluation.

Loads a trained checkpoint from classifykatdpnew.py and evaluates per-horizon
performance while preserving the graph-level aggregation logic and DP metadata.
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

# Reuse shared utilities and DP-specific functions
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


def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    """Aggregate node-level features to graph-level via mean pooling."""
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    """Convert node-level targets to graph-level."""
    if target.dim() == 0:
        return target.unsqueeze(0)
    elif target.dim() == 1:
        return target.mean(dim=0, keepdim=True)
    else:
        return target.mean(dim=0, keepdim=True)


def _evaluate_multistep_dp(
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
    """Evaluate with graph-level aggregation matching classifykatdpnew."""
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    model.eval()

    logits_list, reg_list = [], []
    targets_cls_list, targets_reg_list = [], []
    
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            node_logits, node_reg = model(data)
            
            # Apply graph-level aggregation
            graph_logit = aggregate_node_to_graph(node_logits)
            graph_reg = aggregate_node_to_graph(node_reg)
            
            # Aggregate targets to graph-level
            graph_cls_target = ensure_graph_level_target(test_y_cls[i])
            graph_reg_target = ensure_graph_level_target(test_y_reg[i])
            
            logits_list.append(torch.sigmoid(graph_logit).cpu().numpy())
            reg_list.append(graph_reg.cpu().numpy())
            targets_cls_list.append(graph_cls_target.cpu().numpy())
            targets_reg_list.append(graph_reg_target.cpu().numpy())

    test_probs = np.concatenate(logits_list, axis=0)  # (num_samples, 1)
    test_reg_preds = np.concatenate(reg_list, axis=0)  # (num_samples, out_channels)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)  # (num_samples, 1)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)  # (num_samples, out_channels)

    # Classification metrics
    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_cls_targets.reshape(-1, 1),
    )

    # Apply gating
    test_mask = test_probs >= class_threshold
    gated_preds = test_reg_preds * test_mask

    # Denormalize
    num_forecast_steps = len(horizons)
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(gated_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
    else:
        preds_denorm = gated_preds
        targets_denorm = test_reg_targets

    # Reshape for per-horizon analysis: from (samples, horizons*delay_dim) to (samples, horizons, delay_dim)
    preds_h = preds_denorm.reshape(-1, num_forecast_steps, delay_dim)
    targets_h = targets_denorm.reshape(-1, num_forecast_steps, delay_dim)

    per_horizon_metrics: Dict[int, Dict[str, float]] = {}
    for idx, horizon in enumerate(horizons):
        arrival_preds = preds_h[:, idx, 0]
        arrival_targets = targets_h[:, idx, 0]
        dep_preds = preds_h[:, idx, 1]
        dep_targets = targets_h[:, idx, 1]

        arr_mae = np.mean(np.abs(arrival_preds - arrival_targets))
        arr_rmse = np.sqrt(np.mean((arrival_preds - arrival_targets) ** 2))
        arr_r2 = 1 - np.sum((arrival_targets - arrival_preds) ** 2) / (np.sum((arrival_targets - np.mean(arrival_targets)) ** 2) + 1e-10)
        
        dep_mae = np.mean(np.abs(dep_preds - dep_targets))
        dep_rmse = np.sqrt(np.mean((dep_preds - dep_targets) ** 2))
        dep_r2 = 1 - np.sum((dep_targets - dep_preds) ** 2) / (np.sum((dep_targets - np.mean(dep_targets)) ** 2) + 1e-10)
        
        per_horizon_metrics[horizon] = {
            "arrival_mae": arr_mae,
            "arrival_rmse": arr_rmse,
            "arrival_r2": arr_r2,
            "departure_mae": dep_mae,
            "departure_rmse": dep_rmse,
            "departure_r2": dep_r2,
        }

    # Overall regression metrics on delayed samples
    delayed_mask = test_cls_targets.flatten() >= class_threshold
    reg_metrics = {}
    if delayed_mask.sum() > 0:
        delayed_preds = preds_denorm[delayed_mask]
        delayed_targets = targets_denorm[delayed_mask]
        reg_metrics['mae'] = np.mean(np.abs(delayed_preds - delayed_targets))
        reg_metrics['rmse'] = np.sqrt(np.mean((delayed_preds - delayed_targets) ** 2))
    else:
        reg_metrics['mae'] = 0.0
        reg_metrics['rmse'] = 0.0

    return test_cls_metrics, per_horizon_metrics, reg_metrics, preds_h, targets_h


def _load_model_with_dp_metadata(
    model_path: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    device: torch.device,
) -> Tuple[SequentialTwoStagePredictor, float, float]:
    """Load model and extract DP metadata from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    
    # Load model weights
    if all(key in checkpoint for key in ("encoder", "classifier", "regressor")):
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.regressor.load_state_dict(checkpoint["regressor"])
    else:
        model.load_state_dict(checkpoint)
    
    # Extract DP metadata
    final_epsilon = checkpoint.get('final_epsilon', float('inf'))
    final_delta = checkpoint.get('final_delta', 0.0)
    
    return model, final_epsilon, final_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test classifykatdpnew.py DP models on 3/6/12-step horizons.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="kan_gat_dp_proper.pth",
        help="Path to the DP-trained checkpoint",
    )
    parser.add_argument("--data_source", type=str, default="udata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--summary_csv", type=str, default="classifykatdpnew_test_summary.csv")
    parser.add_argument("--predictions_csv", type=str, default="classifykatdpnew_test_predictions.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

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

    # Build test sequences
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

    # Load model with DP metadata
    model, final_epsilon, final_delta = _load_model_with_dp_metadata(
        args.model_path,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        device=device,
    )

    print(f"\nLoaded DP model from: {args.model_path}")
    print(f"Final ε: {final_epsilon:.3f}")
    print(f"Final δ: {final_delta:.2e}")

    # Evaluate
    (
        cls_metrics,
        per_horizon_metrics,
        reg_metrics,
        preds_h,
        targets_h,
    ) = _evaluate_multistep_dp(
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

    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    
    print("\nCLASSIFICATION METRICS:")
    print(
        f"  Precision: {cls_metrics['precision']:.4f} | "
        f"Recall: {cls_metrics['recall']:.4f} | "
        f"F1: {cls_metrics['f1']:.4f} | Accuracy: {cls_metrics['accuracy']:.4f}"
    )

    print("\nREGRESSION (Delayed Samples):")
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

    # Save summary CSV with DP params
    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            
            # DP parameters
            writer.writerow(["final_epsilon", final_epsilon])
            writer.writerow(["final_delta", final_delta])
            writer.writerow(["model_path", args.model_path])
            writer.writerow(["data_source", args.data_source])
            writer.writerow(["seq_len", args.seq_len])
            writer.writerow(["delay_threshold", args.delay_threshold])
            writer.writerow(["class_threshold", args.class_threshold])
            
            # Classification metrics
            for key, value in cls_metrics.items():
                writer.writerow([f"classification_{key}", value])
            
            # Regression metrics
            writer.writerow(["regression_mae", reg_metrics["mae"]])
            writer.writerow(["regression_rmse", reg_metrics["rmse"]])
            
            # Per-horizon metrics
            for horizon, metrics in per_horizon_metrics.items():
                for metric_name, metric_value in metrics.items():
                    writer.writerow([f"{metric_name}_h{horizon}", metric_value])
        
        print(f"\n✓ Summary saved to: {args.summary_csv}")

    # Save predictions CSV
    if args.predictions_csv:
        with open(args.predictions_csv, "w", newline="") as f:
            fieldnames = [
                "sample_index",
                "horizon",
                "arrival_pred",
                "arrival_target",
                "departure_pred",
                "departure_target",
                "epsilon",
                "delta",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            num_samples = preds_h.shape[0]
            for idx in range(num_samples):
                for h_idx, horizon in enumerate(horizons):
                    arrival_pred = preds_h[idx, h_idx, 0]
                    arrival_target = targets_h[idx, h_idx, 0]
                    departure_pred = preds_h[idx, h_idx, 1]
                    departure_target = targets_h[idx, h_idx, 1]
                    writer.writerow(
                        {
                            "sample_index": idx,
                            "horizon": horizon,
                            "arrival_pred": arrival_pred,
                            "arrival_target": arrival_target,
                            "departure_pred": departure_pred,
                            "departure_target": departure_target,
                            "epsilon": final_epsilon,
                            "delta": final_delta,
                        }
                    )
        
        print(f"✓ Predictions saved to: {args.predictions_csv}")


if __name__ == "__main__":
    main()
