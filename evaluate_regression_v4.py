"""Test script for threestagev4noise.py models (single-horizon, DP, three-stage).

Designed to evaluate checkpoints produced by threestagev4noise.py, which trains
with a single forecast horizon, fixed Gaussian noise multiplier, and a Stage 3
fine-tune on flights under the delay threshold.
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch_geometric.data import Data

# Make plotting safe in headless runs (saves PNGs without interactive windows).
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

# Reuse shared utilities
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (  # noqa: E402
    SequentialTwoStagePredictor,
    build_sequences,
    classification_metrics,
    load_flight_data,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level  # noqa: E402

# Optional visualization utilities (shared across scripts)
try:
    from visualize_training_classification import (  # noqa: E402
        visualize_classification_results,
        visualize_regression_after_classification,
        visualize_regression_timeseries,
        visualize_three_stage_pipeline,
    )

    VISUALIZATION_AVAILABLE = True
except Exception:
    VISUALIZATION_AVAILABLE = False

# Import per-channel metrics from threestagev4noise
try:
    from threestagev4noise import classification_metrics_per_channel
except ImportError:
    # Fallback: define a simple wrapper
    def classification_metrics_per_channel(preds, targets, channel_names=('arrival', 'departure')):
        """Fallback wrapper for per-channel classification metrics."""
        basic_metrics = classification_metrics(preds, targets)
        # Add channel-specific placeholders
        for name in channel_names:
            basic_metrics[f'precision_{name}'] = basic_metrics['precision']
            basic_metrics[f'recall_{name}'] = basic_metrics['recall']
            basic_metrics[f'f1_{name}'] = basic_metrics['f1']
            basic_metrics[f'accuracy_{name}'] = basic_metrics['accuracy']
        return basic_metrics


def _evaluate_three_stage_per_horizon(
    model: SequentialTwoStagePredictor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    scaler,
    horizons: List[int],
    delay_dim: int,
    test_x: torch.Tensor,
    test_y_reg: torch.Tensor,
    test_y_cls: torch.Tensor,
    class_threshold: float,
    delay_threshold: float,
) -> Tuple[
    Dict[str, float],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
    Dict[int, Dict[str, float]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Evaluate three-stage model with separate metrics per horizon for delayed/non-delayed.
    
    Since the model uses classifier gating, predictions for delayed flights come through
    the classifier gate, while non-delayed flights get zero predictions.
    
    Returns:
        - Classification metrics
        - Per-horizon metrics for DELAYED flights (>5 min)
        - Per-horizon metrics for NON-DELAYED flights (<5 min)
        - Per-horizon metrics for OVERALL
        - Predictions array
        - Targets array
    """
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    
    model.eval()
    logits_list, reg_list = [], []
    targets_cls_list, targets_reg_list = [], []
    
    print("\n[EVALUATION] Processing test samples...")
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            
            # Get predictions (classifier + regressor)
            node_logits, node_reg = model(data)
            
            logits_list.append(torch.sigmoid(node_logits).cpu().numpy())
            reg_list.append(node_reg.cpu().numpy())
            targets_cls_list.append(test_y_cls[i].cpu().numpy())
            targets_reg_list.append(test_y_reg[i].cpu().numpy())
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")
    
    test_probs = np.concatenate(logits_list, axis=0)  # (num_samples*num_nodes, out_channels)
    test_reg_preds = np.concatenate(reg_list, axis=0)  # (num_samples*num_nodes, out_channels)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)  # (num_samples*num_nodes, out_channels)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)  # (num_samples*num_nodes, out_channels)
    
    # Classification metrics (per-channel: arrival/departure)
    test_cls_metrics = classification_metrics_per_channel(
        test_probs,
        test_cls_targets,
        channel_names=('arrival', 'departure'),
    )
    
    # Apply classifier gating (as done in training)
    test_mask = (test_probs >= class_threshold)
    gated_preds = test_reg_preds * test_mask
    
    print(f"\n[DENORMALIZATION] Checking predictions...")
    print(f"  Gated predictions shape: {gated_preds.shape}")
    print(f"  Gated predictions (scaled): min={gated_preds.min():.3f}, max={gated_preds.max():.3f}, mean={gated_preds.mean():.3f}")
    print(f"  Test targets (scaled): min={test_reg_targets.min():.3f}, max={test_reg_targets.max():.3f}, mean={test_reg_targets.mean():.3f}")
    
    # Denormalize
    num_forecast_steps = len(horizons)
    if scaler is not None:
        print(f"  Applying inverse transform with scaler...")
        print(f"    Scaler mean: {scaler.mean}, std: {scaler.std}")
        preds_denorm = scaler.inverse_transform(gated_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
        print(f"  After denormalization:")
        print(f"    Predictions: min={preds_denorm.min():.2f}, max={preds_denorm.max():.2f}, mean={preds_denorm.mean():.2f}")
        print(f"    Targets: min={targets_denorm.min():.2f}, max={targets_denorm.max():.2f}, mean={targets_denorm.mean():.2f}")
    else:
        preds_denorm = gated_preds
        targets_denorm = test_reg_targets
    
    # Treat negative values as on time (0 min)
    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)
    
    # Reshape for per-horizon analysis
    preds_h = preds_denorm.reshape(-1, num_forecast_steps, delay_dim)
    targets_h = targets_denorm.reshape(-1, num_forecast_steps, delay_dim)
    
    # Get ground truth masks based on ACTUAL delays (not predictions)
    # Delayed: >= threshold
    true_delayed_mask = np.any(targets_h >= delay_threshold, axis=(1, 2))
    
    # Non-delayed: 1 <= delay < threshold
    # Exclude samples that are purely < 1 min (on time)
    true_nondelayed_mask = (~true_delayed_mask) & np.any(targets_h >= 1.0, axis=(1, 2))
    
    print(f"\n[GROUND TRUTH] Delayed samples: {true_delayed_mask.sum()} | Non-delayed samples: {true_nondelayed_mask.sum()}")
    
    # Per-horizon metrics for DELAYED flights (>5 min)
    per_horizon_delayed_metrics: Dict[int, Dict[str, float]] = {}
    for idx, horizon in enumerate(horizons):
        if true_delayed_mask.sum() > 0:
            arrival_preds = preds_h[true_delayed_mask, idx, 0]
            arrival_targets = targets_h[true_delayed_mask, idx, 0]
            dep_preds = preds_h[true_delayed_mask, idx, 1]
            dep_targets = targets_h[true_delayed_mask, idx, 1]
            
            arr_mae = np.mean(np.abs(arrival_preds - arrival_targets))
            arr_rmse = np.sqrt(np.mean((arrival_preds - arrival_targets) ** 2))
            arr_r2 = 1 - np.sum((arrival_targets - arrival_preds) ** 2) / (
                np.sum((arrival_targets - np.mean(arrival_targets)) ** 2) + 1e-10
            )
            
            dep_mae = np.mean(np.abs(dep_preds - dep_targets))
            dep_rmse = np.sqrt(np.mean((dep_preds - dep_targets) ** 2))
            dep_r2 = 1 - np.sum((dep_targets - dep_preds) ** 2) / (
                np.sum((dep_targets - np.mean(dep_targets)) ** 2) + 1e-10
            )
            
            per_horizon_delayed_metrics[horizon] = {
                "arrival_mae": arr_mae,
                "arrival_rmse": arr_rmse,
                "arrival_r2": arr_r2,
                "departure_mae": dep_mae,
                "departure_rmse": dep_rmse,
                "departure_r2": dep_r2,
                "num_samples": int(true_delayed_mask.sum()),
                "mean_arrival_pred": np.mean(arrival_preds),
                "mean_arrival_target": np.mean(arrival_targets),
                "mean_departure_pred": np.mean(dep_preds),
                "mean_departure_target": np.mean(dep_targets),
            }
        else:
            per_horizon_delayed_metrics[horizon] = {
                "arrival_mae": 0.0,
                "arrival_rmse": 0.0,
                "arrival_r2": 0.0,
                "departure_mae": 0.0,
                "departure_rmse": 0.0,
                "departure_r2": 0.0,
                "num_samples": 0,
                "mean_arrival_pred": 0.0,
                "mean_arrival_target": 0.0,
                "mean_departure_pred": 0.0,
                "mean_departure_target": 0.0,
            }
    
    # Per-horizon metrics for NON-DELAYED flights (<5 min)
    # These should be very small due to Stage 3 training
    per_horizon_nondelayed_metrics: Dict[int, Dict[str, float]] = {}
    for idx, horizon in enumerate(horizons):
        if true_nondelayed_mask.sum() > 0:
            arrival_preds = preds_h[true_nondelayed_mask, idx, 0]
            arrival_targets = targets_h[true_nondelayed_mask, idx, 0]
            dep_preds = preds_h[true_nondelayed_mask, idx, 1]
            dep_targets = targets_h[true_nondelayed_mask, idx, 1]
            
            arr_mae = np.mean(np.abs(arrival_preds - arrival_targets))
            arr_rmse = np.sqrt(np.mean((arrival_preds - arrival_targets) ** 2))
            arr_r2 = 1 - np.sum((arrival_targets - arrival_preds) ** 2) / (
                np.sum((arrival_targets - np.mean(arrival_targets)) ** 2) + 1e-10
            )
            
            dep_mae = np.mean(np.abs(dep_preds - dep_targets))
            dep_rmse = np.sqrt(np.mean((dep_preds - dep_targets) ** 2))
            dep_r2 = 1 - np.sum((dep_targets - dep_preds) ** 2) / (
                np.sum((dep_targets - np.mean(dep_targets)) ** 2) + 1e-10
            )
            
            per_horizon_nondelayed_metrics[horizon] = {
                "arrival_mae": arr_mae,
                "arrival_rmse": arr_rmse,
                "arrival_r2": arr_r2,
                "departure_mae": dep_mae,
                "departure_rmse": dep_rmse,
                "departure_r2": dep_r2,
                "num_samples": int(true_nondelayed_mask.sum()),
                "mean_arrival_pred": np.mean(arrival_preds),
                "mean_arrival_target": np.mean(arrival_targets),
                "mean_departure_pred": np.mean(dep_preds),
                "mean_departure_target": np.mean(dep_targets),
            }
        else:
            per_horizon_nondelayed_metrics[horizon] = {
                "arrival_mae": 0.0,
                "arrival_rmse": 0.0,
                "arrival_r2": 0.0,
                "departure_mae": 0.0,
                "departure_rmse": 0.0,
                "departure_r2": 0.0,
                "num_samples": 0,
                "mean_arrival_pred": 0.0,
                "mean_arrival_target": 0.0,
                "mean_departure_pred": 0.0,
                "mean_departure_target": 0.0,
            }
    
    # Per-horizon metrics for OVERALL (all flights)
    per_horizon_overall_metrics: Dict[int, Dict[str, float]] = {}
    for idx, horizon in enumerate(horizons):
        arrival_preds = preds_h[:, idx, 0]
        arrival_targets = targets_h[:, idx, 0]
        dep_preds = preds_h[:, idx, 1]
        dep_targets = targets_h[:, idx, 1]
        
        arr_mae = np.mean(np.abs(arrival_preds - arrival_targets))
        arr_rmse = np.sqrt(np.mean((arrival_preds - arrival_targets) ** 2))
        arr_r2 = 1 - np.sum((arrival_targets - arrival_preds) ** 2) / (
            np.sum((arrival_targets - np.mean(arrival_targets)) ** 2) + 1e-10
        )
        
        dep_mae = np.mean(np.abs(dep_preds - dep_targets))
        dep_rmse = np.sqrt(np.mean((dep_preds - dep_targets) ** 2))
        dep_r2 = 1 - np.sum((dep_targets - dep_preds) ** 2) / (
            np.sum((dep_targets - np.mean(dep_targets)) ** 2) + 1e-10
        )
        
        per_horizon_overall_metrics[horizon] = {
            "arrival_mae": arr_mae,
            "arrival_rmse": arr_rmse,
            "arrival_r2": arr_r2,
            "departure_mae": dep_mae,
            "departure_rmse": dep_rmse,
            "departure_r2": dep_r2,
            "num_samples": len(preds_h),
            "mean_arrival_pred": np.mean(arrival_preds),
            "mean_arrival_target": np.mean(arrival_targets),
            "mean_departure_pred": np.mean(dep_preds),
            "mean_departure_target": np.mean(dep_targets),
        }
    
    return (
        test_cls_metrics,
        per_horizon_delayed_metrics,
        per_horizon_nondelayed_metrics,
        per_horizon_overall_metrics,
        preds_h,
        targets_h,
        test_probs,
        test_cls_targets,
    )


def _load_three_stage_model(
    model_path: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    device: torch.device,
) -> Tuple[SequentialTwoStagePredictor, float, float, Optional[float], bool]:
    """Load threestagev4noise checkpoint and extract DP metadata."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    
    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=hidden_channels,
    ).to(device)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Load model weights
    if all(key in checkpoint for key in ("encoder", "classifier", "regressor")):
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.classifier.load_state_dict(checkpoint["classifier"])
        model.regressor.load_state_dict(checkpoint["regressor"])
        print("✓ Loaded three-stage trained model (regressor trained on both delayed and non-delayed)")
    else:
        model.load_state_dict(checkpoint)
    
    # Extract DP metadata
    final_epsilon = checkpoint.get("final_epsilon", float("inf"))
    final_delta = checkpoint.get("final_delta", 0.0)
    target_epsilon = checkpoint.get("target_epsilon")
    epsilon_exceeded = checkpoint.get("epsilon_exceeded", False)
    
    return model, final_epsilon, final_delta, target_epsilon, epsilon_exceeded


def find_latest_model(pattern: str = "kan_gat_dp_three_stage_sigma*.pth") -> str:
    """Find the most recently created model file matching the pattern.

    Defaults to the sigma-stamped filenames produced by threestagev4noise.py and
    falls back to the broader three-stage pattern if none are found.
    """
    model_files = glob.glob(pattern)
    if not model_files and pattern != "kan_gat_dp_three_stage_*.pth":
        model_files = glob.glob("kan_gat_dp_three_stage_*.pth")
    if not model_files:
        raise FileNotFoundError(
            f"No model files found matching pattern: {pattern}\n"
            f"Please train a model first using threestagev4noise.py or specify --model_path"
        )
    latest_model = max(model_files, key=os.path.getmtime)
    return latest_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test three-stage DP models with separate horizon-wise evaluation.",
    )
    parser.add_argument("--model_path",       type=str,        default="auto",
        help="Path to the three-stage trained checkpoint. Use 'auto' to find the latest kan_gat_dp_three_stage_*.pth file.",
    )
    parser.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument(
        "--horizons",
        type=int,
        nargs=1,
        default=[12],
        choices=[3, 6, 12, 24],
        help="Single forecast horizon (must match threestagev4noise training run).",
    )
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--use_node_level", action="store_true", default=True, help="Use node-level labels")
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--hidden_channels", type=int, default=64, help="Hidden channels (must match training: threestagev4noise uses 64)")
    parser.add_argument("--summary_csv", type=str, default="three_stage_test_summary.csv")
    parser.add_argument("--predictions_csv", type=str, default="three_stage_test_predictions.csv")
    parser.add_argument("--results_table_csv", type=str, default="results_table.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_results_table(
    csv_path: str,
    epsilon: float,
    delta: float,
    target_epsilon: Optional[float],
    epsilon_exceeded: bool,
    per_horizon_delayed_metrics: Dict[int, Dict[str, float]],
    per_horizon_nondelayed_metrics: Dict[int, Dict[str, float]],
    per_horizon_overall_metrics: Dict[int, Dict[str, float]],
    cls_metrics: Dict[str, float],
    horizons: List[int],
    model_path: str,
    args,
) -> None:
    """Save comprehensive results table with all metrics."""
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header section
        writer.writerow(["=" * 80])
        writer.writerow(["COMPREHENSIVE EVALUATION RESULTS"])
        writer.writerow(["=" * 80])
        writer.writerow([])
        
        # Model and configuration info
        writer.writerow(["MODEL INFORMATION"])
        writer.writerow(["Model Path", model_path])
        writer.writerow(["Model Type", "Three-Stage DP"])
        writer.writerow(["Data Source", args.data_source])
        writer.writerow(["Sequence Length", args.seq_len])
        writer.writerow(["Delay Threshold", f"{args.delay_threshold} min"])
        writer.writerow(["Classification Threshold", args.class_threshold])
        writer.writerow(["Final Epsilon", f"{epsilon:.3f}"])
        if target_epsilon is not None:
            writer.writerow(["Target Epsilon", f"{target_epsilon:.3f}"])
        writer.writerow(["Final Delta", f"{delta:.2e}"])
        writer.writerow(["Epsilon Exceeded", epsilon_exceeded])
        writer.writerow([])
        
        # Classification metrics
        writer.writerow(["=" * 80])
        writer.writerow(["CLASSIFICATION METRICS (Macro over Arrival/Departure)"])
        writer.writerow(["=" * 80])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Precision", f"{cls_metrics['precision']:.4f}"])
        writer.writerow(["Recall", f"{cls_metrics['recall']:.4f}"])
        writer.writerow(["F1 Score", f"{cls_metrics['f1']:.4f}"])
        writer.writerow(["Accuracy", f"{cls_metrics['accuracy']:.4f}"])
        writer.writerow([])
        writer.writerow(["Per-Channel Metrics:"])
        writer.writerow(["Channel", "Precision", "Recall", "F1 Score", "Accuracy"])
        writer.writerow([
            "Arrival",
            f"{cls_metrics.get('precision_arrival', 0):.4f}",
            f"{cls_metrics.get('recall_arrival', 0):.4f}",
            f"{cls_metrics.get('f1_arrival', 0):.4f}",
            f"{cls_metrics.get('accuracy_arrival', 0):.4f}"
        ])
        writer.writerow([
            "Departure",
            f"{cls_metrics.get('precision_departure', 0):.4f}",
            f"{cls_metrics.get('recall_departure', 0):.4f}",
            f"{cls_metrics.get('f1_departure', 0):.4f}",
            f"{cls_metrics.get('accuracy_departure', 0):.4f}"
        ])
        writer.writerow([])
        
        # Overall summary table (the main table you wanted)
        writer.writerow(["=" * 80])
        writer.writerow(["OVERALL SUMMARY TABLE"])
        writer.writerow(["=" * 80])
        
        # Calculate overall MAE
        all_arrival_maes = [per_horizon_overall_metrics[h]["arrival_mae"] for h in horizons]
        all_departure_maes = [per_horizon_overall_metrics[h]["departure_mae"] for h in horizons]
        overall_mae = np.mean(all_arrival_maes + all_departure_maes)
        
        # Create header row
        header = ["Epsilon", "Overall MAE (min)"]
        for h in horizons:
            header.append(f"{h}-step Arrival")
        for h in horizons:
            header.append(f"{h}-step Departure")
        header.extend(["Precision", "Recall", "F1 Score", "Accuracy"])
        writer.writerow(header)
        
        # Create data row
        row = [f"{epsilon:.2f}", f"{overall_mae:.4f}"]
        for h in horizons:
            row.append(f"{per_horizon_overall_metrics[h]['arrival_mae']:.4f}")
        for h in horizons:
            row.append(f"{per_horizon_overall_metrics[h]['departure_mae']:.4f}")
        row.extend([
            f"{cls_metrics['precision']:.4f}",
            f"{cls_metrics['recall']:.4f}",
            f"{cls_metrics['f1']:.4f}",
            f"{cls_metrics['accuracy']:.4f}"
        ])
        writer.writerow(row)
        writer.writerow([])
        
        # Detailed per-horizon metrics for DELAYED flights
        writer.writerow(["=" * 80])
        writer.writerow(["DETAILED METRICS - DELAYED FLIGHTS (>5 min)"])
        writer.writerow(["=" * 80])
        writer.writerow(["Horizon", "Samples", "Arrival MAE", "Arrival RMSE", "Arrival R2", 
                        "Departure MAE", "Departure RMSE", "Departure R2",
                        "Mean Arr Pred", "Mean Arr Target", "Mean Dep Pred", "Mean Dep Target"])
        
        for horizon in horizons:
            m = per_horizon_delayed_metrics[horizon]
            writer.writerow([
                f"{horizon}-step",
                m["num_samples"],
                f"{m['arrival_mae']:.4f}",
                f"{m['arrival_rmse']:.4f}",
                f"{m['arrival_r2']:.4f}",
                f"{m['departure_mae']:.4f}",
                f"{m['departure_rmse']:.4f}",
                f"{m['departure_r2']:.4f}",
                f"{m['mean_arrival_pred']:.2f}",
                f"{m['mean_arrival_target']:.2f}",
                f"{m['mean_departure_pred']:.2f}",
                f"{m['mean_departure_target']:.2f}",
            ])
        writer.writerow([])
        
        # Detailed per-horizon metrics for NON-DELAYED flights
        writer.writerow(["=" * 80])
        writer.writerow(["DETAILED METRICS - NON-DELAYED FLIGHTS (<5 min)"])
        writer.writerow(["=" * 80])
        writer.writerow(["Horizon", "Samples", "Arrival MAE", "Arrival RMSE", "Arrival R2", 
                        "Departure MAE", "Departure RMSE", "Departure R2",
                        "Mean Arr Pred", "Mean Arr Target", "Mean Dep Pred", "Mean Dep Target"])
        
        for horizon in horizons:
            m = per_horizon_nondelayed_metrics[horizon]
            writer.writerow([
                f"{horizon}-step",
                m["num_samples"],
                f"{m['arrival_mae']:.4f}",
                f"{m['arrival_rmse']:.4f}",
                f"{m['arrival_r2']:.4f}",
                f"{m['departure_mae']:.4f}",
                f"{m['departure_rmse']:.4f}",
                f"{m['departure_r2']:.4f}",
                f"{m['mean_arrival_pred']:.2f}",
                f"{m['mean_arrival_target']:.2f}",
                f"{m['mean_departure_pred']:.2f}",
                f"{m['mean_departure_target']:.2f}",
            ])
        writer.writerow([])
        
        # Detailed per-horizon metrics for OVERALL (all flights)
        writer.writerow(["=" * 80])
        writer.writerow(["DETAILED METRICS - ALL FLIGHTS (OVERALL)"])
        writer.writerow(["=" * 80])
        writer.writerow(["Horizon", "Samples", "Arrival MAE", "Arrival RMSE", "Arrival R2", 
                        "Departure MAE", "Departure RMSE", "Departure R2",
                        "Mean Arr Pred", "Mean Arr Target", "Mean Dep Pred", "Mean Dep Target"])
        
        for horizon in horizons:
            m = per_horizon_overall_metrics[horizon]
            writer.writerow([
                f"{horizon}-step",
                m["num_samples"],
                f"{m['arrival_mae']:.4f}",
                f"{m['arrival_rmse']:.4f}",
                f"{m['arrival_r2']:.4f}",
                f"{m['departure_mae']:.4f}",
                f"{m['departure_rmse']:.4f}",
                f"{m['departure_r2']:.4f}",
                f"{m['mean_arrival_pred']:.2f}",
                f"{m['mean_arrival_target']:.2f}",
                f"{m['mean_departure_pred']:.2f}",
                f"{m['mean_departure_target']:.2f}",
            ])
        writer.writerow([])
    
    print(f"\n✓ Comprehensive results table saved to: {csv_path}")


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
    if len(horizons) != 1:
        raise ValueError(
            "threestagev4noise trains a SINGLE horizon. Pass exactly one of 3/6/12/24 via --horizons."
        )
    
    max_horizon = horizons[0]
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = delay_dim
    
    # Build test sequences
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    test_x, test_y_reg, test_y_cls = build_fn(
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
    
    # Auto-detect model path if set to 'auto'
    if args.model_path == "auto":
        args.model_path = find_latest_model()
        print(f"Auto-detected latest model: {args.model_path}")
    
    # Load model with DP metadata
    model, final_epsilon, final_delta, target_epsilon, epsilon_exceeded = _load_three_stage_model(
        args.model_path,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        device=device,
    )
    
    print(f"\nLoaded three-stage DP model from: {args.model_path}")
    print(f"Final ε: {final_epsilon:.3f}")
    print(f"Final δ: {final_delta:.2e}")
    if target_epsilon is not None:
        status = "OK" if final_epsilon <= target_epsilon else "EXCEEDED"
        print(f"Target ε: {target_epsilon:.3f} | Status: {status}")
    if epsilon_exceeded:
        print("⚠️  Checkpoint reports epsilon budget exceeded during training.")
    
    # Generate unique filenames with epsilon and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eps_str = f"eps{final_epsilon:.2f}".replace(".", "_")
    
    # Update filenames if using defaults
    if args.summary_csv == "three_stage_test_summary.csv":
        args.summary_csv = f"three_stage_test_summary_{eps_str}_{timestamp}.csv"
    if args.predictions_csv == "three_stage_test_predictions.csv":
        args.predictions_csv = f"three_stage_test_predictions_{eps_str}_{timestamp}.csv"
    if args.results_table_csv == "results_table.csv":
        args.results_table_csv = f"results_table_{eps_str}_{timestamp}.csv"
    
    print(f"\nOutput files will be:")
    print(f"  Summary: {args.summary_csv}")
    print(f"  Results Table: {args.results_table_csv}")
    print(f"  Predictions: {args.predictions_csv}")
    
    # Evaluate
    (
        cls_metrics,
        per_horizon_delayed_metrics,
        per_horizon_nondelayed_metrics,
        per_horizon_overall_metrics,
        preds_h,
        targets_h,
        test_probs,
        test_cls_targets,
    ) = _evaluate_three_stage_per_horizon(
        model,
        edge_indices,
        device,
        scaler,
        horizons,
        delay_dim,
        test_x,
        test_y_reg,
        test_y_cls,
        args.class_threshold,
        args.delay_threshold,
    )

    # Visualizations (saved as PNGs)
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating evaluation plots...")
        try:
            # preds_h/targets_h: (N, num_horizons, delay_dim). This script enforces single horizon.
            preds_2d = preds_h[:, 0, :] if preds_h.ndim == 3 else preds_h
            targets_2d = targets_h[:, 0, :] if targets_h.ndim == 3 else targets_h

            # Use arrival channel (0) for classification/regression combo plots (consistent with training script).
            if test_probs.ndim > 1 and test_probs.shape[1] > 1:
                y_pred_cls = (test_probs[:, 0] >= args.class_threshold).astype(int)
                y_true_cls = (test_cls_targets[:, 0] >= 0.5).astype(int)
            else:
                y_pred_cls = (test_probs.reshape(-1) >= args.class_threshold).astype(int)
                y_true_cls = (test_cls_targets.reshape(-1) >= 0.5).astype(int)

            y_reg_true_arrival = targets_2d[:, 0].reshape(-1) if targets_2d.ndim > 1 else targets_2d.reshape(-1)
            y_reg_pred_arrival = preds_2d[:, 0].reshape(-1) if preds_2d.ndim > 1 else preds_2d.reshape(-1)

            visualize_classification_results(
                y_true_cls,
                y_pred_cls,
                y_reg_true_arrival,
                y_reg_pred_arrival,
                threshold=args.delay_threshold,
                save_path=f"eval_classification_results_{eps_str}_{timestamp}.png",
            )

            visualize_regression_timeseries(
                targets_2d,
                preds_2d,
                title="Regression Over Time (True vs Predicted)",
                xlabel="Time (sample index)",
                ylabel="Delay (minutes)",
                save_path=f"eval_regression_timeseries_{eps_str}_{timestamp}.png",
            )

            visualize_regression_after_classification(
                y_reg_true_arrival,
                y_reg_pred_arrival,
                y_true_cls,
                y_pred_cls,
                threshold=args.delay_threshold,
                stage="Evaluation (Arrival)",
                save_path=f"eval_regression_results_{eps_str}_{timestamp}.png",
            )

            results_dict = {
                "y_true_cls": y_true_cls,
                "y_pred_cls": y_pred_cls,
                "y_true_reg": y_reg_true_arrival,
                # This evaluation script does not separate stage2/stage3 regressors; reuse final preds.
                "y_pred_stage2": y_reg_pred_arrival,
                "y_pred_stage3": y_reg_pred_arrival,
                "final_predictions": y_reg_pred_arrival,
            }
            visualize_three_stage_pipeline(
                results_dict,
                threshold=args.delay_threshold,
                save_path=f"eval_three_stage_pipeline_{eps_str}_{timestamp}.png",
            )

            print("  ✓ Saved evaluation plots:")
            print(f"    - eval_classification_results_{eps_str}_{timestamp}.png")
            print(f"    - eval_regression_timeseries_{eps_str}_{timestamp}.png")
            print(f"    - eval_regression_results_{eps_str}_{timestamp}.png")
            print(f"    - eval_three_stage_pipeline_{eps_str}_{timestamp}.png")
        except Exception as e:
            print(f"  ✗ Error generating evaluation plots: {e}")
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS - THREE-STAGE MODEL")
    print("="*80)
    
    print("\nCLASSIFICATION METRICS (macro over arrival/departure):")
    print(
        f"  Precision: {cls_metrics['precision']:.4f} | "
        f"Recall: {cls_metrics['recall']:.4f} | "
        f"F1: {cls_metrics['f1']:.4f} | Accuracy: {cls_metrics['accuracy']:.4f}"
    )
    print("  Per-channel:")
    print(
        f"    Arrival   - P: {cls_metrics.get('precision_arrival', 0):.4f} "
        f"R: {cls_metrics.get('recall_arrival', 0):.4f} F1: {cls_metrics.get('f1_arrival', 0):.4f} "
        f"Acc: {cls_metrics.get('accuracy_arrival', 0):.4f}"
    )
    print(
        f"    Departure - P: {cls_metrics.get('precision_departure', 0):.4f} "
        f"R: {cls_metrics.get('recall_departure', 0):.4f} F1: {cls_metrics.get('f1_departure', 0):.4f} "
        f"Acc: {cls_metrics.get('accuracy_departure', 0):.4f}"
    )
    
    print("\n" + "="*80)
    print("PER-HORIZON RESULTS FOR DELAYED FLIGHTS (>5 min)")
    print("="*80)
    for horizon in horizons:
        metrics = per_horizon_delayed_metrics[horizon]
        print(f"\n{horizon}-STEP AHEAD PREDICTIONS (Delayed Flights):")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Arrival Delay:")
        print(f"    MAE: {metrics['arrival_mae']:.4f} min | RMSE: {metrics['arrival_rmse']:.4f} min | R²: {metrics['arrival_r2']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_arrival_pred']:.2f} min | Mean Target: {metrics['mean_arrival_target']:.2f} min")
        print(f"  Departure Delay:")
        print(f"    MAE: {metrics['departure_mae']:.4f} min | RMSE: {metrics['departure_rmse']:.4f} min | R²: {metrics['departure_r2']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_departure_pred']:.2f} min | Mean Target: {metrics['mean_departure_target']:.2f} min")
    
    print("\n" + "="*80)
    print("PER-HORIZON RESULTS FOR NON-DELAYED FLIGHTS (<5 min)")
    print("="*80)
    print("NOTE: Stage 3 trains the regressor on non-delayed flights to produce accurate small delays")
    for horizon in horizons:
        metrics = per_horizon_nondelayed_metrics[horizon]
        print(f"\n{horizon}-STEP AHEAD PREDICTIONS (Non-Delayed Flights):")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Arrival Delay:")
        print(f"    MAE: {metrics['arrival_mae']:.4f} min | RMSE: {metrics['arrival_rmse']:.4f} min | R²: {metrics['arrival_r2']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_arrival_pred']:.2f} min | Mean Target: {metrics['mean_arrival_target']:.2f} min")
        print(f"  Departure Delay:")
        print(f"    MAE: {metrics['departure_mae']:.4f} min | RMSE: {metrics['departure_rmse']:.4f} min | R²: {metrics['departure_r2']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_departure_pred']:.2f} min | Mean Target: {metrics['mean_departure_target']:.2f} min")
    
    print("\n" + "="*80)
    print("PER-HORIZON RESULTS FOR ALL FLIGHTS (OVERALL)")
    print("="*80)
    for horizon in horizons:
        metrics = per_horizon_overall_metrics[horizon]
        print(f"\n{horizon}-STEP AHEAD PREDICTIONS (All Flights):")
        print(f"  Samples: {metrics['num_samples']}")
        print(f"  Arrival Delay:")
        print(f"    MAE: {metrics['arrival_mae']:.4f} min | RMSE: {metrics['arrival_rmse']:.4f} min | R²: {metrics['arrival_r2']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_arrival_pred']:.2f} min | Mean Target: {metrics['mean_arrival_target']:.2f} min")
        print(f"  Departure Delay:")
        print(f"    MAE: {metrics['departure_mae']:.4f} min | RMSE: {metrics['departure_rmse']:.4f} min | R²: {metrics['departure_r2']:.4f}")
        print(f"    Mean Predicted: {metrics['mean_departure_pred']:.2f} min | Mean Target: {metrics['mean_departure_target']:.2f} min")
    
    # Save summary CSV
    if args.summary_csv:
        with open(args.summary_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            
            # DP parameters
            writer.writerow(["final_epsilon", final_epsilon])
            writer.writerow(["final_delta", final_delta])
            writer.writerow(["target_epsilon", target_epsilon if target_epsilon is not None else "n/a"])
            writer.writerow(["epsilon_exceeded", epsilon_exceeded])
            writer.writerow(["model_path", args.model_path])
            writer.writerow(["model_type", "three_stage"])
            writer.writerow(["data_source", args.data_source])
            writer.writerow(["seq_len", args.seq_len])
            writer.writerow(["delay_threshold", args.delay_threshold])
            writer.writerow(["class_threshold", args.class_threshold])
            
            # Classification metrics
            for key, value in cls_metrics.items():
                writer.writerow([f"classification_{key}", value])
            
            # Per-channel classification metrics (if available)
            for channel in ['arrival', 'departure']:
                for metric in ['precision', 'recall', 'f1', 'accuracy']:
                    key = f"{metric}_{channel}"
                    if key in cls_metrics:
                        writer.writerow([f"classification_{key}", cls_metrics[key]])
            
            # Per-horizon metrics for DELAYED flights
            for horizon, metrics in per_horizon_delayed_metrics.items():
                writer.writerow([f"delayed_h{horizon}_num_samples", metrics["num_samples"]])
                writer.writerow([f"delayed_h{horizon}_arrival_mae", metrics["arrival_mae"]])
                writer.writerow([f"delayed_h{horizon}_arrival_rmse", metrics["arrival_rmse"]])
                writer.writerow([f"delayed_h{horizon}_arrival_r2", metrics["arrival_r2"]])
                writer.writerow([f"delayed_h{horizon}_departure_mae", metrics["departure_mae"]])
                writer.writerow([f"delayed_h{horizon}_departure_rmse", metrics["departure_rmse"]])
                writer.writerow([f"delayed_h{horizon}_departure_r2", metrics["departure_r2"]])
                writer.writerow([f"delayed_h{horizon}_mean_arrival_pred", metrics["mean_arrival_pred"]])
                writer.writerow([f"delayed_h{horizon}_mean_arrival_target", metrics["mean_arrival_target"]])
                writer.writerow([f"delayed_h{horizon}_mean_departure_pred", metrics["mean_departure_pred"]])
                writer.writerow([f"delayed_h{horizon}_mean_departure_target", metrics["mean_departure_target"]])
            
            # Per-horizon metrics for NON-DELAYED flights
            for horizon, metrics in per_horizon_nondelayed_metrics.items():
                writer.writerow([f"nondelayed_h{horizon}_num_samples", metrics["num_samples"]])
                writer.writerow([f"nondelayed_h{horizon}_arrival_mae", metrics["arrival_mae"]])
                writer.writerow([f"nondelayed_h{horizon}_arrival_rmse", metrics["arrival_rmse"]])
                writer.writerow([f"nondelayed_h{horizon}_arrival_r2", metrics["arrival_r2"]])
                writer.writerow([f"nondelayed_h{horizon}_departure_mae", metrics["departure_mae"]])
                writer.writerow([f"nondelayed_h{horizon}_departure_rmse", metrics["departure_rmse"]])
                writer.writerow([f"nondelayed_h{horizon}_departure_r2", metrics["departure_r2"]])
                writer.writerow([f"nondelayed_h{horizon}_mean_arrival_pred", metrics["mean_arrival_pred"]])
                writer.writerow([f"nondelayed_h{horizon}_mean_arrival_target", metrics["mean_arrival_target"]])
                writer.writerow([f"nondelayed_h{horizon}_mean_departure_pred", metrics["mean_departure_pred"]])
                writer.writerow([f"nondelayed_h{horizon}_mean_departure_target", metrics["mean_departure_target"]])
            
            # Per-horizon metrics for OVERALL
            for horizon, metrics in per_horizon_overall_metrics.items():
                writer.writerow([f"overall_h{horizon}_num_samples", metrics["num_samples"]])
                writer.writerow([f"overall_h{horizon}_arrival_mae", metrics["arrival_mae"]])
                writer.writerow([f"overall_h{horizon}_arrival_rmse", metrics["arrival_rmse"]])
                writer.writerow([f"overall_h{horizon}_arrival_r2", metrics["arrival_r2"]])
                writer.writerow([f"overall_h{horizon}_departure_mae", metrics["departure_mae"]])
                writer.writerow([f"overall_h{horizon}_departure_rmse", metrics["departure_rmse"]])
                writer.writerow([f"overall_h{horizon}_departure_r2", metrics["departure_r2"]])
                writer.writerow([f"overall_h{horizon}_mean_arrival_pred", metrics["mean_arrival_pred"]])
                writer.writerow([f"overall_h{horizon}_mean_arrival_target", metrics["mean_arrival_target"]])
                writer.writerow([f"overall_h{horizon}_mean_departure_pred", metrics["mean_departure_pred"]])
                writer.writerow([f"overall_h{horizon}_mean_departure_target", metrics["mean_departure_target"]])
        
        print(f"\n✓ Summary saved to: {args.summary_csv}")
    
    # Save results table in organized format
    if args.results_table_csv:
        save_results_table(
            args.results_table_csv,
            final_epsilon,
            final_delta,
            target_epsilon,
            epsilon_exceeded,
            per_horizon_delayed_metrics,
            per_horizon_nondelayed_metrics,
            per_horizon_overall_metrics,
            cls_metrics,
            horizons,
            args.model_path,
            args,
        )
    
    # Save predictions CSV
    if args.predictions_csv:
        with open(args.predictions_csv, "w", newline="") as f:
            fieldnames = [
                "sample_index",
                "horizon",
                "is_delayed_ground_truth",
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
            
            # Determine ground truth delayed status
            targets_reshaped = targets_h.reshape(num_samples, len(horizons), delay_dim)
            is_delayed = np.any(targets_reshaped >= args.delay_threshold, axis=(1, 2))
            
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
                            "is_delayed_ground_truth": int(is_delayed[idx]),
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
