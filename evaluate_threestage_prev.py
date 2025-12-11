"""Test script for three-stage DP models with horizon-wise evaluation.

Tests a model trained with train_stage3 which continues training the SAME regressor
on non-delayed flights after training on delayed flights in stage 2.
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
    set_seed,
)
from classifykat_balanced import build_sequences_node_level  # noqa: E402


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
            graph_logit = aggregate_node_to_graph(node_logits)
            graph_reg = aggregate_node_to_graph(node_reg)
            
            # Aggregate targets to graph-level
            graph_cls_target = ensure_graph_level_target(test_y_cls[i])
            graph_reg_target = ensure_graph_level_target(test_y_reg[i])
            
            logits_list.append(torch.sigmoid(graph_logit).cpu().numpy())
            reg_list.append(graph_reg.cpu().numpy())
            targets_cls_list.append(graph_cls_target.cpu().numpy())
            targets_reg_list.append(graph_reg_target.cpu().numpy())
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")
    
    test_probs = np.concatenate(logits_list, axis=0)  # (num_samples, 1)
    test_reg_preds = np.concatenate(reg_list, axis=0)  # (num_samples, out_channels)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)  # (num_samples, 1)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)  # (num_samples, out_channels)
    
    # Classification metrics
    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_cls_targets.reshape(-1, 1),
    )
    
    # Apply classifier gating (as done in training)
    test_mask = (test_probs >= class_threshold)
    gated_preds = test_reg_preds * test_mask
    
    # Denormalize
    num_forecast_steps = len(horizons)
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(gated_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
    else:
        preds_denorm = gated_preds
        targets_denorm = test_reg_targets
    
    # Reshape for per-horizon analysis
    preds_h = preds_denorm.reshape(-1, num_forecast_steps, delay_dim)
    targets_h = targets_denorm.reshape(-1, num_forecast_steps, delay_dim)
    
    # Get ground truth masks based on ACTUAL delays (not predictions)
    true_delayed_mask = np.any(np.abs(targets_denorm.reshape(-1, num_forecast_steps, delay_dim)) >= delay_threshold, axis=(1, 2))
    true_nondelayed_mask = ~true_delayed_mask
    
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
    )


def _load_three_stage_model(
    model_path: str,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    device: torch.device,
) -> Tuple[SequentialTwoStagePredictor, float, float]:
    """Load three-stage model and extract DP metadata."""
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
    final_epsilon = checkpoint.get('final_epsilon', float('inf'))
    final_delta = checkpoint.get('final_delta', 0.0)
    
    return model, final_epsilon, final_delta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test three-stage DP models with separate horizon-wise evaluation.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="kan_gat_dp_three_stage_eps5_20_20251209_190223.pth",
        help="Path to the three-stage trained checkpoint",
    )
    parser.add_argument("--data_source", type=str, default="udata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--use_node_level", action="store_true", default=True, help="Use node-level labels")
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--summary_csv", type=str, default="three_stage_test_summary.csv")
    parser.add_argument("--predictions_csv", type=str, default="three_stage_test_predictions.csv")
    parser.add_argument("--results_table_csv", type=str, default="results_table.csv")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save_results_table(
    csv_path: str,
    epsilon: float,
    delta: float,
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
        writer.writerow(["Final Delta", f"{delta:.2e}"])
        writer.writerow([])
        
        # Classification metrics
        writer.writerow(["=" * 80])
        writer.writerow(["CLASSIFICATION METRICS"])
        writer.writerow(["=" * 80])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Precision", f"{cls_metrics['precision']:.4f}"])
        writer.writerow(["Recall", f"{cls_metrics['recall']:.4f}"])
        writer.writerow(["F1 Score", f"{cls_metrics['f1']:.4f}"])
        writer.writerow(["Accuracy", f"{cls_metrics['accuracy']:.4f}"])
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
    if not horizons:
        raise ValueError("Please provide at least one positive horizon.")
    
    max_horizon = max(horizons)
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = len(horizons) * delay_dim
    
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
    
    # Load model with DP metadata
    model, final_epsilon, final_delta = _load_three_stage_model(
        args.model_path,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        device=device,
    )
    
    print(f"\nLoaded three-stage DP model from: {args.model_path}")
    print(f"Final ε: {final_epsilon:.3f}")
    print(f"Final δ: {final_delta:.2e}")
    
    # Evaluate
    (
        cls_metrics,
        per_horizon_delayed_metrics,
        per_horizon_nondelayed_metrics,
        per_horizon_overall_metrics,
        preds_h,
        targets_h,
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
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS - THREE-STAGE MODEL")
    print("="*80)
    
    print("\nCLASSIFICATION METRICS:")
    print(
        f"  Precision: {cls_metrics['precision']:.4f} | "
        f"Recall: {cls_metrics['recall']:.4f} | "
        f"F1: {cls_metrics['f1']:.4f} | Accuracy: {cls_metrics['accuracy']:.4f}"
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
            writer.writerow(["model_path", args.model_path])
            writer.writerow(["model_type", "three_stage"])
            writer.writerow(["data_source", args.data_source])
            writer.writerow(["seq_len", args.seq_len])
            writer.writerow(["delay_threshold", args.delay_threshold])
            writer.writerow(["class_threshold", args.class_threshold])
            
            # Classification metrics
            for key, value in cls_metrics.items():
                writer.writerow([f"classification_{key}", value])
            
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