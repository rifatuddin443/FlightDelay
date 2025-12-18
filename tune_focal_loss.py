"""Hyperparameter tuning for Focal Loss alpha and gamma parameters.

Tests different combinations of alpha and gamma to find optimal values for
flight delay classification task with class imbalance.
"""

from __future__ import annotations
import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (
    ResidualKANPredictor,
    build_sequences,
    classification_metrics,
    load_flight_data,
    set_seed,
    EarlyStopping,
)
from classifykat_balanced import build_sequences_node_level


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


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


def train_classifier_with_focal_loss(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
    alpha: float,
    gamma: float,
    patience: int = 5,
) -> Tuple[Dict[str, float], float]:
    """Train classifier with specific focal loss parameters."""
    
    # Freeze regressor
    for param in model.regressor.parameters():
        param.requires_grad = False
    
    trainable_params = list(model.encoder.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    
    # Use Focal Loss with specified parameters
    cls_loss_fn = FocalLoss(alpha=alpha, gamma=gamma)
    
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    
    best_f1 = 0.0
    early_stopping = EarlyStopping(patience=patience, mode="max")
    training_time = 0.0
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        epoch_losses = []
        
        num_samples = train_x.shape[0]
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            
            batch_x = train_x[batch_indices].to(device)
            batch_y_cls = train_y_cls[batch_indices].to(device)
            
            optimizer.zero_grad()
            
            batch_logits = []
            batch_targets = []
            
            for i in range(batch_x.shape[0]):
                data = Data(
                    x=batch_x[i],
                    edge_index_adj=edge_index_adj,
                    edge_index_od=edge_index_od,
                    edge_index_od_t=edge_index_od_t,
                )
                
                node_encoded = model.encoder(data)
                node_logits = model.classifier(node_encoded)
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(batch_y_cls[i])
                
                batch_logits.append(graph_logit)
                batch_targets.append(graph_target)
            
            batch_logits_tensor = torch.cat(batch_logits, dim=0)
            batch_targets_tensor = torch.cat(batch_targets, dim=0)
            
            loss = cls_loss_fn(batch_logits_tensor, batch_targets_tensor)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_index_adj,
                    edge_index_od=edge_index_od,
                    edge_index_od_t=edge_index_od_t,
                )
                
                node_encoded = model.encoder(data)
                node_logits = model.classifier(node_encoded)
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(val_y_cls[i])
                
                val_probs.append(torch.sigmoid(graph_logit).cpu().numpy())
                val_targets.append(graph_target.cpu().numpy())
        
        val_probs_np = np.concatenate(val_probs, axis=0)
        val_targets_np = np.concatenate(val_targets, axis=0)
        val_metrics = classification_metrics(
            val_probs_np.reshape(-1, 1),
            val_targets_np.reshape(-1, 1),
        )
        
        epoch_time = time.time() - epoch_start
        training_time += epoch_time
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
        
        if early_stopping(val_metrics['f1'], epoch):
            break
    
    # Final validation metrics
    model.eval()
    val_probs, val_targets = [], []
    with torch.no_grad():
        for i in range(len(val_x)):
            data = Data(
                x=val_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            
            node_encoded = model.encoder(data)
            node_logits = model.classifier(node_encoded)
            graph_logit = aggregate_node_to_graph(node_logits)
            graph_target = ensure_graph_level_target(val_y_cls[i])
            
            val_probs.append(torch.sigmoid(graph_logit).cpu().numpy())
            val_targets.append(graph_target.cpu().numpy())
    
    val_probs_np = np.concatenate(val_probs, axis=0)
    val_targets_np = np.concatenate(val_targets, axis=0)
    final_metrics = classification_metrics(
        val_probs_np.reshape(-1, 1),
        val_targets_np.reshape(-1, 1),
    )
    
    # Unfreeze regressor
    for param in model.regressor.parameters():
        param.requires_grad = True
    
    return final_metrics, training_time


def grid_search_focal_loss(
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    in_channels: int,
    out_channels: int,
    hidden_channels: int,
    epochs: int,
    lr: float,
    batch_size: int,
    alpha_values: List[float],
    gamma_values: List[float],
) -> List[Dict]:
    """Perform grid search over focal loss hyperparameters."""
    
    results = []
    total_combinations = len(alpha_values) * len(gamma_values)
    current = 0
    
    print("\n" + "="*80)
    print(f"FOCAL LOSS HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Testing {len(alpha_values)} alpha values × {len(gamma_values)} gamma values = {total_combinations} combinations")
    print(f"Alpha values: {alpha_values}")
    print(f"Gamma values: {gamma_values}")
    print("="*80 + "\n")
    
    for alpha in alpha_values:
        for gamma in gamma_values:
            current += 1
            print(f"\n[{current}/{total_combinations}] Testing alpha={alpha}, gamma={gamma}")
            print("-" * 40)
            
            # Create fresh model for each combination
            model = ResidualKANPredictor(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=hidden_channels,
            ).to(device)
            
            start_time = time.time()
            metrics, training_time = train_classifier_with_focal_loss(
                model=model,
                train_x=train_x,
                train_y_cls=train_y_cls,
                val_x=val_x,
                val_y_cls=val_y_cls,
                edge_indices=edge_indices,
                device=device,
                epochs=epochs,
                lr=lr,
                batch_size=batch_size,
                alpha=alpha,
                gamma=gamma,
            )
            total_time = time.time() - start_time
            
            result = {
                'alpha': alpha,
                'gamma': gamma,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'accuracy': metrics['accuracy'],
                'training_time': training_time,
                'total_time': total_time,
            }
            results.append(result)
            
            print(f"Results: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
                  f"F1={metrics['f1']:.4f}, Accuracy={metrics['accuracy']:.4f}")
            print(f"Time: {training_time:.2f}s")
    
    return results


def save_results(results: List[Dict], output_file: str) -> None:
    """Save grid search results to CSV."""
    
    # Sort by F1 score (descending)
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'rank', 'alpha', 'gamma', 'f1', 'precision', 'recall', 'accuracy', 
            'training_time', 'total_time'
        ])
        writer.writeheader()
        
        for rank, result in enumerate(results_sorted, 1):
            writer.writerow({
                'rank': rank,
                'alpha': result['alpha'],
                'gamma': result['gamma'],
                'f1': f"{result['f1']:.4f}",
                'precision': f"{result['precision']:.4f}",
                'recall': f"{result['recall']:.4f}",
                'accuracy': f"{result['accuracy']:.4f}",
                'training_time': f"{result['training_time']:.2f}",
                'total_time': f"{result['total_time']:.2f}",
            })
    
    print(f"\n✓ Results saved to: {output_file}")


def print_summary(results: List[Dict]) -> None:
    """Print summary of best configurations."""
    
    # Sort by F1 score
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    
    print("\n" + "="*80)
    print("TOP 5 CONFIGURATIONS BY F1 SCORE")
    print("="*80)
    
    for i, result in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. Alpha={result['alpha']:.3f}, Gamma={result['gamma']:.2f}")
        print(f"   F1: {result['f1']:.4f} | Precision: {result['precision']:.4f} | "
              f"Recall: {result['recall']:.4f} | Accuracy: {result['accuracy']:.4f}")
        print(f"   Training Time: {result['training_time']:.2f}s")
    
    # Best for precision
    best_precision = max(results, key=lambda x: x['precision'])
    print(f"\nBest Precision: Alpha={best_precision['alpha']:.3f}, Gamma={best_precision['gamma']:.2f} "
          f"→ {best_precision['precision']:.4f}")
    
    # Best for recall
    best_recall = max(results, key=lambda x: x['recall'])
    print(f"Best Recall: Alpha={best_recall['alpha']:.3f}, Gamma={best_recall['gamma']:.2f} "
          f"→ {best_recall['recall']:.4f}")
    
    # Best for F1
    best_f1 = results_sorted[0]
    print(f"Best F1: Alpha={best_f1['alpha']:.3f}, Gamma={best_f1['gamma']:.2f} "
          f"→ {best_f1['f1']:.4f}")
    
    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune Focal Loss hyperparameters for flight delay classification")
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 6, 12])
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--use_node_level', action='store_true', default=True)
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=15, help='Max epochs per configuration')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--patience', type=int, default=5)
    
    # Focal loss hyperparameter ranges
    parser.add_argument('--alpha_values', type=float, nargs='+', 
                       default=[0.1, 0.25, 0.5, 0.75, 0.9],
                       help='Alpha values to test (weight for positive class)')
    parser.add_argument('--gamma_values', type=float, nargs='+',
                       default=[0.5, 1.0, 2.0, 3.0, 5.0],
                       help='Gamma values to test (focusing parameter)')
    
    parser.add_argument('--output_csv', type=str, default='focal_loss_tuning_results.csv')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    
    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
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
    max_horizon = max(horizons)
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = len(horizons) * delay_dim
    
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    
    print(f"\n[DATA] Train: {len(train_x)} samples | Val: {len(val_x)} samples")
    print(f"[DATA] Class balance (delayed): {train_y_cls.mean().item():.2%}")
    
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    # Perform grid search
    start_time = time.time()
    results = grid_search_focal_loss(
        train_x=train_x,
        train_y_cls=train_y_cls,
        val_x=val_x,
        val_y_cls=val_y_cls,
        edge_indices=edge_indices,
        device=device,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=args.hidden_channels,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        alpha_values=args.alpha_values,
        gamma_values=args.gamma_values,
    )
    total_time = time.time() - start_time
    
    print(f"\n\nTotal grid search time: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    # Add timestamp to output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_csv == 'focal_loss_tuning_results.csv':
        args.output_csv = f'focal_loss_tuning_results_{timestamp}.csv'
    
    # Save and print results
    save_results(results, args.output_csv)
    print_summary(results)


if __name__ == '__main__':
    main()
