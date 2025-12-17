"""Benchmark different architectures for classification accuracy.

Tests:
1. Baseline (current with Focal Loss)
2. Attention Fusion
3. Residual Blocks
4. Graph Transformer
5. Hybrid (Attention + Residual)

Run with: python benchmark_architectures.py --quick_test (5 epochs each)
         python benchmark_architectures.py --full_test (20 epochs each)
"""

import argparse
import time
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (
    load_flight_data,
    build_sequences,
    classification_metrics,
    set_seed,
    StandardScaler,
)
from classifykat_balanced import build_sequences_node_level

# Add efficient-kan
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
from kan import KAN


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification."""
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


# ==================== ARCHITECTURE 1: BASELINE WITH IMPROVEMENTS ====================
class BaselineEncoder(nn.Module):
    """Current architecture with dropout."""
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 2):
        super().__init__()
        self.alpha_adj = nn.Parameter(torch.tensor(1.0))
        self.alpha_od = nn.Parameter(torch.tensor(1.0))
        self.alpha_od_t = nn.Parameter(torch.tensor(1.0))

        self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)

        fusion_input_dim = hidden_channels * 3 + 3
        self.fusion_kan = KAN(
            layers_hidden=[fusion_input_dim, hidden_channels, hidden_channels],
            grid_size=3,
            spline_order=2,
        )
        self.dropout = nn.Dropout(0.3)

    def forward(self, data: Data) -> torch.Tensor:
        weights = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
        w_adj, w_od, w_od_t = weights

        x_adj = self.gat_adj(data.x, data.edge_index_adj)
        x_od = self.gat_od(data.x, data.edge_index_od)
        x_od_t = self.gat_od_t(data.x, data.edge_index_od_t)

        num_nodes = x_adj.size(0)
        scalars = torch.cat([
            w_adj.expand(num_nodes, 1),
            w_od.expand(num_nodes, 1),
            w_od_t.expand(num_nodes, 1),
        ], dim=1)

        x_concat = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
        fused = F.relu(self.fusion_kan(x_concat))
        fused = self.dropout(fused)
        return fused


# ==================== ARCHITECTURE 2: ATTENTION FUSION ====================
class AttentionFusionEncoder(nn.Module):
    """Multi-head attention fusion of graph views."""
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4):
        super().__init__()
        self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
        self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
        
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)
        
        self.fusion_kan = KAN(
            layers_hidden=[hidden_channels, hidden_channels * 2, hidden_channels],
            grid_size=4,
            spline_order=2,
        )
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data: Data) -> torch.Tensor:
        x_adj = F.elu(self.gat_adj(data.x, data.edge_index_adj))
        x_od = F.elu(self.gat_od(data.x, data.edge_index_od))
        x_od_t = F.elu(self.gat_od_t(data.x, data.edge_index_od_t))
        
        # Stack for attention [num_nodes, 3, hidden_dim]
        x_stack = torch.stack([x_adj, x_od, x_od_t], dim=1)
        
        # Self-attention across graph views
        attn_out, _ = self.multihead_attn(x_stack, x_stack, x_stack)
        attn_out = self.ln1(attn_out.mean(dim=1) + x_stack.mean(dim=1))
        
        # Refine with KAN
        fused = self.fusion_kan(attn_out)
        fused = self.ln2(fused)
        fused = self.dropout(fused)
        
        return fused


# ==================== ARCHITECTURE 3: RESIDUAL BLOCKS ====================
class ResidualEncoder(nn.Module):
    """Deep residual architecture."""
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 2):
        super().__init__()
        self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        
        # Initial fusion
        self.fusion1 = nn.Linear(hidden_channels * 3, hidden_channels)
        
        # Residual blocks
        self.res_block1 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        
        self.res_block2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data: Data) -> torch.Tensor:
        x_adj = self.gat_adj(data.x, data.edge_index_adj)
        x_od = self.gat_od(data.x, data.edge_index_od)
        x_od_t = self.gat_od_t(data.x, data.edge_index_od_t)
        
        x_concat = torch.cat([x_adj, x_od, x_od_t], dim=1)
        x = F.elu(self.fusion1(x_concat))
        
        # Residual block 1
        residual = x
        x = self.res_block1(x) + residual
        x = F.elu(x)
        
        # Residual block 2
        residual = x
        x = self.res_block2(x) + residual
        x = F.elu(x)
        
        x = self.dropout(x)
        return x


# ==================== ARCHITECTURE 4: LIGHTWEIGHT TRANSFORMER ====================
class TransformerEncoder(nn.Module):
    """Lightweight transformer for graph nodes."""
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 2):
        super().__init__()
        # Initial GAT processing
        self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        
        # Fusion
        self.fusion = nn.Linear(hidden_channels * 3, hidden_channels)
        
        # Transformer layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=4,
            dim_feedforward=hidden_channels * 2,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data: Data) -> torch.Tensor:
        x_adj = self.gat_adj(data.x, data.edge_index_adj)
        x_od = self.gat_od(data.x, data.edge_index_od)
        x_od_t = self.gat_od_t(data.x, data.edge_index_od_t)
        
        x_concat = torch.cat([x_adj, x_od, x_od_t], dim=1)
        x = F.gelu(self.fusion(x_concat))
        
        # Add batch dimension for transformer
        x = x.unsqueeze(0)  # [1, num_nodes, hidden]
        x = self.transformer(x)
        x = x.squeeze(0)  # [num_nodes, hidden]
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


# ==================== ARCHITECTURE 5: HYBRID ====================
class HybridEncoder(nn.Module):
    """Attention + Residual hybrid."""
    def __init__(self, in_channels: int, hidden_channels: int = 64, heads: int = 4):
        super().__init__()
        self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
        self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.2)
        
        # Attention fusion
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_channels,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Residual refinement
        self.res_block = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, data: Data) -> torch.Tensor:
        x_adj = F.elu(self.gat_adj(data.x, data.edge_index_adj))
        x_od = F.elu(self.gat_od(data.x, data.edge_index_od))
        x_od_t = F.elu(self.gat_od_t(data.x, data.edge_index_od_t))
        
        # Attention fusion
        x_stack = torch.stack([x_adj, x_od, x_od_t], dim=1)
        attn_out, _ = self.multihead_attn(x_stack, x_stack, x_stack)
        x = attn_out.mean(dim=1)
        
        # Residual refinement
        residual = x
        x = self.res_block(x) + residual
        x = F.elu(x)
        
        x = self.layer_norm(x)
        x = self.dropout(x)
        return x


# ==================== UNIVERSAL CLASSIFIER ====================
class UniversalClassifier(nn.Module):
    """Classifier head that works with any encoder."""
    def __init__(self, encoder, hidden_channels: int = 64):
        super().__init__()
        self.encoder = encoder
        
        self.classifier = KAN(
            layers_hidden=[hidden_channels, hidden_channels // 2, 1],
            grid_size=3,
            spline_order=2,
        )
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, data: Data) -> torch.Tensor:
        hidden = self.encoder(data)
        hidden_dropped = self.dropout(hidden)
        logits = self.classifier(hidden_dropped)
        return logits


# ==================== TRAINING AND EVALUATION ====================
def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    """Aggregate node-level to graph-level."""
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    """Convert targets to graph-level."""
    if target.dim() == 0:
        return target.unsqueeze(0)
    elif target.dim() == 1:
        return target.mean(dim=0, keepdim=True)
    else:
        return target.mean(dim=0, keepdim=True)


def train_and_evaluate(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict[str, float]:
    """Train model and return validation metrics."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    
    best_f1 = 0.0
    best_metrics = {}
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for i in range(len(train_x)):
            data = Data(
                x=train_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )
            
            optimizer.zero_grad()
            node_logits = model(data)  # [num_nodes, 1]
            node_targets = train_y[i].to(device)  # [num_nodes, 1]
            
            loss = loss_fn(node_logits, node_targets)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Validation - NODE LEVEL
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                node_logits = model(data)  # [num_nodes, 1]
                node_targets = val_y[i]  # [num_nodes, 1]
                
                val_probs.append(torch.sigmoid(node_logits).cpu().numpy())
                val_targets.append(node_targets.numpy())
        
        # Flatten all node predictions across all samples
        val_probs_np = np.concatenate(val_probs, axis=0)  # [total_nodes, 1]
        val_targets_np = np.concatenate(val_targets, axis=0)  # [total_nodes, 1]
        
        # Debug: Check predictions
        if epoch == 0:
            print(f"  Debug: Val probs range [{val_probs_np.min():.3f}, {val_probs_np.max():.3f}], "
                  f"Targets mean: {val_targets_np.mean():.3f}, Unique preds: {len(np.unique(val_probs_np.round(3)))}")
        
        val_metrics = classification_metrics(
            val_probs_np,
            val_targets_np,
        )
        
        # Handle potential NaN values
        current_f1 = val_metrics.get('f1', 0.0)
        if current_f1 is None or np.isnan(current_f1):
            current_f1 = 0.0
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_metrics = {k: (v if v is not None and not np.isnan(v) else 0.0) 
                           for k, v in val_metrics.items()}
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val F1: {val_metrics.get('f1', 0.0):.4f}")
    
    # Ensure best_metrics has all keys
    if not best_metrics:
        best_metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
    
    return best_metrics


def benchmark_all_architectures(args):
    """Run benchmark on all architectures."""
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading data...")
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=24,
        data_source=args.data_source,
    )
    
    # Build sequences
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, args.horizons[0], args.delay_threshold, args.horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, args.horizons[0], args.delay_threshold, args.horizons
    )
    
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    feature_dim = train_inputs.shape[2]
    in_channels = args.seq_len * feature_dim
    hidden_channels = args.hidden_channels
    
    print(f"Input config: in_channels={in_channels}, hidden_channels={hidden_channels}")
    print(f"Train sample shape: {train_x[0].shape}")
    print(f"Edge index shapes: adj={edge_indices[0].shape}, od={edge_indices[1].shape}, od_t={edge_indices[2].shape}")
    print(f"Num nodes: {num_nodes}\n")
    
    print(f"Data loaded: {len(train_x)} train, {len(val_x)} val samples")
    print(f"Class balance: {train_y_cls.mean().item():.2%} delayed\n")
    
    # Define architectures to test
    architectures = {
        '1_Baseline': lambda: BaselineEncoder(in_channels, hidden_channels),
        '2_Attention': lambda: AttentionFusionEncoder(in_channels, hidden_channels),
        '3_Residual': lambda: ResidualEncoder(in_channels, hidden_channels),
        '4_Transformer': lambda: TransformerEncoder(in_channels, hidden_channels),
        '5_Hybrid': lambda: HybridEncoder(in_channels, hidden_channels),
    }
    
    results = {}
    
    print("="*80)
    print("BENCHMARKING ARCHITECTURES")
    print("="*80)
    
    for name, encoder_fn in architectures.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        encoder = encoder_fn()
        model = UniversalClassifier(encoder, hidden_channels)
        
        metrics = train_and_evaluate(
            model, train_x, train_y_cls, val_x, val_y_cls,
            edge_indices, device, args.epochs, args.lr
        )
        
        elapsed = time.time() - start_time
        
        # Ensure metrics is a dictionary with all required keys
        if not isinstance(metrics, dict):
            print(f"  WARNING: metrics is not a dict, got {type(metrics)}")
            metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
        
        results[name] = {
            'f1': metrics.get('f1', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'time_seconds': elapsed,
            'time_per_epoch': elapsed / args.epochs,
        }
        
        print(f"\n[DONE] {name} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  F1: {metrics.get('f1', 0.0):.4f} | Precision: {metrics.get('precision', 0.0):.4f} | "
              f"Recall: {metrics.get('recall', 0.0):.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print(f"\n{'Architecture':<20} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'Time (s)':<12}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for name, metrics in sorted_results:
        print(f"{name:<20} {metrics['f1']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['time_seconds']:<12.1f}")
    
    best_arch = sorted_results[0][0]
    best_f1 = sorted_results[0][1]['f1']
    
    print("\n" + "="*80)
    print(f"WINNER: {best_arch} with F1 = {best_f1:.4f}")
    print("="*80)
    
    # Save results
    import json
    with open(f'benchmark_results_{args.epochs}epochs.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to benchmark_results_{args.epochs}epochs.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark KAN-GAT architectures")
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 6, 12])
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--use_node_level', action='store_true', default=True)
    parser.add_argument('--weather_file', type=str, default='weather2016_2021.npy')
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10, help='Epochs per architecture')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--quick_test', action='store_true', help='Run 5 epochs only')
    parser.add_argument('--full_test', action='store_true', help='Run 20 epochs')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.quick_test:
        args.epochs = 5
        print("Quick test mode: 5 epochs per architecture\n")
    elif args.full_test:
        args.epochs = 20
        print("Full test mode: 20 epochs per architecture\n")
    
    benchmark_all_architectures(args)
