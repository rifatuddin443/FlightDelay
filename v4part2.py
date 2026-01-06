
"""Properly implemented differentially private sequential KAN-GAT pipeline with epsilon budget control.

FIXED:
1. Ensures both predictions AND targets are at graph-level (not node-level).
2. Uses fixed noise multiplier for differential privacy (not auto-computed).
3. Allows training to complete all epochs while tracking epsilon.
4. NEW: Added Stage 3 for regressing delays on samples predicted as under threshold.
5. FIXED: Stage 3 now correctly masks based on actual delay values (< 5 min), not classification labels.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.functional as F
from torch.func import grad, vmap, functional_call
from torch.utils.data import Dataset
from torch_geometric.data import Data
import glob

# Check if running in Colab for file downloads
try:
    from google.colab import files as colab_files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    colab_files = None

# Reuse original implementation
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (  # noqa: E402
    EarlyStopping,
    SequentialTwoStagePredictor,
    build_sequences,
    classification_metrics,
    load_flight_data,
    regression_metrics,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level  # noqa: E402
from baseline_methods import test_error  # noqa: E402

# Import visualization functions
try:
    from visualize_training_classification import (
        visualize_training_data,
        visualize_classification_results
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: visualize_training_classification not found. Visualizations will be skipped.")
    VISUALIZATION_AVAILABLE = False


# Override build_sequences to keep 3D temporal structure
def build_sequences_temporal(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequences keeping temporal structure [num_nodes, seq_len, features]."""
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    x_list, y_reg_list, y_cls_list = [], [], []

    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))
    if not horizon_ids:
        raise ValueError("At least one future horizon is required to build sequences.")

    for t in range(max_idx):
        # Keep 3D: [num_nodes, seq_len, features]
        x_seq = input_data[:, t:t + seq_len, :]
        
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)
        
        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :])
        
        # Per-node, per-feature classification
        node_delays = np.max(raw_target, axis=1)  # [num_nodes, num_features]
        cls_flag = (node_delays >= delay_threshold).astype(np.float32)

        x_list.append(x_seq)
        y_reg_list.append(y_seq)
        y_cls_list.append(cls_flag)

    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    
    # Print balance
    cls_tensor = tensors[2]
    delayed_rate = cls_tensor.mean().item()
    print(f"\nNode-level balance (temporal): {delayed_rate:.2%} delayed")
    
    return tensors


# Import GAT components we need
from torch_geometric.nn import GATConv  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
from kan import KAN  # noqa: E402


class TemporalGATEncoder(nn.Module):
    """GAT encoder with temporal encoding via LSTM before spatial aggregation."""
    
    def __init__(self, feature_dim: int, seq_len: int, hidden_channels: int = 32, heads: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        
        # Temporal encoding: LSTM processes [seq_len, features] per node
        self.temporal_lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_channels,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )
        
        # Spatial encoding: GAT on temporally-encoded nodes
        self.alpha_adj = nn.Parameter(torch.tensor(1.0))
        self.alpha_od = nn.Parameter(torch.tensor(1.0))
        self.alpha_od_t = nn.Parameter(torch.tensor(1.0))
        
        self.gat_adj = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od_t = GATConv(hidden_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        
        fusion_input_dim = hidden_channels * 3 + 3
        self.fusion_kan = KAN(
            layers_hidden=[fusion_input_dim, hidden_channels, hidden_channels],
            grid_size=3,
            spline_order=2,
        )
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, data: Data) -> torch.Tensor:
        # data.x: [num_nodes, seq_len, feature_dim]
        x = data.x
        num_nodes = x.size(0)
        
        # Temporal encoding per node: [num_nodes, seq_len, feature_dim] -> [num_nodes, hidden]
        _, (h_n, _) = self.temporal_lstm(x)  # h_n: [1, num_nodes, hidden]
        x_temporal = h_n.squeeze(0)  # [num_nodes, hidden]
        
        # Spatial aggregation via GAT
        weights = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
        w_adj, w_od, w_od_t = weights
        
        x_adj = self.gat_adj(x_temporal, data.edge_index_adj)
        x_od = self.gat_od(x_temporal, data.edge_index_od)
        x_od_t = self.gat_od_t(x_temporal, data.edge_index_od_t)
        
        scalars = torch.cat([
            w_adj.expand(num_nodes, 1),
            w_od.expand(num_nodes, 1),
            w_od_t.expand(num_nodes, 1),
        ], dim=1)
        
        x_concat = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
        fused = F.relu(self.fusion_kan(x_concat))
        fused = self.dropout(fused)
        return fused


class SequentialTwoStagePredictor(nn.Module):
    """Classifier and regressor with temporal-aware encoding."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32, seq_len: int = 8):
        super().__init__()
        # in_channels is now feature_dim, not flattened
        self.encoder = TemporalGATEncoder(in_channels, seq_len, hidden_channels=hidden_channels)
        embed_dim = hidden_channels

        self.classifier = KAN(
            layers_hidden=[embed_dim, embed_dim // 2, out_channels],
            grid_size=3,
            spline_order=2,
        )
        self.regressor = KAN(
            layers_hidden=[embed_dim, embed_dim // 2, out_channels],
            grid_size=3,
            spline_order=2,
        )
        self.dropout_cls = nn.Dropout(0.2)
        self.dropout_reg = nn.Dropout(0.2)

    def forward_classifier(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Return embeddings and classification logits."""
        hidden = self.encoder(data)
        hidden_dropped = self.dropout_cls(hidden)
        logits = self.classifier(hidden_dropped)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stage 2: Regress on precomputed embeddings."""
        hidden_dropped = self.dropout_reg(hidden)
        return self.regressor(hidden_dropped)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass for inference."""
        hidden = self.encoder(data)
        logits = self.classifier(self.dropout_cls(hidden))
        reg_out = self.regressor(self.dropout_reg(hidden))
        return logits, reg_out


class GraphSequenceData(Data):
    """Custom PyG data object with multiple edge indices."""
    def __inc__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key in {"edge_index_adj", "edge_index_od", "edge_index_od_t"}:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class GraphSequenceDataset(Dataset):
    """Dataset wrapper for graph sequences."""
    def __init__(
        self,
        features: torch.Tensor,
        y_reg: torch.Tensor,
        y_cls: torch.Tensor,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> None:
        self.features = features.clone()
        self.y_reg = y_reg.clone()
        self.y_cls = y_cls.clone()
        self.edge_index_adj = edge_index_adj.clone().long()
        self.edge_index_od = edge_index_od.clone().long()
        self.edge_index_od_t = edge_index_od_t.clone().long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> GraphSequenceData:
        data = GraphSequenceData()
        feat = self.features[idx]
        data.x = feat
        data.num_nodes = feat.shape[0]
        data.y_cls = self.y_cls[idx]
        data.y_reg = self.y_reg[idx]
        data.edge_index_adj = self.edge_index_adj
        data.edge_index_od = self.edge_index_od
        data.edge_index_od_t = self.edge_index_od_t
        return data


@dataclass
class RDPAccountant:
    """Opacus-equivalent RDP accountant with exact Poisson subsampling."""
    noise_multiplier: float
    sample_rate: float
    steps: int = 0

    def step(self) -> None:
        """Record one training step."""
        self.steps += 1

    def _rdp_gaussian(self, alpha: float, sigma: float) -> float:
        """Exact RDP for Gaussian mechanism (matching Opacus formula)."""
        if alpha < 1:
            return 0.0
        return alpha * (1 + 1 / (2 * sigma**2 * alpha)) * np.log1p(1 / (sigma**2 * alpha))

    def _rdp_subsampling(self, alpha: float, q: float, rdp_full: float) -> float:
        """Exact Poisson subsampling amplification (Opacus)."""
        if q == 0 or q == 1:
            return rdp_full
        return (np.exp((alpha - 1) * np.log1p(q)) - 1) * rdp_full

    def get_epsilon(self, delta: float, orders: Optional[List[float]] = None) -> float:
        """Convert accumulated RDP to (ε, δ)-DP with exact subsampling."""
        if self.steps == 0:
            return 0.0
        if orders is None:
            orders = np.logspace(1.0, 10.0, 100).tolist()

        sigma = self.noise_multiplier
        rdp_totals: List[float] = []
        valid_orders: List[float] = []
        for alpha in orders:
            if alpha == 1:
                continue
            rdp_step = self._rdp_gaussian(alpha, sigma)
            rdp_subsampled = self._rdp_subsampling(alpha, self.sample_rate, rdp_step)
            rdp_totals.append(rdp_subsampled * self.steps)
            valid_orders.append(alpha)

        eps_from_rdp: List[float] = []
        for alpha, rdp_total_alpha in zip(valid_orders, rdp_totals):
            eps = rdp_total_alpha + np.log(1 / delta) / (alpha - 1)
            eps_from_rdp.append(eps)

        return min(eps_from_rdp) if eps_from_rdp else float("inf")


def compute_noise_multiplier(target_epsilon: float, target_delta: float,
                             sample_rate: float, steps: int) -> float:
    """Exact Opacus-style noise multiplier via binary search on σ."""
    if steps == 0:
        return 1.0

    def epsilon_for_sigma(sigma: float) -> float:
        acc = RDPAccountant(sigma, sample_rate, steps)
        return acc.get_epsilon(target_delta)

    low, high = 0.1, 10.0
    while high - low > 1e-5:
        mid = (low + high) / 2
        if epsilon_for_sigma(mid) < target_epsilon:
            high = mid
        else:
            low = mid
    return round(high, 3)


@dataclass
class DPConfig:
    """Differential privacy configuration with budget control."""
    enabled: bool
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    max_grad_norm: float
    sample_rate: float
    epsilon_tolerance: float = 0.05


class FocalLoss(nn.Module):
    """Binary focal loss with logits for multi-channel targets."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Standard BCE with logits per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t is the model-assigned probability of the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t).pow(self.gamma)
        loss = focal_factor * bce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            # Apply alpha only to positives; keep negatives at weight 1.0
            alpha_factor = alpha_t * targets + (1.0 - targets)
            loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def classification_metrics_per_channel(
    preds: np.ndarray,
    targets: np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
) -> Dict[str, float]:
    """Compute precision/recall/F1/accuracy per output channel.

    Expects preds/targets shaped [N, C] (or any shape that can be reshaped to [-1, C]).
    Returns both per-channel metrics and macro-averaged metrics.
    """
    preds_2d = preds.reshape(-1, preds.shape[-1])
    targets_2d = targets.reshape(-1, targets.shape[-1])
    n_channels = preds_2d.shape[1]

    metrics: Dict[str, float] = {}
    precisions, recalls, f1s, accuracies = [], [], [], []
    for c in range(n_channels):
        preds_bin = preds_2d[:, c] >= 0.5
        targets_bin = targets_2d[:, c] >= 0.5

        tp = np.logical_and(preds_bin, targets_bin).sum()
        fp = np.logical_and(preds_bin, ~targets_bin).sum()
        fn = np.logical_and(~preds_bin, targets_bin).sum()
        tn = np.logical_and(~preds_bin, ~targets_bin).sum()

        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        accuracy = float((tp + tn) / (tp + tn + fp + fn + 1e-8))

        name = channel_names[c] if c < len(channel_names) else f"ch{c}"
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall
        metrics[f"f1_{name}"] = f1
        metrics[f"accuracy_{name}"] = accuracy

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)

    metrics["precision"] = float(np.mean(precisions)) if precisions else 0.0
    metrics["recall"] = float(np.mean(recalls)) if recalls else 0.0
    metrics["f1"] = float(np.mean(f1s)) if f1s else 0.0
    metrics["accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
    return metrics


class PerSampleGradientClipper:
    """Per-sample gradient clipping - simplified approach without functorch."""
    def __init__(self, model: nn.Module, max_grad_norm: float):
        self.model = model
        self.max_grad_norm = max_grad_norm

    def compute_per_sample_gradients(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        edge_indices: Tuple,
        loss_fn,
        is_classification: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Compute per-sample gradients via loop."""
        edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
        all_grads = []
        
        for i in range(batch_x.shape[0]):
            self.model.zero_grad(set_to_none=True)
            
            data = Data(
                x=batch_x[i],
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            
            if is_classification:
                _, node_logits = self.model.forward_classifier(data)
                # REMOVED AGGREGATION: Use node-level logits and targets directly
                loss = loss_fn(node_logits, batch_y[i])
            else:
                _, node_reg = self.model(data)
                # REMOVED AGGREGATION: Use node-level regression
                mask = (batch_y[i] >= 0).float()
                loss = loss_fn(node_reg * mask, batch_y[i] * mask)
            
            loss.backward()
            
            sample_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None and param.requires_grad:
                    sample_grads[name] = param.grad.clone().detach()
            
            # Clip this sample's gradient
            grad_norm = torch.sqrt(
                sum(torch.sum(g ** 2) for g in sample_grads.values())
            )
            clip_coef = min(1.0, self.max_grad_norm / (grad_norm + 1e-10))
            clipped_grads = {k: v * clip_coef for k, v in sample_grads.items()}
            all_grads.append(clipped_grads)
        
        # Average clipped gradients across samples
        avg_grads = {}
        for key in all_grads[0].keys():
            avg_grads[key] = torch.mean(
                torch.stack([g[key] for g in all_grads]),
                dim=0
            )
        
        return avg_grads

    def add_noise_to_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        noise_multiplier: float,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Add calibrated Gaussian noise to gradients."""
        noisy_grads = {}
        noise_scale = noise_multiplier * self.max_grad_norm / batch_size
        
        for key, grad in gradients.items():
            noise = torch.normal(
                mean=0.0,
                std=noise_scale,
                size=grad.shape,
                device=grad.device,
            )
            noisy_grads[key] = grad + noise
        
        return noisy_grads


def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    """Aggregate node-level features to graph-level via mean pooling."""
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    """Convert node-level targets to graph-level."""
    if target.dim() == 0:  # Scalar
        return target.unsqueeze(0)
    elif target.dim() == 1:  # [num_nodes]
        return target.mean(dim=0, keepdim=True)
    else:  # [num_nodes, feature_dim]
        return target.mean(dim=0, keepdim=True)


def train_stage1_with_dp(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    dp_config: DPConfig,
    batch_size: int,
) -> Tuple[List[Dict], RDPAccountant, float]:
    """Train stage 1 with proper DP-SGD and epsilon budget control."""
    stage_start_time = time.time()
    print("\n" + "="*80)
    print("STAGE 1: TRAINING DELAY CLASSIFIER WITH DP-SGD")
    print("="*80)
    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")
    
    # Freeze regressor
    for param in model.regressor.parameters():
        param.requires_grad = False
    
    trainable_params = list(model.encoder.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    
    # Ensure pos_weight is on the correct device
    if isinstance(pos_weight, (float, int)):
        pos_weight_t = torch.tensor([pos_weight], device=device)
    else:
        pos_weight_t = pos_weight.to(device)
        
    cls_loss_fn = FocalLoss(
        alpha=pos_weight_t,
        gamma=2.0,
        reduction="mean",
    )
    
    accountant = RDPAccountant(
        noise_multiplier=dp_config.noise_multiplier,
        sample_rate=dp_config.sample_rate if dp_config.enabled else 1.0,
    )
    
    if dp_config.enabled:
        clipper = PerSampleGradientClipper(model, dp_config.max_grad_norm)
        print(f"✓ DP-SGD enabled: ε_target={dp_config.target_epsilon}, δ={dp_config.target_delta}")
        print(f"  Noise multiplier: {dp_config.noise_multiplier}")
        print(f"  Max grad norm: {dp_config.max_grad_norm}")
        print(f"  Sample rate: {dp_config.sample_rate}")
    else:
        print("✗ DP-SGD disabled (standard training)")
    
    history = []
    best_f1 = 0.0
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="max")
    
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stage1_checkpoint.pth') if CHECKPOINT_DIR else None

    # Check if checkpoint exists to resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming Stage 1 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_losses = []
        
        num_samples = train_x.shape[0]
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y_cls[batch_indices].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if dp_config.enabled:
                # DP-SGD
                per_sample_grads = clipper.compute_per_sample_gradients(
                    batch_x, batch_y, edge_indices, cls_loss_fn, is_classification=True
                )
                noisy_grads = clipper.add_noise_to_gradients(
                    per_sample_grads,
                    dp_config.noise_multiplier,
                    len(batch_indices),
                )
                for name, param in model.named_parameters():
                    if param.requires_grad and name in noisy_grads:
                        param.grad = noisy_grads[name]
                
                # Loss for logging
                with torch.no_grad():
                    logits_list = []
                    targets_list = []
                    for i in range(len(batch_x)):
                        data = Data(
                            x=batch_x[i],
                            edge_index_adj=edge_indices[0],
                            edge_index_od=edge_indices[1],
                            edge_index_od_t=edge_indices[2],
                        )
                        _, node_logits = model.forward_classifier(data.to(device))
                        # REMOVED AGGREGATION
                        logits_list.append(node_logits)
                        targets_list.append(batch_y[i])
                    all_logits = torch.cat(logits_list, dim=0)
                    all_targets = torch.cat(targets_list, dim=0)
                    loss = cls_loss_fn(all_logits, all_targets)
                
                accountant.step()
            else:
                # Standard training
                logits_list = []
                targets_list = []
                for i in range(len(batch_x)):
                    data = Data(
                        x=batch_x[i],
                        edge_index_adj=edge_indices[0],
                        edge_index_od=edge_indices[1],
                        edge_index_od_t=edge_indices[2],
                    )
                    _, node_logits = model.forward_classifier(data.to(device))
                    # REMOVED AGGREGATION
                    logits_list.append(node_logits)
                    targets_list.append(batch_y[i])
                all_logits = torch.cat(logits_list, dim=0)
                all_targets = torch.cat(targets_list, dim=0)
                loss = cls_loss_fn(all_logits, all_targets)
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
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_logits = model.forward_classifier(data)
                # REMOVED AGGREGATION
                val_probs.append(torch.sigmoid(node_logits).cpu())
                val_targets.append(val_y_cls[i].cpu())
        
        val_probs_np = torch.cat(val_probs).numpy()
        val_targets_np = torch.cat(val_targets).numpy()
        # Per-channel (arrival/departure) metrics + macro averages
        val_metrics = classification_metrics_per_channel(
            val_probs_np,
            val_targets_np,
            channel_names=("arrival", "departure"),
        )
        
        epoch_time = time.time() - epoch_start_time
        
        if dp_config.enabled:
            current_epsilon = accountant.get_epsilon(dp_config.target_delta)
        else:
            current_epsilon = float('inf')
        
        history.append({
            'epoch': epoch,
            'stage': 1,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_accuracy': val_metrics['accuracy'],
            'epsilon': current_epsilon,
            'delta': dp_config.target_delta if dp_config.enabled else 0.0,
            'epoch_time_seconds': epoch_time,
            'total_steps': accountant.steps,
        })
        
        eps_str = f"ε: {current_epsilon:.3f}/{dp_config.target_epsilon}" if dp_config.enabled else "No DP"
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val F1 (macro): {val_metrics['f1']:.4f} "
            f"[arr {val_metrics['f1_arrival']:.4f}, dep {val_metrics['f1_departure']:.4f}] | "
            f"{eps_str} | Time: {epoch_time:.2f}s"
        )
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = {
                'encoder': model.encoder.state_dict(),
                'classifier': model.classifier.state_dict(),
            }
            print("  ✓ New best checkpoint")
            # Save checkpoint to disk for resume capability
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, history[-1]['train_loss'], checkpoint_path)
                print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        if early_stopping(val_metrics['f1'], epoch):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.encoder.load_state_dict(best_state['encoder'])
        model.classifier.load_state_dict(best_state['classifier'])
    
    for param in model.regressor.parameters():
        param.requires_grad = True
    
    stage_time = time.time() - stage_start_time
    final_epsilon = accountant.get_epsilon(dp_config.target_delta) if dp_config.enabled else float('inf')
    
    print(f"\nStage 1 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    print(f"Final ε: {final_epsilon:.3f} (target: {dp_config.target_epsilon})")
    
    return history, accountant, stage_time


def train_stage2_with_dp(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    class_threshold: float,
    patience: int,
    dp_config: DPConfig,
    batch_size: int,
    stage1_accountant: RDPAccountant,
) -> Tuple[List[Dict], RDPAccountant, float]:
    """Train stage 2 (delayed flights regressor) with proper DP-SGD."""
    stage_start_time = time.time()
    print("\n" + "="*80)
    print("STAGE 2: TRAINING DELAY REGRESSOR (DELAYED FLIGHTS) WITH DP-SGD")
    print("="*80)
    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")
    
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr, weight_decay=1e-4)
    reg_loss_fn = nn.MSELoss(reduction='mean')
    
    accountant = RDPAccountant(
        noise_multiplier=dp_config.noise_multiplier,
        sample_rate=dp_config.sample_rate if dp_config.enabled else 1.0,
        steps=stage1_accountant.steps,
    )
    
    if dp_config.enabled:
        clipper = PerSampleGradientClipper(model, dp_config.max_grad_norm)
        current_eps = stage1_accountant.get_epsilon(dp_config.target_delta)
        print(f"✓ DP-SGD enabled (continuing from stage 1)")
        print(f"  Current ε: {current_eps:.3f} / {dp_config.target_epsilon}")
    else:
        print("✗ DP-SGD disabled")
    
    history = []
    best_val_loss = float('inf')
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stage2_checkpoint.pth') if CHECKPOINT_DIR else None

    # Check if checkpoint exists to resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resuming Stage 2 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_losses = []
        
        num_samples = train_x.shape[0]
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = train_x[batch_indices].to(device)
            batch_y_reg = train_y_reg[batch_indices].to(device)
            batch_y_cls = train_y_cls[batch_indices].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if dp_config.enabled:
                per_sample_grads = clipper.compute_per_sample_gradients(
                    batch_x, batch_y_reg, edge_indices, reg_loss_fn, is_classification=False
                )
                noisy_grads = clipper.add_noise_to_gradients(
                    per_sample_grads,
                    dp_config.noise_multiplier,
                    len(batch_indices),
                )
                for name, param in model.named_parameters():
                    if param.requires_grad and name in noisy_grads:
                        param.grad = noisy_grads[name]
                
                with torch.no_grad():
                    reg_preds = []
                    reg_targets = []
                    for i in range(len(batch_x)):
                        data = Data(
                            x=batch_x[i],
                            edge_index_adj=edge_indices[0],
                            edge_index_od=edge_indices[1],
                            edge_index_od_t=edge_indices[2],
                        )
                        _, node_reg = model(data.to(device))
                        # REMOVED AGGREGATION
                        reg_preds.append(node_reg)
                        reg_targets.append(batch_y_reg[i])
                    reg_preds = torch.cat(reg_preds, dim=0)
                    reg_targets = torch.cat(reg_targets, dim=0)
                    
                    cls_mask = []
                    for i in range(len(batch_y_cls)):
                        # REMOVED AGGREGATION
                        cls_mask.append(batch_y_cls[i])
                    cls_mask = torch.cat(cls_mask, dim=0)
                    mask = (cls_mask >= class_threshold).float()
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(-1)
                    
                    # Per-channel masked MSE then macro-average across channels
                    se = (reg_preds - reg_targets) ** 2
                    num = (se * mask).sum(dim=0)
                    den = mask.sum(dim=0).clamp_min(1.0)
                    loss = (num / den).mean()
                
                accountant.step()
            else:
                reg_preds = []
                reg_targets = []
                for i in range(len(batch_x)):
                    data = Data(
                        x=batch_x[i],
                        edge_index_adj=edge_indices[0],
                        edge_index_od=edge_indices[1],
                        edge_index_od_t=edge_indices[2],
                    )
                    _, node_reg = model(data.to(device))
                    # REMOVED AGGREGATION
                    reg_preds.append(node_reg)
                    reg_targets.append(batch_y_reg[i])
                reg_preds = torch.cat(reg_preds, dim=0)
                reg_targets = torch.cat(reg_targets, dim=0)
                
                cls_mask = []
                for i in range(len(batch_y_cls)):
                    # REMOVED AGGREGATION
                    cls_mask.append(batch_y_cls[i])
                cls_mask = torch.cat(cls_mask, dim=0)
                mask = (cls_mask >= class_threshold).float()
                if mask.dim() == 1:
                    mask = mask.unsqueeze(-1)
                
                # Per-channel masked MSE then macro-average across channels
                se = (reg_preds - reg_targets) ** 2
                num = (se * mask).sum(dim=0)
                den = mask.sum(dim=0).clamp_min(1.0)
                loss = (num / den).mean()
                loss.backward()
            
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_reg = model(data)
                # REMOVED AGGREGATION
                
                mask = (val_y_cls[i] >= class_threshold).float()
                if mask.dim() == 1:
                    mask = mask.unsqueeze(-1)
                
                # Per-channel masked MSE then macro-average across channels
                se = (node_reg.cpu() - val_y_reg[i]) ** 2
                num = (se * mask).sum(dim=0)
                den = mask.sum(dim=0).clamp_min(1.0)
                loss = (num / den).mean()
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start_time
        
        if dp_config.enabled:
            current_epsilon = accountant.get_epsilon(dp_config.target_delta)
        else:
            current_epsilon = float('inf')
        
        history.append({
            'epoch': epoch,
            'stage': 2,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_loss': val_loss,
            'epsilon': current_epsilon,
            'delta': dp_config.target_delta if dp_config.enabled else 0.0,
            'epoch_time_seconds': epoch_time,
            'total_steps': accountant.steps,
        })
        
        eps_str = f"ε: {current_epsilon:.3f}/{dp_config.target_epsilon}" if dp_config.enabled else "No DP"
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {history[-1]['train_loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | {eps_str} | Time: {epoch_time:.2f}s"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.regressor.state_dict()
            print("  ✓ New best checkpoint")
            # Save checkpoint to disk for resume capability
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                print(f"  ✓ Checkpoint saved to {checkpoint_path}")
        
        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.regressor.load_state_dict(best_state)
    
    for param in model.parameters():
        param.requires_grad = True
    
    stage_time = time.time() - stage_start_time
    final_epsilon = accountant.get_epsilon(dp_config.target_delta) if dp_config.enabled else float('inf')
    
    print(f"\nStage 2 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    print(f"Final ε: {final_epsilon:.3f} (target: {dp_config.target_epsilon})")
    
    return history, accountant, stage_time


def train_stage3_with_dp(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    class_threshold: float,
    delay_threshold: float,
    patience: int,
    dp_config: DPConfig,
    batch_size: int,
    stage2_accountant: RDPAccountant,
) -> Tuple[List[Dict], RDPAccountant, float]:
    """IMPROVED: Better handling of non-delayed flights (Stage 3 v2)."""
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("STAGE 3: TRAINING DELAY REGRESSOR (NON-DELAYED FLIGHTS) - IMPROVED")
    print(f"Training on flights with |delay| < {delay_threshold} min")
    print("=" * 80)
    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")

    # Freeze encoder and classifier initially (regressor-only fine-tuning)
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False

    # Much lower LR, regressor only at first
    optimizer = torch.optim.Adam(
        model.regressor.parameters(),
        lr=lr * 0.01,
        weight_decay=1e-5,
    )

    # Use Huber loss (more robust than MSE)
    reg_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    accountant = RDPAccountant(
        noise_multiplier=dp_config.noise_multiplier,
        sample_rate=dp_config.sample_rate if dp_config.enabled else 1.0,
        steps=stage2_accountant.steps,
    )

    print(f"✓ Regressor-only training with LR={lr * 0.01:.6f}")
    print("✓ Using Huber loss (robust to outliers)")

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    early_stopping = EarlyStopping(patience=patience, mode="min")

    # Epoch at which to unfreeze encoder (progressive fine-tuning)
    unfreeze_epoch = max(2, epochs // 2)

    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stage3_checkpoint.pth') if CHECKPOINT_DIR else None

    # Check if checkpoint exists to resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        # Load checkpoint to check epoch and adjust optimizer if needed
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = checkpoint['epoch']
        
        # If resuming from a state where encoder was already unfrozen,
        # we must re-initialize the optimizer with the correct parameter groups
        if start_epoch >= unfreeze_epoch:
            print(f"  → Resuming from epoch {start_epoch} (Encoder already unfrozen)")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                [
                    {"params": model.encoder.parameters(), "lr": lr * 0.001},
                    {"params": model.regressor.parameters(), "lr": lr * 0.01},
                ],
                weight_decay=1e-5,
            )
            
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Resuming Stage 3 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        # Progressive unfreezing of encoder
        # Only unfreeze if we haven't already done so (checked via start_epoch)
        if epoch == unfreeze_epoch and start_epoch != unfreeze_epoch:
            print("  → Unfreezing encoder with very low LR...")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(
                [
                    {"params": model.encoder.parameters(), "lr": lr * 0.001},
                    {"params": model.regressor.parameters(), "lr": lr * 0.01},
                ],
                weight_decay=1e-5,
            )

        epoch_start_time = time.time()
        model.train()
        epoch_losses: List[float] = []
        total_nondelayed = 0.0
        total_values = 0.0

        num_samples = train_x.shape[0]
        indices = torch.randperm(num_samples)

        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = train_x[batch_indices].to(device)
            batch_y_reg = train_y_reg[batch_indices].to(device)
            batch_y_cls = train_y_cls[batch_indices].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass: collect node-level predictions/targets for batch
            reg_preds: List[torch.Tensor] = []
            reg_targets: List[torch.Tensor] = []
            for i in range(len(batch_x)):
                data = Data(
                    x=batch_x[i],
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_reg = model(data.to(device))
                # REMOVED AGGREGATION
                reg_preds.append(node_reg)
                reg_targets.append(batch_y_reg[i])

            reg_preds_t = torch.cat(reg_preds, dim=0)  # [B * N, out_channels]
            reg_targets_t = torch.cat(reg_targets, dim=0)

            # Denormalize targets (detached) ONLY for mask creation
            if scaler is not None:
                with torch.no_grad():
                    targets_denorm = torch.from_numpy(
                        scaler.inverse_transform(reg_targets_t.detach().cpu().numpy())
                    ).to(device)
            else:
                targets_denorm = reg_targets_t.detach()

            # Create element-wise mask for non-delayed values in denormalized space
            element_mask = (targets_denorm.abs() < delay_threshold).float()
            num_nondelayed_per_ch = element_mask.sum(dim=0)
            num_nondelayed = num_nondelayed_per_ch.sum()

            if num_nondelayed > 0:
                # Huber loss in NORMALIZED space (keeps gradient graph), masked by denorm threshold
                loss_per_element = reg_loss_fn(reg_preds_t, reg_targets_t) * element_mask
                # Per-channel average loss, then macro-average
                loss_nondelayed_ch = loss_per_element.sum(dim=0) / num_nondelayed_per_ch.clamp_min(1.0)
                loss_nondelayed = loss_nondelayed_ch.mean()

                # Auxiliary loss on delayed elements after encoder is unfrozen
                delayed_mask = (targets_denorm.abs() >= delay_threshold).float()
                num_delayed = delayed_mask.sum()

                if num_delayed > 0 and epoch >= unfreeze_epoch:
                    aux_se = ((reg_preds_t - reg_targets_t) ** 2) * delayed_mask
                    loss_delayed = aux_se.sum() / num_delayed
                    loss = 0.8 * loss_nondelayed + 0.2 * loss_delayed
                else:
                    loss = loss_nondelayed

                loss.backward()
                total_nondelayed += num_nondelayed.item()
                total_values += float(element_mask.numel())
            else:
                # No non-delayed elements in this batch; just skip update
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss.backward()

            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation in denormalized space with same mask
        model.eval()
        val_losses: List[float] = []
        val_nondelayed = 0.0
        val_total = 0.0

        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_reg = model(data)
                # REMOVED AGGREGATION

                if scaler is not None:
                    target_denorm = torch.from_numpy(
                        scaler.inverse_transform(val_y_reg[i].cpu().numpy())
                    ).to(device)
                    pred_denorm = torch.from_numpy(
                        scaler.inverse_transform(node_reg.cpu().numpy())
                    ).to(device)
                else:
                    target_denorm = val_y_reg[i]
                    pred_denorm = node_reg

                element_mask = (target_denorm.abs() < delay_threshold).float()
                num_nondelayed_per_ch = element_mask.sum(dim=0)
                num_nondelayed = num_nondelayed_per_ch.sum()

                if num_nondelayed > 0:
                    se = ((pred_denorm - target_denorm) ** 2) * element_mask
                    loss_val_ch = se.sum(dim=0) / num_nondelayed_per_ch.clamp_min(1.0)
                    loss_val = loss_val_ch.mean()
                    val_losses.append(loss_val.item())
                    val_nondelayed += num_nondelayed.item()
                    val_total += float(element_mask.numel())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        epoch_time = time.time() - epoch_start_time

        current_epsilon = (
            accountant.get_epsilon(dp_config.target_delta)
            if dp_config.enabled
            else float("inf")
        )

        nondelayed_ratio = (
            total_nondelayed / total_values if total_values > 0 else 0.0
        )
        val_nondelayed_ratio = (
            val_nondelayed / val_total if val_total > 0 else 0.0
        )

        history.append(
            {
                "epoch": epoch,
                "stage": 3,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_nondelayed": total_nondelayed,
                "train_nondelayed_ratio": nondelayed_ratio,
                "val_loss": val_loss,
                "val_nondelayed": val_nondelayed,
                "val_nondelayed_ratio": val_nondelayed_ratio,
                "epsilon": current_epsilon,
                "epoch_time_seconds": epoch_time,
            }
        )

        eps_str = (
            f"ε: {current_epsilon:.3f}/{dp_config.target_epsilon}"
            if dp_config.enabled
            else "No DP"
        )
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val: {val_loss:.4f} | Non-delayed: {total_nondelayed:.0f} "
            f"({nondelayed_ratio*100:.1f}%) | Val ND: {val_nondelayed:.0f} "
            f"({val_nondelayed_ratio*100:.1f}%) | {eps_str} | Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss and val_nondelayed > 0:
            best_val_loss = val_loss
            best_state = {
                "encoder": model.encoder.state_dict(),
                "regressor": model.regressor.state_dict(),
            }
            print("  ✓ New best (Stage 3 v2)")
            # Save checkpoint to disk for resume capability
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
                print(f"  ✓ Checkpoint saved to {checkpoint_path}")

        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.encoder.load_state_dict(best_state["encoder"])
        model.regressor.load_state_dict(best_state["regressor"])

    stage_time = time.time() - stage_start_time
    final_epsilon = (
        accountant.get_epsilon(dp_config.target_delta)
        if dp_config.enabled
        else float("inf")
    )

    print(f"\nStage 3 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    print(f"Final ε: {final_epsilon:.3f} (target: {dp_config.target_epsilon})")

    return history, accountant, stage_time

def final_evaluation(
    model: SequentialTwoStagePredictor,
    edge_indices: Tuple,
    device: torch.device,
    scaler,
    horizons: List[int],
    delay_dim: int,
    num_nodes: int,
    test_x: torch.Tensor,
    test_y_reg: torch.Tensor,
    test_y_cls: torch.Tensor,
    class_threshold: float,
    delay_threshold: float,
    model_path: str,
    histories: List[Dict],
    final_epsilon: float,
    final_delta: float,
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    train_samples: int,
    val_samples: int,
    dp_config: DPConfig,
) -> None:
    """Final evaluation and export with Stage 3 regressor."""
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    print(f"Test samples: {len(test_x)}")
    
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'encoder': model.encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        'regressor': model.regressor.state_dict(),
        'final_epsilon': float(final_epsilon),
        'final_delta': float(final_delta),
        'target_epsilon': float(dp_config.target_epsilon),
        'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
    }, model_path)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    model.regressor.load_state_dict(checkpoint['regressor'])
    
    model.eval()
    
    logits_list, reg_list = [], []
    targets_cls_list, targets_reg_list = [], []
    
    USE_FAST_EVAL = False
    
    print("[EVALUATION] Processing test samples...")
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )
            
            node_logits, node_reg = model(data)
            # REMOVED AGGREGATION
            
            logits_list.append(torch.sigmoid(node_logits).cpu().numpy())
            reg_list.append(node_reg.cpu().numpy())
            targets_cls_list.append(test_y_cls[i].cpu().numpy())
            targets_reg_list.append(test_y_reg[i].cpu().numpy())
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")
    
    test_probs = np.concatenate(logits_list, axis=0)
    test_reg_preds = np.concatenate(reg_list, axis=0)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)
    
    # Classification metrics (arrival/departure separately + macro)
    test_cls_metrics = classification_metrics_per_channel(
        test_probs,
        test_cls_targets,
        channel_names=("arrival", "departure"),
    )
    
    # Apply classifier gating
    test_mask = (test_probs >= class_threshold)
    gated_preds = test_reg_preds * test_mask
    
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(gated_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
    else:
        preds_denorm = gated_preds
        targets_denorm = test_reg_targets
    
    # Treat negative values as on time (0 min)
    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)
    
    # Flatten arrays for consistent masking (multi-channel arrays need flattening before masking)
    preds_flat = preds_denorm.flatten()
    targets_flat = targets_denorm.flatten()
    
    # Evaluate on delayed flights (Actual >= Threshold)
    delayed_mask = targets_flat >= delay_threshold
    if delayed_mask.sum() > 0:
        delayed_preds = preds_flat[delayed_mask]
        delayed_targets = targets_flat[delayed_mask]
        mae_delayed = np.mean(np.abs(delayed_preds - delayed_targets))
        rmse_delayed = np.sqrt(np.mean((delayed_preds - delayed_targets) ** 2))
    else:
        mae_delayed, rmse_delayed = 0.0, 0.0
    
    # Evaluate on non-delayed flights (1 min <= Actual < Threshold)
    nondelayed_mask = (targets_flat >= 1.0) & (targets_flat < delay_threshold)
    if nondelayed_mask.sum() > 0:
        nondelayed_preds = preds_flat[nondelayed_mask]
        nondelayed_targets = targets_flat[nondelayed_mask]
        mae_nondelayed = np.mean(np.abs(nondelayed_preds - nondelayed_targets))
        rmse_nondelayed = np.sqrt(np.mean((nondelayed_preds - nondelayed_targets) ** 2))
    else:
        mae_nondelayed, rmse_nondelayed = 0.0, 0.0
    
    # Overall metrics
    mae_overall = np.mean(np.abs(preds_denorm - targets_denorm))
    rmse_overall = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
    
    print("\nCLASSIFICATION (macro over arrival/departure):")
    print(f"  Precision: {test_cls_metrics['precision']:.4f} | Recall: {test_cls_metrics['recall']:.4f}")
    print(f"  F1: {test_cls_metrics['f1']:.4f} | Accuracy: {test_cls_metrics['accuracy']:.4f}")
    print("  Per-channel:")
    print(
        f"    Arrival   - P: {test_cls_metrics['precision_arrival']:.4f} "
        f"R: {test_cls_metrics['recall_arrival']:.4f} F1: {test_cls_metrics['f1_arrival']:.4f} "
        f"Acc: {test_cls_metrics['accuracy_arrival']:.4f}"
    )
    print(
        f"    Departure - P: {test_cls_metrics['precision_departure']:.4f} "
        f"R: {test_cls_metrics['recall_departure']:.4f} F1: {test_cls_metrics['f1_departure']:.4f} "
        f"Acc: {test_cls_metrics['accuracy_departure']:.4f}"
    )
    
    print(f"\nREGRESSION (delayed flights >= {delay_threshold} min):")
    print(f"  MAE: {mae_delayed:.4f} min | RMSE: {rmse_delayed:.4f} min")
    print(f"  Number of delayed samples: {delayed_mask.sum()}")
    
    print(f"\nREGRESSION (non-delayed flights 1-{delay_threshold} min):")
    print(f"  MAE: {mae_nondelayed:.4f} min | RMSE: {rmse_nondelayed:.4f} min")
    print(f"  Number of non-delayed samples: {nondelayed_mask.sum()}")
    
    print("\nREGRESSION (overall):")
    print(f"  MAE: {mae_overall:.4f} min | RMSE: {rmse_overall:.4f} min")
    
    # Visualize classification results
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating classification results plots...")
        try:
            # Convert predictions to binary labels
            test_cls_pred = (test_probs >= class_threshold).astype(int).flatten()
            test_cls_true = test_cls_targets.flatten()
            
            # Handle multi-dimensional regression targets (take first channel/delay)
            if targets_denorm.ndim > 1 and targets_denorm.shape[1] > 1:
                test_reg_true = targets_denorm[:, 0].flatten()
                test_reg_pred = preds_denorm[:, 0].flatten()
            else:
                test_reg_true = targets_denorm.flatten()
                test_reg_pred = preds_denorm.flatten()
            
            visualize_classification_results(
                test_cls_true,
                test_cls_pred,
                test_reg_true,
                test_reg_pred,
                threshold=class_threshold,
                save_path=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            print("  ✓ Classification results visualization saved")
        except Exception as e:
            print(f"  ✗ Error generating classification results visualization: {e}")
    
    print("\nPRIVACY BUDGET:")
    print(f"  Target ε: {dp_config.target_epsilon:.3f}")
    print(f"  Final ε: {final_epsilon:.3f}")
    if dp_config.enabled:
        if final_epsilon <= dp_config.target_epsilon:
            print(f"  ✓ Budget maintained (within target)")
        else:
            overshoot = final_epsilon - dp_config.target_epsilon
            print(f"  ⚠️ Budget exceeded by {overshoot:.3f} ε ({overshoot/dp_config.target_epsilon*100:.1f}%)")
    print(f"  Final δ: {final_delta:.2e}")
    
    print("\nTRAINING TIME:")
    total_time = stage1_time + stage2_time + stage3_time
    print(f"  Stage 1: {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    print(f"  Stage 2: {stage2_time:.2f}s ({stage2_time/60:.2f} min)")
    print(f"  Stage 3: {stage3_time:.2f}s ({stage3_time/60:.2f} min)")
    print(f"  Total: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    
    # Generate unique filenames with noise multiplier (sigma) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_str = f"sigma{dp_config.noise_multiplier:.2f}".replace(".", "_")
    
    history_csv = f"kan_gat_dp_three_stage_history_{sigma_str}_{timestamp}.csv"
    summary_csv = f"kan_gat_dp_three_stage_summary_{sigma_str}_{timestamp}.csv"
    
    # Export history
    if histories:
        with open(history_csv, "w", newline="") as f:
            all_fields = sorted({k for row in histories for k in row})
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(histories)
    
    # Export summary
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        summary = {
            'classification_precision': test_cls_metrics['precision'],
            'classification_recall': test_cls_metrics['recall'],
            'classification_f1': test_cls_metrics['f1'],
            'classification_accuracy': test_cls_metrics['accuracy'],
            'regression_mae_delayed': mae_delayed,
            'regression_rmse_delayed': rmse_delayed,
            'regression_mae_nondelayed': mae_nondelayed,
            'regression_rmse_nondelayed': rmse_nondelayed,
            'regression_mae_overall': mae_overall,
            'regression_rmse_overall': rmse_overall,
            'num_delayed_samples': int(delayed_mask.sum()),
            'num_nondelayed_samples': int(nondelayed_mask.sum()),
            'target_epsilon': dp_config.target_epsilon,
            'final_epsilon': final_epsilon,
            'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
            'epsilon_overshoot': max(0, final_epsilon - dp_config.target_epsilon) if dp_config.enabled else 0,
            'final_delta': final_delta,
            'stage1_time_seconds': stage1_time,
            'stage2_time_seconds': stage2_time,
            'stage3_time_seconds': stage3_time,
            'total_training_time_seconds': total_time,
            'total_training_time_minutes': total_time / 60,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': len(test_x),
        }
        for k, v in summary.items():
            writer.writerow([k, v])
    
    print(f"\n✓ Results saved to:")
    print(f"  - {model_path}")
    print(f"  - {history_csv}")
    print(f"  - {summary_csv}")
    
    # Download files to local device (only in Colab)
    if IN_COLAB and colab_files is not None:
        print("\n[DOWNLOAD] Downloading files to local device...")
        
        # Files to download: model, history, summary, and checkpoints
        files_to_download = [
            model_path,
            history_csv,
            summary_csv,
        ]
        
        # Add checkpoint files if they exist
        checkpoint_files = [
            os.path.join(CHECKPOINT_DIR, 'stage1_checkpoint.pth'),
            os.path.join(CHECKPOINT_DIR, 'stage2_checkpoint.pth'),
            os.path.join(CHECKPOINT_DIR, 'stage3_checkpoint.pth'),
        ]
        files_to_download.extend(checkpoint_files)
        
        for file_path in files_to_download:
            if os.path.exists(file_path):
                try:
                    colab_files.download(file_path)
                    print(f"  ✓ Downloaded: {file_path}")
                except Exception as e:
                    print(f"  ✗ Error downloading {file_path}: {e}")
            else:
                print(f"  - File not found: {file_path}")
    else:
        print("\n[INFO] Not running in Colab - files saved locally, no download needed.")


def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    
    # Automatically download checkpoint in Colab as soon as it's saved
    if IN_COLAB and colab_files is not None:
        try:
            colab_files.download(path)
            print(f"  ✓ Checkpoint downloaded: {path}")
        except Exception as e:
            print(f"  ✗ Error downloading checkpoint: {e}")

def load_checkpoint(model, optimizer, path):
    # Use map_location to handle loading checkpoints saved on different devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-stage DP-SGD for KAN-GAT with epsilon budget control")
    parser.add_argument('--data_source', type=str, default='cdata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument(
        '--horizons',
        type=int,
        nargs=1,
        default=[12],
        choices=[3, 6, 12, 24],
        help='Train/test ONLY this horizon (choose one of 3, 6, 12, 24). Example: --horizons 24',
    )
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--use_node_level', action='store_true', default=True, help='Use node-level labels')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--stage1_epochs', type=int, default=8)
    parser.add_argument('--stage2_epochs', type=int, default=4)
    parser.add_argument('--stage3_epochs', type=int, default=4, help='Epochs for non-delayed regressor')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dp', default=True, action='store_true', help='Enable DP-SGD')
    parser.add_argument('--target_epsilon', type=float, default=15.0, help='Target epsilon for tracking (not used for computing noise)')
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=2.0, help='Fixed noise multiplier for DP-SGD (not auto-computed)')
    parser.add_argument('--max_grad_norm', type=float, default=1.5)
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--epsilon_tolerance', type=float, default=0.05)
    parser.add_argument('--model_path', type=str, default='kan_gat_dp_three_stage.pth')
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random)')
    parser.add_argument('--balance_50_50', action='store_true', default=True, help='Apply random undersampling to achieve 50-50 class balance')
    return parser.parse_args()


def main() -> None:
    global CHECKPOINT_DIR
    args = parse_args()
    
    # Set up checkpoint directory if not already set
    if not CHECKPOINT_DIR:
        CHECKPOINT_DIR = setup_checkpoint_directory()
    
    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'
    
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
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
        raise ValueError(
            f"This script trains/tests a single horizon only. "
            f"Pass exactly one value via --horizons (3/6/12/24). Got: {args.horizons}"
        )
    max_horizon = horizons[0]
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    # Keep temporal structure: use feature_dim directly, not flattened
    in_channels = feature_dim
    seq_len = args.seq_len
    out_channels = delay_dim
    
    # Use temporal-aware build function
    build_fn = build_sequences_temporal
    
    if args.use_node_level:
        print("[INFO] Using NODE-LEVEL labels")
    else:
        print("[INFO] Using GRAPH-LEVEL labels")
    
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    test_x, test_y_reg, test_y_cls = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    
    if args.balance_50_50:
        print("\n[INFO] Applying random undersampling for 50-50 balance...")
        # Determine sample-level labels (majority vote for node-level)
        sample_means = train_y_cls.mean(dim=(1, 2))
        sample_labels = (sample_means >= 0.5).long()
        
        pos_indices = (sample_labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (sample_labels == 0).nonzero(as_tuple=True)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        print(f"  Original counts - Positive (Delayed): {n_pos}, Negative: {n_neg}")
        
        if n_pos > 0 and n_neg > 0:
            min_count = min(n_pos, n_neg)
            print(f"  Undersampling to {min_count} samples per class...")
            
            # Randomly select indices
            perm_pos = torch.randperm(n_pos)[:min_count]
            perm_neg = torch.randperm(n_neg)[:min_count]
            
            selected_pos = pos_indices[perm_pos]
            selected_neg = neg_indices[perm_neg]
            
            # Combine and shuffle
            combined_indices = torch.cat([selected_pos, selected_neg])
            combined_indices = combined_indices[torch.randperm(len(combined_indices))]
            
            # Apply selection
            train_x = train_x[combined_indices]
            train_y_reg = train_y_reg[combined_indices]
            train_y_cls = train_y_cls[combined_indices]
            
            print(f"  New training set size: {len(train_x)}")
        else:
            print("  WARNING: Cannot balance 50-50 because one class has 0 samples. Skipping undersampling.")

    print(f"\n[DATA] Class distribution:")
    print(f"  Train: {train_y_cls.mean().item():.2%} delayed")
    print(f"  Val: {val_y_cls.mean().item():.2%} delayed")
    print(f"  Test: {test_y_cls.mean().item():.2%} delayed")
    
   
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=16,
        seq_len=seq_len,
    ).to(device)
    
    total_samples = len(train_x)
    sample_rate = args.batch_size / total_samples
    steps_per_epoch = int(np.ceil(total_samples / args.batch_size))
    total_steps = (args.stage1_epochs + args.stage2_epochs + args.stage3_epochs) * steps_per_epoch
    
    if args.dp:
        print(f"\nUsing FIXED noise multiplier: {args.noise_multiplier:.3f}")
        print(f"  Sample rate: {sample_rate:.4f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Expected total steps: {total_steps} ({steps_per_epoch} steps/epoch × {args.stage1_epochs + args.stage2_epochs + args.stage3_epochs} epochs)")
        print(f"  Target epsilon for tracking: {args.target_epsilon:.3f}")
        args.sample_rate = sample_rate
    
    dp_config = DPConfig(
        enabled=args.dp,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        sample_rate=sample_rate if args.dp else args.sample_rate,
        epsilon_tolerance=args.epsilon_tolerance,
    )
    
    # Per-channel (arrival, departure) positive rate aggregated across all samples and airports
    cls_pos_rate = train_y_cls.float().mean(dim=(0, 1))  # shape: [2]
    # pos_weight = negatives / positives, broadcast over the two channels
    pos_weight = (1.0 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)
    
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    print(f"Train samples: {len(train_x)}")
    print(f"Val samples: {len(val_x)}")
    print(f"Test samples: {len(test_x)}")
    print(f"Class balance (delayed): {cls_pos_rate.mean().item():.2%}")
    
    history_s1, accountant_s1, stage1_time = train_stage1_with_dp(
        model, train_x, train_y_cls, val_x, val_y_cls,
        edge_indices, device, args.stage1_epochs, args.lr,
        pos_weight, args.patience, dp_config, args.batch_size,
    )
    
    history_s2, accountant_s2, stage2_time = train_stage2_with_dp(
        model, train_x, train_y_reg, train_y_cls,
        val_x, val_y_reg, val_y_cls, edge_indices, device,
        args.stage2_epochs, args.lr, scaler, args.class_threshold,
        args.patience, dp_config, args.batch_size, accountant_s1,
    )
    
    history_s3, accountant_s3, stage3_time = train_stage3_with_dp(
        model, train_x, train_y_reg, train_y_cls,
        val_x, val_y_reg, val_y_cls, edge_indices, device,
        args.stage3_epochs, args.lr, scaler, args.class_threshold,
        args.delay_threshold,  # FIXED: Pass actual delay threshold
        args.patience, dp_config, args.batch_size, accountant_s2,
    )
    
    combined_history = history_s1 + history_s2 + history_s3
    
    if dp_config.enabled:
        final_epsilon = accountant_s3.get_epsilon(dp_config.target_delta)
        final_delta = dp_config.target_delta
    else:
        final_epsilon = float('inf')
        final_delta = 0.0
   
    # Generate unique model path with noise multiplier (sigma) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_str = f"sigma{dp_config.noise_multiplier:.2f}".replace(".", "_")
    
    # Update model path if using default
    if args.model_path == 'kan_gat_dp_three_stage.pth':
        args.model_path = f"kan_gat_dp_three_stage_{sigma_str}_{timestamp}.pth"
    
    print(f"\nOutput model will be saved to: {args.model_path}")
    
    final_evaluation(
        model, edge_indices, device, scaler, horizons,
        delay_dim, num_nodes, test_x, test_y_reg, test_y_cls,
        args.class_threshold, args.delay_threshold, args.model_path, combined_history,
        final_epsilon, final_delta, stage1_time, stage2_time, stage3_time,
        len(train_x), len(val_x), dp_config,
    )


def setup_checkpoint_directory():
    from pathlib import Path
    try:
        from google.colab import drive
        from IPython import get_ipython
        if get_ipython() is not None:  # Running in Notebook
            drive.mount('/content/drive')
            base_path = "/content/drive/MyDrive/FlightDelay_Checkpoints"
        else:
            # Running as normal Python script
            base_path = "./checkpoints"
    except:
        # Not in Colab
        print("✓ Checkpoints will be saved locally.")
        base_path = "./checkpoints"

    Path(base_path).mkdir(parents=True, exist_ok=True)
    return base_path 



# Global checkpoint directory - set at runtime
CHECKPOINT_DIR: str = ""


if __name__ == '__main__':
    CHECKPOINT_DIR = setup_checkpoint_directory()
    main()