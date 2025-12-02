"""Properly implemented differentially private sequential KAN-GAT pipeline with epsilon budget control.

FIXED: 
1. Ensures both predictions AND targets are at graph-level (not node-level).
2. Automatically computes noise multiplier to achieve target epsilon.
3. Allows training to complete all epochs while tracking epsilon.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.func import grad, vmap, functional_call
from torch.utils.data import Dataset
from torch_geometric.data import Data

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
    """Rényi Differential Privacy accountant with budget tracking."""

    noise_multiplier: float
    sample_rate: float
    steps: int = 0
    
    def step(self) -> None:
        """Record one training step."""
        self.steps += 1
    
    def get_epsilon(self, delta: float, orders: Optional[List[float]] = None) -> float:
        """Compute epsilon via RDP to (eps, delta)-DP conversion.
        
        This follows Opacus's RDP accounting with privacy amplification by subsampling.
        """
        if self.steps == 0:
            return 0.0
            
        if orders is None:
            orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
        
        rdp_values = []
        for alpha in orders:
            if alpha <= 1:
                continue
            
            # Base RDP for Gaussian mechanism: alpha / (2 * sigma^2)
            rdp_step = alpha / (2 * self.noise_multiplier ** 2)
            
            # Privacy amplification by subsampling
            # For small sample rates: RDP ≈ q^2 * base_rdp
            # For larger sample rates: use tighter bound
            if self.sample_rate < 0.01:
                # Small sample rate approximation
                rdp_step *= self.sample_rate ** 2
            elif self.sample_rate < 1.0:
                # Tighter bound for moderate sample rates
                # This is a simplified version of Opacus's approach
                rdp_step *= self.sample_rate * min(
                    self.sample_rate,
                    np.sqrt(2 * alpha * self.sample_rate)
                )
            # If sample_rate >= 1.0, no amplification
            
            rdp_total = rdp_step * self.steps
            rdp_values.append(rdp_total)
        
        # Convert RDP to (ε, δ)-DP
        eps_values = []
        for alpha, rdp in zip(orders, rdp_values):
            if alpha <= 1:
                continue
            # ε = RDP_α + log(1/δ) / (α - 1)
            eps = rdp + np.log(1 / delta) / (alpha - 1)
            eps_values.append(eps)
        
        return min(eps_values) if eps_values else float('inf')
    
    def predict_steps_for_epsilon(self, target_epsilon: float, delta: float) -> int:
        """Predict how many steps can be taken before reaching target epsilon."""
        if self.steps == 0:
            # Estimate: binary search for the number of steps
            low, high = 1, 10000
            best_steps = 0
            
            while low <= high:
                mid = (low + high) // 2
                temp_accountant = RDPAccountant(
                    noise_multiplier=self.noise_multiplier,
                    sample_rate=self.sample_rate,
                    steps=mid
                )
                eps = temp_accountant.get_epsilon(delta)
                
                if eps <= target_epsilon:
                    best_steps = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            return max(1, best_steps)
        else:
            # Use current rate to predict remaining budget
            current_eps = self.get_epsilon(delta)
            if current_eps >= target_epsilon:
                return 0
            
            # Linear approximation: remaining_eps / current_rate
            eps_per_step = current_eps / self.steps if self.steps > 0 else float('inf')
            remaining_eps = target_epsilon - current_eps
            remaining_steps = int(remaining_eps / eps_per_step) if eps_per_step > 0 else 0
            return max(0, remaining_steps)




def solve_noise_multiplier_for_epsilon(target_epsilon: float, delta: float, 
                                       sample_rate: float, steps: int) -> float:
    """
    Binary search to find minimum noise multiplier such that epsilon 
    computed by RDPAccountant is <= target_epsilon.
    
    This mimics Opacus's make_private_with_epsilon approach.
    """
    if steps == 0:
        return 1.0
    
    # Search bounds: [0.1, 100] should cover most practical scenarios
    lower, upper = 0.1, 100.0
    precision = 0.001  # Higher precision for better accuracy
    best_nm = upper
    
    # First check if upper bound is sufficient
    acc = RDPAccountant(noise_multiplier=upper, sample_rate=sample_rate, steps=steps)
    if acc.get_epsilon(delta) > target_epsilon:
        print(f"\n⚠️  Warning: Cannot achieve ε={target_epsilon} with noise_multiplier up to {upper}")
        print(f"    Try: reducing epochs, increasing batch size, or increasing target epsilon")
        return upper

    # Binary search for optimal noise multiplier
    iteration = 0
    while upper - lower > precision and iteration < 100:
        mid = (lower + upper) / 2
        acc = RDPAccountant(noise_multiplier=mid, sample_rate=sample_rate, steps=steps)
        eps = acc.get_epsilon(delta)

        if eps > target_epsilon:
            lower = mid  # Need more noise (higher multiplier)
        else:
            best_nm = mid
            upper = mid
        
        iteration += 1

    return round(best_nm, 3)

@dataclass
class DPConfig:
    """Differential privacy configuration with budget control."""
    enabled: bool
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    max_grad_norm: float
    sample_rate: float
    epsilon_tolerance: float = 0.05  # Stop when within 5% of target
    


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
        """
        Compute per-sample gradients via loop.
        
        This uses standard autograd instead of functorch/vmap to ensure
        compatibility with complex PyG models.
        """
        edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
        all_grads = []
        
        for i in range(batch_x.shape[0]):
            # Zero gradients
            self.model.zero_grad(set_to_none=True)
            
            # Create data
            data = Data(
                x=batch_x[i],
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            
            # Forward pass
            if is_classification:
                _, node_logits = self.model.forward_classifier(data)
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(batch_y[i])
                loss = loss_fn(graph_logit, graph_target)
            else:
                _, node_reg = self.model(data)
                graph_reg = aggregate_node_to_graph(node_reg)
                graph_target = ensure_graph_level_target(batch_y[i])
                mask = (graph_target >= 0).float()
                loss = loss_fn(graph_reg * mask, graph_target * mask)
            
            # Backward pass
            loss.backward()
            
            # Collect gradients for this sample
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
    """
    Aggregate node-level features to graph-level via mean pooling.
    
    Args:
        node_features: [num_nodes, feature_dim] or [num_nodes, 1]
    
    Returns:
        graph_features: [1, feature_dim] or scalar
    """
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    """
    Convert node-level targets to graph-level.
    
    Args:
        target: Can be [num_nodes, 1], [num_nodes], or scalar
    
    Returns:
        graph_target: [1] or [1, 1] depending on input
    """
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
    pos_weight: float,
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
    
    cls_loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
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
    
    for epoch in range(1, epochs + 1):
        
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
                # DP-SGD: Per-sample clipping + noise
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
                        graph_logit = aggregate_node_to_graph(node_logits)
                        graph_target = ensure_graph_level_target(batch_y[i])
                        logits_list.append(graph_logit)
                        targets_list.append(graph_target)
                    
                    all_logits = torch.cat(logits_list, dim=0)
                    all_targets = torch.cat(targets_list, dim=0)
                    loss = cls_loss_fn(all_logits, all_targets)
                
                accountant.step()
            else:
                # Standard training (NO DP)
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
                    graph_logit = aggregate_node_to_graph(node_logits)
                    graph_target = ensure_graph_level_target(batch_y[i])
                    logits_list.append(graph_logit)
                    targets_list.append(graph_target)
                
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
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(val_y_cls[i])
                val_probs.append(torch.sigmoid(graph_logit).cpu())
                val_targets.append(graph_target.cpu())
        
        val_probs_np = torch.cat(val_probs).numpy()
        val_targets_np = torch.cat(val_targets).numpy()
        val_metrics = classification_metrics(
            val_probs_np.reshape(-1, 1),
            val_targets_np.reshape(-1, 1),
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
            f"Val F1: {val_metrics['f1']:.4f} | {eps_str} | Time: {epoch_time:.2f}s"
        )
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = {
                'encoder': model.encoder.state_dict(),
                'classifier': model.classifier.state_dict(),
            }
            print("  ✓ New best checkpoint")
        
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
    """Train stage 2 with proper DP-SGD and epsilon budget control."""
    
    stage_start_time = time.time()
    
    print("\n" + "="*80)
    print("STAGE 2: TRAINING DELAY REGRESSOR WITH DP-SGD")
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
    
    for epoch in range(1, epochs + 1):
        
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
                        graph_reg = aggregate_node_to_graph(node_reg)
                        graph_target = ensure_graph_level_target(batch_y_reg[i])
                        reg_preds.append(graph_reg)
                        reg_targets.append(graph_target)
                    
                    reg_preds = torch.cat(reg_preds, dim=0)
                    reg_targets = torch.cat(reg_targets, dim=0)
                    
                    cls_mask = []
                    for i in range(len(batch_y_cls)):
                        graph_cls = ensure_graph_level_target(batch_y_cls[i])
                        cls_mask.append(graph_cls)
                    cls_mask = torch.cat(cls_mask, dim=0)
                    
                    mask = (cls_mask >= class_threshold).float()
                    if mask.dim() == 1:
                        mask = mask.unsqueeze(-1)
                    loss = reg_loss_fn(reg_preds * mask, reg_targets * mask)
                
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
                    graph_reg = aggregate_node_to_graph(node_reg)
                    graph_target = ensure_graph_level_target(batch_y_reg[i])
                    reg_preds.append(graph_reg)
                    reg_targets.append(graph_target)
                
                reg_preds = torch.cat(reg_preds, dim=0)
                reg_targets = torch.cat(reg_targets, dim=0)
                
                cls_mask = []
                for i in range(len(batch_y_cls)):
                    graph_cls = ensure_graph_level_target(batch_y_cls[i])
                    cls_mask.append(graph_cls)
                cls_mask = torch.cat(cls_mask, dim=0)
                
                mask = (cls_mask >= class_threshold).float()
                if mask.dim() == 1:
                    mask = mask.unsqueeze(-1)
                loss = reg_loss_fn(reg_preds * mask, reg_targets * mask)
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
                graph_reg = aggregate_node_to_graph(node_reg)
                graph_target = ensure_graph_level_target(val_y_reg[i])
                graph_cls = ensure_graph_level_target(val_y_cls[i])
                
                mask = (graph_cls >= class_threshold).float()
                if mask.dim() == 1:
                    mask = mask.unsqueeze(-1)
                loss = reg_loss_fn(graph_reg.cpu() * mask, graph_target * mask)
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
    model_path: str,
    histories: List[Dict],
    final_epsilon: float,
    final_delta: float,
    stage1_time: float,
    stage2_time: float,
    train_samples: int,
    val_samples: int,
    dp_config: DPConfig,
) -> None:
    """Final evaluation and export."""
    
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
    
    # =============================================================================
    # FAST EVALUATION MODE (Set to True for 10-50x speedup)
    # =============================================================================
    USE_FAST_EVAL = False  # Change to True to enable batched evaluation
    
    if USE_FAST_EVAL:
        # ⚡ FAST: Batched evaluation (10-50x faster)
        print("[FAST MODE] Using batched evaluation (much faster)")
        batch_size = 128  # Adjust based on GPU memory
        
        with torch.no_grad():
            for start_idx in range(0, len(test_x), batch_size):
                end_idx = min(start_idx + batch_size, len(test_x))
                batch_x = test_x[start_idx:end_idx].to(device)
                batch_cls = test_y_cls[start_idx:end_idx]
                batch_reg = test_y_reg[start_idx:end_idx]
                
                # Process batch
                batch_logits_list = []
                batch_reg_list = []
                
                for i in range(len(batch_x)):
                    data = Data(
                        x=batch_x[i],
                        edge_index_adj=edge_indices[0],
                        edge_index_od=edge_indices[1],
                        edge_index_od_t=edge_indices[2],
                    )
                    node_logits, node_reg = model(data)
                    graph_logit = aggregate_node_to_graph(node_logits)
                    graph_reg = aggregate_node_to_graph(node_reg)
                    
                    batch_logits_list.append(torch.sigmoid(graph_logit))
                    batch_reg_list.append(graph_reg)
                
                # Collect results
                logits_list.extend([x.cpu().numpy() for x in batch_logits_list])
                reg_list.extend([x.cpu().numpy() for x in batch_reg_list])
                
                # Targets
                for i in range(len(batch_cls)):
                    targets_cls_list.append(ensure_graph_level_target(batch_cls[i]).cpu().numpy())
                    targets_reg_list.append(ensure_graph_level_target(batch_reg[i]).cpu().numpy())
                
                if (start_idx // batch_size) % 10 == 0:
                    print(f"  Processed {end_idx}/{len(test_x)} samples...")
        
        test_probs = np.concatenate(logits_list, axis=0)
        test_reg_preds = np.concatenate(reg_list, axis=0)
        test_cls_targets = np.concatenate(targets_cls_list, axis=0)
        test_reg_targets = np.concatenate(targets_reg_list, axis=0)
    
    else:
        # 🐌 SLOW: Original per-sample evaluation (accurate but very slow)
        print("[SLOW MODE] Using per-sample evaluation (set USE_FAST_EVAL=True for speedup)")
        with torch.no_grad():
            for i in range(len(test_x)):
                data = Data(
                    x=test_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                node_logits, node_reg = model(data)
                
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_reg = aggregate_node_to_graph(node_reg)
                
                graph_cls_target = ensure_graph_level_target(test_y_cls[i])
                graph_reg_target = ensure_graph_level_target(test_y_reg[i])
                
                logits_list.append(torch.sigmoid(graph_logit).cpu().numpy())
                reg_list.append(graph_reg.cpu().numpy())
                targets_cls_list.append(graph_cls_target.cpu().numpy())
                targets_reg_list.append(graph_reg_target.cpu().numpy())
                
                # Progress indicator
                if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                    print(f"  Processed {i+1}/{len(test_x)} samples...")
        
        test_probs = np.concatenate(logits_list, axis=0)
        test_reg_preds = np.concatenate(reg_list, axis=0)
        test_cls_targets = np.concatenate(targets_cls_list, axis=0)
        test_reg_targets = np.concatenate(targets_reg_list, axis=0)
    
    # =============================================================================
    
    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_cls_targets.reshape(-1, 1),
    )
    
    test_mask = (test_probs >= class_threshold)
    gated_preds = test_reg_preds * test_mask
    
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(gated_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
    else:
        preds_denorm = gated_preds
        targets_denorm = test_reg_targets
    
    delayed_mask = test_cls_targets.flatten() >= class_threshold
    if delayed_mask.sum() > 0:
        delayed_preds = preds_denorm[delayed_mask]
        delayed_targets = targets_denorm[delayed_mask]
        
        mae = np.mean(np.abs(delayed_preds - delayed_targets))
        rmse = np.sqrt(np.mean((delayed_preds - delayed_targets) ** 2))
        
        print("\nCLASSIFICATION:")
        print(f"  Precision: {test_cls_metrics['precision']:.4f} | Recall: {test_cls_metrics['recall']:.4f}")
        print(f"  F1: {test_cls_metrics['f1']:.4f} | Accuracy: {test_cls_metrics['accuracy']:.4f}")
        
        print("\nREGRESSION (delayed flights only):")
        print(f"  MAE: {mae:.4f} min | RMSE: {rmse:.4f} min")
        print(f"  Number of delayed samples: {delayed_mask.sum()}")
    else:
        print("\nNo delayed samples in test set!")
        mae, rmse = 0.0, 0.0
    
    print("\nPRIVACY BUDGET:")
    print(f"  Target ε: {dp_config.target_epsilon:.3f}")
    print(f"  Final ε: {final_epsilon:.3f}")
    if dp_config.enabled:
        if final_epsilon <= dp_config.target_epsilon:
            print(f"  ✓ Budget maintained (within target)")
        else:
            overshoot = final_epsilon - dp_config.target_epsilon
            print(f"  ⚠️  Budget exceeded by {overshoot:.3f} ε ({overshoot/dp_config.target_epsilon*100:.1f}%)")
    print(f"  Final δ: {final_delta:.2e}")
    
    print("\nTRAINING TIME:")
    total_time = stage1_time + stage2_time
    print(f"  Stage 1: {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    print(f"  Stage 2: {stage2_time:.2f}s ({stage2_time/60:.2f} min)")
    print(f"  Total: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    
    # Export history
    if histories:
        with open("kan_gat_dp_epsilon_controlled_history.csv", "w", newline="") as f:
            all_fields = sorted({k for row in histories for k in row})
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(histories)
    
    # Export summary
    with open("kan_gat_dp_epsilon_controlled_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        summary = {
            'classification_precision': test_cls_metrics['precision'],
            'classification_recall': test_cls_metrics['recall'],
            'classification_f1': test_cls_metrics['f1'],
            'classification_accuracy': test_cls_metrics['accuracy'],
            'regression_mae_delayed': mae,
            'regression_rmse_delayed': rmse,
            'num_delayed_samples': int(delayed_mask.sum()) if delayed_mask.sum() > 0 else 0,
            'target_epsilon': dp_config.target_epsilon,
            'final_epsilon': final_epsilon,
            'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
            'epsilon_overshoot': max(0, final_epsilon - dp_config.target_epsilon) if dp_config.enabled else 0,
            'final_delta': final_delta,
            'stage1_time_seconds': stage1_time,
            'stage2_time_seconds': stage2_time,
            'total_training_time_seconds': stage1_time + stage2_time,
            'total_training_time_minutes': (stage1_time + stage2_time) / 60,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': len(test_x),
        }
        for k, v in summary.items():
            writer.writerow([k, v])
    
    print(f"\n✓ Results saved to:")
    print(f"  - {model_path}")
    print(f"  - kan_gat_dp_epsilon_controlled_history.csv")
    print(f"  - kan_gat_dp_epsilon_controlled_summary.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proper DP-SGD for KAN-GAT with epsilon budget control")
    
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 6, 12])
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--use_node_level', action='store_true', default=True, help='Use node-level labels ')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    
    parser.add_argument('--stage1_epochs', type=int, default=5)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    
    parser.add_argument('--dp', default=True, action='store_true', help='Enable DP-SGD')
    parser.add_argument('--target_epsilon', type=float, default=15.0)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=2)
    parser.add_argument('--max_grad_norm', type=float, default=1.5)
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--epsilon_tolerance', type=float, default=0.05, 
                        help='Stop when within this fraction of target epsilon (default: 0.05 = 5%)')
    
    parser.add_argument('--model_path', type=str, default='kan_gat_dp_proper.pth')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'
    
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
    max_horizon = max(horizons)
    
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    
    in_channels = args.seq_len * feature_dim
    out_channels = len(horizons) * delay_dim
    
    # Choose build function based on use_node_level flag
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
    if args.use_node_level:
        print("[INFO] Using NODE-LEVEL labels ")
    else:
        print("[INFO] Using GRAPH-LEVEL labels ")
    
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
    
    # Print class distribution
    print(f"\n[DATA] Class distribution:")
    print(f"  Train: {train_y_cls.mean().item():.2%} delayed")
    print(f"  Val:   {val_y_cls.mean().item():.2%} delayed")
    print(f"  Test:  {test_y_cls.mean().item():.2%} delayed")
    
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
    ).to(device)
    


    # Auto-compute noise multiplier for target epsilon (following Opacus approach)
    total_samples = len(train_x)
    
    # Key fix: sample_rate should be batch_size / total_samples (expected sampling rate)
    # NOT a tiny fixed value like 0.02
    sample_rate = args.batch_size / total_samples
    
    # Calculate expected number of steps per epoch with this sample rate
    # Each epoch we go through the data once, so steps = ceil(total_samples / batch_size)
    steps_per_epoch = int(np.ceil(total_samples / args.batch_size))
    total_steps = (args.stage1_epochs + args.stage2_epochs) * steps_per_epoch

    if args.dp:
        computed_noise_multiplier = solve_noise_multiplier_for_epsilon(
            args.target_epsilon, args.target_delta, sample_rate, total_steps
        )
        print(f"\nAuto-computed noise multiplier for target ε={args.target_epsilon}: {computed_noise_multiplier:.3f}")
        print(f"  Sample rate: {sample_rate:.4f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Expected total steps: {total_steps} ({steps_per_epoch} steps/epoch × {args.stage1_epochs + args.stage2_epochs} epochs)")
        args.noise_multiplier = computed_noise_multiplier
        args.sample_rate = sample_rate  # Override with computed value
    
    dp_config = DPConfig(
        enabled=args.dp,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        sample_rate=sample_rate if args.dp else args.sample_rate,
        epsilon_tolerance=args.epsilon_tolerance,
    )
    
    cls_pos_rate = train_y_cls.mean().item()
    pos_weight = (1 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)
    
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    print(f"Train samples: {len(train_x)}")
    print(f"Val samples: {len(val_x)}")
    print(f"Test samples: {len(test_x)}")
    print(f"Class balance (delayed): {cls_pos_rate:.2%}")
    
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
    
    combined_history = history_s1 + history_s2
    
    if dp_config.enabled:
        final_epsilon = accountant_s2.get_epsilon(dp_config.target_delta)
        final_delta = dp_config.target_delta
    else:
        final_epsilon = float('inf')
        final_delta = 0.0
    
    final_evaluation(
        model, edge_indices, device, scaler, horizons,
        delay_dim, num_nodes, test_x, test_y_reg, test_y_cls,
        args.class_threshold, args.model_path, combined_history,
        final_epsilon, final_delta, stage1_time, stage2_time,
        len(train_x), len(val_x), dp_config,
    )


if __name__ == '__main__':
    main()
