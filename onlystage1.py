"""Differentially private Stage 1 KAN-GAT classifier only.

This version only trains and tests Stage 1 (classification).
Stage 2 and Stage 3 (regression) have been removed.
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
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(batch_y[i])
                loss = loss_fn(graph_logit, graph_target)
            else:
                _, node_reg = self.model(data)
                graph_reg = aggregate_node_to_graph(node_reg)
                graph_target = ensure_graph_level_target(batch_y[i])
                mask = (graph_target >= 0).float()
                loss = loss_fn(graph_reg * mask, graph_target * mask)
            
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
                        graph_logit = aggregate_node_to_graph(node_logits)
                        graph_target = ensure_graph_level_target(batch_y[i])
                        logits_list.append(graph_logit)
                        targets_list.append(graph_target)
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


def final_evaluation(

    model: SequentialTwoStagePredictor,
    edge_indices: Tuple,
    device: torch.device,
    test_x: torch.Tensor,
    test_y_cls: torch.Tensor,
    class_threshold: float,
    model_path: str,
    histories: List[Dict],
    final_epsilon: float,
    final_delta: float,
    stage1_time: float,
    train_samples: int,
    val_samples: int,
    dp_config: DPConfig,
) -> None:
    """Final evaluation and export for Stage 1 classification only."""
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION - STAGE 1 (CLASSIFICATION ONLY)")
    print("="*80)
    print(f"Test samples: {len(test_x)}")
    
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    torch.save({
        'encoder': model.encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        'final_epsilon': float(final_epsilon),
        'final_delta': float(final_delta),
        'target_epsilon': float(dp_config.target_epsilon),
        'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
    }, model_path)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    
    model.eval()
    
    logits_list = []
    targets_cls_list = []
    
    print("[EVALUATION] Processing test samples...")
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )
            
            _, node_logits = model.forward_classifier(data)
            graph_logit = aggregate_node_to_graph(node_logits)
            
            graph_cls_target = ensure_graph_level_target(test_y_cls[i])
            
            logits_list.append(torch.sigmoid(graph_logit).cpu().numpy())
            targets_cls_list.append(graph_cls_target.cpu().numpy())
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")
    
    test_probs = np.concatenate(logits_list, axis=0)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)
    
    # Classification metrics
    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_cls_targets.reshape(-1, 1),
    )
    
    print("\nCLASSIFICATION RESULTS:")
    print(f"  Precision: {test_cls_metrics['precision']:.4f}")
    print(f"  Recall: {test_cls_metrics['recall']:.4f}")
    print(f"  F1: {test_cls_metrics['f1']:.4f}")
    print(f"  Accuracy: {test_cls_metrics['accuracy']:.4f}")
    
    # Visualize classification results
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating classification results plots...")
        try:
            # Convert predictions to binary labels
            test_cls_pred = (test_probs >= class_threshold).astype(int).flatten()
            # Binarize targets (since they might be soft labels from graph aggregation)
            test_cls_true = (test_cls_targets >= 0.5).astype(int).flatten()
            
            # Create dummy regression values based on class (just for visualization)
            test_reg_dummy_true = test_cls_true * class_threshold * 2
            test_reg_dummy_pred = test_cls_pred * class_threshold * 2
            
            visualize_classification_results(
                test_cls_true,
                test_cls_pred,
                test_reg_dummy_true,
                test_reg_dummy_pred,
                threshold=class_threshold,
                save_path=f"stage1_classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            print("  ✓ Stage 1 classification results visualization saved")
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
    print(f"  Stage 1: {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    
    # Generate unique filenames with noise multiplier (sigma) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_str = f"sigma{dp_config.noise_multiplier:.2f}".replace(".", "_")
    
    history_csv = f"kan_gat_dp_stage1_history_{sigma_str}_{timestamp}.csv"
    summary_csv = f"kan_gat_dp_stage1_summary_{sigma_str}_{timestamp}.csv"
    
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
            'target_epsilon': dp_config.target_epsilon,
            'final_epsilon': final_epsilon,
            'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
            'epsilon_overshoot': max(0, final_epsilon - dp_config.target_epsilon) if dp_config.enabled else 0,
            'final_delta': final_delta,
            'stage1_time_seconds': stage1_time,
            'stage1_time_minutes': stage1_time / 60,
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
        
        # Add checkpoint file if it exists
        checkpoint_file = os.path.join(CHECKPOINT_DIR, 'stage1_checkpoint.pth')
        files_to_download.append(checkpoint_file)
        
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
    parser = argparse.ArgumentParser(description="Stage 1 DP-SGD for KAN-GAT classification with epsilon budget control")
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
    parser.add_argument('--stage1_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dp', default=True, action='store_true', help='Enable DP-SGD')
    parser.add_argument('--target_epsilon', type=float, default=15.0, help='Target epsilon for tracking (not used for computing noise)')
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=2.0, help='Fixed noise multiplier for DP-SGD (not auto-computed)')
    parser.add_argument('--max_grad_norm', type=float, default=1.5)
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--epsilon_tolerance', type=float, default=0.05)
    parser.add_argument('--model_path', type=str, default='kan_gat_dp_stage1.pth')
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
    in_channels = args.seq_len * feature_dim
    out_channels = delay_dim
    
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
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
    
    # Visualize training data distribution
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating training data distribution plots...")
        try:
            # Prepare data for visualization (convert to numpy)
            train_x_viz = train_x.cpu().numpy() if isinstance(train_x, torch.Tensor) else train_x
            train_y_cls_viz = train_y_cls.cpu().numpy() if isinstance(train_y_cls, torch.Tensor) else train_y_cls
            
            # For stage 1 only, we don't need regression values for visualization
            # Just use classification labels as "delay values" for the plots
            train_y_cls_flat = train_y_cls_viz.mean(axis=(1, 2)) if train_y_cls_viz.ndim > 1 else train_y_cls_viz
            # Create dummy regression values based on class (just for visualization)
            train_y_reg_dummy = train_y_cls_flat * args.delay_threshold * 2
            
            visualize_training_data(
                train_x_viz,
                train_y_cls_flat,
                train_y_reg_dummy,
                threshold=args.delay_threshold,
                sample_size=min(2000, len(train_x_viz)),
                save_path=f"stage1_training_data_visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            print("  ✓ Stage 1 training data visualization saved")
        except Exception as e:
            print(f"  ✗ Error generating training data visualization: {e}")
    
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
    
    total_samples = len(train_x)
    sample_rate = args.batch_size / total_samples
    steps_per_epoch = int(np.ceil(total_samples / args.batch_size))
    total_steps = args.stage1_epochs * steps_per_epoch
    
    if args.dp:
        print(f"\nUsing FIXED noise multiplier: {args.noise_multiplier:.3f}")
        print(f"  Sample rate: {sample_rate:.4f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Expected total steps: {total_steps} ({steps_per_epoch} steps/epoch × {args.stage1_epochs} epochs)")
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
    
    if dp_config.enabled:
        final_epsilon = accountant_s1.get_epsilon(dp_config.target_delta)
        final_delta = dp_config.target_delta
    else:
        final_epsilon = float('inf')
        final_delta = 0.0
   
    # Generate unique model path with noise multiplier (sigma) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_str = f"sigma{dp_config.noise_multiplier:.2f}".replace(".", "_")
    
    # Update model path if using default
    if args.model_path == 'kan_gat_dp_stage1.pth':
        args.model_path = f"kan_gat_dp_stage1_{sigma_str}_{timestamp}.pth"
    
    print(f"\nOutput model will be saved to: {args.model_path}")
    
    final_evaluation(
        model, edge_indices, device,
        test_x, test_y_cls, args.class_threshold, args.model_path, history_s1,
        final_epsilon, final_delta, stage1_time,
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
