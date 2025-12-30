"""
Compare different undersampling/oversampling methods for Stage 1 Classifier.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data
from torch.utils.data import Dataset

try:
    # Optional dependency for --mode imblearn
    from imblearn.over_sampling import ADASYN, RandomOverSampler, SMOTE, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTEENN, SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score, make_scorer
except Exception as exc:
    print("[WARN] Failed to import imblearn/sklearn dependencies:", exc)
    traceback.print_exc()
    ADASYN = RandomOverSampler = SMOTE = BorderlineSMOTE = None
    RandomUnderSampler = TomekLinks = None
    SMOTEENN = SMOTETomek = None
    ImbPipeline = None
    LogisticRegression = None
    GridSearchCV = StratifiedKFold = None
    StandardScaler = None
    f1_score = make_scorer = None

# Reuse original implementation
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (
    EarlyStopping,
    SequentialTwoStagePredictor,
    build_sequences,
    classification_metrics,
    train_stage1_classifier,
    load_flight_data,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level
# Inlined utilities (originally from threestagev3debug2)

@dataclass
class RDPAccountant:
    """Opacus-style RDP accountant with Poisson subsampling."""
    noise_multiplier: float
    sample_rate: float
    steps: int = 0

    def step(self) -> None:
        self.steps += 1

    def _rdp_gaussian(self, alpha: float, sigma: float) -> float:
        if alpha < 1:
            return 0.0
        return alpha * (1 + 1 / (2 * sigma**2 * alpha)) * np.log1p(1 / (sigma**2 * alpha))

    def _rdp_subsampling(self, alpha: float, q: float, rdp_full: float) -> float:
        if q == 0 or q == 1:
            return rdp_full
        return (np.exp((alpha - 1) * np.log1p(q)) - 1) * rdp_full

    def get_epsilon(self, delta: float, orders: Optional[List[float]] = None) -> float:
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


@dataclass
class DPConfig:
    enabled: bool
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    max_grad_norm: float
    sample_rate: float
    epsilon_tolerance: float = 0.05


class PerSampleGradientClipper:
    """Per-sample gradient clipping without functorch."""

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
                target = batch_y[i]
                loss = loss_fn(node_logits, target)
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

            grad_norm = torch.sqrt(sum(torch.sum(g ** 2) for g in sample_grads.values()))
            clip_coef = min(1.0, self.max_grad_norm / (grad_norm + 1e-10))
            clipped_grads = {k: v * clip_coef for k, v in sample_grads.items()}
            all_grads.append(clipped_grads)

        avg_grads = {}
        for key in all_grads[0].keys():
            avg_grads[key] = torch.mean(torch.stack([g[key] for g in all_grads]), dim=0)

        return avg_grads

    def add_noise_to_gradients(
        self,
        gradients: Dict[str, torch.Tensor],
        noise_multiplier: float,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        noisy_grads = {}
        noise_scale = noise_multiplier * self.max_grad_norm / batch_size

        for key, grad in gradients.items():
            noise = torch.normal(mean=0.0, std=noise_scale, size=grad.shape, device=grad.device)
            noisy_grads[key] = grad + noise

        return noisy_grads


def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 0:
        return target.unsqueeze(0)
    if target.dim() == 1:
        return target.mean(dim=0, keepdim=True)
    return target.mean(dim=0, keepdim=True)


def graph_level_binary_labels(y_cls: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if y_cls.dim() == 1:
        graph_vals = y_cls
    else:
        reduce_dims = tuple(range(1, y_cls.dim()))
        graph_vals = y_cls.mean(dim=reduce_dims)
    return (graph_vals >= threshold).float()


def build_balanced_indices(
    labels: torch.Tensor,
    desired_pos_fraction: float,
    total_samples: Optional[int] = None,
    threshold: float = 0.5,
) -> torch.Tensor:
    labels_cpu = labels.detach().cpu()
    total_samples = total_samples or labels_cpu.shape[0]
    desired_pos_fraction = float(np.clip(desired_pos_fraction, 1e-3, 0.5))

    graph_labels = graph_level_binary_labels(labels_cpu, threshold)
    pos_idx = torch.nonzero(graph_labels >= 0.5, as_tuple=False).view(-1)
    neg_idx = torch.nonzero(graph_labels < 0.5, as_tuple=False).view(-1)

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return torch.randperm(labels_cpu.shape[0])

    pos_target = max(1, int(round(total_samples * desired_pos_fraction)))
    neg_target = max(1, total_samples - pos_target)

    pos_pool = pos_idx[torch.randint(len(pos_idx), (pos_target,))]
    neg_pool = neg_idx[torch.randint(len(neg_idx), (neg_target,))]

    combined = torch.cat([pos_pool, neg_pool], dim=0)
    return combined[torch.randperm(len(combined))]


def setup_checkpoint_directory() -> str:
    try:
        from google.colab import drive
        from IPython import get_ipython
        if get_ipython() is not None:
            drive.mount('/content/drive')
            base_path = "/content/drive/MyDrive/FlightDelay_Checkpoints"
        else:
            base_path = "./checkpoints"
    except Exception:
        print("Checkpoints will be saved locally.")
        base_path = "./checkpoints"

    Path(base_path).mkdir(parents=True, exist_ok=True)
    return base_path

# Global checkpoint directory
CHECKPOINT_DIR = ""

def train_stage1_comparison(
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
    sampling_config: Dict,
    class_threshold: float,
) -> Dict:
    """Train stage 1 with specific sampling configuration."""
    
    # Reset model parameters to ensure fair comparison
    # (In a real scenario we might re-init, but here we assume model is passed fresh or we re-init)
    # For this script, we will re-initialize the model outside or just accept it's a new run.
    
    print(f"\nTraining with config: {sampling_config}")
    
    # Freeze regressor
    for param in model.regressor.parameters():
        param.requires_grad = False
    
    trainable_params = list(model.encoder.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    
    # Adjust pos_weight based on sampling?
    # If we balance the batch 50/50, pos_weight should probably be 1.0 (no weighting needed).
    # If we use baseline, we use the calculated pos_weight.
    
    if sampling_config['method'] == 'baseline':
        current_pos_weight = torch.tensor([pos_weight], device=device)
    else:
        # If we are balancing, we might not need strong pos_weight, or we might want to keep it.
        # Usually if we balance 50/50, we remove the weight.
        # Let's assume for 'balanced' methods we set weight to 1.0, unless specified.
        current_pos_weight = torch.tensor([1.0], device=device)

    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=current_pos_weight)
    
    accountant = RDPAccountant(
        noise_multiplier=dp_config.noise_multiplier,
        sample_rate=dp_config.sample_rate if dp_config.enabled else 1.0,
    )
    
    if dp_config.enabled:
        clipper = PerSampleGradientClipper(model, dp_config.max_grad_norm)
    
    best_f1 = 0.0
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="max")
    
    # Calculate sampling parameters once
    num_total = len(train_y_cls)
    # Use graph-level labels for counting to match build_balanced_indices logic
    train_y_graph = graph_level_binary_labels(train_y_cls, class_threshold)
    num_pos = (train_y_graph >= 0.5).sum().item()
    num_neg = num_total - num_pos
    
    if sampling_config['method'] == 'random_under':
        # Undersample majority to match minority * ratio
        # e.g. ratio 0.5 -> pos = 0.5 * total -> pos = neg
        # total = pos / ratio? No.
        # If ratio is 0.5 (50% pos), then we want pos_count == neg_count.
        # So we limit neg_count to num_pos. Total = 2 * num_pos.
        target_ratio = sampling_config['ratio']
        # pos / (pos + neg') = ratio
        # pos = ratio * (pos + neg')
        # pos = ratio*pos + ratio*neg'
        # pos * (1 - ratio) = ratio * neg'
        # neg' = pos * (1 - ratio) / ratio
        
        target_neg = int(num_pos * (1 - target_ratio) / target_ratio)
        epoch_samples = num_pos + target_neg
        desired_pos_fraction = target_ratio
        
    elif sampling_config['method'] == 'random_over':
        # Oversample minority
        target_ratio = sampling_config['ratio']
        # pos' / (pos' + neg) = ratio
        # pos' = ratio * (pos' + neg)
        # pos' * (1 - ratio) = ratio * neg
        # pos' = neg * ratio / (1 - ratio)
        
        target_pos = int(num_neg * target_ratio / (1 - target_ratio))
        epoch_samples = target_pos + num_neg
        desired_pos_fraction = target_ratio
        
    else: # baseline
        epoch_samples = num_total
        desired_pos_fraction = num_pos / num_total

    print(f"  Epoch samples: {epoch_samples} (Original: {num_total})")
    print(f"  Target Pos Fraction: {desired_pos_fraction:.2%}")
    print(f"  Pos Weight: {current_pos_weight.item():.2f}")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        
        # Build indices for this epoch
        if sampling_config['method'] == 'baseline':
            indices = torch.randperm(num_total)
        else:
            indices = build_balanced_indices(
                train_y_cls,
                desired_pos_fraction,
                total_samples=epoch_samples,
                threshold=class_threshold,
            )
            
        for start_idx in range(0, len(indices), batch_size):
            batch_indices = indices[start_idx : start_idx + batch_size]
            batch_x = train_x[batch_indices].to(device)
            batch_y = train_y_cls[batch_indices].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if dp_config.enabled:
                per_sample_grads = clipper.compute_per_sample_gradients(
                    batch_x, batch_y, edge_indices, cls_loss_fn, is_classification=True
                )
                noisy_grads = clipper.add_noise_to_gradients(
                    per_sample_grads, dp_config.noise_multiplier, len(batch_indices)
                )
                for name, param in model.named_parameters():
                    if param.requires_grad and name in noisy_grads:
                        param.grad = noisy_grads[name]
                
                # Forward for loss
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
                        logits_list.append(node_logits)
                        targets_list.append(batch_y[i])
                    all_logits = torch.cat(logits_list, dim=0)
                    all_targets = torch.cat(targets_list, dim=0)
                    loss = cls_loss_fn(all_logits, all_targets)
                
                accountant.step()
            else:
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
                val_probs.append(torch.sigmoid(node_logits).cpu())
                val_targets.append(val_y_cls[i].cpu())
        
        val_probs_np = torch.cat(val_probs).numpy()
        val_targets_np = torch.cat(val_targets).numpy()
        val_metrics = classification_metrics(
            val_probs_np.reshape(-1, 1),
            val_targets_np.reshape(-1, 1),
        )
        
        print(f"  Epoch {epoch} | Loss: {np.mean(epoch_losses):.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = {
                'encoder': model.encoder.state_dict(),
                'classifier': model.classifier.state_dict(),
            }
        
        if early_stopping(val_metrics['f1'], epoch):
            break
            
    return {
        'config': sampling_config,
        'best_f1': best_f1,
        'best_state': best_state,
        'val_metrics': val_metrics # Last epoch metrics
    }


def _require_imblearn() -> None:
    if ImbPipeline is None:
        raise RuntimeError(
            "imblearn/scikit-learn are required for --mode imblearn. "
            "Install with: pip install imbalanced-learn scikit-learn"
        )


def _to_graph_level_tabular(
    x_seq: torch.Tensor,
    y_node: torch.Tensor,
    class_threshold: float,
    agg: str = "mean_std",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (num_samples, num_nodes, feat) -> tabular (num_samples, feat*) and node labels -> graph labels."""
    # Aggregate nodes to graph features. This is only used for --mode imblearn.
    if agg == "mean":
        x_graph = x_seq.mean(dim=1)
    elif agg == "mean_std":
        mean = x_seq.mean(dim=1)
        std = x_seq.std(dim=1, unbiased=False)
        x_graph = torch.cat([mean, std], dim=1)
    else:
        raise ValueError(f"Unknown aggregation: {agg}. Use 'mean' or 'mean_std'.")
    y_graph = graph_level_binary_labels(y_node, class_threshold)

    x_np = x_graph.detach().cpu().numpy().astype(np.float32)
    y_np = y_graph.detach().cpu().numpy().reshape(-1).astype(np.int64)
    return x_np, y_np


@dataclass(frozen=True)
class ImblearnMethod:
    name: str
    sampler: object
    param_grid: Dict[str, List]


def run_imblearn_comparison(
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    test_x: torch.Tensor,
    test_y_cls: torch.Tensor,
    class_threshold: float,
    cv_folds: int,
    random_state: int,
    agg: str,
) -> List[Dict]:
    """Compare imbalanced-learn sampling techniques with per-technique hyperparameter tuning."""
    _require_imblearn()

    x_train, y_train = _to_graph_level_tabular(train_x, train_y_cls, class_threshold, agg=agg)
    x_test, y_test = _to_graph_level_tabular(test_x, test_y_cls, class_threshold, agg=agg)

    scorer = make_scorer(f1_score, average="binary", pos_label=1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    base_clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=random_state,
    )

    methods: List[ImblearnMethod] = [
        ImblearnMethod(
            name="Baseline (no sampling)",
            sampler="passthrough",
            param_grid={
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        ImblearnMethod(
            name="RandomUnderSampler",
            sampler=RandomUnderSampler(random_state=random_state),
            param_grid={
                "sampler__sampling_strategy": [0.3, 0.5, 0.7, 1.0],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
        ImblearnMethod(
            name="RandomOverSampler",
            sampler=RandomOverSampler(random_state=random_state),
            param_grid={
                "sampler__sampling_strategy": [0.3, 0.5, 0.7, 1.0],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
        ImblearnMethod(
            name="SMOTE",
            sampler=SMOTE(random_state=random_state),
            param_grid={
                "sampler__sampling_strategy": [0.3, 0.5, 0.7, 1.0],
                "sampler__k_neighbors": [3, 5, 7],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
        ImblearnMethod(
            name="BorderlineSMOTE",
            sampler=BorderlineSMOTE(random_state=random_state),
            param_grid={
                "sampler__sampling_strategy": [0.3, 0.5, 0.7, 1.0],
                "sampler__k_neighbors": [3, 5, 7],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
        ImblearnMethod(
            name="ADASYN",
            sampler=ADASYN(random_state=random_state),
            param_grid={
                "sampler__sampling_strategy": [0.3, 0.5, 0.7, 1.0],
                "sampler__n_neighbors": [3, 5, 7],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
        ImblearnMethod(
            name="TomekLinks (cleaning)",
            sampler=TomekLinks(),
            param_grid={
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None, "balanced"],
            },
        ),
        ImblearnMethod(
            name="SMOTEENN",
            sampler=SMOTEENN(
                random_state=random_state,
                smote=SMOTE(random_state=random_state),
            ),
            param_grid={
                "sampler__smote__k_neighbors": [3, 5, 7],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
        ImblearnMethod(
            name="SMOTETomek",
            sampler=SMOTETomek(
                random_state=random_state,
                smote=SMOTE(random_state=random_state),
            ),
            param_grid={
                "sampler__smote__k_neighbors": [3, 5, 7],
                "clf__C": [0.01, 0.1, 1.0, 10.0],
                "clf__class_weight": [None],
            },
        ),
    ]

    results: List[Dict] = []
    for m in methods:
        print(f"\n[imblearn] Tuning: {m.name}")

        pipe = ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("sampler", m.sampler),
                ("clf", base_clf),
            ]
        )

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=m.param_grid,
            scoring=scorer,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

        grid.fit(x_train, y_train)

        best_est = grid.best_estimator_
        test_pred = best_est.predict(x_test)
        test_f1 = f1_score(y_test, test_pred, average="binary", pos_label=1)

        results.append(
            {
                "name": m.name,
                "best_cv_f1": float(grid.best_score_),
                "test_f1": float(test_f1),
                "best_params": grid.best_params_,
            }
        )
        print(f"  Best CV F1: {grid.best_score_:.4f} | Test F1: {test_f1:.4f}")

    results.sort(key=lambda r: r["test_f1"], reverse=True)
    return results

def main():
    global CHECKPOINT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="imblearn",
        choices=["torch", "imblearn"],
        help="torch: existing GNN training comparison; imblearn: sklearn+imbalanced-learn tuned comparison",
    )
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--dp', action='store_true', default=True)
    parser.add_argument('--noise_multiplier', type=float, default=1.0)
    parser.add_argument('--cv_folds', type=int, default=5)
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument(
        '--imblearn_agg',
        type=str,
        default='mean_std',
        choices=['mean', 'mean_std'],
        help='Graph->tabular aggregation for --mode imblearn',
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Only the torch training path needs checkpoint setup. The imblearn path is sklearn-only.
    if args.mode == "torch":
        CHECKPOINT_DIR = setup_checkpoint_directory()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Data
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data('udata', weather_file='weather2016_2021.npy', data_source='udata')
    
    # Build Sequences (Node Level)
    horizons = [12]
    max_horizon = 12
    train_x, _, train_y_cls = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw, 8, max_horizon, 5.0, horizons
    )
    val_x, _, val_y_cls = build_sequences_node_level(
        val_inputs, val_delay_scaled, val_raw, 8, max_horizon, 5.0, horizons
    )
    test_x, _, test_y_cls = build_sequences_node_level(
        test_inputs, test_delay_scaled, test_raw, 8, max_horizon, 5.0, horizons
    )
    
    edge_indices = (edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device))
    
    if args.mode == "imblearn":
        # Use Stage 1 Classifier instead of Logistic Regression
        print("\n[INFO] Using Stage 1 Classifier (SequentialTwoStagePredictor) with imblearn samplers")
        
        # Flatten for sampling: (N, Nodes, Feat) -> (N, Nodes*Feat)
        N, Nodes, Feat = train_x.shape
        
        # Combine features and node-level labels for sampling so we preserve node labels
        # train_x: (N, Nodes, Feat)
        # train_y_cls: (N, Nodes, 1)
        train_combined = torch.cat([train_x, train_y_cls], dim=2) # (N, Nodes, Feat+1)
        x_train_flat = train_combined.reshape(N, -1).numpy() # (N, Nodes*(Feat+1))
        
        # Graph-level labels for the sampler to use (target for resampling)
        y_train_flat = train_y_cls.mean(dim=(1, 2)).numpy() >= 0.5 
        y_train_flat = y_train_flat.astype(int)
        
        # For validation/test, we keep them as tensors for the model
        # But we need to evaluate on them
        
        methods = [
            ("Baseline (no sampling)", None),
            ("RandomUnderSampler", RandomUnderSampler(random_state=args.seed, sampling_strategy=0.5)),
            ("RandomOverSampler", RandomOverSampler(random_state=args.seed, sampling_strategy=0.5)),
            ("SMOTE", SMOTE(random_state=args.seed, sampling_strategy=0.5)),
        ]
        
        results = []
        
        for name, sampler in methods:
            print(f"\n[Method] {name}")
            
            if sampler:
                print("  Resampling training data...")
                x_res_flat, _ = sampler.fit_resample(x_train_flat, y_train_flat)
                
                # Reshape back to (N_res, Nodes, Feat+1)
                res_tensor = torch.tensor(x_res_flat.reshape(-1, Nodes, Feat + 1), dtype=torch.float32)
                
                # Split back into features and labels
                x_res_tensor = res_tensor[:, :, :-1]
                y_res_tensor = res_tensor[:, :, -1:]
                
                # For SMOTE, the labels might be interpolated. Threshold them.
                if "SMOTE" in name:
                    y_res_tensor = (y_res_tensor > 0.5).float()
            else:
                x_res_tensor = train_x
                y_res_tensor = train_y_cls
            
            print(f"  Training samples: {len(x_res_tensor)}")
            
            # Initialize fresh model
            model = SequentialTwoStagePredictor(
                in_channels=8 * train_inputs.shape[2],
                out_channels=1, # Dummy
                hidden_channels=32
            ).to(device)
            
            # Train
            train_stage1_classifier(
                model=model,
                train_x=x_res_tensor,
                train_y_cls=y_res_tensor,
                val_x=val_x,
                val_y_cls=val_y_cls,
                edge_indices=edge_indices,
                device=device,
                epochs=5, # Short training for comparison
                lr=0.001,
                pos_weight=1.0, # Balanced by sampling
                batch_size=32,
                patience=2
            )
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                val_probs = []
                val_targets = []
                for i in range(len(val_x)):
                    data = Data(
                        x=val_x[i].to(device),
                        edge_index_adj=edge_indices[0],
                        edge_index_od=edge_indices[1],
                        edge_index_od_t=edge_indices[2],
                    )
                    _, logits = model.forward_classifier(data)
                    # Graph-level aggregation for metric
                    graph_prob = torch.sigmoid(logits).mean().item()
                    graph_target = val_y_cls[i].mean().item() >= 0.5
                    val_probs.append(graph_prob)
                    val_targets.append(graph_target)
            
            metrics = classification_metrics(np.array(val_probs), np.array(val_targets))
            print(f"  Result: F1={metrics['f1']:.4f}")
            results.append({'method': name, **metrics})
            
        # Print summary
        print("\n" + "="*40)
        print("SUMMARY")
        print("="*40)
        for r in results:
            print(f"{r['method']:<25} | F1: {r['f1']:.4f}")
            
        return results

    # Configurations to test (existing torch mode)
    configs = [
        {'method': 'baseline', 'name': 'Baseline (Weighted Loss)'},
        {'method': 'random_under', 'ratio': 0.5, 'name': 'Undersample (50/50)'},
        {'method': 'random_under', 'ratio': 0.35, 'name': 'Undersample (35/65)'},
        {'method': 'random_over', 'ratio': 0.5, 'name': 'Oversample (50/50)'},
    ]

    results = []

    for config in configs:
        print(f"\nRunning: {config['name']}")

        # Re-init model
        feature_dim = train_inputs.shape[2]
        delay_dim = train_delay_scaled.shape[2]
        model = SequentialTwoStagePredictor(
            in_channels=8 * feature_dim,
            out_channels=len(horizons) * delay_dim,
            hidden_channels=32,
        ).to(device)

        dp_config = DPConfig(
            enabled=args.dp,
            target_epsilon=10.0,
            target_delta=1e-5,
            noise_multiplier=1.0,
            max_grad_norm=1.5,
            sample_rate=args.batch_size / len(train_x),
        )

        # Calculate pos_weight for baseline
        cls_pos_rate = train_y_cls.mean().item()
        pos_weight = (1 - cls_pos_rate) / cls_pos_rate

        res = train_stage1_comparison(
            model, train_x, train_y_cls, val_x, val_y_cls,
            edge_indices, device, args.epochs, args.lr,
            pos_weight, 5, dp_config, args.batch_size,
            config, args.class_threshold
        )

        # Test Evaluation
        if res['best_state']:
            model.encoder.load_state_dict(res['best_state']['encoder'])
            model.classifier.load_state_dict(res['best_state']['classifier'])

        model.eval()
        test_probs, test_targets = [], []
        with torch.no_grad():
            for i in range(len(test_x)):
                data = Data(
                    x=test_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_logits = model.forward_classifier(data)
                test_probs.append(torch.sigmoid(node_logits).cpu())
                test_targets.append(test_y_cls[i].cpu())

        test_probs_np = torch.cat(test_probs).numpy()
        test_targets_np = torch.cat(test_targets).numpy()
        test_metrics = classification_metrics(
            test_probs_np.reshape(-1, 1),
            test_targets_np.reshape(-1, 1),
        )

        print(f"Test Results for {config['name']}:")
        print(f"  F1: {test_metrics['f1']:.4f}")
        print(f"  Precision: {test_metrics['precision']:.4f}")
        print(f"  Recall: {test_metrics['recall']:.4f}")

        results.append({
            'name': config['name'],
            'val_f1': res['best_f1'],
            'test_f1': test_metrics['f1'],
            'test_prec': test_metrics['precision'],
            'test_rec': test_metrics['recall']
        })

    print("\n" + "="*60)
    print("FINAL COMPARISON RESULTS")
    print("="*60)
    print(f"{'Method':<25} | {'Val F1':<10} | {'Test F1':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name']:<25} | {r['val_f1']:.4f}     | {r['test_f1']:.4f}      | {r['test_prec']:.4f}     | {r['test_rec']:.4f}")

if __name__ == '__main__':
    main()
