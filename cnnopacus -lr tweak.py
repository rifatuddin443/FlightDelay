"""Three-stage CNN pipeline with optional DP-SGD and epsilon tracking.

NOTES:
- Mirrors the data loading + sequence building flow from `threestagev4noise.py`.
- Uses a CNN encoder (Conv1d over time) with two heads:
    - Stage 1: node-level delay classification (arrival/departure)
    - Stage 2: delayed-flight regression
    - Stage 3: non-delayed-flight regression
"""

from __future__ import annotations

import argparse
import csv
import copy
import os
import sys
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch_geometric.data import Data
import glob

# Opacus DP
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except Exception:
    PrivacyEngine = None  # type: ignore[assignment]
    ModuleValidator = None  # type: ignore[assignment]
    OPACUS_AVAILABLE = False

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


def _script_stem(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    # keep it filesystem-friendly
    return stem.replace(" ", "_")


def _sigma_tag(noise_multiplier: float, dp_enabled: bool) -> str:
    sigma_val = float(noise_multiplier) if dp_enabled else 0.0
    return f"sigma{sigma_val:.2f}".replace(".", "_")


def _run_tag(*, train_script: str, data_source: str, noise_multiplier: float, dp_enabled: bool) -> str:
    return f"{_script_stem(train_script)}_{data_source}_{_sigma_tag(noise_multiplier, dp_enabled)}"


def _ensure_tagged_filename(path: str, *, run_tag: str, timestamp: str, suffix: str, ext: str, epsilon: Optional[float] = None) -> str:
    """Return a filename which always includes run_tag and timestamp.

    If `path` contains a directory, it is preserved.
    """
    directory = os.path.dirname(path)
    base = os.path.basename(path)
    stem, existing_ext = os.path.splitext(base)

    ext = ext if ext.startswith(".") else f".{ext}"
    if existing_ext:
        ext = existing_ext

    # Always enforce naming for reproducibility
    epsilon_tag = f"_eps{epsilon:.2f}" if epsilon is not None else ""
    filename = f"{run_tag}_{suffix}{epsilon_tag}_{timestamp}{ext}"
    return os.path.join(directory, filename) if directory else filename

# Import visualization functions
try:
    from visualize_training_classification import (
        visualize_training_data,
        visualize_classification_results,
        visualize_regression_timeseries,
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


def _flatten_node_level(x: torch.Tensor) -> torch.Tensor:
    """Flatten [S, N, ...] to [S*N, ...] for node-level training."""
    if x.dim() < 3:
        return x
    return x.reshape(-1, *x.shape[2:])


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
    prob_threshold: float = 0.5,
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
        preds_bin = preds_2d[:, c] >= prob_threshold
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


# --------------------------------------------------------------------------------------
# Architecture override (this file only)
# --------------------------------------------------------------------------------------


class TemporalBlock(nn.Module):
    """TCN-style residual block with dilated Conv1d."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        # DP-safe normalization (BatchNorm is not supported by Opacus)
        self.bn1 = nn.GroupNorm(1, out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.bn2 = nn.GroupNorm(1, out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(p=dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.out_act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.act1(self.bn1(self.conv1(x))))
        out = self.drop2(self.act2(self.bn2(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)


class TemporalConvNet(nn.Module):
    """Stack of `TemporalBlock`s with exponentially increasing dilation."""

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        *,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev = int(in_channels)
        for i, ch in enumerate(channels):
            dilation = 2**i
            layers.append(
                TemporalBlock(
                    prev,
                    int(ch),
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            prev = int(ch)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SequentialTwoStagePredictor(nn.Module):
    """CNN-based encoder + fully-connected heads.

    This class intentionally shadows the imported `SequentialTwoStagePredictor` so the
    architecture can be changed without editing other files.

        Expected input: `data.x` shaped either
            - [num_nodes, in_channels] where in_channels = seq_len * feature_dim, or
            - [num_nodes, seq_len, feature_dim].

    Outputs are node-level:
      - classifier logits: [num_nodes, out_channels]
      - regressor preds:   [num_nodes, out_channels]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        regressor_extra_layer: bool = False,
        seq_len: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.hidden_channels = int(hidden_channels)

        self.seq_len = int(seq_len) if seq_len is not None else None
        self.feature_dim: Optional[int] = None
        if self.seq_len is not None and self.seq_len > 0 and (self.in_channels % self.seq_len == 0):
            self.feature_dim = self.in_channels // self.seq_len

        c1 = max(16, self.hidden_channels // 2)
        c2 = max(16, self.hidden_channels)

        # Preferred encoder: TCN-style temporal Conv1d (dilations + residual blocks)
        # If seq_len can't be inferred, fall back to a simple flattened Conv1d.
        if self.feature_dim is not None:
            tcn = TemporalConvNet(
                self.feature_dim,
                [c1, c2, self.hidden_channels],
                kernel_size=3,
                dropout=0.1,
            )
            self.encoder = nn.Sequential(
                tcn,
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
            )
        else:
            # Fallback: Conv1d over flattened history vector.
            self.encoder = nn.Sequential(
                nn.Conv1d(1, c1, kernel_size=7, padding=3),
                nn.GELU(),
                nn.Conv1d(c1, c2, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(c2, self.hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(start_dim=1),
            )

        self.dropout_cls = nn.Dropout(p=0.1)
        self.dropout_reg = nn.Dropout(p=0.1)

        # Fully-connected classifier head (ends in FC).
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_channels, self.hidden_channels),
            nn.ReLU(),
            nn.Linear(self.hidden_channels, self.out_channels),
        )

        # Fully-connected regressor head (ends in FC).
        if regressor_extra_layer:
            self.regressor = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.ReLU(),
                nn.Linear(self.hidden_channels, self.out_channels),
            )
        else:
            self.regressor = nn.Linear(self.hidden_channels, self.out_channels)

    def _encode_x(self, x: torch.Tensor) -> torch.Tensor:
        # Preferred path: [N, seq_len, feature_dim] -> [N, feature_dim, seq_len]
        if self.feature_dim is not None:
            if x.dim() == 3:
                x_t = x.permute(0, 2, 1).contiguous()
                return self.encoder(x_t)
            if x.dim() == 2 and self.seq_len is not None:
                x3 = x.view(x.shape[0], self.seq_len, self.feature_dim)
                x_t = x3.permute(0, 2, 1).contiguous()
                return self.encoder(x_t)

        # Fallback path: flatten and run 1-channel Conv1d
        if x.dim() == 3:
            x_flat = x.reshape(x.shape[0], -1)
        elif x.dim() == 2:
            x_flat = x
        else:
            x_flat = x.view(x.shape[0], -1)
        return self.encoder(x_flat.unsqueeze(1))

    def forward_classifier(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_x(data.x)
        logits = self.classifier(self.dropout_cls(hidden))
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.dropout_reg(hidden))

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode_x(data.x)
        node_reg = self.forward_regressor(hidden)
        return hidden, node_reg


def _unwrap_opacus_model(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module if model is Opacus-wrapped."""
    return getattr(model, "_module", model)


def _safe_get_epsilon(privacy_engine: Optional[object], target_delta: float) -> float:
    """Get current epsilon from Opacus without crashing on OOM.

    Opacus' default PRV accountant can be memory-intensive for large step counts.
    If epsilon computation fails (e.g., MemoryError), return NaN and keep training.
    """
    if privacy_engine is None:
        return float("inf")
    try:
        # PrivacyEngine.get_epsilon(delta)
        return float(privacy_engine.get_epsilon(float(target_delta)))
    except MemoryError:
        print(
            "[DP WARNING] MemoryError while computing epsilon. "
            "Consider using --dp_accountant rdp (default) or lowering epochs/steps. "
            "Continuing with epsilon=NaN."
        )
        return float("nan")
    except Exception as e:
        print(f"[DP WARNING] Failed to compute epsilon ({type(e).__name__}: {e}). Continuing with epsilon=NaN.")
        return float("nan")


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


def _set_optimizer_hparams(optimizer: torch.optim.Optimizer, lr: float, weight_decay: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
        group["weight_decay"] = weight_decay


def train_stage1_opacus(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    privacy_engine: Optional[object],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    target_delta: float,
    target_epsilon: float,
) -> Tuple[List[Dict], float]:
    """Stage 1: node-level delay classification."""
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING DELAY CLASSIFIER (OPACUS)")
    print("=" * 80)

    base = _unwrap_opacus_model(model)

    for p in base.regressor.parameters():
        p.requires_grad = False
    for p in base.encoder.parameters():
        p.requires_grad = True
    for p in base.classifier.parameters():
        p.requires_grad = True

    _set_optimizer_hparams(optimizer, lr=lr, weight_decay=1e-4)
    cls_loss_fn = FocalLoss(alpha=pos_weight.to(device), gamma=2.0, reduction="mean")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience//2, min_lr=lr*0.01
    )

    history: List[Dict] = []
    best_f1 = 0.0
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    early_stopping = EarlyStopping(patience=patience, mode="max")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_losses: List[float] = []
        step_count = 0

        for batch_x, batch_y_cls, _ in train_loader:
            step_count += 1
            batch_x = batch_x.to(device)
            batch_y_cls = batch_y_cls.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = base._encode_x(batch_x)
            logits = base.classifier(base.dropout_cls(hidden))
            loss = cls_loss_fn(logits, batch_y_cls)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        val_probs_list: List[torch.Tensor] = []
        val_targets_list: List[torch.Tensor] = []
        with torch.no_grad():
            for batch_x, batch_y_cls, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y_cls = batch_y_cls.to(device)
                hidden = base._encode_x(batch_x)
                logits = base.classifier(hidden)
                val_probs_list.append(torch.sigmoid(logits).cpu())
                val_targets_list.append(batch_y_cls.cpu())

        val_probs_np = torch.cat(val_probs_list, dim=0).numpy()
        val_targets_np = torch.cat(val_targets_list, dim=0).numpy()
        val_metrics = classification_metrics_per_channel(
            val_probs_np,
            val_targets_np,
            channel_names=("arrival", "departure"),
        )

        epoch_time = time.time() - epoch_start_time
        current_epsilon = _safe_get_epsilon(privacy_engine, target_delta)

        history.append(
            {
                "epoch": epoch,
                "stage": 1,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "epsilon": current_epsilon,
                "delta": target_delta if privacy_engine is not None else 0.0,
                "epoch_time_seconds": epoch_time,
            }
        )

        eps_str = (
            f"ε: {current_epsilon:.3f}/{target_epsilon}" if privacy_engine is not None else "No DP"
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val F1 (macro): {val_metrics['f1']:.4f} "
            f"[arr {val_metrics['f1_arrival']:.4f}, dep {val_metrics['f1_departure']:.4f}] | "
            f"{eps_str} | LR: {current_lr:.6f} | Steps: {step_count} | Time: {epoch_time:.2f}s"
        )
        
        # Step scheduler
        scheduler.step(val_metrics['f1'])

        if float(val_metrics["f1"]) > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_state = {
                "encoder": copy.deepcopy(base.encoder.state_dict()),
                "classifier": copy.deepcopy(base.classifier.state_dict()),
            }
            print("  ✓ New best checkpoint")

        if early_stopping(val_metrics["f1"], epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        base.encoder.load_state_dict(best_state["encoder"])
        base.classifier.load_state_dict(best_state["classifier"])

    for p in base.regressor.parameters():
        p.requires_grad = True

    stage_time = time.time() - stage_start_time
    print(f"\nStage 1 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    return history, stage_time
def train_stage2_opacus(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    privacy_engine: Optional[object],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    delay_threshold: float,
    patience: int,
    target_delta: float,
    target_epsilon: float,
    freeze_encoder: bool = False,
) -> Tuple[List[Dict], float]:
    """Stage 2: delayed-flight regressor fine-tuning (Opacus).

    Mask is defined using *ground-truth* delays in scaled space.
    """
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING DELAY REGRESSOR (DELAYED FLIGHTS) (OPACUS)")
    print("=" * 80)

    base = _unwrap_opacus_model(model)

    # Keep encoder trainable by default for better learning
    # Only freeze if explicitly requested via freeze_encoder_stage2
    for p in base.encoder.parameters():
        p.requires_grad = not freeze_encoder
    for p in base.classifier.parameters():
        p.requires_grad = False
    for p in base.regressor.parameters():
        p.requires_grad = True
    
    # Reset optimizer momentum for fresh start
    for group in optimizer.param_groups:
        for key in ['momentum_buffer', 'exp_avg', 'exp_avg_sq']:
            if key in group:
                del group[key]

    _set_optimizer_hparams(optimizer, lr=lr, weight_decay=1e-4)
    huber_loss = nn.HuberLoss(reduction="none", delta=2.0)
    
    # Learning rate scheduler for Stage 2
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2,min_lr=lr*0.001
    )

    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32), device=device)
        std_t = torch.tensor(np.array(scaler.std, dtype=np.float32), device=device)
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
        delay_threshold_scaled = (torch.full_like(mean_t, float(delay_threshold)) - mean_t) / std_t
    else:
        delay_threshold_scaled = torch.tensor([float(delay_threshold)], device=device)

    def masked_huber_loss(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        thr = delay_threshold_scaled
        if thr.numel() == 1 and targets.shape[-1] > 1:
            thr = thr.expand(targets.shape[-1])
        thr = thr.to(targets.device)
        mask = (targets > thr).float()
        per_elem = huber_loss(preds, targets) * mask
        denom = mask.sum(dim=0).clamp_min(1.0)
        loss_ch = per_elem.sum(dim=0) / denom
        loss = loss_ch.mean()
        return loss, mask

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    early_stopping = EarlyStopping(patience=patience, mode="min")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_losses: List[float] = []
        total_masked = 0.0
        total_elements = 0.0

        for batch_x, _, batch_y_reg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = base._encode_x(batch_x)
            preds = base.forward_regressor(hidden)
            loss, mask = masked_huber_loss(preds, batch_y_reg)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
            total_masked += float(mask.sum().item())
            total_elements += float(mask.numel())

        model.eval()
        val_losses: List[float] = []
        val_masked = 0.0
        val_elements = 0.0
        with torch.no_grad():
            for batch_x, _, batch_y_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_reg = batch_y_reg.to(device)
                hidden = base._encode_x(batch_x)
                preds = base.forward_regressor(hidden)
                loss, mask = masked_huber_loss(preds, batch_y_reg)
                val_losses.append(float(loss.item()))
                val_masked += float(mask.sum().item())
                val_elements += float(mask.numel())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        epoch_time = time.time() - epoch_start_time
        current_epsilon = _safe_get_epsilon(privacy_engine, target_delta)

        masked_ratio = (total_masked / total_elements) if total_elements > 0 else 0.0
        val_masked_ratio = (val_masked / val_elements) if val_elements > 0 else 0.0

        history.append(
            {
                "epoch": epoch,
                "stage": 2,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_masked_ratio": masked_ratio,
                "val_loss": val_loss,
                "val_masked_ratio": val_masked_ratio,
                "epsilon": current_epsilon,
                "delta": target_delta if privacy_engine is not None else 0.0,
                "epoch_time_seconds": epoch_time,
            }
        )

        eps_str = (
            f"ε: {current_epsilon:.3f}/{target_epsilon}" if privacy_engine is not None else "No DP"
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {history[-1]['train_loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | Masked: {masked_ratio*100:.1f}% "
            f"(val {val_masked_ratio*100:.1f}%) | {eps_str} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
        )
        
        # Step scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(base.regressor.state_dict())
            print("  ✓ New best checkpoint")

        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        base.regressor.load_state_dict(best_state)

    for p in base.parameters():
        p.requires_grad = True

    stage_time = time.time() - stage_start_time
    final_epsilon = _safe_get_epsilon(privacy_engine, target_delta)
    print(f"\nStage 2 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    if privacy_engine is not None:
        print(f"Final ε: {final_epsilon:.3f} (target: {target_epsilon})")
    return history, stage_time
def train_stage3_opacus(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    privacy_engine: Optional[object],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    delay_threshold: float,
    patience: int,
    target_delta: float,
    target_epsilon: float,
) -> Tuple[List[Dict], float]:
    """Stage 3: non-delayed regressor fine-tuning (Opacus).

    Mask is created in denormalized space using |delay| < delay_threshold.
    Loss is computed in normalized space (so gradients remain well-behaved).
    """
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("STAGE 3: TRAINING DELAY REGRESSOR (NON-DELAYED FLIGHTS) (OPACUS)")
    print(f"Training on flights with |delay| < {delay_threshold} min")
    print("=" * 80)

    base = _unwrap_opacus_model(model)

    for p in base.encoder.parameters():
        p.requires_grad = False
    for p in base.classifier.parameters():
        p.requires_grad = False
    for p in base.regressor.parameters():
        p.requires_grad = True

    _set_optimizer_hparams(optimizer, lr=lr*2, weight_decay=1e-5)
    reg_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    
    # Learning rate scheduler for Stage 3
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience//2, min_lr=lr*0.001
    )

    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32), device=device)
        std_t = torch.tensor(np.array(scaler.std, dtype=np.float32), device=device)
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
    else:
        mean_t = None
        std_t = None

    def nondelayed_mask_from_targets(targets_scaled: torch.Tensor) -> torch.Tensor:
        if mean_t is None or std_t is None:
            targets_denorm = targets_scaled.detach()
        else:
            targets_denorm = targets_scaled.detach() * std_t + mean_t
        return (targets_denorm.abs() < float(delay_threshold)).float()

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    early_stopping = EarlyStopping(patience=patience, mode="min")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        epoch_losses: List[float] = []
        total_nondelayed = 0.0
        total_elements = 0.0

        for batch_x, _, batch_y_reg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            element_mask = nondelayed_mask_from_targets(batch_y_reg)
            num_nondelayed = float(element_mask.sum().item())
            total_nondelayed += num_nondelayed
            total_elements += float(element_mask.numel())

            if num_nondelayed <= 0.0:
                continue

            optimizer.zero_grad(set_to_none=True)
            hidden = base._encode_x(batch_x)
            preds = base.forward_regressor(hidden)
            loss_per = reg_loss_fn(preds, batch_y_reg) * element_mask
            denom = element_mask.sum(dim=0).clamp_min(1.0)
            loss_ch = loss_per.sum(dim=0) / denom
            loss = loss_ch.mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        nondelayed_ratio = (total_nondelayed / total_elements) if total_elements > 0 else 0.0

        model.eval()
        val_losses: List[float] = []
        val_nondelayed = 0.0
        val_elements = 0.0
        with torch.no_grad():
            for batch_x, _, batch_y_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_reg = batch_y_reg.to(device)
                hidden = base._encode_x(batch_x)
                preds = base.forward_regressor(hidden)

                element_mask = nondelayed_mask_from_targets(batch_y_reg)
                num_nondelayed = float(element_mask.sum().item())
                val_nondelayed += num_nondelayed
                val_elements += float(element_mask.numel())

                if num_nondelayed <= 0.0:
                    continue

                if mean_t is not None and std_t is not None:
                    pred_denorm = preds * std_t + mean_t
                    target_denorm = batch_y_reg * std_t + mean_t
                else:
                    pred_denorm = preds
                    target_denorm = batch_y_reg

                se = ((pred_denorm - target_denorm) ** 2) * element_mask
                denom = element_mask.sum(dim=0).clamp_min(1.0)
                loss_val_ch = se.sum(dim=0) / denom
                val_losses.append(float(loss_val_ch.mean().item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        val_nondelayed_ratio = (val_nondelayed / val_elements) if val_elements > 0 else 0.0

        epoch_time = time.time() - epoch_start_time
        current_epsilon = _safe_get_epsilon(privacy_engine, target_delta)

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
                "delta": target_delta if privacy_engine is not None else 0.0,
                "epoch_time_seconds": epoch_time,
            }
        )

        eps_str = (
            f"ε: {current_epsilon:.3f}/{target_epsilon}" if privacy_engine is not None else "No DP"
        )
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val: {val_loss:.4f} | Non-delayed: {total_nondelayed:.0f} "
            f"({nondelayed_ratio*100:.1f}%) | Val ND: {val_nondelayed:.0f} "
            f"({val_nondelayed_ratio*100:.1f}%) | {eps_str} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s"
        )
        
        # Step scheduler
        scheduler.step(val_loss)

        if val_loss < best_val_loss and val_nondelayed > 0:
            best_val_loss = val_loss
            best_state = copy.deepcopy(base.regressor.state_dict())
            print("  ✓ New best (Stage 3)")

        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        base.regressor.load_state_dict(best_state)

    for p in base.parameters():
        p.requires_grad = True

    stage_time = time.time() - stage_start_time
    final_epsilon = _safe_get_epsilon(privacy_engine, target_delta)
    print(f"\nStage 3 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    if privacy_engine is not None:
        print(f"Final ε: {final_epsilon:.3f} (target: {target_epsilon})")
    return history, stage_time

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
    run_tag: str,
    timestamp: str,
    histories: List[Dict],
    final_epsilon: float,
    final_delta: float,
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    train_samples: int,
    val_samples: int,
    dp_enabled: bool,
    target_epsilon: float,
    noise_multiplier: float,
    *,
    save_model: bool = True,
    artifact_prefix: str = "train",
    seq_len: Optional[int] = None,
    args: Optional[object] = None,
    checkpoint_dir: Optional[str] = None,
) -> None:
    """Final evaluation and export with Stage 3 regressor."""
    model = _unwrap_opacus_model(model)  # type: ignore[assignment]
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    print(f"Test samples: {len(test_x)}")
    
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    if save_model:
        # Save both regressors if available (Stage 2 delayed + Stage 3 non-delayed).
        to_save = {
            'encoder': model.encoder.state_dict(),
            'classifier': model.classifier.state_dict(),
            # Backwards-compatible key: treat 'regressor' as the delayed regressor.
            'regressor': getattr(model, 'regressor_delayed', model.regressor).state_dict(),
            'final_epsilon': float(final_epsilon),
            'final_delta': float(final_delta),
            'target_epsilon': float(target_epsilon),
            'epsilon_exceeded': final_epsilon > float(target_epsilon) if dp_enabled else False,
            'run_tag': str(run_tag),
            'timestamp': str(timestamp),
            'dp_enabled': bool(dp_enabled),
            'noise_multiplier': float(noise_multiplier),
        }
        if hasattr(model, 'regressor_delayed'):
            to_save['regressor_delayed'] = model.regressor_delayed.state_dict()
        if hasattr(model, 'regressor_nondelayed'):
            to_save['regressor_nondelayed'] = model.regressor_nondelayed.state_dict()
        torch.save(to_save, model_path)
        checkpoint = to_save
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load weights supporting multiple checkpoint formats.
    # New format: separate submodule state_dicts.
    if isinstance(checkpoint, dict) and 'encoder' in checkpoint and 'classifier' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder'])
        model.classifier.load_state_dict(checkpoint['classifier'])

        # Recreate delayed/non-delayed regressors if present.
        if 'regressor_delayed' in checkpoint and 'regressor_nondelayed' in checkpoint:
            model.regressor_delayed = copy.deepcopy(model.regressor)
            model.regressor_nondelayed = copy.deepcopy(model.regressor)
            model.regressor_delayed.load_state_dict(checkpoint['regressor_delayed'])
            model.regressor_nondelayed.load_state_dict(checkpoint['regressor_nondelayed'])
        else:
            if 'regressor' not in checkpoint:
                raise ValueError("Checkpoint missing required key: 'regressor'")
            model.regressor.load_state_dict(checkpoint['regressor'])

    # Wrapper formats: {'state_dict': ...} or {'model_state_dict': ...}
    elif isinstance(checkpoint, dict) and ('state_dict' in checkpoint or 'model_state_dict' in checkpoint):
        sd = checkpoint.get('state_dict', checkpoint.get('model_state_dict'))
        if not isinstance(sd, dict):
            raise ValueError("Checkpoint has 'state_dict' but it is not a dict")
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading state_dict (ok if older format): {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading state_dict (ok if older format): {len(unexpected)}")

    # Plain state_dict format: torch.save(model.state_dict(), path)
    elif isinstance(checkpoint, dict) and checkpoint and all(isinstance(k, str) for k in checkpoint.keys()) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"[WARN] Missing keys when loading state_dict (ok if older format): {len(missing)}")
        if unexpected:
            print(f"[WARN] Unexpected keys when loading state_dict (ok if older format): {len(unexpected)}")

    else:
        raise ValueError(
            "Unrecognized checkpoint format. Expected either {encoder/classifier/...} dict or a full model state_dict."
        )
    
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
            
            # Always compute classifier once.
            hidden, node_logits = model.forward_classifier(data)
            probs = torch.sigmoid(node_logits)

            # If we have two regressors, route by the classifier gate.
            if hasattr(model, 'regressor_delayed') and hasattr(model, 'regressor_nondelayed'):
                hidden_dropped = model.dropout_reg(hidden)
                reg_delayed = model.regressor_delayed(hidden_dropped)
                reg_nondelayed = model.regressor_nondelayed(hidden_dropped)
                # Soft gating: smoothly mix regressors based on delayed probability.
                # class_threshold is treated as the midpoint (gate=0.5 when prob==threshold).
                gate = torch.sigmoid((probs - class_threshold) * 10.0)
                node_reg = reg_delayed * gate + reg_nondelayed * (1.0 - gate)
            else:
                node_reg = model.forward_regressor(hidden)

            logits_list.append(probs.cpu().numpy())
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
        prob_threshold=class_threshold,
    )
    
    # NOTE: If a dual-regressor checkpoint was loaded, routing was already applied
    # in the per-sample loop above. For legacy single-regressor checkpoints, we
    # keep the original behavior (delayed-only gating).
    if 'regressor_delayed' in checkpoint and 'regressor_nondelayed' in checkpoint:
        gated_preds = test_reg_preds
    else:
        test_mask = (test_probs >= class_threshold)
        gated_preds = test_reg_preds * test_mask
    
    print(f"\n[DENORMALIZATION] Checking predictions...")
    print(f"  Gated predictions shape: {gated_preds.shape}")
    print(f"  Gated predictions (scaled): min={gated_preds.min():.3f}, max={gated_preds.max():.3f}, mean={gated_preds.mean():.3f}")
    print(f"  Test targets (scaled): min={test_reg_targets.min():.3f}, max={test_reg_targets.max():.3f}, mean={test_reg_targets.mean():.3f}")
    
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
    
    # Flatten both predictions and targets consistently for element-wise evaluation
    preds_flat = preds_denorm.flatten()
    targets_flat = targets_denorm.flatten()
    
    # Evaluate on delayed flights (Actual > Threshold)
    delayed_mask = targets_flat > delay_threshold
    if delayed_mask.sum() > 0:
        delayed_preds = preds_flat[delayed_mask]
        delayed_targets = targets_flat[delayed_mask]
        mae_delayed = np.mean(np.abs(delayed_preds - delayed_targets))
        rmse_delayed = np.sqrt(np.mean((delayed_preds - delayed_targets) ** 2))
    else:
        mae_delayed, rmse_delayed = 0.0, 0.0
    
    # Evaluate on non-delayed flights (Actual <= Threshold)
    nondelayed_mask = targets_flat <= delay_threshold
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
    
    print(f"\nREGRESSION (delayed flights > {delay_threshold} min):")
    print(f"  MAE: {mae_delayed:.4f} min | RMSE: {rmse_delayed:.4f} min")
    print(f"  Number of delayed samples: {delayed_mask.sum()}")
    
    print(f"\nREGRESSION (non-delayed flights <= {delay_threshold} min):")
    print(f"  MAE: {mae_nondelayed:.4f} min | RMSE: {rmse_nondelayed:.4f} min")
    print(f"  Number of non-delayed samples: {nondelayed_mask.sum()}")
    
    print("\nREGRESSION (overall):")
    print(f"  MAE: {mae_overall:.4f} min | RMSE: {rmse_overall:.4f} min")

    # Per-channel regression metrics (keep separate from flattened/merged metrics above)
    arr_mae_d = arr_rmse_d = arr_n_d = 0.0
    dep_mae_d = dep_rmse_d = dep_n_d = 0.0
    arr_mae_nd = arr_rmse_nd = arr_n_nd = 0.0
    dep_mae_nd = dep_rmse_nd = dep_n_nd = 0.0
    arr_mae_all = arr_rmse_all = 0.0
    dep_mae_all = dep_rmse_all = 0.0
    
    if targets_denorm.ndim == 2 and targets_denorm.shape[1] >= 2:
        arr_targets = targets_denorm[:, 0].reshape(-1)
        dep_targets = targets_denorm[:, 1].reshape(-1)
        arr_preds = preds_denorm[:, 0].reshape(-1)
        dep_preds = preds_denorm[:, 1].reshape(-1)

        def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int]:
            if int(mask.sum()) == 0:
                return 0.0, 0.0, 0
            yt = y_true[mask]
            yp = y_pred[mask]
            mae = float(np.mean(np.abs(yp - yt)))
            rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
            return mae, rmse, int(mask.sum())

        arr_delayed_mask = arr_targets > delay_threshold
        dep_delayed_mask = dep_targets > delay_threshold
        arr_nondelayed_mask = ~arr_delayed_mask
        dep_nondelayed_mask = ~dep_delayed_mask

        arr_mae_d, arr_rmse_d, arr_n_d = _mae_rmse(arr_targets, arr_preds, arr_delayed_mask)
        dep_mae_d, dep_rmse_d, dep_n_d = _mae_rmse(dep_targets, dep_preds, dep_delayed_mask)
        arr_mae_nd, arr_rmse_nd, arr_n_nd = _mae_rmse(arr_targets, arr_preds, arr_nondelayed_mask)
        dep_mae_nd, dep_rmse_nd, dep_n_nd = _mae_rmse(dep_targets, dep_preds, dep_nondelayed_mask)
        arr_mae_all, arr_rmse_all, _ = _mae_rmse(arr_targets, arr_preds, np.ones_like(arr_targets, dtype=bool))
        dep_mae_all, dep_rmse_all, _ = _mae_rmse(dep_targets, dep_preds, np.ones_like(dep_targets, dtype=bool))

        print("\nREGRESSION (per-channel, Arrival vs Departure):")
        print(f"  Delayed   (> {delay_threshold} min): Arrival MAE {arr_mae_d:.4f} | RMSE {arr_rmse_d:.4f} (n={arr_n_d}), "
              f"Departure MAE {dep_mae_d:.4f} | RMSE {dep_rmse_d:.4f} (n={dep_n_d})")
        print(f"  Non-delay (<= {delay_threshold} min): Arrival MAE {arr_mae_nd:.4f} | RMSE {arr_rmse_nd:.4f} (n={arr_n_nd}), "
              f"Departure MAE {dep_mae_nd:.4f} | RMSE {dep_rmse_nd:.4f} (n={dep_n_nd})")
        print(f"  Overall: Arrival MAE {arr_mae_all:.4f} | RMSE {arr_rmse_all:.4f}, "
              f"Departure MAE {dep_mae_all:.4f} | RMSE {dep_rmse_all:.4f}")
    
    # Visualize classification results
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating classification results plots...")
        try:
            # Keep plotting arrays aligned in length.
            # We plot a single channel consistently (arrival = channel 0) for both
            # classification and regression.
            if test_probs.ndim > 1 and test_probs.shape[1] > 1:
                test_cls_pred = (test_probs[:, 0] >= class_threshold).astype(int)
                test_cls_true = test_cls_targets[:, 0].astype(int)
            else:
                test_cls_pred = (test_probs >= class_threshold).astype(int).reshape(-1)
                test_cls_true = test_cls_targets.astype(int).reshape(-1)

            if targets_denorm.ndim > 1 and targets_denorm.shape[1] > 1:
                test_reg_true = targets_denorm[:, 0].reshape(-1)
                test_reg_pred = preds_denorm[:, 0].reshape(-1)
            else:
                test_reg_true = targets_denorm.reshape(-1)
                test_reg_pred = preds_denorm.reshape(-1)
            
            visualize_classification_results(
                test_cls_true,
                test_cls_pred,
                test_reg_true,
                test_reg_pred,
                threshold=delay_threshold,
                save_path=f"{run_tag}_{artifact_prefix}_classification_results_{timestamp}.png",
            )
            print("  ✓ Classification results visualization saved")

            visualize_regression_timeseries(
                targets_denorm,
                preds_denorm,
                title="Regression Over Time (True vs Predicted)",
                xlabel="Time (sample index)",
                ylabel="Delay (minutes)",
                save_path=f"{run_tag}_{artifact_prefix}_regression_timeseries_{timestamp}.png",
            )
            print("  ✓ Regression time-series visualization saved")
        except Exception as e:
            print(f"  ✗ Error generating classification results visualization: {e}")
    
    print("\nPRIVACY BUDGET:")
    print(f"  Target ε: {float(target_epsilon):.3f}")
    print(f"  Final ε: {final_epsilon:.3f}")
    if dp_enabled:
        if final_epsilon <= float(target_epsilon):
            print("  ✓ Budget maintained (within target)")
        else:
            overshoot = final_epsilon - float(target_epsilon)
            pct = (overshoot / float(target_epsilon) * 100.0) if float(target_epsilon) > 0 else float('inf')
            print(f"  ⚠️ Budget exceeded by {overshoot:.3f} ε ({pct:.1f}%)")
    print(f"  Final δ: {final_delta:.2e}")
    
    print("\nTRAINING TIME:")
    total_time = stage1_time + stage2_time + stage3_time
    print(f"  Stage 1: {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    print(f"  Stage 2: {stage2_time:.2f}s ({stage2_time/60:.2f} min)")
    print(f"  Stage 3: {stage3_time:.2f}s ({stage3_time/60:.2f} min)")
    print(f"  Total: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    
    out_dir = os.path.dirname(model_path)
    epsilon_tag = f"_eps{final_epsilon:.2f}" if dp_enabled else ""
    history_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_history{epsilon_tag}_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_history{epsilon_tag}_{timestamp}.csv"
    results_table_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_results_table{epsilon_tag}_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_results_table{epsilon_tag}_{timestamp}.csv"
    
    # Export history
    if histories:
        with open(history_csv, "w", newline="") as f:
            all_fields = sorted({k for row in histories for k in row})
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(histories)
    
    # Prepare summary metrics for results table (no separate summary CSV)
    summary = {
        # Classification metrics (macro-averaged)
        'classification_precision': test_cls_metrics['precision'],
        'classification_recall': test_cls_metrics['recall'],
        'classification_f1': test_cls_metrics['f1'],
        'classification_accuracy': test_cls_metrics['accuracy'],
        # Classification metrics (per-channel: arrival)
        'classification_precision_arrival': test_cls_metrics['precision_arrival'],
        'classification_recall_arrival': test_cls_metrics['recall_arrival'],
        'classification_f1_arrival': test_cls_metrics['f1_arrival'],
        'classification_accuracy_arrival': test_cls_metrics['accuracy_arrival'],
        # Classification metrics (per-channel: departure)
        'classification_precision_departure': test_cls_metrics['precision_departure'],
        'classification_recall_departure': test_cls_metrics['recall_departure'],
        'classification_f1_departure': test_cls_metrics['f1_departure'],
        'classification_accuracy_departure': test_cls_metrics['accuracy_departure'],
        # Regression metrics (overall/flattened)
        'regression_mae_delayed': mae_delayed,
        'regression_rmse_delayed': rmse_delayed,
        'regression_mae_nondelayed': mae_nondelayed,
        'regression_rmse_nondelayed': rmse_nondelayed,
        'regression_mae_overall': mae_overall,
        'regression_rmse_overall': rmse_overall,
        # Regression metrics (per-channel: arrival - delayed)
        'regression_mae_delayed_arrival': arr_mae_d,
        'regression_rmse_delayed_arrival': arr_rmse_d,
        'num_delayed_samples_arrival': int(arr_n_d),
        # Regression metrics (per-channel: departure - delayed)
        'regression_mae_delayed_departure': dep_mae_d,
        'regression_rmse_delayed_departure': dep_rmse_d,
        'num_delayed_samples_departure': int(dep_n_d),
        # Regression metrics (per-channel: arrival - non-delayed)
        'regression_mae_nondelayed_arrival': arr_mae_nd,
        'regression_rmse_nondelayed_arrival': arr_rmse_nd,
        'num_nondelayed_samples_arrival': int(arr_n_nd),
        # Regression metrics (per-channel: departure - non-delayed)
        'regression_mae_nondelayed_departure': dep_mae_nd,
        'regression_rmse_nondelayed_departure': dep_rmse_nd,
        'num_nondelayed_samples_departure': int(dep_n_nd),
        # Regression metrics (per-channel: arrival - overall)
        'regression_mae_overall_arrival': arr_mae_all,
        'regression_rmse_overall_arrival': arr_rmse_all,
        # Regression metrics (per-channel: departure - overall)
        'regression_mae_overall_departure': dep_mae_all,
        'regression_rmse_overall_departure': dep_rmse_all,
        # Sample counts
        'num_delayed_samples': int(delayed_mask.sum()),
        'num_nondelayed_samples': int(nondelayed_mask.sum()),
        'target_epsilon': float(target_epsilon),
        'final_epsilon': final_epsilon,
        'epsilon_exceeded': (final_epsilon > float(target_epsilon)) if dp_enabled else False,
        'epsilon_overshoot': max(0.0, final_epsilon - float(target_epsilon)) if dp_enabled else 0.0,
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

    # Export comprehensive results table (evaluate_regression_v4-style)
    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-10
        return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))

    # Per-channel masks based on actual delays
    arr_targets = targets_denorm[:, 0] if targets_denorm.ndim == 2 and targets_denorm.shape[1] >= 2 else targets_denorm.reshape(-1)
    dep_targets = targets_denorm[:, 1] if targets_denorm.ndim == 2 and targets_denorm.shape[1] >= 2 else targets_denorm.reshape(-1)
    arr_preds = preds_denorm[:, 0] if preds_denorm.ndim == 2 and preds_denorm.shape[1] >= 2 else preds_denorm.reshape(-1)
    dep_preds = preds_denorm[:, 1] if preds_denorm.ndim == 2 and preds_denorm.shape[1] >= 2 else preds_denorm.reshape(-1)

    arr_delayed = arr_targets > delay_threshold
    dep_delayed = dep_targets > delay_threshold
    arr_nondelayed = arr_targets <= delay_threshold
    dep_nondelayed = dep_targets <= delay_threshold

    def _channel_metrics(mask: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {
                'mae': 0.0,
                'rmse': 0.0,
                'r2': 0.0,
                'mean_pred': 0.0,
                'mean_target': 0.0,
                'num_samples': 0,
            }
        yt = y_true[mask]
        yp = y_pred[mask]
        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': _safe_r2(yt, yp),
            'mean_pred': float(np.mean(yp)),
            'mean_target': float(np.mean(yt)),
            'num_samples': int(mask.sum()),
        }

    overall_arr = _channel_metrics(np.ones_like(arr_targets, dtype=bool), arr_targets, arr_preds)
    overall_dep = _channel_metrics(np.ones_like(dep_targets, dtype=bool), dep_targets, dep_preds)
    delayed_arr = _channel_metrics(arr_delayed, arr_targets, arr_preds)
    delayed_dep = _channel_metrics(dep_delayed, dep_targets, dep_preds)
    nondelayed_arr = _channel_metrics(arr_nondelayed, arr_targets, arr_preds)
    nondelayed_dep = _channel_metrics(dep_nondelayed, dep_targets, dep_preds)

    with open(results_table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["=" * 80])
        writer.writerow(["COMPREHENSIVE EVALUATION RESULTS"])
        writer.writerow(["=" * 80])
        writer.writerow([])

        writer.writerow(["MODEL INFORMATION"])
        writer.writerow(["Model Path", model_path])
        writer.writerow(["Run Tag", run_tag])
        writer.writerow(["Timestamp", timestamp])
        writer.writerow(["Artifact Prefix", artifact_prefix])
        writer.writerow([])
        
        writer.writerow(["=== COMMAND-LINE ARGUMENTS ==="])
        if args is not None:
            writer.writerow(["data_source", getattr(args, 'data_source', 'N/A')])
            writer.writerow(["seq_len", getattr(args, 'seq_len', seq_len)])
            writer.writerow(["horizons", ",".join(str(h) for h in horizons)])
            writer.writerow(["delay_threshold", f"{delay_threshold} min"])
            writer.writerow(["class_threshold", class_threshold])
            writer.writerow(["use_node_level", getattr(args, 'use_node_level', 'N/A')])
            writer.writerow(["exclude_time_features", getattr(args, 'exclude_time_features', 'N/A')])
            writer.writerow(["weather_file", getattr(args, 'weather_file', 'N/A')])
            writer.writerow(["period_hours", getattr(args, 'period_hours', 'N/A')])
            writer.writerow(["stage1_epochs", getattr(args, 'stage1_epochs', 'N/A')])
            writer.writerow(["stage2_epochs", getattr(args, 'stage2_epochs', 'N/A')])
            writer.writerow(["stage3_epochs", getattr(args, 'stage3_epochs', 'N/A')])
            writer.writerow(["batch_size", getattr(args, 'batch_size', 'N/A')])
            writer.writerow(["lr", getattr(args, 'lr', 'N/A')])
            writer.writerow(["patience", getattr(args, 'patience', 'N/A')])
            writer.writerow(["dp", getattr(args, 'dp', dp_enabled)])
            writer.writerow(["dp_accountant", getattr(args, 'dp_accountant', 'N/A')])
            writer.writerow(["epsilon (target)", f"{float(target_epsilon):.3f}"])
            writer.writerow(["target_delta", getattr(args, 'target_delta', 'N/A')])
            writer.writerow(["noise_multiplier", float(noise_multiplier)])
            writer.writerow(["max_grad_norm", getattr(args, 'max_grad_norm', 'N/A')])
            writer.writerow(["sample_rate", getattr(args, 'sample_rate', 'N/A')])
            writer.writerow(["epsilon_tolerance", getattr(args, 'epsilon_tolerance', 'N/A')])
            writer.writerow(["model_path", getattr(args, 'model_path', 'N/A')])
            writer.writerow(["checkpoint_dir", getattr(args, 'checkpoint_dir', 'N/A')])
            writer.writerow(["seed", getattr(args, 'seed', 'N/A')])
            writer.writerow(["balance_50_50", getattr(args, 'balance_50_50', 'N/A')])
            writer.writerow(["epsilonfixed", getattr(args, 'epsilonfixed', 'N/A')])
        else:
            writer.writerow(["Arguments object not provided"])
        writer.writerow([])
        
        writer.writerow(["=== TRAINING RESULTS ==="])
        writer.writerow(["Final Epsilon", f"{final_epsilon:.3f}"])
        writer.writerow(["Final Delta", f"{final_delta:.2e}"])
        writer.writerow(["Epsilon Exceeded", summary['epsilon_exceeded']])
        writer.writerow(["Train Samples", train_samples])
        writer.writerow(["Val Samples", val_samples])
        writer.writerow(["Test Samples", len(test_x)])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow(["CLASSIFICATION METRICS (Macro over Arrival/Departure)"])
        writer.writerow(["=" * 80])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Precision", f"{test_cls_metrics['precision']:.4f}"])
        writer.writerow(["Recall", f"{test_cls_metrics['recall']:.4f}"])
        writer.writerow(["F1 Score", f"{test_cls_metrics['f1']:.4f}"])
        writer.writerow(["Accuracy", f"{test_cls_metrics['accuracy']:.4f}"])
        writer.writerow([])
        writer.writerow(["Per-Channel Metrics:"])
        writer.writerow(["Channel", "Precision", "Recall", "F1 Score", "Accuracy"])
        writer.writerow([
            "Arrival",
            f"{test_cls_metrics.get('precision_arrival', 0):.4f}",
            f"{test_cls_metrics.get('recall_arrival', 0):.4f}",
            f"{test_cls_metrics.get('f1_arrival', 0):.4f}",
            f"{test_cls_metrics.get('accuracy_arrival', 0):.4f}",
        ])
        writer.writerow([
            "Departure",
            f"{test_cls_metrics.get('precision_departure', 0):.4f}",
            f"{test_cls_metrics.get('recall_departure', 0):.4f}",
            f"{test_cls_metrics.get('f1_departure', 0):.4f}",
            f"{test_cls_metrics.get('accuracy_departure', 0):.4f}",
        ])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow(["OVERALL SUMMARY TABLE"])
        writer.writerow(["=" * 80])
        writer.writerow([
            "Epsilon",
            "Overall MAE (min)",
            "Arrival MAE (min)",
            "Departure MAE (min)",
            "Precision",
            "Recall",
            "F1 Score",
            "Accuracy",
        ])
        writer.writerow([
            f"{final_epsilon:.2f}",
            f"{mae_overall:.4f}",
            f"{overall_arr['mae']:.4f}",
            f"{overall_dep['mae']:.4f}",
            f"{test_cls_metrics['precision']:.4f}",
            f"{test_cls_metrics['recall']:.4f}",
            f"{test_cls_metrics['f1']:.4f}",
            f"{test_cls_metrics['accuracy']:.4f}",
        ])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow([f"DETAILED METRICS - DELAYED (>{delay_threshold} min)"])
        writer.writerow(["=" * 80])
        writer.writerow([
            "Group",
            "Samples",
            "Arrival MAE",
            "Arrival RMSE",
            "Arrival R2",
            "Departure MAE",
            "Departure RMSE",
            "Departure R2",
            "Mean Arr Pred",
            "Mean Arr Target",
            "Mean Dep Pred",
            "Mean Dep Target",
        ])
        writer.writerow([
            "Delayed",
            int(summary['num_delayed_samples']),
            f"{delayed_arr['mae']:.4f}",
            f"{delayed_arr['rmse']:.4f}",
            f"{delayed_arr['r2']:.4f}",
            f"{delayed_dep['mae']:.4f}",
            f"{delayed_dep['rmse']:.4f}",
            f"{delayed_dep['r2']:.4f}",
            f"{delayed_arr['mean_pred']:.2f}",
            f"{delayed_arr['mean_target']:.2f}",
            f"{delayed_dep['mean_pred']:.2f}",
            f"{delayed_dep['mean_target']:.2f}",
        ])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow([f"DETAILED METRICS - NON-DELAYED (<= {delay_threshold} min)"])
        writer.writerow(["=" * 80])
        writer.writerow([
            "Group",
            "Samples",
            "Arrival MAE",
            "Arrival RMSE",
            "Arrival R2",
            "Departure MAE",
            "Departure RMSE",
            "Departure R2",
            "Mean Arr Pred",
            "Mean Arr Target",
            "Mean Dep Pred",
            "Mean Dep Target",
        ])
        writer.writerow([
            "Non-delayed",
            int(summary['num_nondelayed_samples']),
            f"{nondelayed_arr['mae']:.4f}",
            f"{nondelayed_arr['rmse']:.4f}",
            f"{nondelayed_arr['r2']:.4f}",
            f"{nondelayed_dep['mae']:.4f}",
            f"{nondelayed_dep['rmse']:.4f}",
            f"{nondelayed_dep['r2']:.4f}",
            f"{nondelayed_arr['mean_pred']:.2f}",
            f"{nondelayed_arr['mean_target']:.2f}",
            f"{nondelayed_dep['mean_pred']:.2f}",
            f"{nondelayed_dep['mean_target']:.2f}",
        ])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow(["SUMMARY (raw metric/value, matches summary CSV)"])
        writer.writerow(["=" * 80])
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])
    
    print(f"\n✓ Results saved to:")
    print(f"  - {model_path}")
    print(f"  - {history_csv}")
    print(f"  - {results_table_csv}")
    
    # Rename checkpoint directory to match model filename
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        from pathlib import Path
        old_dir = Path(checkpoint_dir)
        # Extract model filename without extension
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        # Create new directory name
        new_dir = old_dir.parent / model_basename
        if old_dir != new_dir and not new_dir.exists():
            try:
                old_dir.rename(new_dir)
                print(f"\n✓ Renamed checkpoint directory:")
                print(f"  From: {old_dir}")
                print(f"  To: {new_dir}")
                # Update latest_run.txt
                latest_file = old_dir.parent / "latest_run.txt"
                if latest_file.exists():
                    with open(latest_file, "w", encoding="utf-8") as f:
                        f.write(str(new_dir))
            except Exception as e:
                print(f"\n⚠ Could not rename checkpoint directory: {e}")
    
    # Download files to local device (only in Colab)
    if IN_COLAB and colab_files is not None:
        print("\n[DOWNLOAD] Downloading files to local device...")
        
        # Files to download: model, history, results table, and checkpoints
        files_to_download = [
            model_path,
            history_csv,
            results_table_csv,
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
    parser = argparse.ArgumentParser(description="Three-stage CNN (with optional DP-SGD) with epsilon tracking")
    parser.add_argument('--data_source', type=str, default='cdata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=48)
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
    parser.add_argument('--exclude_time_features', default=True, action='store_true', help='Exclude time features (hour, day of week) from input')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--stage3_epochs', type=int, default=20, help='Epochs for non-delayed regressor')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--freeze_encoder_stage2', action='store_true', default=False, help='Freeze encoder during Stage 2 (default: trainable)')
    parser.add_argument('--dp', default=False, action='store_true', help='Enable DP-SGD')
    parser.add_argument('--dp_accountant',type=str,default='rdp',choices=['rdp', 'prv', 'gdp'],help="Opacus privacy accountant. 'prv' can be memory-heavy and may crash on large runs; 'rdp' is usually much lighter.")    
    parser.add_argument(
        '--epsilon',
        type=float,
        default=15,
        help='Target epsilon (used for tracking; and with --epsilonfixed for noise calibration)'
    )
    # Backward-compatible alias (do not advertise)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=0, help='Fixed noise multiplier for DP-SGD (lower=less noise, less privacy)')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Max gradient norm for clipping (higher allows larger gradients)')
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--epsilon_tolerance', type=float, default=0.05)
    parser.add_argument('--model_path', type=str, default='cnn_dp_three_stage.pth')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default="auto",
        help=(
            "Where to save/load stage checkpoints. "
            "Use 'auto' to create a new per-run subfolder under ./checkpoints, "
            "use 'latest' to reuse the most recent run folder, "
            "or pass an explicit folder name/path."
        ),
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random)')
    parser.add_argument('--balance_50_50', action='store_true', default=False, help='Apply random undersampling to achieve 50-50 class balance')
    parser.add_argument(
        '--epsilonfixed',
        dest='epsilonfixed',
        action='store_true',
        help='Enable epsilon-calibrated DP-SGD (uses Opacus make_private_with_epsilon)'
    )
    # Backward-compatible alias (do not advertise)
    parser.add_argument('--make_private', dest='epsilonfixed', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--no-epsilonfixed', dest='epsilonfixed', action='store_false', help='Disable epsilon-calibrated mode (use fixed noise multiplier instead)')
    return parser.parse_args()


def main() -> None:
    global CHECKPOINT_DIR
    args = parse_args()

    # Always pick checkpoint directory from args, so re-runs can resume consistently.
    CHECKPOINT_DIR = setup_checkpoint_directory(args.checkpoint_dir)
    
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
    
    # Optionally exclude time features (last 2 columns: hour and day_of_week)
    if args.exclude_time_features:
        print(f"\n[FEATURE FILTERING] Excluding time features from input")
        print(f"  Original feature dimension: {train_inputs.shape[2]}")
        # Assuming last 2 features are time-based (hour, day_of_week)
        train_inputs = train_inputs[:, :, :-2]
        val_inputs = val_inputs[:, :, :-2]
        test_inputs = test_inputs[:, :, :-2]
        print(f"  New feature dimension: {train_inputs.shape[2]}")
    
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
    
    # Comprehensive data validation
    print(f"\n[DATA VALIDATION] Checking data quality...")
    print(f"  Train features shape: {train_x.shape}")
    print(f"  Train targets (reg) shape: {train_y_reg.shape}")
    print(f"  Train targets (cls) shape: {train_y_cls.shape}")
    
    # Check for NaN/Inf
    print(f"  Train features: NaN={torch.isnan(train_x).any().item()}, Inf={torch.isinf(train_x).any().item()}")
    print(f"  Train reg targets: NaN={torch.isnan(train_y_reg).any().item()}, Inf={torch.isinf(train_y_reg).any().item()}")
    print(f"  Train cls targets: NaN={torch.isnan(train_y_cls).any().item()}, Inf={torch.isinf(train_y_cls).any().item()}")
    
    # Target statistics (scaled)
    print(f"  Train reg targets (scaled): min={train_y_reg.min().item():.3f}, max={train_y_reg.max().item():.3f}, mean={train_y_reg.mean().item():.3f}, std={train_y_reg.std().item():.3f}")
    print(f"  Train cls targets: min={train_y_cls.min().item():.3f}, max={train_y_cls.max().item():.3f}, unique values={train_y_cls.unique().tolist()}")
    
    # Per-channel distribution for regression targets
    if train_y_reg.dim() >= 2 and train_y_reg.shape[-1] == 2:
        print(f"  Arrival channel (scaled): min={train_y_reg[..., 0].min().item():.3f}, max={train_y_reg[..., 0].max().item():.3f}, mean={train_y_reg[..., 0].mean().item():.3f}")
        print(f"  Departure channel (scaled): min={train_y_reg[..., 1].min().item():.3f}, max={train_y_reg[..., 1].max().item():.3f}, mean={train_y_reg[..., 1].mean().item():.3f}")
    
    # Check classification label distribution per channel
    if train_y_cls.dim() >= 2 and train_y_cls.shape[-1] == 2:
        arr_delayed = train_y_cls[..., 0].mean().item()
        dep_delayed = train_y_cls[..., 1].mean().item()
        print(f"  Per-channel delayed rate: Arrival={arr_delayed:.2%}, Departure={dep_delayed:.2%}")
    
    # Scaler diagnostics
    print(f"\n[SCALER] Checking normalization parameters...")
    print(f"  Scaler mean: {scaler.mean}")
    print(f"  Scaler std: {scaler.std}")
    
    # Test denormalization on a sample
    sample_scaled = train_y_reg[0, 0, :].cpu().numpy()  # First node, both channels
    sample_denorm = scaler.inverse_transform(sample_scaled.reshape(1, -1))
    print(f"  Test denormalization:")
    print(f"    Scaled: {sample_scaled}")
    print(f"    Denormalized: {sample_denorm.flatten()}")
    print(f"    Expected range: 0-100 min (negative means early arrival)")
    
    print(f"  ✓ Data validation complete")
    
   
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,  # Increased from 16 for better capacity
        regressor_extra_layer=True,
        seq_len=args.seq_len,
    ).to(device)
    
    # Build node-level datasets/loaders for Opacus FIRST (before calculating steps).
    train_x_flat = _flatten_node_level(train_x)
    train_y_cls_flat = _flatten_node_level(train_y_cls)
    train_y_reg_flat = _flatten_node_level(train_y_reg)

    val_x_flat = _flatten_node_level(val_x)
    val_y_cls_flat = _flatten_node_level(val_y_cls)
    val_y_reg_flat = _flatten_node_level(val_y_reg)

    train_ds = TensorDataset(train_x_flat, train_y_cls_flat, train_y_reg_flat)
    val_ds = TensorDataset(val_x_flat, val_y_cls_flat, val_y_reg_flat)

    # Calculate steps based on ACTUAL flattened dataset size (node-level samples)
    total_samples = len(train_ds)  # Node-level count, NOT graph-level
    sample_rate = args.batch_size / total_samples
    # Note: DPDataLoader ignores drop_last, so use floor division for accurate step count
    steps_per_epoch = total_samples // args.batch_size
    total_epochs = args.stage1_epochs + args.stage2_epochs + args.stage3_epochs
    total_steps = total_epochs * steps_per_epoch
    
    if args.dp or args.epsilonfixed:
        print(f"\nDIFFERENTIAL PRIVACY CONFIGURATION:")
        if args.epsilonfixed:
            print(f"  Mode: EPSILON-CALIBRATED (make_private_with_epsilon)")
            print(f"  Target epsilon: {args.epsilon:.3f}")
            print(f"  Noise multiplier will be auto-calibrated by Opacus")
        else:
            print(f"  Mode: FIXED-NOISE (make_private)")
            print(f"  Noise multiplier (σ): {args.noise_multiplier:.3f}")
            print(f"  Tracking target epsilon: {args.epsilon:.3f}")
        print(f"  Accountant: {args.dp_accountant}")
        print(f"  Sampling: Without-replacement (all samples per epoch)")
        print(f"  Training samples (node-level): {total_samples}")
        print(f"  Sample rate per step (q): {sample_rate:.6f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total epochs (all stages): {total_epochs}")
        print(f"  Expected total steps: {total_steps} ({steps_per_epoch} steps/epoch × {total_epochs} epochs)")
        print(f"  Note: Privacy accounting is conservative (actual privacy may be better)")
        
        # Calculate expected noise scale for diagnostics (only for fixed-noise mode)
        if not args.epsilonfixed:
            noise_scale = args.noise_multiplier * args.max_grad_norm / args.batch_size
            print(f"\n[DP DIAGNOSTICS]")
            print(f"  Noise scale: {noise_scale:.6f} (noise_multiplier × max_grad_norm / batch_size)")
            print(f"  For good learning, gradient norms should be > {noise_scale * 3:.6f} (3x noise scale)")
            print(f"  Max gradient norm (clip threshold): {args.max_grad_norm}")
        else:
            print(f"\n[DP DIAGNOSTICS]")
            print(f"  Noise multiplier will be calibrated by Opacus after initialization...")
            print(f"  Max gradient norm (clip threshold): {args.max_grad_norm}")
        
        args.sample_rate = sample_rate

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # Per-channel (arrival, departure) positive rate aggregated across all airports/nodes.
    cls_pos_rate = train_y_cls_flat.float().mean(dim=0)  # shape: [2]
    pos_weight = (1.0 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)

    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Train samples (graphs): {len(train_x)}")
    print(f"Val samples (graphs): {len(val_x)}")
    print(f"Test samples (graphs): {len(test_x)}")
    print(f"Train examples (nodes): {len(train_ds)}")
    print(f"Class balance (delayed): {cls_pos_rate.mean().item():.2%}")

    privacy_engine = None
    dp_enabled = bool(getattr(args, "dp", False) or getattr(args, "epsilonfixed", False))
    tracking_target_epsilon = float(args.epsilon) if dp_enabled else 0.0
    effective_noise_multiplier = float(args.noise_multiplier) if dp_enabled else 0.0
    if dp_enabled:
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for --dp but is not installed. "
                "Install with: pip install opacus"
            )
        if ModuleValidator is not None:
            model = ModuleValidator.fix(model)
            ModuleValidator.validate(model, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        accountant = str(args.dp_accountant)
        if getattr(args, "epsilonfixed", False) and accountant != "rdp":
            print("[DP WARNING] --epsilonfixed uses make_private_with_epsilon; forcing --dp_accountant rdp for noise calibration.")
            accountant = "rdp"
        privacy_engine = PrivacyEngine(accountant=accountant)

        if getattr(args, "epsilonfixed", False):
            if not hasattr(privacy_engine, "make_private_with_epsilon"):
                raise RuntimeError(
                    "Your installed Opacus does not support PrivacyEngine.make_private_with_epsilon. "
                    "Upgrade Opacus (e.g., pip install -U opacus) or run without --epsilonfixed and use --noise_multiplier."
                )
            print(f"\n[EPSILON-CALIBRATED DP] Calling make_private_with_epsilon:")
            print(f"  target_epsilon={args.epsilon}, target_delta={args.target_delta}, epochs={total_epochs}")
            print(f"  max_grad_norm={args.max_grad_norm}")
            print(f"  Training samples (node-level): {len(train_ds)}, batch_size: {args.batch_size}")
            print(f"  Steps per epoch: {steps_per_epoch}")
            print(f"  Total steps: {total_steps}")
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=float(args.epsilon),
                target_delta=float(args.target_delta),
                epochs=total_epochs,
                max_grad_norm=float(args.max_grad_norm),
            )
            # After make_private_with_epsilon, noise_multiplier is on the optimizer, not privacy_engine
            try:
                effective_noise_multiplier = float(optimizer.noise_multiplier)
                noise_scale = effective_noise_multiplier * args.max_grad_norm / args.batch_size
                print(f"\n  ✓ CALIBRATION RESULTS:")
                print(f"    Noise multiplier (σ): {effective_noise_multiplier:.4f}")
                print(f"    Noise scale: {noise_scale:.6f} (σ × max_grad_norm / batch_size)")
                print(f"    For good learning, gradient norms should be > {noise_scale * 3:.6f} (3x noise scale)")
                print(f"    This noise is calibrated for {total_epochs} epochs total (across all 3 stages)")
            except AttributeError:
                print(f"  ⚠️ Could not retrieve calibrated noise_multiplier (not on optimizer)")
                effective_noise_multiplier = float(args.noise_multiplier)
        else:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=float(args.noise_multiplier),
                max_grad_norm=float(args.max_grad_norm),
            )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history_s1, stage1_time = train_stage1_opacus(
        model=model,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.stage1_epochs,
        lr=args.lr,
        pos_weight=pos_weight,
        patience=args.patience,
        target_delta=float(args.target_delta),
        target_epsilon=tracking_target_epsilon,
    )

    # Stage 2 delayed-sample diagnostics (sample-level, not node-level)
    cls_delayed_per_sample = (train_y_cls >= args.class_threshold)
    while cls_delayed_per_sample.dim() > 1:
        cls_delayed_per_sample = cls_delayed_per_sample.any(dim=-1)
    delayed_samples_cls = int(cls_delayed_per_sample.sum().item())

    if scaler is not None and hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
        mean_np = np.array(scaler.mean, dtype=np.float32)
        std_np = np.array(scaler.std, dtype=np.float32)
        std_np = np.where(std_np == 0, 1.0, std_np)
        thr_scaled_np = (np.full_like(mean_np, args.delay_threshold, dtype=np.float32) - mean_np) / std_np
        thr_scaled_t = torch.tensor(thr_scaled_np, dtype=train_y_reg.dtype)
    else:
        thr_scaled_t = torch.tensor(args.delay_threshold, dtype=train_y_reg.dtype)

    reg_delayed_per_sample = (train_y_reg > thr_scaled_t)
    while reg_delayed_per_sample.dim() > 1:
        reg_delayed_per_sample = reg_delayed_per_sample.any(dim=-1)
    delayed_samples_reg = int(reg_delayed_per_sample.sum().item())

    print(f"\n[STAGE 2 DIAGNOSTIC] Delayed samples by cls threshold: {delayed_samples_cls}/{len(train_x)}")
    print(f"[STAGE 2 DIAGNOSTIC] Delayed samples by reg threshold: {delayed_samples_reg}/{len(train_x)}")
    
    history_s2, stage2_time = train_stage2_opacus(
        model=model,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.stage2_epochs,
        lr=args.lr,
        scaler=scaler,
        delay_threshold=float(args.delay_threshold),
        patience=args.patience,
        target_delta=float(args.target_delta),
        target_epsilon=tracking_target_epsilon,
        freeze_encoder=args.freeze_encoder_stage2,
    )

    # Preserve delayed regressor after Stage 2 (do not swap modules; keep optimizer valid).
    base_model = _unwrap_opacus_model(model)
    base_model.regressor_delayed = copy.deepcopy(base_model.regressor).to(device)
    setattr(model, "regressor_delayed", base_model.regressor_delayed)

    history_s3, stage3_time = train_stage3_opacus(
        model=model,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.stage3_epochs,
        lr=args.lr,
        scaler=scaler,
        delay_threshold=float(args.delay_threshold),
        patience=args.patience,
        target_delta=float(args.target_delta),
        target_epsilon=tracking_target_epsilon,
    )

    base_model.regressor_nondelayed = copy.deepcopy(base_model.regressor).to(device)
    setattr(model, "regressor_nondelayed", base_model.regressor_nondelayed)

    combined_history = history_s1 + history_s2 + history_s3

    if dp_enabled and privacy_engine is not None:
        final_epsilon = _safe_get_epsilon(privacy_engine, float(args.target_delta))
        final_delta = float(args.target_delta)
    else:
        final_epsilon = 0.0
        final_delta = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_val = 0.0 if not dp_enabled else float(effective_noise_multiplier)
    run_tag = _run_tag(
        train_script=__file__,
        data_source=str(args.data_source),
        noise_multiplier=sigma_val,
        dp_enabled=dp_enabled,
    )

    # Always enforce reproducible naming (preserve directory if provided).
    default_model_path = os.path.join(CHECKPOINT_DIR, "cnn_dp_three_stage.pth")
    if args.model_path == "cnn_dp_three_stage.pth":
        args.model_path = default_model_path
    args.model_path = _ensure_tagged_filename(
        args.model_path,
        run_tag=run_tag,
        timestamp=timestamp,
        suffix="model",
        ext=".pth",
        epsilon=final_epsilon,
    )

    print(f"\nOutput model will be saved to: {args.model_path}")

    final_evaluation(
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
        args.delay_threshold,
        args.model_path,
        run_tag,
        timestamp,
        combined_history,
        final_epsilon,
        final_delta,
        stage1_time,
        stage2_time,
        stage3_time,
        len(train_x),
        len(val_x),
        bool(args.dp),
        float(args.epsilon),
        sigma_val,
        artifact_prefix="train",
        seq_len=args.seq_len,
        args=args,
        checkpoint_dir=CHECKPOINT_DIR,
    )


def setup_checkpoint_directory(checkpoint_dir: str = 'auto') -> str:
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

    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    latest_file = base_dir / "latest_run.txt"

    def _write_latest(path: Path) -> None:
        try:
            latest_file.write_text(str(path), encoding="utf-8")
        except Exception:
            # Don't fail training just because we can't write the marker.
            pass

    # Resolve run directory
    ck = (checkpoint_dir or "auto").strip().lower()

    if ck in {"auto", "new"}:
        # Create a unique subfolder per run so parallel debugger runs don't overwrite checkpoints.
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}_{uuid.uuid4().hex[:6]}"
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_latest(run_dir)
    elif ck == "latest":
        if not latest_file.exists():
            raise FileNotFoundError(
                f"No latest run marker found at {latest_file}. "
                f"Run once with --checkpoint_dir auto to create it, or pass an explicit --checkpoint_dir."
            )
        txt = latest_file.read_text(encoding="utf-8").strip()
        candidate = Path(txt)
        run_dir = candidate if candidate.is_absolute() else (base_dir / candidate)
        if not run_dir.exists():
            raise FileNotFoundError(f"Latest run folder does not exist: {run_dir}")
    else:
        candidate = Path(checkpoint_dir)
        run_dir = candidate if candidate.is_absolute() else (base_dir / candidate)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_latest(run_dir)

    print(f"✓ Checkpoints for this run: {run_dir}")
    return str(run_dir)



# Global checkpoint directory - set at runtime
CHECKPOINT_DIR: str = ""


if __name__ == '__main__':
    main()