"""Single-stage CNN/TCN regressor with optional DP-SGD (Opacus) and epsilon tracking.

NOTES:
- Mirrors the data loading + sequence building flow from `threestagev4noise.py`.
- Trains ONE regressor on ALL samples (no classification stage, no delayed/non-delayed stages).
- Supports node-level labels (flattened) or graph-level labels (as returned by build_sequences).
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
    build_sequences,
    load_flight_data,
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


def _ensure_tagged_filename(path: str, *, run_tag: str, timestamp: str, suffix: str, ext: str) -> str:
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
    filename = f"{run_tag}_{suffix}_{timestamp}{ext}"
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


def _np_mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    mae = float(np.mean(np.abs(y_pred_f - y_true_f))) if y_true_f.size else 0.0
    rmse = float(np.sqrt(np.mean((y_pred_f - y_true_f) ** 2))) if y_true_f.size else 0.0
    return mae, rmse


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


class SequentialRegressor(nn.Module):
    """CNN/TCN encoder + fully-connected regressor head.

    This class intentionally shadows the imported `SequentialTwoStagePredictor` so the
    architecture can be changed without editing other files.

        Expected input: `data.x` shaped either
            - [num_nodes, in_channels] where in_channels = seq_len * feature_dim, or
            - [num_nodes, seq_len, feature_dim].

        Output is node-level regression:
            - regressor preds: [num_nodes, out_channels]
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

        self.dropout_reg = nn.Dropout(p=0.1)

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
        raise RuntimeError("Classifier stage removed; use forward()/forward_regressor().")

    def forward_regressor(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.dropout_reg(hidden))

    def forward(self, data: Data) -> torch.Tensor:
        hidden = self._encode_x(data.x)
        return self.forward_regressor(hidden)


def _unwrap_opacus_model(model: nn.Module) -> nn.Module:
    """Return the underlying nn.Module if model is Opacus-wrapped."""
    return getattr(model, "_module", model)


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


def train_regressor_opacus(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    privacy_engine: Optional[object],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    target_delta: float,
    target_epsilon: float,
) -> Tuple[List[Dict], float]:
    """Single-stage: node-level delay regression on ALL samples."""
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("TRAINING: SINGLE-STAGE DELAY REGRESSOR (OPACUS optional)")
    print("=" * 80)

    base = _unwrap_opacus_model(model)

    for p in base.parameters():
        p.requires_grad = True

    _set_optimizer_hparams(optimizer, lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(reduction="mean", delta=2.0)

    history: List[Dict] = []
    best_val = float("inf")
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    early_stopping = EarlyStopping(patience=patience, mode="min")

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_losses: List[float] = []

        for batch_x, batch_y_reg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden = base._encode_x(batch_x)
            preds = base.forward_regressor(hidden)
            loss = loss_fn(preds, batch_y_reg)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch_x, batch_y_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_reg = batch_y_reg.to(device)
                hidden = base._encode_x(batch_x)
                preds = base.forward_regressor(hidden)
                val_losses.append(float(loss_fn(preds, batch_y_reg).item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        epoch_time = time.time() - epoch_start_time
        current_epsilon = (
            float(privacy_engine.get_epsilon(target_delta))
            if privacy_engine is not None
            else float("inf")
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epsilon": current_epsilon,
                "delta": target_delta if privacy_engine is not None else 0.0,
                "epoch_time_seconds": epoch_time,
            }
        )

        eps_str = (
            f"ε: {current_epsilon:.3f}/{target_epsilon}" if privacy_engine is not None else "No DP"
        )
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"{eps_str} | Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "encoder": copy.deepcopy(base.encoder.state_dict()),
                "regressor": copy.deepcopy(base.regressor.state_dict()),
            }
            print("  ✓ New best checkpoint")

        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        base.encoder.load_state_dict(best_state["encoder"])
        base.regressor.load_state_dict(best_state["regressor"])

    stage_time = time.time() - stage_start_time
    print(f"\nTraining completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    return history, stage_time

def final_evaluation(
    model: SequentialRegressor,
    edge_indices: Tuple,
    device: torch.device,
    scaler,
    horizons: List[int],
    delay_dim: int,
    num_nodes: int,
    test_x: torch.Tensor,
    test_y_reg: torch.Tensor,
    delay_threshold: float,
    model_path: str,
    run_tag: str,
    timestamp: str,
    histories: List[Dict],
    final_epsilon: float,
    final_delta: float,
    train_time: float,
    train_samples: int,
    val_samples: int,
    dp_enabled: bool,
    target_epsilon: float,
    noise_multiplier: float,
    *,
    save_model: bool = True,
    artifact_prefix: str = "train",
) -> None:
    """Final evaluation and export for single-stage regressor."""
    model = _unwrap_opacus_model(model)  # type: ignore[assignment]
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    print(f"Test samples: {len(test_x)}")
    
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    if save_model:
        to_save = {
            'encoder': model.encoder.state_dict(),
            'regressor': model.regressor.state_dict(),
            'final_epsilon': float(final_epsilon),
            'final_delta': float(final_delta),
            'target_epsilon': float(target_epsilon),
            'epsilon_exceeded': final_epsilon > float(target_epsilon) if dp_enabled else False,
            'run_tag': str(run_tag),
            'timestamp': str(timestamp),
            'dp_enabled': bool(dp_enabled),
            'noise_multiplier': float(noise_multiplier),
        }
        torch.save(to_save, model_path)
        checkpoint = to_save
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load weights supporting multiple checkpoint formats.
    # Preferred format: separate submodule state_dicts.
    if isinstance(checkpoint, dict) and 'encoder' in checkpoint and 'regressor' in checkpoint:
        model.encoder.load_state_dict(checkpoint['encoder'])
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
    
    reg_list: List[np.ndarray] = []
    targets_reg_list: List[np.ndarray] = []
    
    print("[EVALUATION] Processing test samples...")
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )
            node_reg = model(data)
            reg_list.append(node_reg.cpu().numpy())
            targets_reg_list.append(test_y_reg[i].cpu().numpy())
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")
    
    test_reg_preds = np.concatenate(reg_list, axis=0)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)
    
    print(f"\n[DENORMALIZATION] Checking predictions...")
    print(f"  Predictions shape: {test_reg_preds.shape}")
    print(f"  Predictions (scaled): min={test_reg_preds.min():.3f}, max={test_reg_preds.max():.3f}, mean={test_reg_preds.mean():.3f}")
    print(f"  Test targets (scaled): min={test_reg_targets.min():.3f}, max={test_reg_targets.max():.3f}, mean={test_reg_targets.mean():.3f}")
    
    if scaler is not None:
        print(f"  Applying inverse transform with scaler...")
        print(f"    Scaler mean: {scaler.mean}, std: {scaler.std}")
        preds_denorm = scaler.inverse_transform(test_reg_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
        print(f"  After denormalization:")
        print(f"    Predictions: min={preds_denorm.min():.2f}, max={preds_denorm.max():.2f}, mean={preds_denorm.mean():.2f}")
        print(f"    Targets: min={targets_denorm.min():.2f}, max={targets_denorm.max():.2f}, mean={targets_denorm.mean():.2f}")
    else:
        preds_denorm = test_reg_preds
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
    mae_overall, rmse_overall = _np_mae_rmse(targets_denorm, preds_denorm)
    
    print(f"\nREGRESSION (delayed flights > {delay_threshold} min):")
    print(f"  MAE: {mae_delayed:.4f} min | RMSE: {rmse_delayed:.4f} min")
    print(f"  Number of delayed samples: {delayed_mask.sum()}")
    
    print(f"\nREGRESSION (non-delayed flights <= {delay_threshold} min):")
    print(f"  MAE: {mae_nondelayed:.4f} min | RMSE: {rmse_nondelayed:.4f} min")
    print(f"  Number of non-delayed samples: {nondelayed_mask.sum()}")
    
    print("\nREGRESSION (overall):")
    print(f"  MAE: {mae_overall:.4f} min | RMSE: {rmse_overall:.4f} min")
    
    # Visualize regression results (optional)
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating regression plots...")
        try:
            if targets_denorm.ndim > 1 and targets_denorm.shape[1] > 1:
                test_reg_true = targets_denorm[:, 0].reshape(-1)
                test_reg_pred = preds_denorm[:, 0].reshape(-1)
            else:
                test_reg_true = targets_denorm.reshape(-1)
                test_reg_pred = preds_denorm.reshape(-1)

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
            print(f"  ✗ Error generating regression visualization: {e}")
    
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
    print(f"  Total: {train_time:.2f}s ({train_time/60:.2f} min)")
    
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    
    out_dir = os.path.dirname(model_path)
    history_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_history_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_history_{timestamp}.csv"
    summary_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_summary_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_summary_{timestamp}.csv"
    results_table_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_results_table_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_results_table_{timestamp}.csv"
    
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
            'regression_mae_delayed': mae_delayed,
            'regression_rmse_delayed': rmse_delayed,
            'regression_mae_nondelayed': mae_nondelayed,
            'regression_rmse_nondelayed': rmse_nondelayed,
            'regression_mae_overall': mae_overall,
            'regression_rmse_overall': rmse_overall,
            'num_delayed_samples': int(delayed_mask.sum()),
            'num_nondelayed_samples': int(nondelayed_mask.sum()),
            'target_epsilon': float(target_epsilon),
            'final_epsilon': final_epsilon,
            'epsilon_exceeded': (final_epsilon > float(target_epsilon)) if dp_enabled else False,
            'epsilon_overshoot': max(0.0, final_epsilon - float(target_epsilon)) if dp_enabled else 0.0,
            'final_delta': final_delta,
            'train_time_seconds': train_time,
            'train_time_minutes': train_time / 60,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': len(test_x),
        }
        for k, v in summary.items():
            writer.writerow([k, v])

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
        writer.writerow(["Horizons", ",".join(str(h) for h in horizons)])
        writer.writerow(["Delay Threshold", f"{delay_threshold} min (Delayed if > threshold)"])
        writer.writerow(["Classification Threshold", "N/A (single-stage regression)"])
        writer.writerow(["DP Enabled", dp_enabled])
        writer.writerow(["Noise Multiplier (sigma)", float(noise_multiplier)])
        writer.writerow(["Final Epsilon", f"{final_epsilon:.3f}"]) 
        writer.writerow(["Target Epsilon", f"{float(target_epsilon):.3f}"])
        writer.writerow(["Final Delta", f"{final_delta:.2e}"])
        writer.writerow(["Epsilon Exceeded", summary['epsilon_exceeded']])
        writer.writerow(["Train Samples", train_samples])
        writer.writerow(["Val Samples", val_samples])
        writer.writerow(["Test Samples", len(test_x)])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow(["REGRESSION METRICS"])
        writer.writerow(["=" * 80])
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Overall MAE (min)", f"{mae_overall:.4f}"])
        writer.writerow(["Overall RMSE (min)", f"{rmse_overall:.4f}"])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow(["OVERALL SUMMARY TABLE"])
        writer.writerow(["=" * 80])
        writer.writerow([
            "Epsilon",
            "Overall MAE (min)",
            "Arrival MAE (min)",
            "Departure MAE (min)",
        ])
        writer.writerow([
            f"{final_epsilon:.2f}",
            f"{mae_overall:.4f}",
            f"{overall_arr['mae']:.4f}",
            f"{overall_dep['mae']:.4f}",
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
    print(f"  - {summary_csv}")
    print(f"  - {results_table_csv}")
    
    # Download files to local device (only in Colab)
    if IN_COLAB and colab_files is not None:
        print("\n[DOWNLOAD] Downloading files to local device...")
        
        # Files to download: model, history, summary, and checkpoints
        files_to_download = [
            model_path,
            history_csv,
            summary_csv,
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
    parser = argparse.ArgumentParser(description="Single-stage CNN/TCN regressor (optional DP-SGD) with epsilon tracking")
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'])
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
    # class_threshold removed (no classification stage)
    parser.add_argument('--use_node_level', action='store_true', default=True, help='Use node-level labels')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dp', default=True, action='store_true', help='Enable DP-SGD')
    parser.add_argument('--target_epsilon', type=float, default=15.0, help='Target epsilon for tracking (not used for computing noise)')
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=2, help='Fixed noise multiplier for DP-SGD (lower=less noise, less privacy)')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Max gradient norm for clipping (higher allows larger gradients)')
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--epsilon_tolerance', type=float, default=0.05)
    parser.add_argument('--model_path', type=str, default='cnn_dp_regressor.pth')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='auto',
        help=(
            "Where to save/load stage checkpoints. "
            "Use 'auto' to create a new per-run subfolder under ./checkpoints, "
            "use 'latest' to reuse the most recent run folder, "
            "or pass an explicit folder name/path."
        ),
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random)')
    parser.add_argument('--balance_50_50', action='store_true', default=False, help='Apply random undersampling to achieve 50-50 class balance')
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

    # NOTE: classification labels are ignored in this script (single-stage regression).
    
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
    
    model = SequentialRegressor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,  # Increased from 16 for better capacity
        regressor_extra_layer=True,
        seq_len=args.seq_len,
    ).to(device)
    
    total_samples = len(train_x)
    sample_rate = args.batch_size / total_samples
    steps_per_epoch = int(np.ceil(total_samples / args.batch_size))
    total_steps = int(args.epochs) * steps_per_epoch
    
    if args.dp:
        print(f"\nDIFFERENTIAL PRIVACY CONFIGURATION:")
        print(f"  Noise multiplier (σ): {args.noise_multiplier:.3f}")
        print(f"  Sampling: Without-replacement (all samples per epoch)")
        print(f"  Sample rate per step (q): {sample_rate:.4f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Expected total steps: {total_steps} ({steps_per_epoch} steps/epoch × {int(args.epochs)} epochs)")
        print(f"  Target epsilon: {args.target_epsilon:.3f}")
        print(f"  Note: Privacy accounting is conservative (actual privacy may be better)")
        
        # Calculate expected noise scale for diagnostics
        noise_scale = args.noise_multiplier * args.max_grad_norm / args.batch_size
        print(f"\n[DP DIAGNOSTICS]")
        print(f"  Noise scale: {noise_scale:.6f} (noise_multiplier × max_grad_norm / batch_size)")
        print(f"  For good learning, gradient norms should be > {noise_scale * 3:.6f} (3x noise scale)")
        print(f"  Max gradient norm (clip threshold): {args.max_grad_norm}")
        
        args.sample_rate = sample_rate
    
    # Build node-level datasets/loaders for Opacus.
    train_x_flat = _flatten_node_level(train_x)
    train_y_reg_flat = _flatten_node_level(train_y_reg)

    val_x_flat = _flatten_node_level(val_x)
    val_y_reg_flat = _flatten_node_level(val_y_reg)

    train_ds = TensorDataset(train_x_flat, train_y_reg_flat)
    val_ds = TensorDataset(val_x_flat, val_y_reg_flat)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Train samples (graphs): {len(train_x)}")
    print(f"Val samples (graphs): {len(val_x)}")
    print(f"Test samples (graphs): {len(test_x)}")
    print(f"Train examples (nodes): {len(train_ds)}")
    print("Task: Regression on ALL samples")

    privacy_engine = None
    if args.dp:
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "Opacus is required for --dp but is not installed. "
                "Install with: pip install opacus"
            )
        if ModuleValidator is not None:
            model = ModuleValidator.fix(model)
            ModuleValidator.validate(model, strict=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        privacy_engine = PrivacyEngine()
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=float(args.noise_multiplier),
            max_grad_norm=float(args.max_grad_norm),
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    history, train_time = train_regressor_opacus(
        model=model,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(args.epochs),
        lr=args.lr,
        patience=args.patience,
        target_delta=float(args.target_delta),
        target_epsilon=float(args.target_epsilon),
    )

    combined_history = history

    if args.dp and privacy_engine is not None:
        final_epsilon = float(privacy_engine.get_epsilon(float(args.target_delta)))
        final_delta = float(args.target_delta)
    else:
        final_epsilon = float("inf")
        final_delta = 0.0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_val = float(args.noise_multiplier) if args.dp else 0.0
    run_tag = _run_tag(
        train_script=__file__,
        data_source=str(args.data_source),
        noise_multiplier=float(args.noise_multiplier),
        dp_enabled=bool(args.dp),
    )

    # Always enforce reproducible naming (preserve directory if provided).
    default_model_path = os.path.join(CHECKPOINT_DIR, "cnn_dp_regressor.pth")
    if args.model_path == "cnn_dp_regressor.pth":
        args.model_path = default_model_path
    args.model_path = _ensure_tagged_filename(
        args.model_path,
        run_tag=run_tag,
        timestamp=timestamp,
        suffix="model",
        ext=".pth",
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
        args.delay_threshold,
        args.model_path,
        run_tag,
        timestamp,
        combined_history,
        final_epsilon,
        final_delta,
        train_time,
        len(train_x),
        len(val_x),
        bool(args.dp),
        float(args.target_epsilon),
        sigma_val,
        artifact_prefix="train",
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