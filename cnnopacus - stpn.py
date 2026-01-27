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

# Optional STPN backbone (model.py)
try:
    from model import STPN as STPNBackbone  # type: ignore
    STPN_AVAILABLE = True
except Exception:
    STPNBackbone = None  # type: ignore[assignment]
    STPN_AVAILABLE = False


def _edge_index_to_random_walk(edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
    """Convert a PyG-style edge_index [2, E] into a dense random-walk matrix [V, V]."""
    if edge_index.numel() == 0:
        return torch.eye(int(num_nodes), device=device)
    ei = edge_index.to(device)
    adj = torch.zeros((int(num_nodes), int(num_nodes)), device=device, dtype=torch.float32)
    adj[ei[0].long(), ei[1].long()] = 1.0
    # Add self-loops for stability
    adj.fill_diagonal_(1.0)
    deg = adj.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adj / deg


def build_stpn_supports(
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> List[torch.Tensor]:
    """Build STPN supports list (random-walk matrices) from the three edge sets."""
    return [
        _edge_index_to_random_walk(edge_index_adj, num_nodes, device),
        _edge_index_to_random_walk(edge_index_od, num_nodes, device),
        _edge_index_to_random_walk(edge_index_od_t, num_nodes, device),
    ]


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

        # DEBUG: Log encoder selection
        print(f"[SequentialTwoStagePredictor] in_channels={self.in_channels}, seq_len={self.seq_len}, "
              f"feature_dim={self.feature_dim}")

        c1 = max(16, self.hidden_channels // 2)
        c2 = max(16, self.hidden_channels)

        # Preferred encoder: TCN-style temporal Conv1d (dilations + residual blocks)
        # If seq_len can't be inferred, fall back to a simple flattened Conv1d.
        if self.feature_dim is not None:
            print(f"[Encoder] Using TCN encoder (sequence-aware: [N, {self.seq_len}, {self.feature_dim}])")
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
            print(f"[Encoder] FALLBACK to plain Conv1d (flat input: [N, 1, {self.in_channels}])")
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
        print(f"[_encode_x] Using FALLBACK Conv1d path with input shape {x.shape}")
        if x.dim() == 3:
            x_flat = x.reshape(x.shape[0], -1)
        elif x.dim() == 2:
            x_flat = x
        else:
            x_flat = x.view(x.shape[0], -1)
        print(f"[_encode_x] Flattened to {x_flat.shape}, adding channel dim -> {x_flat.unsqueeze(1).shape}")
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


class STPNThreeStagePredictor(nn.Module):
    """STPN backbone + per-node classifier/regressor heads.

    Expects batch inputs shaped [B, V, T, F] (from build_sequences*), and builds
    node embeddings via STPN's spatio-temporal blocks. Heads are simple MLP/Linear
    layers over the node embeddings.

    This keeps the same stage-1/2/3 semantics used elsewhere in this script.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        seq_len: int,
        out_channels: int,
        hidden_channels: int = 64,
        h_layers: int = 2,
        emb_size: int = 16,
        heads: int = 4,
        time_d: int = 4,
        dropout: float = 0.1,
        support_len: int = 3,
        order: int = 2,
        use_se: bool = True,
    ) -> None:
        super().__init__()
        if not STPN_AVAILABLE or STPNBackbone is None:
            raise ImportError("STPN backbone not available (failed to import from model.py)")

        # STPN expects a hidden_channels list of length h_layers+1
        hidden_list = [int(hidden_channels)] * (int(h_layers) + 1)

        self.stpn = STPNBackbone(
            h_layers=int(h_layers),
            in_channels=int(feature_dim),
            hidden_channels=hidden_list,
            out_channels=int(hidden_list[-1]),
            emb_size=int(emb_size),
            dropout=float(dropout),
            time_d=int(time_d),
            heads=int(heads),
            support_len=int(support_len),
            order=int(order),
            use_se=bool(use_se),
            use_cov=False,
        )

        self.feature_dim = int(feature_dim)
        self.seq_len = int(seq_len)
        self.hidden_dim = int(hidden_list[-1])
        self.out_channels = int(out_channels)

        self.dropout_cls = nn.Dropout(p=float(dropout))
        self.dropout_reg = nn.Dropout(p=float(dropout))

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_channels),
        )

        # Keep the regressor simple so Stage 2/3 can clone and specialize.
        self.regressor = nn.Linear(self.hidden_dim, self.out_channels)

    def encode_graph(self, x_bvtf: torch.Tensor, t_in: torch.Tensor, supports: List[torch.Tensor]) -> torch.Tensor:
        """Return node embeddings [B, V, H].

        Accepts either:
        - [B, V, T, F] (preferred)
        - [B, V, T*F]  (flattened time/features, as produced by build_sequences_node_level)
        """
        # Some pipelines may inadvertently construct sparse tensors.
        # STPN expects dense/strided tensors.
        if getattr(x_bvtf, "layout", torch.strided) != torch.strided:
            x_bvtf = x_bvtf.to_dense()

        if x_bvtf.dim() == 3:
            b, v, c = x_bvtf.shape
            expected = int(self.seq_len) * int(self.feature_dim)
            if int(c) != expected:
                raise RuntimeError(
                    "STPNThreeStagePredictor expected x shaped [B, V, T, F] or [B, V, T*F]. "
                    f"Got 3D input with shape {tuple(x_bvtf.shape)} where last dim={c} but expected T*F={expected} "
                    f"(T={self.seq_len}, F={self.feature_dim})."
                )
            x_bvtf = x_bvtf.reshape(int(b), int(v), int(self.seq_len), int(self.feature_dim))
        elif x_bvtf.dim() != 4:
            raise RuntimeError(
                "STPNThreeStagePredictor expected x with 3 or 4 dims. "
                f"Got shape {tuple(x_bvtf.shape)}."
            )

        # STPN expects [B, F, V, T]
        x = x_bvtf.permute(0, 3, 1, 2).contiguous()
        # Run all STPN hidden layers (exclude the final_conv which is task-specific in the original paper)
        for i in range(self.stpn.h_layers + 1):
            x = self.stpn.convs[i](x, t_in, supports)
            if i < self.stpn.h_layers and getattr(self.stpn, "use_se", False):
                x = self.stpn.se[i](x)
        # x: [B, H, V, T] -> [B, V, H]
        return x.mean(dim=-1).permute(0, 2, 1).contiguous()

    def forward_classifier(
        self,
        x_bvtf: torch.Tensor,
        t_in: torch.Tensor,
        supports: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_bvh = self.encode_graph(x_bvtf, t_in, supports)
        logits_bvc = self.classifier(self.dropout_cls(hidden_bvh))
        return hidden_bvh, logits_bvc

    def forward_regressor(self, hidden_bvh: torch.Tensor) -> torch.Tensor:
        return self.regressor(self.dropout_reg(hidden_bvh))


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


def _make_time_labels(batch_size: int, length: int, device: torch.device) -> torch.Tensor:
    # STPN's learnEmbedding expects floating inputs.
    base = torch.arange(int(length), device=device, dtype=torch.float32)
    return base.unsqueeze(0).expand(int(batch_size), -1).contiguous()


def train_regression_opacus(
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
    *,
    supports: Optional[List[torch.Tensor]] = None,
    seq_len: Optional[int] = None,
) -> Tuple[List[Dict], float]:
    """Regression-only training (no stages).

    Works for both backbones:
    - TCN: batches are node-level [B, ...]
    - STPN: batches are graph-level [B, V, T, F]
    """
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("REGRESSION-ONLY TRAINING (OPACUS)")
    print("=" * 80)

    base = _unwrap_opacus_model(model)
    for p in base.parameters():
        p.requires_grad = True

    _set_optimizer_hparams(optimizer, lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(reduction="mean", delta=1.0)

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, torch.Tensor]] = None
    early_stopping = EarlyStopping(patience=patience, mode="min")

    for epoch in range(1, int(epochs) + 1):
        epoch_start_time = time.time()
        model.train()
        train_losses: List[float] = []
        steps = 0

        for batch in train_loader:
            steps += 1
            optimizer.zero_grad(set_to_none=True)

            if hasattr(base, "stpn"):
                if supports is None:
                    raise ValueError("supports must be provided for STPN regression training")
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                t_in = _make_time_labels(batch_x.shape[0], int(seq_len or batch_x.shape[2]), device)
                hidden = base.encode_graph(batch_x, t_in, supports)  # type: ignore[attr-defined]
                preds = base.forward_regressor(hidden)
                loss = loss_fn(preds.reshape(-1, preds.shape[-1]), batch_y.reshape(-1, batch_y.shape[-1]))
            else:
                batch_x, batch_y = batch
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                hidden = base._encode_x(batch_x)  # type: ignore[attr-defined]
                preds = base.forward_regressor(hidden)  # type: ignore[attr-defined]
                loss = loss_fn(preds, batch_y)

            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch in val_loader:
                if hasattr(base, "stpn"):
                    if supports is None:
                        raise ValueError("supports must be provided for STPN regression validation")
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    t_in = _make_time_labels(batch_x.shape[0], int(seq_len or batch_x.shape[2]), device)
                    hidden = base.encode_graph(batch_x, t_in, supports)  # type: ignore[attr-defined]
                    preds = base.forward_regressor(hidden)
                    loss = loss_fn(preds.reshape(-1, preds.shape[-1]), batch_y.reshape(-1, batch_y.shape[-1]))
                else:
                    batch_x, batch_y = batch
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    hidden = base._encode_x(batch_x)  # type: ignore[attr-defined]
                    preds = base.forward_regressor(hidden)  # type: ignore[attr-defined]
                    loss = loss_fn(preds, batch_y)

                val_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        epoch_time = time.time() - epoch_start_time
        current_epsilon = _safe_get_epsilon(privacy_engine, target_delta)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
                "val_loss": val_loss,
                "epsilon": current_epsilon,
                "delta": target_delta if privacy_engine is not None else 0.0,
                "epoch_time_seconds": epoch_time,
                "steps": steps,
            }
        )

        eps_str = f"ε: {current_epsilon:.3f}/{target_epsilon}" if privacy_engine is not None else "No DP"
        print(
            f"Epoch {epoch}/{epochs} | Train: {history[-1]['train_loss']:.4f} | Val: {val_loss:.4f} | "
            f"{eps_str} | Time: {epoch_time:.2f}s",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save only the regression-relevant weights
            if hasattr(base, "stpn"):
                best_state = {
                    "stpn": copy.deepcopy(base.stpn.state_dict()),  # type: ignore[attr-defined]
                    "regressor": copy.deepcopy(base.regressor.state_dict()),  # type: ignore[attr-defined]
                }
            else:
                best_state = {
                    "encoder": copy.deepcopy(base.encoder.state_dict()),  # type: ignore[attr-defined]
                    "regressor": copy.deepcopy(base.regressor.state_dict()),  # type: ignore[attr-defined]
                }
            print("  ✓ New best checkpoint")

        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        if hasattr(base, "stpn"):
            base.stpn.load_state_dict(best_state["stpn"])  # type: ignore[attr-defined]
            base.regressor.load_state_dict(best_state["regressor"])  # type: ignore[attr-defined]
        else:
            base.encoder.load_state_dict(best_state["encoder"])  # type: ignore[attr-defined]
            base.regressor.load_state_dict(best_state["regressor"])  # type: ignore[attr-defined]

    stage_time = time.time() - stage_start_time
    final_epsilon = _safe_get_epsilon(privacy_engine, target_delta)
    print(f"\nTraining completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    if privacy_engine is not None:
        print(f"Final ε: {final_epsilon:.3f} (target: {target_epsilon})")
    return history, stage_time


def final_evaluation_regression_only(
    model: nn.Module,
    *,
    edge_indices: Tuple,
    supports: Optional[List[torch.Tensor]],
    device: torch.device,
    scaler,
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
    save_model: bool = True,
    artifact_prefix: str = "train",
    seq_len: Optional[int] = None,
    args: Optional[object] = None,
    checkpoint_dir: Optional[str] = None,
    enable_visualization: bool = True,
) -> None:
    """Final regression-only evaluation and export (no classification, no staging)."""
    model = _unwrap_opacus_model(model)  # type: ignore[assignment]

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)

    if save_model:
        to_save: Dict[str, object] = {
            "final_epsilon": float(final_epsilon),
            "final_delta": float(final_delta),
            "target_epsilon": float(target_epsilon),
            "epsilon_exceeded": final_epsilon > float(target_epsilon) if dp_enabled else False,
            "run_tag": str(run_tag),
            "timestamp": str(timestamp),
            "dp_enabled": bool(dp_enabled),
            "noise_multiplier": float(noise_multiplier),
        }
        if hasattr(model, "stpn"):
            to_save["stpn"] = model.stpn.state_dict()  # type: ignore[attr-defined]
            to_save["regressor"] = model.regressor.state_dict()  # type: ignore[attr-defined]
        else:
            to_save["encoder"] = model.encoder.state_dict()  # type: ignore[attr-defined]
            to_save["regressor"] = model.regressor.state_dict()  # type: ignore[attr-defined]

        torch.save(to_save, model_path)
        checkpoint = to_save
    else:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Load regression-only weights
    if isinstance(checkpoint, dict):
        if hasattr(model, "stpn") and "stpn" in checkpoint:
            model.stpn.load_state_dict(checkpoint["stpn"])  # type: ignore[attr-defined]
            model.regressor.load_state_dict(checkpoint["regressor"])  # type: ignore[attr-defined]
        elif (not hasattr(model, "stpn")) and "encoder" in checkpoint:
            model.encoder.load_state_dict(checkpoint["encoder"])  # type: ignore[attr-defined]
            model.regressor.load_state_dict(checkpoint["regressor"])  # type: ignore[attr-defined]

    model.eval()

    reg_list: List[np.ndarray] = []
    targets_reg_list: List[np.ndarray] = []

    print("\n" + "=" * 80)
    print("FINAL REGRESSION TEST EVALUATION")
    print("=" * 80)
    print(f"Test samples: {len(test_x)}")

    with torch.no_grad():
        for i in range(len(test_x)):
            if hasattr(model, "stpn"):
                if supports is None:
                    raise ValueError("supports must be provided for STPN evaluation")
                x_bvtf = test_x[i].unsqueeze(0).to(device)
                t_in = _make_time_labels(1, int(seq_len or x_bvtf.shape[2]), device)
                hidden = model.encode_graph(x_bvtf, t_in, supports)  # type: ignore[attr-defined]
                node_reg = model.forward_regressor(hidden).squeeze(0)  # type: ignore[attr-defined]
            else:
                data = Data(
                    x=test_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_reg = model.forward(data)  # type: ignore[arg-type]

            reg_list.append(node_reg.cpu().numpy())
            targets_reg_list.append(test_y_reg[i].cpu().numpy())

    test_reg_preds = np.concatenate(reg_list, axis=0)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)

    print(f"\n[DENORMALIZATION] preds shape={test_reg_preds.shape}, targets shape={test_reg_targets.shape}")
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(test_reg_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
    else:
        preds_denorm = test_reg_preds
        targets_denorm = test_reg_targets

    # Match previous implementation: treat negative values as on time (0 min)
    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)

    preds_flat = preds_denorm.flatten()
    targets_flat = targets_denorm.flatten()

    delayed_mask = targets_flat > float(delay_threshold)
    nondelayed_mask = targets_flat <= float(delay_threshold)

    def _mae_rmse(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        mae = float(np.mean(np.abs(a - b))) if a.size else 0.0
        rmse = float(np.sqrt(np.mean((a - b) ** 2))) if a.size else 0.0
        return mae, rmse

    mae_overall, rmse_overall = _mae_rmse(preds_flat, targets_flat)
    mae_delayed, rmse_delayed = _mae_rmse(preds_flat[delayed_mask], targets_flat[delayed_mask])
    mae_nondelayed, rmse_nondelayed = _mae_rmse(preds_flat[nondelayed_mask], targets_flat[nondelayed_mask])

    # Optional: per-channel (arrival/departure) evaluation when out_channels == 2.
    arr_metrics: Optional[Dict[str, float]] = None
    dep_metrics: Optional[Dict[str, float]] = None
    if preds_denorm.ndim == 2 and targets_denorm.ndim == 2 and preds_denorm.shape[1] == 2 and targets_denorm.shape[1] == 2:
        arr_preds = preds_denorm[:, 0].astype(np.float64, copy=False)
        dep_preds = preds_denorm[:, 1].astype(np.float64, copy=False)
        arr_targets = targets_denorm[:, 0].astype(np.float64, copy=False)
        dep_targets = targets_denorm[:, 1].astype(np.float64, copy=False)

        arr_delayed = arr_targets > float(delay_threshold)
        dep_delayed = dep_targets > float(delay_threshold)
        arr_nondelayed = ~arr_delayed
        dep_nondelayed = ~dep_delayed

        arr_mae_o, arr_rmse_o = _mae_rmse(arr_preds, arr_targets)
        arr_mae_d, arr_rmse_d = _mae_rmse(arr_preds[arr_delayed], arr_targets[arr_delayed])
        arr_mae_nd, arr_rmse_nd = _mae_rmse(arr_preds[arr_nondelayed], arr_targets[arr_nondelayed])

        dep_mae_o, dep_rmse_o = _mae_rmse(dep_preds, dep_targets)
        dep_mae_d, dep_rmse_d = _mae_rmse(dep_preds[dep_delayed], dep_targets[dep_delayed])
        dep_mae_nd, dep_rmse_nd = _mae_rmse(dep_preds[dep_nondelayed], dep_targets[dep_nondelayed])

        arr_metrics = {
            "mae_overall": float(arr_mae_o),
            "rmse_overall": float(arr_rmse_o),
            "mae_delayed": float(arr_mae_d),
            "rmse_delayed": float(arr_rmse_d),
            "mae_nondelayed": float(arr_mae_nd),
            "rmse_nondelayed": float(arr_rmse_nd),
            "n_delayed": int(arr_delayed.sum()),
            "n_nondelayed": int(arr_nondelayed.sum()),
        }
        dep_metrics = {
            "mae_overall": float(dep_mae_o),
            "rmse_overall": float(dep_rmse_o),
            "mae_delayed": float(dep_mae_d),
            "rmse_delayed": float(dep_rmse_d),
            "mae_nondelayed": float(dep_mae_nd),
            "rmse_nondelayed": float(dep_rmse_nd),
            "n_delayed": int(dep_delayed.sum()),
            "n_nondelayed": int(dep_nondelayed.sum()),
        }

    print("\nREGRESSION (overall):")
    print(f"  MAE: {mae_overall:.4f} min | RMSE: {rmse_overall:.4f} min")
    print(f"  min/max target: {targets_flat.min():.2f}/{targets_flat.max():.2f}")
    print(f"  min/max pred:   {preds_flat.min():.2f}/{preds_flat.max():.2f}")

    if arr_metrics is not None and dep_metrics is not None:
        print("\nREGRESSION (per channel):")
        print(
            f"  Arrival  | MAE {arr_metrics['mae_overall']:.4f} | RMSE {arr_metrics['rmse_overall']:.4f}"
        )
        print(
            f"  Departure| MAE {dep_metrics['mae_overall']:.4f} | RMSE {dep_metrics['rmse_overall']:.4f}"
        )

    print(f"\nREGRESSION (delayed > {delay_threshold} min):")
    print(f"  MAE: {mae_delayed:.4f} min | RMSE: {rmse_delayed:.4f} min | n={int(delayed_mask.sum())}")

    if arr_metrics is not None and dep_metrics is not None:
        print(
            f"  Arrival  | MAE {arr_metrics['mae_delayed']:.4f} | RMSE {arr_metrics['rmse_delayed']:.4f} | n={arr_metrics['n_delayed']}"
        )
        print(
            f"  Departure| MAE {dep_metrics['mae_delayed']:.4f} | RMSE {dep_metrics['rmse_delayed']:.4f} | n={dep_metrics['n_delayed']}"
        )

    print(f"\nREGRESSION (non-delayed <= {delay_threshold} min):")
    print(f"  MAE: {mae_nondelayed:.4f} min | RMSE: {rmse_nondelayed:.4f} min | n={int(nondelayed_mask.sum())}")

    if arr_metrics is not None and dep_metrics is not None:
        print(
            f"  Arrival  | MAE {arr_metrics['mae_nondelayed']:.4f} | RMSE {arr_metrics['rmse_nondelayed']:.4f} | n={arr_metrics['n_nondelayed']}"
        )
        print(
            f"  Departure| MAE {dep_metrics['mae_nondelayed']:.4f} | RMSE {dep_metrics['rmse_nondelayed']:.4f} | n={dep_metrics['n_nondelayed']}"
        )

    # Export history + a comprehensive results table (evaluate_regression_v4-style)
    out_dir = os.path.dirname(model_path)
    epsilon_tag = f"_eps{final_epsilon:.2f}" if dp_enabled else ""
    history_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_history{epsilon_tag}_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_history{epsilon_tag}_{timestamp}.csv"
    results_table_csv = os.path.join(out_dir, f"{run_tag}_{artifact_prefix}_results_table{epsilon_tag}_{timestamp}.csv") if out_dir else f"{run_tag}_{artifact_prefix}_results_table{epsilon_tag}_{timestamp}.csv"

    if histories:
        with open(history_csv, "w", newline="") as f:
            fields = sorted({k for row in histories for k in row})
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(histories)
    # Prepare summary metrics (key/value)
    epsilon_exceeded = (final_epsilon > float(target_epsilon)) if dp_enabled else False
    summary: Dict[str, object] = {
        "regression_mae_delayed": float(mae_delayed),
        "regression_rmse_delayed": float(rmse_delayed),
        "regression_mae_nondelayed": float(mae_nondelayed),
        "regression_rmse_nondelayed": float(rmse_nondelayed),
        "regression_mae_overall": float(mae_overall),
        "regression_rmse_overall": float(rmse_overall),
        "num_delayed_samples": int(delayed_mask.sum()),
        "num_nondelayed_samples": int(nondelayed_mask.sum()),
        "target_epsilon": float(target_epsilon),
        "final_epsilon": float(final_epsilon),
        "epsilon_exceeded": bool(epsilon_exceeded),
        "epsilon_overshoot": float(max(0.0, float(final_epsilon) - float(target_epsilon))) if dp_enabled else 0.0,
        "final_delta": float(final_delta),
        "total_training_time_seconds": float(train_time),
        "total_training_time_minutes": float(train_time) / 60.0,
        "train_samples": int(train_samples),
        "val_samples": int(val_samples),
        "test_samples": int(len(test_x)),
    }

    if arr_metrics is not None and dep_metrics is not None:
        summary.update(
            {
                "regression_mae_delayed_arrival": float(arr_metrics["mae_delayed"]),
                "regression_rmse_delayed_arrival": float(arr_metrics["rmse_delayed"]),
                "num_delayed_samples_arrival": int(arr_metrics["n_delayed"]),
                "regression_mae_delayed_departure": float(dep_metrics["mae_delayed"]),
                "regression_rmse_delayed_departure": float(dep_metrics["rmse_delayed"]),
                "num_delayed_samples_departure": int(dep_metrics["n_delayed"]),
                "regression_mae_nondelayed_arrival": float(arr_metrics["mae_nondelayed"]),
                "regression_rmse_nondelayed_arrival": float(arr_metrics["rmse_nondelayed"]),
                "num_nondelayed_samples_arrival": int(arr_metrics["n_nondelayed"]),
                "regression_mae_nondelayed_departure": float(dep_metrics["mae_nondelayed"]),
                "regression_rmse_nondelayed_departure": float(dep_metrics["rmse_nondelayed"]),
                "num_nondelayed_samples_departure": int(dep_metrics["n_nondelayed"]),
                "regression_mae_overall_arrival": float(arr_metrics["mae_overall"]),
                "regression_rmse_overall_arrival": float(arr_metrics["rmse_overall"]),
                "regression_mae_overall_departure": float(dep_metrics["mae_overall"]),
                "regression_rmse_overall_departure": float(dep_metrics["rmse_overall"]),
            }
        )

    def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        denom = float(np.sum((y_true - np.mean(y_true)) ** 2)) + 1e-10
        return float(1.0 - (np.sum((y_true - y_pred) ** 2) / denom))

    def _channel_metrics(mask: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        if mask.sum() == 0:
            return {
                "mae": 0.0,
                "rmse": 0.0,
                "r2": 0.0,
                "mean_pred": 0.0,
                "mean_target": 0.0,
                "num_samples": 0,
            }
        yt = y_true[mask]
        yp = y_pred[mask]
        mae = float(np.mean(np.abs(yp - yt)))
        rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
        return {
            "mae": mae,
            "rmse": rmse,
            "r2": _safe_r2(yt, yp),
            "mean_pred": float(np.mean(yp)),
            "mean_target": float(np.mean(yt)),
            "num_samples": int(mask.sum()),
        }

    # Per-channel masks based on actual delays (only if we have two channels)
    overall_arr = overall_dep = delayed_arr = delayed_dep = nondelayed_arr = nondelayed_dep = None
    if preds_denorm.ndim == 2 and targets_denorm.ndim == 2 and preds_denorm.shape[1] == 2 and targets_denorm.shape[1] == 2:
        arr_targets = targets_denorm[:, 0]
        dep_targets = targets_denorm[:, 1]
        arr_preds = preds_denorm[:, 0]
        dep_preds = preds_denorm[:, 1]

        arr_delayed = arr_targets > float(delay_threshold)
        dep_delayed = dep_targets > float(delay_threshold)
        arr_nondelayed = arr_targets <= float(delay_threshold)
        dep_nondelayed = dep_targets <= float(delay_threshold)

        overall_arr = _channel_metrics(np.ones_like(arr_targets, dtype=bool), arr_targets, arr_preds)
        overall_dep = _channel_metrics(np.ones_like(dep_targets, dtype=bool), dep_targets, dep_preds)
        delayed_arr = _channel_metrics(arr_delayed, arr_targets, arr_preds)
        delayed_dep = _channel_metrics(dep_delayed, dep_targets, dep_preds)
        nondelayed_arr = _channel_metrics(arr_nondelayed, arr_targets, arr_preds)
        nondelayed_dep = _channel_metrics(dep_nondelayed, dep_targets, dep_preds)

    with open(results_table_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["=" * 80])
        writer.writerow(["COMPREHENSIVE EVALUATION RESULTS (REGRESSION-ONLY)"])
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
            writer.writerow(["data_source", getattr(args, "data_source", "N/A")])
            writer.writerow(["seq_len", getattr(args, "seq_len", seq_len)])
            horizons_val = getattr(args, "horizons", None)
            if isinstance(horizons_val, (list, tuple)):
                writer.writerow(["horizons", ",".join(str(h) for h in horizons_val)])
            else:
                writer.writerow(["horizons", str(horizons_val) if horizons_val is not None else "N/A"])
            writer.writerow(["delay_threshold", f"{delay_threshold} min"])
            writer.writerow(["use_node_level", getattr(args, "use_node_level", "N/A")])
            writer.writerow(["exclude_time_features", getattr(args, "exclude_time_features", "N/A")])
            writer.writerow(["weather_file", getattr(args, "weather_file", "N/A")])
            writer.writerow(["period_hours", getattr(args, "period_hours", "N/A")])
            writer.writerow(["epochs", getattr(args, "epochs", "N/A")])
            writer.writerow(["batch_size", getattr(args, "batch_size", "N/A")])
            writer.writerow(["lr", getattr(args, "lr", "N/A")])
            writer.writerow(["patience", getattr(args, "patience", "N/A")])
            writer.writerow(["model", getattr(args, "model", "N/A")])
            writer.writerow(["stpn_layers", getattr(args, "stpn_layers", "N/A")])
            writer.writerow(["stpn_emb_size", getattr(args, "stpn_emb_size", "N/A")])
            writer.writerow(["stpn_heads", getattr(args, "stpn_heads", "N/A")])
            writer.writerow(["stpn_time_d", getattr(args, "stpn_time_d", "N/A")])
            writer.writerow(["dp", getattr(args, "dp", dp_enabled)])
            writer.writerow(["dp_accountant", getattr(args, "dp_accountant", "N/A")])
            writer.writerow(["epsilon (target)", f"{float(target_epsilon):.3f}"])
            writer.writerow(["target_delta", getattr(args, "target_delta", "N/A")])
            writer.writerow(["noise_multiplier", float(noise_multiplier)])
            writer.writerow(["max_grad_norm", getattr(args, "max_grad_norm", "N/A")])
            writer.writerow(["sample_rate", getattr(args, "sample_rate", "N/A")])
            writer.writerow(["epsilon_tolerance", getattr(args, "epsilon_tolerance", "N/A")])
            writer.writerow(["model_path", getattr(args, "model_path", "N/A")])
            writer.writerow(["checkpoint_dir", getattr(args, "checkpoint_dir", "N/A")])
            writer.writerow(["seed", getattr(args, "seed", "N/A")])
            writer.writerow(["balance_50_50", getattr(args, "balance_50_50", "N/A")])
            writer.writerow(["epsilonfixed", getattr(args, "epsilonfixed", "N/A")])
            writer.writerow(["skip_visualization", getattr(args, "skip_visualization", "N/A")])
        else:
            writer.writerow(["Arguments object not provided"])
        writer.writerow([])

        writer.writerow(["=== TRAINING RESULTS ==="])
        writer.writerow(["Final Epsilon", f"{final_epsilon:.3f}"])
        writer.writerow(["Final Delta", f"{final_delta:.2e}"])
        writer.writerow(["Epsilon Exceeded", bool(epsilon_exceeded)])
        writer.writerow(["Train Samples", train_samples])
        writer.writerow(["Val Samples", val_samples])
        writer.writerow(["Test Samples", len(test_x)])
        writer.writerow(["Total Training Time (s)", f"{float(train_time):.2f}"])
        writer.writerow(["Total Training Time (min)", f"{float(train_time)/60.0:.2f}"])
        writer.writerow([])

        writer.writerow(["=" * 80])
        writer.writerow(["OVERALL SUMMARY TABLE"])
        writer.writerow(["=" * 80])
        if overall_arr is not None and overall_dep is not None:
            writer.writerow(["Epsilon", "Overall MAE (min)", "Arrival MAE (min)", "Departure MAE (min)"])
            writer.writerow([
                f"{final_epsilon:.2f}",
                f"{mae_overall:.4f}",
                f"{overall_arr['mae']:.4f}",
                f"{overall_dep['mae']:.4f}",
            ])
        else:
            writer.writerow(["Epsilon", "Overall MAE (min)"])
            writer.writerow([f"{final_epsilon:.2f}", f"{mae_overall:.4f}"])
        writer.writerow([])

        if delayed_arr is not None and delayed_dep is not None and nondelayed_arr is not None and nondelayed_dep is not None:
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
                int(summary["num_delayed_samples"]),
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
                int(summary["num_nondelayed_samples"]),
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
        writer.writerow(["SUMMARY (raw metric/value)"])
        writer.writerow(["=" * 80])
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])

    print("\nPRIVACY BUDGET:")
    print(f"  Target ε: {float(target_epsilon):.3f}")
    print(f"  Final ε: {final_epsilon:.3f}")
    if dp_enabled:
        if final_epsilon <= float(target_epsilon):
            print("  ✓ Budget maintained (within target)")
        else:
            overshoot = final_epsilon - float(target_epsilon)
            pct = (overshoot / float(target_epsilon) * 100.0) if float(target_epsilon) > 0 else float("inf")
            print(f"  ⚠️ Budget exceeded by {overshoot:.3f} ε ({pct:.1f}%)")
    print(f"  Final δ: {final_delta:.2e}")

    print("\nTRAINING TIME:")
    print(f"  Total: {train_time:.2f}s ({train_time/60:.2f} min)")
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    print("\n✓ Results saved to:")
    print(f"  - {model_path}")
    if histories:
        print(f"  - {history_csv}")
    print(f"  - {results_table_csv}")

    # Rename checkpoint directory to match model filename (optional, for consistency)
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        from pathlib import Path

        old_dir = Path(checkpoint_dir)
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        new_dir = old_dir.parent / model_basename
        if old_dir != new_dir and not new_dir.exists():
            try:
                old_dir.rename(new_dir)
                print("\n✓ Renamed checkpoint directory:")
                print(f"  From: {old_dir}")
                print(f"  To: {new_dir}")
                latest_file = old_dir.parent / "latest_run.txt"
                if latest_file.exists():
                    latest_file.write_text(str(new_dir), encoding="utf-8")
            except Exception as e:
                print(f"\n⚠ Could not rename checkpoint directory: {e}")



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
    parser = argparse.ArgumentParser(description="Three-stage model (TCN or STPN) with optional DP-SGD and epsilon tracking")
    parser.add_argument('--data_source', type=str, default='cdata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=18)
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
    # Regression-only training
    parser.add_argument('--epochs', type=int, default=10, help='Training epochs (regression only)')
    # Back-compat (ignored)
    parser.add_argument('--stage1_epochs', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--stage2_epochs', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--stage3_epochs', type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels in encoder')
    parser.add_argument(
        '--model',
        type=str,
        default='stpn',
        choices=['tcn', 'stpn'],
        help="Backbone to use: 'tcn' (existing Conv/TCN encoder) or 'stpn' (spatio-temporal GCN + attention from model.py)",
    )
    parser.add_argument('--stpn_layers', type=int, default=2, help='Number of STPN hidden layers (h_layers)')
    parser.add_argument('--stpn_emb_size', type=int, default=16, help='STPN time embedding size (emb_size)')
    parser.add_argument('--stpn_heads', type=int, default=4, help='STPN attention heads')
    parser.add_argument('--stpn_time_d', type=int, default=4, help='STPN attention projection size (time_d)')
    parser.add_argument('--lr', type=float, default=0.005, help='Global learning rate (used if stage-specific LRs not provided)')
    # Back-compat (ignored)
    parser.add_argument('--stage1_lr', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--stage2_lr', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--stage3_lr', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dp', default=True, action='store_true', help='Enable DP-SGD')
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
    parser.add_argument('--model_path', type=str, default='stpn_dp_regression.pth')
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
    parser.add_argument('--skip_visualization', action='store_true', default=False, help='Skip visualization plots during final evaluation')
    parser.add_argument(
        '--epsilonfixed',
        dest='epsilonfixed',
        default=True,
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
    
    train_x, train_y_reg, _train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    val_x, val_y_reg, _val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    test_x, test_y_reg, _test_y_cls = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    
    if args.balance_50_50:
        print("\n[INFO] --balance_50_50 is a classification option; ignored in regression-only mode.")
    
    # Comprehensive data validation
    print(f"\n[DATA VALIDATION] Checking data quality...")
    print(f"  Train features shape: {train_x.shape}")
    print(f"  Train targets (reg) shape: {train_y_reg.shape}")
    
    # Check for NaN/Inf
    print(f"  Train features: NaN={torch.isnan(train_x).any().item()}, Inf={torch.isinf(train_x).any().item()}")
    print(f"  Train reg targets: NaN={torch.isnan(train_y_reg).any().item()}, Inf={torch.isinf(train_y_reg).any().item()}")
    
    # Target statistics (scaled)
    print(f"  Train reg targets (scaled): min={train_y_reg.min().item():.3f}, max={train_y_reg.max().item():.3f}, mean={train_y_reg.mean().item():.3f}, std={train_y_reg.std().item():.3f}")
    
    # Per-channel distribution for regression targets
    if train_y_reg.dim() >= 2 and train_y_reg.shape[-1] == 2:
        print(f"  Arrival channel (scaled): min={train_y_reg[..., 0].min().item():.3f}, max={train_y_reg[..., 0].max().item():.3f}, mean={train_y_reg[..., 0].mean().item():.3f}")
        print(f"  Departure channel (scaled): min={train_y_reg[..., 1].min().item():.3f}, max={train_y_reg[..., 1].max().item():.3f}, mean={train_y_reg[..., 1].mean().item():.3f}")
    
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

    supports: Optional[List[torch.Tensor]] = None

    if args.model == 'stpn':
        if not STPN_AVAILABLE:
            raise ImportError("--model stpn selected but STPN backbone could not be imported from model.py")

        supports = build_stpn_supports(edge_indices[0], edge_indices[1], edge_indices[2], num_nodes, device)

        model = STPNThreeStagePredictor(
            feature_dim=feature_dim,
            seq_len=int(args.seq_len),
            out_channels=out_channels,
            hidden_channels=args.hidden_channels,
            h_layers=args.stpn_layers,
            emb_size=args.stpn_emb_size,
            heads=args.stpn_heads,
            time_d=args.stpn_time_d,
            dropout=0.1,
            support_len=3,
            order=2,
            use_se=True,
        ).to(device)

        # Graph-level datasets/loaders (regression-only)
        train_ds = TensorDataset(train_x, train_y_reg)
        val_ds = TensorDataset(val_x, val_y_reg)
        sample_unit = "graph"
        total_samples = len(train_ds)
        sample_rate = args.batch_size / max(1, total_samples)
        steps_per_epoch = total_samples // args.batch_size
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)
    else:
        model = SequentialTwoStagePredictor(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=args.hidden_channels,
            regressor_extra_layer=True,
            seq_len=args.seq_len,
        ).to(device)

        # Build node-level datasets/loaders for Opacus FIRST (before calculating steps).
        train_x_flat = _flatten_node_level(train_x)
        train_y_reg_flat = _flatten_node_level(train_y_reg)

        val_x_flat = _flatten_node_level(val_x)
        val_y_reg_flat = _flatten_node_level(val_y_reg)

        train_ds = TensorDataset(train_x_flat, train_y_reg_flat)
        val_ds = TensorDataset(val_x_flat, val_y_reg_flat)
        sample_unit = "node"

        # Calculate steps based on ACTUAL flattened dataset size (node-level samples)
        total_samples = len(train_ds)  # Node-level count, NOT graph-level
        sample_rate = args.batch_size / max(1, total_samples)
        # Note: DPDataLoader ignores drop_last, so use floor division for accurate step count
        steps_per_epoch = total_samples // args.batch_size
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    total_epochs = int(args.epochs)
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
        print(f"  Training samples ({sample_unit}-level): {total_samples}")
        print(f"  Sample rate per step (q): {sample_rate:.6f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Total epochs: {total_epochs}")
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

    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Train samples (graphs): {len(train_x)}")
    print(f"Val samples (graphs): {len(val_x)}")
    print(f"Test samples (graphs): {len(test_x)}")
    print(f"Train examples ({sample_unit}-level): {len(train_ds)}")

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
            print(f"  Training samples ({sample_unit}-level): {len(train_ds)}, batch_size: {args.batch_size}")
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
                print(f"    This noise is calibrated for {total_epochs} epochs total")
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

    print(f"\n[TRAINING] Regression-only: epochs={int(args.epochs)}, lr={float(args.lr)}")
    combined_history, train_time = train_regression_opacus(
        model=model,
        optimizer=optimizer,
        privacy_engine=privacy_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=int(args.epochs),
        lr=float(args.lr),
        scaler=scaler,
        delay_threshold=float(args.delay_threshold),
        patience=args.patience,
        target_delta=float(args.target_delta),
        target_epsilon=tracking_target_epsilon,
        supports=supports,
        seq_len=int(args.seq_len),
    )

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
    default_model_path = os.path.join(CHECKPOINT_DIR, "stpn_dp_regression.pth")
    if args.model_path == "stpn_dp_regression.pth":
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

    final_evaluation_regression_only(
        model,
        edge_indices=edge_indices,
        supports=supports,
        device=device,
        scaler=scaler,
        num_nodes=num_nodes,
        test_x=test_x,
        test_y_reg=test_y_reg,
        delay_threshold=float(args.delay_threshold),
        model_path=args.model_path,
        run_tag=run_tag,
        timestamp=timestamp,
        histories=combined_history,
        final_epsilon=final_epsilon,
        final_delta=final_delta,
        train_time=float(train_time),
        train_samples=len(train_x),
        val_samples=len(val_x),
        dp_enabled=bool(getattr(args, "dp", False) or getattr(args, "epsilonfixed", False)),
        target_epsilon=float(args.epsilon),
        noise_multiplier=float(sigma_val),
        artifact_prefix="train",
        seq_len=args.seq_len,
        args=args,
        checkpoint_dir=CHECKPOINT_DIR,
        enable_visualization=not args.skip_visualization,
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