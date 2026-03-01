"""Three-stage comparison across CNN, LSTM, GRU, and STPN.

Trains and evaluates:
- Stage 1: classification (arrival/departure)
- Stage 2: delayed-flight regression
- Stage 3: non-delayed regression

Outputs (per model):
- history CSV (all stages)
- results table CSV (similar to deepDualReg)
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from classifykat import EarlyStopping, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level, build_sequences

# Optional Opacus DP
try:
    from opacus import PrivacyEngine
    OPACUS_AVAILABLE = True
except Exception:
    PrivacyEngine = None
    OPACUS_AVAILABLE = False

# Try to import STPN
try:
    from model import STPN
    STPN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] STPN not available: {e}")
    STPN = None
    STPN_AVAILABLE = False


def _load_cnn_module() -> object:
    module_path = os.path.join(os.path.dirname(__file__), "cnnopacus - deepDualReg.py")
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"CNN module not found at: {module_path}")
    spec = importlib.util.spec_from_file_location("cnn_dualreg", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load CNN module spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classification_metrics_per_channel(
    preds: np.ndarray,
    targets: np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
    prob_threshold: float = 0.5,
) -> Dict[str, float]:
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


def _scaled_delay_threshold(delay_threshold: float, scaler) -> torch.Tensor:
    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32))
        std_t = torch.tensor(np.array(scaler.std, dtype=np.float32))
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
        return (torch.full_like(mean_t, float(delay_threshold)) - mean_t) / std_t
    return torch.tensor([float(delay_threshold)], dtype=torch.float32)


def _base_model(model: nn.Module) -> nn.Module:
    return getattr(model, "_module", model)


def _make_private_if_enabled(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    *,
    dp_enabled: bool,
    dp_accountant: str,
    noise_multiplier: float,
    max_grad_norm: float,
    model_name: str,
    stage_name: str,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, Optional[object]]:
    if not dp_enabled:
        return model, optimizer, train_loader, None


def _safe_get_epsilon(privacy_engine: Optional[object], target_delta: float) -> float:
    if privacy_engine is None:
        return 0.0
    try:
        return float(privacy_engine.get_epsilon(float(target_delta)))
    except Exception:
        return float("nan")
    if not OPACUS_AVAILABLE or PrivacyEngine is None:
        print(f"[{model_name}] [DP] Opacus not available. Running {stage_name} without DP.")
        return model, optimizer, train_loader, None

    try:
        privacy_engine = PrivacyEngine(accountant=dp_accountant)
        private_model, private_optimizer, private_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=float(noise_multiplier),
            max_grad_norm=float(max_grad_norm),
        )
        print(
            f"[{model_name}] [DP] Enabled for {stage_name} "
            f"(accountant={dp_accountant}, sigma={noise_multiplier}, clip={max_grad_norm})"
        )
        return private_model, private_optimizer, private_loader, privacy_engine
    except Exception as e:
        print(f"[{model_name}] [DP] Failed to enable for {stage_name}: {e}. Running without DP.")
        return model, optimizer, train_loader, None


class CNNAdapter(nn.Module):
    def __init__(self, cnn_model: nn.Module) -> None:
        super().__init__()
        self.model = cnn_model

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.model._encode_x(x)
        logits = self.model.classifier(self.model.dropout_cls(hidden))
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor, which: str) -> torch.Tensor:
        return self.model.forward_regressor(hidden, which=which)


class RecurrentDualModel(nn.Module):
    def __init__(
        self,
        cell_type: str,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        if cell_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        else:
            self.rnn = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_delayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )
        self.regressor_nondelayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.rnn, nn.LSTM):
            _, (h_n, _) = self.rnn(x)
        else:
            _, h_n = self.rnn(x)
        hidden = h_n[-1]
        logits = self.classifier(hidden)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor, which: str) -> torch.Tensor:
        if which == "delayed":
            return self.regressor_delayed(hidden)
        return self.regressor_nondelayed(hidden)


class BiLSTMDualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        feat_dim = hidden_dim * 2
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_delayed = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_nondelayed = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.encoder(x)
        hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.dropout(hidden)

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(x)
        logits = self.classifier(hidden)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor, which: str) -> torch.Tensor:
        if which == "delayed":
            return self.regressor_delayed(hidden)
        return self.regressor_nondelayed(hidden)


class CNNLSTMDualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.dropout_conv = nn.Dropout(dropout)
        self.lstm = nn.LSTM(128, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_delayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_nondelayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout_conv(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        return self.dropout(h_n[-1])

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(x)
        logits = self.classifier(hidden)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor, which: str) -> torch.Tensor:
        if which == "delayed":
            return self.regressor_delayed(hidden)
        return self.regressor_nondelayed(hidden)


class AttentionLSTMDualModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_delayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_nondelayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return self.dropout(context)

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(x)
        logits = self.classifier(hidden)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor, which: str) -> torch.Tensor:
        if which == "delayed":
            return self.regressor_delayed(hidden)
        return self.regressor_nondelayed(hidden)


class STPNDualModel(nn.Module):
    def __init__(
        self,
        supports: List[torch.Tensor],
        in_len: int,
        out_len: int,
        feature_dim: int,
        out_dim: int,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if not STPN_AVAILABLE:
            raise RuntimeError("STPN is not available")
        self.supports = supports
        self.in_len = int(in_len)
        self.out_len = int(out_len)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)

        self.stpn = STPN(
            h_layers=2,
            in_channels=feature_dim,
            hidden_channels=[hidden_dim, hidden_dim, hidden_dim],
            out_channels=hidden_dim,
            emb_size=16,
            dropout=0.2,
            wemb_size=4,
            time_d=4,
            heads=4,
            support_len=3,
            order=2,
            num_weather=8,
            use_se=False,
            use_cov=False,
        )
        self.classifier = nn.Linear(hidden_dim, out_dim)
        self.regressor_delayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.regressor_nondelayed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, T, F] -> [B, F, V, T]
        x_t = x.permute(0, 3, 1, 2).contiguous()
        bsz = x_t.shape[0]
        t_in = torch.arange(self.in_len, device=x.device).unsqueeze(0).repeat(bsz, 1)
        t_out = torch.arange(self.out_len, device=x.device).unsqueeze(0).repeat(bsz, 1)
        out = self.stpn(x_t, t_in, self.supports, t_out, None)
        # out: [B, C, V, T_out] -> pool over time -> [B, C, V]
        pooled = out.mean(dim=3)
        return pooled.permute(0, 2, 1).contiguous()  # [B, V, C]

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(x)
        logits = self.classifier(hidden)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor, which: str) -> torch.Tensor:
        if which == "delayed":
            return self.regressor_delayed(hidden)
        return self.regressor_nondelayed(hidden)


def _train_stage1_node(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    model_name: str,
    dp_enabled: bool = False,
    dp_accountant: str = "rdp",
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_delta: float = 1e-5,
    target_epsilon: float = 0.0,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model_train, optimizer, train_loader_use, privacy_engine = _make_private_if_enabled(
        model,
        optimizer,
        train_loader,
        dp_enabled=dp_enabled,
        dp_accountant=dp_accountant,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        model_name=model_name,
        stage_name="stage1",
    )

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    early_stopping = EarlyStopping(patience=patience, mode="max")

    history: List[Dict[str, float]] = []
    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model_train.train()
        epoch_losses = []
        for batch_x, batch_y_cls, _ in train_loader_use:
            batch_x = batch_x.to(device)
            batch_y_cls = batch_y_cls.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model_train.forward_classifier(batch_x)
            loss = loss_fn(logits, batch_y_cls)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model_train.eval()
        val_probs_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch_x, batch_y_cls, _ in val_loader:
                batch_x = batch_x.to(device)
                batch_y_cls = batch_y_cls.to(device)
                _, logits = model_train.forward_classifier(batch_x)
                val_probs_list.append(torch.sigmoid(logits).cpu())
                val_targets_list.append(batch_y_cls.cpu())

        val_probs = torch.cat(val_probs_list, dim=0).numpy()
        val_targets = torch.cat(val_targets_list, dim=0).numpy()
        val_metrics = classification_metrics_per_channel(val_probs, val_targets)
        eps = _safe_get_epsilon(privacy_engine, target_delta)

        history.append(
            {
                "epoch": epoch,
                "stage": 1,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "val_f1": val_metrics["f1"],
                "val_accuracy": val_metrics["accuracy"],
                "epsilon": eps,
                "delta": float(target_delta) if dp_enabled else 0.0,
                "target_epsilon": float(target_epsilon) if dp_enabled else 0.0,
            }
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_state = {k: v.detach().cpu() for k, v in _base_model(model_train).state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            eps_msg = f" | ε: {eps:.3f}/{target_epsilon:.3f}" if dp_enabled else ""
            print(f"[{model_name}] Stage 1 Epoch {epoch}/{epochs} | Val F1: {val_metrics['f1']:.4f}{eps_msg}")

        if early_stopping(val_metrics["f1"], epoch):
            print(f"[{model_name}] Stage 1 early stopping at epoch {epoch}")
            break

    if best_state is not None:
        _base_model(model_train).load_state_dict(best_state)

    return history


def _train_stage2_node(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    delay_threshold: float,
    scaler,
    patience: int,
    model_name: str,
    dp_enabled: bool = False,
    dp_accountant: str = "rdp",
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_delta: float = 1e-5,
    target_epsilon: float = 0.0,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    model_train, optimizer, train_loader_use, privacy_engine = _make_private_if_enabled(
        model,
        optimizer,
        train_loader,
        dp_enabled=dp_enabled,
        dp_accountant=dp_accountant,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        model_name=model_name,
        stage_name="stage2",
    )

    huber_loss = nn.HuberLoss(reduction="none", delta=2.0)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    delay_threshold_scaled = _scaled_delay_threshold(delay_threshold, scaler)

    def masked_huber(preds: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        thr = delay_threshold_scaled.to(targets.device)
        if thr.numel() == 1 and targets.shape[-1] > 1:
            thr = thr.expand(targets.shape[-1])
        mask = (targets > thr).float()
        per_elem = huber_loss(preds, targets) * mask
        denom = mask.sum(dim=0).clamp_min(1.0)
        loss_ch = per_elem.sum(dim=0) / denom
        return loss_ch.mean(), mask

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model_train.train()
        epoch_losses = []
        for batch_x, _, batch_y_reg in train_loader_use:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden, _ = model_train.forward_classifier(batch_x)
            preds = model_train.forward_regressor(hidden, which="delayed")
            loss, _ = masked_huber(preds, batch_y_reg)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model_train.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _, batch_y_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_reg = batch_y_reg.to(device)
                hidden, _ = model_train.forward_classifier(batch_x)
                preds = model_train.forward_regressor(hidden, which="delayed")
                loss, _ = masked_huber(preds, batch_y_reg)
                val_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        eps = _safe_get_epsilon(privacy_engine, target_delta)
        history.append(
            {
                "epoch": epoch,
                "stage": 2,
                "train_loss": float(np.mean(epoch_losses)),
                "val_loss": val_loss,
                "epsilon": eps,
                "delta": float(target_delta) if dp_enabled else 0.0,
                "target_epsilon": float(target_epsilon) if dp_enabled else 0.0,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in _base_model(model_train).state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            eps_msg = f" | ε: {eps:.3f}/{target_epsilon:.3f}" if dp_enabled else ""
            print(f"[{model_name}] Stage 2 Epoch {epoch}/{epochs} | Val Loss: {val_loss:.4f}{eps_msg}")

        if early_stopping(val_loss, epoch):
            print(f"[{model_name}] Stage 2 early stopping at epoch {epoch}")
            break

    if best_state is not None:
        _base_model(model_train).load_state_dict(best_state)

    return history


def _train_stage3_node(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    delay_threshold: float,
    scaler,
    patience: int,
    model_name: str,
    dp_enabled: bool = False,
    dp_accountant: str = "rdp",
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    target_delta: float = 1e-5,
    target_epsilon: float = 0.0,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    model_train, optimizer, train_loader_use, privacy_engine = _make_private_if_enabled(
        model,
        optimizer,
        train_loader,
        dp_enabled=dp_enabled,
        dp_accountant=dp_accountant,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        model_name=model_name,
        stage_name="stage3",
    )

    reg_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
        mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32), device=device)
        std_t = torch.tensor(np.array(scaler.std, dtype=np.float32), device=device)
        std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
    else:
        mean_t = None
        std_t = None

    def nondelayed_mask(targets_scaled: torch.Tensor) -> torch.Tensor:
        if mean_t is None or std_t is None:
            targets_denorm = targets_scaled.detach()
        else:
            targets_denorm = targets_scaled.detach() * std_t + mean_t
        return (targets_denorm.abs() < float(delay_threshold)).float()

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model_train.train()
        epoch_losses = []
        for batch_x, _, batch_y_reg in train_loader_use:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            mask = nondelayed_mask(batch_y_reg)
            if float(mask.sum().item()) == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            hidden, _ = model_train.forward_classifier(batch_x)
            preds = model_train.forward_regressor(hidden, which="nondelayed")
            loss_per = reg_loss_fn(preds, batch_y_reg) * mask
            denom = mask.sum(dim=0).clamp_min(1.0)
            loss_ch = loss_per.sum(dim=0) / denom
            loss = loss_ch.mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model_train.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _, batch_y_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_reg = batch_y_reg.to(device)
                mask = nondelayed_mask(batch_y_reg)
                if float(mask.sum().item()) == 0:
                    continue
                hidden, _ = model_train.forward_classifier(batch_x)
                preds = model_train.forward_regressor(hidden, which="nondelayed")
                loss_per = reg_loss_fn(preds, batch_y_reg) * mask
                denom = mask.sum(dim=0).clamp_min(1.0)
                loss_ch = loss_per.sum(dim=0) / denom
                val_losses.append(float(loss_ch.mean().item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        eps = _safe_get_epsilon(privacy_engine, target_delta)
        history.append(
            {
                "epoch": epoch,
                "stage": 3,
                "train_loss": float(np.mean(epoch_losses)),
                "val_loss": val_loss,
                "epsilon": eps,
                "delta": float(target_delta) if dp_enabled else 0.0,
                "target_epsilon": float(target_epsilon) if dp_enabled else 0.0,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in _base_model(model_train).state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            eps_msg = f" | ε: {eps:.3f}/{target_epsilon:.3f}" if dp_enabled else ""
            print(f"[{model_name}] Stage 3 Epoch {epoch}/{epochs} | Val Loss: {val_loss:.4f}{eps_msg}")

        if early_stopping(val_loss, epoch):
            print(f"[{model_name}] Stage 3 early stopping at epoch {epoch}")
            break

    if best_state is not None:
        _base_model(model_train).load_state_dict(best_state)

    return history


def _train_stage_stpn(
    stage: int,
    model: STPNDualModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    delay_threshold: float,
    scaler,
    pos_weight: torch.Tensor,
    patience: int,
) -> List[Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience, mode="max" if stage == 1 else "min")

    if stage == 1:
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    elif stage == 2:
        loss_fn = nn.HuberLoss(reduction="none", delta=2.0)
        delay_threshold_scaled = _scaled_delay_threshold(delay_threshold, scaler)
    else:
        loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
        if scaler is not None and hasattr(scaler, "mean") and hasattr(scaler, "std"):
            mean_t = torch.tensor(np.array(scaler.mean, dtype=np.float32), device=device)
            std_t = torch.tensor(np.array(scaler.std, dtype=np.float32), device=device)
            std_t = torch.where(std_t == 0, torch.ones_like(std_t), std_t)
        else:
            mean_t = None
            std_t = None

    history: List[Dict[str, float]] = []
    best_score = float("inf") if stage != 1 else -1.0
    best_state = None

    def nondelayed_mask(targets_scaled: torch.Tensor) -> torch.Tensor:
        if mean_t is None or std_t is None:
            targets_denorm = targets_scaled.detach()
        else:
            targets_denorm = targets_scaled.detach() * std_t + mean_t
        return (targets_denorm.abs() < float(delay_threshold)).float()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y_cls, batch_y_reg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_cls = batch_y_cls.to(device)
            batch_y_reg = batch_y_reg.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden, logits = model.forward_classifier(batch_x)

            if stage == 1:
                loss = loss_fn(logits, batch_y_cls)
            elif stage == 2:
                preds = model.forward_regressor(hidden, which="delayed")
                thr = delay_threshold_scaled.to(batch_y_reg.device)
                if thr.numel() == 1 and batch_y_reg.shape[-1] > 1:
                    thr = thr.expand(batch_y_reg.shape[-1])
                mask = (batch_y_reg > thr).float()
                per_elem = loss_fn(preds, batch_y_reg) * mask
                denom = mask.sum(dim=0).clamp_min(1.0)
                loss_ch = per_elem.sum(dim=0) / denom
                loss = loss_ch.mean()
            else:
                preds = model.forward_regressor(hidden, which="nondelayed")
                mask = nondelayed_mask(batch_y_reg)
                if float(mask.sum().item()) == 0:
                    continue
                per_elem = loss_fn(preds, batch_y_reg) * mask
                denom = mask.sum(dim=0).clamp_min(1.0)
                loss_ch = per_elem.sum(dim=0) / denom
                loss = loss_ch.mean()

            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        if stage == 1:
            val_probs_list = []
            val_targets_list = []
            with torch.no_grad():
                for batch_x, batch_y_cls, _ in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y_cls = batch_y_cls.to(device)
                    _, logits = model.forward_classifier(batch_x)
                    val_probs_list.append(torch.sigmoid(logits).cpu())
                    val_targets_list.append(batch_y_cls.cpu())
            val_probs = torch.cat(val_probs_list, dim=0).numpy()
            val_targets = torch.cat(val_targets_list, dim=0).numpy()
            val_metrics = classification_metrics_per_channel(val_probs, val_targets)
            score = val_metrics["f1"]
        else:
            val_losses = []
            with torch.no_grad():
                for batch_x, _, batch_y_reg in val_loader:
                    batch_x = batch_x.to(device)
                    batch_y_reg = batch_y_reg.to(device)
                    hidden, _ = model.forward_classifier(batch_x)
                    which = "delayed" if stage == 2 else "nondelayed"
                    preds = model.forward_regressor(hidden, which=which)
                    if stage == 2:
                        thr = delay_threshold_scaled.to(batch_y_reg.device)
                        if thr.numel() == 1 and batch_y_reg.shape[-1] > 1:
                            thr = thr.expand(batch_y_reg.shape[-1])
                        mask = (batch_y_reg > thr).float()
                        per_elem = loss_fn(preds, batch_y_reg) * mask
                        denom = mask.sum(dim=0).clamp_min(1.0)
                        loss_ch = per_elem.sum(dim=0) / denom
                        val_losses.append(float(loss_ch.mean().item()))
                    else:
                        mask = nondelayed_mask(batch_y_reg)
                        if float(mask.sum().item()) == 0:
                            continue
                        per_elem = loss_fn(preds, batch_y_reg) * mask
                        denom = mask.sum(dim=0).clamp_min(1.0)
                        loss_ch = per_elem.sum(dim=0) / denom
                        val_losses.append(float(loss_ch.mean().item()))
            score = float(np.mean(val_losses)) if val_losses else 0.0

        history.append({"epoch": epoch, "stage": stage, "train_loss": float(np.mean(epoch_losses)), "val_score": score})

        if stage == 1:
            improved = score > best_score
        else:
            improved = score < best_score

        if improved:
            best_score = score
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            label = "F1" if stage == 1 else "Val Loss"
            print(f"[STPN] Stage {stage} Epoch {epoch}/{epochs} | {label}: {score:.4f}")

        if early_stopping(score, epoch):
            print(f"[STPN] Stage {stage} early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def _evaluate_model(
    model: nn.Module,
    test_x: torch.Tensor,
    test_y_cls: torch.Tensor,
    test_y_reg: torch.Tensor,
    device: torch.device,
    class_threshold: float,
    scaler,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    reg_list = []
    targets_cls_list = []
    targets_reg_list = []

    with torch.no_grad():
        for batch_x, batch_y_cls, batch_y_reg in DataLoader(
            TensorDataset(test_x, test_y_cls, test_y_reg),
            batch_size=256,
            shuffle=False,
        ):
            batch_x = batch_x.to(device)
            batch_y_cls = batch_y_cls.to(device)
            batch_y_reg = batch_y_reg.to(device)
            hidden, logits = model.forward_classifier(batch_x)
            probs = torch.sigmoid(logits)
            reg_delayed = model.forward_regressor(hidden, which="delayed")
            reg_nondelayed = model.forward_regressor(hidden, which="nondelayed")
            gate = torch.sigmoid((probs - class_threshold) * 10.0)
            preds = reg_delayed * gate + reg_nondelayed * (1.0 - gate)
            probs_list.append(probs.cpu().numpy())
            reg_list.append(preds.cpu().numpy())
            targets_cls_list.append(batch_y_cls.cpu().numpy())
            targets_reg_list.append(batch_y_reg.cpu().numpy())

    test_probs = np.concatenate(probs_list, axis=0)
    test_reg_preds = np.concatenate(reg_list, axis=0)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)

    cls_metrics = classification_metrics_per_channel(
        test_probs,
        test_cls_targets,
        channel_names=("arrival", "departure"),
        prob_threshold=class_threshold,
    )
    return cls_metrics, test_probs, test_reg_preds, test_reg_targets


def _evaluate_stpn(
    model: STPNDualModel,
    test_x: torch.Tensor,
    test_y_cls: torch.Tensor,
    test_y_reg: torch.Tensor,
    device: torch.device,
    class_threshold: float,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    probs_list = []
    reg_list = []
    targets_cls_list = []
    targets_reg_list = []

    with torch.no_grad():
        for batch_x, batch_y_cls, batch_y_reg in DataLoader(
            TensorDataset(test_x, test_y_cls, test_y_reg),
            batch_size=16,
            shuffle=False,
        ):
            batch_x = batch_x.to(device)
            batch_y_cls = batch_y_cls.to(device)
            batch_y_reg = batch_y_reg.to(device)
            hidden, logits = model.forward_classifier(batch_x)
            probs = torch.sigmoid(logits)
            reg_delayed = model.forward_regressor(hidden, which="delayed")
            reg_nondelayed = model.forward_regressor(hidden, which="nondelayed")
            gate = torch.sigmoid((probs - class_threshold) * 10.0)
            preds = reg_delayed * gate + reg_nondelayed * (1.0 - gate)
            probs_list.append(probs.cpu().numpy())
            reg_list.append(preds.cpu().numpy())
            targets_cls_list.append(batch_y_cls.cpu().numpy())
            targets_reg_list.append(batch_y_reg.cpu().numpy())

    test_probs = np.concatenate(probs_list, axis=0)
    test_reg_preds = np.concatenate(reg_list, axis=0)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)

    cls_metrics = classification_metrics_per_channel(
        test_probs,
        test_cls_targets,
        channel_names=("arrival", "departure"),
        prob_threshold=class_threshold,
    )
    return cls_metrics, test_probs, test_reg_preds, test_reg_targets


def _write_history_csv(path: str, history: List[Dict[str, float]]) -> None:
    if not history:
        return
    with open(path, "w", newline="") as f:
        fields = sorted({k for row in history for k in row})
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(history)


def _write_results_table(
    path: str,
    model_name: str,
    class_metrics: Dict[str, float],
    reg_preds: np.ndarray,
    reg_targets: np.ndarray,
    class_threshold: float,
    delay_threshold: float,
    scaler,
    final_epsilon: float = 0.0,
    final_delta: float = 0.0,
    target_epsilon: float = 0.0,
    dp_enabled: bool = False,
) -> None:
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(reg_preds)
        targets_denorm = scaler.inverse_transform(reg_targets)
    else:
        preds_denorm = reg_preds
        targets_denorm = reg_targets

    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)

    preds_flat = preds_denorm.flatten()
    targets_flat = targets_denorm.flatten()

    delayed_mask = targets_flat > delay_threshold
    nondelayed_mask = targets_flat <= delay_threshold

    def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        mae = float(np.mean(np.abs(y_pred - y_true))) if y_true.size else 0.0
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2))) if y_true.size else 0.0
        return mae, rmse

    mae_delayed, rmse_delayed = _mae_rmse(targets_flat[delayed_mask], preds_flat[delayed_mask])
    mae_nondelayed, rmse_nondelayed = _mae_rmse(targets_flat[nondelayed_mask], preds_flat[nondelayed_mask])
    mae_overall, rmse_overall = _mae_rmse(targets_flat, preds_flat)

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

        def _mae_rmse_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int]:
            if int(mask.sum()) == 0:
                return 0.0, 0.0, 0
            yt = y_true[mask]
            yp = y_pred[mask]
            return _mae_rmse(yt, yp) + (int(mask.sum()),)

        arr_delayed_mask = arr_targets > delay_threshold
        dep_delayed_mask = dep_targets > delay_threshold
        arr_nondelayed_mask = ~arr_delayed_mask
        dep_nondelayed_mask = ~dep_delayed_mask

        arr_mae_d, arr_rmse_d, arr_n_d = _mae_rmse_mask(arr_targets, arr_preds, arr_delayed_mask)
        dep_mae_d, dep_rmse_d, dep_n_d = _mae_rmse_mask(dep_targets, dep_preds, dep_delayed_mask)
        arr_mae_nd, arr_rmse_nd, arr_n_nd = _mae_rmse_mask(arr_targets, arr_preds, arr_nondelayed_mask)
        dep_mae_nd, dep_rmse_nd, dep_n_nd = _mae_rmse_mask(dep_targets, dep_preds, dep_nondelayed_mask)
        arr_mae_all, arr_rmse_all = _mae_rmse(arr_targets, arr_preds)
        dep_mae_all, dep_rmse_all = _mae_rmse(dep_targets, dep_preds)

    summary = {
        "classification_precision": class_metrics["precision"],
        "classification_recall": class_metrics["recall"],
        "classification_f1": class_metrics["f1"],
        "classification_accuracy": class_metrics["accuracy"],
        "classification_precision_arrival": class_metrics.get("precision_arrival", 0.0),
        "classification_recall_arrival": class_metrics.get("recall_arrival", 0.0),
        "classification_f1_arrival": class_metrics.get("f1_arrival", 0.0),
        "classification_accuracy_arrival": class_metrics.get("accuracy_arrival", 0.0),
        "classification_precision_departure": class_metrics.get("precision_departure", 0.0),
        "classification_recall_departure": class_metrics.get("recall_departure", 0.0),
        "classification_f1_departure": class_metrics.get("f1_departure", 0.0),
        "classification_accuracy_departure": class_metrics.get("accuracy_departure", 0.0),
        "regression_mae_delayed": mae_delayed,
        "regression_rmse_delayed": rmse_delayed,
        "regression_mae_nondelayed": mae_nondelayed,
        "regression_rmse_nondelayed": rmse_nondelayed,
        "regression_mae_overall": mae_overall,
        "regression_rmse_overall": rmse_overall,
        "regression_mae_delayed_arrival": arr_mae_d,
        "regression_rmse_delayed_arrival": arr_rmse_d,
        "num_delayed_samples_arrival": int(arr_n_d),
        "regression_mae_delayed_departure": dep_mae_d,
        "regression_rmse_delayed_departure": dep_rmse_d,
        "num_delayed_samples_departure": int(dep_n_d),
        "regression_mae_nondelayed_arrival": arr_mae_nd,
        "regression_rmse_nondelayed_arrival": arr_rmse_nd,
        "num_nondelayed_samples_arrival": int(arr_n_nd),
        "regression_mae_nondelayed_departure": dep_mae_nd,
        "regression_rmse_nondelayed_departure": dep_rmse_nd,
        "num_nondelayed_samples_departure": int(dep_n_nd),
        "regression_mae_overall_arrival": arr_mae_all,
        "regression_rmse_overall_arrival": arr_rmse_all,
        "regression_mae_overall_departure": dep_mae_all,
        "regression_rmse_overall_departure": dep_rmse_all,
        "num_delayed_samples": int(delayed_mask.sum()),
        "num_nondelayed_samples": int(nondelayed_mask.sum()),
        "target_epsilon": float(target_epsilon),
        "final_epsilon": float(final_epsilon),
        "epsilon_exceeded": bool(final_epsilon > float(target_epsilon)) if dp_enabled else False,
        "epsilon_overshoot": max(0.0, float(final_epsilon) - float(target_epsilon)) if dp_enabled else 0.0,
        "final_delta": float(final_delta),
    }

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["=" * 80])
        writer.writerow(["COMPREHENSIVE EVALUATION RESULTS"])
        writer.writerow(["=" * 80])
        writer.writerow([])
        writer.writerow(["MODEL", model_name])
        writer.writerow(["DP Enabled", bool(dp_enabled)])
        writer.writerow(["Target Epsilon", f"{float(target_epsilon):.4f}"])
        writer.writerow(["Final Epsilon", f"{float(final_epsilon):.4f}"])
        writer.writerow(["Final Delta", f"{float(final_delta):.2e}"])
        writer.writerow([])
        writer.writerow(["CLASSIFICATION METRICS (Macro over Arrival/Departure)"])
        writer.writerow(["Precision", f"{class_metrics['precision']:.4f}"])
        writer.writerow(["Recall", f"{class_metrics['recall']:.4f}"])
        writer.writerow(["F1 Score", f"{class_metrics['f1']:.4f}"])
        writer.writerow(["Accuracy", f"{class_metrics['accuracy']:.4f}"])
        writer.writerow([])
        writer.writerow(["Per-Channel Metrics:"])
        writer.writerow(["Channel", "Precision", "Recall", "F1 Score", "Accuracy"])
        writer.writerow([
            "Arrival",
            f"{class_metrics.get('precision_arrival', 0):.4f}",
            f"{class_metrics.get('recall_arrival', 0):.4f}",
            f"{class_metrics.get('f1_arrival', 0):.4f}",
            f"{class_metrics.get('accuracy_arrival', 0):.4f}",
        ])
        writer.writerow([
            "Departure",
            f"{class_metrics.get('precision_departure', 0):.4f}",
            f"{class_metrics.get('recall_departure', 0):.4f}",
            f"{class_metrics.get('f1_departure', 0):.4f}",
            f"{class_metrics.get('accuracy_departure', 0):.4f}",
        ])
        writer.writerow([])
        writer.writerow(["REGRESSION (delayed)", f"MAE={mae_delayed:.4f}", f"RMSE={rmse_delayed:.4f}"])
        writer.writerow(["REGRESSION (non-delayed)", f"MAE={mae_nondelayed:.4f}", f"RMSE={rmse_nondelayed:.4f}"])
        writer.writerow(["REGRESSION (overall)", f"MAE={mae_overall:.4f}", f"RMSE={rmse_overall:.4f}"])
        writer.writerow([])
        writer.writerow(["SUMMARY (raw metric/value)"])
        writer.writerow(["metric", "value"])
        for k, v in summary.items():
            writer.writerow([k, v])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-stage model comparison")
    parser.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=18)
    parser.add_argument("--horizons", type=int, nargs=1, default=[12], choices=[3, 6, 12, 24])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--use_node_level", action="store_true", default=True)
    parser.add_argument("--exclude_time_features", action="store_true", default=True)
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage3_epochs", type=int, default=14)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--stage1_lr", type=float, default=None)
    parser.add_argument("--stage2_lr", type=float, default=None)
    parser.add_argument("--stage3_lr", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dp", action="store_true", default=False, help="Enable DP-SGD via Opacus")
    parser.add_argument("--dp_accountant", type=str, default="rdp", choices=["rdp", "prv", "gdp"])
    parser.add_argument("--noise_multiplier", type=float, default=1.0, help="DP noise multiplier (sigma)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="DP clipping norm")
    parser.add_argument("--epsilon", type=float, default=7.5, help="Target epsilon for tracking")
    parser.add_argument("--target_delta", type=float, default=1e-5, help="Target delta for epsilon accounting")
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["cnn", "lstm", "gru", "bilstm", "cnnlstm", "attnlstm", "stpn"],
        choices=["cnn", "lstm", "gru", "bilstm", "cnnlstm", "attnlstm", "stpn"],
    )
    parser.add_argument("--output_dir", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed is not None:
        set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

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

    if args.exclude_time_features:
        train_inputs = train_inputs[:, :, :-2]
        val_inputs = val_inputs[:, :, :-2]
        test_inputs = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]

    horizons = sorted({h for h in args.horizons if h > 0})
    if len(horizons) != 1:
        raise ValueError("Pass exactly one horizon via --horizons (3/6/12/24)")

    max_horizon = horizons[0]
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences

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

    # Node-level loaders (CNN/LSTM/GRU)
    train_x_flat = train_x.reshape(-1, args.seq_len, feature_dim)
    val_x_flat = val_x.reshape(-1, args.seq_len, feature_dim)
    test_x_flat = test_x.reshape(-1, args.seq_len, feature_dim)

    train_y_cls_flat = train_y_cls.reshape(-1, delay_dim)
    val_y_cls_flat = val_y_cls.reshape(-1, delay_dim)
    test_y_cls_flat = test_y_cls.reshape(-1, delay_dim)

    train_y_reg_flat = train_y_reg.reshape(-1, delay_dim)
    val_y_reg_flat = val_y_reg.reshape(-1, delay_dim)
    test_y_reg_flat = test_y_reg.reshape(-1, delay_dim)

    train_loader = DataLoader(
        TensorDataset(train_x_flat, train_y_cls_flat, train_y_reg_flat),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x_flat, val_y_cls_flat, val_y_reg_flat),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    pos_rate = train_y_cls_flat.float().mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    stage1_lr = args.stage1_lr if args.stage1_lr is not None else args.lr
    stage2_lr = args.stage2_lr if args.stage2_lr is not None else args.lr
    stage3_lr = args.stage3_lr if args.stage3_lr is not None else (args.lr * 0.01)

    if args.dp:
        if OPACUS_AVAILABLE:
            print(
                f"[DP] Enabled (accountant={args.dp_accountant}, "
                f"noise_multiplier={args.noise_multiplier}, max_grad_norm={args.max_grad_norm})"
            )
            if "stpn" in args.models:
                print("[DP] STPN runs without DP in this script (compatibility path).")
        else:
            print("[DP] Requested but Opacus is not installed. Running without DP.")
            args.dp = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir == "auto":
        output_dir = f"three_stage_compare_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # STPN loaders
    if STPN_AVAILABLE:
        train_x_stpn = train_x.reshape(train_x.shape[0], num_nodes, args.seq_len, feature_dim)
        val_x_stpn = val_x.reshape(val_x.shape[0], num_nodes, args.seq_len, feature_dim)
        test_x_stpn = test_x.reshape(test_x.shape[0], num_nodes, args.seq_len, feature_dim)

        train_stpn_loader = DataLoader(
            TensorDataset(train_x_stpn, train_y_cls, train_y_reg),
            batch_size=max(4, min(16, args.batch_size // 8)),
            shuffle=True,
            drop_last=False,
        )
        val_stpn_loader = DataLoader(
            TensorDataset(val_x_stpn, val_y_cls, val_y_reg),
            batch_size=max(4, min(16, args.batch_size // 8)),
            shuffle=False,
            drop_last=False,
        )
        test_stpn_loader = TensorDataset(test_x_stpn, test_y_cls, test_y_reg)

        base_dir = os.path.dirname(__file__)
        data_dir = args.data_source
        adjacency_candidates = [
            os.path.join(data_dir, "dist_mx.npy"),
            os.path.join(data_dir, "adj_mx.npy"),
            os.path.join(base_dir, data_dir, "dist_mx.npy"),
            os.path.join(base_dir, data_dir, "adj_mx.npy"),
        ]
        adj_file = next((p for p in adjacency_candidates if os.path.exists(p)), None)
        if adj_file is not None:
            adj_mx = np.load(adj_file)
            print(f"[STPN] Loaded adjacency: {adj_file}")
        else:
            adj_mx = np.eye(num_nodes)
            print(
                "[STPN] Using identity adjacency "
                f"(no adjacency file found in {data_dir}; tried dist_mx.npy/adj_mx.npy)"
            )
        adj_norm = adj_mx + np.eye(adj_mx.shape[0])
        row_sum = adj_norm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        adj_norm = adj_norm / row_sum
        supports = [
            torch.FloatTensor(adj_norm).to(device),
            torch.FloatTensor(adj_norm @ adj_norm).to(device),
            torch.FloatTensor(adj_norm @ adj_norm @ adj_norm).to(device),
        ]
    else:
        train_stpn_loader = None
        val_stpn_loader = None
        test_stpn_loader = None
        supports = []

    # Run models
    if "cnn" in args.models:
        print("\n[MODEL] CNN")
        cnn_module = _load_cnn_module()
        cnn_base = cnn_module.SequentialTwoStagePredictor(
            in_channels=args.seq_len * feature_dim,
            out_channels=delay_dim,
            hidden_channels=args.hidden_channels,
            regressor_extra_layer=True,
            seq_len=args.seq_len,
        ).to(device)
        cnn = CNNAdapter(cnn_base).to(device)

        history = []
        history += _train_stage1_node(cnn, train_loader, val_loader, device, args.stage1_epochs, stage1_lr, pos_weight, args.patience, "CNN", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage2_node(cnn, train_loader, val_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, args.patience, "CNN", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage3_node(cnn, train_loader, val_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, args.patience, "CNN", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)

        cls_metrics, _, reg_preds, reg_targets = _evaluate_model(
            cnn,
            test_x_flat,
            test_y_cls_flat,
            test_y_reg_flat,
            device,
            args.class_threshold,
            scaler,
        )

        final_epsilon = float(history[-1].get("epsilon", 0.0)) if history else 0.0
        _write_history_csv(os.path.join(output_dir, "cnn_history.csv"), history)
        _write_results_table(
            os.path.join(output_dir, "cnn_results_table.csv"),
            "CNN",
            cls_metrics,
            reg_preds,
            reg_targets,
            args.class_threshold,
            args.delay_threshold,
            scaler,
            final_epsilon=final_epsilon,
            final_delta=float(args.target_delta) if args.dp else 0.0,
            target_epsilon=float(args.epsilon),
            dp_enabled=bool(args.dp),
        )

    if "lstm" in args.models:
        print("\n[MODEL] LSTM")
        lstm = RecurrentDualModel("lstm", feature_dim, 64, delay_dim).to(device)
        history = []
        history += _train_stage1_node(lstm, train_loader, val_loader, device, args.stage1_epochs, stage1_lr, pos_weight, args.patience, "LSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage2_node(lstm, train_loader, val_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, args.patience, "LSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage3_node(lstm, train_loader, val_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, args.patience, "LSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)

        cls_metrics, _, reg_preds, reg_targets = _evaluate_model(
            lstm,
            test_x_flat,
            test_y_cls_flat,
            test_y_reg_flat,
            device,
            args.class_threshold,
            scaler,
        )

        final_epsilon = float(history[-1].get("epsilon", 0.0)) if history else 0.0
        _write_history_csv(os.path.join(output_dir, "lstm_history.csv"), history)
        _write_results_table(
            os.path.join(output_dir, "lstm_results_table.csv"),
            "LSTM",
            cls_metrics,
            reg_preds,
            reg_targets,
            args.class_threshold,
            args.delay_threshold,
            scaler,
            final_epsilon=final_epsilon,
            final_delta=float(args.target_delta) if args.dp else 0.0,
            target_epsilon=float(args.epsilon),
            dp_enabled=bool(args.dp),
        )

    if "gru" in args.models:
        print("\n[MODEL] GRU")
        gru = RecurrentDualModel("gru", feature_dim, 64, delay_dim).to(device)
        history = []
        history += _train_stage1_node(gru, train_loader, val_loader, device, args.stage1_epochs, stage1_lr, pos_weight, args.patience, "GRU", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage2_node(gru, train_loader, val_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, args.patience, "GRU", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage3_node(gru, train_loader, val_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, args.patience, "GRU", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)

        cls_metrics, _, reg_preds, reg_targets = _evaluate_model(
            gru,
            test_x_flat,
            test_y_cls_flat,
            test_y_reg_flat,
            device,
            args.class_threshold,
            scaler,
        )

        final_epsilon = float(history[-1].get("epsilon", 0.0)) if history else 0.0
        _write_history_csv(os.path.join(output_dir, "gru_history.csv"), history)
        _write_results_table(
            os.path.join(output_dir, "gru_results_table.csv"),
            "GRU",
            cls_metrics,
            reg_preds,
            reg_targets,
            args.class_threshold,
            args.delay_threshold,
            scaler,
            final_epsilon=final_epsilon,
            final_delta=float(args.target_delta) if args.dp else 0.0,
            target_epsilon=float(args.epsilon),
            dp_enabled=bool(args.dp),
        )

    if "bilstm" in args.models:
        print("\n[MODEL] BiLSTM")
        bilstm = BiLSTMDualModel(feature_dim, 64, delay_dim).to(device)
        history = []
        history += _train_stage1_node(bilstm, train_loader, val_loader, device, args.stage1_epochs, stage1_lr, pos_weight, args.patience, "BiLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage2_node(bilstm, train_loader, val_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, args.patience, "BiLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage3_node(bilstm, train_loader, val_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, args.patience, "BiLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)

        cls_metrics, _, reg_preds, reg_targets = _evaluate_model(
            bilstm,
            test_x_flat,
            test_y_cls_flat,
            test_y_reg_flat,
            device,
            args.class_threshold,
            scaler,
        )

        final_epsilon = float(history[-1].get("epsilon", 0.0)) if history else 0.0
        _write_history_csv(os.path.join(output_dir, "bilstm_history.csv"), history)
        _write_results_table(
            os.path.join(output_dir, "bilstm_results_table.csv"),
            "BiLSTM",
            cls_metrics,
            reg_preds,
            reg_targets,
            args.class_threshold,
            args.delay_threshold,
            scaler,
            final_epsilon=final_epsilon,
            final_delta=float(args.target_delta) if args.dp else 0.0,
            target_epsilon=float(args.epsilon),
            dp_enabled=bool(args.dp),
        )

    if "cnnlstm" in args.models:
        print("\n[MODEL] CNN-LSTM")
        cnnlstm = CNNLSTMDualModel(feature_dim, 64, delay_dim).to(device)
        history = []
        history += _train_stage1_node(cnnlstm, train_loader, val_loader, device, args.stage1_epochs, stage1_lr, pos_weight, args.patience, "CNNLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage2_node(cnnlstm, train_loader, val_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, args.patience, "CNNLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage3_node(cnnlstm, train_loader, val_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, args.patience, "CNNLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)

        cls_metrics, _, reg_preds, reg_targets = _evaluate_model(
            cnnlstm,
            test_x_flat,
            test_y_cls_flat,
            test_y_reg_flat,
            device,
            args.class_threshold,
            scaler,
        )

        final_epsilon = float(history[-1].get("epsilon", 0.0)) if history else 0.0
        _write_history_csv(os.path.join(output_dir, "cnnlstm_history.csv"), history)
        _write_results_table(
            os.path.join(output_dir, "cnnlstm_results_table.csv"),
            "CNN-LSTM",
            cls_metrics,
            reg_preds,
            reg_targets,
            args.class_threshold,
            args.delay_threshold,
            scaler,
            final_epsilon=final_epsilon,
            final_delta=float(args.target_delta) if args.dp else 0.0,
            target_epsilon=float(args.epsilon),
            dp_enabled=bool(args.dp),
        )

    if "attnlstm" in args.models:
        print("\n[MODEL] Attention-LSTM")
        attnlstm = AttentionLSTMDualModel(feature_dim, 64, delay_dim).to(device)
        history = []
        history += _train_stage1_node(attnlstm, train_loader, val_loader, device, args.stage1_epochs, stage1_lr, pos_weight, args.patience, "AttentionLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage2_node(attnlstm, train_loader, val_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, args.patience, "AttentionLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)
        history += _train_stage3_node(attnlstm, train_loader, val_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, args.patience, "AttentionLSTM", args.dp, args.dp_accountant, args.noise_multiplier, args.max_grad_norm, args.target_delta, args.epsilon)

        cls_metrics, _, reg_preds, reg_targets = _evaluate_model(
            attnlstm,
            test_x_flat,
            test_y_cls_flat,
            test_y_reg_flat,
            device,
            args.class_threshold,
            scaler,
        )

        final_epsilon = float(history[-1].get("epsilon", 0.0)) if history else 0.0
        _write_history_csv(os.path.join(output_dir, "attnlstm_history.csv"), history)
        _write_results_table(
            os.path.join(output_dir, "attnlstm_results_table.csv"),
            "Attention-LSTM",
            cls_metrics,
            reg_preds,
            reg_targets,
            args.class_threshold,
            args.delay_threshold,
            scaler,
            final_epsilon=final_epsilon,
            final_delta=float(args.target_delta) if args.dp else 0.0,
            target_epsilon=float(args.epsilon),
            dp_enabled=bool(args.dp),
        )

    if "stpn" in args.models:
        if not STPN_AVAILABLE:
            print("[STPN] Skipped (not available)")
        else:
            print("\n[MODEL] STPN")
            stpn_model = STPNDualModel(supports, args.seq_len, len(horizons), feature_dim, delay_dim).to(device)
            history = []
            history += _train_stage_stpn(1, stpn_model, train_stpn_loader, val_stpn_loader, device, args.stage1_epochs, stage1_lr, args.delay_threshold, scaler, pos_weight, args.patience)
            history += _train_stage_stpn(2, stpn_model, train_stpn_loader, val_stpn_loader, device, args.stage2_epochs, stage2_lr, args.delay_threshold, scaler, pos_weight, args.patience)
            history += _train_stage_stpn(3, stpn_model, train_stpn_loader, val_stpn_loader, device, args.stage3_epochs, stage3_lr, args.delay_threshold, scaler, pos_weight, args.patience)

            cls_metrics, _, reg_preds, reg_targets = _evaluate_stpn(
                stpn_model,
                test_stpn_loader.tensors[0],
                test_stpn_loader.tensors[1],
                test_stpn_loader.tensors[2],
                device,
                args.class_threshold,
            )

            _write_history_csv(os.path.join(output_dir, "stpn_history.csv"), history)
            _write_results_table(
                os.path.join(output_dir, "stpn_results_table.csv"),
                "STPN",
                cls_metrics,
                reg_preds.reshape(-1, delay_dim),
                reg_targets.reshape(-1, delay_dim),
                args.class_threshold,
                args.delay_threshold,
                scaler,
                final_epsilon=0.0,
                final_delta=0.0,
                target_epsilon=float(args.epsilon),
                dp_enabled=False,
            )

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
