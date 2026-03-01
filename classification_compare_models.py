"""Classification-only benchmark runner.

This script imports the CNN model from "cnnopacus - deepDualReg.py" and compares
classification performance against LSTM, GRU, and STPN baselines.

Outputs:
- Per-model metrics CSV
- Aggregate comparison CSV
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
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))

from classifykat import EarlyStopping, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level, build_sequences


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


# Try to import STPN
try:
    from model import STPN
    STPN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] STPN not available: {e}")
    STPN = None
    STPN_AVAILABLE = False


class LSTMClassifier(nn.Module):
    """LSTM for multi-label classification (arrival/departure)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2, out_dim: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class GRUClassifier(nn.Module):
    """GRU for multi-label classification (arrival/departure)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2, out_dim: int = 2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h_n = self.gru(x)
        return self.fc(h_n[-1])


class STPNClassifier(nn.Module):
    """STPN wrapper that outputs node-level classification logits."""

    def __init__(
        self,
        stpn: nn.Module,
        supports: List[torch.Tensor],
        in_len: int,
        out_len: int,
    ) -> None:
        super().__init__()
        self.stpn = stpn
        self.supports = supports
        self.in_len = int(in_len)
        self.out_len = int(out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, V, T_in]
        bsz = x.shape[0]
        t_in = torch.arange(self.in_len, device=x.device).unsqueeze(0).repeat(bsz, 1)
        t_out = torch.arange(self.out_len, device=x.device).unsqueeze(0).repeat(bsz, 1)
        outputs = self.stpn(x, t_in, self.supports, t_out, None)
        # outputs: [B, C, V, T_out] -> pool over time
        pooled = outputs.mean(dim=3)  # [B, C, V]
        return pooled.permute(0, 2, 1)  # [B, V, C]


def normalized_adjacency_matrix(adj: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """Compute D^{-1/2} A D^{-1/2} for random walk normalization."""
    if symmetrize:
        adj = (adj + adj.T) / 2.0
    adj = adj + np.eye(adj.shape[0])
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt = np.diag(d_inv_sqrt)
    return d_inv_sqrt @ adj @ d_inv_sqrt


def build_node_level_tensors(
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    test_x: torch.Tensor,
    test_y_cls: torch.Tensor,
    seq_len: int,
    feature_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n_train, n_nodes, _ = train_x.shape
    n_val = val_x.shape[0]
    n_test = test_x.shape[0]

    train_x_4d = train_x.view(n_train, n_nodes, seq_len, feature_dim)
    val_x_4d = val_x.view(n_val, n_nodes, seq_len, feature_dim)
    test_x_4d = test_x.view(n_test, n_nodes, seq_len, feature_dim)

    train_x_nodes = train_x_4d.reshape(n_train * n_nodes, seq_len, feature_dim)
    val_x_nodes = val_x_4d.reshape(n_val * n_nodes, seq_len, feature_dim)
    test_x_nodes = test_x_4d.reshape(n_test * n_nodes, seq_len, feature_dim)

    train_y_nodes = train_y_cls.reshape(n_train * n_nodes, -1)
    val_y_nodes = val_y_cls.reshape(n_val * n_nodes, -1)
    test_y_nodes = test_y_cls.reshape(n_test * n_nodes, -1)

    return (
        train_x_nodes,
        train_y_nodes,
        val_x_nodes,
        val_y_nodes,
        test_x_nodes,
        test_y_nodes,
    )


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


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    model_name: str,
    use_logits: bool = True,
) -> Dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    early_stopping = EarlyStopping(patience=patience, mode="max")

    best_f1 = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        model.eval()
        val_probs_list = []
        val_targets_list = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                probs = torch.sigmoid(logits) if use_logits else logits
                val_probs_list.append(probs.cpu())
                val_targets_list.append(batch_y.cpu())

        val_probs = torch.cat(val_probs_list, dim=0).numpy()
        val_targets = torch.cat(val_targets_list, dim=0).numpy()
        val_metrics = classification_metrics_per_channel(val_probs, val_targets)

        if val_metrics["f1"] > best_f1:
            best_f1 = float(val_metrics["f1"])
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"[{model_name}] Epoch {epoch}/{epochs} | "
                f"Loss: {np.mean(epoch_losses):.4f} | Val F1: {val_metrics['f1']:.4f}"
            )

        if early_stopping(val_metrics["f1"], epoch):
            print(f"[{model_name}] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return val_metrics


def evaluate_classifier(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    use_logits: bool = True,
) -> Dict[str, float]:
    model.eval()
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits) if use_logits else logits
            probs_list.append(probs.cpu())
            targets_list.append(batch_y.cpu())

    probs = torch.cat(probs_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()
    return classification_metrics_per_channel(probs, targets)


def evaluate_cnn_classifier_deepdual(
    model: nn.Module,
    test_x: torch.Tensor,
    test_y_cls: torch.Tensor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    class_threshold: float,
) -> Dict[str, float]:
    model.eval()
    probs_list = []
    targets_list = []

    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            _, node_logits = model.forward_classifier(data)
            probs_list.append(torch.sigmoid(node_logits).cpu())
            targets_list.append(test_y_cls[i].cpu())

    probs = torch.cat(probs_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()
    return classification_metrics_per_channel(
        probs,
        targets,
        channel_names=("arrival", "departure"),
        prob_threshold=class_threshold,
    )


def evaluate_stpn_classifier(
    model: STPNClassifier,
    test_x: torch.Tensor,
    test_y: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> Dict[str, float]:
    model.eval()
    probs_list = []
    targets_list = []
    with torch.no_grad():
        for i in range(0, test_x.shape[0], batch_size):
            batch_x = test_x[i : i + batch_size].to(device)
            batch_y = test_y[i : i + batch_size].to(device)
            logits = model(batch_x)
            probs = torch.sigmoid(logits)
            probs_list.append(probs.cpu())
            targets_list.append(batch_y.cpu())

    probs = torch.cat(probs_list, dim=0).numpy()
    targets = torch.cat(targets_list, dim=0).numpy()
    return classification_metrics_per_channel(probs, targets)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classification-only model comparison")
    parser.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=18)
    parser.add_argument("--horizons", type=int, nargs=1, default=[12], choices=[3, 6, 12, 24])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--use_node_level", action="store_true", default=True)
    parser.add_argument("--exclude_time_features", action="store_true", default=True)
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["cnn", "lstm", "gru", "stpn"],
        choices=["cnn", "lstm", "gru", "stpn"],
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

    (
        train_x_nodes,
        train_y_nodes,
        val_x_nodes,
        val_y_nodes,
        test_x_nodes,
        test_y_nodes,
    ) = build_node_level_tensors(
        train_x,
        train_y_cls,
        val_x,
        val_y_cls,
        test_x,
        test_y_cls,
        args.seq_len,
        feature_dim,
    )

    pos_rate = train_y_nodes.mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    train_loader = DataLoader(
        TensorDataset(train_x_nodes, train_y_nodes),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x_nodes, val_y_nodes),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x_nodes, test_y_nodes),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir == "auto":
        output_dir = f"classification_compare_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    results: Dict[str, Dict[str, float]] = {}

    if "cnn" in args.models:
        print("\n[MODEL] CNN (dual-reg encoder) classification")
        cnn_module = _load_cnn_module()
        cnn_model = cnn_module.SequentialTwoStagePredictor(
            in_channels=args.seq_len * feature_dim,
            out_channels=delay_dim,
            hidden_channels=args.hidden_channels,
            regressor_extra_layer=True,
            seq_len=args.seq_len,
        ).to(device)

        class CNNWrapper(nn.Module):
            def __init__(self, base: nn.Module):
                super().__init__()
                self.base = base

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                hidden = self.base._encode_x(x)
                return self.base.classifier(self.base.dropout_cls(hidden))

        wrapped = CNNWrapper(cnn_model).to(device)
        train_classifier(
            wrapped,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            pos_weight=pos_weight,
            patience=args.patience,
            model_name="CNN",
        )
        metrics = evaluate_cnn_classifier_deepdual(
            cnn_model,
            test_x,
            test_y_cls,
            edge_indices=(edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device)),
            device=device,
            class_threshold=args.class_threshold,
        )
        results["CNN"] = metrics

    if "lstm" in args.models:
        print("\n[MODEL] LSTM classification")
        lstm = LSTMClassifier(feature_dim, hidden_dim=64, dropout=0.2, out_dim=delay_dim).to(device)
        train_classifier(
            lstm,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            pos_weight=pos_weight,
            patience=args.patience,
            model_name="LSTM",
        )
        metrics = evaluate_classifier(lstm, test_loader, device)
        results["LSTM"] = metrics

    if "gru" in args.models:
        print("\n[MODEL] GRU classification")
        gru = GRUClassifier(feature_dim, hidden_dim=64, dropout=0.2, out_dim=delay_dim).to(device)
        train_classifier(
            gru,
            train_loader,
            val_loader,
            device,
            epochs=args.epochs,
            lr=args.lr,
            pos_weight=pos_weight,
            patience=args.patience,
            model_name="GRU",
        )
        metrics = evaluate_classifier(gru, test_loader, device)
        results["GRU"] = metrics

    if "stpn" in args.models:
        print("\n[MODEL] STPN classification")
        if not STPN_AVAILABLE:
            print("[STPN] Skipped (not available)")
        else:
            adj_file = os.path.join(os.path.dirname(__file__), "dist_mx.npy")
            if os.path.exists(adj_file):
                adj_mx = np.load(adj_file)
            else:
                adj_mx = np.eye(num_nodes)
                print("[STPN] Using identity adjacency (dist_mx.npy not found)")

            adj_norm = normalized_adjacency_matrix(adj_mx)
            supports = [
                torch.FloatTensor(adj_norm).to(device),
                torch.FloatTensor(adj_norm @ adj_norm).to(device),
                torch.FloatTensor(adj_norm @ adj_norm @ adj_norm).to(device),
            ]

            stpn = STPN(
                h_layers=2,
                in_channels=feature_dim,
                hidden_channels=[32, 32, 16],
                out_channels=delay_dim,
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
            ).to(device)

            stpn_model = STPNClassifier(stpn, supports, in_len=args.seq_len, out_len=len(horizons)).to(device)
            stpn_optimizer = torch.optim.Adam(stpn_model.parameters(), lr=args.lr, weight_decay=1e-4)
            stpn_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
            stpn_es = EarlyStopping(patience=args.patience, mode="max")

            # Prepare STPN tensors: [B, F, V, T]
            train_x_4d = train_x.view(train_x.shape[0], num_nodes, args.seq_len, feature_dim)
            val_x_4d = val_x.view(val_x.shape[0], num_nodes, args.seq_len, feature_dim)
            test_x_4d = test_x.view(test_x.shape[0], num_nodes, args.seq_len, feature_dim)

            train_x_stpn = train_x_4d.permute(0, 3, 1, 2)
            val_x_stpn = val_x_4d.permute(0, 3, 1, 2)
            test_x_stpn = test_x_4d.permute(0, 3, 1, 2)

            best_state = None
            best_f1 = -1.0

            for epoch in range(1, args.epochs + 1):
                stpn_model.train()
                epoch_losses = []

                for i in range(0, train_x_stpn.shape[0], args.batch_size):
                    batch_x = train_x_stpn[i : i + args.batch_size].to(device)
                    batch_y = train_y_cls[i : i + args.batch_size].to(device)

                    stpn_optimizer.zero_grad(set_to_none=True)
                    logits = stpn_model(batch_x)
                    loss = stpn_loss_fn(logits, batch_y)
                    loss.backward()
                    stpn_optimizer.step()
                    epoch_losses.append(float(loss.item()))

                stpn_model.eval()
                val_metrics = evaluate_stpn_classifier(
                    stpn_model,
                    val_x_stpn,
                    val_y_cls,
                    device,
                    batch_size=args.batch_size,
                )

                if val_metrics["f1"] > best_f1:
                    best_f1 = float(val_metrics["f1"])
                    best_state = {k: v.detach().cpu() for k, v in stpn_model.state_dict().items()}

                if epoch % 5 == 0 or epoch == args.epochs:
                    print(
                        f"[STPN] Epoch {epoch}/{args.epochs} | "
                        f"Loss: {np.mean(epoch_losses):.4f} | Val F1: {val_metrics['f1']:.4f}"
                    )

                if stpn_es(val_metrics["f1"], epoch):
                    print(f"[STPN] Early stopping at epoch {epoch}")
                    break

            if best_state is not None:
                stpn_model.load_state_dict(best_state)

            metrics = evaluate_stpn_classifier(
                stpn_model,
                test_x_stpn,
                test_y_cls,
                device,
                batch_size=args.batch_size,
            )
            results["STPN"] = metrics

    # Write per-model CSVs
    for model_name, metrics in results.items():
        out_path = os.path.join(output_dir, f"{model_name.lower()}_classification_metrics.csv")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric", "value"])
            for k, v in metrics.items():
                writer.writerow([k, v])

    # Write aggregate summary
    summary_path = os.path.join(output_dir, "classification_summary.csv")
    with open(summary_path, "w", newline="") as f:
        fieldnames = ["model", "precision", "recall", "f1", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            writer.writerow(
                {
                    "model": model_name,
                    "precision": metrics.get("precision", 0.0),
                    "recall": metrics.get("recall", 0.0),
                    "f1": metrics.get("f1", 0.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                }
            )

    print(f"\n✓ Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
