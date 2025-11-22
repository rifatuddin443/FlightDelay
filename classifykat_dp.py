"""Differentially private variant of the sequential KAN-GAT pipeline.

This script reuses the original `classifykat` architecture but trains the
stage-1 classifier (encoder + classifier head) with Opacus' PrivacyEngine.
Stage-2 regression remains unchanged. The goal is to let users opt into
DP-SGD when fitting the delay detector, while keeping the remainder of the
workflow (regressor training, evaluation, CSV exports) identical to the
non-DP version.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from opacus import PrivacyEngine
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader

# Reuse the original implementation details from classifykat
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (  # noqa: E402
    EarlyStopping,
    SequentialTwoStagePredictor,
    build_sequences,
    classification_metrics,
    load_flight_data,
    regression_metrics,
    set_seed,
    train_stage2_regressor,
)
from baseline_methods import test_error  # noqa: E402


class GraphSequenceData(Data):
    """Custom PyG data object that keeps multiple edge_index tensors."""

    def __inc__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key in {"edge_index_adj", "edge_index_od", "edge_index_od_t"}:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class GraphSequenceDataset(Dataset):
    """Wraps windowed sequences into PyG graph samples for batching."""

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
class DPConfig:
    enabled: bool
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    max_grad_norm: float


def _manual_dp_clip_and_noise(
    param_groups: List[Dict],
    max_grad_norm: float,
    noise_multiplier: float,
    batch_size: int,
    device: torch.device,
) -> None:
    grads = [p.grad for g in param_groups for p in g["params"] if p.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))
    clip_coef = min(1.0, max_grad_norm / (total_norm + 1e-6))
    for g in grads:
        g.mul_(clip_coef)
        noise = torch.normal(
            mean=0.0,
            std=noise_multiplier * max_grad_norm,
            size=g.shape,
            device=device,
        )
        g.add_(noise / max(1, batch_size))


def _prepare_dataloaders(
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    batch_size: int,
    dp_enabled: bool,
) -> Tuple[PyGDataLoader, PyGDataLoader]:
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices

    train_dataset = GraphSequenceDataset(
        train_x.cpu(),
        train_y_reg.cpu(),
        train_y_cls.cpu(),
        edge_index_adj.cpu(),
        edge_index_od.cpu(),
        edge_index_od_t.cpu(),
    )
    val_dataset = GraphSequenceDataset(
        val_x.cpu(),
        val_y_reg.cpu(),
        val_y_cls.cpu(),
        edge_index_adj.cpu(),
        edge_index_od.cpu(),
        edge_index_od_t.cpu(),
    )

    train_loader = PyGDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=dp_enabled,  # required for Opacus to keep sample rate fixed
    )
    val_loader = PyGDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def train_stage1_classifier_dp(
    model: SequentialTwoStagePredictor,
    train_loader: PyGDataLoader,
    val_loader: PyGDataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: float,
    patience: int,
    dp_config: DPConfig,
) -> Tuple[List[Dict], PrivacyEngine | None]:
    """Train encoder + classifier with optional DP-SGD."""

    for param in model.regressor.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    cls_loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    privacy_engine: PrivacyEngine | None = None
    manual_dp = False
    if dp_config.enabled:
        privacy_engine = PrivacyEngine()
        try:
            if dp_config.target_epsilon > 0:
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=dp_config.target_epsilon,
                    target_delta=dp_config.target_delta,
                    epochs=epochs,
                    max_grad_norm=dp_config.max_grad_norm,
                )
            else:
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=dp_config.noise_multiplier,
                    max_grad_norm=dp_config.max_grad_norm,
                )
        except NotImplementedError as err:
            print("⚠️  Opacus could not wrap the model (" + str(err) + ")")
            print("   Falling back to manual gradient clipping + Gaussian noise.")
            privacy_engine = None
            manual_dp = True

    history: List[Dict] = []
    best_f1 = 0.0
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="max")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model.forward_classifier(batch)
            loss = cls_loss_fn(logits, batch.y_cls)
            loss.backward()
            if manual_dp:
                _manual_dp_clip_and_noise(
                    optimizer.param_groups,
                    max_grad_norm=dp_config.max_grad_norm,
                    noise_multiplier=dp_config.noise_multiplier,
                    batch_size=getattr(batch, "num_graphs", batch.x.shape[0]),
                    device=device,
                )
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, logits = model.forward_classifier(batch)
                val_probs.append(torch.sigmoid(logits).cpu())
                val_targets.append(batch.y_cls.cpu())

        val_probs_np = torch.cat(val_probs, dim=0).numpy()
        val_targets_np = torch.cat(val_targets, dim=0).numpy()
        val_metrics = classification_metrics(
            val_probs_np.reshape(-1, 1),
            val_targets_np.reshape(-1, 1),
        )

        current_epsilon = "No DP"
        if manual_dp and dp_config.enabled:
            current_epsilon = "manual"
        elif dp_config.enabled and privacy_engine is not None:
            try:
                current_epsilon = f"{privacy_engine.get_epsilon(dp_config.target_delta):.3f}"
            except Exception:
                current_epsilon = "N/A"

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
            }
        )

        print(
            f"Epoch {epoch}/{epochs} | Loss {history[-1]['train_loss']:.4f} | "
            f"Val F1 {val_metrics['f1']:.4f} | Epsilon {current_epsilon}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "encoder": model.encoder.state_dict(),
                "classifier": model.classifier.state_dict(),
            }
            print("  -> New best classifier checkpoint")

        if early_stopping(val_metrics["f1"], epoch):
            break

    if best_state is not None:
        model.encoder.load_state_dict(best_state["encoder"])
        model.classifier.load_state_dict(best_state["classifier"])

    for param in model.regressor.parameters():
        param.requires_grad = True

    return history, privacy_engine


def final_evaluation(
    model: SequentialTwoStagePredictor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
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
) -> None:
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices

    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    torch.save(
        {
            "encoder": model.encoder.state_dict(),
            "classifier": model.classifier.state_dict(),
            "regressor": model.regressor.state_dict(),
        },
        model_path,
    )

    checkpoint = torch.load(model_path, map_location=device)
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.classifier.load_state_dict(checkpoint["classifier"])
    model.regressor.load_state_dict(checkpoint["regressor"])

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION (DP Variant)")
    print("=" * 80)

    model.eval()
    logits_list, reg_list = [], []
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            logits, reg = model(data)
            logits_list.append(torch.sigmoid(logits).cpu().numpy())
            reg_list.append(reg.cpu().numpy())

    test_probs = np.array(logits_list)
    test_reg_preds = np.array(reg_list)

    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_y_cls.cpu().numpy().reshape(-1, 1),
    )

    test_mask = test_probs >= class_threshold
    gated_preds = test_reg_preds * test_mask

    num_forecast_steps = len(horizons)
    preds_flat = gated_preds.reshape(-1, num_forecast_steps * delay_dim)
    targets_flat = test_y_reg.cpu().numpy().reshape(-1, num_forecast_steps * delay_dim)

    preds_denorm = scaler.inverse_transform(preds_flat).reshape(gated_preds.shape)
    targets_denorm = scaler.inverse_transform(targets_flat).reshape(test_y_reg.shape)

    preds_h = preds_denorm.reshape(-1, num_nodes, num_forecast_steps, delay_dim)
    targets_h = targets_denorm.reshape(-1, num_nodes, num_forecast_steps, delay_dim)

    per_horizon_metrics = {}
    for idx, horizon in enumerate(horizons):
        arrival_preds = preds_h[:, :, idx, 0]
        arrival_targets = targets_h[:, :, idx, 0]
        dep_preds = preds_h[:, :, idx, 1]
        dep_targets = targets_h[:, :, idx, 1]

        arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
        dep_mae, dep_rmse, dep_r2 = test_error(dep_preds, dep_targets)
        per_horizon_metrics[horizon] = {
            "arrival_mae": arr_mae,
            "arrival_rmse": arr_rmse,
            "arrival_r2": arr_r2,
            "departure_mae": dep_mae,
            "departure_rmse": dep_rmse,
            "departure_r2": dep_r2,
        }

        print(f"\n{horizon}-STEP AHEAD PREDICTIONS:")
        print(
            f"  Arrival Delay  -> MAE: {arr_mae:.4f} min, "
            f"RMSE: {arr_rmse:.4f} min, R²: {arr_r2:.4f}"
        )
        print(
            f"  Departure Delay -> MAE: {dep_mae:.4f} min, "
            f"RMSE: {dep_rmse:.4f} min, R²: {dep_r2:.4f}"
        )

    targets_cls = test_y_cls.cpu().numpy()
    delayed_mask = np.broadcast_to(targets_cls.astype(bool), test_reg_preds.shape)
    reg_metrics = regression_metrics(preds_denorm, targets_denorm, delayed_mask)

    print("\nCLASSIFICATION PERFORMANCE:")
    print(
        f"  Precision: {test_cls_metrics['precision']:.4f} | "
        f"Recall: {test_cls_metrics['recall']:.4f} | "
        f"F1: {test_cls_metrics['f1']:.4f} | Accuracy: {test_cls_metrics['accuracy']:.4f}"
    )
    print("\nREGRESSION PERFORMANCE (delayed nodes):")
    print(f"  MAE: {reg_metrics['mae']:.4f} min | RMSE: {reg_metrics['rmse']:.4f} min")

    if histories:
        all_fields = sorted({key for row in histories for key in row})
        with open("kan_gat_sequential_dp_history.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for row in histories:
                writer.writerow({field: row.get(field, "") for field in all_fields})

    with open("kan_gat_sequential_dp_test_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        summary = {
            "classification_precision": test_cls_metrics["precision"],
            "classification_recall": test_cls_metrics["recall"],
            "classification_f1": test_cls_metrics["f1"],
            "classification_accuracy": test_cls_metrics["accuracy"],
            "regression_mae_delayed": reg_metrics["mae"],
            "regression_rmse_delayed": reg_metrics["rmse"],
        }
        for k, v in summary.items():
            writer.writerow([k, v])
        for horizon, metrics in per_horizon_metrics.items():
            for metric_name, metric_value in metrics.items():
                writer.writerow([f"{metric_name}_h{horizon}", metric_value])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Differentially private sequential two-stage KAN-GAT predictor",
    )
    parser.add_argument("--data_source", type=str, default="udata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=8)
    parser.add_argument("--horizons", type=int, nargs="+", default=[3, 6, 12])
    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--model_path", type=str, default="kan_gat_sequential_dp.pth")
    parser.add_argument("--seed", type=int, default=42)

    # DP arguments
    parser.add_argument("--dp",default=True,action="store_true", help="Enable DP-SGD for stage-1 training")
    parser.add_argument("--target_epsilon", type=float, default=3.0)
    parser.add_argument("--target_delta", type=float, default=1e-5)
    parser.add_argument("--noise_multiplier", type=float, default=1.2)
    parser.add_argument("--max_grad_norm", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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

    horizons = sorted({h for h in args.horizons if h > 0})
    if not horizons:
        raise ValueError("Provide at least one positive prediction horizon.")
    max_horizon = max(horizons)

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]

    in_channels = args.seq_len * feature_dim
    out_channels = len(horizons) * delay_dim

    train_x, train_y_reg, train_y_cls = build_sequences(
        train_inputs,
        train_delay_scaled,
        train_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        target_horizons=horizons,
    )
    val_x, val_y_reg, val_y_cls = build_sequences(
        val_inputs,
        val_delay_scaled,
        val_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        target_horizons=horizons,
    )
    test_x, test_y_reg, test_y_cls = build_sequences(
        test_inputs,
        test_delay_scaled,
        test_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        target_horizons=horizons,
    )

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

    cls_pos_rate = train_y_cls.mean().item()
    pos_weight = (1 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)

    train_loader, val_loader = _prepare_dataloaders(
        train_x,
        train_y_reg,
        train_y_cls,
        val_x,
        val_y_reg,
        val_y_cls,
        edge_indices,
        batch_size=args.batch_size,
        dp_enabled=args.dp,
    )

    dp_config = DPConfig(
        enabled=args.dp,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
    )

    history_stage1, _ = train_stage1_classifier_dp(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.stage1_epochs,
        lr=args.lr,
        pos_weight=pos_weight,
        patience=args.patience,
        dp_config=dp_config,
    )

    history_stage2 = train_stage2_regressor(
        model,
        train_x,
        train_y_reg,
        train_y_cls,
        val_x,
        val_y_reg,
        val_y_cls,
        edge_indices,
        device,
        epochs=args.stage2_epochs,
        lr=args.lr,
        scaler=scaler,
        class_threshold=args.class_threshold,
        batch_size=args.batch_size,
        patience=args.patience,
    )

    combined_history = history_stage1 + history_stage2

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
        args.model_path,
        combined_history,
    )


if __name__ == "__main__":
    main()
