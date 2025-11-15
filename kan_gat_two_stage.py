"""
Lightweight two-stage KAN-GAT pipeline for flight delay handling.

Stage 1: GAT encoder + KAN classifier predicts delay vs. no-delay per airport node.
Stage 2: KAN regressor outputs delay magnitudes, conditioned on the classifier.
Only nodes predicted as delayed obtain non-zero regression outputs at inference.

The model reuses the adaptive fusion idea (adjacency + OD + OD^T graphs) but keeps
all hidden sizes small so it can train quickly on modest hardware.
"""

import argparse
import csv
import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Add efficient-kan to path for local import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
from kan import KAN  # noqa: E402
from baseline_methods import test_error  # noqa: E402


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""

    def __init__(self, mean: float, std: float):
        self.mean = mean
        self.std = std if std != 0 else 1.0

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class LightweightGATEncoder(nn.Module):
    """Smaller KAN-GAT encoder that outputs compact node embeddings."""

    def __init__(self, in_channels: int, hidden_channels: int = 32, heads: int = 2):
        super().__init__()
        self.alpha_adj = nn.Parameter(torch.tensor(1.0))
        self.alpha_od = nn.Parameter(torch.tensor(1.0))
        self.alpha_od_t = nn.Parameter(torch.tensor(1.0))

        self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
        self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)

        fusion_input_dim = hidden_channels * 3 + 3
        self.fusion_kan = KAN(
            layers_hidden=[fusion_input_dim, hidden_channels, hidden_channels],
            grid_size=3,
            spline_order=2,
        )

    def forward(self, data: Data) -> torch.Tensor:
        weights = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
        w_adj, w_od, w_od_t = weights

        x_adj = self.gat_adj(data.x, data.edge_index_adj)
        x_od = self.gat_od(data.x, data.edge_index_od)
        x_od_t = self.gat_od_t(data.x, data.edge_index_od_t)

        num_nodes = x_adj.size(0)
        scalars = torch.cat([
            w_adj.expand(num_nodes, 1),
            w_od.expand(num_nodes, 1),
            w_od_t.expand(num_nodes, 1),
        ], dim=1)

        x_concat = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
        fused = F.relu(self.fusion_kan(x_concat))
        return fused

    def get_graph_weights(self) -> Dict[str, float]:
        weights = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
        return {
            'adj_weight': weights[0].item(),
            'od_weight': weights[1].item(),
            'od_t_weight': weights[2].item(),
        }


class TwoStageDelayPredictor(nn.Module):
    """Classifier + regressor sharing a lightweight encoder."""

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.encoder = LightweightGATEncoder(in_channels, hidden_channels=hidden_channels)
        embed_dim = hidden_channels

        self.classifier = KAN(
            layers_hidden=[embed_dim, embed_dim // 2, 1],
            grid_size=3,
            spline_order=2,
        )
        self.regressor = KAN(
            layers_hidden=[embed_dim, embed_dim // 2, out_channels],
            grid_size=3,
            spline_order=2,
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(data)
        logits = self.classifier(hidden)
        reg_out = self.regressor(hidden)
        return logits, reg_out


def load_flight_data(
    data_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
    od_mx = np.load(os.path.join(data_dir, 'od_mx.npy'))
    adj_mx = np.load(os.path.join(data_dir, 'dist_mx.npy'))
    delay_data = np.load(os.path.join(data_dir, 'delay.npy'))

    num_nodes = od_mx.shape[0]
    edge_index_od = torch.tensor(np.array(od_mx.nonzero()), dtype=torch.long)
    edge_index_od_t = torch.tensor(np.array(od_mx.T.nonzero()), dtype=torch.long)
    edge_index_adj = torch.tensor(np.array(adj_mx.nonzero()), dtype=torch.long)

    total_steps = delay_data.shape[1]
    train_end = int(train_ratio * total_steps)
    val_end = int((train_ratio + val_ratio) * total_steps)

    train_raw = delay_data[:, :train_end, :]
    val_raw = delay_data[:, train_end:val_end, :]
    test_raw = delay_data[:, val_end:, :]

    scaler = StandardScaler(
        mean=np.nanmean(train_raw),
        std=np.nanstd(train_raw),
    )

    def scale_and_fill(arr: np.ndarray) -> np.ndarray:
        scaled = scaler.transform(arr)
        return np.nan_to_num(scaled)

    train_scaled = scale_and_fill(train_raw)
    val_scaled = scale_and_fill(val_raw)
    test_scaled = scale_and_fill(test_raw)

    return (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_scaled,
        val_scaled,
        test_scaled,
        train_raw,
        val_raw,
        test_raw,
        scaler,
        num_nodes,
    )


def build_sequences(
    scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = scaled.shape[0]
    max_idx = scaled.shape[1] - seq_len - horizon
    x_list, y_reg_list, y_cls_list = [], [], []

    for t in range(max_idx):
        x_seq = scaled[:, t:t + seq_len, :].reshape(num_nodes, -1)
        y_seq = scaled[:, t + seq_len:t + seq_len + horizon, :].reshape(num_nodes, -1)

        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target)
        cls_flag = (np.max(np.abs(raw_target), axis=(1, 2)) >= delay_threshold).astype(np.float32)
        cls_flag = cls_flag.reshape(num_nodes, 1)

        x_list.append(x_seq)
        y_reg_list.append(y_seq)
        y_cls_list.append(cls_flag)

    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    return tensors


def classification_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    preds_bin = preds >= 0.5
    targets_bin = targets >= 0.5

    tp = np.logical_and(preds_bin, targets_bin).sum()
    fp = np.logical_and(preds_bin, ~targets_bin).sum()
    fn = np.logical_and(~preds_bin, targets_bin).sum()
    tn = np.logical_and(~preds_bin, ~targets_bin).sum()

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }


def regression_metrics(preds: np.ndarray, targets: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    if mask.sum() == 0:
        return {'mae': 0.0, 'rmse': 0.0}
    preds_sel = preds[mask]
    targets_sel = targets[mask]
    mae = float(np.mean(np.abs(preds_sel - targets_sel)))
    rmse = float(np.sqrt(np.mean((preds_sel - targets_sel) ** 2)))
    return {'mae': mae, 'rmse': rmse}


def infer_sequences(
    model: TwoStageDelayPredictor,
    x_tensors: torch.Tensor,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    device: torch.device,
):
    model.eval()
    logits_all, reg_all = [], []
    with torch.no_grad():
        for i in range(len(x_tensors)):
            data = Data(
                x=x_tensors[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            logits, reg = model(data)
            logits_all.append(torch.sigmoid(logits).cpu().numpy())
            reg_all.append(reg.cpu().numpy())
    probs = np.array(logits_all)
    reg_preds = np.array(reg_all)
    return probs, reg_preds


def compute_metrics_and_denorm(
    probs: np.ndarray,
    reg_preds: np.ndarray,
    y_reg_tensors: torch.Tensor,
    y_cls_tensors: torch.Tensor,
    scaler: StandardScaler,
    class_threshold: float,
):
    preds_mask = probs >= class_threshold
    gated_preds = reg_preds * preds_mask

    out_channels = gated_preds.shape[-1]
    reg_targets = y_reg_tensors.cpu().numpy()

    preds_flat = gated_preds.reshape(-1, out_channels)
    targets_flat = reg_targets.reshape(-1, out_channels)

    reg_preds_denorm = scaler.inverse_transform(preds_flat).reshape(gated_preds.shape)
    reg_targets_denorm = scaler.inverse_transform(targets_flat).reshape(gated_preds.shape)

    targets_cls = y_cls_tensors.cpu().numpy()
    cls_metrics = classification_metrics(
        probs.reshape(-1, 1),
        targets_cls.reshape(-1, 1),
    )

    delayed_mask = np.broadcast_to(targets_cls.astype(bool), reg_preds.shape)
    reg_metrics = regression_metrics(reg_preds_denorm, reg_targets_denorm, delayed_mask)

    metrics = {**cls_metrics, **reg_metrics}
    return metrics, reg_preds_denorm, reg_targets_denorm, preds_mask


def evaluate_split(
    model: TwoStageDelayPredictor,
    x_tensors: torch.Tensor,
    y_reg_tensors: torch.Tensor,
    y_cls_tensors: torch.Tensor,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    device: torch.device,
    scaler: StandardScaler,
    class_threshold: float,
) -> Dict[str, float]:
    probs, reg_preds = infer_sequences(
        model,
        x_tensors,
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        device,
    )

    metrics, _, _, _ = compute_metrics_and_denorm(
        probs,
        reg_preds,
        y_reg_tensors,
        y_cls_tensors,
        scaler,
        class_threshold,
    )

    return metrics


def run_multihorizon_test(
    model: TwoStageDelayPredictor,
    test_scaled: np.ndarray,
    test_raw: np.ndarray,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    device: torch.device,
    scaler: StandardScaler,
    seq_len: int,
    horizons: Tuple[int, ...],
    class_threshold: float,
):
    num_nodes, _, base_features = test_scaled.shape
    results = {}

    print("\n" + "=" * 80)
    print("MULTI-HORIZON TESTING")
    print("=" * 80)

    for horizon in horizons:
        max_idx = test_scaled.shape[1] - seq_len - horizon
        if max_idx <= 0:
            print(f"Skipping horizon {horizon}: insufficient timesteps")
            continue

        test_x_list = []
        test_y_list = []

        for t in range(max_idx):
            x_seq = test_scaled[:, t:t + seq_len, :].reshape(num_nodes, -1)
            y_seq = test_raw[:, t + seq_len:t + seq_len + horizon, :]
            y_seq = np.nan_to_num(y_seq).reshape(num_nodes, -1)
            test_x_list.append(x_seq)
            test_y_list.append(y_seq)

        test_x_all = torch.tensor(np.stack(test_x_list), dtype=torch.float32)
        test_y_all = np.stack(test_y_list)

        print(f"Horizon {horizon}: created {len(test_x_list)} sequences")

        all_preds = []

        model.eval()
        with torch.no_grad():
            for i in range(len(test_x_all)):
                current_input = test_x_all[i].to(device)
                step_preds = []

                for step in range(horizon):
                    data = Data(
                        x=current_input,
                        edge_index_adj=edge_index_adj,
                        edge_index_od=edge_index_od,
                        edge_index_od_t=edge_index_od_t,
                    )
                    logits, reg = model(data)
                    probs = torch.sigmoid(logits)
                    mask = (probs >= class_threshold).float()
                    gated_reg = reg * mask
                    step_preds.append(gated_reg.cpu().numpy())

                    if step < horizon - 1:
                        current_input = torch.cat([
                            current_input[:, base_features:],
                            gated_reg,
                        ], dim=1)

                all_preds.append(np.concatenate(step_preds, axis=1))

        all_preds = np.array(all_preds)
        preds_denorm = scaler.inverse_transform(
            all_preds.reshape(-1, base_features)
        ).reshape(all_preds.shape)

        preds_h = preds_denorm.reshape(-1, num_nodes, horizon, base_features)
        targets_h = test_y_all.reshape(-1, num_nodes, horizon, base_features)

        arrival_preds = preds_h[:, :, horizon - 1, 0]
        arrival_targets = targets_h[:, :, horizon - 1, 0]
        departure_preds = preds_h[:, :, horizon - 1, 1]
        departure_targets = targets_h[:, :, horizon - 1, 1]

        arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
        dep_mae, dep_rmse, dep_r2 = test_error(departure_preds, departure_targets)

        results[horizon] = {
            'arr_mae': arr_mae,
            'arr_rmse': arr_rmse,
            'arr_r2': arr_r2,
            'dep_mae': dep_mae,
            'dep_rmse': dep_rmse,
            'dep_r2': dep_r2,
        }

        print(f"{horizon}-step ARRIVAL  -> MAE: {arr_mae:.4f}, RMSE: {arr_rmse:.4f}, R²: {arr_r2:.4f}")
        print(f"{horizon}-step DEPARTURE -> MAE: {dep_mae:.4f}, RMSE: {dep_rmse:.4f}, R²: {dep_r2:.4f}")

    return results


def save_multihorizon_results(results: Dict[int, Dict[str, float]], path: str):
    if not results:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'horizon', 'arr_mae', 'arr_rmse', 'arr_r2',
            'dep_mae', 'dep_rmse', 'dep_r2'
        ])
        for horizon in sorted(results.keys()):
            res = results[horizon]
            writer.writerow([
                horizon,
                res['arr_mae'], res['arr_rmse'], res['arr_r2'],
                res['dep_mae'], res['dep_rmse'], res['dep_r2'],
            ])


def print_multihorizon_table(results: Dict[int, Dict[str, float]]):
    if not results:
        return
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - Multi-Horizon Two-Stage Predictions")
    print("=" * 80)
    print(f"{'Horizon':<10} {'Delay Type':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 80)
    for horizon in sorted(results.keys()):
        res = results[horizon]
        print(f"{horizon}-step    {'Arrival':<15} {res['arr_mae']:<12.4f} {res['arr_rmse']:<12.4f} {res['arr_r2']:<12.4f}")
        print(f"{'':10} {'Departure':<15} {res['dep_mae']:<12.4f} {res['dep_rmse']:<12.4f} {res['dep_r2']:<12.4f}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Two-stage lightweight KAN-GAT delay predictor')
    parser.add_argument('--data_dir', type=str, default='cdata')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--reg_loss_weight', type=float, default=1.0)
    parser.add_argument('--delay_threshold', type=float, default=5.0, help='Minutes to tag as delayed')
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_scaled,
        val_scaled,
        test_scaled,
        train_raw,
        val_raw,
        test_raw,
        scaler,
        num_nodes,
    ) = load_flight_data(args.data_dir)

    in_channels = args.seq_len * train_scaled.shape[2]
    out_channels = args.horizon * train_scaled.shape[2]

    train_x, train_y_reg, train_y_cls = build_sequences(
        train_scaled,
        train_raw,
        args.seq_len,
        args.horizon,
        args.delay_threshold,
    )
    val_x, val_y_reg, val_y_cls = build_sequences(
        val_scaled,
        val_raw,
        args.seq_len,
        args.horizon,
        args.delay_threshold,
    )
    test_x, test_y_reg, test_y_cls = build_sequences(
        test_scaled,
        test_raw,
        args.seq_len,
        args.horizon,
        args.delay_threshold,
    )

    edge_index_adj = edge_index_adj.to(device)
    edge_index_od = edge_index_od.to(device)
    edge_index_od_t = edge_index_od_t.to(device)

    model = TwoStageDelayPredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cls_pos_rate = train_y_cls.mean().item()
    pos_weight = (1 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    reg_loss_fn = nn.MSELoss()

    history = []
    best_val_f1 = 0.0
    best_model_path = 'kan_gat_two_stage_best.pth'

    num_sequences = len(train_x)
    print(f'Training on {num_sequences} sequences, device={device}')

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(num_sequences)
        losses = []
        for idx in perm:
            x = train_x[idx].to(device)
            y_reg = train_y_reg[idx].to(device)
            y_cls = train_y_cls[idx].to(device)

            data = Data(
                x=x,
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )

            logits, reg = model(data)

            cls_loss = cls_loss_fn(logits, y_cls)
            mask = y_cls > 0.5
            if mask.sum() > 0:
                expanded_mask = mask.expand_as(y_reg)
                reg_loss = reg_loss_fn(reg[expanded_mask], y_reg[expanded_mask])
            else:
                reg_loss = torch.tensor(0.0, device=device)

            loss = cls_loss + args.reg_loss_weight * reg_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_metrics = evaluate_split(
            model,
            val_x,
            val_y_reg,
            val_y_cls,
            edge_index_adj,
            edge_index_od,
            edge_index_od_t,
            device,
            scaler,
            args.class_threshold,
        )
        history.append({
            'epoch': epoch + 1,
            'train_loss': float(np.mean(losses)),
            **val_metrics,
        })

        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.3f} | Val MAE: {val_metrics['mae']:.3f}"
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), best_model_path)
            print('  -> Saved new best model')

    print('Training complete. Evaluating on test split...')
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_probs, test_reg_preds = infer_sequences(
        model,
        test_x,
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        device,
    )

    test_metrics, test_preds_denorm, test_targets_denorm, test_mask = compute_metrics_and_denorm(
        test_probs,
        test_reg_preds,
        test_y_reg,
        test_y_cls,
        scaler,
        args.class_threshold,
    )

    num_features = train_scaled.shape[2]
    preds_h = test_preds_denorm.reshape(-1, num_nodes, args.horizon, num_features)
    targets_h = test_targets_denorm.reshape(-1, num_nodes, args.horizon, num_features)

    horizon_label = args.horizon
    arrival_preds = preds_h[:, :, horizon_label - 1, 0]
    arrival_targets = targets_h[:, :, horizon_label - 1, 0]
    dep_preds = preds_h[:, :, horizon_label - 1, 1]
    dep_targets = targets_h[:, :, horizon_label - 1, 1]

    arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
    dep_mae, dep_rmse, dep_r2 = test_error(dep_preds, dep_targets)

    test_metrics.update({
        'arrival_mae': arr_mae,
        'arrival_rmse': arr_rmse,
        'arrival_r2': arr_r2,
        'departure_mae': dep_mae,
        'departure_rmse': dep_rmse,
        'departure_r2': dep_r2,
    })

    print("\n" + "=" * 80)
    print("TEST RESULTS - Two-Stage KAN-GAT")
    print("=" * 80)
    print(f"{horizon_label}-step ahead ARRIVAL delay:")
    print(f"  MAE: {arr_mae:.4f} min, RMSE: {arr_rmse:.4f} min, R²: {arr_r2:.4f}")
    print(f"{horizon_label}-step ahead DEPARTURE delay:")
    print(f"  MAE: {dep_mae:.4f} min, RMSE: {dep_rmse:.4f} min, R²: {dep_r2:.4f}")
    print("-" * 80)
    print(
        f"Classification -> Precision: {test_metrics['precision']:.4f}, "
        f"Recall: {test_metrics['recall']:.4f}, F1: {test_metrics['f1']:.4f}, "
        f"Accuracy: {test_metrics['accuracy']:.4f}"
    )
    print(f"Regression (delayed nodes) -> MAE: {test_metrics['mae']:.4f}, RMSE: {test_metrics['rmse']:.4f}")
    print("=" * 80)

    print("\nSUMMARY TABLE - Two-Stage Flight Delay Predictions")
    print("=" * 80)
    print(f"{'Horizon':<10} {'Delay Type':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 80)
    print(f"{horizon_label}-step    {'Arrival':<15} {arr_mae:<12.4f} {arr_rmse:<12.4f} {arr_r2:<12.4f}")
    print(f"{'':10} {'Departure':<15} {dep_mae:<12.4f} {dep_rmse:<12.4f} {dep_r2:<12.4f}")
    print("=" * 80)

    with open('kan_gat_two_stage_history.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    with open('kan_gat_two_stage_test_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in test_metrics.items():
            writer.writerow([key, value])

    with open('kan_gat_two_stage_test_predictions.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'sequence_idx',
            'node_idx',
            'delay_probability',
            'delay_predicted',
            'arrival_true',
            'arrival_pred',
            'departure_true',
            'departure_pred',
        ])
        max_seq = min(10, preds_h.shape[0])
        for seq_idx in range(max_seq):
            for node_idx in range(num_nodes):
                writer.writerow([
                    seq_idx,
                    node_idx,
                    test_probs[seq_idx, node_idx, 0],
                    int(test_mask[seq_idx, node_idx, 0]),
                    targets_h[seq_idx, node_idx, horizon_label - 1, 0],
                    preds_h[seq_idx, node_idx, horizon_label - 1, 0],
                    targets_h[seq_idx, node_idx, horizon_label - 1, 1],
                    preds_h[seq_idx, node_idx, horizon_label - 1, 1],
                ])

    multihorizon_results = run_multihorizon_test(
        model,
        test_scaled,
        test_raw,
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        device,
        scaler,
        args.seq_len,
        horizons=(3, 6, 12),
        class_threshold=args.class_threshold,
    )
    print_multihorizon_table(multihorizon_results)
    save_multihorizon_results(multihorizon_results, 'kan_gat_two_stage_test_multihorizon.csv')


if __name__ == '__main__':
    main()
