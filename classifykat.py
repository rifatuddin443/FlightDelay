"""
Sequential two-stage KAN-GAT pipeline with decoupled training.

Stage 1: Train GAT encoder + KAN classifier to predict delay vs. no-delay.
Stage 2: Freeze classifier, train KAN regressor ONLY on predicted delayed nodes.

This architecture ensures the regressor sees only relevant samples during training,
improving regression quality on imbalanced delay data.
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


class SequentialTwoStagePredictor(nn.Module):
    """Classifier and regressor with independent training phases."""

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

    def forward_classifier(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Return embeddings and classification logits."""
        hidden = self.encoder(data)
        logits = self.classifier(hidden)
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stage 2: Regress on precomputed embeddings."""
        return self.regressor(hidden)

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass for inference."""
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


def train_stage1_classifier(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: float,
) -> Dict:
    """Stage 1: Train encoder + classifier only."""
    print("\n" + "=" * 80)
    print("STAGE 1: TRAINING CLASSIFIER (Delay Detection)")
    print("=" * 80)

    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    
    # Only optimize encoder + classifier parameters
    optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.classifier.parameters()),
        lr=lr,
        weight_decay=1e-4
    )
    cls_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    history = []
    best_val_f1 = 0.0
    best_state = None

    num_sequences = len(train_x)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_sequences)
        losses = []
        
        for idx in perm:
            x = train_x[idx].to(device)
            y_cls = train_y_cls[idx].to(device)

            data = Data(
                x=x,
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )

            _, logits = model.forward_classifier(data)
            loss = cls_loss_fn(logits, y_cls)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        model.eval()
        val_logits_list = []
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_index_adj,
                    edge_index_od=edge_index_od,
                    edge_index_od_t=edge_index_od_t,
                )
                _, logits = model.forward_classifier(data)
                val_logits_list.append(torch.sigmoid(logits).cpu().numpy())

        val_probs = np.array(val_logits_list)
        val_metrics = classification_metrics(
            val_probs.reshape(-1, 1),
            val_y_cls.cpu().numpy().reshape(-1, 1),
        )

        history.append({
            'epoch': epoch + 1,
            'stage': 1,
            'train_loss': float(np.mean(losses)),
            **val_metrics,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}"
        )

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_state = {
                'encoder': model.encoder.state_dict(),
                'classifier': model.classifier.state_dict(),
            }
            print('  -> New best F1 score')

    # Load best classifier
    model.encoder.load_state_dict(best_state['encoder'])
    model.classifier.load_state_dict(best_state['classifier'])
    
    print(f"\nStage 1 Complete. Best Val F1: {best_val_f1:.4f}")
    return history


def train_stage2_regressor(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int,
    lr: float,
    scaler: StandardScaler,
    class_threshold: float,
) -> Dict:
    """Stage 2: Freeze classifier, train regressor ONLY on delayed samples."""
    print("\n" + "=" * 80)
    print("STAGE 2: TRAINING REGRESSOR (Delay Magnitude Prediction)")
    print("=" * 80)

    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices

    # Freeze encoder and classifier
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False

    # Only optimize regressor
    optimizer = torch.optim.Adam(
        model.regressor.parameters(),
        lr=lr * 0.5,  # Lower learning rate for fine-tuning
        weight_decay=1e-4
    )
    reg_loss_fn = nn.MSELoss()

    history = []
    best_val_mae = float('inf')
    best_state = None

    # Pre-compute which training samples have delays
    num_sequences = len(train_x)
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_sequences)
        losses = []
        samples_used = 0
        
        for idx in perm:
            x = train_x[idx].to(device)
            y_reg = train_y_reg[idx].to(device)
            y_cls = train_y_cls[idx].to(device)

            # Skip samples with no delays
            if y_cls.sum() == 0:
                continue

            data = Data(
                x=x,
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )

            # Get classifier predictions (frozen)
            with torch.no_grad():
                hidden, logits = model.forward_classifier(data)
                pred_mask = torch.sigmoid(logits) >= class_threshold

            # Train regressor only on predicted delayed nodes
            if pred_mask.sum() == 0:
                continue

            reg_out = model.forward_regressor(hidden)
            
            # Compute loss only on nodes predicted as delayed
            expanded_mask = pred_mask.expand_as(y_reg)
            loss = reg_loss_fn(reg_out[expanded_mask], y_reg[expanded_mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            samples_used += 1

        if len(losses) == 0:
            print(f"Epoch {epoch + 1}/{epochs} | No delayed samples found, skipping")
            continue

        # Validation
        val_metrics = evaluate_stage2(
            model,
            val_x,
            val_y_reg,
            val_y_cls,
            edge_indices,
            device,
            scaler,
            class_threshold,
        )

        history.append({
            'epoch': epoch + 1,
            'stage': 2,
            'train_loss': float(np.mean(losses)),
            'samples_used': samples_used,
            **val_metrics,
        })

        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Samples: {samples_used} | Val MAE: {val_metrics['mae']:.4f} | "
            f"Val RMSE: {val_metrics['rmse']:.4f}"
        )

        if val_metrics['mae'] < best_val_mae:
            best_val_mae = val_metrics['mae']
            best_state = model.regressor.state_dict()
            print('  -> New best MAE')

    # Load best regressor
    model.regressor.load_state_dict(best_state)
    
    print(f"\nStage 2 Complete. Best Val MAE: {best_val_mae:.4f}")
    
    # Unfreeze for inference
    for param in model.encoder.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True

    return history


def evaluate_stage2(
    model: SequentialTwoStagePredictor,
    x_tensors: torch.Tensor,
    y_reg_tensors: torch.Tensor,
    y_cls_tensors: torch.Tensor,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    scaler: StandardScaler,
    class_threshold: float,
) -> Dict[str, float]:
    """Evaluate regressor on delayed samples only."""
    edge_index_adj, edge_index_od, edge_index_od_t = edge_indices
    
    model.eval()
    reg_preds_list = []
    pred_masks_list = []
    
    with torch.no_grad():
        for i in range(len(x_tensors)):
            data = Data(
                x=x_tensors[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            hidden, logits = model.forward_classifier(data)
            reg_out = model.forward_regressor(hidden)
            
            pred_mask = torch.sigmoid(logits) >= class_threshold
            reg_preds_list.append(reg_out.cpu().numpy())
            pred_masks_list.append(pred_mask.cpu().numpy())

    reg_preds = np.array(reg_preds_list)
    pred_masks = np.array(pred_masks_list)
    
    # Gate predictions
    gated_preds = reg_preds * pred_masks
    
    # Denormalize
    out_channels = gated_preds.shape[-1]
    preds_flat = gated_preds.reshape(-1, out_channels)
    targets_flat = y_reg_tensors.cpu().numpy().reshape(-1, out_channels)
    
    reg_preds_denorm = scaler.inverse_transform(preds_flat).reshape(gated_preds.shape)
    reg_targets_denorm = scaler.inverse_transform(targets_flat).reshape(gated_preds.shape)
    
    # Compute metrics only on truly delayed samples
    targets_cls = y_cls_tensors.cpu().numpy()
    delayed_mask = np.broadcast_to(targets_cls.astype(bool), reg_preds.shape)
    
    return regression_metrics(reg_preds_denorm, reg_targets_denorm, delayed_mask)


def main():
    parser = argparse.ArgumentParser(description='Sequential two-stage KAN-GAT delay predictor')
    parser.add_argument('--data_dir', type=str, default='cdata')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizon', type=int, default=1)
    parser.add_argument('--stage1_epochs', type=int, default=15, help='Epochs for classifier training')
    parser.add_argument('--stage2_epochs', type=int, default=10, help='Epochs for regressor training')
    parser.add_argument('--lr', type=float, default=0.003)
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
    edge_indices = (edge_index_adj, edge_index_od, edge_index_od_t)

    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32
    ).to(device)

    cls_pos_rate = train_y_cls.mean().item()
    pos_weight = (1 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)

    # Stage 1: Train classifier
    history_stage1 = train_stage1_classifier(
        model,
        train_x,
        train_y_cls,
        val_x,
        val_y_cls,
        edge_indices,
        device,
        args.stage1_epochs,
        args.lr,
        pos_weight,
    )

    # Stage 2: Train regressor on delayed samples only
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
        args.stage2_epochs,
        args.lr,
        scaler,
        args.class_threshold,
    )

    best_model_path = 'kan_gat_sequential_best.pth'
    torch.save({
        'encoder': model.encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        'regressor': model.regressor.state_dict(),
    }, best_model_path)

    checkpoint = torch.load(best_model_path, map_location=device)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    model.regressor.load_state_dict(checkpoint['regressor'])

    # Test evaluation
    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION")
    print("=" * 80)

    model.eval()
    test_probs_list = []
    test_reg_list = []
    
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            logits, reg = model(data)
            test_probs_list.append(torch.sigmoid(logits).cpu().numpy())
            test_reg_list.append(reg.cpu().numpy())

    test_probs = np.array(test_probs_list)
    test_reg_preds = np.array(test_reg_list)

    # Classification metrics
    test_cls_metrics = classification_metrics(
        test_probs.reshape(-1, 1),
        test_y_cls.cpu().numpy().reshape(-1, 1),
    )

    # Gated regression predictions
    test_mask = test_probs >= args.class_threshold
    gated_preds = test_reg_preds * test_mask

    # Denormalize
    num_features = train_scaled.shape[2]
    preds_flat = gated_preds.reshape(-1, out_channels)
    targets_flat = test_y_reg.cpu().numpy().reshape(-1, out_channels)
    
    test_preds_denorm = scaler.inverse_transform(preds_flat).reshape(gated_preds.shape)
    test_targets_denorm = scaler.inverse_transform(targets_flat).reshape(test_y_reg.shape)

    # Reshape for horizon analysis
    preds_h = test_preds_denorm.reshape(-1, num_nodes, args.horizon, num_features)
    targets_h = test_targets_denorm.reshape(-1, num_nodes, args.horizon, num_features)

    horizon_label = args.horizon
    arrival_preds = preds_h[:, :, horizon_label - 1, 0]
    arrival_targets = targets_h[:, :, horizon_label - 1, 0]
    dep_preds = preds_h[:, :, horizon_label - 1, 1]
    dep_targets = targets_h[:, :, horizon_label - 1, 1]

    arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
    dep_mae, dep_rmse, dep_r2 = test_error(dep_preds, dep_targets)

    # Regression metrics on delayed nodes only
    targets_cls = test_y_cls.cpu().numpy()
    delayed_mask = np.broadcast_to(targets_cls.astype(bool), test_reg_preds.shape)
    test_reg_metrics = regression_metrics(test_preds_denorm, test_targets_denorm, delayed_mask)

    print("\nCLASSIFICATION PERFORMANCE:")
    print(f"  Precision: {test_cls_metrics['precision']:.4f}")
    print(f"  Recall: {test_cls_metrics['recall']:.4f}")
    print(f"  F1 Score: {test_cls_metrics['f1']:.4f}")
    print(f"  Accuracy: {test_cls_metrics['accuracy']:.4f}")

    print("\nREGRESSION PERFORMANCE (on delayed nodes):")
    print(f"  MAE: {test_reg_metrics['mae']:.4f} min")
    print(f"  RMSE: {test_reg_metrics['rmse']:.4f} min")

    print(f"\n{horizon_label}-STEP AHEAD PREDICTIONS:")
    print(f"  Arrival Delay  -> MAE: {arr_mae:.4f} min, RMSE: {arr_rmse:.4f} min, R²: {arr_r2:.4f}")
    print(f"  Departure Delay -> MAE: {dep_mae:.4f} min, RMSE: {dep_rmse:.4f} min, R²: {dep_r2:.4f}")
    print("=" * 80)

    # Save results
    combined_history = history_stage1 + history_stage2
    if combined_history:
        all_fields = sorted({key for row in combined_history for key in row.keys()})
        normalized_rows = []
        for row in combined_history:
            normalized = {field: row.get(field, '') for field in all_fields}
            normalized_rows.append(normalized)

        with open('kan_gat_sequential_history.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(normalized_rows)

    test_metrics = {
        **test_cls_metrics,
        **test_reg_metrics,
        'arrival_mae': arr_mae,
        'arrival_rmse': arr_rmse,
        'arrival_r2': arr_r2,
        'departure_mae': dep_mae,
        'departure_rmse': dep_rmse,
        'departure_r2': dep_r2,
    }

    with open('kan_gat_sequential_test_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in test_metrics.items():
            writer.writerow([key, value])

    print("\nTraining complete. Results saved.")


if __name__ == '__main__':
    main()
