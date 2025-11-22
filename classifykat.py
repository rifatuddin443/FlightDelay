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
from typing import Dict, List, Optional, Tuple

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
        std = np.array(std)
        self.std = np.where(std == 0, 1.0, std)

    def transform(self, data: np.ndarray) -> np.ndarray:
        return (data - self.mean) / self.std

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return data * self.std + self.mean


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    
    def __init__(self, patience: int = 5, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
            
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # max
            improved = score > (self.best_score + self.min_delta)
            
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⏹️  Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"   Best score: {self.best_score:.4f} at epoch {self.best_epoch}")
                return True
        return False


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
    weather_file: str = 'weather_cn.npy',
    period_hours: int = 24,
    data_source: str = 'cdata',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, int]:
    if data_source == 'udata':
        od_file = 'od_pair.npy'
        adj_file = 'adj_mx.npy'
        delay_file = 'udelay.npy'
    else:
        od_file = 'od_mx.npy'
        adj_file = 'dist_mx.npy'
        delay_file = 'delay.npy'

    od_mx = np.load(os.path.join(data_dir, od_file))
    adj_mx = np.load(os.path.join(data_dir, adj_file))
    delay_data = np.load(os.path.join(data_dir, delay_file))

    weather_path = os.path.join(data_dir, weather_file)
    if not os.path.exists(weather_path):
        raise FileNotFoundError(f"Weather file not found: {weather_path}")
    weather_data = np.load(weather_path)
    if weather_data.ndim == 2:
        weather_data = weather_data[..., np.newaxis]

    if weather_data.shape[0] != delay_data.shape[0] or weather_data.shape[1] != delay_data.shape[1]:
        raise ValueError('Weather data shape must align with delay data (num_nodes, timesteps, features).')

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

    # Delay scaler (targets)
    scaler = StandardScaler(
        mean=np.nanmean(train_raw),
        std=np.nanstd(train_raw),
    )
    delay_scaled = scaler.transform(delay_data)
    delay_scaled = np.nan_to_num(delay_scaled)

    # Weather scaler (inputs only)
    weather_train = weather_data[:, :train_end, :]
    weather_mean = np.nanmean(weather_train, axis=(0, 1))
    weather_std = np.nanstd(weather_train, axis=(0, 1))
    weather_scaler = StandardScaler(weather_mean, weather_std)
    weather_scaled = weather_scaler.transform(weather_data)
    weather_scaled = np.nan_to_num(weather_scaled)

    # Temporal embeddings (sin/cos encoding of 24h cycle)
    time_indices = np.arange(total_steps)
    radians = 2 * np.pi * ((time_indices % period_hours) / period_hours)
    time_embed = np.stack([np.sin(radians), np.cos(radians)], axis=-1)
    time_embed = np.broadcast_to(time_embed, (num_nodes, total_steps, 2))

    # Split datasets
    train_delay = delay_scaled[:, :train_end, :]
    val_delay = delay_scaled[:, train_end:val_end, :]
    test_delay = delay_scaled[:, val_end:, :]

    train_inputs = np.concatenate([
        train_delay,
        weather_scaled[:, :train_end, :],
        time_embed[:, :train_end, :],
    ], axis=2)
    val_inputs = np.concatenate([
        val_delay,
        weather_scaled[:, train_end:val_end, :],
        time_embed[:, train_end:val_end, :],
    ], axis=2)
    test_inputs = np.concatenate([
        test_delay,
        weather_scaled[:, val_end:, :],
        time_embed[:, val_end:, :],
    ], axis=2)

    return (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_inputs,
        val_inputs,
        test_inputs,
        train_delay,
        val_delay,
        test_delay,
        train_raw,
        val_raw,
        test_raw,
        scaler,
        num_nodes,
    )


def build_sequences(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    x_list, y_reg_list, y_cls_list = [], [], []

    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))
    if not horizon_ids:
        raise ValueError("At least one future horizon is required to build sequences.")

    for t in range(max_idx):
        x_seq = input_data[:, t:t + seq_len, :].reshape(num_nodes, -1)
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)

        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :])
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
    batch_size: int = 16,
    patience: int = 5,
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
    early_stopping = EarlyStopping(patience=patience, mode='max')

    num_sequences = len(train_x)
    num_batches = (num_sequences + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(num_sequences)
        losses = []
        
        # Process in mini-batches for efficiency
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_sequences)
            batch_indices = perm[start_idx:end_idx]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            
            for idx in batch_indices:
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
                batch_loss += loss
                
            batch_loss = batch_loss / len(batch_indices)
            batch_loss.backward()
            optimizer.step()
            losses.append(batch_loss.item())

        # Validation - sample subset for speed
        model.eval()
        val_sample_size = len(val_x)  # Validate on subset
        val_indices = torch.randperm(len(val_x))[:val_sample_size]
        val_logits_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for i in val_indices:
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_index_adj,
                    edge_index_od=edge_index_od,
                    edge_index_od_t=edge_index_od_t,
                )
                _, logits = model.forward_classifier(data)
                val_logits_list.append(torch.sigmoid(logits).cpu().numpy())
                val_targets_list.append(val_y_cls[i].numpy())

        val_probs = np.array(val_logits_list)
        val_targets = np.array(val_targets_list)
        val_metrics = classification_metrics(
            val_probs.reshape(-1, 1),
            val_targets.reshape(-1, 1),
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
        
        # Early stopping check
        if early_stopping(val_metrics['f1'], epoch + 1):
            break

    if best_state is None:
        best_state = {
            'encoder': model.encoder.state_dict(),
            'classifier': model.classifier.state_dict(),
        }

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
    batch_size: int = 16,
    patience: int = 5,
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
    early_stopping = EarlyStopping(patience=patience, mode='min')

    # Pre-filter indices with delays for efficiency
    delayed_indices = [i for i in range(len(train_x)) if train_y_cls[i].sum() > 0]
    print(f"Training on {len(delayed_indices)} samples with delays (out of {len(train_x)} total)")
    
    for epoch in range(epochs):
        model.train()
        perm = torch.tensor(delayed_indices)[torch.randperm(len(delayed_indices))]
        losses = []
        samples_used = 0
        num_batches = (len(perm) + batch_size - 1) // batch_size
        
        # Process in mini-batches
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(perm))
            batch_indices = perm[start_idx:end_idx]
            
            optimizer.zero_grad()
            batch_loss = 0.0
            batch_count = 0
            
            for idx in batch_indices:
                x = train_x[idx].to(device)
                y_reg = train_y_reg[idx].to(device)
                y_cls = train_y_cls[idx].to(device)

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

                if pred_mask.sum() == 0:
                    gt_mask = y_cls >= class_threshold
                    if gt_mask.sum() == 0:
                        continue
                    pred_mask = gt_mask

                reg_out = model.forward_regressor(hidden)
                
                # Compute loss only on nodes predicted as delayed
                expanded_mask = pred_mask.expand_as(y_reg)
                loss = reg_loss_fn(reg_out[expanded_mask], y_reg[expanded_mask])
                batch_loss += loss
                batch_count += 1
            
            if batch_count > 0:
                batch_loss = batch_loss / batch_count
                batch_loss.backward()
                optimizer.step()
                losses.append(batch_loss.item())
                samples_used += batch_count

        if len(losses) == 0:
            print(f"Epoch {epoch + 1}/{epochs} | No delayed samples found, skipping")
            continue

        # Validation - sample subset for speed
        val_sample_size = min(len(val_x), 100)
        val_indices = torch.randperm(len(val_x))[:val_sample_size]
        val_x_sample = val_x[val_indices]
        val_y_reg_sample = val_y_reg[val_indices]
        val_y_cls_sample = val_y_cls[val_indices]
        
        val_metrics = evaluate_stage2(
            model,
            val_x_sample,
            val_y_reg_sample,
            val_y_cls_sample,
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
        
        # Early stopping check
        if early_stopping(val_metrics['mae'], epoch + 1):
            break

    if best_state is not None:
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
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'],
                        help='Data source folder: cdata (China) or udata (USA)')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 6, 12],
                        help='List of step-ahead horizons to train/evaluate (e.g., 3 6 12)')
    parser.add_argument('--stage1_epochs', type=int, default=15, help='Epochs for classifier training')
    parser.add_argument('--stage2_epochs', type=int, default=15, help='Epochs for regressor training')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--delay_threshold', type=float, default=5.0, help='Minutes to tag as delayed')
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--model_path', type=str, default='kan_gat_sequential_best.pth',
                        help='Destination path for the trained checkpoint (.pth)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'

    set_seed(args.seed)
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

    data_dir = args.data_source

    horizons = sorted({h for h in args.horizons if h > 0})
    if not horizons:
        raise ValueError('Please provide at least one positive prediction horizon (e.g., --horizons 3 6 12).')
    max_horizon = max(horizons)

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
        data_dir,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

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
        batch_size=args.batch_size,
        patience=args.patience,
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
        batch_size=args.batch_size,
        patience=args.patience,
    )

    best_model_path = args.model_path
    model_dir = os.path.dirname(best_model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
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
    num_forecast_steps = len(horizons)
    num_features = delay_dim
    preds_flat = gated_preds.reshape(-1, num_forecast_steps * num_features)
    targets_flat = test_y_reg.cpu().numpy().reshape(-1, num_forecast_steps * num_features)
    
    test_preds_denorm = scaler.inverse_transform(preds_flat).reshape(gated_preds.shape)
    test_targets_denorm = scaler.inverse_transform(targets_flat).reshape(test_y_reg.shape)

    # Reshape for horizon analysis
    preds_h = test_preds_denorm.reshape(-1, num_nodes, num_forecast_steps, num_features)
    targets_h = test_targets_denorm.reshape(-1, num_nodes, num_forecast_steps, num_features)

    per_horizon_metrics = {}
    for idx, horizon in enumerate(horizons):
        arrival_preds = preds_h[:, :, idx, 0]
        arrival_targets = targets_h[:, :, idx, 0]
        dep_preds = preds_h[:, :, idx, 1]
        dep_targets = targets_h[:, :, idx, 1]

        arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
        dep_mae, dep_rmse, dep_r2 = test_error(dep_preds, dep_targets)

        per_horizon_metrics[horizon] = {
            'arrival_mae': arr_mae,
            'arrival_rmse': arr_rmse,
            'arrival_r2': arr_r2,
            'departure_mae': dep_mae,
            'departure_rmse': dep_rmse,
            'departure_r2': dep_r2,
        }

        print(f"\n{horizon}-STEP AHEAD PREDICTIONS:")
        print(f"  Arrival Delay  -> MAE: {arr_mae:.4f} min, RMSE: {arr_rmse:.4f} min, R²: {arr_r2:.4f}")
        print(f"  Departure Delay -> MAE: {dep_mae:.4f} min, RMSE: {dep_rmse:.4f} min, R²: {dep_r2:.4f}")

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

    with open('kan_gat_sequential_test_summary.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for key, value in {
            'classification_precision': test_cls_metrics['precision'],
            'classification_recall': test_cls_metrics['recall'],
            'classification_f1': test_cls_metrics['f1'],
            'classification_accuracy': test_cls_metrics['accuracy'],
            'regression_mae_delayed': test_reg_metrics['mae'],
            'regression_rmse_delayed': test_reg_metrics['rmse'],
        }.items():
            writer.writerow([key, value])

        for horizon in horizons:
            metrics = per_horizon_metrics[horizon]
            for metric_name, metric_value in metrics.items():
                writer.writerow([f"{metric_name}_h{horizon}", metric_value])

    print("\nTraining complete. Results saved.")


if __name__ == '__main__':
    main()
