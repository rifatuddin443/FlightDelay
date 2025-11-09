import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
import pandas as pd
import os
import time
from datetime import datetime, timedelta
import copy
import csv
import util
from torch.utils.data import Dataset, DataLoader
from baseline_methods import test_error, StandardScaler

# === NEW: opacus for DP ===
from opacus import PrivacyEngine
from opacus.utils.uniform_sampler import UniformWithReplacementSampler

# Try to import Opacus' DP-compatible MultiheadAttention (if available)
OpacusMultiheadAttention = None
try:
    from opacus.layers import DPMultiheadAttention as OpacusMultiheadAttention
except Exception:
    OpacusMultiheadAttention = None

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Optimize PyTorch performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# ===== OPTIMIZED DSAFNet Model =====
class OptimizedSpatialAttentionStream(nn.Module):
    """Vectorized spatial attention - no loops"""
    def __init__(self, input_dim=6, hidden_dim=64, attention_class=None):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Use DP-compatible attention if provided
        if attention_class is not None:
            self.attention = attention_class(hidden_dim, 1, batch_first=False)
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=False)
    
    def forward(self, x):
        # x: [batch, airports, time_steps, features]
        batch_size, airports, time_steps, features = x.shape
        
        # Process all time steps at once by reshaping
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch_size * time_steps, airports, features)
        x_emb = self.embedding(x_reshaped)  # [batch*time, airports, hidden]
        x_emb = x_emb.transpose(0, 1)  # [airports, batch*time, hidden]
        
        out, _ = self.attention(x_emb, x_emb, x_emb)
        out = out.transpose(0, 1)  # [batch*time, airports, hidden]
        
        # Reshape back
        out = out.view(batch_size, time_steps, airports, -1)
        out = out.permute(0, 2, 1, 3)  # [batch, airports, time_steps, hidden]
        
        return out

class OptimizedTemporalAttentionStream(nn.Module):
    """Vectorized temporal attention - no loops"""
    def __init__(self, input_dim=6, hidden_dim=64, attention_class=None):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        if attention_class is not None:
            self.attention = attention_class(hidden_dim, 1, batch_first=False)
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=False)
    
    def forward(self, x):
        # x: [batch, airports, time_steps, features]
        batch_size, airports, time_steps, features = x.shape
        
        # Process all airports at once by reshaping
        x_reshaped = x.reshape(batch_size * airports, time_steps, features)
        x_emb = self.embedding(x_reshaped)  # [batch*airports, time, hidden]
        x_emb = x_emb.transpose(0, 1)  # [time, batch*airports, hidden]
        
        out, _ = self.attention(x_emb, x_emb, x_emb)
        out = out.transpose(0, 1)  # [batch*airports, time, hidden]
        
        # Reshape back
        out = out.view(batch_size, airports, time_steps, -1)
        
        return out

class OptimizedContextualCrossAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        # initialize lazily so we can inject DP-compatible attention later
        self.cross_attn = None
        self._hidden_dim = hidden_dim
        self._attention_class = None
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, spatial, temporal):
        # spatial, temporal: [batch, airports, hidden]
        if self.cross_attn is None:
            # default to native MultiheadAttention
            self.cross_attn = nn.MultiheadAttention(self._hidden_dim, 1, batch_first=True)
        fusion, _ = self.cross_attn(spatial, temporal, temporal)
        cat = torch.cat([spatial, fusion], dim=-1)
        gate_w = self.gate(cat)
        return gate_w * spatial + (1 - gate_w) * fusion

class SimpleGraphEncoder(nn.Module):
    def __init__(self, num_graphs=3):
        super().__init__()
        self.graph_weights = nn.Parameter(torch.ones(num_graphs) / num_graphs)
    
    def forward(self, features, adj_matrices):
        combined_adj = sum(w * adj for w, adj in zip(self.graph_weights, adj_matrices))
        return torch.matmul(combined_adj, features)

class OutputProjection(nn.Module):
    def __init__(self, hidden_dim=64, output_steps=12):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, output_steps * 2)
        self.output_steps = output_steps
    
    def forward(self, fused_features):
        out = self.projection(fused_features)
        batch_size, airports, _ = out.shape
        out = out.view(batch_size, airports, self.output_steps, 2)
        return out

class OptimizedDSAFNet(nn.Module):
    """Optimized DSAFNet with vectorized operations"""
    def __init__(self, input_dim=6, hidden_dim=64, output_steps=12, num_graphs=3, attention_class=None):
        super().__init__()
        self.spatial_stream = OptimizedSpatialAttentionStream(input_dim, hidden_dim, attention_class=attention_class)
        self.temporal_stream = OptimizedTemporalAttentionStream(input_dim, hidden_dim, attention_class=attention_class)
        self.cross_fusion = OptimizedContextualCrossAttention(hidden_dim)
        # if a DP-compatible attention class is provided, use it for cross attention
        if attention_class is not None:
            try:
                self.cross_fusion.cross_attn = attention_class(hidden_dim, 1, batch_first=True)
                self.cross_fusion._attention_class = attention_class
            except Exception:
                # fall back silently to native attention if creation fails
                self.cross_fusion.cross_attn = None
        self.graph_encoder = SimpleGraphEncoder(num_graphs)
        self.output_proj = OutputProjection(hidden_dim, output_steps)
    
    def forward(self, x, ti=None, supports=None, to=None, w=None):
        # Handle input shape transformation
        if x.dim() == 4:
            # x is (batch_size, in_channels, num_nodes, in_len)
            # Transform to (batch_size, num_nodes, in_len, in_channels)
            x = x.permute(0, 2, 3, 1)
        
        batch_size, airports, time_steps, features = x.shape
        adj_matrices = supports if supports is not None else []
        
        # Vectorized spatial and temporal processing - NO LOOPS!
        spatial_features = self.spatial_stream(x)  # [batch, airports, time, hidden]
        temporal_features = self.temporal_stream(x)  # [batch, airports, time, hidden]
        
        # Average across time dimension
        spatial_avg = spatial_features.mean(dim=2)  # [batch, airports, hidden]
        temporal_avg = temporal_features.mean(dim=2)  # [batch, airports, hidden]
        
        # Cross attention fusion
        fused = self.cross_fusion(spatial_avg, temporal_avg)
        
        # Graph encoding
        if adj_matrices:
            graph_enhanced = self.graph_encoder(fused, adj_matrices)
        else:
            graph_enhanced = fused
        
        # Output projection
        out = self.output_proj(graph_enhanced)
        
        # Transform to match expected output format
        out = out.permute(0, 3, 1, 2)  # [batch, 2, airports, output_steps]
        
        return out

# ===== Optimized Data Loader =====
class OptimizedTrainWindowDataset(Dataset):
    def __init__(self, data, weather, period, in_len, out_len, indices):
        self.data = data
        self.weather = weather
        self.period = period
        self.in_len = in_len
        self.out_len = out_len
        self.indices = indices
        
        print(f"Dataset initialized with {len(indices)} training samples.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        x = self.data[:, s:s + self.in_len, :]
        y = self.data[:, s + self.in_len:s + self.in_len + self.out_len, :]
        w = self.weather[:, s:s + self.in_len]

        ti = (np.arange(s, s + self.in_len) % self.period) * np.ones([1, self.in_len]) / (self.period - 1)
        to = (np.arange(s + self.in_len, s + self.in_len + self.out_len) % self.period) * np.ones([1, self.out_len]) / (self.period - 1)

        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
        y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)
        w = torch.tensor(w, dtype=torch.long)
        ti = torch.tensor(ti, dtype=torch.float32)
        to = torch.tensor(to, dtype=torch.float32)
        
        return x, y, ti, to, w

# ===== Training Time Estimator =====
class TrainingTimeEstimator:
    def __init__(self, estimate_epochs=3, total_epochs=50):
        self.estimate_epochs = estimate_epochs
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.start_time = None
        self.training_started = False
        
    def start_training(self):
        self.start_time = time.time()
        self.training_started = True
        print(f"🕐 Training started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Will estimate total time after {self.estimate_epochs} epochs")
        print("-" * 60)
    
    def record_epoch_time(self, epoch, epoch_start_time):
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        
        elapsed_total = time.time() - self.start_time
        print(f"⏱️  Epoch {epoch+1}: {epoch_time:.2f}s | Total elapsed: {elapsed_total:.2f}s")
        
        if len(self.epoch_times) == self.estimate_epochs:
            self._provide_estimate()
        elif len(self.epoch_times) > self.estimate_epochs and (epoch + 1) % 10 == 0:
            self._update_estimate(epoch + 1)
    
    def _provide_estimate(self):
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - len(self.epoch_times)
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        print("\n" + "="*60)
        print("📈 TRAINING TIME ESTIMATION")
        print("="*60)
        print(f"✅ Completed {len(self.epoch_times)} epochs for estimation")
        print(f"⏱️  Average time per epoch: {avg_epoch_time:.2f} seconds")
        print(f"📊 Remaining epochs: {remaining_epochs}")
        print(f"🕐 Estimated remaining time: {self._format_time(estimated_remaining_time)}")
        
        total_estimated_time = avg_epoch_time * self.total_epochs
        print(f"🎯 Estimated total training time: {self._format_time(total_estimated_time)}")
        
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        print(f"🏁 Estimated completion: {completion_time.strftime('%H:%M:%S')}")
        print("="*60 + "\n")
    
    def _update_estimate(self, current_epoch):
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        print(f"\n📊 Updated estimate at epoch {current_epoch}:")
        print(f"   Average epoch time: {avg_epoch_time:.2f}s")
        print(f"   Estimated remaining: {self._format_time(estimated_remaining_time)}")
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        print(f"   Estimated completion: {completion_time.strftime('%H:%M:%S')}\n")
    
    def _format_time(self, seconds):
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    def finish_training(self):
        if not self.training_started:
            return None
            
        total_time = time.time() - self.start_time
        avg_epoch_time = np.mean(self.epoch_times) if self.epoch_times else 0
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETED!")
        print("="*60)
        print(f"✅ Total epochs completed: {len(self.epoch_times)}")
        print(f"⏱️  Total training time: {self._format_time(total_time)}")
        print(f"📊 Average time per epoch: {avg_epoch_time:.2f} seconds")
        print(f"🚀 Training efficiency: {len(self.epoch_times)/total_time*60:.1f} epochs/minute")
        
        if len(self.epoch_times) > 1:
            fastest = min(self.epoch_times)
            slowest = max(self.epoch_times)
            print(f"⚡ Fastest epoch: {fastest:.2f}s")
            print(f"🐌 Slowest epoch: {slowest:.2f}s")
        
        print(f"🏁 Training finished at: {datetime.now().strftime('%H:%M:%S')}")
        print("="*60)
        
        return {
            'total_time': total_time,
            'avg_epoch_time': avg_epoch_time,
            'total_epochs': len(self.epoch_times),
            'epoch_times': self.epoch_times.copy()
        }


class EarlyStopping:
    """Simple early stopping on validation loss."""
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = max(1, patience)
        self.min_delta = min_delta
        self.best = float('inf')
        self.counter = 0
        self.triggered = False

    def step(self, value):
        if value < self.best - self.min_delta:
            self.best = value
            self.counter = 0
            return False
        self.counter += 1
        if self.counter > self.patience:
            self.triggered = True
            return True
        return False

# ===== Training Script =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--data', type=str, default='US', help='data type')
    parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/val/test ratio', type=float)
    parser.add_argument('--in_channels', type=int, default=2, help='input variable')
    parser.add_argument('--out_channels', type=int, default=2, help='output variable')
    parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
    parser.add_argument('--support_len', type=int, default=3, help='number of spatial adjacency matrix')
    parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--in_len', type=int, default=12, help='input time series length')
    parser.add_argument('--out_len', type=int, default=12, help='output time series length')
    parser.add_argument('--batch', type=int, default=64, help='training batch size')
    parser.add_argument('--episode', type=int, default=10, help='training episodes')
    parser.add_argument('--period', type=int, default=36, help='periodic for temporal embedding')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loader workers')
    
    # DP parameters
    parser.add_argument('--dp', default=True, action='store_true', help='enable differential privacy')
    parser.add_argument('--target_epsilon', type=float, default=4.0, help='target epsilon')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='delta for DP')
    parser.add_argument('--noise_multiplier', type=float, default=1.5, help='noise multiplier')
    parser.add_argument('--max_grad_norm', type=float, default=1.5, help='gradient clipping norm')
    parser.add_argument('--early_stop_patience', type=int, default=5, help='early stopping patience (epochs)')
    parser.add_argument('--lr_patience', type=int, default=3, help='ReduceLROnPlateau patience')
    parser.add_argument('--lr_factor', type=float, default=0.5, help='ReduceLROnPlateau factor')
    parser.add_argument('--min_lr', type=float, default=1e-7, help='minimum learning rate for scheduler')
    
    parser.add_argument("--output_dir", type=str, default="./results", help="output directory")
    args = parser.parse_args()

    # Device setup
    print("="*60)
    print("🚀 OPTIMIZED DSAFNET TRAINING")
    print("="*60)
    
    if torch.cuda.is_available():
        device = torch.device(args.device if args.device.startswith('cuda') else 'cuda:0')
        print(f"✅ Using GPU: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU")
    
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading data...")
    start_time = time.time()
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    print(f"✅ Data loaded in {time.time() - start_time:.2f}s")

    # Create optimized model
    print("\n" + "="*60)
    print("🤖 CREATING OPTIMIZED MODEL")
    print("="*60)
    # If DP requested but Opacus does not provide a DP-compatible MultiheadAttention,
    # warn and disable DP so training can continue. To enable DP with attention, install
    # a newer opacus that exposes DPMultiheadAttention (or implement a compatible layer).
    if args.dp and OpacusMultiheadAttention is None:
        print("⚠️  Requested DP training but Opacus DPMultiheadAttention is not available.")
        print("   Disabling --dp for this run. To enable DP with attention, upgrade opacus or provide a compatible DPMultiheadAttention.")
        args.dp = False

    attention_cls = OpacusMultiheadAttention if args.dp else None

    model = OptimizedDSAFNet(
        input_dim=args.in_channels,
        hidden_dim=args.hidden_dim,
        output_steps=args.out_len,
        num_graphs=args.support_len,
        attention_class=attention_cls,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model parameters: {num_params:,}")
    print(f"✨ Using VECTORIZED forward pass (no loops!)")
    
    model = model.to(device)
    print(f"✅ Model on device: {device}")
    print("="*60)

    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        min_lr=args.min_lr,
    )

    # Data preprocessing
    print("Preprocessing data...")
    start_time = time.time()
    scaler_obj = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                                training_data[~np.isnan(training_data)].std())
    training_data = scaler_obj.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    print(f"✅ Preprocessing done in {time.time() - start_time:.2f}s")

    # Create dataset
    train_indices = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    dataset = OptimizedTrainWindowDataset(
        data=training_data,
        weather=training_w,
        period=args.period,
        in_len=args.in_len,
        out_len=args.out_len,
        indices=train_indices
    )

    if args.dp:
        sample_rate = args.batch / len(dataset)
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataset),
            sample_rate=sample_rate,
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(dataset, batch_sampler=batch_sampler,
                                num_workers=args.num_workers, pin_memory=True)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True,
                                drop_last=True, num_workers=args.num_workers,
                                pin_memory=True, persistent_workers=True)

    # DP setup
    if args.dp:
        privacy_engine = PrivacyEngine()
        try:
            if args.target_epsilon > 0:
                model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    target_epsilon=args.target_epsilon,
                    target_delta=args.target_delta,
                    epochs=args.episode,
                    max_grad_norm=args.max_grad_norm,
                )
                csv_filename = f'optimized_dsafnet_ε={args.target_epsilon}_max={args.max_grad_norm}.csv'
            else:
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=args.noise_multiplier,
                    max_grad_norm=args.max_grad_norm,
                )
                csv_filename = f'optimized_dsafnet_noise={args.noise_multiplier}_max={args.max_grad_norm}.csv'
        except Exception as e:
            print(f"ERROR during DP setup: {e}")
            raise
    else:
        privacy_engine = None
        csv_filename = 'optimized_dsafnet_training_no_dp.csv'

    # Validation data
    print("Preparing validation data...")
    start_time = time.time()
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)
    print(f"✅ Validation data ready in {time.time() - start_time:.2f}s")

    print("\n" + "="*60)
    print("🚀 STARTING TRAINING")
    print("="*60)
    
    if device.type == 'cuda':
        print(f"💾 Initial GPU Memory: {torch.cuda.memory_allocated(device)/1024**2:.1f} MB")
    
    # Initialize time estimator
    time_estimator = TrainingTimeEstimator(estimate_epochs=3, total_epochs=args.episode)
    time_estimator.start_training()

    # CSV logging
    csv_headers = ['epoch', 'training_loss', 'validation_loss', 'validation_mae', 'validation_r2', 'validation_rmse', 'epsilon', 'learning_rate', 'epoch_time']
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)

    training_metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_mae': [],
        'val_r2': [],
        'val_rmse': [],
        'learning_rate': [],
        'epoch_time': []
    }
    early_stopper = EarlyStopping(patience=args.early_stop_patience)
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    for ep in range(1, 1 + args.episode):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_start_time = time.time()

        epoch_training_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            trainx, trainy, trainti, trainto, trainw = batch

            # Move to device
            trainx = trainx.to(device, non_blocking=True)
            trainy = trainy.to(device, non_blocking=True)
            trainti = trainti.to(device, non_blocking=True)
            trainto = trainto.to(device, non_blocking=True)
            trainw = trainw.to(device, non_blocking=True)

            if trainx.dim() == 3:
                trainx = trainx.unsqueeze(0)
                trainy = trainy.unsqueeze(0)
                trainti = trainti.unsqueeze(0)
                trainto = trainto.unsqueeze(0)
                trainw = trainw.unsqueeze(0)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            # Forward pass
            output = model(trainx, trainti, supports, trainto, trainw)
            if output.shape != trainy.shape:
                if output.shape[-1] == 1 and trainy.shape[-1] > 1:
                    trainy = trainy[..., -1:]
            loss = util.masked_rmse(output, trainy, 0.0)

            # Backward pass
            loss.backward()

            # Optimizer step
            if not args.dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            optimizer.step()

            epoch_training_loss += loss.item()
            num_batches += 1

        # Validation
        model.eval()
        outputs = []
        validation_loss = 0.0
        val_loss_count = 0

        with torch.no_grad():
            for i in range(len(val_index)):
                testx = np.expand_dims(val_data[:, val_index[i]: val_index[i] + args.in_len, :], axis=0)
                testx = scaler_obj.transform(testx)
                testw = np.expand_dims(val_w[:, val_index[i]: val_index[i] + args.in_len], axis=0)
                testw = torch.LongTensor(testw).to(device)
                testx[np.isnan(testx)] = 0
                testti = (np.arange(int(training_data.shape[1]) + val_index[i],
                                    int(training_data.shape[1]) + val_index[i] + args.in_len) % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
                testto = (np.arange(int(training_data.shape[1]) + val_index[i] + args.in_len,
                                    int(training_data.shape[1]) + val_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len]) / (args.period - 1)

                testx = torch.Tensor(testx).to(device).permute(0, 3, 1, 2)
                testti = torch.Tensor(testti).to(device)
                testto = torch.Tensor(testto).to(device)

                out = model(testx, testti, supports, testto, testw)

                testy = np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0)
                testy = scaler_obj.transform(testy)
                testy[np.isnan(testy)] = 0
                testy = torch.Tensor(testy).to(device).permute(0, 3, 1, 2)

                if out.shape != testy.shape:
                    if out.shape[-1] == 1 and testy.shape[-1] > 1:
                        testy = testy[..., -1:]

                val_loss = util.masked_rmse(out, testy, 0.0)
                validation_loss += val_loss.item()
                val_loss_count += 1

                out = out.permute(0, 2, 3, 1)
                out = out.detach().cpu().numpy()
                out = scaler_obj.inverse_transform(out)
                outputs.append(out)

        yhat = np.concatenate(outputs)
        amae, ar2, armse = [], [], []
        for i in range(12):
            metrics = test_error(yhat[:, :, i, :], label[:, :, i, :])
            amae.append(metrics[0])
            ar2.append(metrics[2])
            armse.append(metrics[1])

        mean_mae = float(np.mean(amae))
        mean_r2 = float(np.mean(ar2))
        mean_rmse = float(np.mean(armse))

        avg_training_loss = epoch_training_loss / num_batches if num_batches > 0 else 0.0
        avg_validation_loss = validation_loss / val_loss_count if val_loss_count > 0 else 0.0

        current_epsilon = None
        if args.dp and privacy_engine:
            try:
                current_epsilon = privacy_engine.get_epsilon(args.target_delta)
            except Exception:
                current_epsilon = 'N/A'
        else:
            current_epsilon = 'No DP'

        if device.type == 'cuda':
            torch.cuda.synchronize()
        epoch_time = time.time() - epoch_start_time

        training_metrics['epoch'].append(ep)
        training_metrics['train_loss'].append(avg_training_loss)
        training_metrics['val_loss'].append(avg_validation_loss)
        training_metrics['val_mae'].append(mean_mae)
        training_metrics['val_r2'].append(mean_r2)
        training_metrics['val_rmse'].append(mean_rmse)
        training_metrics['learning_rate'].append(optimizer.param_groups[0]['lr'])
        training_metrics['epoch_time'].append(epoch_time)

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                ep,
                avg_training_loss,
                avg_validation_loss,
                mean_mae,
                mean_r2,
                mean_rmse,
                current_epsilon,
                optimizer.param_groups[0]['lr'],
                epoch_time
            ])

        time_estimator.record_epoch_time(ep-1, epoch_start_time)
        scheduler.step(avg_validation_loss)

        if args.dp:
            try:
                eps = privacy_engine.get_epsilon(args.target_delta)
                print(f"Epoch {ep:03d} | Train {avg_training_loss:.4f} | Val {avg_validation_loss:.4f} | MAE {mean_mae:.4f} R2 {mean_r2:.4f} | ε={eps:.2f}")
            except Exception:
                print(f"Epoch {ep:03d} | Train {avg_training_loss:.4f} | Val {avg_validation_loss:.4f} | MAE {mean_mae:.4f} R2 {mean_r2:.4f}")
        else:
            print(f"Epoch {ep:03d} | Train {avg_training_loss:.4f} | Val {avg_validation_loss:.4f} | MAE {mean_mae:.4f} R2 {mean_r2:.4f}")

        if avg_validation_loss < best_val_loss - 1e-6:
            best_val_loss = avg_validation_loss
            best_model = copy.deepcopy(model.state_dict())

        if early_stopper.step(avg_validation_loss):
            print(f"⏹️  Early stopping triggered at epoch {ep}. Best val loss: {early_stopper.best:.4f}")
            break

    final_stats = time_estimator.finish_training()
    
    model.load_state_dict(best_model)
    
    # Save metrics
    df = pd.DataFrame(training_metrics)
    excel_path = os.path.join(args.output_dir, 'optimized_dsafnet_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"📊 Metrics saved to: {excel_path}")

    if final_stats:
        time_stats = pd.DataFrame([{
            'Total_Time_Seconds': final_stats['total_time'],
            'Total_Time_Minutes': final_stats['total_time'] / 60,
            'Avg_Epoch_Time': final_stats['avg_epoch_time'],
            'Total_Epochs': final_stats['total_epochs'],
            'Epochs_Per_Minute': final_stats['total_epochs'] / (final_stats['total_time'] / 60)
        }])
        time_stats_path = os.path.join(args.output_dir, 'optimized_dsafnet_time_stats.xlsx')
        time_stats.to_excel(time_stats_path, index=False)
        print(f"⏱️  Time stats saved to: {time_stats_path}")

    model_path = os.path.join(args.output_dir, "optimized_dsafnet_" + args.data + ".pth")
    torch.save(model.state_dict(), model_path)
    print(f"💾 Model saved to: {model_path}")
    print("✅ Training complete!")

if __name__ == "__main__":
    main()
