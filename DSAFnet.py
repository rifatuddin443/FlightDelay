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

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Optimize PyTorch performance
torch.backends.cudnn.benchmark = True  # Optimize CUDA kernels
torch.backends.cudnn.deterministic = False  # Allow non-deterministic operations for speed

# ===== DSAFNet Model Definition =====
class SpatialAttentionStream(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 1)
    def forward(self, x):  # x: [batch, airports, features]
        x_emb = self.embedding(x).transpose(0, 1)
        out, _ = self.attention(x_emb, x_emb, x_emb)
        return out.transpose(0, 1)

class TemporalAttentionStream(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 1)
    def forward(self, x):  # x: [batch, time_steps, features]
        x_emb = self.embedding(x).transpose(0, 1)
        out, _ = self.attention(x_emb, x_emb, x_emb)
        return out.transpose(0, 1)

class ContextualCrossAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 1)
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
    def forward(self, spatial, temporal):
        fusion, _ = self.cross_attn(spatial.transpose(0, 1), temporal.transpose(0, 1), temporal.transpose(0, 1))
        fusion = fusion.transpose(0, 1)
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
        # fused_features: [batch, airports, hidden_dim]
        out = self.projection(fused_features)  # [batch, airports, output_steps * 2]
        batch_size, airports, _ = out.shape
        out = out.view(batch_size, airports, self.output_steps, 2)  # [batch, airports, output_steps, 2]
        return out

class DSAFNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_steps=12, num_graphs=3):
        super().__init__()
        self.spatial_stream = SpatialAttentionStream(input_dim, hidden_dim)
        self.temporal_stream = TemporalAttentionStream(input_dim, hidden_dim)
        self.cross_fusion = ContextualCrossAttention(hidden_dim)
        self.graph_encoder = SimpleGraphEncoder(num_graphs)
        self.output_proj = OutputProjection(hidden_dim, output_steps)
    def forward(self, x, ti=None, supports=None, to=None, w=None):
        """
        Forward pass matching STPN signature
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, num_nodes, in_len)
            ti: Time input embeddings (optional)
            supports: List of adjacency matrices for spatial modeling
            to: Time output embeddings (optional)
            w: Weather conditions (optional)
        """
        
        # Handle the input shape transformation for DSAFNet
        # Expected: (batch_size, airports, time_steps, features)
        if x.dim() == 4:
            # x is (batch_size, in_channels, num_nodes, in_len)
            # Transform to (batch_size, num_nodes, in_len, in_channels)
            x = x.permute(0, 2, 3, 1)
        
        batch_size, airports, time_steps, features = x.shape
        
        # Use supports as adjacency matrices if provided
        adj_matrices = supports if supports is not None else []
        
        spatial_features = [self.spatial_stream(x[:,:,t,:]) for t in range(time_steps)]
        temporal_features = [self.temporal_stream(x[:,a,:,:]) for a in range(airports)]
        spatial_stack = torch.stack(spatial_features, dim=2)
        temporal_stack = torch.stack(temporal_features, dim=1)
        fused = self.cross_fusion(spatial_stack.mean(2), temporal_stack.mean(2))
        
        if adj_matrices:
            graph_enhanced = self.graph_encoder(fused, adj_matrices)
        else:
            # If no adjacency matrices provided, use identity
            graph_enhanced = fused
            
        out = self.output_proj(graph_enhanced)
        
        # Transform output back to match expected STPN output format
        # out is (batch_size, airports, output_steps, 2)
        # Transform to (batch_size, 2, airports, output_steps)
        out = out.permute(0, 3, 1, 2)
        
        return out

# ===== Optimized Data Loader Class =====
class TrainWindowDataset(Dataset):
    def __init__(self, data, weather, period, in_len, out_len, indices):
        # Pre-compute all samples to avoid repeated computation during training
        self.samples = []
        self.period = period
        
        print(f"Pre-computing {len(indices)} training samples...")
        for idx in indices:
            s = idx
            x = data[:, s:s + in_len, :]              # (N, L_in, F)
            y = data[:, s + in_len:s + in_len + out_len, :]  # (N, L_out, F)
            w = weather[:, s:s + in_len]              # (N, L_in)

            ti = (np.arange(s, s + in_len) % period) * np.ones([1, in_len]) / (period - 1)
            to = (np.arange(s + in_len, s + in_len + out_len) % period) * np.ones([1, out_len]) / (period - 1)

            # Convert to tensors once and store
            x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  # (F, N, L_in)
            y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)  # (F, N, L_out)
            w = torch.tensor(w, dtype=torch.long)                      # (N, L_in)
            ti = torch.tensor(ti, dtype=torch.float32)                 # (1, L_in)
            to = torch.tensor(to, dtype=torch.float32)                 # (1, L_out)
            
            self.samples.append((x, y, ti, to, w))
        
        print(f"Pre-computation completed. {len(self.samples)} samples ready.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

# ===== Training Time Estimator =====
class TrainingTimeEstimator:
    """
    A class to estimate total training time based on initial epochs performance
    """
    def __init__(self, estimate_epochs=3, total_epochs=50):
        self.estimate_epochs = estimate_epochs
        self.total_epochs = total_epochs
        self.epoch_times = []
        self.start_time = None
        self.training_started = False
        
    def start_training(self):
        """Mark the start of training"""
        self.start_time = time.time()
        self.training_started = True
        print(f"🕐 Training started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📊 Will estimate total time after {self.estimate_epochs} epochs")
        print("-" * 60)
    
    def record_epoch_time(self, epoch, epoch_start_time):
        """Record time taken for current epoch"""
        epoch_time = time.time() - epoch_start_time
        self.epoch_times.append(epoch_time)
        
        # Display current epoch info
        elapsed_total = time.time() - self.start_time
        print(f"⏱️  Epoch {epoch+1}: {epoch_time:.2f}s | Total elapsed: {elapsed_total:.2f}s")
        
        # Provide estimate after initial epochs
        if len(self.epoch_times) == self.estimate_epochs:
            self._provide_estimate()
        
        # Update estimate periodically
        elif len(self.epoch_times) > self.estimate_epochs and (epoch + 1) % 10 == 0:
            self._update_estimate(epoch + 1)
    
    def _provide_estimate(self):
        """Provide initial time estimate based on first few epochs"""
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
        
        # Estimated completion time
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        print(f"🏁 Estimated completion: {completion_time.strftime('%H:%M:%S')}")
        print("="*60 + "\n")
    
    def _update_estimate(self, current_epoch):
        """Update estimate based on all epochs so far"""
        avg_epoch_time = np.mean(self.epoch_times)
        remaining_epochs = self.total_epochs - current_epoch
        estimated_remaining_time = avg_epoch_time * remaining_epochs
        
        print(f"\n📊 Updated estimate at epoch {current_epoch}:")
        print(f"   Average epoch time: {avg_epoch_time:.2f}s")
        print(f"   Estimated remaining: {self._format_time(estimated_remaining_time)}")
        completion_time = datetime.now() + timedelta(seconds=estimated_remaining_time)
        print(f"   Estimated completion: {completion_time.strftime('%H:%M:%S')}\n")
    
    def _format_time(self, seconds):
        """Format seconds into human-readable time"""
        if seconds < 60:
            return f"{seconds:.1f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.2f} hours"
    
    def finish_training(self):
        """Mark the end of training and provide final statistics"""
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
        
        # Show fastest and slowest epochs
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

# ===== Training Script =====

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device',type=str,default='cuda:0',help='')
    parser.add_argument('--data',type=str,default='US',help='data type')
    parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/val/test ratio', type=float)
    parser.add_argument('--in_channels',type=int,default=2,help='input variable')
    parser.add_argument('--out_channels',type=int,default=2,help='output variable')
    parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
    parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
    parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
    parser.add_argument('--lr', type=float, default=0.0005, help='learning rate ')
    parser.add_argument('--in_len',type=int,default=12,help='input time series length')
    parser.add_argument('--out_len',type=int,default=12,help='output time series length')
    parser.add_argument('--batch',type=int,default=64,help='training batch size (increased from 32)')
    parser.add_argument('--episode',type=int,default=25,help='training episodes')
    parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension for DSAFNet')
    parser.add_argument('--num_workers', type=int, default=8, help='number of data loader workers')
    parser.add_argument('--val_frequency', type=int, default=5, help='validate every N epochs')
    
    # === NEW: DP hyperparameters ===
    parser.add_argument('--dp', default=False,  action='store_true', help='enable differential privacy with Opacus')
    parser.add_argument('--target_epsilon', type=float, default=1.0, help='if >0, use make_private_with_epsilon')
    parser.add_argument('--target_delta', type=float, default=1e-5, help='delta for DP accounting')
    parser.add_argument('--noise_multiplier', type=float, default=1.5, help='sigma; used if target_epsilon <= 0')
    parser.add_argument('--max_grad_norm', type=float, default=1.5, help='per-sample gradient clipping norm')
    
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save training results")
    args = parser.parse_args()

    # === ENHANCED DEVICE SETUP ===
    print("="*60)
    print("🚀 DEVICE CONFIGURATION")
    print("="*60)
    
    # Force CUDA if available, fallback to CPU if not
    if torch.cuda.is_available():
        if args.device.startswith('cuda'):
            device = torch.device(args.device)
        else:
            device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    
    print(f"✅ Final device: {device}")
    print("="*60)
    
    # Enable optimizations
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        print(f"🎮 Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("⚠️  Using CPU")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Loading data...")
    start_time = time.time()
    # Load data using util.load_data
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")

    # Create DSAFNet model
    print("\n" + "="*60)
    print("🤖 MODEL INITIALIZATION")
    print("="*60)
    model = DSAFNet(
        input_dim=args.in_channels, 
        hidden_dim=args.hidden_dim, 
        output_steps=args.out_len, 
        num_graphs=args.support_len
    )
    print(f"📦 Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Move model to device
    model = model.to(device)
    
    print(f"✅ Model moved to device: {device}")
    print("Using DSAFNet model")
    print("="*60)

    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj]
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    print("Preprocessing data...")
    start_time = time.time()
    # Data preprocessing
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                            training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    print(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")

    # Create dataset and data loader with optimizations
    print("Creating dataset...")
    start_time = time.time()
    train_indices = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    dataset = TrainWindowDataset(
        data=training_data,
        weather=training_w,
        period=args.period,
        in_len=args.in_len,
        out_len=args.out_len,
        indices=train_indices
    )
    print(f"Dataset creation completed in {time.time() - start_time:.2f} seconds")

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

    # Differential Privacy setup
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
                csv_filename = f'dsafnet_training_ε={args.target_epsilon}_max_grad_norm={args.max_grad_norm}.csv'
            else:
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=args.noise_multiplier,
                    max_grad_norm=args.max_grad_norm,
                )
                csv_filename = f'dsafnet_noise_multiplier={args.noise_multiplier}_max_grad_norm={args.max_grad_norm}.csv'
            
        except Exception as e:
            print(f"ERROR during DP setup: {e}")
            print(f"Error type: {type(e)}")
            raise
    else:
        privacy_engine = None
        csv_filename = 'dsafnet_training_no_dp.csv'

    # Validation data preparation (optimized)
    print("Preparing validation data...")
    start_time = time.time()
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    
    # Subsample validation data for faster validation
    if len(val_index) > 1000:
        print(f"Large validation set detected ({len(val_index)} samples). Subsampling to 1000 for faster validation.")
        val_index = val_index[::len(val_index)//1000][:1000]
    
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)
    print(f"Validation data preparation completed in {time.time() - start_time:.2f} seconds")

    print("start training...", flush=True)
    print("\n" + "="*60)
    print("🚀 TRAINING LOOP STARTED")
    print("="*60)

    # Initialize CSV file for logging training metrics
    csv_headers = ['epoch', 'training_loss', 'validation_loss', 'validation_mae', 'validation_r2', 'validation_rmse', 'epsilon']
    
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
        
        # Log training arguments as metadata rows
        writer.writerow(['# Training Arguments'])
        args_dict = vars(args)
        for key, value in args_dict.items():
            writer.writerow([f'# {key}', str(value)])
        writer.writerow(['# Start of Training Data'])
        writer.writerow(csv_headers)

    # Initialize lists to store metrics for each epoch
    training_metrics = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_mae': [],
        'val_mae': [],
        'val_r2': [],
        'val_rmse': []
    }

    # Initialize time estimator
    time_estimator = TrainingTimeEstimator(estimate_epochs=3, total_epochs=args.episode)
    time_estimator.start_training()

    MAE_list = []
    best_model = copy.deepcopy(model.state_dict())

    for ep in range(1, 1 + args.episode):
        epoch_start_time = time.time()
        
        # Initialize epoch training loss tracking
        epoch_training_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            trainx, trainy, trainti, trainto, trainw = batch
            
            # Move to device
            trainx = trainx.to(device, non_blocking=False)
            trainy = trainy.to(device, non_blocking=False)
            trainti = trainti.to(device, non_blocking=False)
            trainto = trainto.to(device, non_blocking=False)
            trainw = trainw.to(device, non_blocking=False)

            if trainx.dim() == 3:
                trainx = trainx.unsqueeze(0)
                trainy = trainy.unsqueeze(0)
                trainti = trainti.unsqueeze(0)
                trainto = trainto.unsqueeze(0)
                trainw = trainw.unsqueeze(0)

            model.train()

            optimizer.zero_grad(set_to_none=True)
            
            output = model(trainx, trainti, supports, trainto, trainw)
            
            # Handle shape mismatch if output and target have different shapes
            if output.shape != trainy.shape:
                if output.shape[-1] == 1 and trainy.shape[-1] > 1:
                    # Model predicts 1 step, target has 12 steps - take last step of target
                    trainy = trainy[..., -1:]

            loss = util.masked_rmse(output, trainy, 0.0)
            
            loss.backward()

            if not args.dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            try:
                optimizer.step()
            except Exception as e:
                print(f"ERROR in optimizer.step(): {e}")
                print(f"Error type: {type(e)}")
                raise Exception("Error occurred during optimizer.step()")
            
            # Accumulate training loss for this epoch
            epoch_training_loss += loss.item()
            num_batches += 1

        # Validation phase
        model.eval()
        outputs = []
        validation_loss = 0.0
        val_loss_count = 0
        
        with torch.no_grad():
            for i in range(len(val_index)):
                testx = np.expand_dims(val_data[:, val_index[i]: val_index[i] + args.in_len, :], axis=0)
                testx = scaler.transform(testx)
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
                
                # Calculate validation loss for this sample
                testy = np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0)
                testy = scaler.transform(testy)
                testy[np.isnan(testy)] = 0
                testy = torch.Tensor(testy).to(device).permute(0, 3, 1, 2)
                
                # Adjust shapes if necessary (same logic as training)
                if out.shape != testy.shape:
                    if out.shape[-1] == 1 and testy.shape[-1] > 1:
                        testy = testy[..., -1:]
                
                val_loss = util.masked_rmse(out, testy, 0.0)
                validation_loss += val_loss.item()
                val_loss_count += 1
                
                out = out.permute(0, 2, 3, 1)
                out = out.detach().cpu().numpy()
                out = scaler.inverse_transform(out)
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
        
        # Calculate average losses for this epoch
        avg_training_loss = epoch_training_loss / num_batches if num_batches > 0 else 0.0
        avg_validation_loss = validation_loss / val_loss_count if val_loss_count > 0 else 0.0

        # Get epsilon value for DP
        current_epsilon = None
        if args.dp and privacy_engine:
            try:
                current_epsilon = privacy_engine.get_epsilon(args.target_delta)
            except Exception:
                current_epsilon = 'N/A'
        else:
            current_epsilon = 'No DP'

        # Store metrics
        training_metrics['epoch'].append(ep)
        training_metrics['train_loss'].append(avg_training_loss)
        training_metrics['val_loss'].append(avg_validation_loss)
        training_metrics['train_mae'].append(mean_mae)  # Using validation MAE as train MAE for consistency
        training_metrics['val_mae'].append(mean_mae)
        training_metrics['val_r2'].append(mean_r2)
        training_metrics['val_rmse'].append(mean_rmse)

        # Write metrics to CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep, avg_training_loss, avg_validation_loss, mean_mae, mean_r2, mean_rmse, current_epsilon])

        # Record epoch time and get estimates
        time_estimator.record_epoch_time(ep-1, epoch_start_time)

        if args.dp:
            try:
                eps = privacy_engine.get_epsilon(args.target_delta)
                print(f"Epoch {ep:03d} | Train Loss {avg_training_loss:.4f} | Val Loss {avg_validation_loss:.4f} | Val MAE {mean_mae:.4f} R2 {mean_r2:.4f} RMSE {mean_rmse:.4f} | ε ~ {eps:.2f}, δ={args.target_delta}")
            except Exception:
                print(f"Epoch {ep:03d} | Train Loss {avg_training_loss:.4f} | Val Loss {avg_validation_loss:.4f} | Val MAE {mean_mae:.4f} R2 {mean_r2:.4f} RMSE {mean_rmse:.4f}")
        else:
            print(f"Epoch {ep:03d} | Train Loss {avg_training_loss:.4f} | Val Loss {avg_validation_loss:.4f} | Val MAE {mean_mae:.4f} R2 {mean_r2:.4f} RMSE {mean_rmse:.4f}")

        MAE_list.append(mean_mae)
        if mean_mae == min(MAE_list):  # store best
            best_model = copy.deepcopy(model.state_dict())

    # Finish training and get final stats
    final_stats = time_estimator.finish_training()
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Save metrics to Excel file
    df = pd.DataFrame(training_metrics)
    excel_path = os.path.join(args.output_dir, 'dsafnet_training_metrics.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Training metrics saved to: {excel_path}")

    # Save training time statistics
    if final_stats:
        time_stats = pd.DataFrame([{
            'Total_Training_Time_Seconds': final_stats['total_time'],
            'Total_Training_Time_Minutes': final_stats['total_time'] / 60,
            'Average_Epoch_Time_Seconds': final_stats['avg_epoch_time'],
            'Total_Epochs': final_stats['total_epochs'],
            'Epochs_Per_Minute': final_stats['total_epochs'] / (final_stats['total_time'] / 60),
            'Training_Efficiency': f"{final_stats['total_epochs'] / final_stats['total_time'] * 60:.1f} epochs/min"
        }])
        time_stats_path = os.path.join(args.output_dir, 'dsafnet_training_time_statistics.xlsx')
        time_stats.to_excel(time_stats_path, index=False)
        print(f"Training time statistics saved to: {time_stats_path}")

    # Save model
    model_path = os.path.join(args.output_dir, "dsafnet_" + args.data + ".pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    print("Training complete.")

if __name__ == "__main__":
    main()
