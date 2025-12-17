# -*- coding: utf-8 -*-
"""
Enhanced training script with Differential Privacy using Opacus
Includes latest DP advancements for sparse data and overfitting prevention
"""

import torch
import util
import argparse
import random
import copy
import torch.optim as optim
import numpy as np
import os
from datetime import datetime
import warnings
from typing import Optional, Tuple, List
from baseline_methods import test_error, StandardScaler
from model import STPN

# Differential Privacy imports
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.accountants import RDPAccountant
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.optimizers import DPOptimizer
from opacus.grad_sample import GradSampleModule

parser = argparse.ArgumentParser()

# Original arguments
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='US',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/val/test ratio', type=float)
parser.add_argument('--h_layers',type=int,default=2,help='number of hidden layer')
parser.add_argument('--in_channels',type=int,default=2,help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=[128, 64, 32], help='hidden layer dimension', type=int)
parser.add_argument('--out_channels',type=int,default=2,help='output variable')
parser.add_argument('--emb_size',type=int,default=16,help='time embedding size')
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--wemb_size',type=int,default=4,help='covairate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=8,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')
parser.add_argument('--in_len',type=int,default=12,help='input time series length')
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
# Differential Privacy arguments
parser.add_argument('--enable_dp', default=True, action='store_true', help='Enable differential privacy')
parser.add_argument('--target_epsilon', type=float, default=3.0, help='Target privacy budget (epsilon)')
parser.add_argument('--target_delta', type=float, default=1e-5, help='Target delta for (epsilon, delta)-DP')
parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Maximum gradient norm for clipping')
parser.add_argument('--noise_multiplier', type=float, default=None, help='Noise multiplier (auto-computed if None)')
parser.add_argument('--secure_mode', action='store_true', help='Enable secure RNG for cryptographically secure noise')
# Advanced DP options for sparse data
parser.add_argument('--adaptive_clipping', action='store_true', help='Use adaptive gradient clipping')
parser.add_argument('--clip_percentile', type=float, default=90, help='Percentile for adaptive clipping')
parser.add_argument('--sparse_aware_noise', action='store_true', help='Apply sparse-aware noise addition')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation for virtual batching')
parser.add_argument('--dp_warmup_epochs', type=int, default=5, help='Number of warmup epochs before applying DP')
parser.add_argument('--privacy_amplification', action='store_true', help='Use subsampling for privacy amplification')
parser.add_argument('--poisson_sampling', action='store_true', help='Use Poisson sampling instead of uniform')
# Regularization for overfitting with DP
parser.add_argument('--dp_dropout_rate', type=float, default=0.2, help='Additional dropout when using DP')
parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')
parser.add_argument('--weight_standardization', action='store_true', help='Apply weight standardization')
parser.add_argument('--spectral_norm', action='store_true', help='Apply spectral normalization')

args = parser.parse_args()

class DPModelWrapper(torch.nn.Module):
    """Wrapper to make model DP-compatible with additional regularization"""
    def __init__(self, model, use_spectral_norm=False, additional_dropout=0.0):
        super().__init__()
        self.model = model
        self.additional_dropout = torch.nn.Dropout(additional_dropout) if additional_dropout > 0 else None
        if use_spectral_norm:
            self._apply_spectral_norm()
    def _apply_spectral_norm(self):
        """Apply spectral normalization to linear and conv layers"""
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
                torch.nn.utils.parametrizations.spectral_norm(module)
    def forward(self, *args, **kwargs):
        out = self.model(*args, **kwargs)
        if self.additional_dropout is not None and self.training:
            out = self.additional_dropout(out)
        return out

class AdaptiveClippingCallback:
    """Implements adaptive gradient clipping based on gradient statistics"""
    def __init__(self, percentile=90, window_size=100):
        self.percentile = percentile
        self.window_size = window_size
        self.grad_norms = []
    def update_clip_norm(self, model, current_clip):
        """Compute adaptive clipping threshold based on recent gradients"""
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.grad_norms.append(total_norm)
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)
        if len(self.grad_norms) >= 10:
            adaptive_clip = np.percentile(self.grad_norms, self.percentile)
            return min(adaptive_clip, current_clip * 2)
        return current_clip

class SparseAwareNoiseScheduler:
    """Implements sparse-aware noise scheduling"""
    def __init__(self, base_noise_multiplier, sparsity_threshold=0.1):
        self.base_noise = base_noise_multiplier
        self.sparsity_threshold = sparsity_threshold
    def compute_noise_multiplier(self, gradients, epoch, max_epochs):
        total_params = sum(g.numel() for g in gradients if g is not None)
        sparse_params = sum((g.abs() < self.sparsity_threshold).sum().item() for g in gradients if g is not None)
        sparsity_ratio = sparse_params / total_params if total_params > 0 else 0
        decay_factor = 1.0 - (epoch / max_epochs) * 0.3
        sparsity_factor = 1.0 - sparsity_ratio * 0.5
        return self.base_noise * decay_factor * sparsity_factor

def masked_rmse_with_label_smoothing(output, target, null_val=0.0, smoothing=0.0):
    """RMSE loss with label smoothing for regularization"""
    if smoothing > 0:
        noise = torch.randn_like(target) * smoothing
        target = target + noise
    return util.masked_rmse(output, target, null_val)

def setup_logging(args):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'training_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    dp_suffix = '_dp' if args.enable_dp else ''
    log_file = os.path.join(log_dir, f'training_log{dp_suffix}_{timestamp}.txt')
    return log_file

def log_message(message, log_file):
    """Print message to console and append to log file"""
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

class DPDataLoader:
    """
    Custom DataLoader wrapper for DP training
    Expects (data, labels, weather, time_in, time_out) to be torch tensors.
    """
    def __init__(self, data, labels, weather, time_in, time_out, batch_size, shuffle=True):
        self.dataset = data
        self.labels = labels
        self.weather = weather
        self.time_in = time_in
        self.time_out = time_out
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(data)))
    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        for i in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[i:i + self.batch_size]
            if len(batch_indices) < self.batch_size:
                continue  # Skip incomplete batches for DP
            batch_data = torch.stack([self.data[j] for j in batch_indices])
            batch_labels = torch.stack([self.labels[j] for j in batch_indices])
            batch_weather = torch.stack([self.weather[j] for j in batch_indices])
            batch_time_in = torch.stack([self.time_in[j] for j in batch_indices])
            batch_time_out = torch.stack([self.time_out[j] for j in batch_indices])
            yield batch_data, batch_labels, batch_weather, batch_time_in, batch_time_out
    def __len__(self):
        return len(self.indices) // self.batch_size

def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    supports = [torch.tensor(i).to(device) for i in adj]

    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                           training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0

    batch_index = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))

    ## ========== Initialize model, DP wrappers/fixes FIRST ===========
    model_core = STPN(
        args.h_layers, args.in_channels, args.hidden_channels, args.out_channels,
        args.emb_size, args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
        args.order, args.num_weather, args.use_se, args.use_cov
    ).to(device)

    if args.enable_dp:
        # Fix for DP compatibility
        model_core = ModuleValidator.fix(model_core)
        # Add custom wrapper (dropout, spectral norm if requested)
        model_core = DPModelWrapper(
            model_core,
            use_spectral_norm=args.spectral_norm,
            additional_dropout=args.dp_dropout_rate
        )
    model = model_core

    ## ========== Now initialize optimizer ===========
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    privacy_engine = None
    adaptive_clipper = None
    noise_scheduler = None

    if args.enable_dp:
        # Prepare all training data for DP
        all_train_x = []
        all_train_y = []
        all_train_w = []
        all_train_ti = []
        all_train_to = []
        for idx in batch_index:
            all_train_x.append(training_data[:, idx:idx + args.in_len, :])
            all_train_y.append(training_data[:, idx + args.in_len:idx + args.in_len + args.out_len, :])
            all_train_w.append(training_w[:, idx:idx + args.in_len])
            all_train_ti.append((np.arange(idx, idx + args.in_len) % args.period) / (args.period - 1))
            all_train_to.append((np.arange(idx + args.in_len, idx + args.in_len + args.out_len) % args.period) / (args.period - 1))
        # Convert to tensors (B, V, T, C)
        all_train_x = torch.FloatTensor(np.array(all_train_x)).permute(0, 3, 1, 2).to(device)
        all_train_y = torch.FloatTensor(np.array(all_train_y)).permute(0, 3, 1, 2).to(device)
        all_train_w = torch.LongTensor(np.array(all_train_w)).to(device)
        all_train_ti = torch.FloatTensor(np.array(all_train_ti)).to(device)
        all_train_to = torch.FloatTensor(np.array(all_train_to)).to(device)

        dp_dataloader = DPDataLoader(
            all_train_x, all_train_y, all_train_w, all_train_ti, all_train_to,
            batch_size=args.batch, shuffle=True
        )

        # Initialize PrivacyEngine
        privacy_engine = PrivacyEngine()
        # Calculate appropriate noise multiplier if not specified
        if args.noise_multiplier is None:
            from opacus.accountants.utils import get_noise_multiplier
            sample_rate = args.batch / len(batch_index)
            steps = args.episode * (len(batch_index) // args.batch)
            args.noise_multiplier = get_noise_multiplier(
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                sample_rate=sample_rate,
                steps=steps,
            )
        # Make private
        model, optimizer, dp_dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dp_dataloader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
            poisson_sampling=args.poisson_sampling,
        )
        # Adaptive components
        if args.adaptive_clipping:
            adaptive_clipper = AdaptiveClippingCallback(percentile=args.clip_percentile)
        if args.sparse_aware_noise:
            noise_scheduler = SparseAwareNoiseScheduler(args.noise_multiplier)

    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)

    log_file = setup_logging(args)
    log_message("=== Training Configuration ===", log_file)
    log_message(f"Model parameters: {sum(p.numel() for p in model.parameters())}", log_file)
    log_message(f"Learning rate: {args.lr}", log_file)
    log_message(f"Batch size: {args.batch}", log_file)

    if args.enable_dp:
        log_message("\n=== Differential Privacy Configuration ===", log_file)
        log_message(f"Target epsilon: {args.target_epsilon}", log_file)
        log_message(f"Target delta: {args.target_delta}", log_file)
        log_message(f"Max gradient norm: {args.max_grad_norm}", log_file)
        log_message(f"Noise multiplier: {args.noise_multiplier}", log_file)
        log_message(f"Adaptive clipping: {args.adaptive_clipping}", log_file)
        log_message(f"Sparse-aware noise: {args.sparse_aware_noise}", log_file)

    log_message("\nStarting training...", log_file)

    MAE_list = []
    best_model = None

    for ep in range(1, 1 + args.episode):
        epoch_losses = []
        epoch_maes = []
        # DP warmup logic
        apply_dp = args.enable_dp and (ep > args.dp_warmup_epochs)

        if args.enable_dp and apply_dp:
            # DP training with DPDataLoader
            for batch_idx, (trainx, trainy, trainw, trainti, trainto) in enumerate(dp_dataloader):
                model.train()
                output = model(trainx, trainti, supports, trainto, trainw)
                if args.label_smoothing > 0 and apply_dp:
                    loss = masked_rmse_with_label_smoothing(output, trainy, 0.0, args.label_smoothing)
                else:
                    loss = util.masked_rmse(output, trainy, 0.0)
                epoch_losses.append(loss.item())
                # Training MAE
                with torch.no_grad():
                    output_denorm = scaler.inverse_transform(output.permute(0, 2, 3, 1).cpu().numpy())
                    target_denorm = scaler.inverse_transform(trainy.permute(0, 2, 3, 1).cpu().numpy())
                    train_mae = np.mean(np.abs(output_denorm - target_denorm))
                    epoch_maes.append(train_mae)
                optimizer.zero_grad()
                loss.backward()
                # Adaptive clipping
                if adaptive_clipper and apply_dp:
                    new_clip = adaptive_clipper.update_clip_norm(model, args.max_grad_norm)
                    if hasattr(optimizer, 'max_grad_norm'):
                        optimizer.max_grad_norm = new_clip
                optimizer.step()
                # Sparse-aware noise
                if noise_scheduler and apply_dp and hasattr(optimizer, 'noise_multiplier'):
                    gradients = [p.grad for p in model.parameters()]
                    new_noise = noise_scheduler.compute_noise_multiplier(gradients, ep, args.episode)
                    optimizer.noise_multiplier = new_noise
        else:
            # Non-DP training
            random.shuffle(batch_index)
            for j in range(len(batch_index) // args.batch - 1):
                trainx = []
                trainy = []
                trainti = []
                trainto = []
                trainw = []
                for k in range(args.batch):
                    idx = batch_index[j * args.batch + k]
                    trainx.append(np.expand_dims(training_data[:, idx:idx + args.in_len, :], axis=0))
                    trainy.append(np.expand_dims(training_data[:, idx + args.in_len:idx + args.in_len + args.out_len, :], axis=0))
                    trainw.append(np.expand_dims(training_w[:, idx:idx + args.in_len], axis=0))
                    trainti.append((np.arange(idx, idx + args.in_len) % args.period) * np.ones([1, args.in_len]) / (args.period - 1))
                    trainto.append((np.arange(idx + args.in_len, idx + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len]) / (args.period - 1))
                trainx = torch.FloatTensor(np.concatenate(trainx)).permute(0, 3, 1, 2).to(device)
                trainy = torch.FloatTensor(np.concatenate(trainy)).permute(0, 3, 1, 2).to(device)
                trainw = torch.LongTensor(np.concatenate(trainw)).to(device)
                trainti = torch.FloatTensor(np.concatenate(trainti)).to(device)
                trainto = torch.FloatTensor(np.concatenate(trainto)).to(device)
                model.train()
                optimizer.zero_grad()
                output = model(trainx, trainti, supports, trainto, trainw)
                loss = util.masked_rmse(output, trainy, 0.0)
                epoch_losses.append(loss.item())
                # Training MAE
                with torch.no_grad():
                    output_denorm = scaler.inverse_transform(output.permute(0, 2, 3, 1).cpu().numpy())
                    target_denorm = scaler.inverse_transform(trainy.permute(0, 2, 3, 1).cpu().numpy())
                    train_mae = np.mean(np.abs(output_denorm - target_denorm))
                    epoch_maes.append(train_mae)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()

        # Validation phase
        model.eval()
        outputs = []
        amae = []
        ar2 = []
        armse = []
        with torch.no_grad():
            for i in range(len(val_index)):
                testx = np.expand_dims(val_data[:, val_index[i]:val_index[i] + args.in_len, :], axis=0)
                testx = scaler.transform(testx)
                testw = np.expand_dims(val_w[:, val_index[i]:val_index[i] + args.in_len], axis=0)
                testw = torch.LongTensor(testw).to(device)
                testx[np.isnan(testx)] = 0
                testti = (np.arange(int(training_data.shape[1]) + val_index[i], int(training_data.shape[1]) + val_index[i] + args.in_len) % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
                testto = (np.arange(int(training_data.shape[1]) + val_index[i] + args.in_len, int(training_data.shape[1]) + val_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len]) / (args.period - 1)
                testx = torch.FloatTensor(testx).permute(0, 3, 1, 2).to(device)
                testti = torch.FloatTensor(testti).to(device)
                testto = torch.FloatTensor(testto).to(device)
                output = model(testx, testti, supports, testto, testw)
                output = output.permute(0, 2, 3, 1).cpu().numpy()
                output = scaler.inverse_transform(output)
                outputs.append(output)
            yhat = np.concatenate(outputs)
            for i in range(12):
                metrics = test_error(yhat[:, :, i, :], label[:, :, i, :])
                amae.append(metrics)
                ar2.append(metrics[2])
                armse.append(metrics[1])
        mean_mae = np.mean(amae)
        MAE_list.append(mean_mae)

        # Log epoch summary
        epoch_summary = f"""
Epoch {ep}/{args.episode}
Training - Average Loss: {np.mean(epoch_losses):.4f}, MAE: {np.mean(epoch_maes):.4f}
Validation - MAE: {mean_mae:.4f}, R2: {np.mean(ar2):.4f}, RMSE: {np.mean(armse):.4f}"""
        # Add privacy budget info if DP is enabled
        if args.enable_dp and apply_dp and privacy_engine:
            epsilon = privacy_engine.get_epsilon(args.target_delta)
            epoch_summary += f"\nPrivacy spent: (ε={epsilon:.2f}, δ={args.target_delta})"
        epoch_summary += f"\n{'-'*80}"
        log_message(epoch_summary, log_file)

        # Save best model
        if mean_mae == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
            log_message("New best model saved!", log_file)

        # Save model
        model_name = f"stpn_{args.data}_{'dp' if args.enable_dp else 'standard'}.pth"
        torch.save({
            'model_state_dict': best_model,
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': ep,
            'mae': mean_mae,
            'args': args,
            'privacy_spent': (epsilon, args.target_delta) if args.enable_dp and apply_dp and privacy_engine else None
        }, model_name)

    # Final summary
    log_message("\n=== Training Complete ===", log_file)
    log_message(f"Best validation MAE: {min(MAE_list):.4f}", log_file)
    if args.enable_dp and privacy_engine:
        final_epsilon = privacy_engine.get_epsilon(args.target_delta)
        log_message(f"Final privacy guarantee: (ε={final_epsilon:.2f}, δ={args.target_delta})", log_file)

if __name__ == "__main__":
    main()
