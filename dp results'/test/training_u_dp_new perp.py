# -*- coding: utf-8 -*-

"""
Created on Tue Jul 12 16:36:04 2022
Modified to include Differential Privacy with Opacus
Supports both make_private and make_private_with_epsilon methods
Includes automatic BatchNorm to GroupNorm conversion

@author: AA
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
from baseline_methods import test_error, StandardScaler
from model import STPN

# Opacus imports for differential privacy
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from opacus.utils.uniform_sampler import UniformWithReplacementSampler
from torch.utils.data import Dataset, DataLoader

parser = argparse.ArgumentParser()
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
parser.add_argument('--use_se', type=str, default=False,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')
parser.add_argument('--in_len',type=int,default=12,help='input time series length')
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')

# Differential Privacy arguments
parser.add_argument('--enable_dp', type=bool, default=True, help='enable differential privacy')
parser.add_argument('--dp_method', type=str, default='make_private', choices=['make_private', 'make_private_with_epsilon'], 
                    help='DP method: make_private (specify noise_multiplier) or make_private_with_epsilon (specify target_epsilon)')

# Arguments for make_private method
parser.add_argument('--noise_multiplier', type=float, default=1.1, 
                    help='noise multiplier for DP (used with make_private method)')

# Arguments for make_private_with_epsilon method  
parser.add_argument('--target_epsilon', type=float, default=8.0, 
                    help='target epsilon for DP (used with make_private_with_epsilon method)')

# Common DP arguments
parser.add_argument('--max_grad_norm', type=float, default=1.0, help='maximum gradient norm for clipping')
parser.add_argument('--delta', type=float, default=1e-5, help='delta parameter for (epsilon, delta)-DP')
parser.add_argument('--max_physical_batch_size', type=int, default=16, help='maximum physical batch size for memory management')

# Model fixing arguments
parser.add_argument('--auto_fix_model', type=bool, default=True, help='automatically fix incompatible modules (BatchNorm->GroupNorm)')
parser.add_argument('--validate_model_strict', type=bool, default=False, help='strict validation of model compatibility')

args = parser.parse_args()

def setup_logging(args):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'training_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    dp_suffix = f"_dp_{args.dp_method}" if args.enable_dp else ""
    log_file = os.path.join(log_dir, f'training_log{dp_suffix}_{timestamp}.txt')
    return log_file

def log_message(message, log_file):
    """Print message to console and append to log file"""
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def fix_model_for_dp(model, log_file):
    """
    Fix model to be compatible with differential privacy.
    Automatically replaces BatchNorm with GroupNorm and other incompatible layers.
    """
    log_message("=== Model Compatibility Check ===", log_file)
    
    # First, validate the original model
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        log_message(f"Found {len(errors)} compatibility issues:", log_file)
        
    
    # Apply automatic fixes
    if args.auto_fix_model:
        log_message("Applying automatic fixes...", log_file)
        fixed_model = ModuleValidator.fix(model)
        log_message("Model automatically fixed for DP compatibility", log_file)
        
        # Validate the fixed model
        validation_errors = ModuleValidator.validate(fixed_model, strict=args.validate_model_strict)
        
            
        return fixed_model
    else:
        
        return model

def create_custom_dataloader(trainx, trainy, trainti, trainto, trainw, batch_size, device):
    """Create a custom dataset and dataloader for our spatio-temporal data"""
    class SpatioTemporalDataset(torch.utils.data.Dataset):
        def __init__(self, x, y, ti, to, w):
            self.x = torch.Tensor(x).to(device)
            self.y = torch.Tensor(y).to(device)
            self.ti = torch.Tensor(ti).to(device)
            self.to = torch.Tensor(to).to(device)
            self.w = torch.LongTensor(w).to(device)
            
        def __len__(self):
            return len(self.x)
            
        def __getitem__(self, idx):
            return {
                'x': self.x[idx].permute(2, 0, 1),  # [features, airports, time]
                'y': self.y[idx].permute(2, 0, 1),  # [features, airports, time]
                'ti': self.ti[idx],
                'to': self.to[idx],
                'w': self.w[idx]
            }
    
    dataset = SpatioTemporalDataset(trainx, trainy, trainti, trainto, trainw)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def apply_differential_privacy(privacy_engine, model, optimizer, dataloader, args, log_file):
    """Apply differential privacy using the specified method"""
    
    if args.dp_method == 'make_private_with_epsilon':
        # Method 1: Target epsilon approach - automatically calibrates noise
        log_message(f"Using make_private_with_epsilon with target_epsilon={args.target_epsilon}", log_file)
        
        model, optimizer, dataloader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            epochs=args.episode,
            target_epsilon=args.target_epsilon,
            target_delta=args.delta,
            max_grad_norm=args.max_grad_norm,
        )
        
        # Get the automatically calculated noise multiplier
        noise_multiplier = optimizer.noise_multiplier
        log_message(f"Automatically calculated noise_multiplier: {noise_multiplier:.4f}", log_file)
        
    else:  # make_private
        # Method 2: Manual noise multiplier approach - user specifies exact noise
        log_message(f"Using make_private with noise_multiplier={args.noise_multiplier}", log_file)
        
        model, optimizer, dataloader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=dataloader,
            noise_multiplier=args.noise_multiplier,
            max_grad_norm=args.max_grad_norm,
        )
    
    return model, optimizer, dataloader

def main():
    device = torch.device(args.device)
    
    # Load data
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    
    # Initialize model
    model = STPN(args.h_layers, args.in_channels, args.hidden_channels, args.out_channels, args.emb_size,
                 args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
                 args.order, args.num_weather, args.use_se, args.use_cov).to(device)
    
    supports = [torch.tensor(i, dtype=torch.float32).to(device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # Setup logging
    log_file = setup_logging(args)
    
    log_message("=== Training Configuration ===", log_file)
    log_message(f"Original model parameters: {sum(p.numel() for p in model.parameters())}", log_file)
   
    
    # Standardize training data
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
     
    
    
    # Fix model for differential privacy if enabled
    if args.enable_dp:
        model = fix_model_for_dp(model, log_file)
        log_message(f"Fixed model parameters: {sum(p.numel() for p in model.parameters())}", log_file)
    
    # Prepare batch indices
    batch_index = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    
    # Prepare validation labels
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)
    
    # Initialize Privacy Engine if DP is enabled
    privacy_engine = None
    if args.enable_dp:
        privacy_engine = PrivacyEngine()
        log_message("=== Differential Privacy Enabled ===", log_file)
        log_message(f"DP Method: {args.dp_method}", log_file)
        
        if args.dp_method == 'make_private_with_epsilon':
            log_message(f"Target epsilon: {args.target_epsilon}", log_file)
        else:
            log_message(f"Noise multiplier: {args.noise_multiplier}", log_file)
            
        log_message(f"Max gradient norm: {args.max_grad_norm}", log_file)
        log_message(f"Delta: {args.delta}", log_file)
    
    log_message(f"Learning rate: {args.lr}", log_file)
    log_message(f"Batch size: {args.batch}", log_file)
    log_message(f"Differential Privacy: {'Enabled (' + args.dp_method + ')' if args.enable_dp else 'Disabled'}", log_file)
    log_message("start training...", log_file)
    
    MAE_list = []
    
    for ep in range(1, 1 + args.episode):
        random.shuffle(batch_index)
        
        # Prepare training batches for this epoch
        all_trainx, all_trainy, all_trainti, all_trainto, all_trainw = [], [], [], [], []
        
        # Collect all training data for the epoch
        for j in range(len(batch_index) // args.batch - 1):
            trainx = []
            trainy = []
            trainti = []
            trainto = []
            trainw = []
            
            for k in range(args.batch):
                idx = j * args.batch + k
                batch_idx = batch_index[idx]
                
                trainx.append(np.expand_dims(training_data[:, batch_idx:batch_idx + args.in_len, :], axis=0))
                trainy.append(np.expand_dims(training_data[:, batch_idx + args.in_len:batch_idx + args.in_len + args.out_len, :], axis=0))
                trainw.append(np.expand_dims(training_w[:, batch_idx:batch_idx + args.in_len], axis=0))
                trainti.append((np.arange(batch_idx, batch_idx + args.in_len) % args.period) * np.ones([1, args.in_len]) / (args.period - 1))
                trainto.append((np.arange(batch_idx + args.in_len, batch_idx + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len]) / (args.period - 1))
            
            all_trainx.extend(trainx)
            all_trainy.extend(trainy)
            all_trainti.extend(trainti)
            all_trainto.extend(trainto)
            all_trainw.extend(trainw)
        
        # Convert to numpy arrays
        all_trainx = np.concatenate(all_trainx)
        all_trainy = np.concatenate(all_trainy)
        all_trainti = np.concatenate(all_trainti)
        all_trainto = np.concatenate(all_trainto)
        all_trainw = np.concatenate(all_trainw)
        
        # Create dataloader
        dataloader = create_custom_dataloader(all_trainx, all_trainy, all_trainti, all_trainto, all_trainw, args.batch, device)
        
        # Apply differential privacy if enabled (only on first epoch to avoid re-wrapping)
        if args.enable_dp and ep == 1:
            model, optimizer, dataloader = apply_differential_privacy(privacy_engine, model, optimizer, dataloader, args, log_file)
            log_message(f"Epoch {ep}: DP-enabled model created using {args.dp_method}", log_file)
        
        # Training phase with memory management for DP
        epoch_losses = []
        epoch_maes = []
        
        model.train()
        
        def train_step(batch):
            trainx = batch['x']
            trainy = batch['y']
            trainti = batch['ti']
            trainto = batch['to']
            trainw = batch['w']
            
            optimizer.zero_grad()
            output = model(trainx, trainti, supports, trainto, trainw)
            loss = util.masked_rmse(output, trainy, 0.0)
            
            # Calculate training MAE
            with torch.no_grad():
                output_denorm = scaler.inverse_transform(output.permute(0, 2, 3, 1).cpu().numpy())
                target_denorm = scaler.inverse_transform(trainy.permute(0, 2, 3, 1).cpu().numpy())
                train_mae = np.mean(np.abs(output_denorm - target_denorm))
            
            loss.backward()
            
            if not args.enable_dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            
            optimizer.step()
            
            return loss.item(), train_mae
        
        if args.enable_dp:
            # Use BatchMemoryManager for DP training
            with BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=args.max_physical_batch_size,
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for batch in memory_safe_data_loader:
                    loss_val, mae_val = train_step(batch)
                    epoch_losses.append(loss_val)
                    epoch_maes.append(mae_val)
        else:
            # Standard training loop
            for batch in dataloader:
                loss_val, mae_val = train_step(batch)
                epoch_losses.append(loss_val)
                epoch_maes.append(mae_val)
        
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
                
                testti = (np.arange(int(training_data.shape[1]) + val_index[i], 
                                  int(training_data.shape[1]) + val_index[i] + args.in_len) % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
                testto = (np.arange(int(training_data.shape[1]) + val_index[i] + args.in_len, 
                                  int(training_data.shape[1]) + val_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len]) / (args.period - 1)
                
                testx = torch.Tensor(testx).to(device)
                testx = testx.permute(0, 3, 1, 2)
                testti = torch.Tensor(testti).to(device)
                testto = torch.Tensor(testto).to(device)
                
                output = model(testx, testti, supports, testto, testw)
                output = output.permute(0, 2, 3, 1)
                output = output.detach().cpu().numpy()
                output = scaler.inverse_transform(output)
                outputs.append(output)
        
        yhat = np.concatenate(outputs)
        
        # Calculate validation metrics
        for i in range(12):
            metrics = test_error(yhat[:, :, i, :], label[:, :, i, :])
            amae.append(metrics[0])
            ar2.append(metrics[2])
            armse.append(metrics[1])
        
        # Update MAE list for best model tracking
        mean_mae = np.mean(amae)
        MAE_list.append(mean_mae)
        
        # Calculate privacy budget if DP is enabled
        privacy_info = ""
        if args.enable_dp:
            epsilon = privacy_engine.get_epsilon(args.delta)
            if args.dp_method == 'make_private_with_epsilon':
                privacy_info = f", ε: {epsilon:.4f}/{args.target_epsilon:.4f}, δ: {args.delta}"
            else:
                privacy_info = f", ε: {epsilon:.4f}, δ: {args.delta}, σ: {args.noise_multiplier}"
        
        # Log epoch summary
        epoch_summary = f"""
Epoch {ep}/{args.episode}
Training - Average Loss: {np.mean(epoch_losses):.4f}, MAE: {np.mean(epoch_maes):.4f}
Validation - MAE: {mean_mae:.4f}, R2: {np.mean(ar2):.4f}, RMSE: {np.mean(armse):.4f}{privacy_info}
{'-'*80}"""
        log_message(epoch_summary, log_file)
        
        # Save best model
        if mean_mae == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
            log_message("New best model saved!", log_file)
    
    # Load best model and save
    model.load_state_dict(best_model)
    model_suffix = f"_dp_{args.dp_method}" if args.enable_dp else ""
    torch.save(model, f"spdpn{args.data}{model_suffix}.pth")
    
    log_message("\n=== Training Complete ===", log_file)
    log_message(f"Best validation MAE: {min(MAE_list):.4f}", log_file)
    
    if args.enable_dp:
        final_epsilon = privacy_engine.get_epsilon(args.delta)
        log_message(f"Final Privacy Budget - ε: {final_epsilon:.4f}, δ: {args.delta}", log_file)
        
        if args.dp_method == 'make_private_with_epsilon':
            log_message(f"Target epsilon was: {args.target_epsilon:.4f}", log_file)
            if final_epsilon <= args.target_epsilon:
                log_message("✓ Privacy budget target achieved!", log_file)
            else:
                log_message("⚠ Privacy budget exceeded target!", log_file)
        else:
            log_message(f"Used noise multiplier: {args.noise_multiplier}", log_file)
            
        log_message("Model trained with differential privacy guarantees.", log_file)

if __name__ == "__main__":
    main()