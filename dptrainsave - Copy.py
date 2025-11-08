# training_u.py

# -*- coding: utf-8 -*-

import torch
import util
import argparse
import random
import copy
import torch.optim as optim
import numpy as np
from baseline_methods import test_error, StandardScaler
from model import STPN
import csv

# === NEW: opacus for DP ===
from opacus import PrivacyEngine
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
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate for DP training')
parser.add_argument('--wemb_size',type=int,default=4,help='covairate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=8,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate ')
parser.add_argument('--in_len',type=int,default=12,help='input time series length')
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')
# === NEW: DP hyperparameters ===
parser.add_argument('--dp', default=1, action='store_true', help='enable differential privacy with Opacus')
parser.add_argument('--target_epsilon', type=float, default=15.0, help='if >0, use make_private_with_epsilon')
parser.add_argument('--target_delta', type=float, default=1e-5, help='delta for DP accounting')
parser.add_argument('--noise_multiplier', type=float, default=0.5, help='sigma; used if target_epsilon <= 0')
parser.add_argument('--max_grad_norm', type=float, default=1.5, help='per-sample gradient clipping norm')
# resume
#parser.add_argument('--resume', type=str, default="spdpnUS_ep4_MAE21.7426.pth", help='path to checkpoint to resume from')
#parser.add_argument('--start_epoch', type=int, default=1, help='epoch to start from')
args = parser.parse_args()

class TrainWindowDataset(Dataset):
    def __init__(self, data, weather, period, in_len, out_len, indices):
        self.data = data
        self.weather = weather
        self.period = period
        self.in_len = in_len
        self.out_len = out_len
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        s = self.indices[idx]
        x = self.data[:, s:s + self.in_len, :] # (N, L_in, F)
        y = self.data[:, s + self.in_len:s + self.in_len + self.out_len, :] # (N, L_out, F)
        w = self.weather[:, s:s + self.in_len] # (N, L_in)
        ti = (np.arange(s, s + self.in_len) % self.period) * np.ones([1, self.in_len]) / (self.period - 1)
        to = (np.arange(s + self.in_len, s + self.in_len + self.out_len) % self.period) * np.ones([1, self.out_len]) / (self.period - 1)
        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1) # (F, N, L_in)
        y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1) # (F, N, L_out)
        w = torch.tensor(w, dtype=torch.long) # (N, L_in)
        ti = torch.tensor(ti, dtype=torch.float32) # (1, L_in)
        to = torch.tensor(to, dtype=torch.float32) # (1, L_out)
        return x, y, ti, to, w
    def check_model_parameters(model, epoch, prefix=""):
        """Debug function to check model parameters"""
        stats = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                param_norm = param.data.norm(2).item()
                stats[name] = {
                    'param_norm': param_norm,
                    'grad_norm': grad_norm,
                    'param_mean': param.data.mean().item(),
                    'grad_mean': param.grad.data.mean().item() if param.grad is not None else 0
                }
        
        # Print top 5 largest gradient norms
        if stats:
            sorted_grads = sorted(stats.items(), key=lambda x: x[1]['grad_norm'], reverse=True)[:5]
            print(f"\n{prefix} Epoch {epoch} - Top 5 gradient norms:")
            for name, stat in sorted_grads:
                print(f"  {name}: grad_norm={stat['grad_norm']:.4f}, param_norm={stat['param_norm']:.4f}")
        
        return stats

def regularized_loss(output, target, model, l2_lambda=0.01):
    base_loss = util.masked_rmse(output, target, 0.0)
    l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
    return base_loss + l2_lambda * l2_reg

def compute_detailed_metrics(pred, true, threshold=1e-6):
    """Compute robust metrics with validity tracking"""
    # Convert numpy arrays to tensors if needed
    if isinstance(pred, np.ndarray):
        pred = torch.from_numpy(pred)
    if isinstance(true, np.ndarray):
        true = torch.from_numpy(true)
        
    abs_error = torch.abs(pred - true)
    valid_mask = ~(torch.isnan(abs_error) | torch.isinf(abs_error))
    
    if valid_mask.sum() == 0:
        return float('inf'), float('inf'), 0.0
    
    mae = torch.mean(abs_error[valid_mask])
    rmse = torch.sqrt(torch.mean((pred - true)[valid_mask] ** 2))
    
    return float(mae), float(rmse), valid_mask.float().mean()

def analyze_gradients(model, epoch):
    total_norm = 0
    param_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
    
    total_norm = total_norm ** (1. / 2)
    avg_norm = total_norm / param_count if param_count > 0 else 0
    
    if total_norm > 10 or avg_norm > 1:
        print(f"Epoch {epoch}: High gradient norm detected - Total: {total_norm:.4f}, Avg: {avg_norm:.4f}")
    
    return total_norm, avg_norm

def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)

    # ========== DEBUG: Check data statistics ==========
    print("\n=== DATA STATISTICS ===")
    print(f"Training data shape: {training_data.shape}")
    print(f"Training data range: [{np.nanmin(training_data):.4f}, {np.nanmax(training_data):.4f}]")
    print(f"Training data mean: {np.nanmean(training_data):.4f}, std: {np.nanstd(training_data):.4f}")
    print(f"NaN count in training data: {np.isnan(training_data).sum()}")

    model = STPN(
        args.h_layers, args.in_channels, args.hidden_channels, args.out_channels, args.emb_size,
        args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
        args.order, args.num_weather, args.use_se, args.use_cov
    ).to(device)
    
    # Load checkpoint if resuming
    #if args.resume:
        #print(f"Loading checkpoint: {args.resume}")
        #model.load_state_dict(torch.load(args.resume))
        #print("Checkpoint loaded successfully")
    
    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3
    )
    print("Learning rate scheduler initialized")
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                           training_data[~np.isnan(training_data)].std())
    
    # Apply consistent normalization to all datasets
    print("\n=== Applying normalization and NaN handling ===")
    
    # Training data
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    print(f"Training data - range: [{training_data.min():.4f}, {training_data.max():.4f}]")
    
    # Validation data
    val_data = scaler.transform(val_data)
    val_data[np.isnan(val_data)] = 0
    print(f"Validation data - range: [{val_data.min():.4f}, {val_data.max():.4f}]")
    
    # Test data
    test_data = scaler.transform(test_data)
    test_data[np.isnan(test_data)] = 0
    print(f"Test data - range: [{test_data.min():.4f}, {test_data.max():.4f}]")

    train_indices = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    dataset = TrainWindowDataset(
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
        train_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    if args.dp:
        privacy_engine = PrivacyEngine()
        if args.target_epsilon and args.target_epsilon > 0:
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=args.target_epsilon,
                target_delta=args.target_delta,
                epochs=args.episode,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"[DP] Using target ε={args.target_epsilon}, δ={args.target_delta}, max_grad_norm={args.max_grad_norm}")
        else:
            model, optimizer, train_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=args.noise_multiplier,
                max_grad_norm=args.max_grad_norm,
            )
            print(f"[DP] Using noise_multiplier={args.noise_multiplier}, max_grad_norm={args.max_grad_norm}")
    else:
        privacy_engine = None

 # ========== Prepare validation data ==========
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)

    print("start training...", flush=True)
    best_mae = float("inf")

    # Store per epoch stats
    metrics_log = []
    epoch_losses = []  # New code tracks epoch losses differently

    for ep in range(1, 1 + args.episode):
        model.train()
        train_maes = []
        train_rmses = []
        train_valid_ratios = []
        for batch_idx, batch in enumerate(train_loader):
            trainx, trainy, trainti, trainto, trainw = batch


            trainx = trainx.to(device)
            trainy = trainy.to(device)
            trainti = trainti.to(device)
            trainto = trainto.to(device)
            trainw = trainw.to(device)
            if trainx.dim() == 3:
                trainx = trainx.unsqueeze(0)
                trainy = trainy.unsqueeze(0)
                trainti = trainti.unsqueeze(0)
                trainto = trainto.unsqueeze(0)
                trainw = trainw.unsqueeze(0)
            optimizer.zero_grad(set_to_none=True)
            output = model(trainx, trainti, supports, trainto, trainw)
            loss = regularized_loss(output, trainy, model) if args.dp else util.masked_rmse(output, trainy, 0.0)
            
            # Track training metrics
            with torch.no_grad():
                train_mae, train_rmse, train_valid = compute_detailed_metrics(output, trainy)
                train_maes.append(train_mae)
                train_rmses.append(train_rmse)
                train_valid_ratios.append(train_valid)
            
            loss.backward()
            
            # Analyze gradients before optimizer step
            total_norm, avg_norm = analyze_gradients(model, ep)
            
            # Skip update if gradients are too large
            if total_norm > 100:
                print(f"WARNING: Skipping update due to large gradient norm")
                optimizer.zero_grad(set_to_none=True)
                continue
                
            if not args.dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()

            # New code: track loss for this epoch
            epoch_losses.append(loss.item())

            # Check for NaN/Inf in output
            if torch.isnan(output).any() or torch.isinf(output).any():
                print(f"WARNING: NaN/Inf in output at batch {batch_idx}")
                continue

            # Safe gradient norm calculation
            try:
                # Only consider parameters that have gradients
                grad_norms = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norms.append(torch.norm(p.grad))
                
                if grad_norms:  # Only compute total norm if we have gradients
                    total_norm = torch.norm(torch.stack(grad_norms), p=2).item()
                    if total_norm > 100:
                        print(f"WARNING: Large gradient norm {total_norm:.2f}")
                        continue
            except Exception as e:
                print(f"Warning: Error computing gradient norms: {e}")
                continue

        # Replace your validation section end with this:
        model.eval()
        outputs = []
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
                out = out.permute(0, 2, 3, 1)
                out = out.detach().cpu().numpy()
                
                # Consistent NaN handling before inverse transform
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                out = scaler.inverse_transform(out)
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Additional bounds check after processing
                out = np.clip(out, -1e6, 1e6)  # Reasonable bounds for your data
                outputs.append(out)

        yhat = np.concatenate(outputs)

        # Add validation sanity checks
        print("\n=== Validation Batch Statistics ===")
        for i in range(12):
            pred_slice = yhat[:, :, i, :]
            true_slice = label[:, :, i, :]
            
            print(f"\nTimestep {i}:")
            print(f"Val pred min: {np.nanmin(pred_slice):.4f}, max: {np.nanmax(pred_slice):.4f}, mean: {np.nanmean(pred_slice):.4f}")
            print(f"Val true min: {np.nanmin(true_slice):.4f}, max: {np.nanmax(true_slice):.4f}, mean: {np.nanmean(true_slice):.4f}")
            print(f"Non-NaN count pred: {np.isfinite(pred_slice).sum()}, Non-NaN count true: {np.isfinite(true_slice).sum()}")
            print(f"Output shape: {pred_slice.shape}, Label shape: {true_slice.shape}")

        # ROBUST metrics calculation that ensures MAE >= 0
        amae, ar2, armse = [], [], []
        for i in range(12):
            pred_slice = yhat[:, :, i, :]
            true_slice = label[:, :, i, :]
            
            # Ensure no invalid values
            pred_slice = np.nan_to_num(pred_slice, nan=0.0, posinf=0.0, neginf=0.0)
            true_slice = np.nan_to_num(true_slice, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Manual robust calculation
            err = pred_slice - true_slice
            mae_val = np.mean(np.abs(err))  # This MUST be >= 0
            rmse_val = np.sqrt(np.mean(err ** 2))  # This MUST be >= 0
            
            # R² calculation
            ss_res = np.sum((true_slice - pred_slice) ** 2)
            ss_tot = np.sum((true_slice - np.mean(true_slice)) ** 2)
            r2_val = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0
            
            # Ensure non-negative (safety check)
            mae_val = max(0.0, mae_val)
            rmse_val = max(0.0, rmse_val)
            
            amae.append(mae_val)
            armse.append(rmse_val)
            ar2.append(r2_val)

        mean_mae = float(np.mean(amae))
        mean_r2 = float(np.mean(ar2))
        mean_rmse = float(np.mean(armse))

        # Final safety check
        mean_mae = max(0.0, mean_mae)
        mean_rmse = max(0.0, mean_rmse)


        if args.dp:
            try:
                eps = privacy_engine.get_epsilon(args.target_delta)
                print(f"Epoch {ep:03d} | Loss {loss.item():.4f} | Val MAE {mean_mae:.4f} R2 {mean_r2:.4f} RMSE {mean_rmse:.4f} | ε ~ {eps:.2f}, δ={args.target_delta}")
            except Exception:
                print(f"Epoch {ep:03d} | Loss {loss.item():.4f} | Val MAE {mean_mae:.4f} R2 {mean_r2:.4f} RMSE {mean_rmse:.4f}")
        else:
            print(f"Epoch {ep:03d} | Loss {loss.item():.4f} | Val MAE {mean_mae:.4f} R2 {mean_r2:.4f} RMSE {mean_rmse:.4f}")

        # Save every time new minimum is reached
        if mean_mae >= 0 and mean_mae < best_mae:
            best_mae = mean_mae
            torch.save(model.state_dict(), f"spdpn{args.data}_ep{ep}_MAE{mean_mae:.4f}.pth")
            print(f"[Checkpoint] New minimum MAE {mean_mae:.4f} at epoch {ep}, model saved.")
        

        # Calculate average training metrics
        avg_train_mae = np.mean(train_maes)
        avg_train_rmse = np.mean(train_rmses)
        avg_train_valid = np.mean(train_valid_ratios)
        
        # Calculate overfitting score
        overfitting_score = mean_mae / (avg_train_mae + 1e-8)  # Prevent division by zero
        
        print(f"Training MAE: {avg_train_mae:.4f}, Valid ratio: {avg_train_valid:.2%}")
        print(f"Overfitting score: {overfitting_score:.2f} (closer to 1 is better)")

        # Accumulate stats for saving
        log_row = {
            'epoch': ep, 
            'loss': float(loss), 
            'mae': mean_mae, 
            'r2': mean_r2, 
            'rmse': mean_rmse,
            'train_mae': avg_train_mae,
            'train_rmse': avg_train_rmse,
            'train_valid_ratio': avg_train_valid,
            'overfitting_score': overfitting_score
        }
        
        if args.dp:
            # Note: eps may only be defined if DP is active and privacy_engine available
            log_row['epsilon'] = float(eps) if 'eps' in locals() else None
            log_row['delta'] = args.target_delta
            
        metrics_log.append(log_row)

        # Update learning rate based on validation MAE
        scheduler.step(mean_mae)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr:.6f}")

    # Save metrics to external file
    csv_name = f'training_metrics_{args.data}dp.csv' if args.dp else f'training_metrics{args.data}.csv'
    fieldnames = metrics_log[0].keys()
    with open(csv_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_log)
    print(f"All epoch metrics saved to: {csv_name}")

if __name__ == "__main__":
    main()