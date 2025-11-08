# training_u.py
# -*- coding: utf-8 -*-


from statsmodels.tsa.vector_ar import output
import torch
import util
import argparse
import random
import copy
import torch.optim as optim
import numpy as np
from datetime import datetime
import os
import csv

from baseline_methods import test_error, StandardScaler
from model import STPN
import torch.nn as nn

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
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--wemb_size',type=int,default=4,help='covairate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=8,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=False,help="use SE block")
parser.add_argument('--use_cov', type=str, default=False,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.0005, help='learning rate ')
parser.add_argument('--in_len',type=int,default=12,help='input time series length')
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')

# === NEW: DP hyperparameters ===
parser.add_argument('--dp', default=True,  action='store_true', help='enable differential privacy with Opacus')
parser.add_argument('--target_epsilon', type=float, default=-5.0, help='if >0, use make_private_with_epsilon')
parser.add_argument('--target_delta', type=float, default=1e-5, help='delta for DP accounting')
parser.add_argument('--noise_multiplier', type=float, default=1.5, help='sigma; used if target_epsilon <= 0')
parser.add_argument('--max_grad_norm', type=float, default=1.5, help='per-sample gradient clipping norm')

args = parser.parse_args()


def setup_logging(args):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'training_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    return log_file
def log_message(message, log_file):
    """Print message to console and append to log file"""
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x, ti=None, supports=None, to=None, w=None):
        # Ignore additional args for this test model
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
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
        x = self.data[:, s:s + self.in_len, :]              # (N, L_in, F)
        y = self.data[:, s + self.in_len:s + self.in_len + self.out_len, :]  # (N, L_out, F)
        w = self.weather[:, s:s + self.in_len]              # (N, L_in)

        ti = (np.arange(s, s + self.in_len) % self.period) * np.ones([1, self.in_len]) / (self.period - 1)
        to = (np.arange(s + self.in_len, s + self.in_len + self.out_len) % self.period) * np.ones([1, self.out_len]) / (self.period - 1)

        x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)  # (F, N, L_in)
        y = torch.tensor(y, dtype=torch.float32).permute(2, 0, 1)  # (F, N, L_out)
        w = torch.tensor(w, dtype=torch.long)                      # (N, L_in)
        ti = torch.tensor(ti, dtype=torch.float32)                 # (1, L_in)
        to = torch.tensor(to, dtype=torch.float32)                 # (1, L_out)
        return x, y, ti, to, w
def debug_tensor_shapes(obj, prefix=''):
    try:
        if isinstance(obj, torch.Tensor):
            print(f"{prefix} Tensor shape: {obj.shape}, numel: {obj.numel()}")
        elif isinstance(obj, (list, tuple)):
            print(f"{prefix} List/Tuple of length {len(obj)}")
            for i, item in enumerate(obj):
                debug_tensor_shapes(item, prefix=f"{prefix}[{i}]")
        elif isinstance(obj, dict):
            print(f"{prefix} Dict with keys: {list(obj.keys())}")
            for k, v in obj.items():
                debug_tensor_shapes(v, prefix=f"{prefix}[{k}]")
        else:
            print(f"{prefix} Unknown type: {type(obj)}")
    except Exception as e:
        print(f"{prefix} Exception when printing: {e}")


def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)

    # For debugging DP issues, try a simple CNN instead
    # model = SimpleCNN(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    model = STPN(
        args.h_layers, args.in_channels, args.hidden_channels, args.out_channels, args.emb_size,
        args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
        args.order, args.num_weather, args.use_se, args.use_cov
    ).to(device)

    supports = [torch.tensor(i, dtype=torch.float32, device=device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                            training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0

    

    train_indices = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    dataset = TrainWindowDataset(
        data=training_data,
        weather=training_w,
        period=args.period,
        in_len=args.in_len,
        out_len=args.out_len,
        indices=train_indices
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batch size: {args.batch}")

    if args.dp:
        print("Setting up DataLoader for Differential Privacy...")
        sample_rate = args.batch / len(dataset)
        print(f"Sample rate: {sample_rate}")
        batch_sampler = UniformWithReplacementSampler(
            num_samples=len(dataset),
            sample_rate=sample_rate,
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(dataset, batch_sampler=batch_sampler)
        print(f"DP DataLoader created with batch_sampler")
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
        print(f"Standard DataLoader created with batch_size={args.batch}")

    # Test the first batch to check shapes
    print("\n=== Testing first batch ===")
    test_batch = next(iter(train_loader))
    print(f"First batch shapes:")
    for i, tensor in enumerate(test_batch):
        print(f"  Tensor {i}: {tensor.shape if hasattr(tensor, 'shape') else type(tensor)}")
    
    # Reset the DataLoader
    if args.dp:
        train_loader = DataLoader(dataset, batch_sampler=batch_sampler)
    else:
        train_loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)

    if args.dp:
        print("\n=== Setting up Differential Privacy ===")
        print(f"Model parameters before DP:")
        param_count = 0
        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
            param_count += param.numel()
        print(f"Total parameters: {param_count:,}")
        
        privacy_engine = PrivacyEngine()
        
        try:
            if args.target_epsilon > 0:
                print(f"Making private with target epsilon: {args.target_epsilon}")
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
                csv_filename = f'training_ε={args.target_epsilon}_max_grad_norm={args.max_grad_norm}.csv'
            else:
                print(f"Making private with noise multiplier: {args.noise_multiplier}")
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=args.noise_multiplier,
                    max_grad_norm=args.max_grad_norm,
                )
                print(f"[DP] Using noise_multiplier={args.noise_multiplier}, max_grad_norm={args.max_grad_norm}")
                csv_filename = f'noise_multiplier={args.noise_multiplier}_max_grad_norm={args.max_grad_norm}.csv'
            
            print("Differential Privacy setup completed successfully")
            
        except Exception as e:
            print(f"ERROR during DP setup: {e}")
            print(f"Error type: {type(e)}")
            raise
    else:
        privacy_engine = None
        csv_filename = 'training_no_dp.csv'

    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)

    print("start training...", flush=True)

    # Initialize CSV file for logging training metrics
    #timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if 'csv_filename' not in locals():
        csv_filename = 'training_default.csv'
    
    csv_headers = ['epoch', 'training_loss', 'validation_loss', 'validation_mae']
    
    print(f"Creating CSV file: {csv_filename}")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(csv_headers)
    
    print(f"Training metrics will be saved to: {csv_filename}")

    MAE_list = []
    best_model = copy.deepcopy(model.state_dict())

    

    for ep in range(1, 1 + args.episode):
        
        # Initialize epoch training loss tracking
        epoch_training_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            trainx, trainy, trainti, trainto, trainw = batch
            
            # Limit debug output to first 3 batches to avoid spam
            debug_this_batch = batch_idx < 3
            
            if debug_this_batch:
                print(f"\n=== Debug: Epoch {ep}, Batch {batch_idx + 1} ===")
                print(f"Original shapes:")
                print(f"  trainx: {trainx.shape}")
                print(f"  trainy: {trainy.shape}")
                print(f"  trainti: {trainti.shape}")
                print(f"  trainto: {trainto.shape}")
                print(f"  trainw: {trainw.shape}")
            
            trainx = trainx.to(device)
            trainy = trainy.to(device)
            trainti = trainti.to(device)
            trainto = trainto.to(device)
            trainw = trainw.to(device)

            if trainx.dim() == 3:
                if debug_this_batch:
                    print("Expanding dimensions...")
                trainx = trainx.unsqueeze(0)
                trainy = trainy.unsqueeze(0)
                trainti = trainti.unsqueeze(0)
                trainto = trainto.unsqueeze(0)
                trainw = trainw.unsqueeze(0)
                if debug_this_batch:
                    print(f"After unsqueeze:")
                    print(f"  trainx: {trainx.shape}")
                    print(f"  trainy: {trainy.shape}")

            model.train()

            # for name, param in model.named_parameters():
            #     print(f"{name} requires_grad: {param.requires_grad}")

            
            optimizer.zero_grad(set_to_none=True)
            if debug_this_batch:
                print(f"Forward pass...")
            output = model(trainx, trainti, supports, trainto, trainw)
            if debug_this_batch:
                print(f"Model output shape: {output.shape}")
            
            # print(f"Before fix - Output: {output.shape}, Target: {trainy.shape}")
            if output.shape != trainy.shape:
                if debug_this_batch:
                    print(f"Shape mismatch detected - Output: {output.shape}, Target: {trainy.shape}")
                if output.shape[-1] == 1 and trainy.shape[-1] > 1:
                    # Model predicts 1 step, target has 12 steps - take last step of target
                    trainy = trainy[..., -1:]
                    if debug_this_batch:
                        print(f"Adjusted target shape to: {trainy.shape}")

            # print(f"After fix - Output: {output.shape}, Target: {trainy.shape}")
            #loss = torch.nn.MSELoss()(output, trainy)
            if debug_this_batch:
                print(f"Computing loss...")
            loss = util.masked_rmse(output, trainy, 0.0)
            if debug_this_batch:
                print(f"Loss value: {loss.item()}")
            
            if debug_this_batch:
                print(f"Backward pass...")
            loss.backward()
            
            # print("=== Debug data shapes ===")
            # debug_tensor_shapes(trainx, prefix="trainx")
            # debug_tensor_shapes(trainti, prefix="trainti")
            # debug_tensor_shapes(trainto, prefix="trainto")
            # debug_tensor_shapes(trainw, prefix="trainw")

            # print("=== Debug model output ===")
            # debug_tensor_shapes(output, prefix="output")

            # print("=== Debug target shapes ===")
            # debug_tensor_shapes(trainy, prefix="trainy")
            

            # Debug: Check gradient shapes before optimizer.step()
            if debug_this_batch:
                print(f"\n=== Debug: Before optimizer.step() ===")
                print(f"Batch size: {trainx.shape[0] if trainx.dim() > 0 else 'unknown'}")
                print(f"Loss value: {loss.item()}")
                
                # Check regular gradients
                print("\n--- Regular gradients ---")
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        print(f"{name}: grad.shape={param.grad.shape}, param.shape={param.shape}")
                    elif param.requires_grad:
                        print(f"{name}: grad=None, param.shape={param.shape}")
                
                # Debug per-sample gradients for Opacus
                if args.dp:
                    print("\n--- Per-sample gradients (Opacus) ---")
                    missing_per_sample_grad = []
                    mismatched_shapes = []
                    
                    for name, param in model.named_parameters():
                        if param.requires_grad:
                            if not hasattr(param, "grad_sample") or param.grad_sample is None:
                                missing_per_sample_grad.append(name)
                            else:
                                grad_sample_shape = param.grad_sample.shape
                                expected_shape = (trainx.shape[0],) + param.shape
                                print(f"{name}: grad_sample.shape={grad_sample_shape}, expected={expected_shape}")
                                if grad_sample_shape != expected_shape:
                                    mismatched_shapes.append((name, grad_sample_shape, expected_shape))

                    if missing_per_sample_grad:
                        print("Parameters missing per-sample gradients:")
                        for name in missing_per_sample_grad:
                            print(f" - {name}")
                    
                    if mismatched_shapes:
                        print("Parameters with mismatched per-sample gradient shapes:")
                        for name, actual, expected in mismatched_shapes:
                            print(f" - {name}: actual={actual}, expected={expected}")
                    
                    if not missing_per_sample_grad and not mismatched_shapes:
                        print("All parameters have correctly shaped per-sample gradients.")

            if not args.dp:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)

            if debug_this_batch:
                print(f"\n=== Debug: About to call optimizer.step() ===")
            try:
                optimizer.step()
                if debug_this_batch:
                    print("optimizer.step() completed successfully")
            except Exception as e:
                print(f"ERROR in optimizer.step(): {e}")
                print(f"Error type: {type(e)}")
                
                # Additional debug for the specific error
                if "stack expects each tensor to be equal size" in str(e):
                    print("\n--- Debugging tensor stacking issue ---")
                    if args.dp:
                        print("Checking grad_sample tensors that might be stacked:")
                        for name, param in model.named_parameters():
                            if param.requires_grad and hasattr(param, "grad_sample") and param.grad_sample is not None:
                                print(f"{name}: grad_sample.shape={param.grad_sample.shape}, numel={param.grad_sample.numel()}")
                
                # Re-raise the exception to see the full traceback
                raise Exception("Error occurred during optimizer.step()")
            
            # Accumulate training loss for this epoch
            epoch_training_loss += loss.item()
            num_batches += 1
            

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

        # Write metrics to CSV file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([ep, avg_training_loss, avg_validation_loss, mean_mae])

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

    model.load_state_dict(best_model)
    torch.save(model.state_dict(), "spdpn" + args.data + ".pth")
    print("Saved weights to:", "spdpn" + args.data + ".pth")

if __name__ == "__main__":
    main()
