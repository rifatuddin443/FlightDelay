"""Benchmark with STPN

Simple benchmark that includes STPN model along with basic baselines.
No differential privacy requirements.

Usage:
    python benchmark_with_stpn.py --quick_test
    python benchmark_with_stpn.py --epochs 30
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from baseline_methods import test_error

# Try to import STPN
try:
    from model import STPN
    STPN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] STPN not available: {e}")
    STPN = None
    STPN_AVAILABLE = False

# Configure matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# SIMPLE MODELS
# ============================================================================

class SimpleLSTM(nn.Module):
    """Simple LSTM for flight delay prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class SimpleGRU(nn.Module):
    """Simple GRU for flight delay prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        return self.fc(h_n[-1])


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data_for_stpn(data_dir='cdata', test_ratio=0.2, in_len=12, out_len=3):
    """Load flight delay data for STPN format."""
    print(f"\n[Data] Loading from {data_dir}...")
    
    # Load delay data
    delay_file = os.path.join(data_dir, 'delay.npy')
    if not os.path.exists(delay_file):
        delay_file = os.path.join(data_dir, 'udelay.npy')
    
    delay_data = np.load(delay_file)  # [nodes, time, features]
    print(f"[Data] Delay shape: {delay_data.shape}")
    
    # Load adjacency matrix
    adj_file = os.path.join(data_dir, 'dist_mx.npy')
    if not os.path.exists(adj_file):
        adj_file = os.path.join(data_dir, 'adj_mx.npy')
    
    if os.path.exists(adj_file):
        adj_mx = np.load(adj_file)
        print(f"[Data] Adjacency shape: {adj_mx.shape}")
    else:
        # Create identity matrix as fallback
        adj_mx = np.eye(delay_data.shape[0])
        print(f"[Data] Using identity matrix for adjacency")
    
    # Load OD matrix
    od_file = os.path.join(data_dir, 'od_mx.npy')
    if not os.path.exists(od_file):
        od_file = os.path.join(data_dir, 'od_pair.npy')
    
    if os.path.exists(od_file):
        od_mx = np.load(od_file)
        print(f"[Data] OD matrix shape: {od_mx.shape}")
    else:
        od_mx = adj_mx.copy()
        print(f"[Data] Using adjacency as OD matrix")
    
    num_nodes, total_time, num_features = delay_data.shape
    
    # Replace NaN with 0
    delay_data = np.nan_to_num(delay_data)
    
    # Normalize
    mean = delay_data.mean()
    std = delay_data.std()
    delay_data_norm = (delay_data - mean) / (std + 1e-8)
    
    # Create STPN-format data
    # STPN expects: (batch, channels, nodes, time)
    max_samples = total_time - in_len - out_len
    n_train = int(max_samples * (1 - test_ratio))
    
    # For STPN format
    stpn_x_train = []
    stpn_y_train = []
    stpn_x_test = []
    stpn_y_test = []
    
    # For simple models (flatten)
    simple_x_train = []
    simple_y_train = []
    simple_x_test = []
    simple_y_test = []
    
    for t in range(max_samples):
        # STPN format: (channels, nodes, time)
        x_sample = delay_data_norm[:, t:t+in_len, :].transpose(2, 0, 1)  # [features, nodes, time]
        y_sample = delay_data[:, t+in_len:t+in_len+out_len, :].transpose(2, 0, 1)  # [features, nodes, time]
        
        # Simple format: flatten for LSTM/GRU
        x_simple = delay_data_norm[:, t:t+in_len, :].reshape(-1, in_len * num_features)  # [nodes, time*features]
        y_simple = delay_data[:, t+in_len:t+in_len+out_len, :].mean(axis=(1, 2))  # [nodes] - average over time and features
        
        if t < n_train:
            stpn_x_train.append(x_sample)
            stpn_y_train.append(y_sample)
            simple_x_train.extend(x_simple)
            simple_y_train.extend(y_simple)
        else:
            stpn_x_test.append(x_sample)
            stpn_y_test.append(y_sample)
            simple_x_test.extend(x_simple)
            simple_y_test.extend(y_simple)
    
    # Convert to numpy
    stpn_x_train = np.array(stpn_x_train)
    stpn_y_train = np.array(stpn_y_train)
    stpn_x_test = np.array(stpn_x_test)
    stpn_y_test = np.array(stpn_y_test)
    
    simple_x_train = np.array(simple_x_train).reshape(-1, in_len, num_features)
    simple_y_train = np.array(simple_y_train).reshape(-1, 1)
    simple_x_test = np.array(simple_x_test).reshape(-1, in_len, num_features)
    simple_y_test = np.array(simple_y_test).reshape(-1, 1)
    
    # Create support matrices for STPN (random walk normalization)
    def normalize_adj(mx):
        """Convert adjacency to random walk matrix."""
        mx = mx + np.eye(mx.shape[0])  # Add self-loops
        rowsum = mx.sum(axis=1).clip(min=1.0)
        return mx / rowsum[:, np.newaxis]
    
    supports = [
        torch.FloatTensor(normalize_adj(adj_mx)),
        torch.FloatTensor(normalize_adj(od_mx)),
        torch.FloatTensor(normalize_adj(od_mx.T))
    ]
    
    print(f"[Data] STPN Train: {stpn_x_train.shape}, Test: {stpn_x_test.shape}")
    print(f"[Data] Simple Train: {simple_x_train.shape}, Test: {simple_x_test.shape}")
    
    return {
        'stpn_x_train': stpn_x_train,
        'stpn_y_train': stpn_y_train,
        'stpn_x_test': stpn_x_test,
        'stpn_y_test': stpn_y_test,
        'simple_x_train': simple_x_train,
        'simple_y_train': simple_y_train,
        'simple_x_test': simple_x_test,
        'simple_y_test': simple_y_test,
        'supports': supports,
        'num_nodes': num_nodes,
        'in_len': in_len,
        'out_len': out_len,
        'mean': mean,
        'std': std
    }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkWithSTPNRunner:
    """Benchmark runner with STPN."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"stpn_benchmark_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"{'BENCHMARK WITH STPN':^80}")
        print(f"{'='*80}")
        print(f"\nDevice: {self.device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"STPN Available: {STPN_AVAILABLE}")
        print(f"Output: {self.output_dir}\n")
    
    def clear_gpu_memory(self):
        """Clear GPU memory between models."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def run_historical_average(self, y_train, y_test):
        """Historical average baseline."""
        print("\n[1/4] Running Historical Average...")
        start = time.time()
        
        y_pred = np.full_like(y_test, y_train.mean())
        mae, rmse, r2 = test_error(y_pred, y_test)
        elapsed = time.time() - start
        
        self.results['Historical_Average'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': 0, 'inference_time': elapsed
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        self.clear_gpu_memory()
    
    def run_lstm(self, X_train, y_train, X_test, y_test):
        """LSTM baseline."""
        print("\n[2/4] Running LSTM...")
        
        model = SimpleLSTM(X_train.shape[-1], hidden_dim=64, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=self.args.batch_size, shuffle=True
        )
        
        # Train
        start = time.time()
        model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start
        
        # Evaluate in batches to save memory
        model.eval()
        start = time.time()
        y_pred = []
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test)),
            batch_size=32, shuffle=False
        )
        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                batch_pred = model(X_batch).cpu().numpy()
                y_pred.append(batch_pred)
        y_pred = np.vstack(y_pred)
        inference_time = time.time() - start
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['LSTM'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': training_time, 'inference_time': inference_time
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
        
        # Clean up
        del model, optimizer, train_loader, test_loader
        self.clear_gpu_memory()
    
    def run_gru(self, X_train, y_train, X_test, y_test):
        """GRU baseline."""
        print("\n[3/4] Running GRU...")
        
        model = SimpleGRU(X_train.shape[-1], hidden_dim=64, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=self.args.batch_size, shuffle=True
        )
        
        # Train
        start = time.time()
        model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start
        
        # Evaluate in batches to save memory
        model.eval()
        start = time.time()
        y_pred = []
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test)),
            batch_size=32, shuffle=False
        )
        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                batch_pred = model(X_batch).cpu().numpy()
                y_pred.append(batch_pred)
        y_pred = np.vstack(y_pred)
        inference_time = time.time() - start
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['GRU'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': training_time, 'inference_time': inference_time
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
        
        # Clean up
        del model, optimizer, train_loader, test_loader
        self.clear_gpu_memory()
    
    def run_stpn(self, data):
        """Run STPN model."""
        if not STPN_AVAILABLE:
            print("\n[4/4] STPN: Skipped (not available)")
            return
        
        print("\n[4/4] Running STPN...")
        
        try:
            # Prepare data - keep on CPU, move batches to GPU
            X_train_cpu = data['stpn_x_train']
            y_train_cpu = data['stpn_y_train']
            y_test_np = data['stpn_y_test']
            
            supports = [s.to(self.device) for s in data['supports']]
            num_nodes = data['num_nodes']
            in_len = data['in_len']
            out_len = data['out_len']
            
            # Create STPN model
            model = STPN(
                h_layers=2,
                in_channels=X_train_cpu.shape[1],  # Number of features
                hidden_channels=[32, 32, 16],
                out_channels=X_train_cpu.shape[1],
                emb_size=16,
                dropout=0.2,
                wemb_size=4,
                time_d=4,
                heads=4,
                support_len=3,
                order=2,
                num_weather=8,
                use_se=False,
                use_cov=False  # Disable weather for simplicity
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            criterion = nn.MSELoss()
            
            # Train with smaller batches
            start = time.time()
            model.train()
            batch_size = min(8, X_train_cpu.shape[0])  # Use small batches to save memory
            w_type = None
            
            for epoch in range(self.args.epochs):
                total_loss = 0
                num_batches = 0
                
                for i in range(0, X_train_cpu.shape[0], batch_size):
                    # Move batch to GPU
                    batch_x = torch.FloatTensor(X_train_cpu[i:i+batch_size]).to(self.device)
                    batch_y = torch.FloatTensor(y_train_cpu[i:i+batch_size]).to(self.device)
                    batch_t_in = torch.arange(in_len).unsqueeze(0).repeat(batch_x.shape[0], 1).to(self.device)
                    batch_t_out = torch.arange(out_len).unsqueeze(0).repeat(batch_x.shape[0], 1).to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_x, batch_t_in, supports, batch_t_out, w_type)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clean up batch from GPU
                    del batch_x, batch_y, batch_t_in, batch_t_out, outputs, loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
            
            training_time = time.time() - start
            
            # Evaluate in batches to save memory
            model.eval()
            start = time.time()
            
            y_pred_list = []
            eval_batch_size = 8  # Smaller batch for evaluation
            X_test_cpu = data['stpn_x_test']
            
            with torch.no_grad():
                for i in range(0, X_test_cpu.shape[0], eval_batch_size):
                    batch_x = torch.FloatTensor(X_test_cpu[i:i+eval_batch_size]).to(self.device)
                    batch_t_in = torch.arange(in_len).unsqueeze(0).repeat(batch_x.shape[0], 1).to(self.device)
                    batch_t_out = torch.arange(out_len).unsqueeze(0).repeat(batch_x.shape[0], 1).to(self.device)
                    
                    batch_pred = model(batch_x, batch_t_in, supports, batch_t_out, w_type).cpu().numpy()
                    y_pred_list.append(batch_pred)
                    
                    # Clear batch from GPU
                    del batch_x, batch_t_in, batch_t_out, batch_pred
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            y_pred = np.concatenate(y_pred_list, axis=0)
            inference_time = time.time() - start
            
            # Reshape for evaluation (average over nodes, time, features)
            y_pred_flat = y_pred.reshape(-1, 1)
            y_test_flat = y_test_np.reshape(-1, 1)
            
            mae, rmse, r2 = test_error(y_pred_flat, y_test_flat)
            
            self.results['STPN'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'training_time': training_time, 'inference_time': inference_time
            }
            print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
            
            # Clean up
            del model, optimizer, supports, criterion
            self.clear_gpu_memory()
            
        except Exception as e:
            print(f"    STPN failed: {e}")
            import traceback
            traceback.print_exc()
            self.clear_gpu_memory()
    
    def visualize_results(self):
        """Create comparison plots."""
        if len(self.results) == 0:
            return
        
        print("\n[Viz] Creating comparison plots...")
        
        models = list(self.results.keys())
        mae_vals = [self.results[m]['mae'] for m in models]
        rmse_vals = [self.results[m]['rmse'] for m in models]
        r2_vals = [self.results[m]['r2'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE
        axes[0].barh(models, mae_vals, color=sns.color_palette("husl", len(models)))
        axes[0].set_xlabel('MAE (Lower is Better)')
        axes[0].set_title('Mean Absolute Error')
        for i, v in enumerate(mae_vals):
            axes[0].text(v, i, f' {v:.3f}', va='center')
        
        # RMSE
        axes[1].barh(models, rmse_vals, color=sns.color_palette("husl", len(models)))
        axes[1].set_xlabel('RMSE (Lower is Better)')
        axes[1].set_title('Root Mean Squared Error')
        for i, v in enumerate(rmse_vals):
            axes[1].text(v, i, f' {v:.3f}', va='center')
        
        # R²
        axes[2].barh(models, r2_vals, color=sns.color_palette("husl", len(models)))
        axes[2].set_xlabel('R² (Higher is Better)')
        axes[2].set_title('R² Score')
        for i, v in enumerate(r2_vals):
            axes[2].text(v, i, f' {v:.3f}', va='center')
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'results_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"[Viz] Saved to {plot_path}")
        plt.close()
    
    def save_results(self):
        """Save results to CSV and JSON."""
        if len(self.results) == 0:
            return
        
        print("\n[Save] Saving results...")
        
        # CSV
        csv_path = os.path.join(self.output_dir, f'results_{self.timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'MAE', 'RMSE', 'R²', 'Train_Time', 'Inference_Time'])
            for model, metrics in self.results.items():
                writer.writerow([
                    model, 
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['r2']:.4f}",
                    f"{metrics['training_time']:.2f}",
                    f"{metrics['inference_time']:.4f}"
                ])
        print(f"[Save] CSV: {csv_path}")
        
        # JSON
        json_path = os.path.join(self.output_dir, f'results_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[Save] JSON: {json_path}")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"{'RESULTS SUMMARY':^80}")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
        print(f"{'-'*80}")
        for model, metrics in sorted(self.results.items(), key=lambda x: x[1]['mae']):
            print(f"{model:<20} {metrics['mae']:<12.4f} {metrics['rmse']:<12.4f} {metrics['r2']:<12.4f}")
        print(f"{'='*80}\n")
    
    def run(self):
        """Run complete benchmark."""
        # Load data
        data = load_data_for_stpn(
            self.args.data_dir, 
            self.args.test_ratio, 
            self.args.in_len,
            self.args.out_len
        )
        
        # Run models
        self.run_historical_average(data['simple_y_train'], data['simple_y_test'])
        self.run_lstm(data['simple_x_train'], data['simple_y_train'], 
                     data['simple_x_test'], data['simple_y_test'])
        self.run_gru(data['simple_x_train'], data['simple_y_train'], 
                    data['simple_x_test'], data['simple_y_test'])
        self.run_stpn(data)
        
        # Visualize and save
        self.visualize_results()
        self.save_results()
        
        print("\n✅ Benchmark complete!\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Benchmark with STPN')
    parser.add_argument('--data_dir', type=str, default='cdata', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio')
    parser.add_argument('--in_len', type=int, default=18, help='Input sequence length')
    parser.add_argument('--out_len', type=int, default=12, help='Output sequence length')
    parser.add_argument('--quick_test', default=True, action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 10
        print("[Quick Test Mode] Epochs reduced to 10")
    
    return args


if __name__ == '__main__':
    args = parse_args()
    benchmark = BenchmarkWithSTPNRunner(args)
    benchmark.run()
