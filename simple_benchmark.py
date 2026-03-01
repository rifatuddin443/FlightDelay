"""Simplified Benchmark Runner - No DP, No Graph Models

Quick benchmark for testing basic models without complex dependencies.
Excludes: DSAFNet, STPN, graph-based models, differential privacy

Usage:
    python simple_benchmark.py --quick_test
    python simple_benchmark.py --epochs 30
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))

from baseline_methods import test_error

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

def load_simple_data(data_dir='cdata', test_ratio=0.2, seq_length=12):
    """Load flight delay data in simple format."""
    print(f"\n[Data] Loading from {data_dir}...")
    
    # Load delay data
    delay_file = os.path.join(data_dir, 'delay.npy')
    if not os.path.exists(delay_file):
        delay_file = os.path.join(data_dir, 'udelay.npy')
    
    delay_data = np.load(delay_file)  # [nodes, time, features]
    print(f"[Data] Delay shape: {delay_data.shape}")
    
    num_nodes, total_time, num_features = delay_data.shape
    
    # Create sequences
    sequences = []
    targets = []
    
    for t in range(total_time - seq_length - 1):
        for node in range(num_nodes):
            seq = delay_data[node, t:t+seq_length, :]
            target = delay_data[node, t+seq_length, :].mean()  # Average delay
            
            if not np.isnan(target):  # Skip NaN targets
                sequences.append(seq)
                targets.append(target)
    
    sequences = np.array(sequences)
    targets = np.array(targets).reshape(-1, 1)
    
    # Replace NaN with 0
    sequences = np.nan_to_num(sequences)
    targets = np.nan_to_num(targets)
    
    # Train/test split
    n_samples = len(sequences)
    n_train = int(n_samples * (1 - test_ratio))
    
    X_train = sequences[:n_train]
    y_train = targets[:n_train]
    X_test = sequences[n_train:]
    y_test = targets[n_train:]
    
    # Normalize
    X_mean = X_train.mean()
    X_std = X_train.std()
    y_mean = y_train.mean()
    y_std = y_train.std()
    
    X_train = (X_train - X_mean) / (X_std + 1e-8)
    X_test = (X_test - X_mean) / (X_std + 1e-8)
    
    print(f"[Data] Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, y_train, X_test, y_test, y_mean, y_std


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class SimpleBenchmark:
    """Simple benchmark runner."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"simple_benchmark_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"{'SIMPLE BENCHMARK SUITE':^80}")
        print(f"{'='*80}")
        print(f"\nDevice: {self.device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Output: {self.output_dir}\n")
    
    def run_historical_average(self, y_train, y_test):
        """Historical average baseline."""
        print("\n[1/5] Running Historical Average...")
        start = time.time()
        
        y_pred = np.full_like(y_test, y_train.mean())
        mae, rmse, r2 = test_error(y_pred, y_test)
        elapsed = time.time() - start
        
        self.results['Historical_Average'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': 0, 'inference_time': elapsed
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    
    def run_lstm(self, X_train, y_train, X_test, y_test):
        """LSTM baseline."""
        print("\n[2/5] Running LSTM...")
        
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
        
        # Evaluate
        model.eval()
        start = time.time()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test).to(self.device)).cpu().numpy()
        inference_time = time.time() - start
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['LSTM'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': training_time, 'inference_time': inference_time
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
    
    def run_gru(self, X_train, y_train, X_test, y_test):
        """GRU baseline."""
        print("\n[3/5] Running GRU...")
        
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
        
        # Evaluate
        model.eval()
        start = time.time()
        with torch.no_grad():
            y_pred = model(torch.FloatTensor(X_test).to(self.device)).cpu().numpy()
        inference_time = time.time() - start
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['GRU'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': training_time, 'inference_time': inference_time
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
    
    def run_tree_models(self, X_train, y_train, X_test, y_test):
        """Run tree-based models if available."""
        try:
            from sklearn.ensemble import RandomForestRegressor
            import xgboost as xgb
            
            # Flatten sequences
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_train_flat = y_train.ravel()
            
            # Random Forest
            print("\n[4/5] Running Random Forest...")
            start = time.time()
            rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
            rf.fit(X_train_flat, y_train_flat)
            training_time = time.time() - start
            
            start = time.time()
            y_pred = rf.predict(X_test_flat).reshape(-1, 1)
            inference_time = time.time() - start
            
            mae, rmse, r2 = test_error(y_pred, y_test)
            self.results['Random_Forest'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'training_time': training_time, 'inference_time': inference_time
            }
            print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            # XGBoost
            print("\n[5/5] Running XGBoost...")
            start = time.time()
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=6, n_jobs=-1, random_state=42)
            xgb_model.fit(X_train_flat, y_train_flat)
            training_time = time.time() - start
            
            start = time.time()
            y_pred = xgb_model.predict(X_test_flat).reshape(-1, 1)
            inference_time = time.time() - start
            
            mae, rmse, r2 = test_error(y_pred, y_test)
            self.results['XGBoost'] = {
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'training_time': training_time, 'inference_time': inference_time
            }
            print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
        except ImportError as e:
            print(f"\n[4-5/5] Tree models skipped: {e}")
    
    def visualize_results(self):
        """Create comparison plots."""
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
        X_train, y_train, X_test, y_test, y_mean, y_std = load_simple_data(
            self.args.data_dir, self.args.test_ratio, self.args.seq_length
        )
        
        # Run models
        self.run_historical_average(y_train, y_test)
        self.run_lstm(X_train, y_train, X_test, y_test)
        self.run_gru(X_train, y_train, X_test, y_test)
        self.run_tree_models(X_train, y_train, X_test, y_test)
        
        # Visualize and save
        self.visualize_results()
        self.save_results()
        
        print("\n✅ Benchmark complete!\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Simple Flight Delay Benchmark')
    parser.add_argument('--data_dir', type=str, default='cdata', help='Data directory')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio')
    parser.add_argument('--seq_length', type=int, default=12, help='Sequence length')
    parser.add_argument('--quick_test', action='store_true', help='Quick test mode')
    
    args = parser.parse_args()
    
    if args.quick_test:
        args.epochs = 10
        print("[Quick Test Mode] Epochs reduced to 10")
    
    return args


if __name__ == '__main__':
    args = parse_args()
    benchmark = SimpleBenchmark(args)
    benchmark.run()
