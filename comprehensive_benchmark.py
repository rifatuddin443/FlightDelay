"""Comprehensive Benchmark: Compare Your Model with State-of-the-Art Methods

This script compares multiple flight delay prediction models:
1. Your CNN-KAN Model (from cnnopacus_other model.py)
2. STPN (Spatio-Temporal Pattern Network)
3. DSAFnet (Dual-Stream Attention Fusion Network)
4. Historical Average
5. VAR (Vector Auto-Regressive)
6. LSTM Baseline
7. GRU Baseline
8. Transformer Baseline

Usage:
    python comprehensive_benchmark.py --data_dir cdata --epochs 50
    python comprehensive_benchmark.py --quick_test --epochs 10  # Fast test
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

# Import existing modules
from classifykat import (
    load_flight_data,
    build_sequences,
    classification_metrics,
    regression_metrics,
    set_seed,
    StandardScaler,
    EarlyStopping,
)
from classifykat_balanced import build_sequences_node_level
from baseline_methods import (
    test_error,
    historical_average_predict,
    var_predict,
)

# Set random seeds for reproducibility
set_seed(42)

# Configure matplotlib for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# LSTM/GRU/TRANSFORMER BASELINE MODELS
# ============================================================================

class LSTMBaseline(nn.Module):
    """LSTM baseline for flight delay prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Use last hidden state
        out = self.fc(self.dropout(h_n[-1]))
        return out


class GRUBaseline(nn.Module):
    """GRU baseline for flight delay prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        gru_out, h_n = self.gru(x)
        out = self.fc(self.dropout(h_n[-1]))
        return out


class TransformerBaseline(nn.Module):
    """Transformer baseline for flight delay prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_heads: int = 4, 
                 num_layers: int = 2, output_dim: int = 1, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        x = self.embedding(x)
        transformer_out = self.transformer(x)
        # Use mean pooling across sequence
        pooled = transformer_out.mean(dim=1)
        out = self.fc(self.dropout(pooled))
        return out


# ============================================================================
# MODEL WRAPPER FOR UNIFIED INTERFACE
# ============================================================================

class ModelWrapper:
    """Wrapper to provide unified interface for all models."""
    
    def __init__(self, model, model_name: str, device: str = 'cuda'):
        self.model = model
        self.model_name = model_name
        self.device = device
        if hasattr(model, 'to'):
            self.model.to(device)
    
    def train_epoch(self, train_loader, optimizer, criterion, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        total_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    inputs, targets = batch
                else:
                    inputs = batch[0]
                    targets = batch[-1]
            else:
                inputs = batch.x
                targets = batch.y
            
            # Move to device
            if hasattr(inputs, 'to'):
                inputs = inputs.to(self.device)
            if hasattr(targets, 'to'):
                targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * targets.size(0)
            total_samples += targets.size(0)
        
        return total_loss / total_samples
    
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in test_loader:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        inputs, targets = batch
                    else:
                        inputs = batch[0]
                        targets = batch[-1]
                else:
                    inputs = batch.x
                    targets = batch.y
                
                # Move to device
                if hasattr(inputs, 'to'):
                    inputs = inputs.to(self.device)
                if hasattr(targets, 'to'):
                    targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Compute metrics
        mae, rmse, r2 = test_error(predictions, actuals)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals
        }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Main benchmark orchestrator."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        self.output_dir = f"benchmark_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"[Benchmark] Using device: {self.device}")
        print(f"[Benchmark] Results will be saved to: {self.output_dir}")
    
    def load_data(self):
        """Load and prepare datasets."""
        print("\n[Benchmark] Loading flight data...")
        
        # Load data using existing utilities
        # Returns: edge_index_adj, edge_index_od, edge_index_od_t, 
        #          train_inputs, val_inputs, test_inputs,
        #          train_delay, val_delay, test_delay,
        #          train_raw, val_raw, test_raw, scaler, num_nodes
        (edge_index_adj, edge_index_od, edge_index_od_t,
         train_inputs, val_inputs, test_inputs,
         train_delay, val_delay, test_delay,
         train_raw, val_raw, test_raw, scaler, num_nodes) = load_flight_data(
            self.args.data_dir,
            train_ratio=1.0 - self.args.test_ratio,
            val_ratio=0.0,  # No validation split for benchmark
            data_source=self.args.data_dir
        )
        
        # Combine train and validation for simplicity
        all_inputs = np.concatenate([train_inputs, val_inputs, test_inputs], axis=1)
        all_raw = np.concatenate([train_raw, val_raw, test_raw], axis=1)
        
        # Build sequences using the new interface
        # We'll use build_sequences_node_level from classifykat_balanced
        from classifykat_balanced import build_sequences_node_level
        
        # Calculate split point
        total_samples = all_inputs.shape[1]
        n_train = int(total_samples * (1 - self.args.test_ratio))
        
        # Create simple sequences for benchmark (flatten temporal data)
        # Shape: [nodes, time, features] -> [time * nodes, seq_len, features]
        seq_data = []
        label_data = []
        
        for t in range(total_samples - self.args.seq_length - self.args.pred_length):
            for node in range(num_nodes):
                # Input sequence
                seq = all_inputs[node, t:t+self.args.seq_length, :]
                # Target (use delay only)
                target = all_raw[node, t+self.args.seq_length:t+self.args.seq_length+self.args.pred_length, :]
                
                seq_data.append(seq)
                label_data.append(target.mean(axis=0, keepdims=True))  # Average over prediction horizon
        
        sequences = np.array(seq_data)
        labels = np.array(label_data)
        
        # Split into train/test
        n_samples = len(sequences)
        n_train_samples = int(n_samples * (1 - self.args.test_ratio))
        
        self.data = {
            'sequences_arr': sequences,
            'labels_arr': labels,
            'n_train': n_train_samples,
            'n_test': n_samples - n_train_samples,
            'edge_index': edge_index_adj,
            'edge_index_od': edge_index_od,
            'edge_index_od_t': edge_index_od_t,
            'scaler': scaler,
            'num_nodes': num_nodes,
        }
        
        print(f"[Benchmark] Loaded {n_samples} samples ({n_train_samples} train, {n_samples - n_train_samples} test)")
        print(f"[Benchmark] Sequence shape: {sequences.shape}")
        print(f"[Benchmark] Label shape: {labels.shape}")
        
        return self.data
    
    def prepare_baseline_data(self):
        """Prepare data for classical baselines (HA, VAR)."""
        # Use the labels directly (they are delay values)
        # Shape: [samples, pred_length]
        delay_combined = self.data['labels_arr'].squeeze()
        
        # If we need to reshape for time series format
        # Classical methods expect [features, time] format
        if len(delay_combined.shape) == 1:
            delay_combined = delay_combined.reshape(-1, 1)
        
        return delay_combined
    
    def run_historical_average(self):
        """Run Historical Average baseline."""
        print("\n[Benchmark] Running Historical Average...")
        start_time = time.time()
        
        delay_data = self.prepare_baseline_data()
        y_pred, y_true = historical_average_predict(
            delay_data.T,  # Transpose to [nodes, time]
            period=self.args.seq_length,
            test_ratio=self.args.test_ratio
        )
        
        mae, rmse, r2 = test_error(y_pred.T, y_true.T)
        elapsed = time.time() - start_time
        
        self.results['Historical_Average'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': 0,
            'inference_time': elapsed,
        }
        
        print(f"[HA] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"[HA] Inference time: {elapsed:.2f}s")
    
    def run_var(self):
        """Run VAR baseline."""
        print("\n[Benchmark] Running VAR...")
        start_time = time.time()
        
        try:
            delay_data = self.prepare_baseline_data()
            y_pred, y_true = var_predict(
                delay_data.T,  # Transpose to [nodes, time]
                n_forwards=tuple(range(1, self.args.pred_length + 1)),
                n_lags=min(36, self.args.seq_length),
                test_ratio=self.args.test_ratio
            )
            
            # Average across prediction horizons
            y_pred_mean = y_pred.mean(axis=0)
            mae, rmse, r2 = test_error(y_pred_mean.T, y_true.T)
            elapsed = time.time() - start_time
            
            self.results['VAR'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'training_time': elapsed * 0.8,  # Rough estimate
                'inference_time': elapsed * 0.2,
            }
            
            print(f"[VAR] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            print(f"[VAR] Total time: {elapsed:.2f}s")
            
        except Exception as e:
            print(f"[VAR] Failed: {e}")
            self.results['VAR'] = {
                'mae': np.nan,
                'rmse': np.nan,
                'r2': np.nan,
                'error': str(e)
            }
    
    def run_lstm_baseline(self):
        """Run LSTM baseline."""
        print("\n[Benchmark] Running LSTM...")
        
        # Prepare data
        X_train = self.data['sequences_arr'][:self.data['n_train']]
        y_train = self.data['labels_arr'][:self.data['n_train'], :, 0].mean(axis=1, keepdims=True)
        X_test = self.data['sequences_arr'][self.data['n_train']:]
        y_test = self.data['labels_arr'][self.data['n_train']:, :, 0].mean(axis=1, keepdims=True)
        
        # Create model
        input_dim = X_train.shape[-1]
        model = LSTMBaseline(input_dim, hidden_dim=128, num_layers=2, output_dim=1)
        model.to(self.device)
        
        # Train
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"[LSTM] Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy()
        
        inference_time = time.time() - start_time
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['LSTM'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'inference_time': inference_time,
        }
        
        print(f"[LSTM] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"[LSTM] Training time: {training_time:.2f}s, Inference: {inference_time:.2f}s")
    
    def run_gru_baseline(self):
        """Run GRU baseline."""
        print("\n[Benchmark] Running GRU...")
        
        # Prepare data
        X_train = self.data['sequences_arr'][:self.data['n_train']]
        y_train = self.data['labels_arr'][:self.data['n_train'], :, 0].mean(axis=1, keepdims=True)
        X_test = self.data['sequences_arr'][self.data['n_train']:]
        y_test = self.data['labels_arr'][self.data['n_train']:, :, 0].mean(axis=1, keepdims=True)
        
        # Create model
        input_dim = X_train.shape[-1]
        model = GRUBaseline(input_dim, hidden_dim=128, num_layers=2, output_dim=1)
        model.to(self.device)
        
        # Train
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"[GRU] Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy()
        
        inference_time = time.time() - start_time
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['GRU'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'inference_time': inference_time,
        }
        
        print(f"[GRU] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"[GRU] Training time: {training_time:.2f}s, Inference: {inference_time:.2f}s")
    
    def run_transformer_baseline(self):
        """Run Transformer baseline."""
        print("\n[Benchmark] Running Transformer...")
        
        # Prepare data
        X_train = self.data['sequences_arr'][:self.data['n_train']]
        y_train = self.data['labels_arr'][:self.data['n_train'], :, 0].mean(axis=1, keepdims=True)
        X_test = self.data['sequences_arr'][self.data['n_train']:]
        y_test = self.data['labels_arr'][self.data['n_train']:, :, 0].mean(axis=1, keepdims=True)
        
        # Create model
        input_dim = X_train.shape[-1]
        model = TransformerBaseline(input_dim, hidden_dim=128, num_heads=4, num_layers=2, output_dim=1)
        model.to(self.device)
        
        # Train
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        start_time = time.time()
        for epoch in range(self.args.epochs):
            model.train()
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"[Transformer] Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        
        # Evaluate
        model.eval()
        start_time = time.time()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)
            y_pred = model(X_test_tensor).cpu().numpy()
        
        inference_time = time.time() - start_time
        
        mae, rmse, r2 = test_error(y_pred, y_test)
        
        self.results['Transformer'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'training_time': training_time,
            'inference_time': inference_time,
        }
        
        print(f"[Transformer] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        print(f"[Transformer] Training time: {training_time:.2f}s, Inference: {inference_time:.2f}s")
    
    def run_all_baselines(self):
        """Run all baseline methods."""
        if not self.args.skip_classical:
            self.run_historical_average()
            if not self.args.quick_test:
                self.run_var()
        
        if not self.args.skip_dl:
            self.run_lstm_baseline()
            self.run_gru_baseline()
            if not self.args.quick_test:
                self.run_transformer_baseline()
    
    def visualize_results(self):
        """Create comparison visualizations."""
        print("\n[Benchmark] Creating visualizations...")
        
        # Extract metrics
        models = list(self.results.keys())
        mae_values = [self.results[m].get('mae', np.nan) for m in models]
        rmse_values = [self.results[m].get('rmse', np.nan) for m in models]
        r2_values = [self.results[m].get('r2', np.nan) for m in models]
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Flight Delay Prediction: Model Comparison', fontsize=16, fontweight='bold')
        
        # MAE comparison
        ax = axes[0, 0]
        bars = ax.barh(models, mae_values, color=sns.color_palette("husl", len(models)))
        ax.set_xlabel('MAE (Mean Absolute Error)', fontsize=12)
        ax.set_title('MAE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, mae_values)):
            if not np.isnan(val):
                ax.text(val, i, f' {val:.3f}', va='center', fontsize=10)
        
        # RMSE comparison
        ax = axes[0, 1]
        bars = ax.barh(models, rmse_values, color=sns.color_palette("husl", len(models)))
        ax.set_xlabel('RMSE (Root Mean Squared Error)', fontsize=12)
        ax.set_title('RMSE Comparison (Lower is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, rmse_values)):
            if not np.isnan(val):
                ax.text(val, i, f' {val:.3f}', va='center', fontsize=10)
        
        # R² comparison
        ax = axes[1, 0]
        bars = ax.barh(models, r2_values, color=sns.color_palette("husl", len(models)))
        ax.set_xlabel('R² Score', fontsize=12)
        ax.set_title('R² Comparison (Higher is Better)', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, r2_values)):
            if not np.isnan(val):
                ax.text(val, i, f' {val:.3f}', va='center', fontsize=10)
        
        # Training time comparison
        ax = axes[1, 1]
        train_times = [self.results[m].get('training_time', 0) for m in models]
        bars = ax.barh(models, train_times, color=sns.color_palette("husl", len(models)))
        ax.set_xlabel('Training Time (seconds)', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, train_times)):
            if val > 0:
                ax.text(val, i, f' {val:.1f}s', va='center', fontsize=10)
        
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(self.output_dir, f'benchmark_comparison_{self.timestamp}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Benchmark] Saved visualization to: {output_path}")
        
        plt.close()
    
    def save_results(self):
        """Save benchmark results to CSV and JSON."""
        print("\n[Benchmark] Saving results...")
        
        # Save as CSV
        csv_path = os.path.join(self.output_dir, f'benchmark_results_{self.timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'MAE', 'RMSE', 'R²', 'Training Time (s)', 'Inference Time (s)'])
            for model_name, metrics in self.results.items():
                writer.writerow([
                    model_name,
                    f"{metrics.get('mae', np.nan):.4f}",
                    f"{metrics.get('rmse', np.nan):.4f}",
                    f"{metrics.get('r2', np.nan):.4f}",
                    f"{metrics.get('training_time', 0):.2f}",
                    f"{metrics.get('inference_time', 0):.2f}",
                ])
        print(f"[Benchmark] Saved CSV to: {csv_path}")
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f'benchmark_results_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=float)
        print(f"[Benchmark] Saved JSON to: {json_path}")
        
        # Print summary table
        print("\n" + "="*80)
        print("BENCHMARK RESULTS SUMMARY")
        print("="*80)
        print(f"{'Model':<20} {'MAE':<12} {'RMSE':<12} {'R²':<12} {'Train Time':<12}")
        print("-"*80)
        for model_name, metrics in sorted(self.results.items(), key=lambda x: x[1].get('mae', float('inf'))):
            print(f"{model_name:<20} "
                  f"{metrics.get('mae', np.nan):<12.4f} "
                  f"{metrics.get('rmse', np.nan):<12.4f} "
                  f"{metrics.get('r2', np.nan):<12.4f} "
                  f"{metrics.get('training_time', 0):<12.2f}")
        print("="*80)
    
    def run(self):
        """Execute complete benchmark."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FLIGHT DELAY PREDICTION BENCHMARK")
        print("="*80)
        
        # Load data
        self.load_data()
        
        # Run baselines
        self.run_all_baselines()
        
        # TODO: Add STPN, DSAFnet, and your custom model here
        # These will be added in the next iteration
        
        # Visualize and save
        self.visualize_results()
        self.save_results()
        
        print("\n[Benchmark] Complete!")
        return self.results


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Comprehensive Model Benchmark')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='cdata',
                        help='Directory containing flight data')
    parser.add_argument('--seq_length', type=int, default=12,
                        help='Input sequence length')
    parser.add_argument('--pred_length', type=int, default=4,
                        help='Prediction horizon')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Test set ratio')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    
    # Benchmark options
    parser.add_argument('--quick_test', action='store_true',
                        help='Run quick test with fewer epochs and models')
    parser.add_argument('--skip_classical', action='store_true',
                        help='Skip classical baselines (HA, VAR)')
    parser.add_argument('--skip_dl', action='store_true',
                        help='Skip deep learning baselines (LSTM, GRU, Transformer)')
    parser.add_argument('--skip_graph', action='store_true',
                        help='Skip graph-based models (STPN, DSAFnet)')
    
    args = parser.parse_args()
    
    # Quick test overrides
    if args.quick_test:
        args.epochs = 10
        print("[Benchmark] Quick test mode: epochs=10, limited models")
    
    return args


if __name__ == '__main__':
    args = parse_args()
    
    # Run benchmark
    runner = BenchmarkRunner(args)
    results = runner.run()
