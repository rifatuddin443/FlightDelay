"""Benchmark: Flight Delay Prediction Models

Compares 5 flight delay prediction models:
1. Historical Average (baseline)
2. LSTM (sequential baseline)
3. GRU (sequential baseline)
4. STPN (Spatial-Temporal Graph Neural Network)
5. CNN-Opacus (Hybrid CNN-KAN-GAT with optional DP)

Usage:
    # Run all models on delayed flights only
    python benchmark_stpn_vs_cnnopacus.py --delayed_only --epochs 20
    
    # Run all models on all flights
    python benchmark_stpn_vs_cnnopacus.py --epochs 20
    
    # Run specific models
    python benchmark_stpn_vs_cnnopacus.py --models lstm gru stpn --delayed_only --epochs 20
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data

sys.path.insert(0, os.path.dirname(__file__))

from baseline_methods import test_error
from classifykat import (
    build_sequences,
    load_flight_data,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level

# Try to import STPN
try:
    from model import STPN
    STPN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] STPN not available: {e}")
    STPN = None
    STPN_AVAILABLE = False

# Try to import CNN-Opacus model
try:
    from cnnopacuskanMerge import HybridCNNKANGATPredictor
    CNN_OPACUS_AVAILABLE = True
except Exception as e:
    print(f"[Warning] CNN-Opacus not available: {e}")
    HybridCNNKANGATPredictor = None
    CNN_OPACUS_AVAILABLE = False

# Try to import Opacus for DP
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except Exception:
    PrivacyEngine = None
    ModuleValidator = None
    OPACUS_AVAILABLE = False

# Configure matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# SIMPLE BASELINE MODELS
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
# UTILITY FUNCTIONS
# ============================================================================

def normalized_adjacency_matrix(adj: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """Compute D^{-1/2} A D^{-1/2} for random walk normalization."""
    if symmetrize:
        adj = (adj + adj.T) / 2.0
    adj = adj + np.eye(adj.shape[0])
    d = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(d, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    D_inv_sqrt = np.diag(d_inv_sqrt)
    return D_inv_sqrt @ adj @ D_inv_sqrt


def load_benchmark_data(args):
    """Load and prepare data for both STPN and CNN-Opacus models."""
    print(f"\n[Data] Loading from {args.data_dir}...")
    
    # Use load_flight_data from classifykat
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )
    
    print(f"[Data] Loaded {num_nodes} nodes")
    print(f"[Data] Train: {train_inputs.shape}, Val: {val_inputs.shape}, Test: {test_inputs.shape}")
    
    # Build sequences for CNN-Opacus
    max_horizon = args.out_len
    horizons = [max_horizon]
    
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.in_len, max_horizon, args.delay_threshold, horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.in_len, max_horizon, args.delay_threshold, horizons
    )
    test_x, test_y_reg, test_y_cls = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.in_len, max_horizon, args.delay_threshold, horizons
    )
    
    print(f"[Data] Sequences - Train: {train_x.shape}, Test: {test_x.shape}")
    
    # Filter test data to keep only delayed samples if requested
    if args.delayed_only:
        # Compute average delay per sample from regression targets
        # test_y_reg shape: [samples, nodes, horizons] or similar
        test_delay_avg = test_y_reg.mean(dim=(1, 2))  # Average across nodes and horizons
        
        # Use scaler to convert back to minutes (approximate)
        # Assuming delay > args.min_delay_minutes minutes
        # Since data is scaled, we need to check the raw delay values from test_y_cls
        # test_y_cls is binary (0/1), so we filter based on classification labels
        # Alternatively, use raw delay threshold
        
        # Simple approach: keep samples where average delay > threshold
        # For scaled data, we use a relative threshold
        delayed_mask = test_y_cls.float().mean(dim=(1, 2)) > 0.5  # Majority delayed
        
        n_before = test_x.shape[0]
        test_x = test_x[delayed_mask]
        test_y_reg = test_y_reg[delayed_mask]
        test_y_cls = test_y_cls[delayed_mask]
        n_after = test_x.shape[0]
        
        print(f"[Data] Filtered to delayed samples only: {n_before} -> {n_after} ({100*n_after/n_before:.1f}%)")
        
        if n_after == 0:
            raise ValueError("No delayed samples found in test set!")
    
    # Prepare STPN format data
    # Load adjacency matrix for STPN
    adj_file = os.path.join(args.data_dir, 'dist_mx.npy')
    if os.path.exists(adj_file):
        adj_mx = np.load(adj_file)
    else:
        # Create identity matrix as fallback
        adj_mx = np.eye(num_nodes)
        print(f"[Data] Using identity matrix for adjacency")
    
    # Create support matrices for STPN
    adj_norm = normalized_adjacency_matrix(adj_mx)
    supports = [
        torch.FloatTensor(adj_norm),
        torch.FloatTensor(adj_norm @ adj_norm),
        torch.FloatTensor(adj_norm @ adj_norm @ adj_norm),
    ]
    
    # Prepare STPN input format: [samples, features, nodes, time]
    # build_sequences returns [samples, nodes, seq_len*features] for node_level
    # Need to reshape to [samples, nodes, seq_len, features] first
    feature_dim = train_inputs.shape[2]  # Original feature dimension
    n_samples_train = train_x.shape[0]
    n_samples_test = test_x.shape[0]
    
    # Reshape: [samples, nodes, seq_len*features] -> [samples, nodes, seq_len, features]
    train_x_4d = train_x.reshape(n_samples_train, num_nodes, args.in_len, feature_dim)
    test_x_4d = test_x.reshape(n_samples_test, num_nodes, args.in_len, feature_dim)
    
    # Permute to STPN format: [samples, features, nodes, time]
    stpn_train = train_x_4d.permute(0, 3, 1, 2)  # [N, F, nodes, time]
    stpn_test = test_x_4d.permute(0, 3, 1, 2)
    
    # Prepare target: average over nodes and time for simple comparison
    train_y_simple = train_y_reg.mean(dim=(1, 2)).unsqueeze(-1)  # [N, 1]
    test_y_simple = test_y_reg.mean(dim=(1, 2)).unsqueeze(-1)
    
    return {
        # Raw data
        'num_nodes': num_nodes,
        'scaler': scaler,
        
        # Edge indices for GNN
        'edge_index_adj': edge_index_adj,
        'edge_index_od': edge_index_od,
        
        # CNN-Opacus format
        'train_x': train_x,
        'train_y_reg': train_y_reg,
        'train_y_cls': train_y_cls,
        'test_x': test_x,
        'test_y_reg': test_y_reg,
        'test_y_cls': test_y_cls,
        
        # STPN format
        'stpn_train': stpn_train.numpy(),
        'stpn_test': stpn_test.numpy(),
        'train_y_simple': train_y_simple.numpy(),
        'test_y_simple': test_y_simple.numpy(),
        'supports': supports,
        
        # Simple format for LSTM/GRU (flattened sequences)
        'simple_train_x': train_x.reshape(train_x.shape[0], -1),  # [N, nodes*seq_len*features]
        'simple_test_x': test_x.reshape(test_x.shape[0], -1),
        'simple_train_y': train_y_simple,
        'simple_test_y': test_y_simple,
    }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class BenchmarkRunner:
    """Benchmark runner for flight delay prediction models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"benchmark_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"{'FLIGHT DELAY PREDICTION BENCHMARK':^80}")
        print(f"{'='*80}")
        print(f"\nDevice: {self.device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Models to run: {', '.join(args.models)}")
        print(f"STPN Available: {STPN_AVAILABLE}")
        print(f"CNN-Opacus Available: {CNN_OPACUS_AVAILABLE}")
        print(f"Differential Privacy: {'Enabled' if args.use_dp and OPACUS_AVAILABLE else 'Disabled'}")
        print(f"Test Filter: {'Delayed Only (>{}min)'.format(args.min_delay_minutes) if args.delayed_only else 'All Samples'}")
        print(f"Output: {self.output_dir}\n")
    
    def clear_gpu_memory(self):
        """Clear GPU memory between models."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def run_historical_average(self, data):
        """Run Historical Average baseline."""
        if 'historical' not in self.args.models:
            return
        
        print("\n[1/5] Running Historical Average...")
        start = time.time()
        
        y_train = data['simple_train_y']
        y_test = data['simple_test_y']
        
        y_pred = np.full_like(y_test, y_train.mean())
        mae, rmse, r2 = test_error(y_pred, y_test)
        elapsed = time.time() - start
        
        self.results['Historical_Average'] = {
            'mae': mae, 'rmse': rmse, 'r2': r2,
            'training_time': 0, 'inference_time': elapsed
        }
        print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        self.clear_gpu_memory()
    
    def run_lstm(self, data):
        """Run LSTM baseline."""
        if 'lstm' not in self.args.models:
            return
        
        print("\n[2/5] Running LSTM...")
        
        X_train = data['simple_train_x']
        y_train = data['simple_train_y']
        X_test = data['simple_test_x']
        y_test = data['simple_test_y']
        
        # Reshape for LSTM: [batch, seq_len, features]
        # Currently: [batch, nodes*seq_len*features]
        # Reshape to: [batch, seq_len, nodes*features]
        num_nodes = data['num_nodes']
        seq_len = self.args.in_len
        feature_dim = X_train.shape[1] // (num_nodes * seq_len)
        
        X_train_seq = X_train.reshape(X_train.shape[0], seq_len, -1)
        X_test_seq = X_test.reshape(X_test.shape[0], seq_len, -1)
        
        model = SimpleLSTM(X_train_seq.shape[-1], hidden_dim=64, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train)),
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
        
        # Evaluate in batches
        model.eval()
        start = time.time()
        y_pred = []
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_seq)),
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
        
        del model, optimizer, train_loader, test_loader
        self.clear_gpu_memory()
    
    def run_gru(self, data):
        """Run GRU baseline."""
        if 'gru' not in self.args.models:
            return
        
        print("\n[3/5] Running GRU...")
        
        X_train = data['simple_train_x']
        y_train = data['simple_train_y']
        X_test = data['simple_test_x']
        y_test = data['simple_test_y']
        
        # Reshape for GRU
        num_nodes = data['num_nodes']
        seq_len = self.args.in_len
        feature_dim = X_train.shape[1] // (num_nodes * seq_len)
        
        X_train_seq = X_train.reshape(X_train.shape[0], seq_len, -1)
        X_test_seq = X_test.reshape(X_test.shape[0], seq_len, -1)
        
        model = SimpleGRU(X_train_seq.shape[-1], hidden_dim=64, output_dim=1).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.MSELoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train_seq), torch.FloatTensor(y_train)),
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
        
        # Evaluate in batches
        model.eval()
        start = time.time()
        y_pred = []
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test_seq)),
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
        
        del model, optimizer, train_loader, test_loader
        self.clear_gpu_memory()
    
    def run_stpn(self, data):
        """Run STPN model."""
        if 'stpn' not in self.args.models or not STPN_AVAILABLE:
            print("\n[STPN] Skipped")
            return
        
        print("\n[4/5] Running STPN...")
        
        try:
            # Prepare data - keep on CPU, move batches to GPU
            X_train_cpu = data['stpn_train']
            y_train_cpu = data['train_y_simple']
            y_test_np = data['test_y_simple']
            
            supports = [s.to(self.device) for s in data['supports']]
            num_nodes = data['num_nodes']
            in_len = self.args.in_len
            out_len = self.args.out_len
            
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
                use_cov=False
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            criterion = nn.MSELoss()
            
            # Train with smaller batches
            start = time.time()
            model.train()
            batch_size = min(8, X_train_cpu.shape[0])
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
                    
                    # Average output over nodes/time to match target shape
                    outputs_avg = outputs.mean(dim=(1, 2, 3)).unsqueeze(-1)
                    
                    loss = criterion(outputs_avg, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Clean up batch from GPU
                    del batch_x, batch_y, batch_t_in, batch_t_out, outputs, outputs_avg, loss
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
            eval_batch_size = 8
            X_test_cpu = data['stpn_test']
            
            with torch.no_grad():
                for i in range(0, X_test_cpu.shape[0], eval_batch_size):
                    batch_x = torch.FloatTensor(X_test_cpu[i:i+eval_batch_size]).to(self.device)
                    batch_t_in = torch.arange(in_len).unsqueeze(0).repeat(batch_x.shape[0], 1).to(self.device)
                    batch_t_out = torch.arange(out_len).unsqueeze(0).repeat(batch_x.shape[0], 1).to(self.device)
                    
                    batch_pred = model(batch_x, batch_t_in, supports, batch_t_out, w_type)
                    batch_pred_avg = batch_pred.mean(dim=(1, 2, 3)).unsqueeze(-1).cpu().numpy()
                    y_pred_list.append(batch_pred_avg)
                    
                    del batch_x, batch_t_in, batch_t_out, batch_pred, batch_pred_avg
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            
            y_pred = np.concatenate(y_pred_list, axis=0)
            inference_time = time.time() - start
            
            mae, rmse, r2 = test_error(y_pred, y_test_np)
            
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
    
    def run_cnn_opacus(self, data):
        """Run CNN-Opacus (Hybrid CNN-KAN-GAT) model."""
        if 'cnnopacus' not in self.args.models or not CNN_OPACUS_AVAILABLE:
            print("\n[CNN-Opacus] Skipped")
            return
        
        print("\n[5/5] Running CNN-Opacus (Hybrid CNN-KAN-GAT)...")
        
        try:
            # Prepare data
            train_x = data['train_x']
            train_y_reg = data['train_y_reg']
            test_x = data['test_x']
            test_y_reg = data['test_y_reg']
            edge_index = data['edge_index_adj']
            
            # Flatten sequences for CNN: [samples, nodes*time*features]
            train_x_flat = train_x.reshape(train_x.shape[0], -1)
            test_x_flat = test_x.reshape(test_x.shape[0], -1)
            
            # Target: average over nodes and time
            train_y_simple = train_y_reg.mean(dim=(1, 2)).unsqueeze(-1)
            test_y_simple = test_y_reg.mean(dim=(1, 2)).unsqueeze(-1)
            
            # Create model
            feature_dim = train_x.shape[-1]
            in_channels = self.args.in_len * feature_dim
            out_channels = 1
            
            model = HybridCNNKANGATPredictor(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=self.args.hidden_dim,
                regressor_extra_layer=False,
                seq_len=self.args.in_len,
            ).to(self.device)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            criterion = nn.MSELoss()
            
            # Optional: Wrap with Opacus for DP
            privacy_engine = None
            if self.args.use_dp and OPACUS_AVAILABLE:
                print("    [DP] Enabling differential privacy...")
                model = ModuleValidator.fix(model)
                privacy_engine = PrivacyEngine()
                
                train_dataset = TensorDataset(train_x_flat, train_y_simple)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                )
                
                model, optimizer, train_loader = privacy_engine.make_private(
                    module=model,
                    optimizer=optimizer,
                    data_loader=train_loader,
                    noise_multiplier=self.args.noise_multiplier,
                    max_grad_norm=self.args.max_grad_norm,
                )
            else:
                train_dataset = TensorDataset(train_x_flat, train_y_simple)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.args.batch_size,
                    shuffle=True,
                )
            
            # Train (Stage 2: regression only for simplicity)
            start = time.time()
            model.train()
            model.set_stage(2)
            
            for epoch in range(self.args.epochs):
                total_loss = 0
                num_batches = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Create dummy Data object
                    batch_data = Data(
                        x=batch_x,
                        edge_index=edge_index.to(self.device),
                    )
                    
                    optimizer.zero_grad()
                    
                    # CNN encoding + regression
                    hidden = model._encode_x(batch_data.x)
                    outputs = model.forward_regressor(hidden)
                    
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / num_batches
                    eps_str = ""
                    if privacy_engine:
                        try:
                            epsilon = privacy_engine.get_epsilon(delta=self.args.delta)
                            eps_str = f", ε={epsilon:.2f}"
                        except:
                            pass
                    print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}{eps_str}")
            
            training_time = time.time() - start
            
            # Evaluate
            model.eval()
            start = time.time()
            
            y_pred_list = []
            test_dataset = TensorDataset(test_x_flat, test_y_simple)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_data = Data(x=batch_x, edge_index=edge_index.to(self.device))
                    
                    hidden = model._encode_x(batch_data.x)
                    batch_pred = model.forward_regressor(hidden).cpu().numpy()
                    y_pred_list.append(batch_pred)
            
            y_pred = np.vstack(y_pred_list)
            inference_time = time.time() - start
            
            mae, rmse, r2 = test_error(y_pred, test_y_simple.numpy())
            
            result_dict = {
                'mae': mae, 'rmse': rmse, 'r2': r2,
                'training_time': training_time, 'inference_time': inference_time
            }
            
            if privacy_engine:
                try:
                    epsilon = privacy_engine.get_epsilon(delta=self.args.delta)
                    result_dict['epsilon'] = epsilon
                    print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, ε: {epsilon:.2f}")
                except:
                    print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            else:
                print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            
            print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
            
            self.results['CNN_Opacus'] = result_dict
            
            # Clean up
            del model, optimizer, train_loader, test_loader
            self.clear_gpu_memory()
            
        except Exception as e:
            print(f"    CNN-Opacus failed: {e}")
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
        axes[0].invert_yaxis()
        
        # RMSE
        axes[1].barh(models, rmse_vals, color=sns.color_palette("husl", len(models)))
        axes[1].set_xlabel('RMSE (Lower is Better)')
        axes[1].set_title('Root Mean Square Error')
        axes[1].invert_yaxis()
        
        # R²
        axes[2].barh(models, r2_vals, color=sns.color_palette("husl", len(models)))
        axes[2].set_xlabel('R² (Higher is Better)')
        axes[2].set_title('R² Score')
        axes[2].invert_yaxis()
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'comparison_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved plot: {plot_path}")
        plt.close()
    
    def save_results(self):
        """Save results to CSV and JSON."""
        print("\n[Save] Saving results...")

        def _json_default(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if torch.is_tensor(obj):
                return obj.detach().cpu().tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
        
        # CSV
        csv_path = os.path.join(self.output_dir, f'results_{self.timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['Model', 'MAE', 'RMSE', 'R²', 'Train_Time', 'Inference_Time']
            if any('epsilon' in self.results[m] for m in self.results):
                header.append('Epsilon')
            writer.writerow(header)
            
            for model, metrics in self.results.items():
                row = [
                    model, 
                    f"{metrics['mae']:.4f}",
                    f"{metrics['rmse']:.4f}",
                    f"{metrics['r2']:.4f}",
                    f"{metrics['training_time']:.2f}",
                    f"{metrics['inference_time']:.4f}"
                ]
                if 'epsilon' in metrics:
                    row.append(f"{metrics['epsilon']:.2f}")
                writer.writerow(row)
        print(f"[Save] CSV: {csv_path}")
        
        # JSON
        json_path = os.path.join(self.output_dir, f'results_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=_json_default)
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
        if self.args.seed is not None:
            set_seed(self.args.seed)
        
        # Load data
        data = load_benchmark_data(self.args)
        
        # Run models
        self.run_historical_average(data)
        self.run_lstm(data)
        self.run_gru(data)
        self.run_stpn(data)
        self.run_cnn_opacus(data)
        
        # Visualize and save
        self.visualize_results()
        self.save_results()
        
        print("\n✅ Benchmark complete!\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Flight Delay Prediction Benchmark')
    
    # Models to run
    parser.add_argument('--models', nargs='+', 
                        default=['historical', 'lstm', 'gru', 'stpn', 'cnnopacus'],
                        choices=['historical', 'lstm', 'gru', 'stpn', 'cnnopacus'],
                        help='Which models to run (default: all)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='cdata', help='Data directory')
    parser.add_argument('--data_source', type=str, default='cdata', help='Data source name')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy', help='Weather file')
    parser.add_argument('--period_hours', type=int, default=2, help='Period in hours')
    parser.add_argument('--test_ratio', type=float, default=0.2, help='Test ratio')
    
    # Sequence arguments
    parser.add_argument('--in_len', type=int, default=18, help='Input sequence length')
    parser.add_argument('--out_len', type=int, default=12, help='Output sequence length')
    parser.add_argument('--delay_threshold', type=float, default=15.0, help='Delay threshold for classification')
    parser.add_argument('--use_node_level', action='store_true', help='Use node-level sequences')
    
    # Test filtering
    parser.add_argument('--delayed_only', action='store_true', help='Test on delayed samples only (>5 min)')
    parser.add_argument('--min_delay_minutes', type=float, default=5.0, help='Minimum delay in minutes for filtering')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Differential Privacy arguments
    parser.add_argument('--use_dp', action='store_true', help='Enable differential privacy')
    parser.add_argument('--no_dp', dest='use_dp', action='store_false', help='Disable differential privacy')
    parser.add_argument('--noise_multiplier', type=float, default=1.0, help='Noise multiplier for DP')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for DP')
    parser.add_argument('--delta', type=float, default=1e-5, help='Delta for DP')
    
    parser.set_defaults(use_dp=False)
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    benchmark = BenchmarkRunner(args)
    benchmark.run()
