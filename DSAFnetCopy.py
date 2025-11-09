import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import random
import pandas as pd
import os
import time
import copy
from datetime import datetime, timedelta
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader

# ===== DSAFNet Model Definition =====
class SpatialAttentionStream(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 1, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):  # x: [batch, airports, features]
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        x_emb = self.dropout(x_emb)
        x_emb = x_emb.transpose(0, 1)
        out, _ = self.attention(x_emb, x_emb, x_emb)
        out = out.transpose(0, 1)
        return self.dropout(out)

class TemporalAttentionStream(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 1, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):  # x: [batch, time_steps, features]
        x_emb = self.embedding(x)
        x_emb = self.layer_norm(x_emb)
        x_emb = self.dropout(x_emb)
        x_emb = x_emb.transpose(0, 1)
        out, _ = self.attention(x_emb, x_emb, x_emb)
        out = out.transpose(0, 1)
        return self.dropout(out)

class ContextualCrossAttention(nn.Module):
    def __init__(self, hidden_dim=64, dropout_rate=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 1, dropout=dropout_rate)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim), 
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout_rate),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, spatial, temporal):
        fusion, _ = self.cross_attn(spatial.transpose(0, 1), temporal.transpose(0, 1), temporal.transpose(0, 1))
        fusion = fusion.transpose(0, 1)
        fusion = self.dropout(fusion)
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
    def __init__(self, hidden_dim=64, output_steps=12, dropout_rate=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_steps * 2)
        )
        self.output_steps = output_steps
        
    def forward(self, fused_features):
        # fused_features: [batch, airports, hidden_dim]
        out = self.projection(fused_features)  # [batch, airports, output_steps * 2]
        batch_size, airports, _ = out.shape
        out = out.view(batch_size, airports, self.output_steps, 2)  # [batch, airports, output_steps, 2]
        return out

class DSAFNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_steps=12, num_graphs=3, dropout_rate=0.1):
        super().__init__()
        self.spatial_stream = SpatialAttentionStream(input_dim, hidden_dim, dropout_rate)
        self.temporal_stream = TemporalAttentionStream(input_dim, hidden_dim, dropout_rate)
        self.cross_fusion = ContextualCrossAttention(hidden_dim, dropout_rate)
        self.graph_encoder = SimpleGraphEncoder(num_graphs)
        self.output_proj = OutputProjection(hidden_dim, output_steps, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x, adj_matrices):
        batch_size, airports, time_steps, features = x.shape
        
        # OPTIMIZED: Batch process all spatial features at once
        # Reshape to process all time steps in parallel: [batch*time_steps, airports, features]
        x_spatial = x.permute(0, 2, 1, 3).contiguous().view(batch_size * time_steps, airports, features)
        spatial_batch = self.spatial_stream(x_spatial)  # Single call instead of 12
        spatial_features = spatial_batch.view(batch_size, time_steps, airports, -1).permute(0, 2, 1, 3)
        
        # OPTIMIZED: Batch process all temporal features at once  
        # Reshape to process all airports in parallel: [batch*airports, time_steps, features]
        x_temporal = x.permute(0, 1, 2, 3).contiguous().view(batch_size * airports, time_steps, features)
        temporal_batch = self.temporal_stream(x_temporal)  # Single call instead of 70
        temporal_features = temporal_batch.view(batch_size, airports, time_steps, -1)
        
        # Average pooling (much faster than mean on lists)
        spatial_avg = spatial_features.mean(dim=2)  # [batch, airports, hidden_dim]
        temporal_avg = temporal_features.mean(dim=2)  # [batch, airports, hidden_dim]
        
        fused = self.cross_fusion(spatial_avg, temporal_avg)
        fused = self.dropout(fused)
        graph_enhanced = self.graph_encoder(fused, adj_matrices)
        out = self.output_proj(graph_enhanced)
        return out

# ===== Data Loading Functions =====
def load_flight_delay_data(dataset='US', data_dir='.', sequence_length=12, prediction_length=12, normalize=True):
    """
    Load flight delay dataset (US or China)
    
    Args:
        dataset: 'US' or 'China'
        data_dir: Base directory containing udata and cdata folders
        sequence_length: Input sequence length (default 12)
        prediction_length: Output prediction length (default 12)
        normalize: Whether to normalize the data
    
    Returns:
        data: [samples, airports, sequence_length, features] - Input sequences
        labels: [samples, airports, prediction_length, 2] - Target sequences (arrival, departure delays)
        adj_matrices: List of adjacency matrices
        scaler: StandardScaler object (if normalize=True)
        num_airports: Number of airports
    """
    
    print(f"Loading {dataset} flight delay dataset...")
    
    if dataset.upper() == 'US':
        data_folder = os.path.join(data_dir, 'udata')
        delay_file = os.path.join(data_folder, 'udelay.npy')
        adj_file = os.path.join(data_folder, 'adj_mx.npy')
        weather_file = os.path.join(data_folder, 'weather2016_2021.npy')
        od_file = os.path.join(data_folder, 'od_pair.npy')
    else:  # China
        data_folder = os.path.join(data_dir, 'cdata')
        delay_file = os.path.join(data_folder, 'delay.npy')
        adj_file = os.path.join(data_folder, 'dist_mx.npy')
        weather_file = os.path.join(data_folder, 'weather_cn.npy')
        od_file = None  # Check if exists
        
    # Load delay data
    if os.path.exists(delay_file):
        delay_data = np.load(delay_file)
        print(f"Loaded delay data: {delay_data.shape}")
    else:
        raise FileNotFoundError(f"Delay data file not found: {delay_file}")
    
    # Load adjacency matrix
    if os.path.exists(adj_file):
        adj_matrix = np.load(adj_file)
        print(f"Loaded adjacency matrix: {adj_matrix.shape}")
        
        # Create multiple graph views
        adj_matrices = []
        
        # Original adjacency matrix (connectivity)
        adj_matrices.append(adj_matrix)
        
        # Distance-based adjacency (inverse relationship)
        if dataset.upper() == 'CHINA':
            # For distance matrix, convert to similarity
            adj_dist = np.exp(-adj_matrix / np.std(adj_matrix))
            np.fill_diagonal(adj_dist, 1.0)
        else:
            adj_dist = adj_matrix.copy()
        adj_matrices.append(adj_dist)
        
        # Self-connection matrix (identity)
        adj_self = np.eye(adj_matrix.shape[0])
        adj_matrices.append(adj_self)
        
    else:
        raise FileNotFoundError(f"Adjacency matrix file not found: {adj_file}")
    
    # Load weather data if available
    weather_data = None
    if os.path.exists(weather_file):
        weather_data = np.load(weather_file)
        print(f"Loaded weather data: {weather_data.shape}")
    else:
        print(f"⚠️  Weather data not found: {weather_file}")
    
    # Load OD data if available
    od_data = None
    if od_file and os.path.exists(od_file):
        od_data = np.load(od_file)
        print(f"Loaded OD data: {od_data.shape}")
    
    # Process delay data
    # Check data dimensions and correct if needed
    print(f"📊 Original delay data shape: {delay_data.shape}")
    
    if len(delay_data.shape) == 3:
        dim1, dim2, dim3 = delay_data.shape
        
        # The correct format should be [time_steps, airports, features]
        # Check if we need to transpose based on adjacency matrix size
        if adj_matrix.shape[0] == dim1 and dim2 > dim1:
            # Data is [airports, time_steps, features] - need to transpose
            delay_data = delay_data.transpose(1, 0, 2)  # [time_steps, airports, features]
            print(f"🔄 Transposed delay data to: {delay_data.shape}")
        
        time_steps, num_airports, num_features = delay_data.shape
    elif len(delay_data.shape) == 2:
        # If it's [time_steps, airports], assume single feature (total delay)
        time_steps, num_airports = delay_data.shape
        delay_data = delay_data[:, :, np.newaxis]  # Add feature dimension
        num_features = 1
    else:
        raise ValueError(f"Unexpected delay data shape: {delay_data.shape}")
    
    print(f"Dataset: {time_steps} timesteps, {num_airports} airports, {num_features} features")
    
    # Verify dimensions match adjacency matrix
    if num_airports != adj_matrix.shape[0]:
        raise ValueError(f"Number of airports mismatch: delay data has {num_airports}, "
                        f"adjacency matrix has {adj_matrix.shape[0]}")
    
    # Handle NaN values
    nan_count = np.isnan(delay_data).sum()
    if nan_count > 0:
        print(f"Found {nan_count} NaN values ({nan_count/delay_data.size*100:.2f}%), replacing with zeros...")
        delay_data = np.nan_to_num(delay_data, nan=0.0)
    
    # Check for reasonable data ranges
    data_min, data_max = delay_data.min(), delay_data.max()
    print(f"Data range: [{data_min:.4f}, {data_max:.4f}]")
    
    if abs(data_max) > 1000:  # Delays in minutes shouldn't exceed ~1000
        print("⚠️  Data values seem very large - you might want to check units or scaling")
    
    # If we have weather data, keep it separate for efficient handling
    weather_data_processed = None
    if weather_data is not None:
        if len(weather_data.shape) == 3:
            if weather_data.shape[0] == time_steps and weather_data.shape[1] == num_airports:
                weather_data_processed = weather_data
                print(f"✅ Weather data prepared: {weather_data_processed.shape}")
            else:
                print(f"⚠️  Weather data shape mismatch: {weather_data.shape} vs expected ({time_steps}, {num_airports}, ...)")
        elif len(weather_data.shape) == 2:
            if weather_data.shape[0] == time_steps and weather_data.shape[1] == num_airports:
                weather_data_processed = weather_data
                print(f"✅ Weather data prepared: {weather_data_processed.shape}")
    
    scaler = None
    if normalize:
        print("Normalization will be applied using training data only")
    
    return delay_data, weather_data_processed, adj_matrices, scaler, num_airports, time_steps

# ===== Memory-Efficient Dataset Class =====
class FlightDelayDataset(Dataset):
    """
    Memory-efficient dataset class for flight delay prediction using sliding windows
    """
    def __init__(self, data, weather_data, adj_matrices, indices, in_len, out_len, 
                 normalize=True, scaler=None, add_time_features=True):
        """
        Args:
            data: [time_steps, airports, features] - raw delay data
            weather_data: [time_steps, airports] or [time_steps, airports, weather_features] - weather data
            adj_matrices: List of adjacency matrices
            indices: List of valid starting indices for sequences
            in_len: Input sequence length
            out_len: Output sequence length
            normalize: Whether data is normalized
            scaler: StandardScaler object for denormalization if needed
            add_time_features: Whether to add time-based features
        """
        self.data = torch.FloatTensor(data)  # [time_steps, airports, features]
        self.weather_data = torch.FloatTensor(weather_data) if weather_data is not None else None
        self.adj_matrices = [torch.FloatTensor(adj) for adj in adj_matrices]
        self.indices = indices
        self.in_len = in_len
        self.out_len = out_len
        self.normalize = normalize
        self.scaler = scaler
        self.add_time_features = add_time_features
        
        self.time_steps, self.num_airports, self.num_features = self.data.shape
        
        # Pre-calculate some statistics for time features
        if add_time_features:
            self.period = 24  # Assume 24-hour period for time features
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sequence starting at indices[idx]
        Returns data in format compatible with DSAFNet: [airports, time_steps, features]
        """
        start_idx = self.indices[idx]
        
        # Extract input sequence: [in_len, airports, features]
        x_seq = self.data[start_idx:start_idx + self.in_len]
        
        # Extract target sequence: [out_len, airports, 2] (arrival, departure delays)
        y_seq = self.data[start_idx + self.in_len:start_idx + self.in_len + self.out_len]
        
        # Ensure we have at least 2 features for arrival/departure delays
        if self.num_features >= 2:
            y_delays = y_seq[:, :, :2]  # First 2 features are delays
        else:
            # If only 1 feature, duplicate for arrival and departure
            y_delays = y_seq.repeat(1, 1, 2)
        
        # Combine with weather data if available
        if self.weather_data is not None:
            if self.weather_data.dim() == 2:  # [time_steps, airports]
                weather_seq = self.weather_data[start_idx:start_idx + self.in_len].unsqueeze(2)  # Add feature dim
            else:  # [time_steps, airports, weather_features]
                weather_seq = self.weather_data[start_idx:start_idx + self.in_len]
            
            # Concatenate delay and weather features
            x_seq = torch.cat([x_seq, weather_seq], dim=2)
        
        # Add time-based features if requested
        if self.add_time_features:
            time_features = self._get_time_features(start_idx)  # [in_len, airports, 1]
            x_seq = torch.cat([x_seq, time_features], dim=2)
        
        # Transpose to DSAFNet format: [airports, time_steps, features]
        x_seq = x_seq.permute(1, 0, 2)  # [airports, in_len, features]
        y_delays = y_delays.permute(1, 0, 2)  # [airports, out_len, 2]
        
        return x_seq, y_delays
    
    def _get_time_features(self, start_idx):
        """Generate time-based features"""
        time_indices = torch.arange(start_idx, start_idx + self.in_len, dtype=torch.float32)
        # Normalize by period (e.g., hour of day)
        time_features = (time_indices % self.period) / (self.period - 1)
        # Expand to match [in_len, airports, 1]
        time_features = time_features.unsqueeze(1).unsqueeze(2).expand(self.in_len, self.num_airports, 1)
        return time_features
    
    def get_adj_matrices(self):
        """Return adjacency matrices"""
        return self.adj_matrices
    
    def get_raw_data(self):
        """Return raw data for index-based training like training_c.py"""
        return self.data, self.weather_data

def create_datasets(data, weather_data, adj_matrices, in_len, out_len, 
                   train_ratio=0.7, val_ratio=0.15, scaler=None, add_time_features=True):
    """
    Create train, validation, and test datasets with sliding windows
    IMPORTANT: Normalization is done ONLY on training data to prevent data leakage
    
    Args:
        data: [time_steps, airports, features] - processed delay data (NOT normalized)
        weather_data: Weather data (can be None)
        adj_matrices: List of adjacency matrices  
        in_len: Input sequence length
        out_len: Output sequence length
        train_ratio: Training data ratio
        val_ratio: Validation data ratio
        scaler: StandardScaler for the data (will be fitted on training data only)
        add_time_features: Whether to add time features
    
    Returns:
        train_dataset, val_dataset, test_dataset, dataset_info
    """
    time_steps = data.shape[0]
    
    # Calculate valid indices for sequences
    total_valid_indices = time_steps - in_len - out_len + 1
    if total_valid_indices <= 0:
        raise ValueError(f"Not enough time steps ({time_steps}) for sequences "
                        f"(need at least {in_len + out_len})")
    
    # Create all possible starting indices
    all_indices = list(range(total_valid_indices))
    
    # Split indices for train/val/test
    train_size = int(train_ratio * len(all_indices))
    val_size = int(val_ratio * len(all_indices))
    
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:train_size + val_size]
    test_indices = all_indices[train_size + val_size:]
    
    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test sequences")
    
    # CRITICAL: Fit normalization ONLY on training data to prevent leakage
    fitted_scaler = None
    normalized_data = data.copy()  # Start with original data
    
    if scaler is not None or True:  # Always normalize to prevent overfitting
        from sklearn.preprocessing import StandardScaler
        fitted_scaler = StandardScaler()
        
        print("Fitting normalization on training data only...")
        
        # Extract only training sequences for normalization
        train_data_for_norm = []
        for idx in train_indices:
            train_seq = data[idx:idx + in_len]  # [in_len, airports, features]
            train_data_for_norm.append(train_seq)
        
        train_data_array = np.array(train_data_for_norm)  # [train_samples, in_len, airports, features]
        
        # Reshape for fitting: [samples*time*airports, features] 
        train_flattened = train_data_array.reshape(-1, data.shape[2])
        
        # Remove NaN values for fitting
        valid_mask = ~np.isnan(train_flattened).any(axis=1)
        valid_train_data = train_flattened[valid_mask]
        
        if len(valid_train_data) > 0:
            fitted_scaler.fit(valid_train_data)
            original_shape = data.shape
            data_flattened = data.reshape(-1, data.shape[2])
            normalized_flattened = fitted_scaler.transform(data_flattened)
            normalized_data = normalized_flattened.reshape(original_shape)
            normalized_data = np.nan_to_num(normalized_data, nan=0.0)
            print("Normalization applied successfully")
        else:
            print("Warning: No valid training data for normalization")
    
    # Create datasets using normalized data
    train_dataset = FlightDelayDataset(
        normalized_data, weather_data, adj_matrices, train_indices, in_len, out_len,
        normalize=True, scaler=fitted_scaler, add_time_features=add_time_features
    )
    
    val_dataset = FlightDelayDataset(
        normalized_data, weather_data, adj_matrices, val_indices, in_len, out_len,
        normalize=True, scaler=fitted_scaler, add_time_features=add_time_features
    )
    
    test_dataset = FlightDelayDataset(
        normalized_data, weather_data, adj_matrices, test_indices, in_len, out_len,
        normalize=True, scaler=fitted_scaler, add_time_features=add_time_features
    )
    
    dataset_info = {
        'total_time_steps': time_steps,
        'num_airports': data.shape[1],
        'num_features': data.shape[2],
        'train_size': len(train_indices),
        'val_size': len(val_indices),
        'test_size': len(test_indices),
        'input_dim': train_dataset[0][0].shape[2]  # Actual input features after processing
    }
    
    return train_dataset, val_dataset, test_dataset, dataset_info

# ===== Early Stopping Class =====
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=10, min_delta=0.0001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        self.best_weights = model.state_dict().copy()

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
def calculate_mae(pred, target):
    """Calculate Mean Absolute Error"""
    return torch.mean(torch.abs(pred - target)).item()

def calculate_denormalized_mae(pred, target, scaler):
    """Calculate MAE in original scale (real minutes)"""
    if scaler is None:
        return calculate_mae(pred, target)
    
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    if pred_np.shape[-1] >= 2:
        pred_denorm = scaler.inverse_transform(pred_np.reshape(-1, pred_np.shape[-1]))
        target_denorm = scaler.inverse_transform(target_np.reshape(-1, target_np.shape[-1]))
        pred_denorm = pred_denorm.reshape(pred_np.shape)
        target_denorm = target_denorm.reshape(target_np.shape)
        return np.mean(np.abs(pred_denorm - target_denorm))
    else:
        return calculate_mae(pred, target)

def test_error(pred, target):
    """Calculate comprehensive test metrics (MAE, RMSE, R2)"""
    # Remove NaN values
    mask = ~(np.isnan(pred) | np.isnan(target))
    pred_clean = pred[mask]
    target_clean = target[mask]
    
    if len(pred_clean) == 0:
        return np.nan, np.nan, np.nan
    
    # Mean Absolute Error
    mae = np.mean(np.abs(pred_clean - target_clean))
    
    # Root Mean Square Error
    rmse = np.sqrt(np.mean((pred_clean - target_clean) ** 2))
    
    # R-squared
    ss_res = np.sum((target_clean - pred_clean) ** 2)
    ss_tot = np.sum((target_clean - np.mean(target_clean)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return mae, rmse, r2

def evaluate_model_with_loader(model, data_loader, adj_matrices, criterion, device, scaler=None):
    """Evaluate model using DataLoader"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb, adj_matrices)
            loss = criterion(out, yb)
            mae = calculate_mae(out, yb)
            
            if scaler is not None:
                mae_denorm = calculate_denormalized_mae(out, yb, scaler)
            
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    return avg_loss, avg_mae

def evaluate_model(model, data, labels, adj_matrices, criterion, batch_size, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i:i+batch_size]
            yb = labels[i:i+batch_size]
            out = model(xb, adj_matrices)
            loss = criterion(out, yb)
            mae = calculate_mae(out, yb)
            
            total_loss += loss.item()
            total_mae += mae
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    return avg_loss, avg_mae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dataset", type=str, default="US", choices=["US", "China"], help="Choose dataset: US or China")
    parser.add_argument("--data_dir", type=str, default=".", help="Directory containing udata and cdata folders")
    parser.add_argument("--inlen", type=int, default=12)
    parser.add_argument("--outlen", type=int, default=12)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--episode", type=int, default=7)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--decay", type=float, default=1e-4)  # Increased weight decay
    parser.add_argument("--input_dim", type=int, default=None, help="Input dimension (will be auto-detected)")
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.4)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--use_mixup", action="store_true", help="Use mixup data augmentation")
    parser.add_argument("--mixup_alpha", type=float, default=0.5)
    parser.add_argument("--normalize", action="store_true", default=True, help="Normalize input data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save training results")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {args.dataset} flight delay dataset...")
    try:
        delay_data, weather_data, adj_matrices, scaler, num_airports, time_steps = load_flight_delay_data(
            dataset=args.dataset,
            data_dir=args.data_dir,
            sequence_length=args.inlen,
            prediction_length=args.outlen,
            normalize=args.normalize
        )
        
        print(f"Successfully loaded dataset")
        print(f"Time steps: {time_steps}, Airports: {num_airports}, Features: {delay_data.shape[2]}")
        
        # Create memory-efficient datasets using sliding windows
        train_dataset, val_dataset, test_dataset, dataset_info = create_datasets(
            data=delay_data,
            weather_data=weather_data,
            adj_matrices=adj_matrices,
            in_len=args.inlen,
            out_len=args.outlen,
            train_ratio=0.7,
            val_ratio=0.15,
            scaler=scaler,
            add_time_features=True  # Add time-based features
        )
        
        # Auto-detect input dimension
        if args.input_dim is None:
            args.input_dim = dataset_info['input_dim']
            print(f"Auto-detected input dimension: {args.input_dim}")
        
        # Get adjacency matrices for model
        adj_matrices = train_dataset.get_adj_matrices()
        
        print(f"Dataset info: {dataset_info['train_size']} train, {dataset_info['val_size']} val, {dataset_info['test_size']} test")
        
        use_real_data = True
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Falling back to synthetic data...")
        
        # Fallback to synthetic data if real data loading fails
        num_airports = 70
        total_samples = 500
        
        if args.input_dim is None:
            args.input_dim = 6
            
        # Create synthetic datasets
        synthetic_data = np.random.randn(total_samples, num_airports, args.inlen, args.input_dim)
        synthetic_labels = np.random.randn(total_samples, num_airports, args.outlen, 2)
        adj_matrices = [np.eye(num_airports) for _ in range(3)]
        
        # Convert to torch tensors
        full_data = torch.FloatTensor(synthetic_data)
        full_labels = torch.FloatTensor(synthetic_labels)
        
        # Create simple train/val/test split for synthetic data
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
        
        train_data = full_data[:train_size]
        train_labels = full_labels[:train_size]
        val_data = full_data[train_size:train_size + val_size]
        val_labels = full_labels[train_size:train_size + val_size]
        test_data = full_data[train_size + val_size:]
        test_labels = full_labels[train_size + val_size:]
        
        # Create simple tensor datasets (not using DataLoader for synthetic)
        dataset_info = {
            'train_size': train_size,
            'val_size': val_size,
            'test_size': len(test_data),
            'num_airports': num_airports,
            'input_dim': args.input_dim
        }
        
        use_real_data = False
        scaler = None
    
    # Convert adjacency matrices to tensors and move to device
    adj_matrices = [torch.FloatTensor(adj).to(device) for adj in adj_matrices]
    
    # Create DataLoaders for efficient training (only for real data) - OPTIMIZED
    if use_real_data:
        train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, 
                                 num_workers=0, pin_memory=False, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch*2, shuffle=False, 
                               num_workers=0, pin_memory=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch*2, shuffle=False, 
                                num_workers=0, pin_memory=False)
        print(f"DataLoaders created: {len(train_loader)} train batches, {len(val_loader)} val batches")
    else:
        # For synthetic data, move to device
        train_data = train_data.to(device)
        train_labels = train_labels.to(device)
        val_data = val_data.to(device)
        val_labels = val_labels.to(device)
        test_data = test_data.to(device)
        test_labels = test_labels.to(device)
        
        print(f"\n📊 Synthetic Data Shapes:")
        print(f"   Train: {train_data.shape}")
        print(f"   Validation: {val_data.shape}")
        print(f"   Test: {test_data.shape}")

    # Model
    model = DSAFNet(
        input_dim=args.input_dim, 
        hidden_dim=args.hidden_dim, 
        output_steps=args.outlen, 
        num_graphs=len(adj_matrices),
        dropout_rate=args.dropout_rate
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=args.early_stopping_patience, min_delta=0.0001)

    training_metrics = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_mae': [], 'val_mae': []}
    time_estimator = TrainingTimeEstimator(estimate_epochs=3, total_epochs=args.episode)
    time_estimator.start_training()

    print("Starting training...")
    print(f"Dropout: {args.dropout_rate}, Weight decay: {args.decay}, Early stopping patience: {args.early_stopping_patience}")
    print("-" * 60)
    
    # Prepare data for training_c.py style training
    if use_real_data:
        # Create batch indices like in training_c.py for shuffling
        batch_index = list(range(len(train_dataset)))
        val_index = list(range(len(val_dataset)))
        
        print(f"📊 Index-based training setup:")
        print(f"   Train samples: {len(train_dataset)}")
        print(f"   Validation samples: {len(val_dataset)}")
        print(f"   Test samples: {len(test_dataset)}")
    
    best_val_loss = float('inf')
    MAE_list = []
    best_model = None
    
    for ep in range(1, args.episode + 1):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_mae = 0.0
        num_train_batches = 0
        
        if use_real_data:
            # Shuffle batch indices like in training_c.py
            random.shuffle(batch_index)
            
            # Process batches using dataset's processed data
            for j in range(len(batch_index) // args.batch - 1):
                trainx = []
                trainy = []
                
                for k in range(args.batch):
                    # Use dataset's __getitem__ to get processed data with all features
                    sample_idx = j * args.batch + k
                    if sample_idx < len(train_dataset):
                        x_sample, y_sample = train_dataset[sample_idx]
                        trainx.append(x_sample.unsqueeze(0))  # Add batch dimension
                        trainy.append(y_sample.unsqueeze(0))  # Add batch dimension
                
                if not trainx:  # Skip if no valid samples
                    continue
                
                # Concatenate batch
                trainx = torch.cat(trainx, dim=0)  # [batch, airports, inlen, features]
                trainy = torch.cat(trainy, dim=0)  # [batch, airports, outlen, 2]
                
                # Move to device
                trainx = trainx.to(device)
                trainy = trainy.to(device)
                
                optimizer.zero_grad()
                output = model(trainx, adj_matrices)
                loss = criterion(output, trainy)
                mae = calculate_mae(output, trainy)
                
                if use_real_data and train_dataset.scaler is not None:
                    mae_denorm = calculate_denormalized_mae(output, trainy, train_dataset.scaler)
                else:
                    mae_denorm = mae
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)  # Use 3 like training_c.py
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_mae += mae_denorm  # Store denormalized MAE
                num_train_batches += 1
        else:
            # Use tensor-based training for synthetic data
            perm = torch.randperm(train_data.size(0))
            train_data_shuffled = train_data[perm]
            train_labels_shuffled = train_labels[perm]
            
            for i in range(0, train_data_shuffled.size(0), args.batch):
                xb = train_data_shuffled[i:i+args.batch]
                yb = train_labels_shuffled[i:i+args.batch]
                
                optimizer.zero_grad()
                out = model(xb, adj_matrices)
                loss = criterion(out, yb)
                mae = calculate_mae(out, yb)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
                optimizer.step()
                
                epoch_train_loss += loss.item()
                epoch_train_mae += mae
                num_train_batches += 1
        
        # Calculate average training metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_train_mae = epoch_train_mae / num_train_batches
        
        # Validation phase - following training_c.py pattern with dataset
        if use_real_data:
            model.eval()
            outputs = []
            labels = []
            
            with torch.no_grad():
                # Use validation dataset to get processed data
                for i in range(min(len(val_dataset), len(val_index))):
                    # Get processed validation sample
                    testx, testy = val_dataset[i]
                    testx = testx.unsqueeze(0).to(device)  # Add batch dimension
                    
                    # Forward pass
                    output = model(testx, adj_matrices)
                    output = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
                    
                    # Store outputs and labels for evaluation
                    outputs.append(np.expand_dims(output, axis=0))
                    labels.append(np.expand_dims(testy.numpy(), axis=0))
            
            # Concatenate all validation outputs
            yhat = np.concatenate(outputs)  # [samples, airports, outlen, 2]
            val_label = np.concatenate(labels)  # [samples, airports, outlen, 2]
            
            # DENORMALIZE predictions and labels to get real minutes
            dataset_scaler = val_dataset.scaler
            if dataset_scaler is not None:
                original_shape = yhat.shape
                yhat_reshaped = yhat.reshape(-1, yhat.shape[-1])
                val_label_reshaped = val_label.reshape(-1, val_label.shape[-1])
                yhat_denorm = dataset_scaler.inverse_transform(yhat_reshaped)
                val_label_denorm = dataset_scaler.inverse_transform(val_label_reshaped)
                yhat = yhat_denorm.reshape(original_shape)
                val_label = val_label_denorm.reshape(original_shape)
            
            # Calculate validation metrics
            val_mae_list = []
            val_r2_list = []
            val_rmse_list = []
            for i in range(args.outlen):
                val_metrics = test_error(yhat[:, :, i, :], val_label[:, :, i, :])
                if val_metrics:
                    val_mae_list.append(val_metrics[0])  # MAE in real minutes
                    val_rmse_list.append(val_metrics[1])  # RMSE in real minutes
                    val_r2_list.append(val_metrics[2])  # R2
            
            val_mae = np.mean(val_mae_list) if val_mae_list else float('inf')
            val_rmse = np.mean(val_rmse_list) if val_rmse_list else float('inf')
            val_r2 = np.mean(val_r2_list) if val_r2_list else 0.0
            val_loss = val_mae  # Use MAE as validation loss (now in real minutes)
            val_mae_denorm = val_mae  # Already denormalized to real minutes
            
            # Store in MAE_list like training_c.py
            MAE_list.append(val_mae)
            
            # Print validation results like training_c.py (now in REAL MINUTES)
            log = 'On average over all horizons, Val MAE: {:.4f} min, Val R2: {:.4f}, Val RMSE: {:.4f} min'
            print(log.format(val_mae, val_r2, val_rmse))
        else:
            val_loss, val_mae = evaluate_model(model, val_data, val_labels, adj_matrices, criterion, args.batch, device)
            val_mae_denorm = None
            MAE_list.append(val_mae)
        
        # Store metrics
        training_metrics['epoch'].append(ep)
        training_metrics['train_loss'].append(avg_train_loss)
        training_metrics['val_loss'].append(val_loss)
        training_metrics['train_mae'].append(avg_train_mae)
        training_metrics['val_mae'].append(val_mae)
        
        # Early stopping check
        early_stopping(val_loss, model)
        
        # Record epoch time and get estimates
        time_estimator.record_epoch_time(ep - 1, epoch_start_time)  # ep-1 for 0-based indexing
        
        # Save best model like training_c.py
        if use_real_data and val_mae == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
        elif not use_real_data and val_mae < best_val_loss:
            best_model = copy.deepcopy(model.state_dict())
            best_val_loss = val_mae
        
        # Enhanced progress reporting - simplified like training_c.py
        improvement_indicator = "✅" if (use_real_data and val_mae == min(MAE_list)) or (not use_real_data and val_mae < best_val_loss) else "  "
        
        # Display progress - all metrics now in REAL MINUTES (denormalized)
        if val_mae_denorm is not None:
            print(f"Epoch {ep:3d} {improvement_indicator} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.4f} min | "
                  f"Train MAE: {avg_train_mae:.4f} min | Val MAE: {val_mae:.4f} min")
        else:
            print(f"Epoch {ep:3d} {improvement_indicator} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Train MAE: {avg_train_mae:.6f} | Val MAE: {val_mae:.6f}")
        
        if early_stopping.early_stop:
            print(f"\n🛑 Early stopping triggered at epoch {ep}")
            print(f"📈 Best validation loss: {early_stopping.best_loss:.6f}")
            break
    
    if best_model is not None:
        model.load_state_dict(best_model)
        print("Loaded best model weights")
    
    final_stats = time_estimator.finish_training()
    
    print(f"\nTraining stopped at epoch {len(training_metrics['epoch'])}")
    if early_stopping.early_stop:
        print(f"Best validation loss: {early_stopping.best_loss:.6f}")
    
    final_train_loss = training_metrics['train_loss'][-1]
    final_val_loss = training_metrics['val_loss'][-1]
    final_train_mae = training_metrics['train_mae'][-1]
    final_val_mae = training_metrics['val_mae'][-1]
    
    print(f"\nFinal Performance:")
    print(f"  Train Loss: {final_train_loss:.6f} | Val Loss: {final_val_loss:.6f}")
    print(f"  Train MAE:  {final_train_mae:.4f} min | Val MAE:  {final_val_mae:.4f} min")
    
    train_val_gap = final_val_loss - final_train_loss
    print(f"\nTraining Analysis:")
    print(f"  Train-Val Gap: {train_val_gap:.6f}")
    
    if train_val_gap < 0.01:
        print("  Status: Excellent stability")
    elif train_val_gap < 0.05:
        print("  Status: Moderate stability")
    else:
        print("  Status: Needs improvement")
    
    print("="*60)
    
    training_results = {
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'final_train_mae': final_train_mae,
        'final_val_mae': final_val_mae,
        'train_val_gap': train_val_gap,
        'early_stopped': early_stopping.early_stop,
        'best_val_loss': early_stopping.best_loss,
        'epochs_trained': len(training_metrics['epoch']),
        'dropout_rate': args.dropout_rate,
        'weight_decay': args.decay,
        'used_mixup': args.use_mixup
    }
    
    # Save metrics to Excel file
    df = pd.DataFrame(training_metrics)
    excel_path = os.path.join(args.output_dir, 'training_metrics_regularized.xlsx')
    df.to_excel(excel_path, index=False)
    print(f"Training metrics saved to: {excel_path}")
    
    results_df = pd.DataFrame([training_results])
    results_excel_path = os.path.join(args.output_dir, 'training_results_regularized.xlsx')
    results_df.to_excel(results_excel_path, index=False)
    print(f"Training results saved to: {results_excel_path}")

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
        time_stats_path = os.path.join(args.output_dir, 'training_time_statistics_regularized.xlsx')
        time_stats.to_excel(time_stats_path, index=False)
        print(f"Training time statistics saved to: {time_stats_path}")

    # Save model
    model_path = os.path.join(args.output_dir, "dsafnet_flight_delay_regularized.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Training Summary:")
    print(f"  Epochs trained: {len(training_metrics['epoch'])}")
    print(f"  Early stopping: {'Yes' if early_stopping.early_stop else 'No'}")
    print(f"  Final Train MAE: {final_train_mae:.4f} min")
    print(f"  Final Val MAE: {final_val_mae:.4f} min")
    print(f"  Train-Val Gap: {train_val_gap:.6f}")
    print(f"\nTo test the model: python dsafnetcopy_test.py --dataset {args.dataset}")
    print("="*80)

if __name__ == "__main__":
    main()
