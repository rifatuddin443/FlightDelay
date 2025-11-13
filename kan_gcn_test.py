"""
KAN-GCN Test Script for Flight Delay Prediction
Loads a trained model and evaluates multi-horizon predictions (3, 6, 12 steps)
"""

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import csv
import sys
import os

# Add efficient-kan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
from kan import KAN

from baseline_methods import test_error, StandardScaler

# ============================
# Configuration
# ============================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64
SEQUENCE_LENGTH = 12
BASE_FEATURES = 2  # arrival_delay, departure_delay
HIDDEN_CHANNELS = 64
OUT_CHANNELS = BASE_FEATURES

# Model path
MODEL_PATH = 'kan_gcn_best.pth'

# Data paths
DATA_DIR = 'udata'
DELAY_FILE = os.path.join(DATA_DIR, 'udelay.npy')
ADJ_FILE = os.path.join(DATA_DIR, 'adj_mx.npy')

print(f"Using device: {DEVICE}")
print(f"Model path: {MODEL_PATH}")

# ============================
# Model Definition
# ============================
class KAN_GCN(nn.Module):
    """
    KAN-GCN Model: Graph Convolutional Network + Kolmogorov-Arnold Network
    """
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(KAN_GCN, self).__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        self.kan = KAN(layers_hidden=[hidden_channels, hidden_channels // 2, out_channels])
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.gcn(x, edge_index)
        x = torch.relu(x)
        x = self.kan(x)
        return x


# ============================
# Data Loading
# ============================
def load_flight_data():
    """
    Load flight delay data and adjacency matrix.
    Returns train/val/test splits with proper temporal ordering.
    """
    print("\n--- Loading Data ---")
    
    # Load delay data
    if not os.path.exists(DELAY_FILE):
        raise FileNotFoundError(f"Delay file not found: {DELAY_FILE}")
    
    delay_data = np.load(DELAY_FILE)
    print(f"Loaded delay data: {delay_data.shape}")  # (num_nodes, timesteps, features)
    
    # Load adjacency matrix
    if not os.path.exists(ADJ_FILE):
        raise FileNotFoundError(f"Adjacency file not found: {ADJ_FILE}")
    
    adj_mx = np.load(ADJ_FILE)
    print(f"Loaded adjacency matrix: {adj_mx.shape}")
    
    # Create edge index for PyTorch Geometric
    edge_index = []
    for i in range(adj_mx.shape[0]):
        for j in range(adj_mx.shape[1]):
            if adj_mx[i, j] > 0:
                edge_index.append([i, j])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    print(f"Created edge index: {edge_index.shape}")
    
    # Temporal split: 70% train, 10% val, 20% test
    num_samples = delay_data.shape[1]
    train_size = int(0.7 * num_samples)
    val_size = int(0.1 * num_samples)
    
    train_data = delay_data[:, :train_size, :]
    val_data = delay_data[:, train_size:train_size+val_size, :]
    test_data = delay_data[:, train_size+val_size:, :]
    
    print(f"Train shape: {train_data.shape}")
    print(f"Val shape: {val_data.shape}")
    print(f"Test shape: {test_data.shape}")
    
    return train_data, val_data, test_data, edge_index


# ============================
# Multi-Horizon Testing
# ============================
def test_multihorizon(model, test_data, edge_index, scaler, horizons=[3, 6, 12]):
    """
    Test the model on multiple prediction horizons.
    
    Args:
        model: Trained KAN-GCN model
        test_data: Test dataset (num_nodes, timesteps, features)
        edge_index: Graph edge indices
        scaler: Fitted StandardScaler for normalization
        horizons: List of prediction horizons to evaluate
    
    Returns:
        Dictionary of results for each horizon
    """
    model.eval()
    num_nodes = test_data.shape[0]
    results = {}
    
    print("\n" + "="*80)
    print("MULTI-HORIZON TESTING")
    print("="*80)
    
    for horizon in horizons:
        print(f"\nEvaluating {horizon}-step ahead predictions...")
        max_test_idx = test_data.shape[1] - SEQUENCE_LENGTH - horizon
        test_x_list = []
        test_y_list = []
        
        # Create test sequences
        for t in range(max_test_idx):
            x_seq = test_data[:, t:t+SEQUENCE_LENGTH, :]
            x_seq = scaler.transform(x_seq)
            x_seq[np.isnan(x_seq)] = 0
            x_seq = x_seq.reshape(num_nodes, -1)
            
            # Get target for this horizon
            y_seq = test_data[:, t+SEQUENCE_LENGTH:t+SEQUENCE_LENGTH+horizon, :]
            y_seq = y_seq.reshape(num_nodes, -1)
            y_seq[np.isnan(y_seq)] = 0
            
            test_x_list.append(x_seq)
            test_y_list.append(y_seq)
        
        test_x_all = torch.tensor(np.stack(test_x_list), dtype=torch.float)
        test_y_all = torch.tensor(np.stack(test_y_list), dtype=torch.float)
        
        print(f"Created {len(test_x_list)} test sequences")
        
        # Predictions for this horizon
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            num_test_batches = (len(test_x_all) + BATCH_SIZE - 1) // BATCH_SIZE
            
            for batch_idx in range(num_test_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(test_x_all))
                
                batch_x = test_x_all[start_idx:end_idx]
                batch_y = test_y_all[start_idx:end_idx]
                
                for i in range(len(batch_x)):
                    x_single = batch_x[i].to(DEVICE)
                    
                    # Iterative multi-step prediction
                    predictions_for_horizon = []
                    current_input = x_single.clone()
                    
                    for step in range(horizon):
                        data = Data(x=current_input, edge_index=edge_index)
                        out = model(data)
                        predictions_for_horizon.append(out.cpu().numpy())
                        
                        # Update input for next step
                        if step < horizon - 1:
                            current_input = torch.cat([
                                current_input[:, BASE_FEATURES:],
                                out
                            ], dim=1)
                    
                    # Stack predictions
                    pred_horizon = np.concatenate(predictions_for_horizon, axis=1)
                    all_preds.append(pred_horizon)
                    all_targets.append(batch_y[i].numpy())
        
        # Convert to arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Denormalize predictions
        preds_denorm = scaler.inverse_transform(
            all_preds.reshape(-1, BASE_FEATURES)
        ).reshape(all_preds.shape)
        targets_denorm = all_targets
        
        # Reshape to (num_sequences, num_nodes, horizon, features)
        preds_denorm = preds_denorm.reshape(-1, num_nodes, horizon, BASE_FEATURES)
        targets_denorm = targets_denorm.reshape(-1, num_nodes, horizon, BASE_FEATURES)
        
        # Calculate metrics for the final timestep of this horizon
        # Arrival delay (feature 0)
        arrival_preds = preds_denorm[:, :, horizon-1, 0]
        arrival_targets = targets_denorm[:, :, horizon-1, 0]
        arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
        
        # Departure delay (feature 1)
        departure_preds = preds_denorm[:, :, horizon-1, 1]
        departure_targets = targets_denorm[:, :, horizon-1, 1]
        dep_mae, dep_rmse, dep_r2 = test_error(departure_preds, departure_targets)
        
        results[horizon] = {
            'arr_mae': arr_mae,
            'arr_rmse': arr_rmse,
            'arr_r2': arr_r2,
            'dep_mae': dep_mae,
            'dep_rmse': dep_rmse,
            'dep_r2': dep_r2,
            'predictions': preds_denorm,
            'targets': targets_denorm
        }
        
        print(f"  Arrival  - MAE: {arr_mae:.4f}, RMSE: {arr_rmse:.4f}, R²: {arr_r2:.4f}")
        print(f"  Departure - MAE: {dep_mae:.4f}, RMSE: {dep_rmse:.4f}, R²: {dep_r2:.4f}")
    
    return results


def print_results_table(results):
    """Print formatted results table"""
    print("\n" + "="*80)
    print("SUMMARY TABLE - KAN-GCN Flight Delay Predictions")
    print("="*80)
    print(f"{'Horizon':<10} {'Delay Type':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-"*80)
    
    for horizon in sorted(results.keys()):
        res = results[horizon]
        print(f"{horizon}-step    {'Arrival':<15} {res['arr_mae']:<12.4f} {res['arr_rmse']:<12.4f} {res['arr_r2']:<12.4f}")
        print(f"{'':10} {'Departure':<15} {res['dep_mae']:<12.4f} {res['dep_rmse']:<12.4f} {res['dep_r2']:<12.4f}")
    
    print("="*80)


def save_results(results, output_file='kan_gcn_test_results.csv'):
    """Save results to CSV file"""
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['horizon', 'arr_mae', 'arr_rmse', 'arr_r2', 
                        'dep_mae', 'dep_rmse', 'dep_r2'])
        
        for horizon in sorted(results.keys()):
            res = results[horizon]
            writer.writerow([
                horizon,
                res['arr_mae'], res['arr_rmse'], res['arr_r2'],
                res['dep_mae'], res['dep_rmse'], res['dep_r2']
            ])
    
    print(f"\n✓ Results saved to: {output_file}")


# ============================
# Main Execution
# ============================
if __name__ == '__main__':
    try:
        # Check if model exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}\nPlease train the model first using kan_gcn_model.py")
        
        # Load data
        train_data, val_data, test_data, edge_index = load_flight_data()
        num_nodes = test_data.shape[0]
        edge_index = edge_index.to(DEVICE)
        
        # Fit scaler on training data
        print("\n--- Fitting StandardScaler ---")
        train_reshaped = train_data.reshape(-1, BASE_FEATURES)
        mean = np.mean(train_reshaped[~np.isnan(train_reshaped)])
        std = np.std(train_reshaped[~np.isnan(train_reshaped)])
        scaler = StandardScaler(mean=mean, std=std)
        print(f"Scaler mean: {scaler.mean}")
        print(f"Scaler std: {scaler.std}")
        
        # Initialize model
        print("\n--- Loading Model ---")
        in_channels = SEQUENCE_LENGTH * BASE_FEATURES
        model = KAN_GCN(in_channels, HIDDEN_CHANNELS, OUT_CHANNELS).to(DEVICE)
        
        # Load trained weights
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✓ Model loaded from: {MODEL_PATH}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Run multi-horizon testing
        results = test_multihorizon(
            model, test_data, edge_index, scaler, 
            horizons=[3, 6, 12]
        )
        
        # Display results
        print_results_table(results)
        
        # Save results
        save_results(results, output_file='kan_gcn_test_results.csv')
        
        print("\n✓ Testing complete!")
        
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
