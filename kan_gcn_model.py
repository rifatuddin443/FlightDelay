import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import sys
import os
# Add efficient-kan to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
from kan import KAN
import numpy as np
import csv
import math
from torch_geometric.data import Data
import time
from tqdm import tqdm

class KAN_GCN(nn.Module):
    """
    A simple architecture combining a GCN layer with a KAN layer for flight delay prediction.
    Uses efficient-kan implementation for better performance.
    """
    def __init__(self, in_channels, out_channels, hidden_channels=64, grid_size=5, spline_order=3):
        super(KAN_GCN, self).__init__()
        self.gcn = GCNConv(in_channels, hidden_channels)
        
        # The KAN layer will take the node features from the GCN layer as input.
        # layers_hidden defines the network architecture: [input_dim, hidden_dim, output_dim]
        self.kan = KAN(
            layers_hidden=[hidden_channels, hidden_channels // 2, out_channels],
            grid_size=grid_size,
            spline_order=spline_order
        )

    def forward(self, data):
        """
        Forward pass for the KAN-GCN model.
        
        Args:
            data (torch_geometric.data.Data): A PyG Data object containing:
                - x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels]
                - edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges]
        
        Returns:
            torch.Tensor: The predicted flight delay for each node, shape [num_nodes, out_channels]
        """
        x, edge_index = data.x, data.edge_index
        
        # Apply GCN layer and ReLU activation
        x = self.gcn(x, edge_index)
        x = F.relu(x)
        
        # Apply KAN layer for prediction
        x = self.kan(x)
        
        return x

class StandardScaler:
    """Standardize features by removing the mean and scaling to unit variance."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def evaluate_in_batches(model, x_data, y_data, edge_index, device, batch_size, scaler, out_channels):
    """
    Evaluate model on data in batches for efficiency.
    
    Returns:
        predictions, targets (both denormalized)
    """
    predictions = []
    targets = []
    
    num_batches = (len(x_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(x_data))
        
        batch_x = x_data[start_idx:end_idx]
        batch_y = y_data[start_idx:end_idx]
        
        for i in range(len(batch_x)):
            x_single = batch_x[i].to(device)
            y_single = batch_y[i]
            
            data = Data(x=x_single, edge_index=edge_index)
            out = model(data)
            
            predictions.append(out.cpu().numpy())
            targets.append(y_single.numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Denormalize predictions
    preds_denorm = scaler.inverse_transform(predictions.reshape(-1, out_channels)).reshape(predictions.shape)
    targets_denorm = targets  # Already in original scale
    
    return preds_denorm, targets_denorm


def load_flight_data(data_dir='cdata', train_ratio=0.7, val_ratio=0.1):
    """
    Loads flight data following the same pattern as training_u.py.
    
    Args:
        data_dir (str): Directory where the data files are located.
        train_ratio (float): Ratio of data to use for training.
        val_ratio (float): Ratio of data to use for validation.
    
    Returns:
        tuple: (edge_index, train_data, val_data, test_data, scaler, num_nodes)
    """
    print("Loading flight data...")
    
    # Load adjacency matrix
    #adj_mx_path = os.path.join(data_dir, 'adj_mx.npy')
    adj_mx_path = os.path.join(data_dir, 'dist_mx.npy')
    if not os.path.exists(adj_mx_path):
        raise FileNotFoundError(f"Adjacency matrix not found at {adj_mx_path}")
    adj_mx = np.load(adj_mx_path)
    num_nodes = adj_mx.shape[0]
    
    # Create edge_index from adjacency matrix for PyG
    edge_index = torch.tensor(np.array(adj_mx.nonzero()), dtype=torch.long)

    # Load delay data - shape: (num_nodes, time_steps, 2)
    # where [:,:,0] is arrival delay and [:,:,1] is departure delay
    #delay_path = os.path.join(data_dir, 'udelay.npy')
    delay_path = os.path.join(data_dir, 'delay.npy')
    if not os.path.exists(delay_path):
        raise FileNotFoundError(f"Delay data not found at {delay_path}")
    
    data = np.load(delay_path)
    print(f"Raw data shape: {data.shape}")  # (nodes, time_steps, features)
    
    # Split data temporally (as in training_u.py)
    training_data = data[:, :int(train_ratio * data.shape[1]), :]
    val_data = data[:, int(train_ratio * data.shape[1]):int((train_ratio + val_ratio) * data.shape[1]), :]
    test_data = data[:, int((train_ratio + val_ratio) * data.shape[1]):, :]
    
    # Create scaler from training data (excluding NaN values)
    scaler = StandardScaler(
        training_data[~np.isnan(training_data)].mean(), 
        training_data[~np.isnan(training_data)].std()
    )
    
    # Scale training data and replace NaN with 0
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    
    print(f"Data loaded: {num_nodes} nodes, {edge_index.shape[1]} edges.")
    print(f"Training data shape: {training_data.shape}")
    print(f"Validation data shape: {val_data.shape}")
    print(f"Test data shape: {test_data.shape}")

    return edge_index, training_data, val_data, test_data, scaler, num_nodes


if __name__ == '__main__':
    # --- Configuration ---
    DATA_DIR = 'cdata'
    SEQUENCE_LENGTH = 12  # Use last 12 time steps as features
    PREDICT_HORIZON = 1   # Predict next time step
    BASE_FEATURES = 2  # arrival delay + departure delay
    IN_CHANNELS = SEQUENCE_LENGTH * BASE_FEATURES  # Flattened input
    OUT_CHANNELS = PREDICT_HORIZON * BASE_FEATURES  # Flattened output
    HIDDEN_CHANNELS = 64
    LEARNING_RATE = 0.005
    EPOCHS = 25
    BATCH_SIZE = 64  # Process multiple sequences in parallel
    NUM_WORKERS = 4  # For parallel data loading
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")

    # --- Data Loading ---
    try:
        edge_index, training_data, val_data, test_data, scaler, num_nodes = load_flight_data(
            data_dir=DATA_DIR, train_ratio=0.7, val_ratio=0.1
        )

        # --- Model, Optimizer, and Loss Function ---
        model = KAN_GCN(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS, hidden_channels=HIDDEN_CHANNELS).to(DEVICE)
        
        # Note: torch.compile disabled to avoid Triton dependency
        # Model will run in eager mode which is still optimized with batching and GPU
        print("Running model in eager mode (torch.compile disabled)")
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        criterion = nn.MSELoss()

        print("\n--- Starting Training ---")
        print(f"Preparing data with batch size {BATCH_SIZE}...")

        # Pre-create all sequences as tensors for faster batch processing
        # Training sequences
        max_train_idx = training_data.shape[1] - SEQUENCE_LENGTH - PREDICT_HORIZON
        train_x_list = []
        train_y_list = []
        
        for t in range(max_train_idx):
            x_seq = training_data[:, t:t+SEQUENCE_LENGTH, :].reshape(num_nodes, -1)
            y_seq = training_data[:, t+SEQUENCE_LENGTH:t+SEQUENCE_LENGTH+PREDICT_HORIZON, :].reshape(num_nodes, -1)
            train_x_list.append(x_seq)
            train_y_list.append(y_seq)
        
        # Stack into large tensors for vectorized operations
        train_x_all = torch.tensor(np.stack(train_x_list), dtype=torch.float)  # (num_seq, num_nodes, features)
        train_y_all = torch.tensor(np.stack(train_y_list), dtype=torch.float)
        
        print(f"Created {len(train_x_list)} training sequences (shape: {train_x_all.shape})")
        
        # Validation sequences
        max_val_idx = val_data.shape[1] - SEQUENCE_LENGTH - PREDICT_HORIZON
        val_x_list = []
        val_y_list = []
        
        for t in range(max_val_idx):
            x_seq = val_data[:, t:t+SEQUENCE_LENGTH, :]
            x_seq = scaler.transform(x_seq)
            x_seq[np.isnan(x_seq)] = 0
            x_seq = x_seq.reshape(num_nodes, -1)
            
            # Get target and replace NaN values with 0
            y_seq = val_data[:, t+SEQUENCE_LENGTH:t+SEQUENCE_LENGTH+PREDICT_HORIZON, :].reshape(num_nodes, -1)
            y_seq[np.isnan(y_seq)] = 0  # Replace NaN with 0
            
            val_x_list.append(x_seq)
            val_y_list.append(y_seq)
        
        val_x_all = torch.tensor(np.stack(val_x_list), dtype=torch.float)
        val_y_all = torch.tensor(np.stack(val_y_list), dtype=torch.float)
        
        print(f"Created {len(val_x_list)} validation sequences (shape: {val_x_all.shape})")

        history = []
        best_val_mae = float('inf')
        
        # Move edge_index to device
        edge_index = edge_index.to(DEVICE)
        
        num_train_batches = (len(train_x_all) + BATCH_SIZE - 1) // BATCH_SIZE
        num_val_batches = (len(val_x_all) + BATCH_SIZE - 1) // BATCH_SIZE

        # --- Training Loop with Batching ---
        print("\nStarting training loop...")
        for epoch in range(EPOCHS):
            epoch_start_time = time.time()
            model.train()
            epoch_losses = []
            epoch_train_maes = []
            epoch_train_rmses = []
            
            # Shuffle training data
            perm = torch.randperm(len(train_x_all))
            train_x_shuffled = train_x_all[perm]
            train_y_shuffled = train_y_all[perm]
            
            # Process in batches with progress bar
            pbar = tqdm(range(num_train_batches), desc=f'Epoch {epoch+1}/{EPOCHS}', leave=False)
            for batch_idx in pbar:
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min(start_idx + BATCH_SIZE, len(train_x_all))
                
                # Get batch
                batch_x = train_x_shuffled[start_idx:end_idx]  # (batch_size, num_nodes, features)
                batch_y = train_y_shuffled[start_idx:end_idx]
                
                # Process each sequence in batch (PyG processes one graph at a time)
                batch_losses = []
                batch_preds = []
                batch_targets = []
                
                for i in range(len(batch_x)):
                    optimizer.zero_grad()
                    
                    x_single = batch_x[i].to(DEVICE)
                    y_single = batch_y[i].to(DEVICE)
                    
                    data = Data(x=x_single, edge_index=edge_index, y=y_single)
                    
                    # Forward pass
                    out = model(data)
                    
                    # Compute loss
                    loss = criterion(out, y_single)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    batch_losses.append(loss.item())
                    batch_preds.append(out.detach().cpu().numpy())
                    batch_targets.append(y_single.cpu().numpy())
                
                epoch_losses.extend(batch_losses)
                
                # Compute metrics for batch (denormalized)
                batch_preds = np.array(batch_preds)
                batch_targets = np.array(batch_targets)
                
                batch_preds_denorm = scaler.inverse_transform(batch_preds.reshape(-1, OUT_CHANNELS)).reshape(batch_preds.shape)
                batch_targets_denorm = scaler.inverse_transform(batch_targets.reshape(-1, OUT_CHANNELS)).reshape(batch_targets.shape)
                
                batch_mae = np.mean(np.abs(batch_preds_denorm - batch_targets_denorm))
                batch_rmse = np.sqrt(np.mean((batch_preds_denorm - batch_targets_denorm) ** 2))
                
                epoch_train_maes.append(batch_mae)
                epoch_train_rmses.append(batch_rmse)
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{np.mean(batch_losses):.4f}', 'mae': f'{batch_mae:.4f}'})
            
            # Validation with batching
            model.eval()
            with torch.no_grad():
                val_preds_denorm, val_targets_denorm = evaluate_in_batches(
                    model, val_x_all, val_y_all, edge_index, DEVICE, BATCH_SIZE, scaler, OUT_CHANNELS
                )
            
            # Compute validation metrics (NaN values have been replaced with 0)
            val_mae = np.mean(np.abs(val_preds_denorm - val_targets_denorm))
            val_rmse = np.sqrt(np.mean((val_preds_denorm - val_targets_denorm) ** 2))
            
            # Aggregate epoch metrics
            epoch_time = time.time() - epoch_start_time
            avg_train_loss = np.mean(epoch_losses)
            avg_train_mae = np.mean(epoch_train_maes)
            avg_train_rmse = np.mean(epoch_train_rmses)
            
            history.append((epoch+1, avg_train_loss, avg_train_mae, avg_train_rmse, val_mae, val_rmse))
            
            if (epoch + 1) % 2 == 0 or epoch == 0:
                print(f'Epoch [{epoch+1}/{EPOCHS}] ({epoch_time:.1f}s), Loss: {avg_train_loss:.4f}, '
                      f'Train MAE: {avg_train_mae:.4f}, Val MAE: {val_mae:.4f}')
            
            # Save best model (only if val_mae is valid)
            if not np.isnan(val_mae) and not np.isinf(val_mae) and val_mae < best_val_mae:
                best_val_mae = val_mae
                torch.save(model.state_dict(), 'kan_gcn_best.pth')
                print(f"  -> New best model saved! Val MAE: {best_val_mae:.4f}")

        print("--- Training Finished ---")

        # --- Test Set Evaluation with Multiple Horizons ---
        print("\n--- Testing on Test Set ---")
        
        # Define prediction horizons (3, 6, 12 steps)
        PREDICT_HORIZONS = [3, 6, 12]
        
        # Load best model
        model.load_state_dict(torch.load('kan_gcn_best.pth'))
        model.eval()
        
        # Store results for each horizon
        test_results = {}
        
        for horizon in PREDICT_HORIZONS:
            print(f"\nEvaluating {horizon}-step ahead predictions...")
            max_test_idx = test_data.shape[1] - SEQUENCE_LENGTH - horizon
            test_x_list = []
            test_y_list = []
            
            for t in range(max_test_idx):
                x_seq = test_data[:, t:t+SEQUENCE_LENGTH, :]
                x_seq = scaler.transform(x_seq)
                x_seq[np.isnan(x_seq)] = 0
                x_seq = x_seq.reshape(num_nodes, -1)
                
                # Get target for this horizon
                y_seq = test_data[:, t+SEQUENCE_LENGTH:t+SEQUENCE_LENGTH+horizon, :].reshape(num_nodes, -1)
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
                        
                        # For multi-step prediction, we need to predict iteratively
                        predictions_for_horizon = []
                        current_input = x_single.clone()
                        
                        for step in range(horizon):
                            data = Data(x=current_input, edge_index=edge_index)
                            out = model(data)
                            predictions_for_horizon.append(out.cpu().numpy())
                            
                            # Update input for next step (shift and append prediction)
                            if step < horizon - 1:
                                # Shift the input window: remove first timestep, add prediction
                                current_input = torch.cat([
                                    current_input[:, BASE_FEATURES:],  # Remove first timestep
                                    out
                                ], dim=1)
                        
                        # Stack predictions: shape (num_nodes, horizon*2)
                        pred_horizon = np.concatenate(predictions_for_horizon, axis=1)
                        all_preds.append(pred_horizon)
                        all_targets.append(batch_y[i].numpy())
            
            # Convert to arrays
            all_preds = np.array(all_preds)
            all_targets = np.array(all_targets)
            
            # Denormalize
            preds_denorm = scaler.inverse_transform(all_preds.reshape(-1, BASE_FEATURES)).reshape(all_preds.shape)
            targets_denorm = all_targets
            
            # Reshape to (num_sequences, num_nodes, horizon, features)
            preds_denorm = preds_denorm.reshape(-1, num_nodes, horizon, BASE_FEATURES)
            targets_denorm = targets_denorm.reshape(-1, num_nodes, horizon, BASE_FEATURES)
            
            test_results[horizon] = {
                'predictions': preds_denorm,
                'targets': targets_denorm
            }
        
        # Calculate and display metrics for each horizon
        print("\n" + "="*80)
        print("TEST RESULTS - Multiple Horizon Predictions")
        print("="*80)
        
        from baseline_methods import test_error
        
        results_table = []
        
        for horizon in PREDICT_HORIZONS:
            preds = test_results[horizon]['predictions']
            targets = test_results[horizon]['targets']
            
            # For arrival delay (feature 0)
            arrival_preds = preds[:, :, horizon-1, 0]  # Last step prediction
            arrival_targets = targets[:, :, horizon-1, 0]
            arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
            
            # For departure delay (feature 1)
            departure_preds = preds[:, :, horizon-1, 1]
            departure_targets = targets[:, :, horizon-1, 1]
            dep_mae, dep_rmse, dep_r2 = test_error(departure_preds, departure_targets)
            
            results_table.append({
                'horizon': horizon,
                'arr_mae': arr_mae,
                'arr_rmse': arr_rmse,
                'arr_r2': arr_r2,
                'dep_mae': dep_mae,
                'dep_rmse': dep_rmse,
                'dep_r2': dep_r2
            })
            
            print(f"\n{horizon}-step ahead ARRIVAL delay:")
            print(f"  MAE: {arr_mae:.4f} min, RMSE: {arr_rmse:.4f} min, R²: {arr_r2:.4f}")
            
            print(f"{horizon}-step ahead DEPARTURE delay:")
            print(f"  MAE: {dep_mae:.4f} min, RMSE: {dep_rmse:.4f} min, R²: {dep_r2:.4f}")
        
        # Overall metrics (1-step ahead from original evaluation)
        print("\n" + "="*80)
        
        # Compute overall 1-step metrics
        max_test_idx = test_data.shape[1] - SEQUENCE_LENGTH - 1
        test_x_list = []
        test_y_list = []
        
        for t in range(max_test_idx):
            x_seq = test_data[:, t:t+SEQUENCE_LENGTH, :]
            x_seq = scaler.transform(x_seq)
            x_seq[np.isnan(x_seq)] = 0
            x_seq = x_seq.reshape(num_nodes, -1)
            y_seq = test_data[:, t+SEQUENCE_LENGTH:t+SEQUENCE_LENGTH+1, :].reshape(num_nodes, -1)
            y_seq[np.isnan(y_seq)] = 0
            test_x_list.append(x_seq)
            test_y_list.append(y_seq)
        
        test_x_all = torch.tensor(np.stack(test_x_list), dtype=torch.float)
        test_y_all = torch.tensor(np.stack(test_y_list), dtype=torch.float)
        
        with torch.no_grad():
            test_preds_denorm, test_targets_denorm = evaluate_in_batches(
                model, test_x_all, test_y_all, edge_index, DEVICE, BATCH_SIZE, scaler, OUT_CHANNELS
            )
        
        test_mae = np.mean(np.abs(test_preds_denorm - test_targets_denorm))
        test_rmse = np.sqrt(np.mean((test_preds_denorm - test_targets_denorm) ** 2))
        
        print(f"\nOverall Test MAE (1-step): {test_mae:.4f}")
        print(f"Overall Test RMSE (1-step): {test_rmse:.4f}")
        
        # --- CSV Saving ---
        # Save training history
        history_file = 'kan_gcn_training_history.csv'
        with open(history_file, 'w', newline='') as hf:
            writer = csv.writer(hf)
            writer.writerow(['epoch', 'train_loss', 'train_mae', 'train_rmse', 'val_mae', 'val_rmse'])
            for row in history:
                writer.writerow(row)
        
        # Save multi-horizon test results
        multihorizon_file = 'kan_gcn_test_multihorizon.csv'
        with open(multihorizon_file, 'w', newline='') as mf:
            writer = csv.writer(mf)
            writer.writerow(['horizon', 'arr_mae', 'arr_rmse', 'arr_r2', 'dep_mae', 'dep_rmse', 'dep_r2'])
            for result in results_table:
                writer.writerow([
                    result['horizon'],
                    result['arr_mae'], result['arr_rmse'], result['arr_r2'],
                    result['dep_mae'], result['dep_rmse'], result['dep_r2']
                ])
        
        # Save test predictions (first 10 sequences for demonstration)
        test_pred_file = 'kan_gcn_test_predictions.csv'
        with open(test_pred_file, 'w', newline='') as tf:
            writer = csv.writer(tf)
            writer.writerow(['sequence_idx', 'node_idx', 'arrival_true', 'arrival_pred', 
                           'departure_true', 'departure_pred'])
            for seq_idx in range(min(10, len(test_preds_denorm))):
                pred = test_preds_denorm[seq_idx]
                target = test_targets_denorm[seq_idx]
                for node_idx in range(num_nodes):
                    writer.writerow([
                        seq_idx, node_idx,
                        target[node_idx, 0], pred[node_idx, 0],  # arrival
                        target[node_idx, 1], pred[node_idx, 1]   # departure
                    ])
        
        # Save test metrics summary
        test_summary_file = 'kan_gcn_test_summary.csv'
        with open(test_summary_file, 'w', newline='') as tf:
            writer = csv.writer(tf)
            writer.writerow(['metric', 'value'])
            writer.writerow(['1step_test_mae', test_mae])
            writer.writerow(['1step_test_rmse', test_rmse])
            writer.writerow(['best_val_mae', best_val_mae])
        
        # Print summary table
        print("\n" + "="*80)
        print("SUMMARY TABLE - KAN-GCN Flight Delay Predictions")
        print("="*80)
        print(f"{'Horizon':<10} {'Delay Type':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
        print("-"*80)
        for result in results_table:
            print(f"{result['horizon']}-step    {'Arrival':<15} {result['arr_mae']:<12.4f} {result['arr_rmse']:<12.4f} {result['arr_r2']:<12.4f}")
            print(f"{'':10} {'Departure':<15} {result['dep_mae']:<12.4f} {result['dep_rmse']:<12.4f} {result['dep_r2']:<12.4f}")
        print("="*80)
        
        print(f'\n✓ Test Set Results:')
        print(f'  Overall MAE (1-step): {test_mae:.4f} minutes')
        print(f'  Overall RMSE (1-step): {test_rmse:.4f} minutes')
        print(f'\n✓ Files saved:')
        print(f'  - {history_file}')
        print(f'  - {multihorizon_file} (3, 6, 12-step predictions)')
        print(f'  - {test_pred_file}')
        print(f'  - {test_summary_file}')
        print(f'  - kan_gcn_best.pth')

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the dataset is correctly placed and paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

