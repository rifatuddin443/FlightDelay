# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 2025

@author: AI Assistant
Test script for dsafnet-full-fixed (1) architecture
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import sys
import os
import util
import csv

from baseline_methods import test_error, StandardScaler

# Add current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== Model Architecture from dsafnet-full-fixed (1).py =====

class OptimizedSpatialAttentionStream(nn.Module):
    """Vectorized spatial attention - DP compatible (no batch reshaping)"""
    def __init__(self, input_dim=6, hidden_dim=64, attention_class=None):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        # Use DP-compatible attention if provided
        if attention_class is not None:
            self.attention = attention_class(hidden_dim, 1, batch_first=True)
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
    
    def forward(self, x):
        # x: [batch, airports, time_steps, features]
        batch_size, airports, time_steps, features = x.shape

        # We run attention per time-step so attention's batch dimension remains the dataset batch
        time_outputs = []
        for t in range(time_steps):
            # x_t: [batch, airports, features]
            x_t = x[:, :, t, :]
            x_emb = self.embedding(x_t)  # [batch, airports, hidden]

            # attention expects (batch, seq, embed) when batch_first=True
            out_t, _ = self.attention(x_emb, x_emb, x_emb)  # [batch, airports, hidden]

            # add time axis
            time_outputs.append(out_t.unsqueeze(2))  # [batch, airports, 1, hidden]

        # concat along time dimension -> [batch, airports, time_steps, hidden]
        out = torch.cat(time_outputs, dim=2)
        return out

class OptimizedTemporalAttentionStream(nn.Module):
    """Vectorized temporal attention - DP compatible (no batch reshaping)"""
    def __init__(self, input_dim=6, hidden_dim=64, attention_class=None):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        if attention_class is not None:
            self.attention = attention_class(hidden_dim, 1, batch_first=True)
        else:
            self.attention = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
    
    def forward(self, x):
        # x: [batch, airports, time_steps, features]
        batch_size, airports, time_steps, features = x.shape

        # Run attention per airport so attention's batch dimension remains the dataset batch
        node_outputs = []
        for n in range(airports):
            # x_n: [batch, time_steps, features]
            x_n = x[:, n, :, :]
            x_emb = self.embedding(x_n)  # [batch, time_steps, hidden]

            out_n, _ = self.attention(x_emb, x_emb, x_emb)  # [batch, time_steps, hidden]

            # add node axis
            node_outputs.append(out_n.unsqueeze(1))  # [batch, 1, time_steps, hidden]

        # concat along node dimension -> [batch, airports, time_steps, hidden]
        out = torch.cat(node_outputs, dim=1)
        return out

class OptimizedContextualCrossAttention(nn.Module):
    def __init__(self, hidden_dim=64, attention_class=None):
        super().__init__()
        self._hidden_dim = hidden_dim
        
        # Create attention immediately with proper class
        if attention_class is not None:
            self.cross_attn = attention_class(hidden_dim, 1, batch_first=True)
        else:
            self.cross_attn = nn.MultiheadAttention(hidden_dim, 1, batch_first=True)
            
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
    
    def forward(self, spatial, temporal):
        # spatial, temporal: [batch, airports, hidden]
        fusion, _ = self.cross_attn(spatial, temporal, temporal)
        cat = torch.cat([spatial, fusion], dim=-1)
        gate_w = self.gate(cat)
        return gate_w * spatial + (1 - gate_w) * fusion

class SimpleGraphEncoder(nn.Module):
    def __init__(self, num_graphs=3, num_nodes=None):
        super().__init__()
        self.num_graphs = num_graphs
        self.num_nodes = num_nodes
        self.graph_weights = nn.Parameter(torch.ones(num_graphs) / num_graphs)
        
        # DON'T register buffers here - we'll pass adjacency matrices directly
        self._adj_matrices = None
    
    def set_adjacency_matrices(self, adj_list):
        """Store adjacency matrices as a regular Python attribute (not buffer)"""
        # Store as a list - not registered with PyTorch
        self._adj_matrices = adj_list
    
    def forward(self, features, adj_matrices=None):
        # Use passed adjacency matrices or stored ones
        if adj_matrices is not None:
            adj_list = adj_matrices
        elif self._adj_matrices is not None:
            adj_list = self._adj_matrices
        else:
            return features
        
        if not adj_list:
            return features
        
        # Stack and compute weighted combination
        adj_stack = torch.stack(adj_list)
        weights = torch.softmax(self.graph_weights[:len(adj_list)], dim=0)
        weights = weights.view(-1, 1, 1)
        combined_adj = (adj_stack * weights).sum(dim=0)
        
        return torch.matmul(combined_adj, features)

class OutputProjection(nn.Module):
    def __init__(self, hidden_dim=64, output_steps=12):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, output_steps * 2)
        self.output_steps = output_steps
    
    def forward(self, fused_features):
        out = self.projection(fused_features)
        batch_size, airports, _ = out.shape
        out = out.view(batch_size, airports, self.output_steps, 2)
        return out

class OptimizedDSAFNet(nn.Module):
    """Optimized DSAFNet with vectorized operations and DP support"""
    def __init__(self, input_dim=6, hidden_dim=64, output_steps=12, num_graphs=3, num_nodes=None, attention_class=None):
        super().__init__()
        self.spatial_stream = OptimizedSpatialAttentionStream(input_dim, hidden_dim, attention_class=attention_class)
        self.temporal_stream = OptimizedTemporalAttentionStream(input_dim, hidden_dim, attention_class=attention_class)
        self.cross_fusion = OptimizedContextualCrossAttention(hidden_dim, attention_class=attention_class)
        self.graph_encoder = SimpleGraphEncoder(num_graphs, num_nodes)
        self.output_proj = OutputProjection(hidden_dim, output_steps)
    
    def set_adjacency_matrices(self, adj_matrices):
        """Store adjacency matrices in the graph encoder"""
        self.graph_encoder.set_adjacency_matrices(adj_matrices)
    
    def forward(self, x, ti=None, supports=None, to=None, w=None):
        # NOTE: supports parameter is ignored - using stored adjacency matrices
        
        # Handle input shape transformation
        if x.dim() == 4:
            x = x.permute(0, 2, 3, 1)
        
        # Vectorized spatial and temporal processing
        spatial_features = self.spatial_stream(x)
        temporal_features = self.temporal_stream(x)
        
        # Average across time dimension
        spatial_avg = spatial_features.mean(dim=2)
        temporal_avg = temporal_features.mean(dim=2)
        
        # Cross attention fusion
        fused = self.cross_fusion(spatial_avg, temporal_avg)
        
        # Graph encoding (uses internally stored adjacency matrices)
        graph_enhanced = self.graph_encoder(fused)
        
        # Output projection
        out = self.output_proj(graph_enhanced)
        
        # Transform to match expected output format
        out = out.permute(0, 3, 1, 2)  # [batch, 2, airports, output_steps]
        
        return out


# ===== Test Script =====

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='device to run on')
parser.add_argument('--data', type=str, default='US', help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/test/val ratio', type=float)
parser.add_argument('--in_len', type=int, default=12, help='input time series length')
parser.add_argument('--out_len', type=int, default=12, help='output time series length')
parser.add_argument('--period', type=int, default=36, help='periodic for temporal embedding')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--model_path', type=str, default='./results/optimized_dsafnet_US.pth', help='path to saved model')
parser.add_argument('--out_csv', type=str, default='./results/dsaftest_full_fixed_results.csv', help='path to write CSV results')

args = parser.parse_args()

def main():
    print("="*60)
    print("🧪 TESTING DSAFNET-FULL-FIXED MODEL")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data and adjacency matrices using your util function
    print("\nLoading data...")
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    
    # Convert adjacency matrices to torch tensors
    supports = [torch.tensor(i, dtype=torch.float32).to(device) for i in adj]
    num_nodes = test_data.shape[0]
    
    print(f"✅ Data loaded successfully")
    print(f"📊 Test data shape: {test_data.shape}")
    print(f"📊 Number of nodes (airports): {num_nodes}")
    print(f"📊 Number of adjacency matrices: {len(supports)}")
    
    # Determine the correct input dimension based on the data
    actual_input_dim = test_data.shape[2]  # Number of features in the data
    print(f"📊 Input features: {actual_input_dim}")
    
    # Initialize scaler based on training data
    print("\nInitializing scaler...")
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), 
                           training_data[~np.isnan(training_data)].std())
    print("✅ Scaler initialized")
    
    # Prepare test label sequences
    print("\nPreparing test sequences...")
    test_index = list(range(test_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(test_index)):
        label.append(np.expand_dims(test_data[:, test_index[i] + args.in_len:test_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)
    print(f"✅ Prepared {len(test_index)} test sequences")
    print(f"📊 Label shape: {label.shape}")
    
    # Initialize model
    print("\n" + "="*60)
    print("🤖 INITIALIZING MODEL")
    print("="*60)
    model = OptimizedDSAFNet(
        input_dim=actual_input_dim, 
        hidden_dim=args.hidden_dim, 
        output_steps=args.out_len, 
        num_graphs=len(supports),
        num_nodes=num_nodes,
        attention_class=None  # Use standard attention for testing
    ).to(device)
    
    # Set adjacency matrices in the model
    model.set_adjacency_matrices(supports)
    print("✅ Adjacency matrices stored in model")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model parameters: {num_params:,}")
    
    # Try to load the model state dict
    print("\n" + "="*60)
    print("💾 LOADING MODEL WEIGHTS")
    print("="*60)
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        
        # Handle Opacus wrapped model (removes _module. prefix and converts attention keys)
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove _module. prefix if present (from Opacus PrivacyEngine)
            new_key = key.replace('_module.', '')
            
            # Convert Opacus attention layer names to standard PyTorch
            # Opacus uses qlinear, klinear, vlinear; PyTorch uses in_proj_weight/bias
            if 'qlinear.weight' in new_key:
                # For Opacus DPMultiheadAttention, we need to keep the separate linear layers
                new_state_dict[new_key] = value
            elif 'klinear.weight' in new_key:
                new_state_dict[new_key] = value
            elif 'vlinear.weight' in new_key:
                new_state_dict[new_key] = value
            elif 'qlinear.bias' in new_key:
                new_state_dict[new_key] = value
            elif 'klinear.bias' in new_key:
                new_state_dict[new_key] = value
            elif 'vlinear.bias' in new_key:
                new_state_dict[new_key] = value
            else:
                new_state_dict[new_key] = value
        
        # Check if we have Opacus-style attention layers
        has_opacus_attention = any('qlinear' in k for k in new_state_dict.keys())
        
        if has_opacus_attention:
            print("⚠️  Detected Opacus-trained model with DPMultiheadAttention")
            print("⚠️  Standard MultiheadAttention uses different layer structure")
            print("⚠️  Converting attention weights...")
            
            # Convert qlinear, klinear, vlinear to in_proj_weight/bias for standard MHA
            converted_state_dict = {}
            for key, value in new_state_dict.items():
                if 'qlinear.weight' in key or 'klinear.weight' in key or 'vlinear.weight' in key:
                    # Need to concatenate q, k, v weights into in_proj_weight
                    base_key = key.replace('qlinear.weight', '').replace('klinear.weight', '').replace('vlinear.weight', '')
                    in_proj_key = base_key + 'in_proj_weight'
                    
                    if in_proj_key not in converted_state_dict:
                        # Find q, k, v weights
                        q_key = base_key + 'qlinear.weight'
                        k_key = base_key + 'klinear.weight'
                        v_key = base_key + 'vlinear.weight'
                        
                        if q_key in new_state_dict and k_key in new_state_dict and v_key in new_state_dict:
                            # Concatenate [q, k, v] weights
                            q_weight = new_state_dict[q_key]
                            k_weight = new_state_dict[k_key]
                            v_weight = new_state_dict[v_key]
                            converted_state_dict[in_proj_key] = torch.cat([q_weight, k_weight, v_weight], dim=0)
                
                elif 'qlinear.bias' in key or 'klinear.bias' in key or 'vlinear.bias' in key:
                    # Need to concatenate q, k, v biases into in_proj_bias
                    base_key = key.replace('qlinear.bias', '').replace('klinear.bias', '').replace('vlinear.bias', '')
                    in_proj_bias_key = base_key + 'in_proj_bias'
                    
                    if in_proj_bias_key not in converted_state_dict:
                        # Find q, k, v biases
                        q_bias_key = base_key + 'qlinear.bias'
                        k_bias_key = base_key + 'klinear.bias'
                        v_bias_key = base_key + 'vlinear.bias'
                        
                        if q_bias_key in new_state_dict and k_bias_key in new_state_dict and v_bias_key in new_state_dict:
                            # Concatenate [q, k, v] biases
                            q_bias = new_state_dict[q_bias_key]
                            k_bias = new_state_dict[k_bias_key]
                            v_bias = new_state_dict[v_bias_key]
                            converted_state_dict[in_proj_bias_key] = torch.cat([q_bias, k_bias, v_bias], dim=0)
                else:
                    # Keep other weights as-is
                    converted_state_dict[key] = value
            
            new_state_dict = converted_state_dict
        
        # Load the processed state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️  Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 5:
                for k in missing_keys:
                    print(f"    - {k}")
        
        if unexpected_keys:
            print(f"⚠️  Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                for k in unexpected_keys:
                    print(f"    - {k}")
        
        if not missing_keys and not unexpected_keys:
            print(f"✅ Successfully loaded model weights from: {args.model_path}")
        else:
            print(f"⚠️  Partially loaded model weights from: {args.model_path}")
            print(f"⚠️  Some layers may use random initialization")
            
    except FileNotFoundError:
        print(f"⚠️  Model file not found: {args.model_path}")
        print("⚠️  Continuing with randomly initialized model...")
    except Exception as e:
        print(f"⚠️  Error loading model weights: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Continuing with randomly initialized model...")
    
    # Set model to evaluation mode
    model.eval()
    
    # Run inference
    print("\n" + "="*60)
    print("🔮 RUNNING INFERENCE")
    print("="*60)
    
    outputs = []
    with torch.no_grad():
        for i in range(len(test_index)):
            testx = np.expand_dims(test_data[:, test_index[i]:test_index[i] + args.in_len, :], axis=0)
            testx = scaler.transform(testx)
            testw = np.expand_dims(test_w[:, test_index[i]:test_index[i] + args.in_len], axis=0)
            testw = torch.LongTensor(testw).to(device)
            testx[np.isnan(testx)] = 0
            
            # Print progress every 1000 samples
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(test_index)}...")
            
            # Create time embeddings
            testti = (np.arange(int(training_data.shape[1] + val_data.shape[1]) + test_index[i],
                                int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len)
                      % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
            testto = (np.arange(int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len,
                                int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len + args.out_len)
                      % args.period) * np.ones([1, args.out_len]) / (args.period - 1)
            
            # Convert to torch tensors with correct shape
            # testx shape: [batch=1, nodes, time, features] -> [batch, features, nodes, time]
            testx = torch.Tensor(testx).to(device).permute(0, 3, 1, 2)
            testti = torch.Tensor(testti).to(device)
            testto = torch.Tensor(testto).to(device)
            
            # Forward pass - model uses stored adjacency matrices
            output = model(testx, testti, None, testto, testw)
            
            # Convert output back to expected format [batch, nodes, time, features]
            output = output.permute(0, 2, 3, 1)
            output = output.cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
    
    yhat = np.concatenate(outputs)
    
    print(f"\n✅ Inference complete!")
    print(f"📊 Predictions shape: {yhat.shape}")
    print(f"📊 Labels shape: {label.shape}")
    
    # Calculate and print error metrics
    print("\n" + "="*60)
    print("📈 TEST RESULTS")
    print("="*60)
    
    # Prepare results collection for CSV
    results_rows = []
    # Calculate metrics for arrival and departure delay at key time steps
    for step in [2, 5, 11]:
        print(f"\n--- {step+1} Step Ahead Predictions ---")
        
        # Arrival delay (feature 0)
        MAE_a, RMSE_a, R2_a = test_error(yhat[:, :, step, 0], label[:, :, step, 0])
        print(f"Arrival Delay  | MAE: {MAE_a:7.4f} min | RMSE: {RMSE_a:7.4f} min | R²: {R2_a:7.4f}")
        
        # Departure delay (feature 1)
        MAE_d, RMSE_d, R2_d = test_error(yhat[:, :, step, 1], label[:, :, step, 1])
        print(f"Departure Delay| MAE: {MAE_d:7.4f} min | RMSE: {RMSE_d:7.4f} min | R²: {R2_d:7.4f}")

        # Append row for this step
        results_rows.append([
            step + 1,
            float(MAE_a), float(RMSE_a), float(R2_a),
            float(MAE_d), float(RMSE_d), float(R2_d)
        ])
    
    # Calculate average metrics across all time steps
    print("\n" + "="*60)
    print("📊 AVERAGE METRICS (ALL TIME STEPS)")
    print("="*60)
    
    amae_arr, ar2_arr, armse_arr = [], [], []
    amae_dep, ar2_dep, armse_dep = [], [], []
    
    for i in range(args.out_len):
        # Arrival delay
        mae, rmse, r2 = test_error(yhat[:, :, i, 0], label[:, :, i, 0])
        amae_arr.append(mae)
        armse_arr.append(rmse)
        ar2_arr.append(r2)
        
        # Departure delay
        mae, rmse, r2 = test_error(yhat[:, :, i, 1], label[:, :, i, 1])
        amae_dep.append(mae)
        armse_dep.append(rmse)
        ar2_dep.append(r2)
    
    print(f"\nArrival Delay  | MAE: {np.mean(amae_arr):7.4f} min | RMSE: {np.mean(armse_arr):7.4f} min | R²: {np.mean(ar2_arr):7.4f}")
    print(f"Departure Delay| MAE: {np.mean(amae_dep):7.4f} min | RMSE: {np.mean(armse_dep):7.4f} min | R²: {np.mean(ar2_dep):7.4f}")
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)

    # Append average / overall metrics row
    results_rows.append([
        'avg',
        float(np.mean(amae_arr)), float(np.mean(armse_arr)), float(np.mean(ar2_arr)),
        float(np.mean(amae_dep)), float(np.mean(armse_dep)), float(np.mean(ar2_dep))
    ])

    # Write results to CSV
    out_csv = args.out_csv
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    header = ['step', 'arrival_mae', 'arrival_rmse', 'arrival_r2', 'departure_mae', 'departure_rmse', 'departure_r2']
    try:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in results_rows:
                writer.writerow(row)
        print(f"\n💾 CSV results written to: {out_csv}")
    except Exception as e:
        print(f"⚠️  Failed to write CSV results to {out_csv}: {e}")

if __name__ == "__main__":
    main()
