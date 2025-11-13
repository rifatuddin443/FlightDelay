# -*- coding: utf-8 -*-
"""
Test script for Simple GCN DSAFNet model
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

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the model from the training file
from dsafnetgcn import OptimizedDSAFNetWithSimpleGCN

# ===== Test Script =====

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='device to run on')
parser.add_argument('--data', type=str, default='US', help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], type=float)
parser.add_argument('--in_len', type=int, default=12, help='input time series length')
parser.add_argument('--out_len', type=int, default=12, help='output time series length')
parser.add_argument('--period', type=int, default=36, help='periodic for temporal embedding')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--model_path', type=str, default='./results/simple_gcn_US.pth', 
                    help='path to saved model')
parser.add_argument('--out_csv', type=str, default='./results/simple_gcn_test_results.csv', 
                    help='path to write CSV results')

args = parser.parse_args()


def main():
    print("="*60)
    print("🧪 TESTING SIMPLE GCN DSAFNET MODEL")
    print("="*60)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\nLoading data...")
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    
    # Convert adjacency matrices to torch tensors
    supports = [torch.tensor(i, dtype=torch.float32).to(device) for i in adj]
    num_nodes = test_data.shape[0]
    
    print(f"✅ Data loaded successfully")
    print(f"📊 Test data shape: {test_data.shape}")
    print(f"📊 Number of nodes (airports): {num_nodes}")
    print(f"📊 Number of adjacency matrices: {len(supports)} (OD, DO, distance)")
    
    # Determine input dimension
    actual_input_dim = test_data.shape[2]
    print(f"📊 Input features: {actual_input_dim}")
    
    # Initialize scaler
    print("\nInitializing scaler...")
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                           training_data[~np.isnan(training_data)].std())
    print("✅ Scaler initialized")
    
    # Prepare test sequences
    print("\nPreparing test sequences...")
    test_index = list(range(test_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(test_index)):
        label.append(np.expand_dims(
            test_data[:, test_index[i] + args.in_len:test_index[i] + args.in_len + args.out_len, :], 
            axis=0))
    label = np.concatenate(label)
    print(f"✅ Prepared {len(test_index)} test sequences")
    print(f"📊 Label shape: {label.shape}")
    
    # Initialize model
    print("\n" + "="*60)
    print("🤖 INITIALIZING SIMPLE GCN MODEL")
    print("="*60)
    
    model = OptimizedDSAFNetWithSimpleGCN(
        input_dim=actual_input_dim,
        hidden_dim=args.hidden_dim,
        output_steps=args.out_len,
        num_nodes=num_nodes,
        attention_class=None  # Standard attention for testing
    ).to(device)
    
    # Set adjacency matrices
    model.set_adjacency_matrices(supports)
    print("✅ 3 adjacency matrices (OD, DO, distance) set in GCN encoder")
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"📦 Model parameters: {num_params:,}")
    
    # Load model weights
    print("\n" + "="*60)
    print("💾 LOADING MODEL WEIGHTS")
    print("="*60)
    
    try:
        state_dict = torch.load(args.model_path, map_location=device)
        
        # Handle Opacus wrapped models
        new_state_dict = {}
        for key, value in state_dict.items():
            # Remove _module. prefix if present
            new_key = key.replace('_module.', '')
            new_state_dict[new_key] = value
        
        # Check if Opacus-style attention layers exist
        has_opacus_attention = any('qlinear' in k or 'klinear' in k or 'vlinear' in k 
                                  for k in new_state_dict.keys())
        
        if has_opacus_attention:
            print("⚠️ Detected Opacus-trained model with DPMultiheadAttention")
            print("⚠️ Converting attention weights to standard MultiheadAttention format...")
            
            # Convert qlinear, klinear, vlinear to in_proj_weight/bias
            converted_state_dict = {}
            
            # Group keys by base attention module
            attention_modules = set()
            for key in new_state_dict.keys():
                if 'attention' in key or 'cross_attn' in key:
                    # Extract base module name (everything before .qlinear/.klinear/.vlinear)
                    if 'qlinear' in key or 'klinear' in key or 'vlinear' in key:
                        base = key.rsplit('.', 2)[0]
                        attention_modules.add(base)
            
            # Convert each attention module
            for base_module in attention_modules:
                # Get q, k, v weights
                q_weight_key = f"{base_module}.qlinear.weight"
                k_weight_key = f"{base_module}.klinear.weight"
                v_weight_key = f"{base_module}.vlinear.weight"
                
                if all(k in new_state_dict for k in [q_weight_key, k_weight_key, v_weight_key]):
                    q_w = new_state_dict[q_weight_key]
                    k_w = new_state_dict[k_weight_key]
                    v_w = new_state_dict[v_weight_key]
                    converted_state_dict[f"{base_module}.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
                
                # Get q, k, v biases
                q_bias_key = f"{base_module}.qlinear.bias"
                k_bias_key = f"{base_module}.klinear.bias"
                v_bias_key = f"{base_module}.vlinear.bias"
                
                if all(k in new_state_dict for k in [q_bias_key, k_bias_key, v_bias_key]):
                    q_b = new_state_dict[q_bias_key]
                    k_b = new_state_dict[k_bias_key]
                    v_b = new_state_dict[v_bias_key]
                    converted_state_dict[f"{base_module}.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
            
            # Copy other non-attention weights
            for key, value in new_state_dict.items():
                if not any(attn_key in key for attn_key in ['qlinear', 'klinear', 'vlinear']):
                    converted_state_dict[key] = value
            
            new_state_dict = converted_state_dict
        
        # Load state dict
        missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            print(f"⚠️ Missing keys: {len(missing_keys)}")
            if len(missing_keys) <= 5:
                for k in missing_keys:
                    print(f"   - {k}")
        
        if unexpected_keys:
            print(f"⚠️ Unexpected keys: {len(unexpected_keys)}")
            if len(unexpected_keys) <= 5:
                for k in unexpected_keys:
                    print(f"   - {k}")
        
        if not missing_keys and not unexpected_keys:
            print(f"✅ Successfully loaded model weights from: {args.model_path}")
        else:
            print(f"⚠️ Partially loaded model weights from: {args.model_path}")
    
    except FileNotFoundError:
        print(f"⚠️ Model file not found: {args.model_path}")
        print("⚠️ Using randomly initialized model...")
    except Exception as e:
        print(f"⚠️ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️ Using randomly initialized model...")
    
    # Set to evaluation mode
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
            
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(test_index)}...")
            
            # Time embeddings
            testti = (np.arange(int(training_data.shape[1] + val_data.shape[1]) + test_index[i],
                               int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len)
                     % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
            testto = (np.arange(int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len,
                               int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len + args.out_len)
                     % args.period) * np.ones([1, args.out_len]) / (args.period - 1)
            
            # Convert to tensors
            testx = torch.Tensor(testx).to(device).permute(0, 3, 1, 2)
            testti = torch.Tensor(testti).to(device)
            testto = torch.Tensor(testto).to(device)
            
            # Forward pass
            output = model(testx, testti, None, testto, testw)
            
            # Convert output
            output = output.permute(0, 2, 3, 1)
            output = output.cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
    
    yhat = np.concatenate(outputs)
    print(f"\n✅ Inference complete!")
    print(f"📊 Predictions shape: {yhat.shape}")
    print(f"📊 Labels shape: {label.shape}")
    
    # Calculate metrics
    print("\n" + "="*60)
    print("📈 TEST RESULTS")
    print("="*60)
    
    results_rows = []
    
    # Key time steps
    for step in [2, 5, 11]:
        print(f"\n--- {step+1} Step Ahead Predictions ---")
        
        # Arrival delay
        MAE_a, RMSE_a, R2_a = test_error(yhat[:, :, step, 0], label[:, :, step, 0])
        print(f"Arrival Delay  | MAE: {MAE_a:7.4f} min | RMSE: {RMSE_a:7.4f} min | R²: {R2_a:7.4f}")
        
        # Departure delay
        MAE_d, RMSE_d, R2_d = test_error(yhat[:, :, step, 1], label[:, :, step, 1])
        print(f"Departure Delay| MAE: {MAE_d:7.4f} min | RMSE: {RMSE_d:7.4f} min | R²: {R2_d:7.4f}")
        
        results_rows.append([
            step + 1,
            float(MAE_a), float(RMSE_a), float(R2_a),
            float(MAE_d), float(RMSE_d), float(R2_d)
        ])
    
    # Average across all steps
    print("\n" + "="*60)
    print("📊 AVERAGE METRICS (ALL TIME STEPS)")
    print("="*60)
    
    amae_arr, ar2_arr, armse_arr = [], [], []
    amae_dep, ar2_dep, armse_dep = [], [], []
    
    for i in range(args.out_len):
        mae, rmse, r2 = test_error(yhat[:, :, i, 0], label[:, :, i, 0])
        amae_arr.append(mae)
        armse_arr.append(rmse)
        ar2_arr.append(r2)
        
        mae, rmse, r2 = test_error(yhat[:, :, i, 1], label[:, :, i, 1])
        amae_dep.append(mae)
        armse_dep.append(rmse)
        ar2_dep.append(r2)
    
    print(f"\nArrival Delay  | MAE: {np.mean(amae_arr):7.4f} min | RMSE: {np.mean(armse_arr):7.4f} min | R²: {np.mean(ar2_arr):7.4f}")
    print(f"Departure Delay| MAE: {np.mean(amae_dep):7.4f} min | RMSE: {np.mean(armse_dep):7.4f} min | R²: {np.mean(ar2_dep):7.4f}")
    
    print("\n" + "="*60)
    print("✅ TESTING COMPLETE!")
    print("="*60)
    
    results_rows.append([
        'avg',
        float(np.mean(amae_arr)), float(np.mean(armse_arr)), float(np.mean(ar2_arr)),
        float(np.mean(amae_dep)), float(np.mean(armse_dep)), float(np.mean(ar2_dep))
    ])
    
    # Write CSV
    out_csv = args.out_csv
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    header = ['step', 'arrival_mae', 'arrival_rmse', 'arrival_r2', 
              'departure_mae', 'departure_rmse', 'departure_r2']
    
    try:
        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in results_rows:
                writer.writerow(row)
        print(f"\n💾 CSV results written to: {out_csv}")
    except Exception as e:
        print(f"⚠️ Failed to write CSV: {e}")


if __name__ == "__main__":
    main()
