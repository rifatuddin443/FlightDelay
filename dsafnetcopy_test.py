# -*- coding: utf-8 -*-
"""
Test script for DSAFNet model (DSAFnetCopy.py)
Loads trained model and evaluates on test set

Created on: November 9, 2025
"""

import torch
import numpy as np
import argparse
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import model and utilities
from DSAFnetCopy import DSAFNet, load_flight_delay_data, create_datasets, test_error

def main():
    parser = argparse.ArgumentParser(description='Test DSAFNet model on flight delay data')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to run on')
    parser.add_argument('--dataset', type=str, default='US', choices=['US', 'China'], help='dataset type')
    parser.add_argument('--data_dir', type=str, default='.', help='directory containing udata and cdata folders')
    parser.add_argument('--model_path', type=str, default='./results/dsafnet_flight_delay_regularized.pth', 
                        help='path to trained model weights')
    parser.add_argument('--in_len', type=int, default=12, help='input time series length')
    parser.add_argument('--out_len', type=int, default=12, help='output time series length')
    parser.add_argument('--input_dim', type=int, default=None, help='input dimension (auto-detected if None)')
    parser.add_argument('--hidden_dim', type=int, default=32, help='hidden dimension')
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for testing')
    args = parser.parse_args()

    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Using device: {device}")
    print("="*80)
    
    # Load dataset
    print(f"🌍 Loading {args.dataset} flight delay dataset...")
    try:
        delay_data, weather_data, adj_matrices, scaler, num_airports, time_steps = load_flight_delay_data(
            dataset=args.dataset,
            data_dir=args.data_dir,
            sequence_length=args.in_len,
            prediction_length=args.out_len,
            normalize=True
        )
        
        print(f"✅ Successfully loaded dataset")
        print(f"📊 Time steps: {time_steps}")
        print(f"📊 Airports: {num_airports}")
        print(f"📊 Delay features: {delay_data.shape[2]}")
        
        # Create datasets (we only need test dataset)
        _, _, test_dataset, dataset_info = create_datasets(
            data=delay_data,
            weather_data=weather_data,
            adj_matrices=adj_matrices,
            in_len=args.in_len,
            out_len=args.out_len,
            train_ratio=0.7,
            val_ratio=0.15,
            scaler=scaler,
            add_time_features=True
        )
        
        # Auto-detect input dimension if not specified
        if args.input_dim is None:
            args.input_dim = dataset_info['input_dim']
            print(f"🔍 Auto-detected input dimension: {args.input_dim}")
        
        # Get adjacency matrices
        adj_matrices = test_dataset.get_adj_matrices()
        adj_matrices = [adj.to(device) for adj in adj_matrices]
        
        print(f"📊 Test Dataset Information:")
        print(f"   Test sequences: {dataset_info['test_size']}")
        print(f"   Input features: {dataset_info['input_dim']}")
        print(f"   Airports: {dataset_info['num_airports']}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize model
    print("\n" + "="*80)
    print("🤖 Initializing model...")
    model = DSAFNet(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_steps=args.out_len,
        num_graphs=len(adj_matrices),
        dropout_rate=args.dropout_rate
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,}")
    
    # Load trained model weights
    print(f"\n📥 Loading model weights from: {args.model_path}")
    try:
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("✅ Successfully loaded model weights")
    except FileNotFoundError:
        print(f"❌ Model file not found: {args.model_path}")
        print("⚠️  Continuing with randomly initialized model (results will be meaningless)")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        print("⚠️  Continuing with randomly initialized model (results will be meaningless)")
    
    # Set model to evaluation mode
    model.eval()
    
    # Perform testing
    print("\n" + "="*80)
    print("🧪 TESTING MODEL ON TEST SET")
    print("="*80)
    
    outputs = []
    labels = []
    
    print(f"Processing {len(test_dataset)} test samples...")
    
    with torch.no_grad():
        for i in range(len(test_dataset)):
            # Get test sample
            testx, testy = test_dataset[i]
            testx = testx.unsqueeze(0).to(device)  # Add batch dimension
            
            # Forward pass
            output = model(testx, adj_matrices)
            output = output.squeeze(0).detach().cpu().numpy()  # Remove batch dimension
            
            # Store outputs and labels
            outputs.append(np.expand_dims(output, axis=0))
            labels.append(np.expand_dims(testy.numpy(), axis=0))
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"   Processed {i + 1}/{len(test_dataset)} samples...")
    
    # Concatenate all predictions and labels
    yhat = np.concatenate(outputs)  # [samples, airports, out_len, 2]
    label = np.concatenate(labels)  # [samples, airports, out_len, 2]
    
    print(f"\n✅ Testing complete")
    print(f"   Predictions shape: {yhat.shape}")
    print(f"   Labels shape: {label.shape}")
    
    # DENORMALIZE predictions and labels to get real minutes
    test_scaler = test_dataset.scaler
    if test_scaler is not None:
        print(f"\n🔄 Denormalizing test data using scaler...")
        print(f"   Before denorm - Test pred range: [{yhat.min():.4f}, {yhat.max():.4f}]")
        print(f"   Before denorm - Test label range: [{label.min():.4f}, {label.max():.4f}]")
        
        # Reshape for denormalization: [samples*airports*out_len, 2]
        original_shape = yhat.shape
        yhat_reshaped = yhat.reshape(-1, 2)
        label_reshaped = label.reshape(-1, 2)
        
        # Denormalize using fitted scaler
        yhat_denorm = test_scaler.inverse_transform(yhat_reshaped)
        label_denorm = test_scaler.inverse_transform(label_reshaped)
        
        # Reshape back to original
        yhat = yhat_denorm.reshape(original_shape)
        label = label_denorm.reshape(original_shape)
        
        print(f"   After denorm - Test pred range: [{yhat.min():.4f}, {yhat.max():.4f}]")
        print(f"   After denorm - Test label range: [{label.min():.4f}, {label.max():.4f}]")
        print(f"✅ Denormalization complete - metrics now in REAL MINUTES")
    else:
        print(f"⚠️  No scaler found - using normalized values")
    
    # Calculate and display metrics at key time steps (NOW IN REAL MINUTES)
    print("\n" + "="*80)
    print("📊 DETAILED RESULTS BY TIME STEP (Real Minutes)")
    print("="*80)
    
    delay_types = ['arrival', 'departure']
    delay_indices = [0, 1]
    
    for step in [3, 6, 12]:  # 3, 6, 12 steps ahead
        print(f"\n{step}-step ahead predictions:")
        for delay_idx, delay_type in zip(delay_indices, delay_types):
            # Extract predictions and labels for specific step and delay type
            pred = yhat[:, :, step-1, delay_idx]  # step-1 because 0-indexed
            true = label[:, :, step-1, delay_idx]
            
            # Calculate metrics
            mae, rmse, r2 = test_error(pred, true)
            
            # Print results
            log = f'   {step:2d} step ahead {delay_type:9s} delay - Test MAE: {mae:7.4f} min, Test R2: {r2:6.4f}, Test RMSE: {rmse:7.4f} min'
            print(log)
    
    # Overall performance across all time steps
    print(f"\n{'='*80}")
    print("📊 OVERALL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    for delay_idx, delay_type in zip(delay_indices, delay_types):
        # Calculate average across all time steps
        all_pred = yhat[:, :, :, delay_idx].flatten()
        all_true = label[:, :, :, delay_idx].flatten()
        
        mae, rmse, r2 = test_error(all_pred, all_true)
        
        print(f"Overall {delay_type:9s} delay - MAE: {mae:7.4f} min, R2: {r2:6.4f}, RMSE: {rmse:7.4f} min")
    
    # Combined overall performance
    all_pred_combined = yhat.flatten()
    all_true_combined = label.flatten()
    mae_combined, rmse_combined, r2_combined = test_error(all_pred_combined, all_true_combined)
    
    print(f"Overall combined      - MAE: {mae_combined:7.4f} min, R2: {r2_combined:6.4f}, RMSE: {rmse_combined:7.4f} min")
    print(f"{'='*80}")
    
    # Save results summary
    print(f"\n💾 Saving test results...")
    results_summary = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'test_samples': len(test_dataset),
        'overall_mae': mae_combined,
        'overall_rmse': rmse_combined,
        'overall_r2': r2_combined,
        'arrival_mae': None,
        'departure_mae': None
    }
    
    # Calculate per-delay-type metrics
    for delay_idx, delay_type in zip(delay_indices, delay_types):
        all_pred = yhat[:, :, :, delay_idx].flatten()
        all_true = label[:, :, :, delay_idx].flatten()
        mae, _, _ = test_error(all_pred, all_true)
        results_summary[f'{delay_type}_mae'] = mae
    
    # Save to file
    results_file = os.path.join(os.path.dirname(args.model_path), 'test_results_summary.txt')
    with open(results_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("DSAFNet Test Results Summary\n")
        f.write("="*80 + "\n\n")
        for key, value in results_summary.items():
            f.write(f"{key}: {value}\n")
        f.write("\n" + "="*80 + "\n")
    
    print(f"✅ Results saved to: {results_file}")
    
    print("\n" + "🎉" + "="*78 + "🎉")
    print("🎊 TESTING COMPLETE! 🎊")
    print("🎉" + "="*78 + "🎉")

if __name__ == "__main__":
    main()
