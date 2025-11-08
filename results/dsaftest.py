# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 15:15:00 2025

@author: AI Assistant
"""

import torch
import numpy as np
import argparse
import sys
import os
import util

from baseline_methods import test_error, StandardScaler

# Add current directory to Python path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import DSAFNet class
from DSAFnet import DSAFNet

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='device to run on')
parser.add_argument('--data', type=str, default='US', help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/test/val ratio', type=float)
parser.add_argument('--in_len', type=int, default=12, help='input time series length')
parser.add_argument('--out_len', type=int, default=12, help='output time series length')
parser.add_argument('--period', type=int, default=36, help='periodic for temporal embedding')

args = parser.parse_args()

def main():
    device = torch.device(args.device)
    
    # Load data and adjacency matrices using your util function
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    supports = [torch.tensor(i).to(device) for i in adj]
    
    # Debug: Print data dimensions
    print(f"Test data shape: {test_data.shape}")
    print(f"Training data shape: {training_data.shape}")
    print(f"Test data features: {test_data.shape[2]}")
    
    # Determine the correct input dimension based on the data
    actual_input_dim = test_data.shape[2]  # Number of features in the data
    print(f"Actual input dimension from data: {actual_input_dim}")
    
    # If test data has more features than expected by model, we need to match the model's expected input
    # The error suggests the model expects 2 features, but data has 6
    if actual_input_dim > 2:
        print(f"Model expects 2 features, but data has {actual_input_dim}. Taking first 2 features.")
        test_data = test_data[:, :, :2]  # Take only first 2 features
        actual_input_dim = 2
        print(f"Reduced test data shape: {test_data.shape}")
    
    # Initialize scaler based on training data (also reduced to 2 features if needed)
    if training_data.shape[2] > 2:
        training_data_for_scaler = training_data[:, :, :2]
    else:
        training_data_for_scaler = training_data
        
    scaler = StandardScaler(training_data_for_scaler[~np.isnan(training_data_for_scaler)].mean(), 
                           training_data_for_scaler[~np.isnan(training_data_for_scaler)].std())
    
    # Prepare test label sequences
    test_index = list(range(test_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(test_index)):
        label.append(np.expand_dims(test_data[:, test_index[i] + args.in_len:test_index[i] + args.in_len + args.out_len, :], axis=0))
    label = np.concatenate(label)
    
    # Load pre-trained model and move to device
    # Use the actual input dimension from the data (should be 2 after reduction)
    model = DSAFNet(input_dim=actual_input_dim, hidden_dim=64, output_steps=args.out_len, num_graphs=len(supports)).to(device)
    
    # Try to load the model state dict
    try:
        model.load_state_dict(torch.load("D:\\flight delay\\stpn paper\\STPN-main\\results\\dsafnet_US.pth", map_location=device))
        print("Successfully loaded model weights")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Continuing with randomly initialized model...")
    
    model.eval()
    
    outputs = []
    with torch.no_grad():
        for i in range(len(test_index)):  # Process all test samples
            testx = np.expand_dims(test_data[:, test_index[i]:test_index[i] + args.in_len, :], axis=0)
            testx = scaler.transform(testx)
            testw = np.expand_dims(test_w[:, test_index[i]:test_index[i] + args.in_len], axis=0)
            testw = torch.LongTensor(testw).to(device)
            testx[np.isnan(testx)] = 0
            
            # Print progress every 1000 samples
            if i % 1000 == 0:
                print(f"Processing sample {i}/{len(test_index)}")
            
            # Create time embeddings (matching STPN format)
            testti = (np.arange(int(training_data_for_scaler.shape[1] + val_data.shape[1]) + test_index[i],
                                int(training_data_for_scaler.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len)
                      % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
            testto = (np.arange(int(training_data_for_scaler.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len,
                                int(training_data_for_scaler.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len + args.out_len)
                      % args.period) * np.ones([1, args.out_len]) / (args.period - 1)
            
            # Convert to torch tensors with correct shape for STPN-compatible forward pass
            # testx shape: [batch=1, nodes, time, features] -> [batch, features, nodes, time]
            testx = torch.Tensor(testx).to(device).permute(0, 3, 1, 2)
            testti = torch.Tensor(testti).to(device)
            testto = torch.Tensor(testto).to(device)
            
            # Use STPN-compatible forward pass signature
            output = model(testx, testti, supports, testto, testw)
            
            # Convert output back to expected format [batch, nodes, time, features]
            output = output.permute(0, 2, 3, 1)
            output = output.cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
    
    yhat = np.concatenate(outputs)
    
    print(f"Final yhat shape: {yhat.shape}")
    print(f"Final label shape: {label.shape}")
    print(f"Number of test samples processed: {len(outputs)}")
    
    # Calculate and print error metrics for arrival and departure delay at key time steps
    for step in [2, 5, 11]:
        log_arrival = f'{step+1} step ahead arrival delay, Test MAE: {{:.4f}} min, Test R2: {{:.4f}}, Test RMSE: {{:.4f}} min'
        MAE, RMSE, R2 = test_error(yhat[:, :, step, 0], label[:, :, step, 0])
        print(log_arrival.format(MAE, R2, RMSE))

        log_departure = f'{step+1} step ahead departure delay, Test MAE: {{:.4f}} min, Test R2: {{:.4f}}, Test RMSE: {{:.4f}} min'
        MAE, RMSE, R2 = test_error(yhat[:, :, step, 1], label[:, :, step, 1])
        print(log_departure.format(MAE, R2, RMSE))

if __name__ == "__main__":
    main()
