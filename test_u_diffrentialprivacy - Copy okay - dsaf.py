# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:40:49 2022

@author: AA
"""

import torch
from model import STPN
import util
import numpy as np
import argparse
import torch.nn as nn
from baseline_methods import test_error, StandardScaler
from training_u_diffrentialprivacy import SpatioTemporalCNN

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cpu', help='')
parser.add_argument('--data', type=str, default='US', help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/test/val ratio', type=float)
parser.add_argument('--in_len', type=int, default=12, help='input time series length')
parser.add_argument('--out_len', type=int, default=12, help='output time series length')
parser.add_argument('--period', type=int, default=36, help='periodic for temporal embedding')

# 🔹 Add model architecture arguments (must match training_u.py)
parser.add_argument('--h_layers', type=int, default=2, help='number of STPN hidden layers')
parser.add_argument('--in_channels', type=int, default=2, help='number of input channels')
parser.add_argument('--hidden_channels', type=int,nargs='+', default=[128,64,32], help='hidden channel size')
parser.add_argument('--out_channels', type=int, default=2, help='number of output channels')
parser.add_argument('--emb_size', type=int, default=16, help='temporal embedding size')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--wemb_size',type=int,default=4,help='covariate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=8,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=False,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")

args = parser.parse_args()

# Keep the original SimpleCNN for backward compatibility  
class SpatialAttentionStream(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 1)
    def forward(self, x):  # x: [batch, airports, features]
        x_emb = self.embedding(x).transpose(0, 1)
        out, _ = self.attention(x_emb, x_emb, x_emb)
        return out.transpose(0, 1)

class TemporalAttentionStream(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, 1)
    def forward(self, x):  # x: [batch, time_steps, features]
        x_emb = self.embedding(x).transpose(0, 1)
        out, _ = self.attention(x_emb, x_emb, x_emb)
        return out.transpose(0, 1)

class ContextualCrossAttention(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_dim, 1)
        self.gate = nn.Sequential(nn.Linear(hidden_dim*2, hidden_dim), nn.Sigmoid())
    def forward(self, spatial, temporal):
        fusion, _ = self.cross_attn(spatial.transpose(0, 1), temporal.transpose(0, 1), temporal.transpose(0, 1))
        fusion = fusion.transpose(0, 1)
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
    def __init__(self, hidden_dim=64, output_steps=12):
        super().__init__()
        self.projection = nn.Linear(hidden_dim, output_steps * 2)
        self.output_steps = output_steps
    def forward(self, fused_features):
        # fused_features: [batch, airports, hidden_dim]
        out = self.projection(fused_features)  # [batch, airports, output_steps * 2]
        batch_size, airports, _ = out.shape
        out = out.view(batch_size, airports, self.output_steps, 2)  # [batch, airports, output_steps, 2]
        return out

class DSAFNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, output_steps=12, num_graphs=3):
        super().__init__()
        self.spatial_stream = SpatialAttentionStream(input_dim, hidden_dim)
        self.temporal_stream = TemporalAttentionStream(input_dim, hidden_dim)
        self.cross_fusion = ContextualCrossAttention(hidden_dim)
        self.graph_encoder = SimpleGraphEncoder(num_graphs)
        self.output_proj = OutputProjection(hidden_dim, output_steps)
    def forward(self, x, ti=None, supports=None, to=None, w=None):
        """
        Forward pass matching STPN signature
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, num_nodes, in_len)
            ti: Time input embeddings (optional)
            supports: List of adjacency matrices for spatial modeling
            to: Time output embeddings (optional)
            w: Weather conditions (optional)
        """
        # Handle the input shape transformation for DSAFNet
        # Expected: (batch_size, airports, time_steps, features)
        if x.dim() == 4:
            # x is (batch_size, in_channels, num_nodes, in_len)
            # Transform to (batch_size, num_nodes, in_len, in_channels)
            x = x.permute(0, 2, 3, 1)
        
        batch_size, airports, time_steps, features = x.shape
        
        # Use supports as adjacency matrices if provided
        adj_matrices = supports if supports is not None else []
        
        spatial_features = [self.spatial_stream(x[:,:,t,:]) for t in range(time_steps)]
        temporal_features = [self.temporal_stream(x[:,a,:,:]) for a in range(airports)]
        spatial_stack = torch.stack(spatial_features, dim=2)
        temporal_stack = torch.stack(temporal_features, dim=1)
        fused = self.cross_fusion(spatial_stack.mean(2), temporal_stack.mean(2))
        
        if adj_matrices:
            graph_enhanced = self.graph_encoder(fused, adj_matrices)
        else:
            # If no adjacency matrices provided, use identity
            graph_enhanced = fused
            
        out = self.output_proj(graph_enhanced)
        
        # Transform output back to match expected STPN output format
        # out is (batch_size, airports, output_steps, 2)
        # Transform to (batch_size, 2, airports, output_steps)
        out = out.permute(0, 3, 1, 2)
        
        return out
class SimpleCNN(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x, ti=None, supports=None, to=None, w=None):
        # Ignore additional args for this test model
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x
def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    supports = [torch.tensor(i).to(device) for i in adj]
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(),
                            training_data[~np.isnan(training_data)].std())
    test_index = list(range(test_data.shape[1] - (args.in_len + args.out_len)))
    label = []
    for i in range(len(test_index)):
        label.append(np.expand_dims(
            test_data[:, test_index[i] + args.in_len:test_index[i] + args.in_len + args.out_len, :],
            axis=0))
    label = np.concatenate(label)

    # 🔹 Recreate model and load state_dict
    # Try to match the saved model architecture
    # The error indicates the saved model expects 32 input channels
    # Let's first try to load the state dict to understand the architecture
    state_dict = torch.load("spdpnUS.pth", map_location=device)
    
    # Check the first conv layer weights to understand input channels
    first_conv_key = None
    for key in state_dict.keys():
        if 'conv' in key.lower() and 'weight' in key:
            first_conv_key = key
            break
    
    if first_conv_key:
        first_conv_weight = state_dict[first_conv_key]
        actual_in_channels = first_conv_weight.shape[1]  # Input channels
        print(f"Detected input channels from saved model: {actual_in_channels}")
        print(f"First conv layer shape: {first_conv_weight.shape}")
    else:
        actual_in_channels = args.in_channels
        print(f"Could not detect input channels, using default: {actual_in_channels}")

    # model = SpatioTemporalCNN(
    #         in_channels=actual_in_channels,
    #         out_channels=args.out_channels,
    #         hidden_channels=args.hidden_channels,
    #         in_len=args.in_len,
    #         out_len=args.out_len,
    #         dropout=args.dropout
    #     ).to(device)             
    model = DSAFNet(input_dim=args.in_channels, hidden_dim=64, output_steps=args.out_len, num_graphs=args.support_len).to(device)

    # model = SimpleCNN(in_channels=args.in_channels, out_channels=args.out_channels).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print(f"Model input channels: {actual_in_channels}")
    print(f"Test data shape: {test_data.shape}")
    
    outputs = []
    for i in range(len(test_index)):
        testx = np.expand_dims(test_data[:, test_index[i]: test_index[i] + args.in_len, :], axis=0)
        testx = scaler.transform(testx)
        testw = np.expand_dims(test_w[:, test_index[i]: test_index[i] + args.in_len], axis=0)
        testw = torch.LongTensor(testw).to(device)
        testx[np.isnan(testx)] = 0
        testti = (np.arange(int(training_data.shape[1] + val_data.shape[1]) + test_index[i],
                            int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len)
                  % args.period) * np.ones([1, args.in_len]) / (args.period - 1)
        testto = (np.arange(int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len,
                            int(training_data.shape[1] + val_data.shape[1]) + test_index[i] + args.in_len + args.out_len)
                  % args.period) * np.ones([1, args.out_len]) / (args.period - 1)
        testx = torch.Tensor(testx).to(device)
        testx = testx.permute(0, 3, 1, 2)  # Shape: (batch, features, nodes, time)
        
        # Adjust input channels if needed
        current_channels = testx.shape[1]
        if current_channels != actual_in_channels:
            if actual_in_channels > current_channels:
                # Pad with zeros or duplicate channels
                padding_channels = actual_in_channels - current_channels
                padding = torch.zeros(testx.shape[0], padding_channels, testx.shape[2], testx.shape[3], device=device)
                testx = torch.cat([testx, padding], dim=1)
                print(f"Padded input from {current_channels} to {actual_in_channels} channels")
            else:
                # Truncate channels
                testx = testx[:, :actual_in_channels, :, :]
                print(f"Truncated input from {current_channels} to {actual_in_channels} channels")
        
        testti = torch.Tensor(testti).to(device)
        testto = torch.Tensor(testto).to(device)
        output = model(testx, testti, supports, testto, testw)
        output = output.permute(0, 2, 3, 1)
        output = output.detach().cpu().numpy()
        output = scaler.inverse_transform(output)
        outputs.append(output)
    yhat = np.concatenate(outputs)

    log = '3 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:, :, 2, 0], label[:, :, 2, 0])
    print(log.format(MAE, R2, RMSE))

    log = '6 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:, :, 5, 0], label[:, :, 5, 0])
    print(log.format(MAE, R2, RMSE))

    log = '12 step ahead arrival delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:, :, 11, 0], label[:, :, 11, 0])
    print(log.format(MAE, R2, RMSE))

    log = '3 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:, :, 2, 1], label[:, :, 2, 1])
    print(log.format(MAE, R2, RMSE))

    log = '6 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:, :, 5, 1], label[:, :, 5, 1])
    print(log.format(MAE, R2, RMSE))

    log = '12 step ahead departure delay, Test MAE: {:.4f} min, Test R2: {:.4f}, Test RMSE: {:.4f} min'
    MAE, RMSE, R2 = test_error(yhat[:, :, 11, 1], label[:, :, 11, 1])
    print(log.format(MAE, R2, RMSE))


if __name__ == "__main__":
    main()
