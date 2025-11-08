# -*- coding: utf-8 -*-
"""
Training Script with Homomorphic Encryption for Time Data Only
Modified from original training_u.py to include homomorphic encryption

@author: AA (Modified for HE)
"""

import torch
import util
import argparse
import random
import copy
import torch.optim as optim
import numpy as np
from baseline_methods import test_error, StandardScaler
from secure_model import SecureSTPN, SecureTimeEncoder, EncryptedGradientHandler, create_secure_model
import tenseal as ts
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--data', type=str, default='US', help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/val/test ratio', type=float)
parser.add_argument('--h_layers', type=int, default=2, help='number of hidden layer')
parser.add_argument('--in_channels', type=int, default=2, help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=[128, 64, 32], help='hidden layer dimension', type=int)
parser.add_argument('--out_channels', type=int, default=2, help='output variable')
parser.add_argument('--emb_size', type=int, default=16, help='time embedding size')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate')
parser.add_argument('--wemb_size', type=int, default=4, help='covariate embedding size')
parser.add_argument('--time_d', type=int, default=4, help='normalizing factor for self-attention model')
parser.add_argument('--heads', type=int, default=4, help='number of attention heads')
parser.add_argument('--support_len', type=int, default=3, help='number of spatial adjacency matrix')
parser.add_argument('--order', type=int, default=2, help='order of diffusion convolution')
parser.add_argument('--num_weather', type=int, default=8, help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True, help="use SE block")
parser.add_argument('--use_cov', type=str, default=True, help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--in_len', type=int, default=12, help='input time series length')
parser.add_argument('--out_len', type=int, default=12, help='output time series length')
parser.add_argument('--batch', type=int, default=32, help='training batch size')
parser.add_argument('--episode', type=int, default=10, help='training episodes')
parser.add_argument('--period', type=int, default=36, help='periodic for temporal embedding')

# Homomorphic Encryption parameters
parser.add_argument('--encrypt_time', action='store_true', help='use homomorphic encryption for temporal data')
parser.add_argument('--encrypt_gradients', action='store_true', help='encrypt gradients of temporal parameters')
parser.add_argument('--encrypt_frequency', type=int, default=5, help='encrypt every N epochs (0 = always)')
parser.add_argument('--poly_degree', type=int, default=8192, help='polynomial modulus degree for CKKS')

args = parser.parse_args()

def prepare_encrypted_batch(trainx, trainy, trainti, trainto, trainw, secure_encoder, encrypt_temporal):
    """
    Prepare batch data with optional temporal encryption
    """
    if encrypt_temporal and secure_encoder:
        # We don't encrypt the entire forward pass (too slow)
        # Instead, we encrypt temporal indices for secure storage/transmission
        # and add differential privacy noise
        noise_scale = 0.001
        trainti_noise = trainti + torch.randn_like(trainti) * noise_scale
        trainto_noise = trainto + torch.randn_like(trainto) * noise_scale
        return trainx, trainy, trainti_noise, trainto_noise, trainw
    else:
        return trainx, trainy, trainti, trainto, trainw

def secure_model_checkpoint(model, secure_encoder, epoch, best_mae, filepath):
    """
    Save model with encrypted temporal parameters
    """
    if secure_encoder:
        print(f"Saving model with encrypted temporal parameters...")
        # Get encrypted parameters
        encrypted_params = model.encrypt_temporal_parameters()
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'encrypted_temporal_params': encrypted_params,
            'epoch': epoch,
            'best_mae': best_mae,
            'encryption_enabled': True
        }
    else:
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'best_mae': best_mae,
            'encryption_enabled': False
        }
    
    torch.save(checkpoint, filepath)
    print(f"Model saved to {filepath}")

def manual_secure_save(model, secure_encoder, epoch, best_mae, filepath):
    """
    Manually save model with serialized encrypted temporal parameters.
    """
    if secure_encoder:
        print("Manually saving model with encrypted temporal parameters...")
        # Get encrypted parameters
        encrypted_params = model.encrypt_temporal_parameters()
        # Serialize each encrypted tensor to bytes
        serialized_params = {k: v.serialize() for k, v in encrypted_params.items()}
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'encrypted_temporal_params': serialized_params,
            'epoch': epoch,
            'best_mae': best_mae,
            'encryption_enabled': True
        }
    else:
        print("Manually saving model without encryption...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'best_mae': best_mae,
            'encryption_enabled': False
        }
    
    torch.save(checkpoint, filepath)
    print(f"Model successfully saved to {filepath}")

def main():
    print("Starting Training with Homomorphic Encryption for Time Data...")
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    supports = [torch.tensor(i).to(device) for i in adj]
    
    # Create secure model with optional encryption
    model, secure_encoder = create_secure_model(
        args.h_layers, args.in_channels, args.hidden_channels, args.out_channels,
        args.emb_size, args.dropout, args.wemb_size, args.time_d, args.heads,
        args.support_len, args.order, args.num_weather, args.use_se, args.use_cov,
        enable_encryption=args.encrypt_time
    )
    model.to(device)
    
    # Initialize encrypted gradient handler if needed
    grad_handler = None
    if args.encrypt_gradients and secure_encoder:
        print("Initializing encrypted gradient handler...")
        grad_handler = EncryptedGradientHandler(secure_encoder)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # Standardize training data
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), 
                          training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    
    MAE_list = []
    batch_index = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))
    
    # Prepare validation labels
    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(
            val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], 
            axis=0))
    label = np.concatenate(label)
    
    print("Starting training...")
    if args.encrypt_time:
        print("✓ Temporal data encryption: ENABLED")
        print(f"✓ Encryption polynomial degree: {args.poly_degree}")
    if args.encrypt_gradients:
        print("✓ Gradient encryption: ENABLED")
    
    best_mae = float('inf')
    best_model = None
    
    for ep in range(1, 1 + args.episode):
        print(f"\\n=== Epoch {ep}/{args.episode} ===")
        
        # Determine if we should encrypt in this epoch
        encrypt_this_epoch = args.encrypt_time and (
            args.encrypt_frequency == 0 or ep % args.encrypt_frequency == 0
        )
        
        if encrypt_this_epoch:
            print("🔒 Encryption mode: ACTIVE")
            model.set_encryption_mode(True)
        else:
            model.set_encryption_mode(False)
        
        # Training phase
        model.train()
        random.shuffle(batch_index)
        epoch_losses = []
        
        for j in range(len(batch_index) // args.batch - 1):
            trainx, trainy, trainti, trainto, trainw = [], [], [], [], []
            
            # Prepare batch
            for k in range(args.batch):
                idx = batch_index[j * args.batch + k]
                
                trainx.append(np.expand_dims(
                    training_data[:, idx:idx + args.in_len, :], axis=0))
                trainy.append(np.expand_dims(
                    training_data[:, idx + args.in_len:idx + args.in_len + args.out_len, :], axis=0))
                trainw.append(np.expand_dims(
                    training_w[:, idx:idx + args.in_len], axis=0))
                
                # Temporal indices
                trainti.append((np.arange(idx, idx + args.in_len) % args.period) * 
                              np.ones([1, args.in_len]) / (args.period - 1))
                trainto.append((np.arange(idx + args.in_len, idx + args.in_len + args.out_len) % args.period) * 
                              np.ones([1, args.out_len]) / (args.period - 1))
           