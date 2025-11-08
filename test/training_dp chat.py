# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 16:36:04 2022

@author: AA
"""

import torch
import util
import argparse
import random
import copy
import torch.optim as optim
import numpy as np
import os
from datetime import datetime

from baseline_methods import test_error, StandardScaler
from model import STPN

# NEW IMPORTS for DP
from opacus import PrivacyEngine

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='US',help='data type')
parser.add_argument("--train_val_ratio", nargs="+", default=[0.7, 0.1], help='train/val/test ratio', type=float)
parser.add_argument('--h_layers',type=int,default=2,help='number of hidden layer')
parser.add_argument('--in_channels',type=int,default=2,help='input variable')
parser.add_argument("--hidden_channels", nargs="+", default=[128, 64, 32], help='hidden layer dimension', type=int)
parser.add_argument('--out_channels',type=int,default=2,help='output variable')
parser.add_argument('--emb_size',type=int,default=16,help='time embedding size')
parser.add_argument('--dropout',type=float,default=0,help='dropout rate')
parser.add_argument('--wemb_size',type=int,default=4,help='covairate embedding size')
parser.add_argument('--time_d',type=int,default=4,help='normalizing factor for self-attention model')
parser.add_argument('--heads',type=int,default=4,help='number of attention heads')
parser.add_argument('--support_len',type=int,default=3,help='number of spatial adjacency matrix')
parser.add_argument('--order',type=int,default=2,help='order of diffusion convolution')
parser.add_argument('--num_weather',type=int,default=8,help='number of weather condition')
parser.add_argument('--use_se', type=str, default=True,help="use SE block")
parser.add_argument('--use_cov', type=str, default=True,help="use Covariate")
parser.add_argument('--decay', type=float, default=1e-5, help='decay rate of learning rate ')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate ')
parser.add_argument('--in_len',type=int,default=12,help='input time series length')
parser.add_argument('--out_len',type=int,default=12,help='output time series length')
parser.add_argument('--batch',type=int,default=32,help='training batch size')
parser.add_argument('--episode',type=int,default=50,help='training episodes')
parser.add_argument('--period',type=int,default=36,help='periodic for temporal embedding')

args = parser.parse_args()

def setup_logging(args):
    """Setup logging configuration"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'training_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f'training_log_{timestamp}.txt')
    return log_file

def log_message(message, log_file):
    """Print message to console and append to log file"""
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def main():
    device = torch.device(args.device)
    adj, training_data, val_data, test_data, training_w, val_w, test_w = util.load_data(args.data)
    model = STPN(args.h_layers, args.in_channels, args.hidden_channels, args.out_channels, args.emb_size, 
                 args.dropout, args.wemb_size, args.time_d, args.heads, args.support_len,
                 args.order, args.num_weather, args.use_se, args.use_cov).to(device)
    supports = [torch.tensor(i).to(device) for i in adj]
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    
    # Attach PrivacyEngine for Differential Privacy
    privacy_engine = PrivacyEngine()
    model, optimizer,train_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=None,   # manual batching, no DataLoader
        epochs=args.episode,
        target_epsilon=5.0,  # privacy budget
        target_delta=1e-5,
        max_grad_norm=1.0,
    )
    
    #Standardize mean=0,var=1 training data
    scaler = StandardScaler(training_data[~np.isnan(training_data)].mean(), training_data[~np.isnan(training_data)].std())
    training_data = scaler.transform(training_data)
    training_data[np.isnan(training_data)] = 0
    
    MAE_list = []
    batch_index = list(range(training_data.shape[1] - (args.in_len + args.out_len)))
    val_index = list(range(val_data.shape[1] - (args.in_len + args.out_len)))

    label = []
    for i in range(len(val_index)):
        label.append(np.expand_dims(val_data[:, val_index[i] + args.in_len:val_index[i] + args.in_len + args.out_len, :], axis = 0))
    label = np.concatenate(label)
    
    log_file = setup_logging(args)
    log_message("=== Training Configuration ===", log_file)
    log_message(f"Model parameters: {sum(p.numel() for p in model.parameters())}", log_file)
    log_message(f"Learning rate: {args.lr}", log_file)
    log_message(f"Batch size: {args.batch}", log_file)
    
    log_message("start training...", log_file)
    
    for ep in range(1, 1+args.episode):
        random.shuffle(batch_index)
        epoch_losses = []
        epoch_maes = []
        
        for j in range(len(batch_index) // args.batch - 1):
            trainx, trainy, trainti, trainto, trainw = [], [], [], [], []
            for k in range(args.batch):
                trainx.append(np.expand_dims(training_data[:, batch_index[j * args.batch +k]: batch_index[j * args.batch +k] + args.in_len, :], axis = 0))
                trainy.append(np.expand_dims(training_data[:, batch_index[j * args.batch +k] + args.in_len:batch_index[j * args.batch +k] + args.in_len + args.out_len, :], axis = 0))
                trainw.append(np.expand_dims(training_w[:, batch_index[j * args.batch +k]: batch_index[j * args.batch +k] + args.in_len], axis = 0))
                trainti.append((np.arange(batch_index[j * args.batch +k], batch_index[j * args.batch +k] + args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1))
                trainto.append((np.arange(batch_index[j * args.batch +k] + args.in_len, batch_index[j * args.batch +k] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1))

            trainx = np.concatenate(trainx)
            trainti = np.concatenate(trainti)
            trainto = np.concatenate(trainto)
            trainy = np.concatenate(trainy)
            trainw = np.concatenate(trainw)
            trainw = torch.LongTensor(trainw).to(device)
            trainx = torch.Tensor(trainx).to(device)
            trainx= trainx.permute(0, 3, 1, 2)
            trainy = torch.Tensor(trainy).to(device)
            trainy = trainy.permute(0, 3, 1, 2)
            trainti = torch.Tensor(trainti).to(device)
            trainto = torch.Tensor(trainto).to(device)
            model.train()
            optimizer.zero_grad()
            output = model(trainx, trainti, supports, trainto, trainw)
            loss = util.masked_rmse(output, trainy, 0.0)
            epoch_losses.append(loss.item())
            
            with torch.no_grad():
                output_denorm = scaler.inverse_transform(output.permute(0, 2, 3, 1).cpu().numpy())
                target_denorm = scaler.inverse_transform(trainy.permute(0, 2, 3, 1).cpu().numpy())
                train_mae = np.mean(np.abs(output_denorm - target_denorm))
                epoch_maes.append(train_mae)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            
        # Validation phase
        model.eval()
        outputs, amae, ar2, armse = [], [], [], []
        with torch.no_grad():
            for i in range(len(val_index)):
                testx = np.expand_dims(val_data[:, val_index[i]: val_index[i] + args.in_len, :], axis = 0)
                testx = scaler.transform(testx)
                testw = np.expand_dims(val_w[:, val_index[i]: val_index[i] + args.in_len], axis = 0)
                testw = torch.LongTensor(testw).to(device)
                testx[np.isnan(testx)] = 0
                testti = (np.arange(int(training_data.shape[1])+val_index[i], int(training_data.shape[1])+val_index[i]+ args.in_len) % args.period) * np.ones([1, args.in_len])/(args.period - 1)
                testto = (np.arange(int(training_data.shape[1])+val_index[i] + args.in_len, int(training_data.shape[1])+val_index[i] + args.in_len + args.out_len) % args.period) * np.ones([1, args.out_len])/(args.period - 1)
                testx = torch.Tensor(testx).to(device)
                testx= testx.permute(0, 3, 1, 2)
                testti = torch.Tensor(testti).to(device)
                testto = torch.Tensor(testto).to(device)
                output = model(testx, testti, supports, testto, testw)
                output = output.permute(0, 2, 3, 1)
                output = output.detach().cpu().numpy()
                output = scaler.inverse_transform(output)
                outputs.append(output)
             
            yhat = np.concatenate(outputs)
            for i in range(12):
                metrics = test_error(yhat[:,:,i,:], label[:,:,i,:])
                amae.append(metrics[0])
                ar2.append(metrics[2])
                armse.append(metrics[1])
        
        mean_mae = np.mean(amae)
        MAE_list.append(mean_mae)
        
        epoch_summary = f"""
Epoch {ep}/{args.episode}
Training - Average Loss: {np.mean(epoch_losses):.4f}, MAE: {np.mean(epoch_maes):.4f}
Validation - MAE: {mean_mae:.4f}, R2: {np.mean(ar2):.4f}, RMSE: {np.mean(armse):.4f}
{'-'*80}"""
        log_message(epoch_summary, log_file)
        
        if mean_mae == min(MAE_list):
            best_model = copy.deepcopy(model.state_dict())
            log_message("New best model saved!", log_file)
            model.load_state_dict(best_model)
            torch.save(model, "spdpn" + args.data +".pth")
    
    # Report DP budget
    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    log_message(f"Final DP ε = {epsilon:.2f}, δ = 1e-5", log_file)
    log_message("\n=== Training Complete ===", log_file)
    log_message(f"Best validation MAE: {min(MAE_list):.4f}", log_file)
   
if __name__ == "__main__":   
    main()
