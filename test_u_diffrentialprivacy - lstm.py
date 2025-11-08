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

args = parser.parse_args()
# class SimpleCNN(nn.Module):
#     def __init__(self, in_channels=2, out_channels=2):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

#     def forward(self, x, ti=None, supports=None, to=None, w=None):
#         # Ignore additional args for this test model
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.conv2(x)
#         return x

from opacus.layers import DPLSTM

# LSTMModel class for DP testing (Opacus compatible, matches training)
class LSTMModel(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, hidden_size=64, num_layers=2, dropout=0.0, num_nodes=70):
        super(LSTMModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        input_size = in_channels * num_nodes
        self.lstm = DPLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0 if num_layers == 1 else 0.3
        )
        self.fc = nn.Linear(hidden_size, out_channels * num_nodes)
        self._initialized_input_size = input_size

    def _rebuild_layers_if_needed(self, features, nodes):
        required_input_size = features * nodes
        if required_input_size != self._initialized_input_size or nodes != self.num_nodes:
            self.lstm = DPLSTM(
                input_size=required_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=0.0 if self.num_layers == 1 else 0.3
            )
            self.fc = nn.Linear(self.hidden_size, self.out_channels * nodes)
            self._initialized_input_size = required_input_size
            self.num_nodes = nodes

    def forward(self, x, ti=None, supports=None, to=None, w=None):
        original_shape = x.shape
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() == 4:
            b, f, n, s = x.shape
            self._rebuild_layers_if_needed(f, n)
            x = x.permute(0, 3, 1, 2).contiguous().view(b, s, f * n)
        else:
            raise ValueError("Input tensor must be 3D or 4D")
        out, _ = self.lstm(x)
        out = self.fc(out)
        if len(original_shape) == 3:
            f, n, s = original_shape
            out = out.view(-1, s, self.out_channels, n).permute(0, 2, 3, 1)
            return out.squeeze(0)
        else:
            b, f, n, s = original_shape
            out = out.view(b, s, self.out_channels, n).permute(0, 2, 3, 1)
            return out
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
    # model = STPN(args.h_layers,
    #              args.in_channels,
    #              args.hidden_channels,
    #              args.out_channels,
    #              args.emb_size,
    #              args.dropout).to(device)
    model = LSTMModel(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        hidden_size=64,
        num_layers=2,
        dropout=args.dropout
    ).to(device)
    #              args.in_channels,
    #              args.hidden_channels,
    #              args.out_channels,
    #              args.emb_size,
    #              args.dropout).to(device)
    
    state_dict = torch.load("spdpnUS.pth", map_location=device)
    #state_dict = torch.load("spdpn1" + args.data + ".pth", map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

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
        testx = testx.permute(0, 3, 1, 2)
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
