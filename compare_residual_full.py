"""Extended comparison between baseline and residual predictor with per-epoch metrics and plots.

Runs moderately-sized training (subsample) and records epoch-level metrics for plotting.
"""
import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (
    load_flight_data,
    classification_metrics,
    SequentialTwoStagePredictor,
    ResidualKANPredictor,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level
from torch_geometric.data import Data
import torch.nn as nn


def train_epoch(model, train_x, train_y_cls, edge_indices, device, opt, loss_fn):
    model.train()
    losses = []
    for i in range(len(train_x)):
        data = Data(x=train_x[i].to(device))
        data.edge_index_adj = edge_indices[0]
        data.edge_index_od = edge_indices[1]
        data.edge_index_od_t = edge_indices[2]
        opt.zero_grad()
        if hasattr(model, 'forward_classifier'):
            _, logits = model.forward_classifier(data)
        else:
            logits, _ = model(data)
        loss = loss_fn(logits, train_y_cls[i].to(device))
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def evaluate(model, val_x, val_y_cls, edge_indices, device):
    model.eval()
    probs, targs = [], []
    with torch.no_grad():
        for i in range(len(val_x)):
            data = Data(x=val_x[i].to(device))
            data.edge_index_adj = edge_indices[0]
            data.edge_index_od = edge_indices[1]
            data.edge_index_od_t = edge_indices[2]
            if hasattr(model, 'forward_classifier'):
                _, logits = model.forward_classifier(data)
            else:
                logits, _ = model(data)
            probs.append(torch.sigmoid(logits).cpu().numpy())
            targs.append(val_y_cls[i].numpy())
    probs = np.vstack(probs)
    targs = np.vstack(targs)
    return classification_metrics(probs.reshape(-1,1), targs.reshape(-1,1))


def run_comparison(epochs=20, n_train=4000, n_val=800, hidden=64):
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    edge_index_adj, edge_index_od, edge_index_od_t, train_inputs, val_inputs, test_inputs, train_delay, val_delay, test_delay, train_raw, val_raw, test_raw, scaler, num_nodes = load_flight_data('udata', weather_file='weather2016_2021.npy', period_hours=24, data_source='udata')

    train_x, train_y_reg, train_y_cls = build_sequences_node_level(train_inputs, train_delay, train_raw, seq_len=8, horizon=3, delay_threshold=5.0, target_horizons=[3,6,12])
    val_x, val_y_reg, val_y_cls = build_sequences_node_level(val_inputs, val_delay, val_raw, seq_len=8, horizon=3, delay_threshold=5.0, target_horizons=[3,6,12])

    n_train = min(n_train, len(train_x))
    n_val = min(n_val, len(val_x))
    train_x = train_x[:n_train]
    train_y_cls = train_y_cls[:n_train]
    val_x = val_x[:n_val]
    val_y_cls = val_y_cls[:n_val]

    feature_dim = train_inputs.shape[2]
    in_channels = 8 * feature_dim
    out_channels = 3

    edge_indices = (edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device))

    results = {'baseline': {'epochs': []}, 'residual': {'epochs': []}}

    # prepare models
    baseline = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden)
    residual = ResidualKANPredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden)

    for name, model in [('baseline', baseline), ('residual', residual)]:
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss()
        epoch_times = []
        epoch_metrics = []
        print(f"Running {name} for {epochs} epochs on {n_train} train / {n_val} val samples")
        for e in range(epochs):
            t0 = time.time()
            train_loss = train_epoch(model, train_x, train_y_cls, edge_indices, device, opt, loss_fn)
            metrics = evaluate(model, val_x, val_y_cls, edge_indices, device)
            t1 = time.time()
            epoch_times.append(t1 - t0)
            epoch_metrics.append({'train_loss': train_loss, **metrics})
            print(f"{name} epoch {e+1}/{epochs}: loss={train_loss:.4f}, f1={metrics['f1']:.4f}")
        results[name]['epochs'] = epoch_metrics
        results[name]['total_time'] = sum(epoch_times)

    # save results
    fname = f'compare_residual_full_results_epochs{epochs}_train{n_train}.json'
    with open(fname, 'w') as f:
        json.dump(results, f, indent=2, default=float)
    print('Saved', fname)

    # plot
    for metric in ['f1', 'precision', 'recall', 'train_loss']:
        plt.figure()
        for name in ['baseline', 'residual']:
            vals = [ep.get(metric, 0.0) for ep in results[name]['epochs']]
            plt.plot(range(1, len(vals)+1), vals, label=name)
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.title(metric + ' per epoch')
        plt.legend()
        outpng = f'compare_{metric}_epochs{epochs}_train{n_train}.png'
        plt.savefig(outpng)
        print('Saved', outpng)

    return results

if __name__ == '__main__':
    # moderate run
    run_comparison(epochs=20, n_train=4000, n_val=800, hidden=64)
