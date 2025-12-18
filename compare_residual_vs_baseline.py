"""Quick comparison between baseline SequentialTwoStagePredictor and ResidualKANPredictor.

Runs short training on a small subset and records F1/precision/recall and training time.
"""
import time
import json
import torch
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from classifykat import (
    load_flight_data,
    build_sequences,
    classification_metrics,
    SequentialTwoStagePredictor,
    ResidualKANPredictor,
    set_seed,
)
from torch_geometric.data import Data
import torch.nn as nn


def train_model(model, train_x, train_y_cls, val_x, val_y_cls, edge_indices, device, epochs=5, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    start = time.time()
    best = {"f1": 0.0}
    for epoch in range(epochs):
        model.train()
        losses = []
        for i in range(len(train_x)):
            data = Data(x=train_x[i].to(device), edge_index_adj=edge_indices[0], edge_index_od=edge_indices[1], edge_index_od_t=edge_indices[2])
            opt.zero_grad()
            if hasattr(model, 'forward_classifier'):
                _, logits = model.forward_classifier(data)
            else:
                logits, _ = model(data)
            loss = loss_fn(logits, train_y_cls[i].to(device))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        # val
        model.eval()
        probs, targs = [], []
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(x=val_x[i].to(device), edge_index_adj=edge_indices[0], edge_index_od=edge_indices[1], edge_index_od_t=edge_indices[2])
                if hasattr(model, 'forward_classifier'):
                    _, logits = model.forward_classifier(data)
                else:
                    logits, _ = model(data)
                probs.append(torch.sigmoid(logits).cpu().numpy())
                targs.append(val_y_cls[i].numpy())
        probs = np.vstack(probs)
        targs = np.vstack(targs)
        metrics = classification_metrics(probs.reshape(-1,1), targs.reshape(-1,1))
        if metrics['f1'] > best['f1']:
            best = metrics
    elapsed = time.time() - start
    return best, elapsed


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # load minimal data
    edge_index_adj, edge_index_od, edge_index_od_t, train_inputs, val_inputs, test_inputs, train_delay, val_delay, test_delay, train_raw, val_raw, test_raw, scaler, num_nodes = load_flight_data('udata', weather_file='weather2016_2021.npy', period_hours=24, data_source='udata')

    # use node-level builder
    from classifykat_balanced import build_sequences_node_level
    train_x, train_y_reg, train_y_cls = build_sequences_node_level(train_inputs, train_delay, train_raw, seq_len=8, horizon=3, delay_threshold=5.0, target_horizons=[3,6,12])
    val_x, val_y_reg, val_y_cls = build_sequences_node_level(val_inputs, val_delay, val_raw, seq_len=8, horizon=3, delay_threshold=5.0, target_horizons=[3,6,12])

    # subsample
    n_train = min(2000, len(train_x))
    n_val = min(400, len(val_x))
    train_x = train_x[:n_train]
    train_y_cls = train_y_cls[:n_train]
    val_x = val_x[:n_val]
    val_y_cls = val_y_cls[:n_val]

    feature_dim = train_inputs.shape[2]
    in_channels = 8 * feature_dim
    out_channels = 3

    edge_indices = (edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device))

    results = {}

    # Baseline
    baseline = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=64)
    print('Training baseline...')
    base_metrics, base_time = train_model(baseline, train_x, train_y_cls, val_x, val_y_cls, edge_indices, device, epochs=5)
    results['baseline'] = {'metrics': base_metrics, 'time': base_time}

    # Residual
    residual = ResidualKANPredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=64)
    print('Training residual...')
    res_metrics, res_time = train_model(residual, train_x, train_y_cls, val_x, val_y_cls, edge_indices, device, epochs=5)
    results['residual'] = {'metrics': res_metrics, 'time': res_time}

    with open('compare_residual_vs_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print('Results saved to compare_residual_vs_baseline_results.json')
    print(results)

if __name__ == '__main__':
    main()
