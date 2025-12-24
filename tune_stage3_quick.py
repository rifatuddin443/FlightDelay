"""Quick hyperparameter tuner for Stage 3 (Huber delta and auxiliary weight).

This script runs short grid-searches on small subsets to find good
Huber `delta` and auxiliary weight values for the Stage 3 regressor.

Usage (example):
    python tune_stage3_quick.py --sample_size 512 --epochs 3 --batch_size 16

Notes:
- Uses the same data loader helpers used by the main script (`load_flight_data`).
- Evaluates validation non-delayed MSE in denormalized space (same metric as Stage 3).
- Saves results to `tune_stage3_results.json` in the working directory.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

# Import project utilities (same as main script)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (  # noqa: E402
    SequentialTwoStagePredictor,
    load_flight_data,
    build_sequences,
)

# Local helper implementations (same as main script) to avoid import issues
def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 0:
        return target.unsqueeze(0)
    elif target.dim() == 1:
        return target.mean(dim=0, keepdim=True)
    else:
        return target.mean(dim=0, keepdim=True)


def evaluate_config(
    delta: float,
    aux_weight: float,
    aux_start_epoch: int,
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    scaler,
    edge_indices: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    delay_threshold: float,
    sample_verbose: bool = False,
) -> float:
    """Train Stage 3 briefly with given params and return validation non-delayed MSE."""
    model = SequentialTwoStagePredictor(in_channels=train_x.shape[2] * 1, out_channels=train_y_reg.shape[2], hidden_channels=32).to(device)

    # Freeze encoder/classifier (we tune regressor behavior only)
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.regressor.parameters(), lr=lr * 0.01, weight_decay=1e-5)
    reg_loss_fn = nn.HuberLoss(reduction='none', delta=delta)

    # quick training
    model.train()
    for epoch in range(1, epochs + 1):
        idx = torch.randperm(len(train_x))
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start+batch_size]
            bx = train_x[batch_idx].to(device)
            by = train_y_reg[batch_idx].to(device)

            optimizer.zero_grad()
            preds = []
            targets = []
            for i in range(len(bx)):
                data = Data(x=bx[i], edge_index_adj=edge_indices[0], edge_index_od=edge_indices[1], edge_index_od_t=edge_indices[2])
                _, node_reg = model(data.to(device))
                graph_reg = aggregate_node_to_graph(node_reg)
                graph_target = ensure_graph_level_target(by[i])
                preds.append(graph_reg)
                targets.append(graph_target)

            preds_t = torch.cat(preds, dim=0)
            targets_t = torch.cat(targets, dim=0)

            # denorm for mask
            if scaler is not None:
                with torch.no_grad():
                    targets_denorm = torch.from_numpy(scaler.inverse_transform(targets_t.detach().cpu().numpy())).to(device)
            else:
                targets_denorm = targets_t.detach()

            element_mask = (targets_denorm.abs() < delay_threshold).float()
            num_nd = element_mask.sum()
            if num_nd > 0:
                loss_nd = (reg_loss_fn(preds_t, targets_t) * element_mask).sum() / num_nd
            else:
                loss_nd = torch.tensor(0.0, device=device, requires_grad=True)

            # delayed aux
            delayed_mask = (targets_denorm.abs() >= delay_threshold).float()
            num_d = delayed_mask.sum()
            if num_d > 0 and epoch >= aux_start_epoch and aux_weight > 0.0:
                loss_d = (((preds_t - targets_t) ** 2) * delayed_mask).sum() / num_d
            else:
                loss_d = torch.tensor(0.0, device=device)

            loss = (1.0 - aux_weight) * loss_nd + aux_weight * loss_d
            loss.backward()
            optimizer.step()

    # validation
    model.eval()
    val_losses = []
    with torch.no_grad():
        for i in range(len(val_x)):
            data = Data(x=val_x[i].to(device), edge_index_adj=edge_indices[0], edge_index_od=edge_indices[1], edge_index_od_t=edge_indices[2])
            _, node_reg = model(data)
            graph_reg = aggregate_node_to_graph(node_reg)
            graph_target = ensure_graph_level_target(val_y_reg[i]).to(device)

            if scaler is not None:
                target_denorm = torch.from_numpy(scaler.inverse_transform(graph_target.cpu().numpy())).to(device)
                pred_denorm = torch.from_numpy(scaler.inverse_transform(graph_reg.cpu().numpy())).to(device)
            else:
                target_denorm = graph_target
                pred_denorm = graph_reg

            element_mask = (target_denorm.abs() < delay_threshold).float()
            num_nd = element_mask.sum()
            if num_nd > 0:
                se = ((pred_denorm - target_denorm) ** 2) * element_mask
                val_losses.append((se.sum() / num_nd).item())

    val_loss = float(np.mean(val_losses)) if val_losses else float('inf')
    return val_loss


def parse_list(arg: str) -> List[float]:
    return [float(x) for x in arg.split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--deltas', type=str, default='0.5,1.0,2.0')
    parser.add_argument('--aux_weights', type=str, default='0.0,0.1,0.2')
    parser.add_argument('--aux_start_epochs', type=str, default='1,2')
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'])
    parser.add_argument('--weather_file', type=str, default=None)
    parser.add_argument('--period_hours', type=int, default=24)
    args = parser.parse_args()

    deltas = parse_list(args.deltas)
    aux_weights = parse_list(args.aux_weights)
    aux_start_epochs = [int(x) for x in args.aux_start_epochs.split(',') if x.strip()]

    if args.data_source == 'udata' and args.weather_file is None:
        args.weather_file = 'weather2016_2021.npy'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # Load data (small subset for quick tuning)
    # Load real dataset; do not fall back to synthetic data.
    try:
        (
            edge_index_adj, edge_index_od, edge_index_od_t,
            train_inputs, val_inputs, test_inputs,
            train_delay_scaled, val_delay_scaled, test_delay_scaled,
            train_raw, val_raw, test_raw,
            scaler, num_nodes,
        ) = load_flight_data(
            args.data_source,
            weather_file=args.weather_file,
            period_hours=args.period_hours,
            data_source=args.data_source,
        )
    except FileNotFoundError as e:
        print(f"[ERROR] Required dataset files not found: {e}")
        print("Place the dataset files under the expected path or run the main data preparation script.")
        print("Expected files include 'udata/od_mx.npy' when using '--data_source udata'.")
        raise SystemExit(1)
    except Exception as e:
        print(f"[ERROR] load_flight_data failed: {e}")
        raise

    # build sequences (graph- or node-level as default in main)
    train_x, train_y_reg, train_y_cls = build_sequences(train_inputs, train_delay_scaled, train_raw, 8, max([3,6,12]), args.delay_threshold, [3,6,12])
    val_x, val_y_reg, val_y_cls = build_sequences(val_inputs, val_delay_scaled, val_raw, 8, max([3,6,12]), args.delay_threshold, [3,6,12])

    # subsample
    idx_train = np.random.choice(len(train_x), min(args.sample_size, len(train_x)), replace=False)
    idx_val = np.random.choice(len(val_x), min(args.sample_size//4, len(val_x)), replace=False)
    train_x_s = train_x[idx_train]
    train_y_reg_s = train_y_reg[idx_train]
    val_x_s = val_x[idx_val]
    val_y_reg_s = val_y_reg[idx_val]

    edge_indices = (edge_index_adj.to(device), edge_index_od.to(device), edge_index_od_t.to(device))

    results = []
    total_runs = len(deltas) * len(aux_weights) * len(aux_start_epochs)
    run_i = 0
    start_all = time.time()
    for delta, aw, ase in itertools.product(deltas, aux_weights, aux_start_epochs):
        run_i += 1
        t0 = time.time()
        print(f"Run {run_i}/{total_runs}: delta={delta}, aux_weight={aw}, aux_start_epoch={ase}")
        val_loss = evaluate_config(
            delta=delta,
            aux_weight=aw,
            aux_start_epoch=ase,
            train_x=train_x_s,
            train_y_reg=train_y_reg_s,
            val_x=val_x_s,
            val_y_reg=val_y_reg_s,
            scaler=scaler,
            edge_indices=edge_indices,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            delay_threshold=args.delay_threshold,
        )
        t1 = time.time()
        print(f"  -> val_non_delayed_MSE={val_loss:.6f} | time={t1-t0:.1f}s")
        results.append({'delta': delta, 'aux_weight': aw, 'aux_start_epoch': ase, 'val_loss': val_loss, 'time_s': t1-t0})

    # pick best
    best = min(results, key=lambda r: r['val_loss'])
    total_time = time.time() - start_all
    summary = {'best': best, 'all': results, 'total_time_s': total_time}

    out_path = Path('tune_stage3_results.json')
    out_path.write_text(json.dumps(summary, indent=2))
    print('\nTuning finished. Best config:')
    print(json.dumps(best, indent=2))
    print(f"Results saved to {out_path.resolve()}")


if __name__ == '__main__':
    main()
