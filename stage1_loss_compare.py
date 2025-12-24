"""Quick Stage 1 loss comparison: BCE vs Huber vs Focal.

Runs each loss for 5 epochs on a subset of the dataset and reports val metrics.
Default data_source is 'udata' using existing loaders.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
from torch import nn
from torch_geometric.data import Data

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from classifykat import (
    SequentialTwoStagePredictor,
    load_flight_data,
    build_sequences,
    classification_metrics,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level


def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    if target.dim() == 0:
        return target.unsqueeze(0)
    elif target.dim() == 1:
        return target.mean(dim=0, keepdim=True)
    else:
        return target.mean(dim=0, keepdim=True)


def focal_loss_with_logits(logits: torch.Tensor, targets: torch.Tensor, gamma: float, alpha: float) -> torch.Tensor:
    # targets in {0,1}
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt.clamp(min=1e-8))
    return loss.mean()


def huber_prob_loss(logits: torch.Tensor, targets: torch.Tensor, delta: float) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    huber = nn.HuberLoss(reduction="none", delta=delta)
    return huber(prob, targets).mean()


def train_one_loss(
    model: SequentialTwoStagePredictor,
    edge_indices,
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    loss_name: str,
    huber_delta: float,
    focal_gamma: float,
    focal_alpha: float,
    pos_weight: float,
):
    # Freeze regressor
    for p in model.regressor.parameters():
        p.requires_grad = False

    trainable_params = list(model.encoder.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)

    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    if loss_name == "huber":
        def loss_fn(logits, targets):
            return huber_prob_loss(logits, targets, huber_delta)
    elif loss_name == "focal":
        def loss_fn(logits, targets):
            return focal_loss_with_logits(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
    else:
        def loss_fn(logits, targets):
            return bce(logits, targets)

    for epoch in range(1, epochs + 1):
        model.train()
        idx = torch.randperm(len(train_x))
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start+batch_size]
            bx = train_x[batch_idx].to(device)
            by = train_y_cls[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits_list = []
            targets_list = []
            for i in range(len(bx)):
                data = Data(
                    x=bx[i],
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_logits = model.forward_classifier(data.to(device))
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(by[i])
                logits_list.append(graph_logit)
                targets_list.append(graph_target)
            all_logits = torch.cat(logits_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            loss = loss_fn(all_logits, all_targets)
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_logits = model.forward_classifier(data)
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(val_y_cls[i])
                val_probs.append(torch.sigmoid(graph_logit).cpu())
                val_targets.append(graph_target.cpu())
        probs_np = torch.cat(val_probs).numpy()
        targets_np = torch.cat(val_targets).numpy()
        metrics = classification_metrics(probs_np.reshape(-1,1), targets_np.reshape(-1,1))
        print(f"Epoch {epoch}/{epochs} [{loss_name}] Val F1={metrics['f1']:.4f} Acc={metrics['accuracy']:.4f} Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f}")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata','udata'])
    parser.add_argument('--use_node_level', action='store_true', default=True)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3,6,12])
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--sample_size', type=int, default=800)
    parser.add_argument('--val_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--huber_delta', type=float, default=2.0)
    parser.add_argument('--focal_gamma', type=float, default=2.0)
    parser.add_argument('--focal_alpha', type=float, default=0.25)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    if args.data_source == 'udata':
        weather_file = 'weather2016_2021.npy'
    else:
        weather_file = 'weather_cn.npy'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=weather_file,
        period_hours=24,
        data_source=args.data_source,
    )

    horizons = sorted({h for h in args.horizons if h > 0})
    max_horizon = max(horizons)
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = len(horizons) * delay_dim

    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    val_x_full, val_y_reg_full, val_y_cls_full = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )

    # Subsample
    train_idx = np.random.choice(len(train_x), min(args.sample_size, len(train_x)), replace=False)
    val_idx = np.random.choice(len(val_x_full), min(args.val_size, len(val_x_full)), replace=False)
    train_x = train_x[train_idx]
    train_y_cls = train_y_cls[train_idx]
    val_x = val_x_full[val_idx]
    val_y_cls = val_y_cls_full[val_idx]

    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )

    cls_pos_rate = train_y_cls.mean().item()
    pos_weight = (1 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)
    print(f"Class balance (train delayed): {cls_pos_rate:.2%}")

    losses = ["bce", "huber", "focal"]
    results = {}
    for loss_name in losses:
        model = SequentialTwoStagePredictor(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=32,
        ).to(device)
        print(f"\n=== Training loss: {loss_name} ===")
        metrics = train_one_loss(
            model, edge_indices,
            train_x, train_y_cls,
            val_x, val_y_cls,
            device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss_name=loss_name,
            huber_delta=args.huber_delta,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            pos_weight=pos_weight,
        )
        results[loss_name] = metrics

    print("\nSummary (val metrics):")
    for k,v in results.items():
        print(f"{k}: F1={v['f1']:.4f} Acc={v['accuracy']:.4f} Prec={v['precision']:.4f} Rec={v['recall']:.4f}")


if __name__ == "__main__":
    main()
