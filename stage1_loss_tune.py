"""Stage 1 loss tuner for classification (BCE vs Huber vs Focal).

Runs a grid search over Huber deltas and Focal (gamma, alpha), compares against BCE,
using a subset for speed. Saves JSON results and prints best configs.
"""
from __future__ import annotations

import argparse
import json
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
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1 - prob) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = -alpha_t * (1 - pt).pow(gamma) * torch.log(pt.clamp(min=1e-8))
    return loss.mean()


def huber_prob_loss(logits: torch.Tensor, targets: torch.Tensor, delta: float) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    huber = nn.HuberLoss(reduction="none", delta=delta)
    return huber(prob, targets).mean()


def sse_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * ((logits - targets) ** 2).mean()


def binary_focal_loss_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    alpha_balance: float,
) -> torch.Tensor:
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    pt = probs * targets + (1 - probs) * (1 - targets)
    modulating = (1 - pt).clamp_min(1e-8).pow(gamma)
    alpha_t = alpha_balance * targets + (1 - alpha_balance) * (1 - targets)
    return (modulating * alpha_t * bce).mean()


def activation_l2_penalty(hidden_states: torch.Tensor, device: torch.device) -> torch.Tensor:
    if hidden_states is None:
        return torch.tensor(0.0, device=device)
    dim = max(hidden_states.shape[-1], 1)
    return (hidden_states.pow(2).sum(dim=-1).mean()) / dim


def curriculum_dp_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    hidden: torch.Tensor,
    alpha_value: torch.Tensor,
    beta: float,
    gamma: float,
    alpha_balance: float,
) -> torch.Tensor:
    l_sse = sse_loss(logits, targets)
    l_focal = binary_focal_loss_logits(logits, targets, gamma=gamma, alpha_balance=alpha_balance)
    l_reg = activation_l2_penalty(hidden, device=logits.device)
    alpha_tensor = torch.as_tensor(alpha_value, device=logits.device, dtype=logits.dtype)
    alpha = alpha_tensor.detach()
    return alpha * l_focal + (1 - alpha) * l_sse + (1 - alpha) * (l_reg / beta)


def dp_tailored_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    epoch: int,
    total_epochs: int,
    gamma: float = 2.0,
    class_weights: torch.Tensor | None = None,
    mid_frac: float = 0.5,
    tau: float = 5.0,
    lambda_sse: float = 1.0,
    lambda_focal: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Curriculum blend: early SSE on probs, late focal on hard examples.

    For binary logits shaped [B, 1], promote to 2-class logits before softmax.
    """

    # Flatten labels to int class ids
    t_ids = labels.view(-1).round().long().clamp(0, 1)

    # Promote binary logits to two-class logits for stable softmax
    if logits.dim() == 2 and logits.shape[1] == 1:
        logits_two = torch.cat([-logits, logits], dim=1)
    else:
        logits_two = logits

    p = torch.softmax(logits_two, dim=1)
    p_t = p.gather(1, t_ids[:, None]).squeeze(1)

    focal = -((1 - p_t).clamp_min(eps) ** gamma) * torch.log(p_t.clamp_min(eps))
    if class_weights is not None:
        focal = class_weights[t_ids] * focal

    y = torch.nn.functional.one_hot(t_ids, num_classes=logits_two.size(1)).float()
    sse = 0.5 * torch.sum((p - y) ** 2, dim=1)

    mid = total_epochs * mid_frac
    alpha = torch.sigmoid(torch.tensor((epoch - mid) / tau, device=logits.device, dtype=logits.dtype))
    loss = (1 - alpha) * (lambda_sse * sse) + alpha * (lambda_focal * focal)
    return loss.mean()


def evaluate_loss(
    loss_name: str,
    model: SequentialTwoStagePredictor,
    edge_indices,
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    huber_delta: float,
    focal_gamma: float,
    focal_alpha: float,
    pos_weight: float,
    composite_beta: float,
    composite_gamma: float,
    composite_alpha_epoch: float,
    use_class_balance: bool,
    dp_tailored_mid_frac: float = 0.5,
    dp_tailored_tau: float = 5.0,
    dp_tailored_lambda_sse: float = 1.0,
    dp_tailored_lambda_focal: float = 1.0,
):
    # Freeze regressor
    for p in model.regressor.parameters():
        p.requires_grad = False

    trainable_params = list(model.encoder.parameters()) + list(model.classifier.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-4)

    pos_weight_tensor = torch.tensor([pos_weight], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    loss_mode = loss_name.lower()
    if loss_mode == "huber":
        def loss_fn(logits, targets, hidden=None, **_):
            return huber_prob_loss(logits, targets, huber_delta)
    elif loss_mode == "focal":
        def loss_fn(logits, targets, hidden=None, **_):
            return focal_loss_with_logits(logits, targets, gamma=focal_gamma, alpha=focal_alpha)
    elif loss_mode == "dp_combo":
        def loss_fn(logits, targets, hidden=None, alpha_value=None, **_):
            if alpha_value is None:
                raise ValueError("alpha_value required for dp_combo")
            return curriculum_dp_loss(
                logits,
                targets,
                hidden=hidden,
                alpha_value=alpha_value,
                beta=composite_beta,
                gamma=composite_gamma,
                alpha_balance=pos_weight if use_class_balance else 0.5,
            )
    elif loss_mode == "dp_tailored":
        def loss_fn(logits, targets, hidden=None, epoch=None, **_):
            if epoch is None:
                raise ValueError("epoch required for dp_tailored")
            return dp_tailored_loss(
                logits,
                targets,
                epoch=epoch,
                total_epochs=epochs,
                gamma=focal_gamma,
                class_weights=None,
                mid_frac=dp_tailored_mid_frac,
                tau=dp_tailored_tau,
                lambda_sse=dp_tailored_lambda_sse,
                lambda_focal=dp_tailored_lambda_focal,
            )
    else:
        def loss_fn(logits, targets, hidden=None, **_):
            return bce(logits, targets)

    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        model.train()
        current_alpha = None
        if loss_mode == "dp_combo":
            current_alpha = torch.sigmoid(
                torch.tensor(epoch - composite_alpha_epoch, device=device, dtype=torch.float32)
            )
        idx = torch.randperm(len(train_x))
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start+batch_size]
            bx = train_x[batch_idx].to(device)
            by = train_y_cls[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits_list = []
            targets_list = []
            hidden_list = []
            for i in range(len(bx)):
                data = Data(
                    x=bx[i],
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                hidden, node_logits = model.forward_classifier(data.to(device))
                graph_hidden = aggregate_node_to_graph(hidden)
                graph_logit = aggregate_node_to_graph(node_logits)
                graph_target = ensure_graph_level_target(by[i])
                logits_list.append(graph_logit)
                targets_list.append(graph_target)
                hidden_list.append(graph_hidden)
            all_logits = torch.cat(logits_list, dim=0)
            all_targets = torch.cat(targets_list, dim=0)
            all_hidden = torch.cat(hidden_list, dim=0)
            loss = loss_fn(
                all_logits,
                all_targets,
                hidden=all_hidden,
                alpha_value=current_alpha,
                epoch=epoch,
            )
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1

        if batch_count > 0:
            mean_loss = epoch_loss / batch_count
            print(f"[{loss_name}] epoch {epoch + 1}/{epochs} mean loss: {mean_loss:.4f}")

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
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata','udata'])
    parser.add_argument('--use_node_level', action='store_true', default=True)
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3,6,12])
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--sample_size', type=int, default=1200)
    parser.add_argument('--val_size', type=int, default=400)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--huber_deltas', type=str, default='1.0,2.0,3.0')
    parser.add_argument('--focal_gammas', type=str, default='1.5,2.0')
    parser.add_argument('--focal_alphas', type=str, default='0.25,0.5')
    parser.add_argument('--dp_tailored_gammas', type=str, default='2.0,3.0')
    parser.add_argument('--dp_tailored_mid_frac', type=float, default=0.5)
    parser.add_argument('--dp_tailored_tau', type=float, default=5.0)
    parser.add_argument('--dp_tailored_lambda_sse', type=float, default=1.0)
    parser.add_argument('--dp_tailored_lambda_focal', type=float, default=1.0)
    parser.add_argument('--composite_betas', type=str, default='2.0,4.0')
    parser.add_argument('--composite_gammas', type=str, default='1.5,2.0')
    parser.add_argument('--composite_alpha_epochs', type=str, default='1,2')
    parser.add_argument('--composite_use_class_balance', dest='composite_use_class_balance', action='store_true', help='Enable class balance weight inside focal term for composite loss')
    parser.add_argument('--composite_no_class_balance', dest='composite_use_class_balance', action='store_false', help='Disable class balance weight for composite loss')
    parser.set_defaults(composite_use_class_balance=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='stage1_loss_tune_results.json')
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
    train_x, _, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    val_x_full, _, val_y_cls_full = build_fn(
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

    results = []

    # BCE baseline
    print("Running BCE baseline...")
    model = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=32).to(device)
    metrics = evaluate_loss(
        "bce", model, edge_indices,
        train_x, train_y_cls, val_x, val_y_cls,
        device, args.epochs, args.batch_size,
        huber_delta=0.0, focal_gamma=0.0, focal_alpha=0.0,
        pos_weight=pos_weight,
        composite_beta=0.0, composite_gamma=0.0, composite_alpha_epoch=0.0,
        use_class_balance=args.composite_use_class_balance,
    )
    bce_acc = metrics.get("accuracy")
    bce_acc_str = f"{bce_acc:.3f}" if bce_acc is not None else "n/a"
    print(f"Finished BCE baseline | F1={metrics['f1']:.3f} Acc={bce_acc_str}")
    results.append({
        "loss": "bce",
        "huber_delta": None,
        "focal_gamma": None,
        "focal_alpha": None,
        "composite_beta": None,
        "composite_gamma": None,
        "composite_alpha_epoch": None,
        **metrics,
    })

    # Huber grid
    for hd in [float(x) for x in args.huber_deltas.split(',') if x.strip()]:
        print(f"Running Huber delta={hd}...")
        model = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=32).to(device)
        metrics = evaluate_loss(
            "huber", model, edge_indices,
            train_x, train_y_cls, val_x, val_y_cls,
            device, args.epochs, args.batch_size,
            huber_delta=hd, focal_gamma=0.0, focal_alpha=0.0,
            pos_weight=pos_weight,
            composite_beta=0.0, composite_gamma=0.0, composite_alpha_epoch=0.0,
            use_class_balance=args.composite_use_class_balance,
        )
        huber_acc = metrics.get("accuracy")
        huber_acc_str = f"{huber_acc:.3f}" if huber_acc is not None else "n/a"
        print(f"Finished Huber delta={hd} | F1={metrics['f1']:.3f} Acc={huber_acc_str}")
        results.append({
            "loss": "huber",
            "huber_delta": hd,
            "focal_gamma": None,
            "focal_alpha": None,
            "composite_beta": None,
            "composite_gamma": None,
            "composite_alpha_epoch": None,
            **metrics,
        })

    # Focal grid
    gammas = [float(x) for x in args.focal_gammas.split(',') if x.strip()]
    alphas = [float(x) for x in args.focal_alphas.split(',') if x.strip()]
    for g in gammas:
        for a in alphas:
            print(f"Running Focal gamma={g}, alpha={a}...")
            model = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=32).to(device)
            metrics = evaluate_loss(
                "focal", model, edge_indices,
                train_x, train_y_cls, val_x, val_y_cls,
                device, args.epochs, args.batch_size,
                huber_delta=0.0, focal_gamma=g, focal_alpha=a,
                pos_weight=pos_weight,
                composite_beta=0.0, composite_gamma=0.0, composite_alpha_epoch=0.0,
                use_class_balance=args.composite_use_class_balance,
            )
            focal_acc = metrics.get("accuracy")
            focal_acc_str = f"{focal_acc:.3f}" if focal_acc is not None else "n/a"
            print(f"Finished Focal gamma={g}, alpha={a} | F1={metrics['f1']:.3f} Acc={focal_acc_str}")
            results.append({
                "loss": "focal",
                "huber_delta": None,
                "focal_gamma": g,
                "focal_alpha": a,
                "composite_beta": None,
                "composite_gamma": None,
                "composite_alpha_epoch": None,
                **metrics,
            })

    # DP-tailored curriculum (SSE → Focal)
    dp_tailored_gammas = [float(x) for x in args.dp_tailored_gammas.split(',') if x.strip()]
    for g in dp_tailored_gammas:
        print(f"Running DP tailored (gamma={g})...")
        model = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=32).to(device)
        metrics = evaluate_loss(
            "dp_tailored", model, edge_indices,
            train_x, train_y_cls, val_x, val_y_cls,
            device, args.epochs, args.batch_size,
            huber_delta=0.0, focal_gamma=g, focal_alpha=0.0,
            pos_weight=pos_weight,
            composite_beta=0.0, composite_gamma=0.0, composite_alpha_epoch=0.0,
            use_class_balance=args.composite_use_class_balance,
            dp_tailored_mid_frac=args.dp_tailored_mid_frac,
            dp_tailored_tau=args.dp_tailored_tau,
            dp_tailored_lambda_sse=args.dp_tailored_lambda_sse,
            dp_tailored_lambda_focal=args.dp_tailored_lambda_focal,
        )
        dp_acc = metrics.get("accuracy")
        dp_acc_str = f"{dp_acc:.3f}" if dp_acc is not None else "n/a"
        print(f"Finished DP tailored gamma={g} | F1={metrics['f1']:.3f} Acc={dp_acc_str}")
        results.append({
            "loss": "dp_tailored",
            "huber_delta": None,
            "focal_gamma": g,
            "focal_alpha": None,
            "composite_beta": None,
            "composite_gamma": None,
            "composite_alpha_epoch": None,
            **metrics,
        })

    # Composite DP-inspired loss grid
    comp_betas = [float(x) for x in args.composite_betas.split(',') if x.strip()]
    comp_gammas = [float(x) for x in args.composite_gammas.split(',') if x.strip()]
    comp_alpha_epochs = [float(x) for x in args.composite_alpha_epochs.split(',') if x.strip()]
    for beta in comp_betas:
        for g in comp_gammas:
            for alpha_epoch in comp_alpha_epochs:
                print(f"Running DP combo beta={beta}, gamma={g}, alpha_epoch={alpha_epoch}...")
                model = SequentialTwoStagePredictor(in_channels=in_channels, out_channels=out_channels, hidden_channels=32).to(device)
                metrics = evaluate_loss(
                    "dp_combo", model, edge_indices,
                    train_x, train_y_cls, val_x, val_y_cls,
                    device, args.epochs, args.batch_size,
                    huber_delta=0.0, focal_gamma=0.0, focal_alpha=0.0,
                    pos_weight=pos_weight,
                    composite_beta=beta, composite_gamma=g, composite_alpha_epoch=alpha_epoch,
                    use_class_balance=args.composite_use_class_balance,
                )
                dp_acc = metrics.get("accuracy")
                dp_acc_str = f"{dp_acc:.3f}" if dp_acc is not None else "n/a"
                print(f"Finished DP combo beta={beta}, gamma={g}, alpha_epoch={alpha_epoch} | F1={metrics['f1']:.3f} Acc={dp_acc_str}")
                results.append({
                    "loss": "dp_combo",
                    "huber_delta": None,
                    "focal_gamma": None,
                    "focal_alpha": None,
                    "composite_beta": beta,
                    "composite_gamma": g,
                    "composite_alpha_epoch": alpha_epoch,
                    **metrics,
                })

    # pick best by F1
    best = max(results, key=lambda r: r['f1'])
    out_path = os.path.join(os.getcwd(), args.out)
    with open(out_path, 'w') as f:
        json.dump({"best": best, "all": results}, f, indent=2)

    print("\nBest config (by F1):", best)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
