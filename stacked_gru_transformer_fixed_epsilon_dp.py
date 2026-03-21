"""
Opacus-free Differentially Private training for StackedGRUTransformer
===================================================================
This script adds manual DP-SGD (per-sample clipping + Gaussian noise)
with a fixed target epsilon.

Key points:
- Reuses the architecture from `stacked_gru_transformer.py`
- Does NOT depend on Opacus
- Computes a noise multiplier from target (epsilon, delta) using a
  conservative analytical upper-bound approximation

Privacy/accounting note:
The epsilon accountant implemented here is an approximation for subsampled
Gaussian mechanism. It is practical and monotonic for tuning sigma, but not as
tight as full RDP accountants.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
import traceback
import types
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

try:
    import torch.func as torch_func
except Exception:
    torch_func = None

from classifykat import EarlyStopping, load_flight_data, set_seed
from classifykat_balanced import build_sequences_node_level
from stacked_gru_transformer import (
    StackedGRUTransformer,
    batch_edge_index,
    build_graph_tensors,
    classification_metrics_per_channel,
)


def epsilon_upper_bound_approx(
    *,
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """Approximate ε upper bound for subsampled Gaussian DP-SGD.

    Uses a simple monotonic bound:
      ε ≈ q * sqrt(2T log(1/δ)) / σ + T q^2 / σ^2
    where q is sample rate, T steps, σ noise multiplier.
    """
    if noise_multiplier <= 0:
        return float("inf")
    if sample_rate <= 0 or steps <= 0 or not (0.0 < delta < 1.0):
        return float("inf")

    term1 = sample_rate * math.sqrt(2.0 * steps * math.log(1.0 / delta)) / noise_multiplier
    term2 = steps * (sample_rate ** 2) / (noise_multiplier ** 2)
    return float(term1 + term2)


def solve_noise_multiplier_for_epsilon(
    *,
    target_epsilon: float,
    delta: float,
    sample_rate: float,
    steps: int,
    tol: float = 1e-3,
    max_iter: int = 80,
) -> float:
    """Binary-search σ such that approximate ε(σ) <= target_epsilon."""
    if target_epsilon <= 0:
        raise ValueError("target_epsilon must be > 0")

    lo, hi = 1e-4, 1.0
    eps_hi = epsilon_upper_bound_approx(
        noise_multiplier=hi,
        sample_rate=sample_rate,
        steps=steps,
        delta=delta,
    )

    expand_guard = 0
    while eps_hi > target_epsilon and expand_guard < 60:
        hi *= 2.0
        eps_hi = epsilon_upper_bound_approx(
            noise_multiplier=hi,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        expand_guard += 1

    if eps_hi > target_epsilon:
        raise RuntimeError(
            "Could not find finite noise multiplier for requested epsilon. "
            "Try a larger epsilon, larger batch size, fewer epochs, or larger delta."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        eps_mid = epsilon_upper_bound_approx(
            noise_multiplier=mid,
            sample_rate=sample_rate,
            steps=steps,
            delta=delta,
        )
        if eps_mid <= target_epsilon:
            hi = mid
        else:
            lo = mid
        if abs(hi - lo) <= tol:
            break

    return float(hi)


def _train_step_dp(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    bx: torch.Tensor,
    by: torch.Tensor,
    ei_adj: torch.Tensor,
    ei_od: torch.Tensor,
    ei_od_t: torch.Tensor,
    n_nodes: int,
    max_grad_norm: float,
    noise_multiplier: float,
    device: torch.device,
    dp_microbatch_size: int,
    try_vmap: bool,
) -> float:
    """One DP-SGD optimizer step using per-sample clipping with vectorized grads."""
    params = [p for p in model.parameters() if p.requires_grad]
    accum_grads = [torch.zeros_like(p, device=device) for p in params]

    bsz = bx.size(0)
    total_loss = 0.0
    micro_bsz = bsz if dp_microbatch_size <= 0 else min(dp_microbatch_size, bsz)

    can_use_vmap = try_vmap and (torch_func is not None) and (not getattr(model, "_dp_disable_vmap", False))

    if can_use_vmap:
        named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        param_names = [name for name, _ in named_params]
        param_values = tuple(p for _, p in named_params)
        buffers = {name: b for name, b in model.named_buffers()}

        bei_adj_1 = batch_edge_index(ei_adj, n_nodes, 1)
        bei_od_1 = batch_edge_index(ei_od, n_nodes, 1)
        bei_od_t_1 = batch_edge_index(ei_od_t, n_nodes, 1)

        def single_loss(param_tuple, x_single, y_single):
            p_dict = {n: v for n, v in zip(param_names, param_tuple)}
            logits = torch_func.functional_call(
                model,
                (p_dict, buffers),
                (x_single.unsqueeze(0), bei_adj_1, bei_od_1, bei_od_t_1),
            )
            return loss_fn(logits, y_single.unsqueeze(0))

        grad_and_value_fn = torch_func.grad_and_value(single_loss)

        try:
            for start in range(0, bsz, micro_bsz):
                end = min(start + micro_bsz, bsz)
                x_chunk = bx[start:end].to(device, non_blocking=True)
                y_chunk = by[start:end].to(device, non_blocking=True)
                per_sample_grads, per_sample_losses = torch_func.vmap(
                    grad_and_value_fn,
                    in_dims=(None, 0, 0),
                )(param_values, x_chunk, y_chunk)

                total_loss += float(per_sample_losses.detach().sum().item())

                sq_norm = torch.zeros((x_chunk.size(0),), device=device)
                for g in per_sample_grads:
                    sq_norm += g.reshape(g.size(0), -1).pow(2).sum(dim=1)
                grad_norm = torch.sqrt(sq_norm + 1e-12)
                clip_coef = (max_grad_norm / (grad_norm + 1e-12)).clamp(max=1.0)

                for j, g in enumerate(per_sample_grads):
                    view_shape = (g.size(0),) + (1,) * (g.ndim - 1)
                    accum_grads[j].add_((g * clip_coef.view(view_shape)).sum(dim=0))
        except RuntimeError as e:
            model._dp_disable_vmap = True
            print(f"    [dp] vmap path failed ({type(e).__name__}); falling back to per-sample loop.")
            can_use_vmap = False

    if not can_use_vmap:
        bx_dev = bx.to(device, non_blocking=True)
        by_dev = by.to(device, non_blocking=True)
        bei_adj_1 = batch_edge_index(ei_adj, n_nodes, 1)
        bei_od_1 = batch_edge_index(ei_od, n_nodes, 1)
        bei_od_t_1 = batch_edge_index(ei_od_t, n_nodes, 1)

        for i in range(bsz):
            optimizer.zero_grad(set_to_none=True)

            xi = bx_dev[i : i + 1]
            yi = by_dev[i : i + 1]

            logits = model(xi, bei_adj_1, bei_od_1, bei_od_t_1)
            loss = loss_fn(logits, yi)
            total_loss += float(loss.item())
            loss.backward()

            sq_norm = torch.zeros((), device=device)
            for p in params:
                if p.grad is not None:
                    sq_norm += p.grad.detach().pow(2).sum()
            grad_norm = torch.sqrt(sq_norm + 1e-12)
            clip_coef = (max_grad_norm / (grad_norm + 1e-12)).clamp(max=1.0)

            for j, p in enumerate(params):
                if p.grad is not None:
                    accum_grads[j].add_(p.grad.detach() * clip_coef)

    optimizer.zero_grad(set_to_none=True)
    noise_std = noise_multiplier * max_grad_norm

    for j, p in enumerate(params):
        noise = torch.randn_like(accum_grads[j]) * noise_std
        p.grad = (accum_grads[j] + noise) / float(bsz)

    optimizer.step()

    return total_loss / float(bsz)


def train_and_evaluate_dp(
    model: StackedGRUTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    n_nodes: int,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    class_threshold: float,
    max_grad_norm: float,
    noise_multiplier: float,
    dp_microbatch_size: int,
    try_vmap: bool,
    save_path: Optional[str] = None,
) -> Tuple[Dict[str, float], float]:
    """Train with manual DP-SGD and evaluate best checkpoint on test set."""
    model = model.to(device)
    ei_adj = edge_index_adj.to(device)
    ei_od = edge_index_od.to(device)
    ei_od_t = edge_index_od_t.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    es = EarlyStopping(patience=patience, mode="max")

    best_f1, best_state = -1.0, None
    t0 = time.time()
    n_train_batches = len(train_loader)

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        print(f"    [train] epoch {epoch:3d}/{epochs} started  (batches={n_train_batches})")
        model.train()

        if getattr(model, "_dp_disable_vmap", False) and hasattr(model, "encoder") and hasattr(model.encoder, "gru"):
            try:
                model.encoder.gru.flatten_parameters()
            except Exception:
                pass

        ep_losses: List[float] = []

        for bx, by in train_loader:
            loss = _train_step_dp(
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                bx=bx,
                by=by,
                ei_adj=ei_adj,
                ei_od=ei_od,
                ei_od_t=ei_od_t,
                n_nodes=n_nodes,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                device=device,
                dp_microbatch_size=dp_microbatch_size,
                try_vmap=try_vmap,
            )
            ep_losses.append(loss)

        model.eval()
        vp, vt = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(device)
                bsz = bx.size(0)
                bei_adj = batch_edge_index(ei_adj, n_nodes, bsz)
                bei_od = batch_edge_index(ei_od, n_nodes, bsz)
                bei_od_t = batch_edge_index(ei_od_t, n_nodes, bsz)
                logits = model(bx, bei_adj, bei_od, bei_od_t)
                vp.append(torch.sigmoid(logits).cpu())
                vt.append(by)

        vm = classification_metrics_per_channel(
            torch.cat(vp).numpy(),
            torch.cat(vt).numpy(),
            threshold=class_threshold,
        )

        if vm["f1"] > best_f1:
            best_f1 = vm["f1"]
            best_state = {k: v.cpu().detach().clone() for k, v in model.state_dict().items()}
            if save_path is not None:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": best_state,
                        "val_f1": best_f1,
                        "classifier_name": model.classifier_name,
                        "gru_dim": model.gru_dim,
                        "gat_hidden": model.gat_hidden,
                        "noise_multiplier": noise_multiplier,
                        "max_grad_norm": max_grad_norm,
                    },
                    save_path,
                )

        print(
            f"    [train] epoch {epoch:3d}/{epochs} done  "
            f"loss={np.mean(ep_losses):.4f}  val_f1={vm['f1']:.4f}  "
            f"sec={time.time() - ep_t0:.1f}"
        )

        if es(vm["f1"], epoch):
            print(f"    early stop @ epoch {epoch}  best_val_f1={best_f1:.4f}")
            break

    train_sec = time.time() - t0

    if save_path is not None and best_state is not None:
        ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
        ckpt["train_sec"] = train_sec
        torch.save(ckpt, save_path)
        print(f"  Saved best model -> {save_path}")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    tp_list, tt_list = [], []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            bsz = bx.size(0)
            bei_adj = batch_edge_index(ei_adj, n_nodes, bsz)
            bei_od = batch_edge_index(ei_od, n_nodes, bsz)
            bei_od_t = batch_edge_index(ei_od_t, n_nodes, bsz)
            logits = model(bx, bei_adj, bei_od, bei_od_t)
            tp_list.append(torch.sigmoid(logits).cpu())
            tt_list.append(by)

    metrics = classification_metrics_per_channel(
        torch.cat(tp_list).numpy(),
        torch.cat(tt_list).numpy(),
        threshold=class_threshold,
    )
    return metrics, train_sec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Opacus-free fixed-epsilon DP training for Stacked GRUAttention -> Transformer"
    )

    p.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    p.add_argument("--weather_file", type=str, default="weather_cn.npy")
    p.add_argument("--period_hours", type=int, default=24)

    p.add_argument("--seq_len", type=int, default=18)
    p.add_argument("--horizons", type=int, nargs="+", default=[12])
    p.add_argument("--delay_threshold", type=float, default=5.0)
    p.add_argument("--class_threshold", type=float, default=0.5)

    p.add_argument("--gru_dim", type=int, default=64)
    p.add_argument("--gru_layers", type=int, default=2)
    p.add_argument("--gru_heads", type=int, default=4)
    p.add_argument("--gat_hidden", type=int, default=64)
    p.add_argument("--gat_heads", type=int, default=2)
    p.add_argument("--classifier", type=str, default="TSiTPlus", choices=["TSiTPlus", "ConvTranPlus"])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--chunk_size", type=int, default=200)

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=8)

    p.add_argument("--epsilon", type=float, default=3.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument(
        "--dp_microbatch_size",
        type=int,
        default=32,
        help="Microbatch size for vectorized per-sample DP gradients; <=0 uses full batch.",
    )
    p.add_argument(
        "--try_vmap",
        action="store_true",
        help="Try torch.func vmap per-sample gradients (may fail for some GRU/checkpoint stacks).",
    )
    p.add_argument(
        "--noise_multiplier",
        type=float,
        default=None,
        help="Optional manual sigma override; if unset, solved from epsilon.",
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--disable_activation_checkpoint",
        action="store_true",
        help="Disable model activation checkpointing (often faster and enables vmap path, but uses more GPU memory).",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=-1,
        help="DataLoader workers. -1=auto (Windows->0, else min(8, cpu_count)).",
    )
    p.add_argument("--output_dir", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    print(f"\n{'=' * 75}")
    print("  Opacus-free Fixed-Epsilon DP Training")
    print(f"  Model: GRUAttention -> multi-edge GAT -> {args.classifier}")
    print(f"  Device: {device}")
    print(f"{'=' * 75}\n")

    print("[1/5] Loading data ...")
    (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_inputs,
        val_inputs,
        test_inputs,
        train_delay_scaled,
        val_delay_scaled,
        test_delay_scaled,
        train_raw,
        val_raw,
        test_raw,
        _,
        num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

    train_inputs = train_inputs[:, :, :-2]
    val_inputs = val_inputs[:, :, :-2]
    test_inputs = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    max_horizon = sorted(set(args.horizons))[0]

    print("[2/5] Building sequences ...")
    train_x, _, train_y_cls = build_sequences_node_level(
        train_inputs,
        train_delay_scaled,
        train_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        args.horizons,
    )
    val_x, _, val_y_cls = build_sequences_node_level(
        val_inputs,
        val_delay_scaled,
        val_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        args.horizons,
    )
    test_x, _, test_y_cls = build_sequences_node_level(
        test_inputs,
        test_delay_scaled,
        test_raw,
        args.seq_len,
        max_horizon,
        args.delay_threshold,
        args.horizons,
    )

    trX, trY, vaX, vaY, teX, teY = build_graph_tensors(
        train_x,
        train_y_cls,
        val_x,
        val_y_cls,
        test_x,
        test_y_cls,
        args.seq_len,
        feature_dim,
    )
    n_nodes = trX.shape[1]

    trY = trY.float()
    vaY = vaY.float()
    teY = teY.float()

    if args.num_workers >= 0:
        worker_count = int(args.num_workers)
    else:
        worker_count = 0 if os.name == "nt" else min(8, os.cpu_count() or 0)

    if os.name == "nt" and worker_count > 0:
        print("  [loader] Windows detected: using worker processes may fail with low page file.")
        print("  [loader] If crashes occur, rerun with --num_workers 0.")

    loader_kwargs = dict(
        num_workers=worker_count,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(worker_count > 0),
    )
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        TensorDataset(trX, trY),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        TensorDataset(vaX, vaY),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        TensorDataset(teX, teY),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    dataset_size = len(train_loader.dataset)
    steps = max(1, args.epochs * len(train_loader))
    sample_rate = args.batch_size / max(1, dataset_size)

    if args.noise_multiplier is None:
        sigma = solve_noise_multiplier_for_epsilon(
            target_epsilon=args.epsilon,
            delta=args.delta,
            sample_rate=sample_rate,
            steps=steps,
        )
    else:
        sigma = float(args.noise_multiplier)

    achieved_eps = epsilon_upper_bound_approx(
        noise_multiplier=sigma,
        sample_rate=sample_rate,
        steps=steps,
        delta=args.delta,
    )

    print("[3/5] Privacy setup")
    print(f"  target epsilon={args.epsilon:.4f}, delta={args.delta:.1e}")
    print(f"  sample_rate={sample_rate:.6f}, steps={steps}")
    print(f"  using noise_multiplier(sigma)={sigma:.6f}")
    print(f"  approx epsilon upper bound at end of training={achieved_eps:.4f}")
    print(f"  dp_microbatch_size={args.dp_microbatch_size}")

    pos_rate = trY.reshape(-1, delay_dim).mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    print("[4/5] Building model and training ...")
    model = StackedGRUTransformer(
        c_in=feature_dim,
        c_out=delay_dim,
        seq_len=args.seq_len,
        gru_dim=args.gru_dim,
        gru_layers=args.gru_layers,
        gru_heads=args.gru_heads,
        gat_hidden=args.gat_hidden,
        gat_heads=args.gat_heads,
        classifier_name=args.classifier,
        dropout=args.dropout,
        chunk_size=args.chunk_size,
    )

    model._dp_disable_vmap = not args.try_vmap
    if not args.try_vmap:
        print("  [dp] using stable per-sample DP path (vmap disabled by default).")

    if args.disable_activation_checkpoint:
        def _encode_chunk_no_ckpt(self, chunk: torch.Tensor) -> torch.Tensor:
            return self.encoder(chunk)

        def _classify_chunk_no_ckpt(self, chunk: torch.Tensor) -> torch.Tensor:
            out = self.classifier(chunk)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if out.dim() > 2:
                out = out.flatten(1)
            return out

        model._encode_chunk = types.MethodType(_encode_chunk_no_ckpt, model)
        model._classify_chunk = types.MethodType(_classify_chunk_no_ckpt, model)
        print("  [perf] activation checkpoint disabled for faster training.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir if args.output_dir != "auto" else f"stacked_gru_dp_fixed_eps_{ts}"
    os.makedirs(out_dir, exist_ok=True)

    save_path = os.path.join(out_dir, f"Stacked_GRUAttn_{args.classifier}_DP_best.pth")

    try:
        metrics, train_sec = train_and_evaluate_dp(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            edge_index_adj=edge_index_adj,
            edge_index_od=edge_index_od,
            edge_index_od_t=edge_index_od_t,
            n_nodes=n_nodes,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            pos_weight=pos_weight,
            patience=args.patience,
            class_threshold=args.class_threshold,
            max_grad_norm=args.max_grad_norm,
            noise_multiplier=sigma,
            dp_microbatch_size=args.dp_microbatch_size,
            try_vmap=args.try_vmap,
            save_path=save_path,
        )
    except Exception:
        tb = traceback.format_exc()
        err_path = os.path.join(out_dir, "dp_training_error.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(tb)
        raise RuntimeError(f"Training failed. See: {err_path}")

    print("[5/5] Writing metrics ...")
    metrics_path = os.path.join(out_dir, "dp_metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["epsilon_target", f"{args.epsilon:.6f}"])
        w.writerow(["epsilon_approx_upper", f"{achieved_eps:.6f}"])
        w.writerow(["delta", f"{args.delta:.12f}"])
        w.writerow(["noise_multiplier", f"{sigma:.6f}"])
        w.writerow(["max_grad_norm", f"{args.max_grad_norm:.6f}"])
        w.writerow(["train_sec", f"{train_sec:.2f}"])
        for k, v in metrics.items():
            w.writerow([k, f"{float(v):.6f}"])

    print(f"\nDone. Output dir: {out_dir}")
    print(f"Checkpoint: {save_path}")
    print(f"Metrics:    {metrics_path}")


if __name__ == "__main__":
    main()
