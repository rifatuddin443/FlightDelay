"""Compare one-stage regression models against CNN deep-dual-reg 3-stage model.

This script:
1) Trains CNN with full 3-stage pipeline (classification + delayed reg + non-delayed reg).
2) Trains other models with ONE-STAGE regression only.
3) Compares regression metrics in one results-table CSV:
   - overall MAE/RMSE
   - delayed MAE/RMSE
   - non-delayed MAE/RMSE
   - per-channel (arrival/departure) delayed/non-delayed/overall
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

import compare_three_stage_models as base
from classifykat import load_flight_data, set_seed
from classifykat_balanced import build_sequences, build_sequences_node_level


def _compute_regression_summary(
    reg_preds: np.ndarray,
    reg_targets: np.ndarray,
    scaler,
    delay_threshold: float,
) -> Dict[str, float]:
    if scaler is not None:
        preds_denorm = scaler.inverse_transform(reg_preds)
        targets_denorm = scaler.inverse_transform(reg_targets)
    else:
        preds_denorm = reg_preds
        targets_denorm = reg_targets

    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)

    preds_flat = preds_denorm.flatten()
    targets_flat = targets_denorm.flatten()

    delayed_mask = targets_flat > delay_threshold
    nondelayed_mask = targets_flat <= delay_threshold

    def _mae_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        if y_true.size == 0:
            return 0.0, 0.0
        mae = float(np.mean(np.abs(y_pred - y_true)))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        return mae, rmse

    mae_delayed, rmse_delayed = _mae_rmse(targets_flat[delayed_mask], preds_flat[delayed_mask])
    mae_nondelayed, rmse_nondelayed = _mae_rmse(targets_flat[nondelayed_mask], preds_flat[nondelayed_mask])
    mae_overall, rmse_overall = _mae_rmse(targets_flat, preds_flat)

    summary: Dict[str, float] = {
        "regression_mae_overall": mae_overall,
        "regression_rmse_overall": rmse_overall,
        "regression_mae_delayed": mae_delayed,
        "regression_rmse_delayed": rmse_delayed,
        "regression_mae_nondelayed": mae_nondelayed,
        "regression_rmse_nondelayed": rmse_nondelayed,
        "num_delayed_samples": int(delayed_mask.sum()),
        "num_nondelayed_samples": int(nondelayed_mask.sum()),
    }

    if targets_denorm.ndim == 2 and targets_denorm.shape[1] >= 2:
        arr_targets = targets_denorm[:, 0].reshape(-1)
        dep_targets = targets_denorm[:, 1].reshape(-1)
        arr_preds = preds_denorm[:, 0].reshape(-1)
        dep_preds = preds_denorm[:, 1].reshape(-1)

        def _by_mask(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> Tuple[float, float, int]:
            if int(mask.sum()) == 0:
                return 0.0, 0.0, 0
            mae = float(np.mean(np.abs(y_pred[mask] - y_true[mask])))
            rmse = float(np.sqrt(np.mean((y_pred[mask] - y_true[mask]) ** 2)))
            return mae, rmse, int(mask.sum())

        arr_delayed = arr_targets > delay_threshold
        dep_delayed = dep_targets > delay_threshold
        arr_nd = ~arr_delayed
        dep_nd = ~dep_delayed

        arr_mae_d, arr_rmse_d, arr_n_d = _by_mask(arr_targets, arr_preds, arr_delayed)
        dep_mae_d, dep_rmse_d, dep_n_d = _by_mask(dep_targets, dep_preds, dep_delayed)
        arr_mae_nd, arr_rmse_nd, arr_n_nd = _by_mask(arr_targets, arr_preds, arr_nd)
        dep_mae_nd, dep_rmse_nd, dep_n_nd = _by_mask(dep_targets, dep_preds, dep_nd)

        arr_mae_all = float(np.mean(np.abs(arr_preds - arr_targets)))
        arr_rmse_all = float(np.sqrt(np.mean((arr_preds - arr_targets) ** 2)))
        dep_mae_all = float(np.mean(np.abs(dep_preds - dep_targets)))
        dep_rmse_all = float(np.sqrt(np.mean((dep_preds - dep_targets) ** 2)))

        summary.update(
            {
                "regression_mae_delayed_arrival": arr_mae_d,
                "regression_rmse_delayed_arrival": arr_rmse_d,
                "num_delayed_samples_arrival": int(arr_n_d),
                "regression_mae_delayed_departure": dep_mae_d,
                "regression_rmse_delayed_departure": dep_rmse_d,
                "num_delayed_samples_departure": int(dep_n_d),
                "regression_mae_nondelayed_arrival": arr_mae_nd,
                "regression_rmse_nondelayed_arrival": arr_rmse_nd,
                "num_nondelayed_samples_arrival": int(arr_n_nd),
                "regression_mae_nondelayed_departure": dep_mae_nd,
                "regression_rmse_nondelayed_departure": dep_rmse_nd,
                "num_nondelayed_samples_departure": int(dep_n_nd),
                "regression_mae_overall_arrival": arr_mae_all,
                "regression_rmse_overall_arrival": arr_rmse_all,
                "regression_mae_overall_departure": dep_mae_all,
                "regression_rmse_overall_departure": dep_rmse_all,
            }
        )

    return summary


def _train_one_stage_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int,
    model_name: str,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=2.0)
    early_stopping = base.EarlyStopping(patience=patience, mode="min")

    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses: List[float] = []
        for batch_x, batch_y_reg in train_loader:
            batch_x = batch_x.to(device)
            batch_y_reg = batch_y_reg.to(device)
            optimizer.zero_grad(set_to_none=True)
            hidden, _ = model.forward_classifier(batch_x)
            preds = model.forward_regressor(hidden, which="delayed")
            loss = loss_fn(preds, batch_y_reg)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses: List[float] = []
        with torch.no_grad():
            for batch_x, batch_y_reg in val_loader:
                batch_x = batch_x.to(device)
                batch_y_reg = batch_y_reg.to(device)
                hidden, _ = model.forward_classifier(batch_x)
                preds = model.forward_regressor(hidden, which="delayed")
                val_losses.append(float(loss_fn(preds, batch_y_reg).item()))

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == epochs:
            print(f"[{model_name}] 1-stage reg epoch {epoch}/{epochs} | val_loss={val_loss:.4f}")

        if early_stopping(val_loss, epoch):
            print(f"[{model_name}] early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)


def _predict_one_stage(
    model: nn.Module,
    test_x: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    preds_list: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in DataLoader(TensorDataset(test_x), batch_size=batch_size, shuffle=False):
            batch_x = batch_x.to(device)
            hidden, _ = model.forward_classifier(batch_x)
            preds = model.forward_regressor(hidden, which="delayed")
            preds_list.append(preds.cpu().numpy())
    return np.concatenate(preds_list, axis=0)


def _write_comparison_results(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    fields = [
        "model",
        "mode",
        "regression_mae_overall",
        "regression_rmse_overall",
        "regression_mae_delayed",
        "regression_rmse_delayed",
        "regression_mae_nondelayed",
        "regression_rmse_nondelayed",
        "regression_mae_overall_arrival",
        "regression_mae_overall_departure",
        "regression_mae_delayed_arrival",
        "regression_mae_delayed_departure",
        "regression_mae_nondelayed_arrival",
        "regression_mae_nondelayed_departure",
        "num_delayed_samples",
        "num_nondelayed_samples",
    ]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _write_combined_markdown(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return

    sorted_rows = sorted(rows, key=lambda r: r.get("regression_mae_overall", float("inf")))

    headers = [
        "Rank",
        "Model",
        "Mode",
        "MAE Overall",
        "MAE Delayed",
        "MAE Non-delayed",
        "RMSE Overall",
        "RMSE Delayed",
        "RMSE Non-delayed",
        "MAE Arr",
        "MAE Dep",
    ]

    lines: List[str] = []
    lines.append("# Combined Regression Results")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for idx, row in enumerate(sorted_rows, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(idx),
                    str(row.get("model", "")),
                    str(row.get("mode", "")),
                    f"{float(row.get('regression_mae_overall', float('nan'))):.4f}",
                    f"{float(row.get('regression_mae_delayed', float('nan'))):.4f}",
                    f"{float(row.get('regression_mae_nondelayed', float('nan'))):.4f}",
                    f"{float(row.get('regression_rmse_overall', float('nan'))):.4f}",
                    f"{float(row.get('regression_rmse_delayed', float('nan'))):.4f}",
                    f"{float(row.get('regression_rmse_nondelayed', float('nan'))):.4f}",
                    f"{float(row.get('regression_mae_overall_arrival', float('nan'))):.4f}",
                    f"{float(row.get('regression_mae_overall_departure', float('nan'))):.4f}",
                ]
            )
            + " |"
        )

    if sorted_rows:
        best = sorted_rows[0]
        lines.append("")
        lines.append(
            f"Best by overall MAE: {best.get('model', '')} ({best.get('mode', '')}) "
            f"with MAE={float(best.get('regression_mae_overall', float('nan'))):.4f}"
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _plot_combined_results(output_dir: str, rows: List[Dict[str, float]]) -> List[str]:
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda r: r.get("regression_mae_overall", float("inf")))
    labels = [str(r.get("model", "")) for r in sorted_rows]

    def _vals(key: str) -> List[float]:
        return [float(r.get(key, np.nan)) for r in sorted_rows]

    created: List[str] = []

    # 1) Overall MAE
    plt.figure(figsize=(10, 5))
    plt.bar(labels, _vals("regression_mae_overall"), color="#4C78A8")
    plt.title("Overall MAE Comparison")
    plt.ylabel("MAE (minutes)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    p1 = os.path.join(output_dir, "plot_mae_overall.png")
    plt.savefig(p1, dpi=180)
    plt.close()
    created.append(p1)

    # 2) Delayed vs non-delayed MAE
    x = np.arange(len(labels))
    width = 0.38
    delayed = np.array(_vals("regression_mae_delayed"), dtype=float)
    nondelayed = np.array(_vals("regression_mae_nondelayed"), dtype=float)
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, delayed, width, label="Delayed MAE")
    plt.bar(x + width / 2, nondelayed, width, label="Non-delayed MAE")
    plt.title("Delayed vs Non-delayed MAE")
    plt.ylabel("MAE (minutes)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    p2 = os.path.join(output_dir, "plot_mae_delayed_vs_nondelayed.png")
    plt.savefig(p2, dpi=180)
    plt.close()
    created.append(p2)

    # 3) Arrival vs departure overall MAE
    arr = np.array(_vals("regression_mae_overall_arrival"), dtype=float)
    dep = np.array(_vals("regression_mae_overall_departure"), dtype=float)
    plt.figure(figsize=(11, 5))
    plt.bar(x - width / 2, arr, width, label="Arrival MAE")
    plt.bar(x + width / 2, dep, width, label="Departure MAE")
    plt.title("Arrival vs Departure Overall MAE")
    plt.ylabel("MAE (minutes)")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    p3 = os.path.join(output_dir, "plot_mae_arrival_vs_departure.png")
    plt.savefig(p3, dpi=180)
    plt.close()
    created.append(p3)

    # 4) Overall RMSE
    plt.figure(figsize=(10, 5))
    plt.bar(labels, _vals("regression_rmse_overall"), color="#F58518")
    plt.title("Overall RMSE Comparison")
    plt.ylabel("RMSE (minutes)")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    p4 = os.path.join(output_dir, "plot_rmse_overall.png")
    plt.savefig(p4, dpi=180)
    plt.close()
    created.append(p4)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare 1-stage regression vs CNN 3-stage")
    parser.add_argument("--data_source", type=str, default="cdata", choices=["cdata", "udata"])
    parser.add_argument("--seq_len", type=int, default=18)
    parser.add_argument("--horizons", type=int, nargs=1, default=[12], choices=[3, 6, 12, 24])
    parser.add_argument("--delay_threshold", type=float, default=5.0)
    parser.add_argument("--class_threshold", type=float, default=0.5)
    parser.add_argument("--use_node_level", action="store_true", default=True)
    parser.add_argument("--exclude_time_features", action="store_true", default=True)
    parser.add_argument("--weather_file", type=str, default="weather_cn.npy")
    parser.add_argument("--period_hours", type=int, default=24)
    parser.add_argument("--stage1_epochs", type=int, default=10)
    parser.add_argument("--stage2_epochs", type=int, default=10)
    parser.add_argument("--stage3_epochs", type=int, default=14)
    parser.add_argument("--one_stage_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_channels", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["lstm", "gru", "bilstm"],
        choices=["lstm", "gru", "bilstm", "cnnlstm", "attnlstm", "stpn"],
    )
    parser.add_argument("--output_dir", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.data_source == "udata":
        args.weather_file = "weather2016_2021.npy"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    (
        _, _, _,
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

    if args.exclude_time_features:
        train_inputs = train_inputs[:, :, :-2]
        val_inputs = val_inputs[:, :, :-2]
        test_inputs = test_inputs[:, :, :-2]

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]

    horizons = sorted({h for h in args.horizons if h > 0})
    if len(horizons) != 1:
        raise ValueError("Use exactly one horizon with --horizons")

    build_fn = build_sequences_node_level if args.use_node_level else build_sequences

    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, horizons[0], args.delay_threshold, horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, horizons[0], args.delay_threshold, horizons
    )
    test_x, test_y_reg, test_y_cls = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, horizons[0], args.delay_threshold, horizons
    )

    # node-level flattened tensors
    train_x_flat = train_x.reshape(-1, args.seq_len, feature_dim)
    val_x_flat = val_x.reshape(-1, args.seq_len, feature_dim)
    test_x_flat = test_x.reshape(-1, args.seq_len, feature_dim)
    train_y_reg_flat = train_y_reg.reshape(-1, delay_dim)
    val_y_reg_flat = val_y_reg.reshape(-1, delay_dim)
    test_y_reg_flat = test_y_reg.reshape(-1, delay_dim)
    train_y_cls_flat = train_y_cls.reshape(-1, delay_dim)
    val_y_cls_flat = val_y_cls.reshape(-1, delay_dim)
    test_y_cls_flat = test_y_cls.reshape(-1, delay_dim)

    train_loader_3s = DataLoader(
        TensorDataset(train_x_flat, train_y_cls_flat, train_y_reg_flat),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader_3s = DataLoader(
        TensorDataset(val_x_flat, val_y_cls_flat, val_y_reg_flat),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    train_loader_1s = DataLoader(
        TensorDataset(train_x_flat, train_y_reg_flat),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader_1s = DataLoader(
        TensorDataset(val_x_flat, val_y_reg_flat),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir == "auto":
        output_dir = f"reg1stage_vs_cnn3stage_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    rows: List[Dict[str, float]] = []

    # CNN deep dual-reg 3-stage
    print("\n[REFERENCE] CNN deepDualReg 3-stage")
    cnn_module = base._load_cnn_module()
    cnn_base = cnn_module.SequentialTwoStagePredictor(
        in_channels=args.seq_len * feature_dim,
        out_channels=delay_dim,
        hidden_channels=args.hidden_channels,
        regressor_extra_layer=True,
        seq_len=args.seq_len,
    ).to(device)
    cnn = base.CNNAdapter(cnn_base).to(device)

    pos_rate = train_y_cls_flat.float().mean(dim=0)
    pos_weight = (1.0 - pos_rate + 1e-6) / (pos_rate + 1e-6)

    base._train_stage1_node(cnn, train_loader_3s, val_loader_3s, device, args.stage1_epochs, args.lr, pos_weight, args.patience, "CNN3Stage")
    base._train_stage2_node(cnn, train_loader_3s, val_loader_3s, device, args.stage2_epochs, args.lr, args.delay_threshold, scaler, args.patience, "CNN3Stage")
    base._train_stage3_node(cnn, train_loader_3s, val_loader_3s, device, args.stage3_epochs, args.lr * 0.01, args.delay_threshold, scaler, args.patience, "CNN3Stage")

    _, _, cnn_reg_preds, cnn_reg_targets = base._evaluate_model(
        cnn,
        test_x_flat,
        test_y_cls_flat,
        test_y_reg_flat,
        device,
        args.class_threshold,
        scaler,
    )
    cnn_summary = _compute_regression_summary(cnn_reg_preds, cnn_reg_targets, scaler, args.delay_threshold)
    cnn_summary["model"] = "CNN_deepDualReg_3stage"
    cnn_summary["mode"] = "3stage"
    rows.append(cnn_summary)

    # one-stage models
    model_builders = {
        "lstm": lambda: base.RecurrentDualModel("lstm", feature_dim, 64, delay_dim).to(device),
        "gru": lambda: base.RecurrentDualModel("gru", feature_dim, 64, delay_dim).to(device),
        "bilstm": lambda: base.BiLSTMDualModel(feature_dim, 64, delay_dim).to(device),
        "cnnlstm": lambda: base.CNNLSTMDualModel(feature_dim, 64, delay_dim).to(device),
        "attnlstm": lambda: base.AttentionLSTMDualModel(feature_dim, 64, delay_dim).to(device),
    }

    for name in args.models:
        if name == "stpn":
            print("\n[ONE-STAGE] STPN")
            if not base.STPN_AVAILABLE:
                print("[STPN] not available, skipped")
                continue

            train_x_stpn = train_x.reshape(train_x.shape[0], num_nodes, args.seq_len, feature_dim)
            val_x_stpn = val_x.reshape(val_x.shape[0], num_nodes, args.seq_len, feature_dim)
            test_x_stpn = test_x.reshape(test_x.shape[0], num_nodes, args.seq_len, feature_dim)

            # adjacency from dataset dir (same policy as base file)
            root_dir = os.path.dirname(__file__)
            candidates = [
                os.path.join(args.data_source, "dist_mx.npy"),
                os.path.join(args.data_source, "adj_mx.npy"),
                os.path.join(root_dir, args.data_source, "dist_mx.npy"),
                os.path.join(root_dir, args.data_source, "adj_mx.npy"),
            ]
            adj_file = next((p for p in candidates if os.path.exists(p)), None)
            if adj_file is None:
                adj_mx = np.eye(num_nodes)
            else:
                adj_mx = np.load(adj_file)
            adj_norm = adj_mx + np.eye(adj_mx.shape[0])
            row_sum = adj_norm.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            adj_norm = adj_norm / row_sum
            supports = [
                torch.FloatTensor(adj_norm).to(device),
                torch.FloatTensor(adj_norm @ adj_norm).to(device),
                torch.FloatTensor(adj_norm @ adj_norm @ adj_norm).to(device),
            ]

            model = base.STPNDualModel(supports, args.seq_len, len(horizons), feature_dim, delay_dim).to(device)
            train_loader = DataLoader(TensorDataset(train_x_stpn, train_y_reg), batch_size=max(4, min(16, args.batch_size // 8)), shuffle=True)
            val_loader = DataLoader(TensorDataset(val_x_stpn, val_y_reg), batch_size=max(4, min(16, args.batch_size // 8)), shuffle=False)

            # one-stage reg training for STPN
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
            loss_fn = nn.HuberLoss(delta=2.0)
            es = base.EarlyStopping(patience=args.patience, mode="min")
            best_val = float("inf")
            best_state = None

            for epoch in range(1, args.one_stage_epochs + 1):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    hidden, _ = model.forward_classifier(batch_x)
                    preds = model.forward_regressor(hidden, which="delayed")
                    loss = loss_fn(preds, batch_y)
                    loss.backward()
                    optimizer.step()

                model.eval()
                vals = []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                        hidden, _ = model.forward_classifier(batch_x)
                        preds = model.forward_regressor(hidden, which="delayed")
                        vals.append(float(loss_fn(preds, batch_y).item()))
                v = float(np.mean(vals)) if vals else 0.0
                if v < best_val:
                    best_val = v
                    best_state = {k: vv.detach().cpu() for k, vv in model.state_dict().items()}
                if es(v, epoch):
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            preds = []
            model.eval()
            with torch.no_grad():
                for (batch_x,) in DataLoader(TensorDataset(test_x_stpn), batch_size=max(4, min(16, args.batch_size // 8)), shuffle=False):
                    batch_x = batch_x.to(device)
                    hidden, _ = model.forward_classifier(batch_x)
                    p = model.forward_regressor(hidden, which="delayed")
                    preds.append(p.cpu().numpy())
            reg_preds = np.concatenate(preds, axis=0).reshape(-1, delay_dim)
            reg_targets = test_y_reg.reshape(-1, delay_dim).cpu().numpy()

            summary = _compute_regression_summary(reg_preds, reg_targets, scaler, args.delay_threshold)
            summary["model"] = "STPN"
            summary["mode"] = "1stage_regression"
            rows.append(summary)
            continue

        print(f"\n[ONE-STAGE] {name.upper()}")
        model = model_builders[name]()
        _train_one_stage_regression(
            model,
            train_loader_1s,
            val_loader_1s,
            device,
            epochs=args.one_stage_epochs,
            lr=args.lr,
            patience=args.patience,
            model_name=name.upper(),
        )
        reg_preds = _predict_one_stage(model, test_x_flat, device, args.batch_size)
        reg_targets = test_y_reg_flat.cpu().numpy()

        summary = _compute_regression_summary(reg_preds, reg_targets, scaler, args.delay_threshold)
        summary["model"] = name.upper()
        summary["mode"] = "1stage_regression"
        rows.append(summary)

    # choose best by overall MAE
    ranked = sorted(rows, key=lambda r: r.get("regression_mae_overall", float("inf")))
    if ranked:
        print("\nBest model by overall MAE:")
        print(f"  {ranked[0]['model']} ({ranked[0]['mode']}) -> MAE={ranked[0]['regression_mae_overall']:.4f}")

    out_csv = os.path.join(output_dir, f"regression_comparison_results_table_{timestamp}.csv")
    _write_comparison_results(out_csv, rows)
    out_md = os.path.join(output_dir, f"regression_comparison_results_table_{timestamp}.md")
    _write_combined_markdown(out_md, rows)
    plot_files = _plot_combined_results(output_dir, rows)

    print(f"\n✓ Comparison table saved: {out_csv}")
    print(f"✓ Combined markdown table: {out_md}")
    if plot_files:
        print("✓ Plots saved:")
        for p in plot_files:
            print(f"  - {p}")


if __name__ == "__main__":
    main()
