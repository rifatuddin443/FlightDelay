"""Sequential two-stage KAN-GAT tester.

Loads the saved checkpoint from `classifykat.py` training and evaluates
3-, 6-, and 12-step ahead arrival/departure delay metrics, mirroring the
multi-horizon reporting style used by `kan_gcn_test.py`.
"""

import argparse
import os
import sys
import csv
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

# Local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
from baseline_methods import test_error  # noqa: E402
from classifykat import SequentialTwoStagePredictor, load_flight_data  # noqa: E402


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_sequences(
    scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Create sliding-window sequences for the specified horizon."""
    num_nodes = scaled.shape[0]
    max_idx = scaled.shape[1] - seq_len - horizon
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for t in range(max_idx):
        x_seq = scaled[:, t:t + seq_len, :].reshape(num_nodes, -1)
        y_seq = raw[:, t + seq_len:t + seq_len + horizon, :]
        y_seq = np.nan_to_num(y_seq)
        y_seq = y_seq.reshape(num_nodes, horizon, -1)

        x_list.append(x_seq)
        y_list.append(y_seq)

    if not x_list:
        return torch.empty(0), np.empty((0, num_nodes, horizon, scaled.shape[2]))

    x_tensor = torch.tensor(np.stack(x_list), dtype=torch.float32)
    y_array = np.stack(y_list)
    return x_tensor, y_array


def evaluate_horizon(
    model: SequentialTwoStagePredictor,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    device: torch.device,
    test_scaled: np.ndarray,
    test_raw: np.ndarray,
    scaler,
    seq_len: int,
    horizon: int,
    class_threshold: float,
) -> Dict[str, float]:
    base_features = test_scaled.shape[2]
    test_x, test_y = prepare_sequences(test_scaled, test_raw, seq_len, horizon)
    if len(test_x) == 0:
        print(f"Skipping horizon {horizon}: insufficient timesteps")
        return {}

    all_preds = []

    model.eval()
    with torch.no_grad():
        for idx in range(len(test_x)):
            current_input = test_x[idx].to(device)
            step_preds = []

            for step in range(horizon):
                data = Data(
                    x=current_input,
                    edge_index_adj=edge_index_adj,
                    edge_index_od=edge_index_od,
                    edge_index_od_t=edge_index_od_t,
                )
                logits, reg = model(data)
                probs = torch.sigmoid(logits)
                mask = (probs >= class_threshold).float()
                gated_reg = reg * mask
                step_preds.append(gated_reg.cpu().numpy())

                if step < horizon - 1:
                    current_input = torch.cat([
                        current_input[:, base_features:],
                        gated_reg,
                    ], dim=1)

            # Stack predictions into (num_nodes, horizon, features)
            seq_pred = np.stack(step_preds, axis=0)  # (horizon, num_nodes, features)
            seq_pred = np.transpose(seq_pred, (1, 0, 2))
            all_preds.append(seq_pred)

    all_preds = np.array(all_preds)
    preds_denorm = scaler.inverse_transform(
        all_preds.reshape(-1, base_features)
    ).reshape(all_preds.shape)
    targets_denorm = test_y  # already in original scale

    arrival_preds = preds_denorm[:, :, horizon - 1, 0]
    arrival_targets = targets_denorm[:, :, horizon - 1, 0]
    dep_preds = preds_denorm[:, :, horizon - 1, 1]
    dep_targets = targets_denorm[:, :, horizon - 1, 1]

    arr_mae, arr_rmse, arr_r2 = test_error(arrival_preds, arrival_targets)
    dep_mae, dep_rmse, dep_r2 = test_error(dep_preds, dep_targets)

    print(
        f"{horizon}-step ARRIVAL  -> MAE: {arr_mae:.4f}, RMSE: {arr_rmse:.4f}, R²: {arr_r2:.4f}"
    )
    print(
        f"{horizon}-step DEPARTURE -> MAE: {dep_mae:.4f}, RMSE: {dep_rmse:.4f}, R²: {dep_r2:.4f}"
    )

    return {
        'arr_mae': arr_mae,
        'arr_rmse': arr_rmse,
        'arr_r2': arr_r2,
        'dep_mae': dep_mae,
        'dep_rmse': dep_rmse,
        'dep_r2': dep_r2,
    }


def print_results_table(results: Dict[int, Dict[str, float]]):
    if not results:
        print("No horizons evaluated.")
        return
    print("\n" + "=" * 80)
    print("SUMMARY TABLE - Sequential Two-Stage Multi-Horizon Test")
    print("=" * 80)
    print(f"{'Horizon':<10} {'Delay Type':<15} {'MAE':<12} {'RMSE':<12} {'R²':<12}")
    print("-" * 80)
    for horizon in sorted(results.keys()):
        res = results[horizon]
        print(
            f"{horizon}-step    {'Arrival':<15} {res['arr_mae']:<12.4f} "
            f"{res['arr_rmse']:<12.4f} {res['arr_r2']:<12.4f}"
        )
        print(
            f"{'':10} {'Departure':<15} {res['dep_mae']:<12.4f} "
            f"{res['dep_rmse']:<12.4f} {res['dep_r2']:<12.4f}"
        )
    print("=" * 80)


def save_results(results: Dict[int, Dict[str, float]], path: str):
    if not results:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'horizon', 'arr_mae', 'arr_rmse', 'arr_r2',
            'dep_mae', 'dep_rmse', 'dep_r2'
        ])
        for horizon in sorted(results.keys()):
            res = results[horizon]
            writer.writerow([
                horizon,
                res['arr_mae'], res['arr_rmse'], res['arr_r2'],
                res['dep_mae'], res['dep_rmse'], res['dep_r2'],
            ])


def main():
    parser = argparse.ArgumentParser(description='Sequential two-stage tester (multi-horizon)')
    parser.add_argument('--data_dir', type=str, default='cdata')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='kan_gat_sequential_best.pth')
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 6, 12])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_scaled,
        val_scaled,
        test_scaled,
        train_raw,
        val_raw,
        test_raw,
        scaler,
        num_nodes,
    ) = load_flight_data(args.data_dir)

    edge_index_adj = edge_index_adj.to(device)
    edge_index_od = edge_index_od.to(device)
    edge_index_od_t = edge_index_od_t.to(device)

    base_features = train_scaled.shape[2]
    in_channels = args.seq_len * base_features
    out_channels = base_features

    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32,
    ).to(device)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.model_path}. Please run classifykat.py first."
        )

    checkpoint = torch.load(args.model_path, map_location=device)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])
    model.regressor.load_state_dict(checkpoint['regressor'])
    print(f"Loaded checkpoint from {args.model_path}")

    print("\n" + "=" * 80)
    print("MULTI-HORIZON TESTING")
    print("=" * 80)

    results = {}
    for horizon in args.horizons:
        res = evaluate_horizon(
            model,
            edge_index_adj,
            edge_index_od,
            edge_index_od_t,
            device,
            test_scaled,
            test_raw,
            scaler,
            args.seq_len,
            horizon,
            args.class_threshold,
        )
        if res:
            results[horizon] = res

    print_results_table(results)
    save_results(results, 'kan_gat_sequential_test_multihorizon.csv')
    print("\n✓ Testing complete. Metrics saved to kan_gat_sequential_test_multihorizon.csv")


if __name__ == '__main__':
    main()
