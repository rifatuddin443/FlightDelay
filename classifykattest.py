"""Sequential two-stage KAN-GAT tester with weather + temporal embeddings."""

import argparse
import csv
import os
import sys
from typing import Dict

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


def evaluate_horizon(
    model: SequentialTwoStagePredictor,
    edge_index_adj: torch.Tensor,
    edge_index_od: torch.Tensor,
    edge_index_od_t: torch.Tensor,
    device: torch.device,
    test_inputs: np.ndarray,
    test_raw: np.ndarray,
    scaler,
    seq_len: int,
    horizon: int,
    class_threshold: float,
    delay_dim: int,
) -> Dict[str, float]:
    num_nodes, total_steps, feature_dim = test_inputs.shape
    aux_dim = feature_dim - delay_dim
    max_idx = total_steps - seq_len - horizon
    if max_idx <= 0:
        print(f"Skipping horizon {horizon}: insufficient timesteps")
        return {}

    inputs_tensor = torch.tensor(test_inputs, dtype=torch.float32, device=device)
    test_raw = np.nan_to_num(test_raw)

    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for start_idx in range(max_idx):
            current_input = inputs_tensor[:, start_idx:start_idx + seq_len, :].reshape(num_nodes, -1)
            future_aux = inputs_tensor[:, start_idx + seq_len:start_idx + seq_len + horizon, delay_dim:]
            target_seq = test_raw[:, start_idx + seq_len:start_idx + seq_len + horizon, :]

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
                    if aux_dim > 0:
                        aux_vector = future_aux[:, step, :]
                        new_features = torch.cat([gated_reg, aux_vector], dim=1)
                    else:
                        new_features = gated_reg
                    current_input = torch.cat([
                        current_input[:, feature_dim:],
                        new_features,
                    ], dim=1)

            seq_pred = np.stack(step_preds, axis=0).transpose(1, 0, 2)
            all_preds.append(seq_pred)
            all_targets.append(target_seq)

    all_preds = np.array(all_preds)
    preds_denorm = scaler.inverse_transform(
        all_preds.reshape(-1, delay_dim)
    ).reshape(all_preds.shape)
    targets_denorm = np.array(all_targets)

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
            f"{horizon}-step    {'Arrival':<15} {res['arr_mae']:<12.4f} {res['arr_rmse']:<12.4f} {res['arr_r2']:<12.4f}"
        )
        print(
            f"{'':10} {'Departure':<15} {res['dep_mae']:<12.4f} {res['dep_rmse']:<12.4f} {res['dep_r2']:<12.4f}"
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
    parser.add_argument('--data_source', type=str, default='udata', choices=['cdata', 'udata'],
                        help='Data source folder: cdata (China) or udata (USA)')
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--model_path', type=str, default='kan_gat_sequential_best.pth')
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--horizons', type=int, nargs='+', default=[3, 6, 12])
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_dir = args.data_source

    (
        edge_index_adj,
        edge_index_od,
        edge_index_od_t,
        train_inputs,
        _,
        test_inputs,
        train_delay_scaled,
        _,
        test_delay_scaled,
        train_raw,
        _,
        test_raw,
        scaler,
        num_nodes,
    ) = load_flight_data(
        data_dir,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )

    edge_index_adj = edge_index_adj.to(device)
    edge_index_od = edge_index_od.to(device)
    edge_index_od_t = edge_index_od_t.to(device)

    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]

    model = SequentialTwoStagePredictor(
        in_channels=args.seq_len * feature_dim,
        out_channels=delay_dim,
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
            test_inputs,
            test_raw,
            scaler,
            args.seq_len,
            horizon,
            args.class_threshold,
            delay_dim,
        )
        if res:
            results[horizon] = res

    print_results_table(results)
    save_results(results, 'kan_gat_sequential_test_multihorizon.csv')
    print("\n✓ Testing complete. Metrics saved to kan_gat_sequential_test_multihorizon.csv")


if __name__ == '__main__':
    main()
