"""Quick architecture comparison using a small subset of data.

Fast test on 1000 train + 200 val samples with 10 epochs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))

from benchmark_architectures import *

def quick_test():
    """Run fast benchmark on subset."""
    args = argparse.Namespace(
        data_source='udata',
        seq_len=8,
        horizons=[3, 6, 12],
        delay_threshold=5.0,
        use_node_level=True,
        weather_file='weather2016_2021.npy',
        hidden_channels=64,
        epochs=10,
        lr=0.001,
        seed=42,
        quick_test=True,
        full_test=False,
    )
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load data
    print("Loading data...")
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=24,
        data_source=args.data_source,
    )
    
    # Build sequences
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, args.horizons[0], args.delay_threshold, args.horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, args.horizons[0], args.delay_threshold, args.horizons
    )
    
    # SUBSAMPLE for speed
    n_train = 2000
    n_val = 400
    train_x = train_x[:n_train]
    train_y_cls = train_y_cls[:n_train]
    val_x = val_x[:n_val]
    val_y_cls = val_y_cls[:n_val]
    
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    feature_dim = train_inputs.shape[2]
    in_channels = args.seq_len * feature_dim
    hidden_channels = args.hidden_channels
    
    print(f"Quick test: {len(train_x)} train, {len(val_x)} val samples")
    print(f"Class balance: {train_y_cls.mean().item():.2%} delayed\n")
    
    # Define architectures to test
    architectures = {
        '1_Baseline': lambda: BaselineEncoder(in_channels, hidden_channels),
        '2_Attention': lambda: AttentionFusionEncoder(in_channels, hidden_channels),
        '3_Residual': lambda: ResidualEncoder(in_channels, hidden_channels),
        '4_Transformer': lambda: TransformerEncoder(in_channels, hidden_channels),
        '5_Hybrid': lambda: HybridEncoder(in_channels, hidden_channels),
    }
    
    results = {}
    
    print("="*80)
    print("QUICK ARCHITECTURE COMPARISON")
    print("="*80)
    
    for name, encoder_fn in architectures.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        encoder = encoder_fn()
        model = UniversalClassifier(encoder, hidden_channels)
        
        metrics = train_and_evaluate(
            model, train_x, train_y_cls, val_x, val_y_cls,
            edge_indices, device, args.epochs, args.lr
        )
        
        elapsed = time.time() - start_time
        
        # Ensure metrics is a dictionary with all required keys
        if not isinstance(metrics, dict):
            print(f"  WARNING: metrics is not a dict, got {type(metrics)}")
            metrics = {'f1': 0.0, 'precision': 0.0, 'recall': 0.0, 'accuracy': 0.0}
        
        results[name] = {
            'f1': metrics.get('f1', 0.0),
            'precision': metrics.get('precision', 0.0),
            'recall': metrics.get('recall', 0.0),
            'accuracy': metrics.get('accuracy', 0.0),
            'time_seconds': elapsed,
            'time_per_epoch': elapsed / args.epochs,
        }
        
        print(f"\n[DONE] {name} completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"  F1: {metrics.get('f1', 0.0):.4f} | Precision: {metrics.get('precision', 0.0):.4f} | "
              f"Recall: {metrics.get('recall', 0.0):.4f}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\n{'Architecture':<20} {'F1 Score':<12} {'Precision':<12} {'Recall':<12} {'Time (s)':<12}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1'], reverse=True)
    
    for name, metrics in sorted_results:
        print(f"{name:<20} {metrics['f1']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['time_seconds']:<12.1f}")
    
    best_arch = sorted_results[0][0]
    best_f1 = sorted_results[0][1]['f1']
    
    print("\n" + "="*80)
    print(f"WINNER: {best_arch} with F1 = {best_f1:.4f}")
    print("="*80)
    
    # Save results
    import json
    with open(f'quick_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to quick_test_results.json")


if __name__ == '__main__':
    quick_test()
