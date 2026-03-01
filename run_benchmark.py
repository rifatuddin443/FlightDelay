"""
Main Benchmark Runner Script

This script orchestrates the complete benchmark comparing all available models:
- Classical: Historical Average, VAR
- Deep Learning: LSTM, GRU, Transformer
- Graph-based: STPN, DSAFNet
- Your Custom Model (CNN-KAN)

Usage:
    # Full benchmark (all models, 50 epochs)
    python run_benchmark.py --data_dir cdata --epochs 50
    
    # Quick test (limited models, 10 epochs)
    python run_benchmark.py --quick_test
    
    # Skip certain model types
    python run_benchmark.py --skip_classical --skip_dl  # Only graph models
    python run_benchmark.py --skip_graph  # Only classical and DL baselines
    
    # Custom configuration
    python run_benchmark.py --epochs 30 --batch_size 64 --lr 0.0001
"""

import sys
import os
import argparse

# Add project path
sys.path.insert(0, os.path.dirname(__file__))

from comprehensive_benchmark import BenchmarkRunner, parse_args
from benchmark_graph_models import add_graph_models_to_benchmark
from benchmark_ml_models import add_ml_models_to_benchmark


def main():
    """Main benchmark execution."""
    
    # Parse arguments
    args = parse_args()
    
    print("\n" + "="*100)
    print(" " * 30 + "COMPREHENSIVE BENCHMARK SUITE")
    print("="*100)
    print(f"\nConfiguration:")
    print(f"  Data Directory: {args.data_dir}")
    print(f"  Sequence Length: {args.seq_length}")
    print(f"  Prediction Length: {args.pred_length}")
    print(f"  Training Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Test Ratio: {args.test_ratio}")
    print(f"\nModel Types:")
    print(f"  Classical (HA, VAR): {'Skipped' if args.skip_classical else 'Included'}")
    print(f"  Deep Learning (LSTM, GRU, Transformer): {'Skipped' if args.skip_dl else 'Included'}")
    print(f"  Graph-based (STPN, DSAFNet): {'Skipped' if args.skip_graph else 'Included'}")
    print("="*100 + "\n")
    
    # Create benchmark runner
    runner = BenchmarkRunner(args)
    
    # Load data
    runner.load_data()
    
    # Run baseline methods
    runner.run_all_baselines()
    
    # Add machine learning models (Random Forest, XGBoost, LightGBM, Bi-LSTM, etc.)
    runner = add_ml_models_to_benchmark(runner)
    
    # Add graph-based models (STPN, DSAFNet)
    runner = add_graph_models_to_benchmark(runner)
    
    # Visualize and save results
    runner.visualize_results()
    runner.save_results()
    
    print("\n" + "="*100)
    print(" " * 35 + "BENCHMARK COMPLETE!")
    print("="*100)
    print(f"\nResults saved to: {runner.output_dir}")
    print(f"  - CSV: benchmark_results_{runner.timestamp}.csv")
    print(f"  - JSON: benchmark_results_{runner.timestamp}.json")
    print(f"  - Plots: benchmark_comparison_{runner.timestamp}.png")
    print("\n" + "="*100 + "\n")
    
    # Print top-3 models by MAE
    print("\n🏆 Top 3 Models (by MAE):")
    print("-" * 60)
    sorted_results = sorted(
        runner.results.items(), 
        key=lambda x: x[1].get('mae', float('inf'))
    )[:3]
    for rank, (model_name, metrics) in enumerate(sorted_results, 1):
        print(f"  {rank}. {model_name:<20} MAE: {metrics.get('mae', 'N/A'):.4f}")
    print("-" * 60 + "\n")


if __name__ == '__main__':
    main()
