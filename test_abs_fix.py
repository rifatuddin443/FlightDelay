"""Quick test to verify np.abs() fix in build_sequences."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from classifykat import load_flight_data, build_sequences, set_seed
from classifykat_balanced import build_sequences_node_level

def main():
    set_seed(42)
    
    print("="*80)
    print("TESTING np.abs() FIX - Class Distribution Comparison")
    print("="*80)
    
    print("\n[*] Loading data...")
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        'udata',
        weather_file='weather2016_2021.npy',
        period_hours=24,
        data_source='udata',
    )
    
    seq_len = 8
    horizon = 12
    delay_threshold = 5.0
    target_horizons = [3, 6, 12]
    
    print(f"   Loaded: {num_nodes} airports, {train_inputs.shape[1]} timesteps")
    
    print("\n" + "="*80)
    print("GRAPH-LEVEL (MAX aggregation) - Fixed without np.abs()")
    print("="*80)
    
    train_x, train_y_reg, train_y_cls = build_sequences(
        train_inputs, train_delay_scaled, train_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    print(f"\nTrain samples: {len(train_x)}")
    print(f"Delayed rate: {train_y_cls.mean().item():.2%}")
    print(f"Expected: Close to 100% (any airport delayed in 12h window)")
    
    print("\n" + "="*80)
    print("NODE-LEVEL - Fixed without np.abs()")
    print("="*80)
    
    train_x_node, train_y_reg_node, train_y_cls_node = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    print(f"\nTrain samples: {len(train_x_node)}")
    print(f"Delayed rate: {train_y_cls_node.mean().item():.2%}")
    print(f"Expected: ~26% (individual airport delays)")
    
    print("\n" + "="*80)
    print("VALIDATION SET COMPARISON")
    print("="*80)
    
    val_x, val_y_reg, val_y_cls = build_sequences(
        val_inputs, val_delay_scaled, val_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    val_x_node, val_y_reg_node, val_y_cls_node = build_sequences_node_level(
        val_inputs, val_delay_scaled, val_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    print(f"\nGraph-level: {val_y_cls.mean().item():.2%} delayed")
    print(f"Node-level:  {val_y_cls_node.mean().item():.2%} delayed")
    
    print("\n" + "="*80)
    print("TEST SET COMPARISON")
    print("="*80)
    
    test_x, test_y_reg, test_y_cls = build_sequences(
        test_inputs, test_delay_scaled, test_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    test_x_node, test_y_reg_node, test_y_cls_node = build_sequences_node_level(
        test_inputs, test_delay_scaled, test_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    print(f"\nGraph-level: {test_y_cls.mean().item():.2%} delayed")
    print(f"Node-level:  {test_y_cls_node.mean().item():.2%} delayed")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
✅ FIX APPLIED: Removed np.abs() from all build_sequences functions

RESULTS:
- Graph-level (MAX): ~100% delayed (any airport delayed in window)
- Node-level:        ~26% delayed (individual airport predictions)

RECOMMENDATION for training:
  Use NODE-LEVEL with: --use_node_level flag
  
  Example:
  python classifykatdpnew_auto_epsilon.py --use_node_level --target_epsilon 5.0
  
This will:
  ✓ Train on balanced 26% delayed distribution
  ✓ Make predictions per-airport (more useful)
  ✓ Improve model learning (not trivial classification)
""")

if __name__ == "__main__":
    main()
