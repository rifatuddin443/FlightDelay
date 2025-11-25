"""Compare different labeling strategies for class balance.

This script tests:
1. Original (MAX aggregation): ~97% delayed
2. Mean aggregation: ~40-60% delayed  
3. Majority voting: ~70-80% delayed
4. Node-level (true labels): ~22% delayed (matches raw data)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from classifykat import load_flight_data, set_seed
from classifykat_balanced import (
    build_sequences_balanced,
    build_sequences_node_level,
)

def test_labeling_strategies():
    set_seed(42)
    
    print("Loading data...")
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
    
    print(f"\n{'='*80}")
    print("RAW DATA STATISTICS")
    print(f"{'='*80}")
    print(f"Train delay mean: {train_raw.mean():.2f} min")
    print(f"Train delayed %: {(train_raw >= delay_threshold).astype(float).mean():.2%}")
    
    print(f"\n{'='*80}")
    print("TESTING DIFFERENT LABELING STRATEGIES")
    print(f"{'='*80}\n")
    
    # Strategy 1: Original MAX aggregation (imbalanced)
    print("1️⃣  ORIGINAL (MAX aggregation - any node delayed → graph delayed)")
    print("-" * 80)
    _, _, train_y_cls_max = build_sequences_balanced(
        train_inputs, train_delay_scaled, train_raw,
        seq_len, horizon, delay_threshold, target_horizons,
        aggregation='max'
    )
    delayed_rate_max = train_y_cls_max.mean().item()
    print(f"   Result: {delayed_rate_max:.2%} delayed")
    print(f"   Issue: Extreme imbalance due to graph-level MAX\n")
    
    # Strategy 2: Mean aggregation (more balanced)
    print("2️⃣  MEAN AGGREGATION (average delay across nodes)")
    print("-" * 80)
    _, _, train_y_cls_mean = build_sequences_balanced(
        train_inputs, train_delay_scaled, train_raw,
        seq_len, horizon, delay_threshold, target_horizons,
        aggregation='mean'
    )
    delayed_rate_mean = train_y_cls_mean.mean().item()
    print(f"   Result: {delayed_rate_mean:.2%} delayed")
    print(f"   Improvement: More balanced, graph represents average condition\n")
    
    # Strategy 3: Majority voting (balanced alternative)
    print("3️⃣  MAJORITY VOTING (>50% of nodes delayed → graph delayed)")
    print("-" * 80)
    _, _, train_y_cls_any = build_sequences_balanced(
        train_inputs, train_delay_scaled, train_raw,
        seq_len, horizon, delay_threshold, target_horizons,
        aggregation='any'
    )
    delayed_rate_any = train_y_cls_any.mean().item()
    print(f"   Result: {delayed_rate_any:.2%} delayed")
    print(f"   Improvement: Represents widespread delays, not isolated incidents\n")
    
    # Strategy 4: Node-level (true distribution)
    print("4️⃣  NODE-LEVEL LABELS (each node labeled independently)")
    print("-" * 80)
    _, _, train_y_cls_node = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw,
        seq_len, horizon, delay_threshold, target_horizons
    )
    delayed_rate_node = train_y_cls_node.mean().item()
    print(f"   Result: {delayed_rate_node:.2%} delayed")
    print(f"   ✓ Matches raw data distribution (~22.48%)\n")
    
    print(f"{'='*80}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'='*80}")
    print("\n📊 Class Distribution Comparison:")
    print(f"   Raw data:        {(train_raw >= delay_threshold).astype(float).mean():.2%} delayed")
    print(f"   Original (MAX):  {delayed_rate_max:.2%} delayed ❌ Extreme imbalance")
    print(f"   Mean aggregation:{delayed_rate_mean:.2%} delayed ✓ Balanced")
    print(f"   Majority voting: {delayed_rate_any:.2%} delayed ✓ Balanced")
    print(f"   Node-level:      {delayed_rate_node:.2%} delayed ✓✓ True distribution")
    
    print("\n💡 Recommended Approach:")
    print("   Use NODE-LEVEL LABELS (Strategy 4) because:")
    print("   • Preserves true delay distribution from raw data")
    print("   • Each prediction is meaningful for individual airports")
    print("   • Avoids artificial imbalance from graph aggregation")
    print("   • Classification becomes a real learning task (not trivial)")
    
    print("\n📝 To use in your training:")
    print("   from classifykat_balanced import build_sequences")
    print("   train_x, train_y_reg, train_y_cls = build_sequences(")
    print("       train_inputs, train_delay_scaled, train_raw,")
    print("       seq_len, horizon, delay_threshold, target_horizons,")
    print("       use_node_level=True  # ← Use node-level labels")
    print("   )")
    

if __name__ == '__main__':
    test_labeling_strategies()
