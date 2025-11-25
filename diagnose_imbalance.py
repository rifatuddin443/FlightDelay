"""Diagnostic script to understand the severe class imbalance."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from classifykat import load_flight_data, set_seed

def main():
    set_seed(42)
    
    print("="*80)
    print("FLIGHT DELAY DATASET DIAGNOSTIC")
    print("="*80)
    
    delay_threshold = 5.0
    
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
    
    print(f"   Loaded: {num_nodes} airports")
    print(f"   Train: {train_raw.shape[1]} timesteps")
    print(f"   Val:   {val_raw.shape[1]} timesteps")
    print(f"   Test:  {test_raw.shape[1]} timesteps")
    
    print("\n" + "="*80)
    print("RAW DELAY STATISTICS (Before Windowing)")
    print("="*80)
    
    for split_name, raw_data in [("TRAIN", train_raw), ("VAL", val_raw), ("TEST", test_raw)]:
        print(f"\n{split_name} Split:")
        print(f"  Shape: {raw_data.shape} (nodes, timesteps, features)")
        
        # Overall statistics
        raw_flat = raw_data.flatten()
        raw_flat = raw_flat[~np.isnan(raw_flat)]
        
        print(f"  Mean delay: {raw_flat.mean():.2f} min")
        print(f"  Std delay:  {raw_flat.std():.2f} min")
        print(f"  Min delay:  {raw_flat.min():.2f} min")
        print(f"  Max delay:  {raw_flat.max():.2f} min")
        print(f"  Median:     {np.median(raw_flat):.2f} min")
        
        # Classification statistics
        pct_delayed = (raw_flat >= delay_threshold).mean()  # Only positive delays
        print(f"  Samples >= {delay_threshold} min: {pct_delayed:.2%}")
        
        # Per-timestep analysis
        max_delays_per_time = np.max(raw_data, axis=(0, 2))  # Max across nodes and features (no abs!)
        max_delays_per_time = max_delays_per_time[~np.isnan(max_delays_per_time)]
        
        ontime_timesteps = (max_delays_per_time < delay_threshold).sum()
        total_timesteps = len(max_delays_per_time)
        
        print(f"  Timesteps with ALL airports on-time: {ontime_timesteps}/{total_timesteps} ({ontime_timesteps/total_timesteps:.2%})")
        
        # Percentiles
        print(f"  Percentiles (absolute value for reference):")
        for p in [10, 25, 50, 75, 90, 95, 99]:
            val = np.percentile(np.abs(raw_flat), p)
            print(f"    {p}th: {val:.2f} min")
    
    print("\n" + "="*80)
    print("SLIDING WINDOW IMPACT ANALYSIS")
    print("="*80)
    
    seq_len = 6
    horizon = 12
    
    print(f"\nWindow config: seq_len={seq_len}, horizon={horizon}")
    print(f"Each sample looks at {seq_len} past hours + {horizon} future hours")
    
    # Simulate window creation
    max_idx = train_raw.shape[1] - seq_len - horizon
    print(f"\nTotal possible windows in TRAIN: {max_idx}")
    
    ontime_windows = 0
    delayed_windows = 0
    
    # Sample 1000 random windows to avoid memory issues
    sample_size = min(1000, max_idx)
    sample_indices = np.random.choice(max_idx, sample_size, replace=False)
    
    print(f"Sampling {sample_size} random windows...")
    
    for t in sample_indices:
        # Get future window
        future = train_raw[:, t + seq_len:t + seq_len + horizon, :]
        future_clean = np.nan_to_num(future, nan=0.0)  # Handle NaN values
        max_delay_in_window = np.max(future_clean)  # Only positive delays count
        
        if max_delay_in_window >= delay_threshold:
            delayed_windows += 1
        else:
            ontime_windows += 1
    
    print(f"\nResults from {sample_size} sampled windows:")
    print(f"  On-time windows: {ontime_windows} ({ontime_windows/sample_size:.2%})")
    print(f"  Delayed windows: {delayed_windows} ({delayed_windows/sample_size:.2%})")
    
    print("\n" + "="*80)
    print("DIAGNOSIS AND RECOMMENDATIONS")
    print("="*80)
    
    if ontime_windows == 0:
        print("\n[CRITICAL] NO ON-TIME WINDOWS FOUND!")
        print("\nThis means:")
        print("  - In EVERY 12-hour prediction window, at least ONE airport has delay >= 5 min")
        print("  - This is not a data processing bug - your data is genuinely this skewed")
        print("  - Classification becomes trivial: always predict 'delayed' = 97% accuracy")
        
        print("\n[SOLUTIONS]")
        print("\n1. INCREASE delay_threshold (easiest):")
        print("   - Try threshold = 10, 15, or 20 minutes")
        print("   - Most airlines consider 15+ minutes as 'significant' delay")
        print("   - Command: Change delay_threshold in your training script")
        
        print("\n2. SWITCH TO REGRESSION ONLY:")
        print("   - Skip classification task entirely")
        print("   - Focus on predicting exact delay minutes (regression)")
        print("   - More useful for operations anyway")
        
        print("\n3. USE NODE-LEVEL PREDICTION:")
        print("   - Predict delay for EACH airport independently")
        print("   - Not graph-level (any airport delayed)")
        print("   - See: classifykat_balanced.py::build_sequences_node_level()")
        
        print("\n4. SHORTEN prediction horizon:")
        print("   - Instead of 12 hours ahead, try 3 or 6 hours")
        print("   - Shorter horizon = more on-time samples")
        print("   - Command: Change horizon=3 or 6")
        
        print("\n5. ACCEPT THE IMBALANCE:")
        print("   - Use techniques from classifykat_advanced_balance.py")
        print("   - Focus on maximizing recall with threshold tuning")
        print("   - Combine with high pos_weight")
    
    else:
        print(f"\n[INFO] Found {ontime_windows/sample_size:.1%} on-time windows")
        print("This is still very imbalanced but workable.")
        print("\nRecommended techniques:")
        print("  1. Use build_sequences_class_balanced() with target_ratio=0.3-0.5")
        print("  2. Apply Focal Loss or Class-Balanced Loss")
        print("  3. Tune classification threshold on validation set")

if __name__ == "__main__":
    main()
