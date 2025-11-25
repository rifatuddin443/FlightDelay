"""Test script to compare all advanced balancing techniques."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from classifykat import load_flight_data, build_sequences, set_seed
from classifykat_advanced_balance import (
    build_sequences_with_temporal_sampling,
    build_sequences_class_balanced,
    build_sequences_with_hard_negatives,
    compute_optimal_pos_weight,
)


def main():
    set_seed(42)
    
    print("="*80)
    print("ADVANCED CLASS BALANCING TECHNIQUES COMPARISON")
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
    
    print(f"   Loaded: {num_nodes} airports, {train_inputs.shape[1]} time steps")
    
    seq_len = 6
    horizon = 12
    target_horizons = [3, 6, 12]
    
    # Use training split for all tests
    input_train = train_inputs
    target_train = train_delay_scaled
    raw_train = train_raw
    
    print("\n" + "="*80)
    print("TECHNIQUE 1: TEMPORAL SAMPLING (Reduce Correlation)")
    print("="*80)
    print("Idea: Skip time steps to reduce redundant, correlated samples")
    
    for stride in [1, 6, 12, 24]:
        print(f"\n[>>] Stride = {stride} (every {stride} hours):")  
        x, y_reg, y_cls = build_sequences_with_temporal_sampling(
            input_train, target_train, raw_train,
            seq_len, horizon, delay_threshold, target_horizons,
            stride=stride
        )
    
    print("\n" + "="*80)
    print("TECHNIQUE 2: CLASS-BALANCED UNDERSAMPLING")
    print("="*80)
    print("Idea: Keep all minority, undersample majority to achieve balance")
    
    for target_ratio in [0.3, 0.4, 0.5]:
        print(f"\n[=] Target ratio = {target_ratio:.0%}:")   
        x, y_reg, y_cls = build_sequences_class_balanced(
            input_train, target_train, raw_train,
            seq_len, horizon, delay_threshold, target_horizons,
            target_ratio=target_ratio
        )
    
    print("\n" + "="*80)
    print("TECHNIQUE 3: HARD NEGATIVE MINING")
    print("="*80)
    print("Idea: Keep all on-time + delayed samples near threshold (hard examples)")
    
    for window in [2.0, 5.0, 10.0]:
        print(f"\n[+] Threshold window = ±{window} minutes:")  
        x, y_reg, y_cls = build_sequences_with_hard_negatives(
            input_train, target_train, raw_train,
            seq_len, horizon, delay_threshold, target_horizons,
            near_threshold_window=window
        )
    
    print("\n" + "="*80)
    print("TECHNIQUE 4: OPTIMAL POS_WEIGHT CALCULATION")
    print("="*80)
    print("Idea: Compute BCEWithLogitsLoss pos_weight to boost recall")
    
    # Use original unbalanced data
    _, _, y_cls_orig = build_sequences(
        input_train, target_train, raw_train,
        seq_len, horizon, delay_threshold, target_horizons
    )
    
    for target_recall in [0.7, 0.8, 0.9]:
        print(f"\n[%] Target recall = {target_recall:.0%}:")  
        pos_weight = compute_optimal_pos_weight(y_cls_orig, target_recall)
    
    print("\n" + "="*80)
    print("[!] RECOMMENDATIONS")  
    print("="*80)
    print("""
1️⃣  TEMPORAL SAMPLING (stride=12):
   • Reduces samples but maintains diversity
   • Use if you have enough data and want faster training
   • Might still have imbalance if data is inherently imbalanced

2️⃣  CLASS BALANCING (target_ratio=0.3-0.5):
   • Forces 30-50% minority class
   • Best for learning decision boundary
   • Loses some information by discarding majority samples
   • ⚠️ Adjust classification threshold on validation set afterward

3️⃣  HARD NEGATIVE MINING (window=5.0):
   • Keeps challenging examples near threshold
   • Good for fine-tuning decision boundary
   • Still discards easy positives

4️⃣  HIGH POS_WEIGHT (15-50):
   • Easiest to implement - just change one parameter
   • Keeps all data
   • Use with BCEWithLogitsLoss(pos_weight=torch.tensor([COMPUTED_VALUE]))
   • Combine with threshold tuning for best results

[*] BEST STRATEGY:
   1. Use TEMPORAL SAMPLING (stride=6-12) to reduce correlation
   2. Apply CLASS BALANCING (target_ratio=0.4) for better learning
   3. Train with HIGH POS_WEIGHT (from technique 4)
   4. Run threshold tuning on validation set (tune_threshold.py)
   5. Evaluate on test set with optimized threshold

[CODE] INTEGRATION:
   In your training script, replace build_sequences() with:
   
   from classifykat_advanced_balance import (
       build_sequences_with_temporal_sampling,
       build_sequences_class_balanced,
   )
   
   # Step 1: Temporal sampling
   x_train, y_reg_train, y_cls_train = build_sequences_with_temporal_sampling(
       input_train, target_train, raw_train,
       seq_len, horizon, delay_threshold, target_horizons,
       stride=12
   )
   
   # Step 2: Class balancing
   x_train, y_reg_train, y_cls_train = build_sequences_class_balanced(
       input_train, target_train, raw_train,
       seq_len, horizon, delay_threshold, target_horizons,
       target_ratio=0.4
   )
   
   # Step 3: Compute pos_weight
   pos_weight = compute_optimal_pos_weight(y_cls_train, target_recall=0.85)
   
   # Step 4: Use in loss
   criterion_cls = nn.BCEWithLogitsLoss(
       pos_weight=torch.tensor([pos_weight], device=device)
   )
""")


if __name__ == "__main__":
    main()
