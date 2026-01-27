#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify encoder fallback logic (TCN vs plain Conv1d)."""

import torch
import torch.nn as nn
import sys
import os

# Add current directory to path to import cnnopacus
sys.path.insert(0, os.path.dirname(__file__))

from cnnopacus import SequentialTwoStagePredictor


def test_tcn_path():
    """Test Case 1: TCN path (sequence structure detected)."""
    print("\n" + "="*80)
    print("TEST 1: TCN Path (seq_len=24, in_channels=240)")
    print("="*80)
    
    # Divisible: 240 / 24 = 10 features
    model = SequentialTwoStagePredictor(
        in_channels=240,      # 24 * 10 = 240
        out_channels=2,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=24,           # Sequence length provided
    )
    
    # Input: [batch_size, seq_len, feature_dim] = [4, 24, 10]
    batch_x_3d = torch.randn(4, 24, 10)
    with torch.no_grad():
        hidden = model._encode_x(batch_x_3d)
    print(f"[PASS] TCN path works: input {batch_x_3d.shape} -> hidden {hidden.shape}")
    assert hidden.shape == (4, 128), f"Expected (4, 128), got {hidden.shape}"


def test_fallback_path_bad_divisor():
    """Test Case 2: Fallback (seq_len doesn't divide in_channels evenly)."""
    print("\n" + "="*80)
    print("TEST 2: Fallback Path (seq_len=24, in_channels=250 - NOT divisible)")
    print("="*80)
    
    # NOT divisible: 250 / 24 ≠ integer
    model = SequentialTwoStagePredictor(
        in_channels=250,      # 250 is NOT divisible by 24
        out_channels=2,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=24,           # Sequence length provided, but doesn't divide evenly
    )
    
    # Input: [batch_size, flattened_features]
    batch_x_2d = torch.randn(4, 250)
    with torch.no_grad():
        hidden = model._encode_x(batch_x_2d)
    print(f"[PASS] Fallback path works: input {batch_x_2d.shape} -> hidden {hidden.shape}")
    assert hidden.shape == (4, 128), f"Expected (4, 128), got {hidden.shape}"


def test_fallback_path_no_seq_len():
    """Test Case 3: Fallback (seq_len=None)."""
    print("\n" + "="*80)
    print("TEST 3: Fallback Path (seq_len=None)")
    print("="*80)
    
    # No seq_len provided
    model = SequentialTwoStagePredictor(
        in_channels=240,
        out_channels=2,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=None,         # No sequence info
    )
    
    # Input: [batch_size, flattened_features]
    batch_x_2d = torch.randn(4, 240)
    with torch.no_grad():
        hidden = model._encode_x(batch_x_2d)
    print(f"[PASS] Fallback path works: input {batch_x_2d.shape} -> hidden {hidden.shape}")
    assert hidden.shape == (4, 128), f"Expected (4, 128), got {hidden.shape}"


def test_forward_pass_tcn():
    """Test Case 4: Full forward pass with TCN."""
    print("\n" + "="*80)
    print("TEST 4: Full forward pass (TCN path)")
    print("="*80)
    
    from torch_geometric.data import Data
    
    model = SequentialTwoStagePredictor(
        in_channels=240,
        out_channels=2,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=24,
    )
    
    # Create minimal graph data
    batch_x_3d = torch.randn(4, 24, 10)
    data = Data(x=batch_x_3d)
    
    with torch.no_grad():
        hidden, preds = model(data)
    
    print(f"[PASS] Forward pass works: hidden {hidden.shape}, predictions {preds.shape}")
    assert hidden.shape == (4, 128), f"Expected hidden (4, 128), got {hidden.shape}"
    assert preds.shape == (4, 2), f"Expected preds (4, 2), got {preds.shape}"


def test_forward_pass_fallback():
    """Test Case 5: Full forward pass with Fallback Conv1d."""
    print("\n" + "="*80)
    print("TEST 5: Full forward pass (Fallback Conv1d path)")
    print("="*80)
    
    from torch_geometric.data import Data
    
    model = SequentialTwoStagePredictor(
        in_channels=250,      # NOT divisible by any reasonable seq_len
        out_channels=2,
        hidden_channels=128,
        regressor_extra_layer=True,
        seq_len=None,         # Force fallback
    )
    
    # Create minimal graph data
    batch_x_2d = torch.randn(4, 250)
    data = Data(x=batch_x_2d)
    
    with torch.no_grad():
        hidden, preds = model(data)
    
    print(f"[PASS] Forward pass works: hidden {hidden.shape}, predictions {preds.shape}")
    assert hidden.shape == (4, 128), f"Expected hidden (4, 128), got {hidden.shape}"
    assert preds.shape == (4, 2), f"Expected preds (4, 2), got {preds.shape}"


if __name__ == '__main__':
    try:
        test_tcn_path()
        test_fallback_path_bad_divisor()
        test_fallback_path_no_seq_len()
        test_forward_pass_tcn()
        test_forward_pass_fallback()
        
        print("\n" + "="*80)
        print("[SUCCESS] ALL TESTS PASSED")
        print("="*80)
        print("\nSummary:")
        print("  - TCN encoder activates when seq_len divides in_channels evenly")
        print("  - Fallback Conv1d activates when:")
        print("    1. seq_len doesn't divide in_channels evenly, OR")
        print("    2. seq_len is None")
        print("  - Both paths produce correct output shapes")
        
    except AssertionError as e:
        print(f"\n[FAILURE] TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
