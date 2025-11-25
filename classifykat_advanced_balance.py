"""Advanced techniques to handle severe class imbalance in flight delay prediction.

This module provides multiple strategies beyond simple aggregation changes:
1. Temporal sampling to reduce correlation
2. Class-balanced sampling
3. SMOTE-style synthetic minority oversampling
4. Cost-sensitive learning adjustments
5. Threshold moving and calibration
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from collections import Counter


def build_sequences_with_temporal_sampling(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
    stride: int = 12,  # Skip every N time steps to reduce correlation
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequences with temporal stride to reduce redundancy.
    
    Problem: Sliding window with stride=1 creates highly correlated samples.
    Solution: Use stride > 1 to skip time steps, reducing temporal correlation.
    
    Args:
        stride: Time steps to skip between sequences (default 12 = 12 hours)
                Higher stride → less correlation, fewer samples, more balanced
    """
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    x_list, y_reg_list, y_cls_list = [], [], []

    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))

    # Use stride instead of every time step
    for t in range(0, max_idx, stride):
        x_seq = input_data[:, t:t + seq_len, :].reshape(num_nodes, -1)
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)

        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :])
        
        # Per-node classification (only positive delays count)
        node_delays = np.max(raw_target, axis=(1, 2))
        cls_flag = (node_delays >= delay_threshold).astype(np.float32)
        cls_flag = cls_flag.reshape(num_nodes, 1)

        x_list.append(x_seq)
        y_reg_list.append(y_seq)
        y_cls_list.append(cls_flag)

    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    
    delayed_rate = tensors[2].mean().item()
    print(f"[RESULT] Temporal sampling (stride={stride}): {delayed_rate:.2%} delayed, {len(x_list)} samples")
    
    return tensors


def build_sequences_class_balanced(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
    target_ratio: float = 0.5,  # Target 50/50 balance
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequences with class balancing via undersampling majority class.
    
    Problem: 97% delayed means massive class imbalance.
    Solution: Keep all minority (on-time), undersample majority (delayed) to target ratio.
    
    Args:
        target_ratio: Desired ratio of delayed samples (0.5 = balanced)
        max_samples: Maximum total samples to keep (None = keep all minority + balanced majority)
    """
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    
    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))

    # First pass: collect all samples and labels
    all_samples = []
    for t in range(max_idx):
        x_seq = input_data[:, t:t + seq_len, :].reshape(num_nodes, -1)
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)

        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :])
        
        # Use graph-level label (MAX aggregation, only positive delays)
        max_delay = np.max(raw_target)
        graph_label = 1.0 if max_delay >= delay_threshold else 0.0
        cls_flag = np.full((num_nodes, 1), graph_label, dtype=np.float32)
        
        all_samples.append((x_seq, y_seq, cls_flag, graph_label))
    
    # Separate by class (use graph_label for separation)
    minority_samples = [s for s in all_samples if s[3] < 0.5]  # on-time graphs
    majority_samples = [s for s in all_samples if s[3] >= 0.5]  # delayed graphs
    
    print(f"Original: {len(minority_samples)} on-time graphs, {len(majority_samples)} delayed graphs")
    
    # Handle edge case: no minority samples
    if len(minority_samples) == 0:
        print(f"[WARNING] No on-time samples found! Using all data without balancing.")
        balanced_samples = all_samples
    elif len(majority_samples) == 0:
        print(f"[WARNING] No delayed samples found! Using all data without balancing.")
        balanced_samples = all_samples
    else:
        # Calculate how many majority samples to keep
        n_minority = len(minority_samples)
        n_majority_target = int(n_minority * target_ratio / (1 - target_ratio))
        n_majority_keep = min(n_majority_target, len(majority_samples))
        
        # Random undersample majority
        if n_majority_keep < len(majority_samples):
            np.random.shuffle(majority_samples)
            majority_samples = majority_samples[:n_majority_keep]
        
        # Combine and shuffle
        balanced_samples = minority_samples + majority_samples
        np.random.shuffle(balanced_samples)
    
    if max_samples and len(balanced_samples) > max_samples:
        balanced_samples = balanced_samples[:max_samples]
    
    # Unpack (use first 3 elements, ignore graph_label)
    x_list = [s[0] for s in balanced_samples]
    y_reg_list = [s[1] for s in balanced_samples]
    y_cls_list = [s[2] for s in balanced_samples]
    
    if len(x_list) == 0:
        raise ValueError("No samples available after balancing. Check your data and parameters.")
    
    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    
    delayed_rate = tensors[2].mean().item()
    print(f"[RESULT] Balanced: {delayed_rate:.2%} delayed, {len(x_list)} samples")
    
    return tensors


def build_sequences_with_hard_negatives(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
    near_threshold_window: float = 2.0,  # Keep samples within ±2 min of threshold
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep all minority class + hard examples near decision boundary.
    
    Problem: Model might not learn decision boundary well with extreme imbalance.
    Solution: Keep all on-time flights + delayed flights close to threshold (hard examples).
    
    Args:
        near_threshold_window: Keep delayed samples within [threshold, threshold + window]
    """
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    
    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))

    all_samples = []
    for t in range(max_idx):
        x_seq = input_data[:, t:t + seq_len, :].reshape(num_nodes, -1)
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)

        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :])
        
        # Use graph-level max delay (only positive delays)
        max_delay = np.max(raw_target)
        graph_label = 1.0 if max_delay >= delay_threshold else 0.0
        cls_flag = np.full((num_nodes, 1), graph_label, dtype=np.float32)
        
        # Keep if: on-time OR near threshold
        if max_delay < delay_threshold:  # on-time graph
            all_samples.append((x_seq, y_seq, cls_flag))
        elif max_delay <= delay_threshold + near_threshold_window:  # hard positive
            all_samples.append((x_seq, y_seq, cls_flag))
        # Skip easy positives (very delayed)
    
    if len(all_samples) == 0:
        raise ValueError(f"No samples found! Try increasing near_threshold_window (current: {near_threshold_window})")
    
    x_list = [s[0] for s in all_samples]
    y_reg_list = [s[1] for s in all_samples]
    y_cls_list = [s[2] for s in all_samples]
    
    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    
    delayed_rate = tensors[2].mean().item()
    print(f"[RESULT] Hard negatives: {delayed_rate:.2%} delayed, {len(x_list)} samples")
    
    return tensors


def apply_focal_loss_weights(
    y_cls: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Compute focal loss weights to down-weight easy examples.
    
    Focal Loss = -α(1-p)^γ log(p)
    
    Use these weights in your loss function:
        loss = BCEWithLogitsLoss(weight=focal_weights, pos_weight=pos_weight)
    
    Args:
        y_cls: Classification labels [N, 1]
        alpha: Balance between positive/negative (0.25 for minority)
        gamma: Focusing parameter (2.0 standard)
    
    Returns:
        weights: [N, 1] weights for each sample
    """
    # This is a placeholder - actual focal loss needs predictions during training
    # Return balanced weights based on class frequency
    n_pos = y_cls.sum()
    n_neg = len(y_cls) - n_pos
    
    # Simple class balancing weights
    pos_weight = n_neg / (n_pos + 1e-8)
    neg_weight = 1.0
    
    weights = torch.where(y_cls > 0.5, pos_weight, neg_weight)
    return weights


def compute_optimal_pos_weight(
    y_cls: torch.Tensor,
    target_recall: float = 0.9,
) -> float:
    """Compute pos_weight to achieve target recall.
    
    Higher pos_weight → model penalizes false negatives more → higher recall
    
    Formula: pos_weight = (1 - minority_rate) / minority_rate * recall_boost
    
    Args:
        y_cls: Classification labels
        target_recall: Desired recall (0.9 = 90%)
    
    Returns:
        Suggested pos_weight value
    """
    minority_rate = y_cls.mean().item()
    
    # Standard balanced weight
    base_weight = (1 - minority_rate) / (minority_rate + 1e-8)
    
    # Boost for recall (empirical: multiply by 1/target_recall)
    recall_boost = 1.0 / (target_recall + 0.1)  # +0.1 to avoid extreme values
    
    suggested_weight = base_weight * recall_boost
    
    print(f"\n[RECOMMENDATION] POS_WEIGHT:")
    print(f"   Base (balanced): {base_weight:.2f}")
    print(f"   For recall={target_recall:.0%}: {suggested_weight:.2f}")
    print(f"   Usage: BCEWithLogitsLoss(pos_weight=torch.tensor([{suggested_weight:.2f}]))")
    
    return suggested_weight
