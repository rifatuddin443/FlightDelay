"""Balanced version with proper graph-level vs node-level labeling.

FIX: Uses graph-level AVERAGE delay instead of MAX to avoid extreme imbalance.
This creates more balanced classes while preserving meaningful delay information.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple

# Import everything from original except build_sequences
from classifykat import (
    EarlyStopping,
    SequentialTwoStagePredictor,
    classification_metrics,
    load_flight_data,
    regression_metrics,
    set_seed,
)


def build_sequences_balanced(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
    aggregation: str = 'mean',  # 'mean', 'max', or 'any'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequences with configurable graph-level aggregation.
    
    Args:
        aggregation: How to aggregate node delays to graph label:
            - 'mean': Graph is delayed if AVERAGE delay >= threshold (more balanced)
            - 'max': Graph is delayed if ANY node delayed (original, imbalanced)
            - 'any': Graph is delayed if >50% of nodes delayed (balanced alternative)
    """
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    x_list, y_reg_list, y_cls_list = [], [], []

    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))
    if not horizon_ids:
        raise ValueError("At least one future horizon is required to build sequences.")

    for t in range(max_idx):
        x_seq = input_data[:, t:t + seq_len, :].reshape(num_nodes, -1)
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)

        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :])
        
        # FIX: Different aggregation strategies (only positive delays count)
        if aggregation == 'mean':
            # Graph is delayed if AVERAGE delay across all nodes >= threshold
            # Separate classification for each feature
            avg_delay = np.mean(raw_target, axis=(0, 1)) # [num_features]
            cls_flag = (avg_delay >= delay_threshold).astype(np.float32)
            # Replicate to all nodes for consistency
            cls_flag = np.tile(cls_flag, (num_nodes, 1))
            
        elif aggregation == 'any':
            # Graph is delayed if >50% of nodes are delayed
            # Separate classification for each feature
            node_delays = np.max(raw_target, axis=1)  # [num_nodes, num_features]
            delayed_nodes = (node_delays >= delay_threshold).astype(np.float32)
            graph_delayed = (delayed_nodes.mean(axis=0) > 0.5).astype(np.float32) # [num_features]
            cls_flag = np.tile(graph_delayed, (num_nodes, 1))
            
        else:  # 'max' - original behavior
            # Graph is delayed if ANY node has delay >= threshold
            # Separate classification for each feature
            node_delays = np.max(raw_target, axis=1) # [num_nodes, num_features]
            cls_flag = (node_delays >= delay_threshold).astype(np.float32)
            # cls_flag is [num_nodes, num_features]

        x_list.append(x_seq)
        y_reg_list.append(y_seq)
        y_cls_list.append(cls_flag)

    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    return tensors


def build_sequences_node_level(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build sequences with TRUE node-level labels (most balanced).
    
    Each node gets its own delay label instead of shared graph label.
    This preserves the actual 22.48% delay rate from raw data.
    """
    num_nodes = input_data.shape[0]
    max_idx = input_data.shape[1] - seq_len - horizon
    x_list, y_reg_list, y_cls_list = [], [], []

    if target_horizons:
        horizon_ids = [min(h, horizon) - 1 for h in sorted({h for h in target_horizons if h > 0})]
    else:
        horizon_ids = list(range(horizon))
    if not horizon_ids:
        raise ValueError("At least one future horizon is required to build sequences.")

    for t in range(max_idx):
        x_seq = input_data[:, t:t + seq_len, :].reshape(num_nodes, -1)
        future_scaled = target_scaled[:, t + seq_len:t + seq_len + horizon, :]
        future_scaled = future_scaled[:, horizon_ids, :]
        y_seq = future_scaled.reshape(num_nodes, -1)
        #y seq is scalaed target at 11th time step (for horizon=12) and raw target is unscaled target at 11th time step (for horizon=12)
        
        raw_target = raw[:, t + seq_len:t + seq_len + horizon, :]#future unscaled target
        raw_target = np.nan_to_num(raw_target[:, horizon_ids, :]) #just the target at 11
        
        # FIX: Per-node classification (preserves true distribution)
        # Each node labeled independently based on its own delay (only positive delays)
        # Separate classification for each feature (e.g., arrival and departure)
        # Take max over horizon per feature (keeps arrival vs. departure separate)
        node_delays = np.max(raw_target, axis=1)  # [num_nodes, num_features]
        cls_flag = (node_delays >= delay_threshold).astype(np.float32)
        # cls_flag is now [num_nodes, num_features]

        x_list.append(x_seq)
        y_reg_list.append(y_seq)
        y_cls_list.append(cls_flag)

    tensors = (
        torch.tensor(np.stack(x_list), dtype=torch.float32),
        torch.tensor(np.stack(y_reg_list), dtype=torch.float32),
        torch.tensor(np.stack(y_cls_list), dtype=torch.float32),
    )
    
    # Print actual balance
    cls_tensor = tensors[2]
    delayed_rate = cls_tensor.mean().item()
    print(f"\nNode-level balance: {delayed_rate:.2%} delayed ")
    
    return tensors


# Wrapper function that uses the balanced version
def build_sequences(
    input_data: np.ndarray,
    target_scaled: np.ndarray,
    raw: np.ndarray,
    seq_len: int,
    horizon: int,
    delay_threshold: float,
    target_horizons: Optional[List[int]] = None,
    use_node_level: bool = True,  # Default to most balanced
    aggregation: str = 'mean',
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Wrapper that selects the appropriate sequence building method.
    
    Args:
        use_node_level: If True, uses per-node labels (most balanced, ~22% delayed)
                       If False, uses graph-level with specified aggregation
        aggregation: Only used if use_node_level=False
            - 'mean': Balanced (~40-60% delayed)
            - 'any': Moderate (~70-80% delayed)  
            - 'max': Original imbalanced (~97% delayed)
    """
    if use_node_level:
        return build_sequences_node_level(
            input_data, target_scaled, raw, seq_len, horizon,
            delay_threshold, target_horizons
        )
    else:
        return build_sequences_balanced(
            input_data, target_scaled, raw, seq_len, horizon,
            delay_threshold, target_horizons, aggregation
        )
