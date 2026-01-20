"""Properly implemented differentially private sequential KAN-GAT pipeline with epsilon budget control.

FIXED:
1. Ensures both predictions AND targets are at graph-level (not node-level).
2. Uses fixed noise multiplier for differential privacy (not auto-computed).
3. Allows training to complete all epochs while tracking epsilon.
4. NEW: Added Stage 3 for regressing delays on samples predicted as under threshold.
5. FIXED: Stage 3 now correctly masks based on actual delay values (< 5 min), not classification labels.
opacus 
"""

from __future__ import annotations

import argparse
import csv
import copy
import os
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from contextlib import nullcontext
from typing import Any, Dict, List, Sequence, Tuple, Optional, TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.data.data import BaseData
import glob

# Monkey-patch PyG's MessagePassing to disable static graph check.
# This is needed because Opacus's GradSampleModule triggers PyG's static graph
# detection (via torch._C._get_tracing_state()) even in hooks mode.
try:
    from torch_geometric.nn.conv.message_passing import MessagePassing
    MessagePassing._is_tracing = staticmethod(lambda: False)
except Exception:
    pass  # Older PyG versions may not have this method

# DP via Opacus (replaces custom DP implementation)
try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    from opacus.utils.batch_memory_manager import BatchMemoryManager
except Exception:
    PrivacyEngine = None  # type: ignore[assignment]
    ModuleValidator = None  # type: ignore[assignment]
    BatchMemoryManager = None  # type: ignore[assignment]

# Check if running in Colab for file downloads
try:
    from google.colab import files as colab_files  # type: ignore[import-not-found]
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    colab_files = None

if TYPE_CHECKING:
    from opacus import PrivacyEngine as PrivacyEngineType

# Reuse original implementation
sys.path.insert(0, os.path.dirname(__file__))
from classifykat import (  # noqa: E402
    EarlyStopping,
    SequentialTwoStagePredictor,
    build_sequences,
    classification_metrics,
    load_flight_data,
    regression_metrics,
    set_seed,
)
from classifykat_balanced import build_sequences_node_level  # noqa: E402
from baseline_methods import test_error  # noqa: E402


class ManualGATLayer(nn.Module):
    """Manual GAT implementation using plain tensors for Opacus compatibility."""
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        
        # Linear transformations
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # x: [N, in_channels], edge_index: [2, E]
        N = x.size(0)
        H = self.heads
        C = self.out_channels
        
        # Linear transformation
        x_src = self.lin_src(x).view(N, H, C)  # [N, H, C]
        x_dst = self.lin_dst(x).view(N, H, C)  # [N, H, C]
        
        # Compute attention scores
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)  # [N, H]
        
        # Get edge indices
        row, col = edge_index[0], edge_index[1]
        
        # Compute attention coefficients
        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)
        
        # Softmax per target node
        alpha = self.edge_softmax(alpha, col, N)
        
        # Apply attention and aggregate
        out = torch.zeros(N, H, C, device=x.device)
        x_src_edges = x_src[row]  # [E, H, C]
        
        for i in range(alpha.size(0)):
            out[col[i]] += alpha[i].unsqueeze(-1) * x_src_edges[i]
        
        # Reshape and add bias
        out = out.view(N, H * C) + self.bias
        return out
    
    def edge_softmax(self, scores: torch.Tensor, indices: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Softmax over edges grouped by target node."""
        # scores: [E, H], indices: [E]
        scores_max = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        scores_max.index_reduce_(0, indices, scores, 'amax', include_self=False)
        scores_max = scores_max[indices]
        
        scores_exp = torch.exp(scores - scores_max)
        scores_sum = torch.zeros(num_nodes, scores.size(1), device=scores.device)
        scores_sum.index_add_(0, indices, scores_exp)
        scores_sum = scores_sum[indices]
        
        return scores_exp / (scores_sum + 1e-16)


# Import visualization functions
try:
    from visualize_training_classification import (
        visualize_training_data,
        visualize_classification_results,
        visualize_regression_timeseries,
    )
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("Warning: visualize_training_classification not found. Visualizations will be skipped.")
    VISUALIZATION_AVAILABLE = False


class GraphSequenceData(Data):
    """Custom PyG data object with multiple edge indices."""
    def __inc__(self, key, value, *args, **kwargs):  # type: ignore[override]
        if key in {"edge_index_adj", "edge_index_od", "edge_index_od_t"}:
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)


class GraphSequenceDataset(Dataset):
    """Dataset wrapper for graph sequences."""
    def __init__(
        self,
        features: torch.Tensor,
        y_reg: Optional[torch.Tensor],
        y_cls: Optional[torch.Tensor],
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> None:
        self.features = features.clone()
        self.y_reg = y_reg.clone() if y_reg is not None else None
        self.y_cls = y_cls.clone() if y_cls is not None else None
        self.edge_index_adj = edge_index_adj.clone().long()
        self.edge_index_od = edge_index_od.clone().long()
        self.edge_index_od_t = edge_index_od_t.clone().long()

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> GraphSequenceData:
        data = GraphSequenceData()
        feat = self.features[idx]
        data.x = feat
        data.num_nodes = feat.shape[0]
        if self.y_cls is not None:
            data.y_cls = self.y_cls[idx]
        if self.y_reg is not None:
            data.y_reg = self.y_reg[idx]
        data.edge_index_adj = self.edge_index_adj
        data.edge_index_od = self.edge_index_od
        data.edge_index_od_t = self.edge_index_od_t
        return data


class TensorDatasetForDP(Dataset):
    """Plain tensor dataset for Opacus-compatible training.
    
    Returns plain tensors instead of PyG Data objects to avoid
    GlobalStorage issues with Opacus hooks.
    """
    def __init__(
        self,
        features: torch.Tensor,
        y_reg: Optional[torch.Tensor],
        y_cls: Optional[torch.Tensor],
    ) -> None:
        self.features = features.clone()
        self.y_reg = y_reg.clone() if y_reg is not None else None
        self.y_cls = y_cls.clone() if y_cls is not None else None

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {'x': self.features[idx]}
        if self.y_cls is not None:
            item['y_cls'] = self.y_cls[idx]
        if self.y_reg is not None:
            item['y_reg'] = self.y_reg[idx]
        return item


def _pyg_collate(batch_list: Sequence[BaseData]) -> Batch:
    return Batch.from_data_list(list(batch_list))


def _dp_collate_to_tensors(batch_list: Sequence[BaseData]) -> Dict[str, torch.Tensor]:
    """Collate PyG Data to plain tensors for Opacus compatibility.
    
    Returns dict with 'x', 'y_cls', 'y_reg', 'edge_index_adj', 'edge_index_od', 'edge_index_od_t'.
    Assumes batch_size=1 (single graph per batch).
    """
    if len(batch_list) != 1:
        raise ValueError(f"DP collate requires batch_size=1, got {len(batch_list)}")
    
    data = batch_list[0]
    result = {'x': data.x}
    
    if hasattr(data, 'y_cls') and data.y_cls is not None:
        result['y_cls'] = data.y_cls
    if hasattr(data, 'y_reg') and data.y_reg is not None:
        result['y_reg'] = data.y_reg
    
    # Edge indices are shared across all samples, not batch-specific
    result['edge_index_adj'] = data.edge_index_adj
    result['edge_index_od'] = data.edge_index_od
    result['edge_index_od_t'] = data.edge_index_od_t
    
    return result


class OpacusCompatibleWrapper(nn.Module):
    """Wrapper that takes plain tensors and constructs PyG Data internally.
    
    Opacus hooks intercept module inputs/outputs and expect plain tensors with .shape.
    PyG Data objects have GlobalStorage that breaks Opacus. This wrapper:
    1. Receives plain tensors as input (x, edge_indices, y_cls/y_reg)
    2. Internally constructs a PyG Data object
    3. Calls the wrapped model with the Data object
    4. Returns plain tensor outputs
    
    The edge indices are stored as buffers so they don't need to be passed every call.
    """
    
    def __init__(
        self,
        model: nn.Module,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ):
        super().__init__()
        self.model = model
        # Store edge indices as non-persistent buffers (won't be saved in state_dict)
        self.register_buffer('_edge_index_adj', edge_index_adj.clone(), persistent=False)
        self.register_buffer('_edge_index_od', edge_index_od.clone(), persistent=False)
        self.register_buffer('_edge_index_od_t', edge_index_od_t.clone(), persistent=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward with plain tensor input, returns (logits, reg_output)."""
        # DP loaders use batch_size=1 which adds a leading batch dimension.
        # PyG's GATConv expects node features shaped [num_nodes, in_channels].
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        # Construct PyG Data object inside the forward pass
        data = Data(
            x=x,
            edge_index_adj=self._edge_index_adj,
            edge_index_od=self._edge_index_od,
            edge_index_od_t=self._edge_index_od_t,
        )
        logits, reg_out = self.model(data)
        return logits, reg_out
    
    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stage 1: Return embeddings and classification logits."""
        if x.dim() == 3 and x.size(0) == 1:
            x = x.squeeze(0)
        data = Data(
            x=x,
            edge_index_adj=self._edge_index_adj,
            edge_index_od=self._edge_index_od,
            edge_index_od_t=self._edge_index_od_t,
        )
        return self.model.forward_classifier(data)
    
    def forward_regressor(self, hidden: torch.Tensor) -> torch.Tensor:
        """Stage 2: Regress on precomputed embeddings."""
        return self.model.forward_regressor(hidden)
    
    # Expose inner model attributes for state dict access
    @property
    def encoder(self):
        return self.model.encoder
    
    @property
    def classifier(self):
        return self.model.classifier
    
    @property
    def regressor(self):
        return self.model.regressor
    
    @regressor.setter
    def regressor(self, value):
        self.model.regressor = value
    
    @property
    def dropout_cls(self):
        return self.model.dropout_cls
    
    @property
    def dropout_reg(self):
        return self.model.dropout_reg


class OpacusTensorOnlyWrapper(nn.Module):
    """Tensor-only wrapper for Opacus + PyG.

    Key idea: Opacus (hooks mode) registers hooks on *all* submodules.
    If any submodule receives a torch_geometric.data.Data input, Opacus may
    crash when trying to infer batch size via `.shape` on PyG storages.

    This wrapper avoids passing PyG `Data` objects to any `nn.Module` by calling
    the underlying encoder internals with plain tensors:
      - x: [num_nodes, in_channels] (or [1, num_nodes, in_channels] -> squeezed)
      - edge_index_*: [2, num_edges]

    It reuses the original model's submodules/parameters (GATConv, KAN, etc.).
    """

    def __init__(
        self,
        model: nn.Module,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> None:
        super().__init__()
        # Register model as a proper submodule so Opacus can track its parameters
        self.add_module('model', model)
        self.register_buffer('_edge_index_adj', edge_index_adj.clone(), persistent=False)
        self.register_buffer('_edge_index_od', edge_index_od.clone(), persistent=False)
        self.register_buffer('_edge_index_od_t', edge_index_od_t.clone(), persistent=False)

    def _ensure_2d_x(self, x: torch.Tensor) -> torch.Tensor:
        # For Opacus: Keep batch dimension! Opacus needs shape[0] to be batch_size.
        # Input from DataLoader with batch_size=1 is [1, 50, 40]
        # We squeeze here for GATConv, but we'll unsqueeze output for Opacus.
        if x.dim() == 3 and x.size(0) == 1:
            return x.squeeze(0)  # [50, 40] for GATConv
        return x

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode with manual graph operations to avoid PyG Data objects.
        
        This method MUST call all encoder submodules through proper forward() calls
        so that Opacus hooks can track gradients correctly.
        """
        # Track if we had batch dim
        had_batch_dim = (x.dim() == 3 and x.size(0) == 1)
        x = self._ensure_2d_x(x)
        enc = self.model.encoder

        # Mirror LightweightGATEncoder.forward(), but with tensor inputs.
        weights = F.softmax(torch.stack([enc.alpha_adj, enc.alpha_od, enc.alpha_od_t]), dim=0)
        w_adj, w_od, w_od_t = weights[0], weights[1], weights[2]

        # Call GAT modules - they expect [num_nodes, features]
        x_adj = enc.gat_adj(x, self._edge_index_adj)
        x_od = enc.gat_od(x, self._edge_index_od)
        x_od_t = enc.gat_od_t(x, self._edge_index_od_t)

        num_nodes = x_adj.size(0)
        scalars = torch.cat(
            [
                w_adj.expand(num_nodes, 1),
                w_od.expand(num_nodes, 1),
                w_od_t.expand(num_nodes, 1),
            ],
            dim=1,
        )

        x_concat = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
        fused = F.relu(enc.fusion_kan(x_concat))
        fused = enc.dropout(fused)
        return fused

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(x)
        logits = self.model.classifier(self.model.dropout_cls(hidden))
        reg_out = self.model.regressor(self.model.dropout_reg(hidden))
        return logits, reg_out

    def forward_classifier(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self._encode(x)
        logits = self.model.classifier(self.model.dropout_cls(hidden))
        return hidden, logits

    def forward_regressor(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.model.regressor(self.model.dropout_reg(hidden))

    @property
    def encoder(self):
        return self.model.encoder

    @property
    def classifier(self):
        return self.model.classifier

    @property
    def regressor(self):
        return self.model.regressor

    @regressor.setter
    def regressor(self, value):
        self.model.regressor = value

    @property
    def dropout_cls(self):
        return self.model.dropout_cls

    @property
    def dropout_reg(self):
        return self.model.dropout_reg


class RegressorOnlyWrapper(nn.Module):
    """Wrapper that only exposes regressor for Opacus, computing encoder in no_grad.
    
    This avoids Opacus hooks firing on frozen encoder layers which causes
    'Per sample gradient is not initialized' errors.
    """
    
    def __init__(
        self,
        full_model: nn.Module,
        edge_index_adj: torch.Tensor,
        edge_index_od: torch.Tensor,
        edge_index_od_t: torch.Tensor,
    ) -> None:
        super().__init__()
        # Only register regressor as a submodule - this is what Opacus will wrap
        self.regressor = full_model.regressor
        self.dropout_reg = full_model.dropout_reg
        
        # Keep references to frozen parts (not registered as submodules)
        self._encoder = full_model.encoder
        self._classifier = full_model.classifier
        self._dropout_cls = full_model.dropout_cls
        
        # Store edge indices
        self.register_buffer('_edge_index_adj', edge_index_adj.clone(), persistent=False)
        self.register_buffer('_edge_index_od', edge_index_od.clone(), persistent=False)
        self.register_buffer('_edge_index_od_t', edge_index_od_t.clone(), persistent=False)
    
    def _ensure_2d_x(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3 and x.size(0) == 1:
            return x.squeeze(0)
        return x
    
    def _encode_no_grad(self, x: torch.Tensor) -> torch.Tensor:
        """Compute encoder output without gradients (frozen encoder)."""
        x = self._ensure_2d_x(x)
        enc = self._encoder
        
        with torch.no_grad():
            weights = F.softmax(torch.stack([enc.alpha_adj, enc.alpha_od, enc.alpha_od_t]), dim=0)
            w_adj, w_od, w_od_t = weights[0], weights[1], weights[2]
            
            x_adj = enc.gat_adj(x, self._edge_index_adj)
            x_od = enc.gat_od(x, self._edge_index_od)
            x_od_t = enc.gat_od_t(x, self._edge_index_od_t)
            
            num_nodes = x_adj.size(0)
            scalars = torch.cat([
                w_adj.expand(num_nodes, 1),
                w_od.expand(num_nodes, 1),
                w_od_t.expand(num_nodes, 1),
            ], dim=1)
            
            x_concat = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
            fused = F.relu(enc.fusion_kan(x_concat))
            fused = enc.dropout(fused)
        
        # Return with requires_grad=True so regressor can compute gradients
        return fused.detach().requires_grad_(True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: frozen encoder -> trainable regressor."""
        hidden = self._encode_no_grad(x)
        reg_out = self.regressor(self.dropout_reg(hidden))
        return reg_out


def _noop_update_grid(x: torch.Tensor, margin: float = 0.01) -> None:
    """No-op replacement for KANLinear.update_grid for Opacus compatibility."""
    pass


def _core_model(model: nn.Module) -> nn.Module:
    """Unwrap wrappers (Opacus/DP/DataParallel) to get the underlying model.

    Opacus wraps modules into GradSampleModule, which doesn't expose custom
    helper methods (e.g., forward_classifier) as attributes.
    """
    seen: set[int] = set()
    cur: nn.Module = model
    while True:
        obj_id = id(cur)
        if obj_id in seen:
            return cur
        seen.add(obj_id)

        inner = getattr(cur, "module", None)
        if isinstance(inner, nn.Module) and inner is not cur:
            cur = inner
            continue

        inner = getattr(cur, "_module", None)
        if isinstance(inner, nn.Module) and inner is not cur:
            cur = inner
            continue

        return cur


def _make_kan_opacus_compatible(model: nn.Module) -> nn.Module:
    """Convert KANLinear grid buffers to frozen tensors for Opacus compatibility.

    Opacus doesn't support modules with trainable buffers. KANLinear uses a
    'grid' buffer that can be updated via update_grid(). This helper converts
    that buffer to a plain tensor attribute (not a buffer) so Opacus doesn't
    flag it. The grid update functionality is disabled but the forward pass
    works normally.
    """
    for _, module in model.named_modules():
        # Only touch modules that actually have a registered 'grid' buffer.
        buffers = dict(module.named_buffers(recurse=False))
        if 'grid' not in buffers:
            continue

        grid_data = buffers['grid'].detach().clone()
        # Remove buffer registration and store as a plain attribute.
        # This keeps forward() working (it reads self.grid) but removes it from
        # named_buffers(), which is what Opacus rejects.
        try:
            del module._buffers['grid']
        except Exception:
            pass
        setattr(module, 'grid', grid_data)

        # Disable update_grid to prevent modifying the grid during training.
        if hasattr(module, 'update_grid') and callable(module.update_grid):
            module.update_grid = _noop_update_grid  # type: ignore[method-assign]

    return model


class ManualDPOptimizer:
    """Manual DP-SGD optimizer for graph neural networks.
    
    Opacus hooks mode doesn't work well with GNNs because it confuses
    num_nodes with batch_size. This class implements DP-SGD manually:
    1. Clip gradient norms per sample (with batch_size=1, clip the full gradient)
    2. Add calibrated Gaussian noise
    3. Track privacy budget using Opacus accountant
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        noise_multiplier: float,
        max_grad_norm: float,
        target_delta: float,
    ):
        self.optimizer = optimizer
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm
        self.target_delta = target_delta
        self.steps = 0
        
        # Use Opacus accountant for privacy tracking
        try:
            from opacus.accountants import RDPAccountant
            self.accountant = RDPAccountant()
        except ImportError:
            self.accountant = None
    
    def zero_grad(self, set_to_none: bool = False):
        self.optimizer.zero_grad(set_to_none=set_to_none)
    
    def clip_and_noise_gradients(self):
        """Clip gradients and add noise for DP-SGD."""
        # Compute total gradient norm across all parameters
        total_norm = 0.0
        params_with_grad = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_coef = self.max_grad_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in params_with_grad:
                p.grad.data.mul_(clip_coef)
        
        # Add Gaussian noise
        for p in params_with_grad:
            noise = torch.randn_like(p.grad) * self.noise_multiplier * self.max_grad_norm
            p.grad.data.add_(noise)
        
        return total_norm
    
    def step(self, sample_rate: float = 1.0):
        """Perform optimizer step with DP noise."""
        # Clip and add noise
        grad_norm = self.clip_and_noise_gradients()
        
        # Update parameters
        self.optimizer.step()
        self.steps += 1
        
        # Update privacy accountant
        if self.accountant is not None:
            self.accountant.step(
                noise_multiplier=self.noise_multiplier,
                sample_rate=sample_rate,
            )
        
        return grad_norm
    
    def get_epsilon(self, delta: float) -> float:
        """Get current epsilon for given delta."""
        if self.accountant is not None:
            return self.accountant.get_epsilon(delta)
        return float('inf')
    
    @property
    def param_groups(self):
        return self.optimizer.param_groups


def _fill_missing_grad_samples(model: nn.Module, batch_size: int = 1) -> None:
    """Fill/fix grad_sample for all parameters to ensure consistent batch dimension.
    
    This is necessary for non-standard layers (like GATConv) that Opacus hooks
    don't recognize, or where Opacus incorrectly infers batch_size from node count.
    
    For batch_size=1 graph training, we FORCE all grad_samples to have shape [1, ...],
    overriding any that Opacus computed with wrong batch dimension (e.g., 50 for nodes).
    
    IMPORTANT: This must be called AFTER loss.backward() and BEFORE optimizer.step().
    """
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            continue
        
        # For batch_size=1, per-sample gradient = full gradient with batch dim of 1
        # We FORCE this for all parameters to ensure consistency
        param.grad_sample = param.grad.detach().unsqueeze(0)


def _make_private_with_opacus(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    dp_config: "DPConfig",
    accountant_state: Optional[Dict] = None,
) -> Tuple[nn.Module, torch.optim.Optimizer, DataLoader, Optional["PrivacyEngineType"]]:
    """Attach Opacus DP-SGD to (model, optimizer, dataloader).

    NOTE: PyG Batch objects are incompatible with Opacus's per-sample gradient
    hooks. We use batch_size=1 loaders and Opacus with poisson_sampling=False
    to avoid GlobalStorage errors during backward.

    If accountant_state is provided, it is loaded to continue privacy accounting
    across sequential stages.
    """
    if not dp_config.enabled:
        return model, optimizer, train_loader, None

    if PrivacyEngine is None or ModuleValidator is None:
        raise ImportError(
            "Opacus is required for DP training but is not installed. Install with: pip install opacus"
        )

    # Make KAN layers Opacus-compatible by freezing their grid buffers
    model = _make_kan_opacus_compatible(model)

    print(f"[DEBUG] Before ModuleValidator.fix():")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Use Opacus auto-fix for other common incompatibilities (e.g., BatchNorm -> GroupNorm)
    model = ModuleValidator.fix(model)

    print(f"[DEBUG] After ModuleValidator.fix():")
    print(f"  Model type: {type(model).__name__}")
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Opacus requires optimizer params to match module params exactly.
    # Since we may have modified/replaced modules above, rebuild the optimizer
    # using the same hyperparameters but the (possibly new) model.parameters().
    def _rebuild_optimizer(
        old_opt: torch.optim.Optimizer,
        new_params,
    ) -> torch.optim.Optimizer:
        opt_cls = old_opt.__class__
        if len(old_opt.param_groups) != 1:
            # Fall back to a single param group with defaults.
            return opt_cls(new_params, **old_opt.defaults)

        group = dict(old_opt.param_groups[0])
        group.pop('params', None)
        return opt_cls(new_params, **group)

    # Only pass trainable params to the optimizer. Opacus expects grad_sample
    # on every parameter in the optimizer; frozen params won't have it.
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        raise RuntimeError("No trainable parameters found for DP optimization.")

    optimizer = _rebuild_optimizer(optimizer, trainable_params)

    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        msg = "\n".join([str(e) for e in errors[:20]])
        raise RuntimeError(
            "Model is not Opacus-compatible (grad_sample hooks). "
            "Disable --dp or refactor unsupported layers.\n\n" + msg
        )

    privacy_engine = PrivacyEngine()
    make_private_result = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_config.noise_multiplier,
        max_grad_norm=dp_config.max_grad_norm,
        poisson_sampling=False,  # Disable for PyG compatibility
        # Use hooks mode (ew mode has compatibility issues with torch.stack)
        grad_sample_mode="hooks",
    )

    # Opacus return signature differs by version/config:
    # - (module, optimizer, data_loader)
    # - (module, optimizer, criterion, data_loader)
    if isinstance(make_private_result, tuple) and len(make_private_result) == 3:
        model, optimizer, train_loader = make_private_result
    elif isinstance(make_private_result, tuple) and len(make_private_result) == 4:
        model, optimizer, _criterion, train_loader = make_private_result
    else:
        raise RuntimeError(
            f"Unexpected return from PrivacyEngine.make_private: {type(make_private_result)}"
        )

    if accountant_state is not None:
        try:
            privacy_engine.accountant.load_state_dict(accountant_state)
        except Exception:
            pass

    return model, optimizer, train_loader, privacy_engine


@dataclass
class DPConfig:
    """Differential privacy configuration with budget control."""
    enabled: bool
    target_epsilon: float
    target_delta: float
    noise_multiplier: float
    max_grad_norm: float
    sample_rate: float
    epsilon_tolerance: float = 0.05
    poisson_sampling: bool = True


class FocalLoss(nn.Module):
    """Binary focal loss with logits for multi-channel targets."""

    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        # Standard BCE with logits per element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        # p_t is the model-assigned probability of the true class
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t).pow(self.gamma)
        loss = focal_factor * bce

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)
            # Apply alpha only to positives; keep negatives at weight 1.0
            alpha_factor = alpha_t * targets + (1.0 - targets)
            loss = alpha_factor * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def classification_metrics_per_channel(
    preds: np.ndarray,
    targets: np.ndarray,
    channel_names: Tuple[str, ...] = ("arrival", "departure"),
    prob_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute precision/recall/F1/accuracy per output channel.

    Expects preds/targets shaped [N, C] (or any shape that can be reshaped to [-1, C]).
    Returns both per-channel metrics and macro-averaged metrics.
    """
    preds_2d = preds.reshape(-1, preds.shape[-1])
    targets_2d = targets.reshape(-1, targets.shape[-1])
    n_channels = preds_2d.shape[1]

    metrics: Dict[str, float] = {}
    precisions, recalls, f1s, accuracies = [], [], [], []
    for c in range(n_channels):
        preds_bin = preds_2d[:, c] >= prob_threshold
        targets_bin = targets_2d[:, c] >= 0.5

        tp = np.logical_and(preds_bin, targets_bin).sum()
        fp = np.logical_and(preds_bin, ~targets_bin).sum()
        fn = np.logical_and(~preds_bin, targets_bin).sum()
        tn = np.logical_and(~preds_bin, ~targets_bin).sum()

        precision = float(tp / (tp + fp + 1e-8))
        recall = float(tp / (tp + fn + 1e-8))
        f1 = float(2 * precision * recall / (precision + recall + 1e-8))
        accuracy = float((tp + tn) / (tp + tn + fp + fn + 1e-8))

        name = channel_names[c] if c < len(channel_names) else f"ch{c}"
        metrics[f"precision_{name}"] = precision
        metrics[f"recall_{name}"] = recall
        metrics[f"f1_{name}"] = f1
        metrics[f"accuracy_{name}"] = accuracy

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        accuracies.append(accuracy)

    metrics["precision"] = float(np.mean(precisions)) if precisions else 0.0
    metrics["recall"] = float(np.mean(recalls)) if recalls else 0.0
    metrics["f1"] = float(np.mean(f1s)) if f1s else 0.0
    metrics["accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
    return metrics


def aggregate_node_to_graph(node_features: torch.Tensor) -> torch.Tensor:
    """Aggregate node-level features to graph-level via mean pooling."""
    return node_features.mean(dim=0, keepdim=True)


def ensure_graph_level_target(target: torch.Tensor) -> torch.Tensor:
    """Convert node-level targets to graph-level."""
    if target.dim() == 0:  # Scalar
        return target.unsqueeze(0)
    elif target.dim() == 1:  # [num_nodes]
        return target.mean(dim=0, keepdim=True)
    else:  # [num_nodes, feature_dim]
        return target.mean(dim=0, keepdim=True)


def train_stage1_with_dp(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    pos_weight: torch.Tensor,
    patience: int,
    dp_config: DPConfig,
    batch_size: int,
) -> Tuple[List[Dict], Optional[Dict], float]:
    """Train stage 1 (classifier) with optional DP-SGD via Opacus."""
    stage_start_time = time.time()
    print("\n" + "="*80)
    print("STAGE 1: TRAINING DELAY CLASSIFIER")
    print("="*80)
    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")
    
    # Get the core model (unwrap if needed)
    core_model = _core_model(model)
    
    print(f"[DEBUG] Core model type: {type(core_model).__name__}")
    print(f"[DEBUG] Core model has encoder: {hasattr(core_model, 'encoder')}")
    print(f"[DEBUG] Core model has classifier: {hasattr(core_model, 'classifier')}")
    print(f"[DEBUG] Core model has regressor: {hasattr(core_model, 'regressor')}")

    # Freeze regressor BEFORE wrapping
    for param in core_model.regressor.parameters():
        param.requires_grad = False
    
    # Verify we have trainable parameters
    trainable_params = [p for p in core_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in core_model.parameters() if p.requires_grad)
    encoder_count = sum(p.numel() for p in core_model.encoder.parameters() if p.requires_grad)
    classifier_count = sum(p.numel() for p in core_model.classifier.parameters() if p.requires_grad)
    regressor_count = sum(p.numel() for p in core_model.regressor.parameters() if p.requires_grad)
    
    print(f"[DEBUG] Core model parameter counts:")
    print(f"  Encoder trainable: {encoder_count}")
    print(f"  Classifier trainable: {classifier_count}")
    print(f"  Regressor trainable: {regressor_count}")
    print(f"  Total trainable: {trainable_count}")
    
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-4)
    
    # Ensure pos_weight is on the correct device
    if isinstance(pos_weight, (float, int)):
        pos_weight_t = torch.tensor([pos_weight], device=device)
    else:
        pos_weight_t = pos_weight.to(device)
        
    cls_loss_fn = FocalLoss(
        alpha=pos_weight_t,
        gamma=2.0,
        reduction="mean",
    )

    if dp_config.enabled:
        print(f"[OK] DP-SGD enabled (Opacus): eps_target={dp_config.target_epsilon}, delta={dp_config.target_delta}")
        print(f"  Noise multiplier: {dp_config.noise_multiplier}")
        print(f"  Max grad norm: {dp_config.max_grad_norm}")
        print(f"  Poisson sampling: {dp_config.poisson_sampling}")
    else:
        print("[X] DP-SGD disabled (standard training)")

    # For DP training, use plain tensor dataset with wrapper (no PyG Data)
    if dp_config.enabled:
        train_ds_dp = TensorDatasetForDP(train_x, y_reg=None, y_cls=train_y_cls)
        # Must use batch_size=1 because each sample is a full graph (50 nodes)
        train_loader_dp_raw = DataLoader(train_ds_dp, batch_size=1, shuffle=True)
        
        # Wrap model to accept plain tensors and manually do graph ops
        # Use core_model to ensure we wrap the actual model with frozen params
        wrapped_model = OpacusTensorOnlyWrapper(
            core_model,
            edge_index_adj=edge_indices[0].to(device),
            edge_index_od=edge_indices[1].to(device),
            edge_index_od_t=edge_indices[2].to(device),
        ).to(device)
        
        print(f"[DEBUG] Wrapped model type: {type(wrapped_model).__name__}")
        print(f"[DEBUG] Wrapped model has 'model' attr: {hasattr(wrapped_model, 'model')}")
        
        # Verify wrapped model has trainable parameters
        wrapped_trainable = [p for p in wrapped_model.parameters() if p.requires_grad]
        wrapped_all = list(wrapped_model.parameters())
        wrapped_named = list(wrapped_model.named_parameters())
        
        wrapped_trainable_count = sum(p.numel() for p in wrapped_trainable)
        wrapped_total_count = sum(p.numel() for p in wrapped_all)
        
        print(f"[DEBUG] Wrapped model parameter counts:")
        print(f"  Total parameter tensors: {len(wrapped_all)}")
        print(f"  Total parameter elements: {wrapped_total_count}")
        print(f"  Trainable parameter tensors: {len(wrapped_trainable)}")
        print(f"  Trainable parameter elements: {wrapped_trainable_count}")
        print(f"[DEBUG] First few parameter names:")
        for name, param in wrapped_named[:10]:
            print(f"  {name}: shape={param.shape}, requires_grad={param.requires_grad}")
        
        if not wrapped_trainable:
            raise RuntimeError("No trainable parameters in wrapped model!")
        
        # Create optimizer with trainable parameters
        optimizer = torch.optim.Adam(wrapped_trainable, lr=lr, weight_decay=1e-5)
        
        print(f"[DEBUG] Optimizer created with {len(optimizer.param_groups[0]['params'])} parameters")
        
        # Call Opacus to make private
        print(f"[DEBUG] Calling _make_private_with_opacus...")
        wrapped_model, optimizer, train_loader_dp, privacy_engine = _make_private_with_opacus(
            model=wrapped_model,
            optimizer=optimizer,
            train_loader=train_loader_dp_raw,
            dp_config=dp_config,
            accountant_state=None,
        )
        
        print(f"[DEBUG] After Opacus wrapping:")
        print(f"  Model type: {type(wrapped_model).__name__}")
        print(f"  Optimizer param groups: {len(optimizer.param_groups)}")
        print(f"  Params in optimizer: {len(optimizer.param_groups[0]['params'])}")
        
        # Check if Opacus-wrapped model has per-sample gradient hooks
        opacus_trainable = [p for p in wrapped_model.parameters() if p.requires_grad]
        print(f"  Trainable params after Opacus: {len(opacus_trainable)}")
        train_loader_to_use = train_loader_dp
        model_for_training = wrapped_model
    else:
        train_ds = GraphSequenceDataset(
            train_x,
            y_reg=None,
            y_cls=train_y_cls,
            edge_index_adj=edge_indices[0].detach().cpu(),
            edge_index_od=edge_indices[1].detach().cpu(),
            edge_index_od_t=edge_indices[2].detach().cpu(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_pyg_collate)
        train_loader_to_use = train_loader
        accum_steps = 1
        model_for_training = model
        privacy_engine = None

    val_ds = GraphSequenceDataset(
        val_x,
        y_reg=None,
        y_cls=val_y_cls,
        edge_index_adj=edge_indices[0].detach().cpu(),
        edge_index_od=edge_indices[1].detach().cpu(),
        edge_index_od_t=edge_indices[2].detach().cpu(),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_pyg_collate)

    base_model = _core_model(model)
    
    history = []
    best_f1 = 0.0
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="max")
    
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stage1_checkpoint.pth') if CHECKPOINT_DIR else None

    # Check if checkpoint exists to resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _, pe_state = load_checkpoint(model, optimizer, checkpoint_path)
        if dp_config.enabled and privacy_engine is not None and pe_state is not None:
            try:
                privacy_engine.load_state_dict(pe_state)
            except Exception:
                pass
        print(f"Resuming Stage 1 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model_for_training.train()
        epoch_losses: List[float] = []

        # Use direct iteration without BatchMemoryManager for batch_size=1 DP training
        loader_ctx = nullcontext(train_loader_to_use)

        with loader_ctx as train_iter:
            for _step_idx, batch in enumerate(train_iter):
                optimizer.zero_grad(set_to_none=True)  # Zero grad BEFORE forward pass for Opacus
                
                # Handle tensor dict from DP vs PyG Data from non-DP
                if dp_config.enabled and isinstance(batch, dict):
                    x_batch = batch['x'].to(device)
                    cls_targets = batch['y_cls'].to(device)
                    # Wrapper squeezes batch dim from x, so squeeze targets too
                    if cls_targets.dim() == 3 and cls_targets.size(0) == 1:
                        cls_targets = cls_targets.squeeze(0)
                    
                    # Debug first batch
                    if _step_idx == 0 and epoch == start_epoch:
                        print(f"\n[DEBUG] First batch forward pass:")
                        print(f"  x_batch shape: {x_batch.shape}")
                        print(f"  cls_targets shape: {cls_targets.shape}")
                        print(f"  Model type: {type(model_for_training).__name__}")
                    
                    # Wrapper handles plain tensors, returns (logits, reg)
                    node_logits, _ = model_for_training(x_batch)
                    
                    if _step_idx == 0 and epoch == start_epoch:
                        print(f"  node_logits shape: {node_logits.shape}")
                        print(f"  node_logits requires_grad: {node_logits.requires_grad}")
                        
                        # Check if any parameters have gradients
                        params_with_grad = [name for name, p in model_for_training.named_parameters() 
                                          if p.requires_grad and p.grad is not None]
                        print(f"  Params with grad before backward: {len(params_with_grad)}")
                else:
                    batch = batch.to(device)
                    _, node_logits = base_model.forward_classifier(batch)
                    cls_targets = batch.y_cls

                loss = cls_loss_fn(node_logits, cls_targets)
                
                if _step_idx == 0 and epoch == start_epoch:
                    print(f"  loss: {loss.item():.4f}")
                    print(f"  loss requires_grad: {loss.requires_grad}")
                
                loss.backward()
                
                # Fill missing grad_sample for parameters where Opacus hooks failed
                # This is necessary for GNN layers that Opacus doesn't natively support
                if dp_config.enabled:
                    _fill_missing_grad_samples(model_for_training, batch_size=1)
                
                if _step_idx == 0 and epoch == start_epoch:
                    # Check gradient statistics after backward
                    params_with_grad = [(name, p.grad.norm().item() if p.grad is not None else 0.0) 
                                       for name, p in model_for_training.named_parameters() 
                                       if p.requires_grad]
                    print(f"  After backward, params with grad: {len([g for _, g in params_with_grad if g > 0])}/{len(params_with_grad)}")
                    if params_with_grad:
                        print(f"  First 5 grad norms: {params_with_grad[:5]}")
                    
                    # Check for grad_sample attribute (Opacus specific) - more detailed
                    print(f"  [GRAD_SAMPLE DEBUG]")
                    for name, p in model_for_training.named_parameters():
                        if p.requires_grad:
                            has_attr = hasattr(p, 'grad_sample')
                            gs_value = getattr(p, 'grad_sample', None)
                            gs_type = type(gs_value).__name__ if gs_value is not None else "None"
                            gs_shape = gs_value.shape if hasattr(gs_value, 'shape') else "N/A"
                            print(f"    {name}: has_attr={has_attr}, type={gs_type}, shape={gs_shape}")
                            if name.count('.') <= 3:  # Limit depth for readability
                                break  # Just show first few
                    
                    # Check if GradSampleModule hooks are active
                    gsm = model_for_training
                    print(f"  GradSampleModule type: {type(gsm).__name__}")
                    if hasattr(gsm, '_hooks_enabled'):
                        print(f"  _hooks_enabled: {gsm._hooks_enabled}")
                    if hasattr(gsm, 'hooks_mode'):
                        print(f"  hooks_mode: {gsm.hooks_mode}")
                    if hasattr(gsm, '_module'):
                        print(f"  _module type: {type(gsm._module).__name__}")
                
                optimizer.step()
                epoch_losses.append(float(loss.item()))
                
                # Only debug first batch
                if _step_idx == 0 and epoch == start_epoch:
                    print(f"[DEBUG] First batch completed successfully\n")
        
        # Diagnostic logging every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                # Sample one batch for diagnostics
                sample_idx = 0
                sample_data = Data(
                    x=train_x[sample_idx].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, sample_logits = base_model.forward_classifier(sample_data)
                sample_probs = torch.sigmoid(sample_logits)
                print(f"  [STAGE 1 Diagnostics - Epoch {epoch}]")
                print(f"    Sample logits: min={sample_logits.min().item():.3f}, max={sample_logits.max().item():.3f}, mean={sample_logits.mean().item():.3f}")
                print(f"    Sample probs: min={sample_probs.min().item():.3f}, max={sample_probs.max().item():.3f}, mean={sample_probs.mean().item():.3f}")
                print(f"    Target mean: {train_y_cls[sample_idx].mean().item():.3f}")
        
        # Validation
        model.eval()
        val_probs, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, node_logits = base_model.forward_classifier(batch)
                val_probs.append(torch.sigmoid(node_logits).detach().cpu())
                val_targets.append(batch.y_cls.detach().cpu())

        val_probs_np = torch.cat(val_probs, dim=0).numpy() if val_probs else np.zeros((0, 2), dtype=np.float32)
        val_targets_np = torch.cat(val_targets, dim=0).numpy() if val_targets else np.zeros((0, 2), dtype=np.float32)
        # Per-channel (arrival/departure) metrics + macro averages
        val_metrics = classification_metrics_per_channel(
            val_probs_np,
            val_targets_np,
            channel_names=("arrival", "departure"),
        )
        
        epoch_time = time.time() - epoch_start_time
        
        if dp_config.enabled and privacy_engine is not None:
            current_epsilon = float(privacy_engine.get_epsilon(dp_config.target_delta))
        else:
            current_epsilon = float('inf')
        
        history.append({
            'epoch': epoch,
            'stage': 1,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_precision': val_metrics['precision'],
            'val_recall': val_metrics['recall'],
            'val_f1': val_metrics['f1'],
            'val_accuracy': val_metrics['accuracy'],
            'epsilon': current_epsilon,
            'delta': dp_config.target_delta if dp_config.enabled else 0.0,
            'epoch_time_seconds': epoch_time,
            'total_steps': int(getattr(getattr(privacy_engine, 'accountant', None), 'steps', 0)) if dp_config.enabled and privacy_engine is not None else 0,
        })
        
        eps_str = f"eps: {current_epsilon:.3f}/{dp_config.target_epsilon}" if dp_config.enabled else "No DP"
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val F1 (macro): {val_metrics['f1']:.4f} "
            f"[arr {val_metrics['f1_arrival']:.4f}, dep {val_metrics['f1_departure']:.4f}] | "
            f"{eps_str} | Time: {epoch_time:.2f}s"
        )
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_state = {
                'encoder': base_model.encoder.state_dict(),
                'classifier': base_model.classifier.state_dict(),
            }
            print("  [OK] New best checkpoint")
            # Save checkpoint to disk for resume capability
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, history[-1]['train_loss'], checkpoint_path, privacy_engine)
                print(f"  [OK] Checkpoint saved to {checkpoint_path}")
        
        if early_stopping(val_metrics['f1'], epoch):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        base_model.encoder.load_state_dict(best_state['encoder'])
        base_model.classifier.load_state_dict(best_state['classifier'])
    
    for param in base_model.regressor.parameters():
        param.requires_grad = True
    
    stage_time = time.time() - stage_start_time
    final_epsilon = float(privacy_engine.get_epsilon(dp_config.target_delta)) if dp_config.enabled and privacy_engine is not None else float('inf')
    
    print(f"\nStage 1 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    print(f"Final eps: {final_epsilon:.3f} (target: {dp_config.target_epsilon})")
    
    accountant_state = privacy_engine.accountant.state_dict() if dp_config.enabled and privacy_engine is not None else None
    return history, accountant_state, stage_time


def train_stage2_with_dp(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    class_threshold: float,
    delay_threshold: float,
    patience: int,
    dp_config: DPConfig,
    batch_size: int,
    stage1_accountant_state: Optional[Dict],
) -> Tuple[List[Dict], Optional[Dict], float]:
    """Train stage 2 (delayed flights regressor) with optional DP-SGD via Opacus."""
    stage_start_time = time.time()
    print("\n" + "="*80)
    print("STAGE 2: TRAINING DELAY REGRESSOR (DELAYED FLIGHTS)")
    print("="*80)
    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")

    core_model = _core_model(model)

    for param in core_model.encoder.parameters():
        param.requires_grad = False
    for param in core_model.classifier.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(core_model.regressor.parameters(), lr=lr, weight_decay=1e-4)
    huber_loss = nn.HuberLoss(reduction='none', delta=2.0)

    # Compute delay threshold in *scaled* space to match train_y_reg/val_y_reg.
    # StandardScaler: scaled = (x - mean) / std
    if scaler is not None and hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
        mean_np = np.array(scaler.mean, dtype=np.float32)
        std_np = np.array(scaler.std, dtype=np.float32)
        std_np = np.where(std_np == 0, 1.0, std_np)
        threshold_scaled_np = (np.full_like(mean_np, delay_threshold, dtype=np.float32) - mean_np) / std_np
        delay_threshold_scaled = torch.tensor(threshold_scaled_np, device=device, dtype=torch.float32)
    else:
        delay_threshold_scaled = torch.tensor(delay_threshold, device=device, dtype=torch.float32)

    def masked_huber_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Masked Huber loss where mask is defined by *ground-truth* delays."""
        thr = delay_threshold_scaled.to(targets.device)
        if thr.dim() == 0:
            thr = thr.unsqueeze(0)
        # Broadcast threshold across nodes/samples
        while thr.dim() < targets.dim():
            thr = thr.unsqueeze(0)

        mask = (targets > thr).float()
        if mask.dim() == 1:
            mask = mask.unsqueeze(-1)

        per_elem = huber_loss(preds, targets)
        num = (per_elem * mask).sum(dim=0)
        den = mask.sum(dim=0).clamp_min(1.0)
        return (num / den).mean()

    if dp_config.enabled:
        print("[OK] DP-SGD enabled (Opacus, continuing from stage 1)")
    else:
        print("[X] DP-SGD disabled")

    # For DP training, use plain tensor dataset with wrapper (no PyG Data)
    if dp_config.enabled:
        train_ds_dp = TensorDatasetForDP(train_x, y_reg=train_y_reg, y_cls=None)
        train_loader_dp_raw = DataLoader(train_ds_dp, batch_size=1, shuffle=True)
        
        wrapped_model = OpacusTensorOnlyWrapper(
            model,
            edge_index_adj=edge_indices[0].to(device),
            edge_index_od=edge_indices[1].to(device),
            edge_index_od_t=edge_indices[2].to(device),
        )
        
        # Create optimizer with all trainable parameters from wrapped model
        # (encoder and classifier are frozen, so only regressor params are trainable)
        trainable_params = [p for p in wrapped_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=1e-5)
        
        wrapped_model, optimizer, train_loader_dp, privacy_engine = _make_private_with_opacus(
            model=wrapped_model,
            optimizer=optimizer,
            train_loader=train_loader_dp_raw,
            dp_config=dp_config,
            accountant_state=stage1_accountant_state,
        )
        train_loader_to_use = train_loader_dp
        model_for_training = wrapped_model
    else:
        train_ds = GraphSequenceDataset(
            train_x,
            y_reg=train_y_reg,
            y_cls=None,
            edge_index_adj=edge_indices[0].detach().cpu(),
            edge_index_od=edge_indices[1].detach().cpu(),
            edge_index_od_t=edge_indices[2].detach().cpu(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_pyg_collate)
        train_loader_to_use = train_loader
        accum_steps = 1
        model_for_training = model
        privacy_engine = None

    val_ds = GraphSequenceDataset(
        val_x,
        y_reg=val_y_reg,
        y_cls=None,
        edge_index_adj=edge_indices[0].detach().cpu(),
        edge_index_od=edge_indices[1].detach().cpu(),
        edge_index_od_t=edge_indices[2].detach().cpu(),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_pyg_collate)
    
    history = []
    best_val_loss = float('inf')
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stage2_checkpoint.pth') if CHECKPOINT_DIR else None

    # Check if checkpoint exists to resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _, pe_state = load_checkpoint(model, optimizer, checkpoint_path)
        if dp_config.enabled and privacy_engine is not None and pe_state is not None:
            try:
                privacy_engine.load_state_dict(pe_state)
            except Exception:
                pass
        print(f"Resuming Stage 2 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model_for_training.train()
        epoch_losses: List[float] = []

        # Use direct iteration without BatchMemoryManager for batch_size=1 DP training
        loader_ctx = nullcontext(train_loader_to_use)

        with loader_ctx as train_iter:
            for _step_idx, batch in enumerate(train_iter):
                optimizer.zero_grad(set_to_none=True)  # Zero grad BEFORE forward pass for Opacus
                
                if dp_config.enabled and isinstance(batch, dict):
                    x_batch = batch['x'].to(device)
                    reg_targets = batch['y_reg'].to(device)
                    # Wrapper squeezes batch dim from x, so squeeze targets too
                    if reg_targets.dim() == 3 and reg_targets.size(0) == 1:
                        reg_targets = reg_targets.squeeze(0)
                    # Wrapper handles encoding internally, returns (logits, reg)
                    _, pred_reg = model_for_training(x_batch)
                else:
                    batch = batch.to(device)
                    pred_reg, _ = base_model.forward_regressor(batch)
                    reg_targets = batch.y_reg

                loss = masked_huber_loss(pred_reg, reg_targets)
                loss.backward()
                
                # Fill missing grad_sample for parameters where Opacus hooks failed
                if dp_config.enabled:
                    _fill_missing_grad_samples(model_for_training, batch_size=1)
                
                optimizer.step()
                epoch_losses.append(float(loss.item()))
        
        # Diagnostic logging every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                sample_idx = 0
                sample_data = Data(
                    x=train_x[sample_idx].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                orig_model = _core_model(model)
                _, sample_reg = orig_model(sample_data)
                # Stage 2 trains on ground-truth delayed targets.
                thr_cpu = delay_threshold_scaled.detach().cpu()
                sample_mask = (train_y_reg[sample_idx] > thr_cpu)
                sample_targets = train_y_reg[sample_idx][sample_mask]
                sample_preds = sample_reg.detach().cpu()[sample_mask]
                print(f"  [STAGE 2 Diagnostics - Epoch {epoch}]")
                if len(sample_targets) > 0:
                    print(f"    Sample predictions (delayed): min={sample_preds.min().item():.3f}, max={sample_preds.max().item():.3f}, mean={sample_preds.mean().item():.3f}")
                    print(f"    Target (scaled, delayed): min={sample_targets.min().item():.3f}, max={sample_targets.max().item():.3f}, mean={sample_targets.mean().item():.3f}")
                else:
                    print(f"    No delayed samples in this example")
                mask_count = sample_mask.sum().item()
                total_count = train_y_cls[sample_idx].numel()
                print(f"    Mask coverage: {mask_count}/{total_count} ({100*mask_count/total_count:.1f}% delayed)")
        
        # Validation
        orig_model = _core_model(model)
        orig_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, node_reg = orig_model(batch)
                loss = masked_huber_loss(node_reg, batch.y_reg)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        epoch_time = time.time() - epoch_start_time
        
        if dp_config.enabled and privacy_engine is not None:
            current_epsilon = float(privacy_engine.get_epsilon(dp_config.target_delta))
        else:
            current_epsilon = float('inf')
        
        history.append({
            'epoch': epoch,
            'stage': 2,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_loss': val_loss,
            'epsilon': current_epsilon,
            'delta': dp_config.target_delta if dp_config.enabled else 0.0,
            'epoch_time_seconds': epoch_time,
            'total_steps': int(getattr(getattr(privacy_engine, 'accountant', None), 'steps', 0)) if dp_config.enabled and privacy_engine is not None else 0,
        })
        
        eps_str = f"eps: {current_epsilon:.3f}/{dp_config.target_epsilon}" if dp_config.enabled else "No DP"
        print(
            f"Epoch {epoch}/{epochs} | Train Loss: {history[-1]['train_loss']:.4f} | "
            f"Val Loss: {val_loss:.4f} | {eps_str} | Time: {epoch_time:.2f}s"
        )
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = orig_model.regressor.state_dict()
            print("  [OK] New best checkpoint")
            # Save checkpoint to disk for resume capability
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, privacy_engine)
                print(f"  [OK] Checkpoint saved to {checkpoint_path}")
        
        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        orig_model = _core_model(model)
        orig_model.regressor.load_state_dict(best_state)
    
    for param in _core_model(model).parameters():
        param.requires_grad = True
    
    stage_time = time.time() - stage_start_time
    final_epsilon = float(privacy_engine.get_epsilon(dp_config.target_delta)) if dp_config.enabled and privacy_engine is not None else float('inf')
    
    print(f"\nStage 2 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    print(f"Final eps: {final_epsilon:.3f} (target: {dp_config.target_epsilon})")
    
    accountant_state = privacy_engine.accountant.state_dict() if dp_config.enabled and privacy_engine is not None else None
    return history, accountant_state, stage_time


def train_stage3_with_dp(
    model: SequentialTwoStagePredictor,
    train_x: torch.Tensor,
    train_y_reg: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_x: torch.Tensor,
    val_y_reg: torch.Tensor,
    val_y_cls: torch.Tensor,
    edge_indices: Tuple,
    device: torch.device,
    epochs: int,
    lr: float,
    scaler,
    class_threshold: float,
    delay_threshold: float,
    patience: int,
    dp_config: DPConfig,
    batch_size: int,
    stage2_accountant_state: Optional[Dict],
) -> Tuple[List[Dict], Optional[Dict], float]:
    """Train stage 3 (non-delayed regressor) with optional DP-SGD via Opacus."""
    stage_start_time = time.time()
    print("\n" + "=" * 80)
    print("STAGE 3: TRAINING DELAY REGRESSOR (NON-DELAYED FLIGHTS) - IMPROVED")
    print(f"Training on flights with |delay| < {delay_threshold} min")
    print("=" * 80)
    print(f"Train samples: {len(train_x)} | Val samples: {len(val_x)}")

    # Freeze encoder and classifier (regressor-only fine-tuning)
    # IMPORTANT: keep the shared encoder frozen so the Stage 2 delayed regressor
    # stays compatible with encoder embeddings.
    core_model = _core_model(model)
    for param in core_model.encoder.parameters():
        param.requires_grad = False
    for param in core_model.classifier.parameters():
        param.requires_grad = False
    # Ensure regressor is trainable
    for param in core_model.regressor.parameters():
        param.requires_grad = True

    # Much lower LR, regressor only
    optimizer = torch.optim.Adam(
        core_model.regressor.parameters(),
        lr=lr * 0.01,
        weight_decay=1e-5,
    )

    # Use Huber loss (more robust than MSE)
    reg_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)

    if dp_config.enabled:
        print("[OK] DP-SGD enabled (Opacus, continuing from stage 2)")
    else:
        print("[X] DP-SGD disabled")

    print(f"[OK] Regressor-only training with LR={lr * 0.01:.6f}")
    print("[OK] Using Huber loss (robust to outliers)")

    # For DP training, use plain tensor dataset with wrapper (no PyG Data)
    if dp_config.enabled:
        train_ds_dp = TensorDatasetForDP(train_x, y_reg=train_y_reg, y_cls=None)
        train_loader_dp_raw = DataLoader(train_ds_dp, batch_size=1, shuffle=True)
        
        # Use RegressorOnlyWrapper to avoid Opacus hooks on frozen encoder
        # This computes encoder output with no_grad and only wraps regressor
        wrapped_model = RegressorOnlyWrapper(
            core_model,
            edge_index_adj=edge_indices[0].to(device),
            edge_index_od=edge_indices[1].to(device),
            edge_index_od_t=edge_indices[2].to(device),
        ).to(device)
        
        # Create optimizer with all trainable parameters from wrapped model
        # (encoder and classifier are frozen, so only regressor params are trainable)
        trainable_params = [p for p in wrapped_model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=lr * 0.01, weight_decay=1e-5)
        
        wrapped_model, optimizer, train_loader_dp, privacy_engine = _make_private_with_opacus(
            model=wrapped_model,
            optimizer=optimizer,
            train_loader=train_loader_dp_raw,
            dp_config=dp_config,
            accountant_state=stage2_accountant_state,
        )
        train_loader_to_use = train_loader_dp
        model_for_training = wrapped_model
    else:
        train_ds = GraphSequenceDataset(
            train_x,
            y_reg=train_y_reg,
            y_cls=None,
            edge_index_adj=edge_indices[0].detach().cpu(),
            edge_index_od=edge_indices[1].detach().cpu(),
            edge_index_od_t=edge_indices[2].detach().cpu(),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_pyg_collate)
        train_loader_to_use = train_loader
        accum_steps = 1
        model_for_training = model
        privacy_engine = None

    val_ds = GraphSequenceDataset(
        val_x,
        y_reg=val_y_reg,
        y_cls=None,
        edge_index_adj=edge_indices[0].detach().cpu(),
        edge_index_od=edge_indices[1].detach().cpu(),
        edge_index_od_t=edge_indices[2].detach().cpu(),
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=_pyg_collate)

    history: List[Dict] = []
    best_val_loss = float("inf")
    best_state: Optional[Dict[str, Dict[str, torch.Tensor]]] = None
    early_stopping = EarlyStopping(patience=patience, mode="min")

    start_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'stage3_checkpoint.pth') if CHECKPOINT_DIR else None

    # Check if checkpoint exists to resume
    if checkpoint_path and os.path.exists(checkpoint_path):
        start_epoch, _, pe_state = load_checkpoint(model, optimizer, checkpoint_path)
        if dp_config.enabled and privacy_engine is not None and pe_state is not None:
            try:
                privacy_engine.load_state_dict(pe_state)
            except Exception:
                pass
        print(f"Resuming Stage 3 from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        model_for_training.train()
        epoch_losses: List[float] = []
        total_nondelayed = 0.0
        total_values = 0.0

        # Use direct iteration without BatchMemoryManager for batch_size=1 DP training
        loader_ctx = nullcontext(train_loader_to_use)

        with loader_ctx as train_iter:
            for _step_idx, batch in enumerate(train_iter):
                optimizer.zero_grad(set_to_none=True)  # Zero grad BEFORE forward for Opacus
                
                if dp_config.enabled and isinstance(batch, dict):
                    x_batch = batch['x'].to(device)
                    reg_targets_t = batch['y_reg'].to(device)
                    # Wrapper squeezes batch dim from x, so squeeze targets too
                    if reg_targets_t.dim() == 3 and reg_targets_t.size(0) == 1:
                        reg_targets_t = reg_targets_t.squeeze(0)
                    # RegressorOnlyWrapper returns only reg_preds (not tuple)
                    reg_preds_t = model_for_training(x_batch)
                else:
                    batch = batch.to(device)
                    _, reg_preds_t = base_model.forward_regressor(batch)
                    reg_targets_t = batch.y_reg

                # Denormalize targets (detached) ONLY for mask creation
                if scaler is not None:
                    with torch.no_grad():
                        targets_denorm = torch.from_numpy(
                            scaler.inverse_transform(reg_targets_t.detach().cpu().numpy())
                        ).to(device)
                else:
                    targets_denorm = reg_targets_t.detach()

                # Create element-wise mask for non-delayed values in denormalized space
                element_mask = (targets_denorm.abs() < delay_threshold).float()
                num_nondelayed_per_ch = element_mask.sum(dim=0)
                num_nondelayed = num_nondelayed_per_ch.sum()

                # Always compute loss - use mask to weight elements
                # If no non-delayed elements, loss will be ~0 but gradient graph is maintained
                loss_per_element = reg_loss_fn(reg_preds_t, reg_targets_t) * element_mask
                # Per-channel average loss (clamp to avoid div by zero)
                loss_nondelayed_ch = loss_per_element.sum(dim=0) / num_nondelayed_per_ch.clamp_min(1.0)
                loss = loss_nondelayed_ch.mean()
                
                if num_nondelayed > 0:
                    total_nondelayed += num_nondelayed.item()
                    total_values += float(element_mask.numel())
                
                loss.backward()
                
                # Fill missing grad_sample for parameters where Opacus hooks failed
                if dp_config.enabled:
                    _fill_missing_grad_samples(model_for_training, batch_size=1)
                
                optimizer.step()
                epoch_losses.append(float(loss.item()))

        # Calculate nondelayed ratio for this epoch
        nondelayed_ratio = (
            total_nondelayed / total_values if total_values > 0 else 0.0
        )

        # Diagnostic logging every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                sample_idx = 0
                sample_data = Data(
                    x=train_x[sample_idx].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, sample_reg = model(sample_data)
                print(f"  [STAGE 3 Diagnostics - Epoch {epoch}]")
                print(f"    Sample predictions: min={sample_reg.min().item():.3f}, max={sample_reg.max().item():.3f}, mean={sample_reg.mean().item():.3f}")
                if scaler is not None:
                    sample_denorm = scaler.inverse_transform(sample_reg.cpu().numpy())
                    print(f"    Denormalized preds: min={sample_denorm.min():.2f}, max={sample_denorm.max():.2f}, mean={sample_denorm.mean():.2f} min")
                print(f"    Non-delayed elements in epoch: {total_nondelayed:.0f}/{total_values:.0f} ({100*nondelayed_ratio:.1f}%)")

        # Validation in denormalized space with same mask
        model.eval()
        val_losses: List[float] = []
        val_nondelayed = 0.0
        val_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, node_reg = model(batch)

                if scaler is not None:
                    target_denorm = torch.from_numpy(
                        scaler.inverse_transform(batch.y_reg.detach().cpu().numpy())
                    ).to(device)
                    pred_denorm = torch.from_numpy(
                        scaler.inverse_transform(node_reg.detach().cpu().numpy())
                    ).to(device)
                else:
                    target_denorm = batch.y_reg
                    pred_denorm = node_reg

                element_mask = (target_denorm.abs() < delay_threshold).float()
                num_nondelayed_per_ch = element_mask.sum(dim=0)
                num_nondelayed = num_nondelayed_per_ch.sum()

                if num_nondelayed > 0:
                    se = ((pred_denorm - target_denorm) ** 2) * element_mask
                    loss_val_ch = se.sum(dim=0) / num_nondelayed_per_ch.clamp_min(1.0)
                    loss_val = loss_val_ch.mean()
                    val_losses.append(loss_val.item())
                    val_nondelayed += num_nondelayed.item()
                    val_total += float(element_mask.numel())

        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        epoch_time = time.time() - epoch_start_time

        current_epsilon = (
            float(privacy_engine.get_epsilon(dp_config.target_delta))
            if dp_config.enabled and privacy_engine is not None
            else float("inf")
        )

        val_nondelayed_ratio = (
            val_nondelayed / val_total if val_total > 0 else 0.0
        )

        history.append(
            {
                "epoch": epoch,
                "stage": 3,
                "train_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "train_nondelayed": total_nondelayed,
                "train_nondelayed_ratio": nondelayed_ratio,
                "val_loss": val_loss,
                "val_nondelayed": val_nondelayed,
                "val_nondelayed_ratio": val_nondelayed_ratio,
                "epsilon": current_epsilon,
                "epoch_time_seconds": epoch_time,
                "total_steps": int(getattr(getattr(privacy_engine, 'accountant', None), 'steps', 0)) if dp_config.enabled and privacy_engine is not None else 0,
            }
        )

        eps_str = (
            f"eps: {current_epsilon:.3f}/{dp_config.target_epsilon}"
            if dp_config.enabled
            else "No DP"
        )
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val: {val_loss:.4f} | Non-delayed: {total_nondelayed:.0f} "
            f"({nondelayed_ratio*100:.1f}%) | Val ND: {val_nondelayed:.0f} "
            f"({val_nondelayed_ratio*100:.1f}%) | {eps_str} | Time: {epoch_time:.2f}s"
        )

        if val_loss < best_val_loss and val_nondelayed > 0:
            best_val_loss = val_loss
            orig_model = _core_model(model)
            best_state = {
                "encoder": orig_model.encoder.state_dict(),
                "regressor": orig_model.regressor.state_dict(),
            }
            print("  [OK] New best (Stage 3 v2)")
            # Save checkpoint to disk for resume capability
            if checkpoint_path:
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path, privacy_engine)
                print(f"  [OK] Checkpoint saved to {checkpoint_path}")

        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        orig_model = _core_model(model)
        orig_model.encoder.load_state_dict(best_state["encoder"])
        orig_model.regressor.load_state_dict(best_state["regressor"])

    stage_time = time.time() - stage_start_time
    final_epsilon = (
        float(privacy_engine.get_epsilon(dp_config.target_delta))
        if dp_config.enabled and privacy_engine is not None
        else float("inf")
    )

    print(f"\nStage 3 completed in {stage_time:.2f}s ({stage_time/60:.2f} min)")
    print(f"Final eps: {final_epsilon:.3f} (target: {dp_config.target_epsilon})")

    accountant_state = privacy_engine.accountant.state_dict() if dp_config.enabled and privacy_engine is not None else None
    return history, accountant_state, stage_time

def final_evaluation(
    model: SequentialTwoStagePredictor,
    edge_indices: Tuple,
    device: torch.device,
    scaler,
    horizons: List[int],
    delay_dim: int,
    num_nodes: int,
    test_x: torch.Tensor,
    test_y_reg: torch.Tensor,
    test_y_cls: torch.Tensor,
    class_threshold: float,
    delay_threshold: float,
    model_path: str,
    histories: List[Dict],
    final_epsilon: float,
    final_delta: float,
    stage1_time: float,
    stage2_time: float,
    stage3_time: float,
    train_samples: int,
    val_samples: int,
    dp_config: DPConfig,
) -> None:
    """Final evaluation and export with Stage 3 regressor."""
    print("\n" + "="*80)
    print("FINAL TEST EVALUATION")
    print("="*80)
    print(f"Test samples: {len(test_x)}")

    # Evaluation doesn't need the Opacus wrapper; unwrap to access helper methods.
    model = _core_model(model)
    
    model_dir = os.path.dirname(model_path)
    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
    
    # Save both regressors if available (Stage 2 delayed + Stage 3 non-delayed).
    to_save = {
        'encoder': model.encoder.state_dict(),
        'classifier': model.classifier.state_dict(),
        # Backwards-compatible key: treat 'regressor' as the delayed regressor.
        'regressor': getattr(model, 'regressor_delayed', model.regressor).state_dict(),
        'final_epsilon': float(final_epsilon),
        'final_delta': float(final_delta),
        'target_epsilon': float(dp_config.target_epsilon),
        'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
    }
    if hasattr(model, 'regressor_delayed'):
        to_save['regressor_delayed'] = model.regressor_delayed.state_dict()
    if hasattr(model, 'regressor_nondelayed'):
        to_save['regressor_nondelayed'] = model.regressor_nondelayed.state_dict()
    torch.save(to_save, model_path)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.encoder.load_state_dict(checkpoint['encoder'])
    model.classifier.load_state_dict(checkpoint['classifier'])

    # Recreate delayed/non-delayed regressors if present.
    if 'regressor_delayed' in checkpoint and 'regressor_nondelayed' in checkpoint:
        model.regressor_delayed = copy.deepcopy(model.regressor)
        model.regressor_nondelayed = copy.deepcopy(model.regressor)
        # Make KAN layers Opacus-compatible (remove grid buffers) before loading
        _make_kan_opacus_compatible(model.regressor_delayed)
        _make_kan_opacus_compatible(model.regressor_nondelayed)
        # Load with strict=False to handle any grid key mismatches
        model.regressor_delayed.load_state_dict(checkpoint['regressor_delayed'], strict=False)
        model.regressor_nondelayed.load_state_dict(checkpoint['regressor_nondelayed'], strict=False)
    else:
        model.regressor.load_state_dict(checkpoint['regressor'])
    
    model.eval()
    
    logits_list, reg_list = [], []
    targets_cls_list, targets_reg_list = [], []
    
    USE_FAST_EVAL = False
    
    print("[EVALUATION] Processing test samples...")
    with torch.no_grad():
        for i in range(len(test_x)):
            data = Data(
                x=test_x[i].to(device),
                edge_index_adj=edge_indices[0],
                edge_index_od=edge_indices[1],
                edge_index_od_t=edge_indices[2],
            )
            
            # Always compute classifier once.
            hidden, node_logits = model.forward_classifier(data)
            probs = torch.sigmoid(node_logits)

            # If we have two regressors, route by the classifier gate.
            if hasattr(model, 'regressor_delayed') and hasattr(model, 'regressor_nondelayed'):
                hidden_dropped = model.dropout_reg(hidden)
                reg_delayed = model.regressor_delayed(hidden_dropped)
                reg_nondelayed = model.regressor_nondelayed(hidden_dropped)
                # Soft gating: smoothly mix regressors based on delayed probability.
                # class_threshold is treated as the midpoint (gate=0.5 when prob==threshold).
                gate = torch.sigmoid((probs - class_threshold) * 10.0)
                node_reg = reg_delayed * gate + reg_nondelayed * (1.0 - gate)
            else:
                node_reg = model.forward_regressor(hidden)

            logits_list.append(probs.cpu().numpy())
            reg_list.append(node_reg.cpu().numpy())
            targets_cls_list.append(test_y_cls[i].cpu().numpy())
            targets_reg_list.append(test_y_reg[i].cpu().numpy())
            
            if (i + 1) % 1000 == 0 or (i + 1) == len(test_x):
                print(f"  Processed {i+1}/{len(test_x)} samples...")
    
    test_probs = np.concatenate(logits_list, axis=0)
    test_reg_preds = np.concatenate(reg_list, axis=0)
    test_cls_targets = np.concatenate(targets_cls_list, axis=0)
    test_reg_targets = np.concatenate(targets_reg_list, axis=0)
    
    # Classification metrics (arrival/departure separately + macro)
    test_cls_metrics = classification_metrics_per_channel(
        test_probs,
        test_cls_targets,
        channel_names=("arrival", "departure"),
        prob_threshold=class_threshold,
    )
    
    # NOTE: If a dual-regressor checkpoint was loaded, routing was already applied
    # in the per-sample loop above. For legacy single-regressor checkpoints, we
    # keep the original behavior (delayed-only gating).
    if 'regressor_delayed' in checkpoint and 'regressor_nondelayed' in checkpoint:
        gated_preds = test_reg_preds
    else:
        test_mask = (test_probs >= class_threshold)
        gated_preds = test_reg_preds * test_mask
    
    print(f"\n[DENORMALIZATION] Checking predictions...")
    print(f"  Gated predictions shape: {gated_preds.shape}")
    print(f"  Gated predictions (scaled): min={gated_preds.min():.3f}, max={gated_preds.max():.3f}, mean={gated_preds.mean():.3f}")
    print(f"  Test targets (scaled): min={test_reg_targets.min():.3f}, max={test_reg_targets.max():.3f}, mean={test_reg_targets.mean():.3f}")
    
    if scaler is not None:
        print(f"  Applying inverse transform with scaler...")
        print(f"    Scaler mean: {scaler.mean}, std: {scaler.std}")
        preds_denorm = scaler.inverse_transform(gated_preds)
        targets_denorm = scaler.inverse_transform(test_reg_targets)
        print(f"  After denormalization:")
        print(f"    Predictions: min={preds_denorm.min():.2f}, max={preds_denorm.max():.2f}, mean={preds_denorm.mean():.2f}")
        print(f"    Targets: min={targets_denorm.min():.2f}, max={targets_denorm.max():.2f}, mean={targets_denorm.mean():.2f}")
    else:
        preds_denorm = gated_preds
        targets_denorm = test_reg_targets
    
    # Treat negative values as on time (0 min)
    preds_denorm = np.maximum(0, preds_denorm)
    targets_denorm = np.maximum(0, targets_denorm)
    
    # Flatten both predictions and targets consistently for element-wise evaluation
    preds_flat = preds_denorm.flatten()
    targets_flat = targets_denorm.flatten()
    
    # Evaluate on delayed flights (Actual >= Threshold)
    delayed_mask = targets_flat >= delay_threshold
    if delayed_mask.sum() > 0:
        delayed_preds = preds_flat[delayed_mask]
        delayed_targets = targets_flat[delayed_mask]
        mae_delayed = np.mean(np.abs(delayed_preds - delayed_targets))
        rmse_delayed = np.sqrt(np.mean((delayed_preds - delayed_targets) ** 2))
    else:
        mae_delayed, rmse_delayed = 0.0, 0.0
    
    # Evaluate on non-delayed flights (1 min <= Actual < Threshold)
    nondelayed_mask = (targets_flat >= 1.0) & (targets_flat < delay_threshold)
    if nondelayed_mask.sum() > 0:
        nondelayed_preds = preds_flat[nondelayed_mask]
        nondelayed_targets = targets_flat[nondelayed_mask]
        mae_nondelayed = np.mean(np.abs(nondelayed_preds - nondelayed_targets))
        rmse_nondelayed = np.sqrt(np.mean((nondelayed_preds - nondelayed_targets) ** 2))
    else:
        mae_nondelayed, rmse_nondelayed = 0.0, 0.0
    
    # Overall metrics
    mae_overall = np.mean(np.abs(preds_denorm - targets_denorm))
    rmse_overall = np.sqrt(np.mean((preds_denorm - targets_denorm) ** 2))
    
    print("\nCLASSIFICATION (macro over arrival/departure):")
    print(f"  Precision: {test_cls_metrics['precision']:.4f} | Recall: {test_cls_metrics['recall']:.4f}")
    print(f"  F1: {test_cls_metrics['f1']:.4f} | Accuracy: {test_cls_metrics['accuracy']:.4f}")
    print("  Per-channel:")
    print(
        f"    Arrival   - P: {test_cls_metrics['precision_arrival']:.4f} "
        f"R: {test_cls_metrics['recall_arrival']:.4f} F1: {test_cls_metrics['f1_arrival']:.4f} "
        f"Acc: {test_cls_metrics['accuracy_arrival']:.4f}"
    )
    print(
        f"    Departure - P: {test_cls_metrics['precision_departure']:.4f} "
        f"R: {test_cls_metrics['recall_departure']:.4f} F1: {test_cls_metrics['f1_departure']:.4f} "
        f"Acc: {test_cls_metrics['accuracy_departure']:.4f}"
    )
    
    print(f"\nREGRESSION (delayed flights >= {delay_threshold} min):")
    print(f"  MAE: {mae_delayed:.4f} min | RMSE: {rmse_delayed:.4f} min")
    print(f"  Number of delayed samples: {delayed_mask.sum()}")
    
    print(f"\nREGRESSION (non-delayed flights 1-{delay_threshold} min):")
    print(f"  MAE: {mae_nondelayed:.4f} min | RMSE: {rmse_nondelayed:.4f} min")
    print(f"  Number of non-delayed samples: {nondelayed_mask.sum()}")
    
    print("\nREGRESSION (overall):")
    print(f"  MAE: {mae_overall:.4f} min | RMSE: {rmse_overall:.4f} min")
    
    # Visualize classification results
    if VISUALIZATION_AVAILABLE:
        print("\n[VISUALIZATION] Generating classification results plots...")
        try:
            # Keep plotting arrays aligned in length.
            # We plot a single channel consistently (arrival = channel 0) for both
            # classification and regression.
            if test_probs.ndim > 1 and test_probs.shape[1] > 1:
                test_cls_pred = (test_probs[:, 0] >= class_threshold).astype(int)
                test_cls_true = test_cls_targets[:, 0].astype(int)
            else:
                test_cls_pred = (test_probs >= class_threshold).astype(int).reshape(-1)
                test_cls_true = test_cls_targets.astype(int).reshape(-1)

            if targets_denorm.ndim > 1 and targets_denorm.shape[1] > 1:
                test_reg_true = targets_denorm[:, 0].reshape(-1)
                test_reg_pred = preds_denorm[:, 0].reshape(-1)
            else:
                test_reg_true = targets_denorm.reshape(-1)
                test_reg_pred = preds_denorm.reshape(-1)
            
            visualize_classification_results(
                test_cls_true,
                test_cls_pred,
                test_reg_true,
                test_reg_pred,
                threshold=delay_threshold,
                save_path=f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            print("  [OK] Classification results visualization saved")

            visualize_regression_timeseries(
                targets_denorm,
                preds_denorm,
                title="Regression Over Time (True vs Predicted)",
                xlabel="Time (sample index)",
                ylabel="Delay (minutes)",
                save_path=f"regression_timeseries_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            )
            print("  [OK] Regression time-series visualization saved")
        except Exception as e:
            print(f"  [X] Error generating classification results visualization: {e}")
    
    print("\nPRIVACY BUDGET:")
    print(f"  Target eps: {dp_config.target_epsilon:.3f}")
    print(f"  Final eps: {final_epsilon:.3f}")
    if dp_config.enabled:
        if final_epsilon <= dp_config.target_epsilon:
            print(f"  [OK] Budget maintained (within target)")
        else:
            overshoot = final_epsilon - dp_config.target_epsilon
            print(f"  [!] Budget exceeded by {overshoot:.3f} eps ({overshoot/dp_config.target_epsilon*100:.1f}%)")
    print(f"  Final delta: {final_delta:.2e}")
    
    print("\nTRAINING TIME:")
    total_time = stage1_time + stage2_time + stage3_time
    print(f"  Stage 1: {stage1_time:.2f}s ({stage1_time/60:.2f} min)")
    print(f"  Stage 2: {stage2_time:.2f}s ({stage2_time/60:.2f} min)")
    print(f"  Stage 3: {stage3_time:.2f}s ({stage3_time/60:.2f} min)")
    print(f"  Total: {total_time:.2f}s ({total_time/60:.2f} min)")
    
    print("\nDATASET SIZES:")
    print(f"  Train: {train_samples} | Val: {val_samples} | Test: {len(test_x)}")
    
    # Generate unique filenames with noise multiplier (sigma) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_str = f"sigma{dp_config.noise_multiplier:.2f}".replace(".", "_")
    
    history_csv = f"kan_gat_dp_three_stage_history_{sigma_str}_{timestamp}.csv"
    summary_csv = f"kan_gat_dp_three_stage_summary_{sigma_str}_{timestamp}.csv"
    
    # Export history
    if histories:
        with open(history_csv, "w", newline="") as f:
            all_fields = sorted({k for row in histories for k in row})
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            writer.writerows(histories)
    
    # Export summary
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        summary = {
            'classification_precision': test_cls_metrics['precision'],
            'classification_recall': test_cls_metrics['recall'],
            'classification_f1': test_cls_metrics['f1'],
            'classification_accuracy': test_cls_metrics['accuracy'],
            'regression_mae_delayed': mae_delayed,
            'regression_rmse_delayed': rmse_delayed,
            'regression_mae_nondelayed': mae_nondelayed,
            'regression_rmse_nondelayed': rmse_nondelayed,
            'regression_mae_overall': mae_overall,
            'regression_rmse_overall': rmse_overall,
            'num_delayed_samples': int(delayed_mask.sum()),
            'num_nondelayed_samples': int(nondelayed_mask.sum()),
            'target_epsilon': dp_config.target_epsilon,
            'final_epsilon': final_epsilon,
            'epsilon_exceeded': final_epsilon > dp_config.target_epsilon if dp_config.enabled else False,
            'epsilon_overshoot': max(0, final_epsilon - dp_config.target_epsilon) if dp_config.enabled else 0,
            'final_delta': final_delta,
            'stage1_time_seconds': stage1_time,
            'stage2_time_seconds': stage2_time,
            'stage3_time_seconds': stage3_time,
            'total_training_time_seconds': total_time,
            'total_training_time_minutes': total_time / 60,
            'train_samples': train_samples,
            'val_samples': val_samples,
            'test_samples': len(test_x),
        }
        for k, v in summary.items():
            writer.writerow([k, v])
    
    print(f"\n[OK] Results saved to:")
    print(f"  - {model_path}")
    print(f"  - {history_csv}")
    print(f"  - {summary_csv}")
    
    # Download files to local device (only in Colab)
    if IN_COLAB and colab_files is not None:
        print("\n[DOWNLOAD] Downloading files to local device...")
        
        # Files to download: model, history, summary, and checkpoints
        files_to_download = [
            model_path,
            history_csv,
            summary_csv,
        ]
        
        # Add checkpoint files if they exist
        checkpoint_files = [
            os.path.join(CHECKPOINT_DIR, 'stage1_checkpoint.pth'),
            os.path.join(CHECKPOINT_DIR, 'stage2_checkpoint.pth'),
            os.path.join(CHECKPOINT_DIR, 'stage3_checkpoint.pth'),
        ]
        files_to_download.extend(checkpoint_files)
        
        for file_path in files_to_download:
            if os.path.exists(file_path):
                try:
                    colab_files.download(file_path)
                    print(f"  [OK] Downloaded: {file_path}")
                except Exception as e:
                    print(f"  [X] Error downloading {file_path}: {e}")
            else:
                print(f"  - File not found: {file_path}")
    else:
        print("\n[INFO] Not running in Colab - files saved locally, no download needed.")


def save_checkpoint(model, optimizer, epoch, loss, path, privacy_engine=None):
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if privacy_engine is not None:
        try:
            payload['privacy_engine_state_dict'] = privacy_engine.state_dict()
        except Exception:
            payload['privacy_engine_state_dict'] = None
    torch.save(payload, path)
    
    # Automatically download checkpoint in Colab as soon as it's saved
    if IN_COLAB and colab_files is not None:
        try:
            colab_files.download(path)
            print(f"  [OK] Checkpoint downloaded: {path}")
        except Exception as e:
            print(f"  [X] Error downloading checkpoint: {e}")

def load_checkpoint(model, optimizer, path):
    # Use map_location to handle loading checkpoints saved on different devices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = int(checkpoint.get('epoch', 0))
    loss = float(checkpoint.get('loss', 0.0))
    pe_state = checkpoint.get('privacy_engine_state_dict', None)
    return epoch, loss, pe_state

def _load_checkpoint_state_dict(model: nn.Module, state_dict: Dict, device: torch.device) -> None:
    """Load state dict handling Opacus/wrapper prefixes.
    
    Checkpoints may have been saved from:
    - GradSampleModule -> OpacusTensorOnlyWrapper -> SequentialTwoStagePredictor
    - OpacusTensorOnlyWrapper -> SequentialTwoStagePredictor
    - SequentialTwoStagePredictor directly
    
    This function tries to match keys by stripping common prefixes.
    """
    core = _core_model(model)
    core_state = core.state_dict()
    
    # Possible prefixes used by Opacus wrappers
    prefixes_to_try = [
        '_module.model.',  # GradSampleModule -> OpacusTensorOnlyWrapper
        '_module.',        # GradSampleModule
        'model.',          # OpacusTensorOnlyWrapper
        '',                # Direct match
    ]
    
    new_state = {}
    for key in core_state.keys():
        matched = False
        for prefix in prefixes_to_try:
            full_key = prefix + key
            if full_key in state_dict:
                new_state[key] = state_dict[full_key]
                matched = True
                break
        if not matched:
            # Keep original value if not found in checkpoint
            print(f"  [WARN] Key not found in checkpoint: {key}")
            new_state[key] = core_state[key]
    
    core.load_state_dict(new_state, strict=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-stage DP-SGD for KAN-GAT with epsilon budget control")
    parser.add_argument('--data_source', type=str, default='cdata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument(
        '--horizons',
        type=int,
        nargs=1,
        default=[12],
        choices=[3, 6, 12, 24],
        help='Train/test ONLY this horizon (choose one of 3, 6, 12, 24). Example: --horizons 24',
    )
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--class_threshold', type=float, default=0.5)
    parser.add_argument('--use_node_level', action='store_true', default=True, help='Use node-level labels')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--stage1_epochs', type=int, default=10)
    parser.add_argument('--stage2_epochs', type=int, default=10)
    parser.add_argument('--stage3_epochs', type=int, default=12, help='Epochs for non-delayed regressor')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--dp', default=True, action='store_true', help='Enable DP-SGD (WARNING: Currently incompatible with PyG - will be disabled)')
    parser.add_argument('--target_epsilon', type=float, default=15.0, help='Target epsilon for tracking (not used for computing noise)')
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=0.5, help='Fixed noise multiplier for DP-SGD (lower=less noise, less privacy)')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='Max gradient norm for clipping (higher allows larger gradients)')
    parser.add_argument('--sample_rate', type=float, default=0.02)
    parser.add_argument('--epsilon_tolerance', type=float, default=0.05)
    parser.add_argument('--model_path', type=str, default='kan_gat_dp_three_stage.pth')
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='auto',
        help=(
            "Where to save/load stage checkpoints. "
            "Use 'auto' to create a new per-run subfolder under ./checkpoints, "
            "use 'latest' to reuse the most recent run folder, "
            "or pass an explicit folder name/path."
        ),
    )
    parser.add_argument('--seed', type=int, default=None, help='Random seed (None for random)')
    parser.add_argument('--balance_50_50', action='store_true', default=False, help='Apply random undersampling to achieve 50-50 class balance')
    parser.add_argument('--skip_to_stage', type=int, default=1, choices=[1, 2, 3], help='Skip to stage N (loads checkpoints for prior stages)')
    return parser.parse_args()


def main() -> None:
    global CHECKPOINT_DIR
    args = parse_args()

    # Always pick checkpoint directory from args, so re-runs can resume consistently.
    CHECKPOINT_DIR = setup_checkpoint_directory(args.checkpoint_dir)
    
    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'
    
    if args.seed is not None:
        set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data(
        args.data_source,
        weather_file=args.weather_file,
        period_hours=args.period_hours,
        data_source=args.data_source,
    )
    
    horizons = sorted({h for h in args.horizons if h > 0})
    if len(horizons) != 1:
        raise ValueError(
            f"This script trains/tests a single horizon only. "
            f"Pass exactly one value via --horizons (3/6/12/24). Got: {args.horizons}"
        )
    max_horizon = horizons[0]
    feature_dim = train_inputs.shape[2]
    delay_dim = train_delay_scaled.shape[2]
    in_channels = args.seq_len * feature_dim
    out_channels = delay_dim
    
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
    if args.use_node_level:
        print("[INFO] Using NODE-LEVEL labels")
    else:
        print("[INFO] Using GRAPH-LEVEL labels")
    
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    val_x, val_y_reg, val_y_cls = build_fn(
        val_inputs, val_delay_scaled, val_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    test_x, test_y_reg, test_y_cls = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    
    if args.balance_50_50:
        print("\n[INFO] Applying random undersampling for 50-50 balance...")
        # Determine sample-level labels (majority vote for node-level)
        sample_means = train_y_cls.mean(dim=(1, 2))
        sample_labels = (sample_means >= 0.5).long()
        
        pos_indices = (sample_labels == 1).nonzero(as_tuple=True)[0]
        neg_indices = (sample_labels == 0).nonzero(as_tuple=True)[0]
        
        n_pos = len(pos_indices)
        n_neg = len(neg_indices)
        
        print(f"  Original counts - Positive (Delayed): {n_pos}, Negative: {n_neg}")
        
        if n_pos > 0 and n_neg > 0:
            min_count = min(n_pos, n_neg)
            print(f"  Undersampling to {min_count} samples per class...")
            
            # Randomly select indices
            perm_pos = torch.randperm(n_pos)[:min_count]
            perm_neg = torch.randperm(n_neg)[:min_count]
            
            selected_pos = pos_indices[perm_pos]
            selected_neg = neg_indices[perm_neg]
            
            # Combine and shuffle
            combined_indices = torch.cat([selected_pos, selected_neg])
            combined_indices = combined_indices[torch.randperm(len(combined_indices))]
            
            # Apply selection
            train_x = train_x[combined_indices]
            train_y_reg = train_y_reg[combined_indices]
            train_y_cls = train_y_cls[combined_indices]
            
            print(f"  New training set size: {len(train_x)}")
        else:
            print("  WARNING: Cannot balance 50-50 because one class has 0 samples. Skipping undersampling.")

    print(f"\n[DATA] Class distribution:")
    print(f"  Train: {train_y_cls.mean().item():.2%} delayed")
    print(f"  Val: {val_y_cls.mean().item():.2%} delayed")
    print(f"  Test: {test_y_cls.mean().item():.2%} delayed")
    
    # Comprehensive data validation
    print(f"\n[DATA VALIDATION] Checking data quality...")
    print(f"  Train features shape: {train_x.shape}")
    print(f"  Train targets (reg) shape: {train_y_reg.shape}")
    print(f"  Train targets (cls) shape: {train_y_cls.shape}")
    
    # Check for NaN/Inf
    print(f"  Train features: NaN={torch.isnan(train_x).any().item()}, Inf={torch.isinf(train_x).any().item()}")
    print(f"  Train reg targets: NaN={torch.isnan(train_y_reg).any().item()}, Inf={torch.isinf(train_y_reg).any().item()}")
    print(f"  Train cls targets: NaN={torch.isnan(train_y_cls).any().item()}, Inf={torch.isinf(train_y_cls).any().item()}")
    
    # Target statistics (scaled)
    print(f"  Train reg targets (scaled): min={train_y_reg.min().item():.3f}, max={train_y_reg.max().item():.3f}, mean={train_y_reg.mean().item():.3f}, std={train_y_reg.std().item():.3f}")
    print(f"  Train cls targets: min={train_y_cls.min().item():.3f}, max={train_y_cls.max().item():.3f}, unique values={train_y_cls.unique().tolist()}")
    
    # Per-channel distribution for regression targets
    if train_y_reg.dim() >= 2 and train_y_reg.shape[-1] == 2:
        print(f"  Arrival channel (scaled): min={train_y_reg[..., 0].min().item():.3f}, max={train_y_reg[..., 0].max().item():.3f}, mean={train_y_reg[..., 0].mean().item():.3f}")
        print(f"  Departure channel (scaled): min={train_y_reg[..., 1].min().item():.3f}, max={train_y_reg[..., 1].max().item():.3f}, mean={train_y_reg[..., 1].mean().item():.3f}")
    
    # Check classification label distribution per channel
    if train_y_cls.dim() >= 2 and train_y_cls.shape[-1] == 2:
        arr_delayed = train_y_cls[..., 0].mean().item()
        dep_delayed = train_y_cls[..., 1].mean().item()
        print(f"  Per-channel delayed rate: Arrival={arr_delayed:.2%}, Departure={dep_delayed:.2%}")
    
    # Scaler diagnostics
    print(f"\n[SCALER] Checking normalization parameters...")
    print(f"  Scaler mean: {scaler.mean}")
    print(f"  Scaler std: {scaler.std}")
    
    # Test denormalization on a sample
    sample_scaled = train_y_reg[0, 0, :].cpu().numpy()  # First node, both channels
    sample_denorm = scaler.inverse_transform(sample_scaled.reshape(1, -1))
    print(f"  Test denormalization:")
    print(f"    Scaled: {sample_scaled}")
    print(f"    Denormalized: {sample_denorm.flatten()}")
    print(f"    Expected range: 0-100 min (negative means early arrival)")
    
    print(f"  [OK] Data validation complete")
    
   
    edge_indices = (
        edge_index_adj.to(device),
        edge_index_od.to(device),
        edge_index_od_t.to(device),
    )
    
    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=128,  # Increased from 16 for better capacity
        regressor_extra_layer=True,
    ).to(device)
    
    total_samples = len(train_x)
    sample_rate = args.batch_size / total_samples
    steps_per_epoch = int(np.ceil(total_samples / args.batch_size))
    total_steps = (args.stage1_epochs + args.stage2_epochs + args.stage3_epochs) * steps_per_epoch
    
    if args.dp:
        print(f"\nDIFFERENTIAL PRIVACY CONFIGURATION:")
        print(f"  Noise multiplier (sigma): {args.noise_multiplier:.3f}")
        print(f"  Sampling: Poisson (Opacus)")
        print(f"  Sample rate per step (q): {sample_rate:.4f} (batch_size={args.batch_size} / total={total_samples})")
        print(f"  Steps per epoch: {steps_per_epoch}")
        print(f"  Expected total steps: {total_steps} ({steps_per_epoch} steps/epoch × {args.stage1_epochs + args.stage2_epochs + args.stage3_epochs} epochs)")
        print(f"  Target epsilon: {args.target_epsilon:.3f}")
        print(f"  Accounting: RDP (Opacus)")
        
        # Calculate expected noise scale for diagnostics
        noise_scale = args.noise_multiplier * args.max_grad_norm / args.batch_size
        print(f"\n[DP DIAGNOSTICS]")
        print(f"  Noise scale: {noise_scale:.6f} (noise_multiplier × max_grad_norm / batch_size)")
        print(f"  For good learning, gradient norms should be > {noise_scale * 3:.6f} (3x noise scale)")
        print(f"  Max gradient norm (clip threshold): {args.max_grad_norm}")
        
        args.sample_rate = sample_rate
    
    dp_config = DPConfig(
        enabled=args.dp,
        target_epsilon=args.target_epsilon,
        target_delta=args.target_delta,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        sample_rate=sample_rate if args.dp else args.sample_rate,
        epsilon_tolerance=args.epsilon_tolerance,
        poisson_sampling=True,
    )
    
    # Per-channel (arrival, departure) positive rate aggregated across all samples and airports
    cls_pos_rate = train_y_cls.float().mean(dim=(0, 1))  # shape: [2]
    # pos_weight = negatives / positives, broadcast over the two channels
    pos_weight = (1.0 - cls_pos_rate + 1e-6) / (cls_pos_rate + 1e-6)
    
    print("\n" + "="*80)
    print("DATASET INFORMATION")
    print("="*80)
    print(f"Train samples: {len(train_x)}")
    print(f"Val samples: {len(val_x)}")
    print(f"Test samples: {len(test_x)}")
    print(f"Class balance (delayed): {cls_pos_rate.mean().item():.2%}")
    
    # Skip to requested stage if checkpoints exist
    if args.skip_to_stage > 1:
        print(f"\n[SKIP] Skipping to Stage {args.skip_to_stage}, loading checkpoints...")
        stage1_ckpt = os.path.join(CHECKPOINT_DIR, 'stage1_checkpoint.pth')
        if os.path.exists(stage1_ckpt):
            checkpoint = torch.load(stage1_ckpt, map_location=device, weights_only=False)
            _load_checkpoint_state_dict(model, checkpoint['model_state_dict'], device)
            print(f"  [OK] Loaded Stage 1 checkpoint from {stage1_ckpt}")
        else:
            raise FileNotFoundError(f"Stage 1 checkpoint not found: {stage1_ckpt}")
        history_s1, accountant_s1_state, stage1_time = [], None, 0.0
    else:
        history_s1, accountant_s1_state, stage1_time = train_stage1_with_dp(
            model, train_x, train_y_cls, val_x, val_y_cls,
            edge_indices, device, args.stage1_epochs, args.lr,
            pos_weight, args.patience, dp_config, args.batch_size,
        )

    # Stage 2 delayed-sample diagnostics (sample-level, not node-level)
    cls_delayed_per_sample = (train_y_cls >= args.class_threshold)
    while cls_delayed_per_sample.dim() > 1:
        cls_delayed_per_sample = cls_delayed_per_sample.any(dim=-1)
    delayed_samples_cls = int(cls_delayed_per_sample.sum().item())

    if scaler is not None and hasattr(scaler, 'mean') and hasattr(scaler, 'std'):
        mean_np = np.array(scaler.mean, dtype=np.float32)
        std_np = np.array(scaler.std, dtype=np.float32)
        std_np = np.where(std_np == 0, 1.0, std_np)
        thr_scaled_np = (np.full_like(mean_np, args.delay_threshold, dtype=np.float32) - mean_np) / std_np
        thr_scaled_t = torch.tensor(thr_scaled_np, dtype=train_y_reg.dtype)
    else:
        thr_scaled_t = torch.tensor(args.delay_threshold, dtype=train_y_reg.dtype)

    reg_delayed_per_sample = (train_y_reg > thr_scaled_t)
    while reg_delayed_per_sample.dim() > 1:
        reg_delayed_per_sample = reg_delayed_per_sample.any(dim=-1)
    delayed_samples_reg = int(reg_delayed_per_sample.sum().item())

    print(f"\n[STAGE 2 DIAGNOSTIC] Delayed samples by cls threshold: {delayed_samples_cls}/{len(train_x)}")
    print(f"[STAGE 2 DIAGNOSTIC] Delayed samples by reg threshold: {delayed_samples_reg}/{len(train_x)}")
    
    if args.skip_to_stage > 2:
        print(f"\n[SKIP] Skipping Stage 2, loading checkpoint...")
        stage2_ckpt = os.path.join(CHECKPOINT_DIR, 'stage2_checkpoint.pth')
        if os.path.exists(stage2_ckpt):
            checkpoint = torch.load(stage2_ckpt, map_location=device, weights_only=False)
            _load_checkpoint_state_dict(model, checkpoint['model_state_dict'], device)
            print(f"  [OK] Loaded Stage 2 checkpoint from {stage2_ckpt}")
        else:
            raise FileNotFoundError(f"Stage 2 checkpoint not found: {stage2_ckpt}")
        history_s2, accountant_s2_state, stage2_time = [], None, 0.0
    else:
        history_s2, accountant_s2_state, stage2_time = train_stage2_with_dp(
            model, train_x, train_y_reg, train_y_cls,
            val_x, val_y_reg, val_y_cls, edge_indices, device,
            args.stage2_epochs, args.lr, scaler, args.class_threshold, args.delay_threshold,
            args.patience, dp_config, args.batch_size, accountant_s1_state,
        )

    # Preserve the delayed-flight regressor learned in Stage 2.
    # Stage 3 will train a separate regressor for non-delayed flights.
    base_model = _core_model(model)
    base_model.regressor_delayed = copy.deepcopy(base_model.regressor).to(device)
    # Freeze the delayed regressor so it's not trained in Stage 3
    for param in base_model.regressor_delayed.parameters():
        param.requires_grad = False

    # Initialize a fresh copy to train on non-delayed flights.
    base_model.regressor = copy.deepcopy(base_model.regressor_delayed).to(device)
    # Unfreeze the new regressor copy for training
    for param in base_model.regressor.parameters():
        param.requires_grad = True
    
    history_s3, accountant_s3_state, stage3_time = train_stage3_with_dp(
        model, train_x, train_y_reg, train_y_cls,
        val_x, val_y_reg, val_y_cls, edge_indices, device,
        args.stage3_epochs, args.lr, scaler, args.class_threshold,
        args.delay_threshold,  # FIXED: Pass actual delay threshold
        args.patience, dp_config, args.batch_size, accountant_s2_state,
    )

    # Capture the non-delayed regressor trained in Stage 3.
    base_model.regressor_nondelayed = copy.deepcopy(base_model.regressor).to(device)
    
    combined_history = history_s1 + history_s2 + history_s3
    
    if dp_config.enabled:
        # Opacus epsilon is already logged per-epoch; use the most recent.
        final_epsilon = float(combined_history[-1].get('epsilon', float('inf'))) if combined_history else float('inf')
        final_delta = dp_config.target_delta
    else:
        final_epsilon = float('inf')
        final_delta = 0.0
   
    # Generate unique model path with noise multiplier (sigma) and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sigma_str = f"sigma{dp_config.noise_multiplier:.2f}".replace(".", "_")
    
    # Update model path if using default
    if args.model_path == 'kan_gat_dp_three_stage.pth':
        args.model_path = f"kan_gat_dp_three_stage_{sigma_str}_{timestamp}.pth"
    
    print(f"\nOutput model will be saved to: {args.model_path}")
    
    final_evaluation(
        model, edge_indices, device, scaler, horizons,
        delay_dim, num_nodes, test_x, test_y_reg, test_y_cls,
        args.class_threshold, args.delay_threshold, args.model_path, combined_history,
        final_epsilon, final_delta, stage1_time, stage2_time, stage3_time,
        len(train_x), len(val_x), dp_config,
    )


def setup_checkpoint_directory(checkpoint_dir: str = 'auto') -> str:
    from pathlib import Path
    try:
        from google.colab import drive  # type: ignore[import-not-found]
        from IPython.core.getipython import get_ipython  # type: ignore[import-not-found]
        if get_ipython() is not None:  # Running in Notebook
            drive.mount('/content/drive')
            base_path = "/content/drive/MyDrive/FlightDelay_Checkpoints"
        else:
            # Running as normal Python script
            base_path = "./checkpoints"
    except Exception:
        # Not in Colab
        print("[OK] Checkpoints will be saved locally.")
        base_path = "./checkpoints"

    base_dir = Path(base_path)
    base_dir.mkdir(parents=True, exist_ok=True)

    latest_file = base_dir / "latest_run.txt"

    def _write_latest(path: Path) -> None:
        try:
            latest_file.write_text(str(path), encoding="utf-8")
        except Exception:
            # Don't fail training just because we can't write the marker.
            pass

    # Resolve run directory
    ck = (checkpoint_dir or "auto").strip().lower()

    if ck in {"auto", "new"}:
        # Create a unique subfolder per run so parallel debugger runs don't overwrite checkpoints.
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}_{uuid.uuid4().hex[:6]}"
        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_latest(run_dir)
    elif ck == "latest":
        if not latest_file.exists():
            raise FileNotFoundError(
                f"No latest run marker found at {latest_file}. "
                f"Run once with --checkpoint_dir auto to create it, or pass an explicit --checkpoint_dir."
            )
        txt = latest_file.read_text(encoding="utf-8").strip()
        candidate = Path(txt)
        run_dir = candidate if candidate.is_absolute() else (base_dir / candidate)
        if not run_dir.exists():
            raise FileNotFoundError(f"Latest run folder does not exist: {run_dir}")
    else:
        candidate = Path(checkpoint_dir)
        run_dir = candidate if candidate.is_absolute() else (base_dir / candidate)
        run_dir.mkdir(parents=True, exist_ok=True)
        _write_latest(run_dir)

    print(f"[OK] Checkpoints for this run: {run_dir}")
    return str(run_dir)



# Global checkpoint directory - set at runtime
CHECKPOINT_DIR: str = ""


if __name__ == '__main__':
    main()