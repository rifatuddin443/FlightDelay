"""Graph-based Models for Benchmark
This module integrates STPN, DSAFnet, and custom CNN-KAN models into the benchmark.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data, Batch

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))

from baseline_methods import test_error

# Try importing models
try:
    from DSAFnet import DSAFNet
    DSAFNET_AVAILABLE = True
except Exception as e:
    print(f"[Warning] DSAFnet not available: {e}")
    DSAFNet = None
    DSAFNET_AVAILABLE = False

try:
    # Try to import STPN from model.py
    from model import STPN as STPNModel
    STPN_AVAILABLE = True
except Exception as e:
    print(f"[Warning] STPN not available: {e}")
    STPNModel = None
    STPN_AVAILABLE = False


# ============================================================================
# GRAPH MODEL WRAPPERS
# ============================================================================

class GraphModelWrapper:
    """Wrapper for graph-based models with unified interface."""
    
    def __init__(self, model, model_name: str, device: str = 'cuda'):
        self.model = model
        self.model_name = model_name
        self.device = device
        if hasattr(model, 'to'):
            self.model.to(device)
        self.optimizer = None
        self.criterion = nn.MSELoss()
    
    def prepare_data_for_training(self, data_dict: Dict[str, Any], batch_size: int = 32):
        """Prepare graph data for training."""
        # This method should be overridden by specific model wrappers
        raise NotImplementedError
    
    def train(self, train_loader, optimizer, epochs: int = 50):
        """Train the model."""
        self.model.train()
        training_start = time.time()
        
        for epoch in range(epochs):
            epoch_loss = 0
            num_batches = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                # Forward pass (model-specific)
                loss = self._compute_loss(batch)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                print(f"[{self.model_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - training_start
        return training_time
    
    def _compute_loss(self, batch):
        """Compute loss for a batch (model-specific)."""
        raise NotImplementedError
    
    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        self.model.eval()
        predictions = []
        actuals = []
        
        inference_start = time.time()
        
        with torch.no_grad():
            for batch in test_loader:
                # Forward pass (model-specific)
                outputs, targets = self._forward_batch(batch)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(targets.cpu().numpy())
        
        inference_time = time.time() - inference_start
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Compute metrics
        mae, rmse, r2 = test_error(predictions, actuals)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals,
            'inference_time': inference_time
        }
    
    def _forward_batch(self, batch):
        """Forward pass for a batch (model-specific)."""
        raise NotImplementedError


class DSAFNetWrapper(GraphModelWrapper):
    """Wrapper for DSAFNet model."""
    
    def __init__(self, input_dim: int = 6, hidden_dim: int = 64, 
                 output_steps: int = 12, device: str = 'cuda'):
        if not DSAFNET_AVAILABLE:
            raise RuntimeError("DSAFNet is not available")
        
        model = DSAFNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_steps=output_steps,
            num_graphs=3
        )
        super().__init__(model, 'DSAFNet', device)
    
    def prepare_data_for_training(self, data_dict: Dict[str, Any], batch_size: int = 32):
        """Prepare data for DSAFNet."""
        # Extract sequences and labels
        sequences = data_dict['sequences_arr']
        labels = data_dict['labels_arr']
        n_train = data_dict['n_train']
        
        # Split train/test
        X_train = sequences[:n_train]
        y_train = labels[:n_train]
        X_test = sequences[n_train:]
        y_test = labels[n_train:]
        
        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def _compute_loss(self, batch):
        """Compute loss for DSAFNet."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # DSAFNet forward pass
        # Reshape if needed: [batch, seq, features] -> [batch, features, nodes, seq]
        if len(inputs.shape) == 3:
            # Assume inputs are [batch, seq, features]
            # Need to adapt to DSAFNet's expected input format
            batch_size, seq_len, feat_dim = inputs.shape
            # For simplicity, treat features as spatial dimension
            inputs = inputs.transpose(1, 2).unsqueeze(2)  # [batch, features, 1, seq]
        
        outputs = self.model(inputs)
        
        # Reshape outputs to match targets
        if len(outputs.shape) > len(targets.shape):
            outputs = outputs.squeeze()
        
        loss = self.criterion(outputs, targets)
        return loss
    
    def _forward_batch(self, batch):
        """Forward pass for DSAFNet."""
        inputs, targets = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        
        # Reshape if needed
        if len(inputs.shape) == 3:
            batch_size, seq_len, feat_dim = inputs.shape
            inputs = inputs.transpose(1, 2).unsqueeze(2)
        
        outputs = self.model(inputs)
        
        # Reshape outputs
        if len(outputs.shape) > len(targets.shape):
            outputs = outputs.squeeze()
        
        return outputs, targets


class STPNWrapper(GraphModelWrapper):
    """Wrapper for STPN model."""
    
    def __init__(self, num_nodes: int, in_channels: int, out_channels: int,
                 device: str = 'cuda'):
        if not STPN_AVAILABLE:
            raise RuntimeError("STPN is not available")
        
        model = STPNModel(
            num_nodes=num_nodes,
            in_channels=in_channels,
            out_channels=out_channels
        )
        super().__init__(model, 'STPN', device)
    
    def prepare_data_for_training(self, data_dict: Dict[str, Any], batch_size: int = 32):
        """Prepare data for STPN."""
        # STPN requires graph structure
        sequences = data_dict['sequences_arr']
        labels = data_dict['labels_arr']
        n_train = data_dict['n_train']
        edge_index = data_dict.get('edge_index', None)
        
        # Split train/test
        X_train = sequences[:n_train]
        y_train = labels[:n_train]
        X_test = sequences[n_train:]
        y_test = labels[n_train:]
        
        # Create graph data objects
        def create_graph_batch(X, y, edge_idx):
            graphs = []
            for i in range(len(X)):
                data = Data(
                    x=torch.FloatTensor(X[i]),
                    y=torch.FloatTensor(y[i]),
                    edge_index=torch.LongTensor(edge_idx) if edge_idx is not None else None
                )
                graphs.append(data)
            return graphs
        
        train_graphs = create_graph_batch(X_train, y_train, edge_index)
        test_graphs = create_graph_batch(X_test, y_test, edge_index)
        
        train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def _compute_loss(self, batch):
        """Compute loss for STPN."""
        if isinstance(batch, list):
            batch = Batch.from_data_list(batch)
        
        batch = batch.to(self.device)
        
        # STPN forward pass
        outputs = self.model(batch.x, batch.edge_index)
        
        loss = self.criterion(outputs, batch.y)
        return loss
    
    def _forward_batch(self, batch):
        """Forward pass for STPN."""
        if isinstance(batch, list):
            batch = Batch.from_data_list(batch)
        
        batch = batch.to(self.device)
        outputs = self.model(batch.x, batch.edge_index)
        
        return outputs, batch.y


# ============================================================================
# BENCHMARK INTEGRATION
# ============================================================================

def add_graph_models_to_benchmark(benchmark_runner):
    """Add graph-based models to an existing benchmark runner.
    
    Args:
        benchmark_runner: Instance of BenchmarkRunner from comprehensive_benchmark.py
    """
    
    print("\n[Benchmark] Adding graph-based models...")
    
    # Get data info
    data = benchmark_runner.data
    device = benchmark_runner.device
    args = benchmark_runner.args
    
    # Run DSAFNet if available
    if DSAFNET_AVAILABLE and not args.skip_graph:
        try:
            print("\n[Benchmark] Running DSAFNet...")
            
            # Create wrapper
            dsafnet_wrapper = DSAFNetWrapper(
                input_dim=data['sequences_arr'].shape[-1],
                hidden_dim=64,
                output_steps=args.pred_length,
                device=str(device)
            )
            
            # Prepare data
            train_loader, test_loader = dsafnet_wrapper.prepare_data_for_training(
                data, batch_size=args.batch_size
            )
            
            # Train
            optimizer = torch.optim.Adam(dsafnet_wrapper.model.parameters(), lr=args.lr)
            training_time = dsafnet_wrapper.train(train_loader, optimizer, epochs=args.epochs)
            
            # Evaluate
            results = dsafnet_wrapper.evaluate(test_loader)
            results['training_time'] = training_time
            
            benchmark_runner.results['DSAFNet'] = results
            
            print(f"[DSAFNet] MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}")
            print(f"[DSAFNet] Training: {training_time:.2f}s, Inference: {results['inference_time']:.2f}s")
            
        except Exception as e:
            print(f"[DSAFNet] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run STPN if available
    if STPN_AVAILABLE and not args.skip_graph:
        try:
            print("\n[Benchmark] Running STPN...")
            
            # Determine num_nodes from data
            num_nodes = data['sequences_arr'].shape[1] if len(data['sequences_arr'].shape) > 2 else 1
            
            # Create wrapper
            stpn_wrapper = STPNWrapper(
                num_nodes=num_nodes,
                in_channels=data['sequences_arr'].shape[-1],
                out_channels=args.pred_length,
                device=str(device)
            )
            
            # Prepare data
            train_loader, test_loader = stpn_wrapper.prepare_data_for_training(
                data, batch_size=args.batch_size
            )
            
            # Train
            optimizer = torch.optim.Adam(stpn_wrapper.model.parameters(), lr=args.lr)
            training_time = stpn_wrapper.train(train_loader, optimizer, epochs=args.epochs)
            
            # Evaluate
            results = stpn_wrapper.evaluate(test_loader)
            results['training_time'] = training_time
            
            benchmark_runner.results['STPN'] = results
            
            print(f"[STPN] MAE: {results['mae']:.4f}, RMSE: {results['rmse']:.4f}, R²: {results['r2']:.4f}")
            print(f"[STPN] Training: {training_time:.2f}s, Inference: {results['inference_time']:.2f}s")
            
        except Exception as e:
            print(f"[STPN] Failed: {e}")
            import traceback
            traceback.print_exc()
    
    return benchmark_runner


if __name__ == '__main__':
    print("This module should be imported by comprehensive_benchmark.py")
    print(f"DSAFNet available: {DSAFNET_AVAILABLE}")
    print(f"STPN available: {STPN_AVAILABLE}")
