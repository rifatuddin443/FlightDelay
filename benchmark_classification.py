"""Benchmark: Flight Delay Classification (Stage 1)

Tests classification performance for predicting delayed vs non-delayed flights.
Evaluates Stage 1 (binary classification) only.

Models tested:
1. Majority Class (baseline)
2. Logistic Regression
3. LSTM Classifier
4. GRU Classifier
5. CNN-Opacus Stage 1

Usage:
    # Run all classification models
    python benchmark_classification.py --epochs 20
    
    # Quick test
    python benchmark_classification.py --epochs 10
    
    # Specific models
    python benchmark_classification.py --models lstm gru cnnopacus --epochs 20
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))

from classifykat import (
    build_sequences,
    load_flight_data,
    set_seed,
    classification_metrics,
)
from classifykat_balanced import build_sequences_node_level

# Try to import CNN-Opacus model
try:
    from cnnopacuskanMerge import HybridCNNKANGATPredictor
    CNN_OPACUS_AVAILABLE = True
except Exception as e:
    print(f"[Warning] CNN-Opacus not available: {e}")
    HybridCNNKANGATPredictor = None
    CNN_OPACUS_AVAILABLE = False

# Configure matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============================================================================
# CLASSIFICATION MODELS
# ============================================================================

class LSTMClassifier(nn.Module):
    """LSTM for binary classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        lstm_out, (h_n, _) = self.lstm(x)
        logits = self.fc(h_n[-1])
        return logits


class GRUClassifier(nn.Module):
    """GRU for binary classification."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, 2, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        gru_out, h_n = self.gru(x)
        logits = self.fc(h_n[-1])
        return logits


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray = None) -> Dict:
    """Compute comprehensive classification metrics."""
    # Ensure binary labels
    y_true_binary = (y_true > 0.5).astype(int).flatten()
    y_pred_binary = (y_pred > 0.5).astype(int).flatten()
    
    # Basic metrics
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average='binary', zero_division=0
    )
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
    
    # Additional metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    # ROC-AUC (if probabilities available)
    try:
        if y_prob is not None:
            roc_auc = roc_auc_score(y_true_binary, y_prob.flatten())
        else:
            roc_auc = roc_auc_score(y_true_binary, y_pred_binary)
    except:
        roc_auc = 0.0
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'balanced_accuracy': balanced_acc,
        'roc_auc': roc_auc,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }


def load_classification_data(args):
    """Load and prepare data for classification."""
    print(f"\n[Data] Loading from {args.data_dir}...")
    
    # Load flight data
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
    
    print(f"[Data] Loaded {num_nodes} nodes")
    print(f"[Data] Train: {train_inputs.shape}, Test: {test_inputs.shape}")
    
    # Build sequences
    max_horizon = args.out_len
    horizons = [max_horizon]
    
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.in_len, max_horizon, args.delay_threshold, horizons
    )
    test_x, test_y_reg, test_y_cls = build_fn(
        test_inputs, test_delay_scaled, test_raw,
        args.in_len, max_horizon, args.delay_threshold, horizons
    )
    
    print(f"[Data] Sequences - Train: {train_x.shape}, Test: {test_x.shape}")
    print(f"[Data] Classification labels - Train: {train_y_cls.shape}, Test: {test_y_cls.shape}")
    
    # For simple classifiers: use sample-level labels (majority vote)
    # Average over nodes and horizons, then binarize
    train_cls_simple = (train_y_cls.float().mean(dim=(1, 2)) > 0.5).float().unsqueeze(-1)  # [N, 1], binary 0/1
    test_cls_simple = (test_y_cls.float().mean(dim=(1, 2)) > 0.5).float().unsqueeze(-1)
    
    # Count class distribution
    train_delayed = train_cls_simple.sum().item()
    test_delayed = test_cls_simple.sum().item()
    print(f"[Data] Train delayed: {int(train_delayed)}/{len(train_cls_simple)} ({100*train_delayed/len(train_cls_simple):.1f}%)")
    print(f"[Data] Test delayed: {int(test_delayed)}/{len(test_cls_simple)} ({100*test_delayed/len(test_cls_simple):.1f}%)")
    
    # Flatten features for simple models
    train_x_flat = train_x.reshape(train_x.shape[0], -1).numpy()
    test_x_flat = test_x.reshape(test_x.shape[0], -1).numpy()
    
    # Reshape for LSTM/GRU: [samples, seq_len, features]
    feature_dim = train_inputs.shape[2]
    seq_len = args.in_len
    train_x_seq = train_x.reshape(train_x.shape[0], seq_len, -1).numpy()
    test_x_seq = test_x.reshape(test_x.shape[0], seq_len, -1).numpy()
    
    # For CNN-Opacus: aggregate over nodes [samples, nodes, seq_len*features] -> [samples, seq_len*features]
    train_x_cnn = train_x.mean(dim=1).numpy()  # Average over nodes
    test_x_cnn = test_x.mean(dim=1).numpy()
    
    return {
        'num_nodes': num_nodes,
        'edge_index_adj': edge_index_adj,
        'edge_index_od': edge_index_od,
        
        # Flat format for logistic regression
        'train_x_flat': train_x_flat,
        'test_x_flat': test_x_flat,
        
        # Sequential format for LSTM/GRU
        'train_x_seq': train_x_seq,
        'test_x_seq': test_x_seq,
        
        # CNN format (aggregated over nodes)
        'train_x_cnn': train_x_cnn,
        'test_x_cnn': test_x_cnn,
        
        # Full format for reference
        'train_x': train_x,
        'test_x': test_x,
        
        # Classification labels (sample-level)
        'train_y_cls': train_cls_simple.numpy(),
        'test_y_cls': test_cls_simple.numpy(),
        
        # Node-level labels for CNN-Opacus
        'train_y_cls_full': train_y_cls,
        'test_y_cls_full': test_y_cls,
    }


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class ClassificationBenchmark:
    """Benchmark runner for classification models."""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"classification_benchmark_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"\n{'='*80}")
        print(f"{'FLIGHT DELAY CLASSIFICATION BENCHMARK (STAGE 1)':^80}")
        print(f"{'='*80}")
        print(f"\nDevice: {self.device}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size}")
        print(f"Models to run: {', '.join(args.models)}")
        print(f"CNN-Opacus Available: {CNN_OPACUS_AVAILABLE}")
        print(f"Output: {self.output_dir}\n")
    
    def clear_gpu_memory(self):
        """Clear GPU memory between models."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def run_majority_class(self, data):
        """Majority class baseline."""
        if 'majority' not in self.args.models:
            return
        
        print("\n[1/5] Running Majority Class Baseline...")
        start = time.time()
        
        y_train = data['train_y_cls']
        y_test = data['test_y_cls']
        
        # Predict most common class
        majority_class = (y_train.mean() > 0.5).astype(float)
        y_pred = np.full_like(y_test, majority_class)
        
        elapsed = time.time() - start
        metrics = compute_classification_metrics(y_test, y_pred)
        metrics['training_time'] = 0
        metrics['inference_time'] = elapsed
        
        self.results['Majority_Class'] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        self.clear_gpu_memory()
    
    def run_logistic_regression(self, data):
        """Logistic regression baseline."""
        if 'logistic' not in self.args.models:
            return
        
        print("\n[2/5] Running Logistic Regression...")
        
        X_train = data['train_x_flat']
        y_train = data['train_y_cls'].flatten().astype(int)  # Convert to integer
        X_test = data['test_x_flat']
        y_test = data['test_y_cls']
        
        start = time.time()
        model = LogisticRegression(max_iter=1000, random_state=self.args.seed)
        model.fit(X_train, y_train)
        training_time = time.time() - start
        
        start = time.time()
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start
        
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        metrics['training_time'] = training_time
        metrics['inference_time'] = inference_time
        
        self.results['Logistic_Regression'] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
        self.clear_gpu_memory()
    
    def run_lstm(self, data):
        """LSTM classifier."""
        if 'lstm' not in self.args.models:
            return
        
        print("\n[3/5] Running LSTM Classifier...")
        
        X_train = data['train_x_seq']
        y_train = data['train_y_cls']
        X_test = data['test_x_seq']
        y_test = data['test_y_cls']
        
        model = LSTMClassifier(X_train.shape[-1], hidden_dim=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=self.args.batch_size, shuffle=True
        )
        
        # Train
        start = time.time()
        model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start
        
        # Evaluate
        model.eval()
        start = time.time()
        y_pred_list = []
        y_prob_list = []
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test)),
            batch_size=32, shuffle=False
        )
        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch).cpu().numpy()
                probs = torch.sigmoid(torch.FloatTensor(logits)).numpy()
                y_prob_list.append(probs)
                y_pred_list.append((probs > 0.5).astype(float))
        
        y_pred = np.vstack(y_pred_list)
        y_prob = np.vstack(y_prob_list)
        inference_time = time.time() - start
        
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        metrics['training_time'] = training_time
        metrics['inference_time'] = inference_time
        
        self.results['LSTM'] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
        
        del model, optimizer, train_loader, test_loader
        self.clear_gpu_memory()
    
    def run_gru(self, data):
        """GRU classifier."""
        if 'gru' not in self.args.models:
            return
        
        print("\n[4/5] Running GRU Classifier...")
        
        X_train = data['train_x_seq']
        y_train = data['train_y_cls']
        X_test = data['test_x_seq']
        y_test = data['test_y_cls']
        
        model = GRUClassifier(X_train.shape[-1], hidden_dim=64).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = nn.BCEWithLogitsLoss()
        
        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train)),
            batch_size=self.args.batch_size, shuffle=True
        )
        
        # Train
        start = time.time()
        model.train()
        for epoch in range(self.args.epochs):
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start
        
        # Evaluate
        model.eval()
        start = time.time()
        y_pred_list = []
        y_prob_list = []
        test_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_test)),
            batch_size=32, shuffle=False
        )
        with torch.no_grad():
            for (X_batch,) in test_loader:
                X_batch = X_batch.to(self.device)
                logits = model(X_batch).cpu().numpy()
                probs = torch.sigmoid(torch.FloatTensor(logits)).numpy()
                y_prob_list.append(probs)
                y_pred_list.append((probs > 0.5).astype(float))
        
        y_pred = np.vstack(y_pred_list)
        y_prob = np.vstack(y_prob_list)
        inference_time = time.time() - start
        
        metrics = compute_classification_metrics(y_test, y_pred, y_prob)
        metrics['training_time'] = training_time
        metrics['inference_time'] = inference_time
        
        self.results['GRU'] = metrics
        print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
        print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
        
        del model, optimizer, train_loader, test_loader
        self.clear_gpu_memory()
    
    def run_cnn_opacus(self, data):
        """CNN-Opacus Stage 1 (classification)."""
        if 'cnnopacus' not in self.args.models or not CNN_OPACUS_AVAILABLE:
            print("\n[CNN-Opacus] Skipped")
            return
        
        print("\n[5/5] Running CNN-Opacus Stage 1...")
        
        try:
            train_x_cnn = data['train_x_cnn']
            train_y_cls = data['train_y_cls']
            test_x_cnn = data['test_x_cnn']
            test_y_cls = data['test_y_cls']
            edge_index = data['edge_index_adj']
            
            # Create model - input should be [batch, seq_len*features]
            # train_x_cnn is already [samples, seq_len*features] after averaging over nodes
            in_channels = train_x_cnn.shape[1]  # seq_len * feature_dim
            
            model = HybridCNNKANGATPredictor(
                in_channels=in_channels,
                out_channels=1,
                hidden_channels=self.args.hidden_dim,
                regressor_extra_layer=False,
                seq_len=self.args.in_len,
            ).to(self.device)
            
            model.set_stage(1)  # Stage 1: Classification
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
            criterion = nn.BCEWithLogitsLoss()
            
            train_dataset = TensorDataset(torch.FloatTensor(train_x_cnn), torch.FloatTensor(train_y_cls))
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
            )
            
            # Train
            start = time.time()
            model.train()
            
            for epoch in range(self.args.epochs):
                total_loss = 0
                num_batches = 0
                
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    # Create dummy Data object
                    batch_data = Data(
                        x=batch_x,
                        edge_index=edge_index.to(self.device),
                    )
                    
                    optimizer.zero_grad()
                    
                    # Forward through classifier
                    hidden, logits = model.forward_classifier(batch_data)
                    loss = criterion(logits, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                
                if (epoch + 1) % 5 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"    Epoch {epoch+1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
            
            training_time = time.time() - start
            
            # Evaluate
            model.eval()
            start = time.time()
            
            y_pred_list = []
            y_prob_list = []
            test_dataset = TensorDataset(torch.FloatTensor(test_x_cnn))
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            with torch.no_grad():
                for (batch_x,) in test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_data = Data(x=batch_x, edge_index=edge_index.to(self.device))
                    
                    _, logits = model.forward_classifier(batch_data)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds = (probs > 0.5).astype(float)
                    
                    y_prob_list.append(probs)
                    y_pred_list.append(preds)
            
            y_pred = np.vstack(y_pred_list)
            y_prob = np.vstack(y_prob_list)
            inference_time = time.time() - start
            
            # Already sample-level since we averaged over nodes in data prep
            metrics = compute_classification_metrics(test_y_cls, y_pred, y_prob)
            metrics['training_time'] = training_time
            metrics['inference_time'] = inference_time
            
            self.results['CNN_Opacus'] = metrics
            print(f"    Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
                  f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}")
            print(f"    Training: {training_time:.1f}s, Inference: {inference_time:.2f}s")
            
            del model, optimizer, train_loader, test_loader
            self.clear_gpu_memory()
            
        except Exception as e:
            print(f"    CNN-Opacus failed: {e}")
            import traceback
            traceback.print_exc()
            self.clear_gpu_memory()
    
    def visualize_results(self):
        """Create comparison plots."""
        if len(self.results) == 0:
            return
        
        print("\n[Viz] Creating comparison plots...")
        
        models = list(self.results.keys())
        accuracy_vals = [self.results[m]['accuracy'] for m in models]
        f1_vals = [self.results[m]['f1'] for m in models]
        precision_vals = [self.results[m]['precision'] for m in models]
        recall_vals = [self.results[m]['recall'] for m in models]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].barh(models, accuracy_vals, color=sns.color_palette("husl", len(models)))
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_title('Classification Accuracy')
        axes[0, 0].set_xlim([0, 1])
        axes[0, 0].invert_yaxis()
        
        # F1 Score
        axes[0, 1].barh(models, f1_vals, color=sns.color_palette("husl", len(models)))
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_title('F1 Score')
        axes[0, 1].set_xlim([0, 1])
        axes[0, 1].invert_yaxis()
        
        # Precision
        axes[1, 0].barh(models, precision_vals, color=sns.color_palette("husl", len(models)))
        axes[1, 0].set_xlabel('Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlim([0, 1])
        axes[1, 0].invert_yaxis()
        
        # Recall
        axes[1, 1].barh(models, recall_vals, color=sns.color_palette("husl", len(models)))
        axes[1, 1].set_xlabel('Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlim([0, 1])
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, f'classification_comparison_{self.timestamp}.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"[Viz] Saved plot: {plot_path}")
        plt.close()
    
    def save_results(self):
        """Save results to CSV and JSON."""
        print("\n[Save] Saving results...")
        
        # CSV
        csv_path = os.path.join(self.output_dir, f'classification_results_{self.timestamp}.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Model', 'Accuracy', 'F1', 'Precision', 'Recall', 'Specificity', 
                           'Balanced_Acc', 'ROC_AUC', 'TP', 'TN', 'FP', 'FN', 
                           'Train_Time', 'Inference_Time'])
            
            for model, metrics in self.results.items():
                writer.writerow([
                    model,
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['specificity']:.4f}",
                    f"{metrics['balanced_accuracy']:.4f}",
                    f"{metrics['roc_auc']:.4f}",
                    metrics['tp'],
                    metrics['tn'],
                    metrics['fp'],
                    metrics['fn'],
                    f"{metrics['training_time']:.2f}",
                    f"{metrics['inference_time']:.4f}"
                ])
        print(f"[Save] CSV: {csv_path}")
        
        # JSON
        json_path = os.path.join(self.output_dir, f'classification_results_{self.timestamp}.json')
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"[Save] JSON: {json_path}")
        
        # Summary
        print(f"\n{'='*80}")
        print(f"{'CLASSIFICATION RESULTS SUMMARY':^80}")
        print(f"{'='*80}")
        print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10}")
        print(f"{'-'*80}")
        for model, metrics in sorted(self.results.items(), key=lambda x: x[1]['f1'], reverse=True):
            print(f"{model:<20} {metrics['accuracy']:<10.4f} {metrics['f1']:<10.4f} "
                  f"{metrics['precision']:<12.4f} {metrics['recall']:<10.4f}")
        print(f"{'='*80}\n")
    
    def run(self):
        """Run complete benchmark."""
        if self.args.seed is not None:
            set_seed(self.args.seed)
        
        # Load data
        data = load_classification_data(self.args)
        
        # Run models
        self.run_majority_class(data)
        self.run_logistic_regression(data)
        self.run_lstm(data)
        self.run_gru(data)
        self.run_cnn_opacus(data)
        
        # Visualize and save
        self.visualize_results()
        self.save_results()
        
        print("\n✅ Classification benchmark complete!\n")


def parse_args():
    parser = argparse.ArgumentParser(description='Flight Delay Classification Benchmark')
    
    # Models to run
    parser.add_argument('--models', nargs='+', 
                        default=['majority', 'logistic', 'lstm', 'gru', 'cnnopacus'],
                        choices=['majority', 'logistic', 'lstm', 'gru', 'cnnopacus'],
                        help='Which models to run (default: all)')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='cdata', help='Data directory')
    parser.add_argument('--data_source', type=str, default='cdata', help='Data source name')
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy', help='Weather file')
    parser.add_argument('--period_hours', type=int, default=2, help='Period in hours')
    
    # Sequence arguments
    parser.add_argument('--in_len', type=int, default=18, help='Input sequence length')
    parser.add_argument('--out_len', type=int, default=12, help='Output sequence length')
    parser.add_argument('--delay_threshold', type=float, default=15.0, help='Delay threshold for classification (minutes)')
    parser.add_argument('--use_node_level', action='store_true', help='Use node-level sequences')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse_args()
    benchmark = ClassificationBenchmark(args)
    benchmark.run()
