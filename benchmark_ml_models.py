"""Additional State-of-the-Art Models for Flight Delay Prediction

This module implements recent models commonly used in flight delay prediction
research based on literature review (2020-2026):

1. Random Forest Regressor
2. XGBoost
3. LightGBM
4. Gradient Boosting
5. Support Vector Regression (SVR)
6. Bi-LSTM (Bidirectional LSTM)
7. CNN-LSTM Hybrid
8. Attention-LSTM

References:
- Flight Delay Prediction: A Dissecting Review of Recent Studies Using Machine Learning
- Recent advances in spatio-temporal prediction models
"""

from __future__ import annotations

import time
from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Classical ML models
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.multioutput import MultiOutputRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[Warning] scikit-learn not available")
    SKLEARN_AVAILABLE = False

# Gradient boosting models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("[Warning] XGBoost not available")
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("[Warning] LightGBM not available")
    LIGHTGBM_AVAILABLE = False

from baseline_methods import test_error


# ============================================================================
# DEEP LEARNING MODELS
# ============================================================================

class BiLSTM(nn.Module):
    """Bidirectional LSTM for flight delay prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2, 
                 output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # Bidirectional doubles the hidden dimension
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, (h_n, c_n) = self.lstm(x)
        # Concatenate forward and backward hidden states
        h_combined = torch.cat((h_n[-2], h_n[-1]), dim=1)
        out = self.fc(self.dropout(h_combined))
        return out


class CNNLSTM(nn.Module):
    """CNN-LSTM hybrid architecture for temporal feature extraction and prediction."""
    
    def __init__(self, input_dim: int, seq_len: int, hidden_dim: int = 128, 
                 output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        
        # CNN for feature extraction
        self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(dropout)
        
        # Calculate LSTM input size
        lstm_input_size = 128
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(lstm_input_size, hidden_dim, 2, batch_first=True, dropout=dropout)
        
        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        # Reshape for Conv1d: [batch, features, seq_len]
        x = x.transpose(1, 2)
        
        # CNN layers
        x = torch.relu(self.conv1(x))
        x = self.dropout_conv(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout_conv(x)
        
        # Reshape back for LSTM: [batch, seq_len, features]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        out = self.fc(self.dropout(h_n[-1]))
        return out


class AttentionLSTM(nn.Module):
    """LSTM with attention mechanism for flight delay prediction."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 2,
                 output_dim: int = 1, dropout: float = 0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: [batch, seq_len, features]
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim]
        
        # Compute attention weights
        attn_weights = self.attention(lstm_out)  # [batch, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attn_weights * lstm_out, dim=1)  # [batch, hidden_dim]
        
        # Output
        out = self.fc(self.dropout(context))
        return out


# ============================================================================
# WRAPPER FOR TREE-BASED MODELS
# ============================================================================

class TreeModelWrapper:
    """Wrapper for scikit-learn tree-based models."""
    
    def __init__(self, model, model_name: str):
        self.model = model
        self.model_name = model_name
        self.is_fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """Train the model."""
        print(f"[{self.model_name}] Training...")
        start_time = time.time()
        
        # Flatten sequences for tree models
        X_flat = X_train.reshape(X_train.shape[0], -1)
        y_flat = y_train.mean(axis=1) if len(y_train.shape) > 1 else y_train
        
        self.model.fit(X_flat, y_flat.ravel())
        self.is_fitted = True
        
        training_time = time.time() - start_time
        print(f"[{self.model_name}] Training completed in {training_time:.2f}s")
        return training_time
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate the model."""
        if not self.is_fitted:
            raise RuntimeError(f"{self.model_name} has not been trained yet")
        
        print(f"[{self.model_name}] Evaluating...")
        start_time = time.time()
        
        # Flatten sequences
        X_flat = X_test.reshape(X_test.shape[0], -1)
        y_flat = y_test.mean(axis=1) if len(y_test.shape) > 1 else y_test
        
        # Predict
        y_pred = self.model.predict(X_flat)
        
        inference_time = time.time() - start_time
        
        # Reshape predictions to match targets
        if len(y_pred.shape) < len(y_flat.shape):
            y_pred = y_pred.reshape(-1, 1)
        
        # Compute metrics
        mae, rmse, r2 = test_error(y_pred, y_flat.reshape(-1, 1))
        
        print(f"[{self.model_name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y_flat,
            'inference_time': inference_time
        }


# ============================================================================
# WRAPPER FOR DEEP LEARNING MODELS
# ============================================================================

class DeepLearningWrapper:
    """Wrapper for PyTorch-based models."""
    
    def __init__(self, model, model_name: str, device: str = 'cuda'):
        self.model = model
        self.model_name = model_name
        self.device = device
        if hasattr(model, 'to'):
            self.model.to(device)
    
    def train(self, train_loader: DataLoader, optimizer, criterion, epochs: int = 50) -> float:
        """Train the model."""
        print(f"[{self.model_name}] Training for {epochs} epochs...")
        self.model.train()
        start_time = time.time()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_batches
                print(f"[{self.model_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        training_time = time.time() - start_time
        print(f"[{self.model_name}] Training completed in {training_time:.2f}s")
        return training_time
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        """Evaluate the model."""
        print(f"[{self.model_name}] Evaluating...")
        self.model.eval()
        predictions = []
        actuals = []
        
        start_time = time.time()
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())
        
        inference_time = time.time() - start_time
        
        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)
        
        # Compute metrics
        mae, rmse, r2 = test_error(predictions, actuals)
        
        print(f"[{self.model_name}] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'actuals': actuals,
            'inference_time': inference_time
        }


# ============================================================================
# BENCHMARK INTEGRATION
# ============================================================================

def add_ml_models_to_benchmark(benchmark_runner):
    """Add machine learning models to the benchmark.
    
    Args:
        benchmark_runner: Instance of BenchmarkRunner from comprehensive_benchmark.py
    """
    
    data = benchmark_runner.data
    args = benchmark_runner.args
    device = benchmark_runner.device
    
    # Prepare data
    X_train = data['sequences_arr'][:data['n_train']]
    y_train = data['labels_arr'][:data['n_train'], :, 0].mean(axis=1, keepdims=True)
    X_test = data['sequences_arr'][data['n_train']:]
    y_test = data['labels_arr'][data['n_train']:, :, 0].mean(axis=1, keepdims=True)
    
    # ========== Tree-based Models ==========
    if SKLEARN_AVAILABLE and not args.skip_classical:
        
        # Random Forest
        print("\n[Benchmark] Running Random Forest...")
        try:
            rf_model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                n_jobs=-1,
                random_state=42
            )
            rf_wrapper = TreeModelWrapper(rf_model, 'Random Forest')
            training_time = rf_wrapper.train(X_train, y_train)
            results = rf_wrapper.evaluate(X_test, y_test)
            results['training_time'] = training_time
            benchmark_runner.results['Random_Forest'] = results
        except Exception as e:
            print(f"[Random Forest] Failed: {e}")
        
        # Gradient Boosting
        if not args.quick_test:
            print("\n[Benchmark] Running Gradient Boosting...")
            try:
                gb_model = GradientBoostingRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                gb_wrapper = TreeModelWrapper(gb_model, 'Gradient Boosting')
                training_time = gb_wrapper.train(X_train, y_train)
                results = gb_wrapper.evaluate(X_test, y_test)
                results['training_time'] = training_time
                benchmark_runner.results['Gradient_Boosting'] = results
            except Exception as e:
                print(f"[Gradient Boosting] Failed: {e}")
    
    # XGBoost
    if XGBOOST_AVAILABLE and not args.skip_classical:
        print("\n[Benchmark] Running XGBoost...")
        try:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_train_flat = y_train.ravel()
            
            xgb_model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
            
            start_time = time.time()
            xgb_model.fit(X_train_flat, y_train_flat)
            training_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = xgb_model.predict(X_test_flat)
            inference_time = time.time() - start_time
            
            mae, rmse, r2 = test_error(y_pred.reshape(-1, 1), y_test)
            
            benchmark_runner.results['XGBoost'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'training_time': training_time,
                'inference_time': inference_time
            }
            print(f"[XGBoost] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        except Exception as e:
            print(f"[XGBoost] Failed: {e}")
    
    # LightGBM
    if LIGHTGBM_AVAILABLE and not args.skip_classical:
        print("\n[Benchmark] Running LightGBM...")
        try:
            X_train_flat = X_train.reshape(X_train.shape[0], -1)
            X_test_flat = X_test.reshape(X_test.shape[0], -1)
            y_train_flat = y_train.ravel()
            
            lgbm_model = lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            
            start_time = time.time()
            lgbm_model.fit(X_train_flat, y_train_flat)
            training_time = time.time() - start_time
            
            start_time = time.time()
            y_pred = lgbm_model.predict(X_test_flat)
            inference_time = time.time() - start_time
            
            mae, rmse, r2 = test_error(y_pred.reshape(-1, 1), y_test)
            
            benchmark_runner.results['LightGBM'] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'training_time': training_time,
                'inference_time': inference_time
            }
            print(f"[LightGBM] MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
        except Exception as e:
            print(f"[LightGBM] Failed: {e}")
    
    # ========== Deep Learning Models ==========
    if not args.skip_dl:
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.FloatTensor(y_test)
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        
        input_dim = X_train.shape[-1]
        seq_len = X_train.shape[1]
        
        # Bi-LSTM
        print("\n[Benchmark] Running Bi-LSTM...")
        try:
            bilstm = BiLSTM(input_dim, hidden_dim=128, num_layers=2, output_dim=1)
            bilstm_wrapper = DeepLearningWrapper(bilstm, 'Bi-LSTM', str(device))
            optimizer = torch.optim.Adam(bilstm.parameters(), lr=args.lr)
            criterion = nn.MSELoss()
            training_time = bilstm_wrapper.train(train_loader, optimizer, criterion, args.epochs)
            results = bilstm_wrapper.evaluate(test_loader)
            results['training_time'] = training_time
            benchmark_runner.results['BiLSTM'] = results
        except Exception as e:
            print(f"[Bi-LSTM] Failed: {e}")
        
        # CNN-LSTM
        if not args.quick_test:
            print("\n[Benchmark] Running CNN-LSTM...")
            try:
                cnnlstm = CNNLSTM(input_dim, seq_len, hidden_dim=128, output_dim=1)
                cnnlstm_wrapper = DeepLearningWrapper(cnnlstm, 'CNN-LSTM', str(device))
                optimizer = torch.optim.Adam(cnnlstm.parameters(), lr=args.lr)
                criterion = nn.MSELoss()
                training_time = cnnlstm_wrapper.train(train_loader, optimizer, criterion, args.epochs)
                results = cnnlstm_wrapper.evaluate(test_loader)
                results['training_time'] = training_time
                benchmark_runner.results['CNN_LSTM'] = results
            except Exception as e:
                print(f"[CNN-LSTM] Failed: {e}")
        
        # Attention-LSTM
        if not args.quick_test:
            print("\n[Benchmark] Running Attention-LSTM...")
            try:
                attnlstm = AttentionLSTM(input_dim, hidden_dim=128, num_layers=2, output_dim=1)
                attnlstm_wrapper = DeepLearningWrapper(attnlstm, 'Attention-LSTM', str(device))
                optimizer = torch.optim.Adam(attnlstm.parameters(), lr=args.lr)
                criterion = nn.MSELoss()
                training_time = attnlstm_wrapper.train(train_loader, optimizer, criterion, args.epochs)
                results = attnlstm_wrapper.evaluate(test_loader)
                results['training_time'] = training_time
                benchmark_runner.results['Attention_LSTM'] = results
            except Exception as e:
                print(f"[Attention-LSTM] Failed: {e}")
    
    return benchmark_runner


if __name__ == '__main__':
    print("Additional ML models module")
    print(f"scikit-learn available: {SKLEARN_AVAILABLE}")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"LightGBM available: {LIGHTGBM_AVAILABLE}")
