"""
Visualization tools for training data and classification results.

This script provides functions to:
1. Visualize raw training data distribution
2. Visualize data after classification (predicted vs actual)
3. Compare features between classes
4. Show confusion patterns

Key Visualization Functions:
1. visualize_training_data() - Training Data Distribution
Shows:

Delay histogram by class (color-coded)
Class balance pie chart
Box plots of delays per class
Scatter plot of sample index vs delay
PCA projection of features (2D)
Statistics table with mean/std
2. visualize_classification_results() - Classification Analysis
Shows:

Confusion matrix (raw and normalized)
Scatter plot of correct vs incorrect predictions
Accuracy by class
Delay distribution by prediction outcome (TN, FP, FN, TP)
Metrics table (accuracy, precision, recall, F1)
3. visualize_regression_after_classification() - Stage 2/3 Results
Shows:

Predicted vs True scatter with perfect prediction line
Residual plot
Residual distribution
MAE by classification correctness
Box plots comparing true vs predicted
Metrics table (MAE, RMSE, R², MAPE)
4. visualize_three_stage_pipeline() - Complete Pipeline
Shows all three stages together with flow analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
import pandas as pd
from typing import Tuple, Optional, Dict


def visualize_training_data(
    X: np.ndarray,
    y_cls: np.ndarray,
    y_reg: np.ndarray,
    threshold: float = 5.0,
    sample_size: int = 1000,
    save_path: Optional[str] = None
):
    """
    Visualize training data distribution using multiple plots.
    
    Args:
        X: Feature matrix (samples, features) or (samples, nodes, features)
        y_cls: Classification labels (0 or 1)
        y_reg: Regression targets (actual delay values)
        threshold: Delay threshold for classification
        sample_size: Number of samples to plot (for large datasets)
        save_path: Path to save figure (optional)
    """
    # Flatten features if multi-dimensional (take mean across nodes)
    if X.ndim > 2:
        X_flat = X.reshape(X.shape[0], -1).mean(axis=1, keepdims=True)
    else:
        X_flat = X
    
    # Sample data if too large
    if len(X_flat) > sample_size:
        idx = np.random.choice(len(X_flat), sample_size, replace=False)
        X_flat = X_flat[idx]
        y_cls = y_cls[idx]
        y_reg = y_reg[idx]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training Data Distribution', fontsize=16, fontweight='bold')
    
    # 1. Delay distribution histogram
    ax = axes[0, 0]
    ax.hist(y_reg[y_cls == 0], bins=50, alpha=0.6, label=f'<{threshold} min', color='green', edgecolor='black')
    ax.hist(y_reg[y_cls == 1], bins=50, alpha=0.6, label=f'>={threshold} min', color='red', edgecolor='black')
    ax.axvline(threshold, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Delay (minutes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Delay Distribution by Class', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Class distribution pie chart
    ax = axes[0, 1]
    class_counts = [np.sum(y_cls == 0), np.sum(y_cls == 1)]
    colors = ['green', 'red']
    wedges, texts, autotexts = ax.pie(
        class_counts, 
        labels=[f'<{threshold} min', f'>={threshold} min'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 11}
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax.set_title(f'Class Balance (n={len(y_cls)})', fontsize=13, fontweight='bold')
    
    # 3. Delay box plot by class
    ax = axes[0, 2]
    data_to_plot = [y_reg[y_cls == 0], y_reg[y_cls == 1]]
    bp = ax.boxplot(data_to_plot, labels=[f'<{threshold}', f'>={threshold}'], 
                    patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('green')
    bp['boxes'][1].set_facecolor('red')
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Delay (minutes)', fontsize=12)
    ax.set_title('Delay Distribution by Class', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Scatter plot: Sample index vs Delay (colored by class)
    ax = axes[1, 0]
    scatter1 = ax.scatter(np.where(y_cls == 0)[0], y_reg[y_cls == 0], 
                         c='green', alpha=0.5, s=20, label=f'<{threshold} min')
    scatter2 = ax.scatter(np.where(y_cls == 1)[0], y_reg[y_cls == 1], 
                         c='red', alpha=0.5, s=20, label=f'>={threshold} min')
    ax.axhline(threshold, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Delay (minutes)', fontsize=12)
    ax.set_title('Delay vs Sample Index', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. PCA visualization (2D projection)
    ax = axes[1, 1]
    if X_flat.shape[1] > 1:
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_flat)
        scatter1 = ax.scatter(X_pca[y_cls == 0, 0], X_pca[y_cls == 0, 1], 
                             c='green', alpha=0.5, s=30, label=f'<{threshold} min')
        scatter2 = ax.scatter(X_pca[y_cls == 1, 0], X_pca[y_cls == 1, 1], 
                             c='red', alpha=0.5, s=30, label=f'>={threshold} min')
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
        ax.set_title('PCA Projection of Features', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'Not enough features\nfor PCA', 
               ha='center', va='center', fontsize=12)
    
    # 6. Statistics table
    ax = axes[1, 2]
    ax.axis('off')
    stats_data = [
        ['Total Samples', f'{len(y_cls)}'],
        ['Class 0 (< threshold)', f'{np.sum(y_cls == 0)} ({100*np.sum(y_cls == 0)/len(y_cls):.1f}%)'],
        ['Class 1 (>= threshold)', f'{np.sum(y_cls == 1)} ({100*np.sum(y_cls == 1)/len(y_cls):.1f}%)'],
        ['', ''],
        ['Mean Delay (Class 0)', f'{y_reg[y_cls == 0].mean():.2f} min'],
        ['Mean Delay (Class 1)', f'{y_reg[y_cls == 1].mean():.2f} min'],
        ['Std Delay (Class 0)', f'{y_reg[y_cls == 0].std():.2f} min'],
        ['Std Delay (Class 1)', f'{y_reg[y_cls == 1].std():.2f} min'],
    ]
    table = ax.table(cellText=stats_data, cellLoc='left', loc='center',
                    colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(len(stats_data)):
        if i == 3:  # Empty row
            continue
        cell = table[(i, 0)]
        cell.set_facecolor('#f0f0f0')
        cell = table[(i, 1)]
        cell.set_facecolor('#ffffff')
    ax.set_title('Dataset Statistics', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_classification_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_reg_true: np.ndarray,
    y_reg_pred: Optional[np.ndarray] = None,
    threshold: float = 5.0,
    save_path: Optional[str] = None
):
    """
    Visualize classification results with confusion matrix and prediction patterns.
    
    Args:
        y_true: True classification labels
        y_pred: Predicted classification labels
        y_reg_true: True delay values
        y_reg_pred: Predicted delay values (optional)
        threshold: Delay threshold
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Classification Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Confusion Matrix (heatmap)
    ax1 = fig.add_subplot(gs[0, 0])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
                xticklabels=[f'<{threshold}', f'>={threshold}'],
                yticklabels=[f'<{threshold}', f'>={threshold}'])
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=13, fontweight='bold')
    
    # 2. Normalized Confusion Matrix
    ax2 = fig.add_subplot(gs[0, 1])
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn', cbar=True, ax=ax2,
                xticklabels=[f'<{threshold}', f'>={threshold}'],
                yticklabels=[f'<{threshold}', f'>={threshold}'])
    ax2.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True', fontsize=12, fontweight='bold')
    ax2.set_title('Normalized Confusion Matrix', fontsize=13, fontweight='bold')
    
    # 3. Prediction Scatter: True vs Predicted (colored by correctness)
    ax3 = fig.add_subplot(gs[0, 2:])
    correct = (y_true == y_pred)
    incorrect = ~correct
    
    # Plot correct predictions
    ax3.scatter(np.arange(len(y_true))[correct], y_reg_true[correct], 
               c='green', alpha=0.6, s=30, label='Correct', marker='o')
    # Plot incorrect predictions
    ax3.scatter(np.arange(len(y_true))[incorrect], y_reg_true[incorrect], 
               c='red', alpha=0.6, s=50, label='Incorrect', marker='x')
    
    ax3.axhline(threshold, color='black', linestyle='--', linewidth=2, 
               label='Threshold', alpha=0.7)
    ax3.set_xlabel('Sample Index', fontsize=12)
    ax3.set_ylabel('True Delay (minutes)', fontsize=12)
    ax3.set_title('Classification Results: Correct vs Incorrect', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 4. Error Analysis by True Class
    ax4 = fig.add_subplot(gs[1, 0])
    
    # Calculate error rates
    class_0_correct = np.sum((y_true == 0) & (y_pred == 0))
    class_0_total = np.sum(y_true == 0)
    class_1_correct = np.sum((y_true == 1) & (y_pred == 1))
    class_1_total = np.sum(y_true == 1)
    
    accuracy_by_class = [
        class_0_correct / class_0_total * 100 if class_0_total > 0 else 0,
        class_1_correct / class_1_total * 100 if class_1_total > 0 else 0
    ]
    
    bars = ax4.bar([f'Class 0\n(<{threshold})', f'Class 1\n(>={threshold})'], 
                   accuracy_by_class, color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax4.set_ylabel('Accuracy (%)', fontsize=12)
    ax4.set_title('Accuracy by True Class', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, 105])
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 5. Delay Distribution by Prediction Outcome
    ax5 = fig.add_subplot(gs[1, 1])
    
    outcomes = {
        'True Neg': y_reg_true[(y_true == 0) & (y_pred == 0)],
        'False Pos': y_reg_true[(y_true == 0) & (y_pred == 1)],
        'False Neg': y_reg_true[(y_true == 1) & (y_pred == 0)],
        'True Pos': y_reg_true[(y_true == 1) & (y_pred == 1)]
    }
    
    data_to_plot = [v for v in outcomes.values() if len(v) > 0]
    labels = [k for k, v in outcomes.items() if len(v) > 0]
    colors_map = {'True Neg': 'lightgreen', 'False Pos': 'orange', 
                  'False Neg': 'pink', 'True Pos': 'lightcoral'}
    box_colors = [colors_map[label] for label in labels]
    
    bp = ax5.boxplot(data_to_plot, labels=labels, patch_artist=True, notch=True)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax5.axhline(threshold, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_ylabel('True Delay (minutes)', fontsize=12)
    ax5.set_title('Delay Distribution by Prediction Outcome', fontsize=13, fontweight='bold')
    ax5.tick_params(axis='x', rotation=45)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Metrics Table
    ax6 = fig.add_subplot(gs[1, 2:])
    ax6.axis('off')
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Accuracy', f'{accuracy_score(y_true, y_pred):.4f}'],
        ['Precision', f'{precision_score(y_true, y_pred, zero_division=0):.4f}'],
        ['Recall', f'{recall_score(y_true, y_pred, zero_division=0):.4f}'],
        ['F1-Score', f'{f1_score(y_true, y_pred, zero_division=0):.4f}'],
        ['', ''],
        ['True Negatives', f'{cm[0,0]}'],
        ['False Positives', f'{cm[0,1]}'],
        ['False Negatives', f'{cm[1,0]}'],
        ['True Positives', f'{cm[1,1]}'],
    ]
    
    table = ax6.table(cellText=metrics_data, cellLoc='center', loc='center',
                     colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style the table
    for i in range(len(metrics_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 5:  # Empty row
                cell.set_facecolor('#ffffff')
            else:
                if j == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
    
    ax6.set_title('Classification Metrics', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_regression_after_classification(
    y_reg_true: np.ndarray,
    y_reg_pred: np.ndarray,
    y_cls_true: np.ndarray,
    y_cls_pred: np.ndarray,
    threshold: float = 5.0,
    stage: str = "Stage 2",
    save_path: Optional[str] = None
):
    """
    Visualize regression results after classification (for Stage 2 or Stage 3).
    
    Args:
        y_reg_true: True delay values
        y_reg_pred: Predicted delay values
        y_cls_true: True classification labels
        y_cls_pred: Predicted classification labels
        threshold: Delay threshold
        stage: Stage name ("Stage 2" or "Stage 3")
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{stage}: Regression Results', fontsize=16, fontweight='bold')
    
    # 1. Predicted vs True (scatter plot)
    ax = axes[0, 0]
    ax.scatter(y_reg_true, y_reg_pred, alpha=0.5, s=30, c='blue', edgecolors='k', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_reg_true.min(), y_reg_pred.min())
    max_val = max(y_reg_true.max(), y_reg_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax.set_xlabel('True Delay (minutes)', fontsize=12)
    ax.set_ylabel('Predicted Delay (minutes)', fontsize=12)
    ax.set_title('Predicted vs True Delay', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Residual plot
    ax = axes[0, 1]
    residuals = y_reg_pred - y_reg_true
    ax.scatter(y_reg_true, residuals, alpha=0.5, s=30, c='purple', edgecolors='k', linewidth=0.5)
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('True Delay (minutes)', fontsize=12)
    ax.set_ylabel('Residual (Predicted - True)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. Residual distribution
    ax = axes[0, 2]
    ax.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residual (minutes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Residual Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Error by classification outcome
    ax = axes[1, 0]
    
    correct_cls = (y_cls_true == y_cls_pred)
    mae_correct = np.abs(residuals[correct_cls]).mean() if np.any(correct_cls) else 0
    mae_incorrect = np.abs(residuals[~correct_cls]).mean() if np.any(~correct_cls) else 0
    
    bars = ax.bar(['Correct\nClassification', 'Incorrect\nClassification'], 
                  [mae_correct, mae_incorrect], 
                  color=['green', 'red'], alpha=0.6, edgecolor='black')
    ax.set_ylabel('Mean Absolute Error (minutes)', fontsize=12)
    ax.set_title('Regression Error by Classification Outcome', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # 5. Box plot of true vs predicted
    ax = axes[1, 1]
    bp = ax.boxplot([y_reg_true, y_reg_pred], labels=['True', 'Predicted'], 
                    patch_artist=True, notch=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    for box in bp['boxes']:
        box.set_alpha(0.7)
    ax.set_ylabel('Delay (minutes)', fontsize=12)
    ax.set_title('True vs Predicted Delay Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Metrics table
    ax = axes[1, 2]
    ax.axis('off')
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    
    mae = mean_absolute_error(y_reg_true, y_reg_pred)
    rmse = np.sqrt(mean_squared_error(y_reg_true, y_reg_pred))
    r2 = r2_score(y_reg_true, y_reg_pred)
    mape = np.mean(np.abs((y_reg_true - y_reg_pred) / (y_reg_true + 1e-8))) * 100
    
    metrics_data = [
        ['Metric', 'Value'],
        ['MAE', f'{mae:.4f} min'],
        ['RMSE', f'{rmse:.4f} min'],
        ['R² Score', f'{r2:.4f}'],
        ['MAPE', f'{mape:.2f}%'],
        ['', ''],
        ['Mean True Delay', f'{y_reg_true.mean():.2f} min'],
        ['Mean Pred Delay', f'{y_reg_pred.mean():.2f} min'],
        ['Std True Delay', f'{y_reg_true.std():.2f} min'],
        ['Std Pred Delay', f'{y_reg_pred.std():.2f} min'],
    ]
    
    table = ax.table(cellText=metrics_data, cellLoc='center', loc='center',
                    colWidths=[0.5, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(len(metrics_data)):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            elif i == 5:
                cell.set_facecolor('#ffffff')
            else:
                if j == 0:
                    cell.set_facecolor('#f0f0f0')
                else:
                    cell.set_facecolor('#ffffff')
    
    ax.set_title('Regression Metrics', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def visualize_regression_timeseries(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Regression Results Over Time",
    xlabel: str = "Time (sample index)",
    ylabel: str = "Delay (minutes)",
    max_points: int = 2000,
    save_path: Optional[str] = None,
    channel_names: Tuple[str, str] = ("arrival", "departure"),
):
    """Plot true vs predicted delay over time (sample index) on a single graph.

    This is useful for visually inspecting how predictions track the real delay
    sequence across the evaluation set.

    Args:
        y_true: True delay values, shape (N,)
        y_pred: Predicted delay values, shape (N,)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        max_points: Downsample to at most this many points for readability/perf
        save_path: Path to save figure
    """
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        print("[VISUALIZATION] No regression samples to plot (empty arrays)")
        return

    # Normalize shapes.
    # Supported:
    # - (N,) / (N,1): single-channel
    # - (N,2): two-channel (arrival, departure)
    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    n = min(y_true_arr.shape[0], y_pred_arr.shape[0])
    c = min(y_true_arr.shape[1], y_pred_arr.shape[1])
    y_true_arr = y_true_arr[:n, :c]
    y_pred_arr = y_pred_arr[:n, :c]

    if n == 0 or c == 0:
        print("[VISUALIZATION] No regression samples to plot after alignment")
        return

    # Downsample consistently across channels.
    if max_points is not None and max_points > 0 and n > max_points:
        step = int(np.ceil(n / max_points))
        idx = np.arange(0, n, step)
    else:
        idx = np.arange(n)

    x = idx
    y_true_plot = y_true_arr[idx]
    y_pred_plot = y_pred_arr[idx]

    # Two-channel: requested two subplots.
    if c >= 2:
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(title, fontsize=14, fontweight="bold")

        for i, ax in enumerate(axes[:2]):
            ch_name = channel_names[i] if i < len(channel_names) else f"ch{i}"
            ax.plot(x, y_true_plot[:, i], label=f"True ({ch_name})", color="black", linewidth=2, alpha=0.85)
            ax.plot(x, y_pred_plot[:, i], label=f"Predicted ({ch_name})", color="tab:orange", linewidth=2, alpha=0.85)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(f"{ch_name}", fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel(xlabel, fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.figure(figsize=(14, 6))
        plt.plot(x, y_true_plot[:, 0], label="True Delay", color="black", linewidth=2, alpha=0.85)
        plt.plot(x, y_pred_plot[:, 0], label="Predicted Delay", color="tab:orange", linewidth=2, alpha=0.85)

        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def visualize_three_stage_pipeline(
    results_dict: Dict,
    threshold: float = 5.0,
    save_path: Optional[str] = None
):
    """
    Comprehensive visualization of the entire three-stage pipeline.
    
    Args:
        results_dict: Dictionary containing:
            - 'y_true_cls': True classification labels
            - 'y_pred_cls': Predicted classification labels (Stage 1)
            - 'y_true_reg': True delay values
            - 'y_pred_stage2': Stage 2 predictions (for high delays)
            - 'y_pred_stage3': Stage 3 predictions (for low delays)
            - 'final_predictions': Combined final predictions
        threshold: Delay threshold
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    fig.suptitle('Three-Stage Pipeline: Complete Analysis', fontsize=18, fontweight='bold')
    
    y_true_cls = results_dict['y_true_cls']
    y_pred_cls = results_dict['y_pred_cls']
    y_true_reg = results_dict['y_true_reg']
    final_pred = results_dict['final_predictions']
    
    # 1. Stage 1: Classification Performance
    ax1 = fig.add_subplot(gs[0, :2])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_cls, y_pred_cls)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax1,
                xticklabels=[f'<{threshold}', f'>={threshold}'],
                yticklabels=[f'<{threshold}', f'>={threshold}'])
    ax1.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True', fontsize=12, fontweight='bold')
    ax1.set_title('Stage 1: Classification Confusion Matrix', fontsize=14, fontweight='bold')
    
    # 2. Pipeline Flow (Sankey-style visualization)
    ax2 = fig.add_subplot(gs[0, 2:])
    
    # Calculate flow
    total = len(y_true_cls)
    true_class0 = np.sum(y_true_cls == 0)
    true_class1 = np.sum(y_true_cls == 1)
    pred_class0 = np.sum(y_pred_cls == 0)
    pred_class1 = np.sum(y_pred_cls == 1)
    
    flow_data = [
        ['True Class 0', true_class0],
        ['True Class 1', true_class1],
        ['Pred Class 0 → Stage 3', pred_class0],
        ['Pred Class 1 → Stage 2', pred_class1],
    ]
    
    bars = ax2.barh([x[0] for x in flow_data], [x[1] for x in flow_data], 
                    color=['lightgreen', 'lightcoral', 'lightblue', 'lightyellow'],
                    edgecolor='black', linewidth=2)
    ax2.set_xlabel('Count', fontsize=12)
    ax2.set_title('Pipeline Flow: Data Routing', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2.,
                f'{int(width)} ({width/total*100:.1f}%)',
                ha='left', va='center', fontsize=10, fontweight='bold')
    
    # 3. Stage 2: High Delay Regression
    mask_stage2 = (y_pred_cls == 1)
    if np.any(mask_stage2) and 'y_pred_stage2' in results_dict:
        ax3 = fig.add_subplot(gs[1, :2])
        y_true_s2 = y_true_reg[mask_stage2]
        y_pred_s2 = results_dict['y_pred_stage2'][mask_stage2]
        
        ax3.scatter(y_true_s2, y_pred_s2, alpha=0.6, s=40, c='red', edgecolors='k', linewidth=0.5)
        min_val = min(y_true_s2.min(), y_pred_s2.min())
        max_val = max(y_true_s2.max(), y_pred_s2.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect')
        
        from sklearn.metrics import mean_absolute_error, r2_score
        mae_s2 = mean_absolute_error(y_true_s2, y_pred_s2)
        r2_s2 = r2_score(y_true_s2, y_pred_s2)
        
        ax3.set_xlabel('True Delay (minutes)', fontsize=12)
        ax3.set_ylabel('Predicted Delay (minutes)', fontsize=12)
        ax3.set_title(f'Stage 2: High Delay Regression (n={len(y_true_s2)})\nMAE={mae_s2:.2f}, R²={r2_s2:.3f}', 
                     fontsize=13, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Stage 3: Low Delay Regression
    mask_stage3 = (y_pred_cls == 0)
    if np.any(mask_stage3) and 'y_pred_stage3' in results_dict:
        ax4 = fig.add_subplot(gs[1, 2:])
        y_true_s3 = y_true_reg[mask_stage3]
        y_pred_s3 = results_dict['y_pred_stage3'][mask_stage3]
        
        ax4.scatter(y_true_s3, y_pred_s3, alpha=0.6, s=40, c='green', edgecolors='k', linewidth=0.5)
        min_val = min(y_true_s3.min(), y_pred_s3.min())
        max_val = max(y_true_s3.max(), y_pred_s3.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect')
        
        mae_s3 = mean_absolute_error(y_true_s3, y_pred_s3)
        r2_s3 = r2_score(y_true_s3, y_pred_s3)
        
        ax4.set_xlabel('True Delay (minutes)', fontsize=12)
        ax4.set_ylabel('Predicted Delay (minutes)', fontsize=12)
        ax4.set_title(f'Stage 3: Low Delay Regression (n={len(y_true_s3)})\nMAE={mae_s3:.2f}, R²={r2_s3:.3f}', 
                     fontsize=13, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 5. Final Combined Results
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.scatter(y_true_reg, final_pred, alpha=0.5, s=40, 
               c=y_pred_cls, cmap='RdYlGn_r', edgecolors='k', linewidth=0.5)
    min_val = min(y_true_reg.min(), final_pred.min())
    max_val = max(y_true_reg.max(), final_pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'b--', linewidth=2, label='Perfect')
    
    from sklearn.metrics import mean_absolute_error, r2_score
    mae_final = mean_absolute_error(y_true_reg, final_pred)
    r2_final = r2_score(y_true_reg, final_pred)
    
    ax5.set_xlabel('True Delay (minutes)', fontsize=12)
    ax5.set_ylabel('Predicted Delay (minutes)', fontsize=12)
    ax5.set_title(f'Final Combined Results\nMAE={mae_final:.2f}, R²={r2_final:.3f}', 
                 fontsize=13, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Error Analysis
    ax6 = fig.add_subplot(gs[2, 2:])
    residuals = final_pred - y_true_reg
    
    # Group by classification outcome
    correct_cls = (y_true_cls == y_pred_cls)
    
    data_groups = [
        residuals[(y_true_cls == 0) & correct_cls],
        residuals[(y_true_cls == 0) & ~correct_cls],
        residuals[(y_true_cls == 1) & correct_cls],
        residuals[(y_true_cls == 1) & ~correct_cls],
    ]
    
    labels = [
        f'Class 0\nCorrect',
        f'Class 0\nWrong',
        f'Class 1\nCorrect',
        f'Class 1\nWrong'
    ]
    
    bp = ax6.boxplot([d for d in data_groups if len(d) > 0], 
                     labels=[l for l, d in zip(labels, data_groups) if len(d) > 0],
                     patch_artist=True, notch=True)
    
    colors = ['lightgreen', 'pink', 'lightcoral', 'orange']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax6.axhline(0, color='red', linestyle='--', linewidth=2)
    ax6.set_ylabel('Prediction Error (minutes)', fontsize=12)
    ax6.set_title('Error Distribution by Classification Outcome', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


# Example usage function
def example_usage():
    """
    Example of how to use the visualization functions.
    """
    print("=" * 60)
    print("VISUALIZATION EXAMPLE USAGE")
    print("=" * 60)
    
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 500
    
    # Simulate features (e.g., graph-level features)
    X = np.random.randn(n_samples, 10)
    
    # Simulate delays (with threshold at 5 minutes)
    threshold = 5.0
    y_reg = np.concatenate([
        np.random.exponential(3, n_samples // 2),  # Class 0: low delays
        np.random.exponential(15, n_samples // 2) + 5  # Class 1: high delays
    ])
    y_cls = (y_reg >= threshold).astype(int)
    
    # Simulate predictions with some noise
    y_pred_cls = y_cls.copy()
    flip_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_pred_cls[flip_indices] = 1 - y_pred_cls[flip_indices]
    
    y_pred_reg = y_reg + np.random.randn(n_samples) * 2
    
    print("\n1. Visualizing training data distribution...")
    visualize_training_data(X, y_cls, y_reg, threshold=threshold, 
                           save_path='training_data_visualization.png')
    
    print("\n2. Visualizing classification results...")
    visualize_classification_results(y_cls, y_pred_cls, y_reg, y_pred_reg, 
                                    threshold=threshold,
                                    save_path='classification_results.png')
    
    print("\n3. Visualizing regression after classification...")
    visualize_regression_after_classification(y_reg, y_pred_reg, y_cls, y_pred_cls,
                                             threshold=threshold, stage="Stage 2",
                                             save_path='regression_results.png')
    
    # For three-stage pipeline
    results_dict = {
        'y_true_cls': y_cls,
        'y_pred_cls': y_pred_cls,
        'y_true_reg': y_reg,
        'y_pred_stage2': y_pred_reg,
        'y_pred_stage3': y_pred_reg,
        'final_predictions': y_pred_reg
    }
    
    print("\n4. Visualizing three-stage pipeline...")
    visualize_three_stage_pipeline(results_dict, threshold=threshold,
                                  save_path='three_stage_pipeline.png')
    
    print("\n" + "=" * 60)
    print("Visualization complete! Check the generated PNG files.")
    print("=" * 60)


if __name__ == "__main__":
    example_usage()
