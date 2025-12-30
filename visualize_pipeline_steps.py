"""
Visualization script for the entire data processing pipeline.
Visualizes:
1. Raw data distribution
2. Train/Val/Test splits
3. Graph structure (Nodes & Edges)
4. Processed sequences (Features & Targets)
5. Node labels and class balance

Usage:
    python visualize_pipeline_steps.py --data_source cdata --delay_threshold 5.0
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import networkx as nx
import pandas as pd
from typing import Tuple, List, Optional

# Add current directory to path to import local modules
sys.path.insert(0, os.path.dirname(__file__))

from classifykat import load_flight_data, build_sequences, SequentialTwoStagePredictor
from classifykat_balanced import build_sequences_node_level
from torch_geometric.data import Data

def set_style():
    """Set plotting style."""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12

def visualize_raw_data(train_raw, val_raw, test_raw, threshold=5.0):
    """Visualize raw delay distributions."""
    print("\n[VISUALIZATION] 1. Raw Data Distribution")
    
    # Combine all raw data for overall distribution
    all_raw = np.concatenate([train_raw, val_raw, test_raw], axis=1)
    # Flatten: (nodes, time, 1) -> (nodes * time)
    flat_delays = all_raw.flatten()
    
    # Remove NaNs if any
    flat_delays = flat_delays[~np.isnan(flat_delays)]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Raw Flight Delay Data Analysis (Threshold={threshold} min)', fontsize=16, fontweight='bold')
    
    # 1. Overall Histogram (log scale)
    sns.histplot(flat_delays, bins=100, kde=False, ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_yscale('log')
    axes[0, 0].axvline(threshold, color='r', linestyle='--', label=f'Threshold ({threshold} min)')
    axes[0, 0].set_title('Overall Delay Distribution (Log Scale)')
    axes[0, 0].set_xlabel('Delay (minutes)')
    axes[0, 0].legend()
    
    # 2. Boxplot by Split
    data_splits = [
        train_raw.flatten()[~np.isnan(train_raw.flatten())],
        val_raw.flatten()[~np.isnan(val_raw.flatten())],
        test_raw.flatten()[~np.isnan(test_raw.flatten())]
    ]
    
    sns.boxplot(data=data_splits, ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_xticklabels(['Train', 'Val', 'Test'])
    axes[0, 1].set_title('Delay Distribution by Split')
    axes[0, 1].set_ylabel('Delay (minutes)')
    axes[0, 1].set_ylim(-20, 120)  # Zoom in on typical range
    
    # 3. Delayed vs Non-Delayed Counts
    delayed_mask = flat_delays >= threshold
    counts = [np.sum(~delayed_mask), np.sum(delayed_mask)]
    labels = [f'<{threshold} min', f'>={threshold} min']
    
    axes[1, 0].pie(counts, labels=labels, autopct='%1.1f%%', colors=['lightgreen', 'salmon'], startangle=90)
    axes[1, 0].set_title('Overall Class Balance')
    
    # 4. Time Series Sample (First Node)
    # Plot a small window of time for the first node
    sample_window = 200
    node_idx = 0
    time_series = all_raw[node_idx, :sample_window, 0]
    
    axes[1, 1].plot(time_series, marker='o', markersize=2, linestyle='-', linewidth=1)
    axes[1, 1].axhline(threshold, color='r', linestyle='--', alpha=0.7)
    axes[1, 1].set_title(f'Sample Time Series (Node {node_idx}, First {sample_window} steps)')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Delay (minutes)')
    
    plt.tight_layout()
    plt.savefig('viz_step1_raw_data.png')
    print("  ✓ Saved viz_step1_raw_data.png")
    plt.close()

def visualize_normalization_effects(raw_data, scaled_data, scaler):
    """Visualize data before/after normalization and after denormalization."""
    print("\n[VISUALIZATION] 1b. Normalization Effects")
    
    # Print scaler stats to confirm it's the correct one
    print(f"  Scaler Statistics -> Mean: {scaler.mean:.4f}, Std: {scaler.std:.4f}")
    
    # Take a sample: First node, first 200 time steps
    node_idx = 0
    time_steps = 200
    
    # Handle shapes: (nodes, time, 1)
    # Ensure we don't go out of bounds
    time_steps = min(time_steps, raw_data.shape[1])
    
    raw_sample = raw_data[node_idx, :time_steps, 0]
    scaled_sample = scaled_data[node_idx, :time_steps, 0]
    
    # Denormalize using the custom StandardScaler from classifykat
    # It performs element-wise operation: data * std + mean
    denorm_sample = scaler.inverse_transform(scaled_sample)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Normalization Check (Node {node_idx}, First {time_steps} steps)', fontsize=16, fontweight='bold')
    
    # 1. Raw
    axes[0].plot(raw_sample, label='Raw Data (Before)', color='blue')
    axes[0].set_title('1. Raw Data (Before Normalization)')
    axes[0].set_ylabel('Delay (min)')
    axes[0].legend()
    
    # 2. Scaled
    axes[1].plot(scaled_sample, label='Scaled Data (After)', color='green')
    axes[1].set_title('2. Scaled Data (After Normalization)')
    axes[1].set_ylabel('Normalized Value')
    axes[1].legend()
    
    # 3. Denormalized vs Raw
    axes[2].plot(raw_sample, label='Original Raw', color='blue', alpha=0.5, linewidth=3)
    axes[2].plot(denorm_sample, label='Denormalized (Reconstructed)', color='red', linestyle='--')
    axes[2].set_title('3. Denormalized Data (Reconstructed)')
    axes[2].set_ylabel('Delay (min)')
    axes[2].legend()
    
    # Add MSE text
    # Use nanmean to ignore NaNs in raw data
    mse = np.nanmean((raw_sample - denorm_sample)**2)
    axes[2].text(0.02, 0.9, f'Reconstruction MSE: {mse:.6f}', transform=axes[2].transAxes, 
                 bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('viz_step1b_normalization.png')
    print("  ✓ Saved viz_step1b_normalization.png")
    plt.close()

def visualize_graph_structure(edge_index_adj, edge_index_od, num_nodes):
    """Visualize the graph structure (Adjacency and OD)."""
    print("\n[VISUALIZATION] 2. Graph Structure")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Graph Structure Visualization', fontsize=16, fontweight='bold')
    
    # Helper to plot graph
    def plot_graph(edge_index, ax, title, color):
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))
        edges = edge_index.t().tolist()
        G.add_edges_from(edges)
        
        # Use spring layout for visualization
        pos = nx.spring_layout(G, seed=42, k=0.15)
        
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=50, node_color='lightgray')
        nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3, edge_color=color)
        
        # Calculate degree centrality
        degrees = [d for n, d in G.degree()]
        avg_degree = np.mean(degrees)
        
        ax.set_title(f'{title}\nNodes: {num_nodes}, Edges: {len(edges)}\nAvg Degree: {avg_degree:.2f}')
        ax.axis('off')

    # 1. Adjacency Graph
    plot_graph(edge_index_adj, axes[0], "Spatial Adjacency Graph", "blue")
    
    # 2. OD Graph
    plot_graph(edge_index_od, axes[1], "Origin-Destination (OD) Graph", "green")
    
    plt.tight_layout()
    plt.savefig('viz_step2_graph_structure.png')
    print("  ✓ Saved viz_step2_graph_structure.png")
    plt.close()

def visualize_processed_sequences(train_x, train_y_cls, train_y_reg, seq_len, horizon):
    """Visualize processed sequences (features and targets)."""
    print("\n[VISUALIZATION] 3. Processed Sequences")
    
    # train_x shape: (num_samples, num_nodes * seq_len * features) OR (num_samples, num_nodes, seq_len * features)
    # Let's check shape first
    is_node_level = (train_x.dim() == 3) # (samples, nodes, features)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Processed Model Inputs (Seq Len={seq_len}, Horizon={horizon})', fontsize=16, fontweight='bold')
    
    # 1. Feature Heatmap (First Sample)
    sample_idx = 0
    if is_node_level:
        # (nodes, features)
        features = train_x[sample_idx].cpu().numpy()
        sns.heatmap(features, ax=axes[0, 0], cmap='viridis', cbar=True)
        axes[0, 0].set_title(f'Input Features Heatmap (Sample {sample_idx})\nRows: Nodes, Cols: Features')
        axes[0, 0].set_xlabel('Features (Time Steps × Channels)')
        axes[0, 0].set_ylabel('Nodes')
    else:
        # (features,) - flattened
        features = train_x[sample_idx].cpu().numpy().reshape(-1, seq_len) # Just a guess at reshaping for viz
        sns.heatmap(features, ax=axes[0, 0], cmap='viridis', cbar=True)
        axes[0, 0].set_title(f'Input Features Heatmap (Sample {sample_idx})')
    
    # 2. Target Distribution (Regression)
    y_reg_flat = train_y_reg.flatten().cpu().numpy()
    sns.histplot(y_reg_flat, bins=50, ax=axes[0, 1], color='orange', kde=True)
    axes[0, 1].set_title('Target Regression Values Distribution (Normalized)')
    axes[0, 1].set_xlabel('Normalized Delay')
    
    # 3. Class Balance in Processed Data
    y_cls_flat = train_y_cls.flatten().cpu().numpy()
    # If node level, y_cls might be (samples, nodes, 1)
    # If graph level, y_cls might be (samples, 1)
    
    # Binarize if not already (sometimes it's float 0.0/1.0)
    y_cls_binary = (y_cls_flat >= 0.5).astype(int)
    
    counts = np.bincount(y_cls_binary)
    if len(counts) < 2:
        counts = np.pad(counts, (0, 2-len(counts)))
        
    sns.barplot(x=['Non-Delayed (0)', 'Delayed (1)'], y=counts, ax=axes[1, 0], palette=['lightgreen', 'salmon'])
    axes[1, 0].set_title('Class Balance in Training Set')
    axes[1, 0].set_ylabel('Count')
    for i, v in enumerate(counts):
        axes[1, 0].text(i, v, str(v), ha='center', va='bottom')
        
    # 4. Feature Correlations (Subset)
    # Take a small subset of features from first 100 samples
    if is_node_level:
        # Average over nodes to get (samples, features)
        subset_features = train_x[:100].mean(dim=1).cpu().numpy()
    else:
        subset_features = train_x[:100].cpu().numpy()
        
    # Only take first 20 features to keep plot readable
    subset_features = subset_features[:, :20]
    corr = np.corrcoef(subset_features.T)
    
    sns.heatmap(corr, ax=axes[1, 1], cmap='coolwarm', center=0)
    axes[1, 1].set_title('Feature Correlation Matrix (First 20 Features)')
    
    plt.tight_layout()
    plt.savefig('viz_step3_processed_sequences.png')
    print("  ✓ Saved viz_step3_processed_sequences.png")
    plt.close()

def visualize_node_labels(train_y_cls, num_nodes):
    """Visualize label distribution across nodes."""
    print("\n[VISUALIZATION] 4. Node Label Distribution")
    
    # Check if we have node-level labels
    # train_y_cls shape: (samples, nodes, 1) or (samples, 1)
    
    if train_y_cls.dim() < 2 or (train_y_cls.dim() == 2 and train_y_cls.shape[1] == 1):
        print("  ! Skipping node-level visualization (Graph-level labels detected)")
        return

    # Calculate delay rate per node
    # (samples, nodes, 1) -> (nodes,)
    delay_rates = train_y_cls.mean(dim=0).flatten().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plot of delay rate per node
    nodes = np.arange(len(delay_rates))
    sns.barplot(x=nodes, y=delay_rates, ax=ax, color='salmon')
    
    ax.set_title('Delay Rate by Node (Airport)')
    ax.set_xlabel('Node Index')
    ax.set_ylabel('Proportion of Delayed Samples')
    ax.set_ylim(0, 1.0)
    
    # Add average line
    avg_rate = np.mean(delay_rates)
    ax.axhline(avg_rate, color='blue', linestyle='--', label=f'Average: {avg_rate:.2%}')
    ax.legend()
    
    # If too many nodes, simplify x-axis
    if len(nodes) > 50:
        ax.set_xticks(nodes[::5])
        ax.set_xticklabels(nodes[::5])
    
    plt.tight_layout()
    plt.savefig('viz_step4_node_labels.png')
    print("  ✓ Saved viz_step4_node_labels.png")
    plt.close()

def visualize_model_data_flow(train_x, edge_indices, num_nodes, seq_len, feature_dim):
    """Visualize how data transforms as it passes through the model (Untrained)."""
    print("\n[VISUALIZATION] 5. Model Data Flow (Untrained)")
    
    # 1. Setup Model
    in_channels = seq_len * feature_dim
    out_channels = 1 # Delay dim
    
    model = SequentialTwoStagePredictor(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=32
    )
    model.eval()
    
    # 2. Prepare Single Batch
    # Take first 32 samples
    batch_size = 32
    if len(train_x) < batch_size:
        batch_size = len(train_x)
        
    batch_x = train_x[:batch_size] # (B, N, F)
    
    # Manual Batching for GATConv
    # GATConv expects (TotalNodes, Features)
    b_size, n_nodes, n_features = batch_x.shape
    
    # 1. Flatten X
    x_flat = batch_x.reshape(-1, n_features) # (B*N, F)
    
    # 2. Batch Edge Indices
    def batch_edge_index(edge_index, batch_size, num_nodes):
        edge_indices = []
        for i in range(batch_size):
            offset = i * num_nodes
            edge_indices.append(edge_index + offset)
        return torch.cat(edge_indices, dim=1)
        
    edge_index_adj_batched = batch_edge_index(edge_indices[0], b_size, n_nodes)
    edge_index_od_batched = batch_edge_index(edge_indices[1], b_size, n_nodes)
    edge_index_od_t_batched = batch_edge_index(edge_indices[2], b_size, n_nodes)
    
    data = Data(
        x=x_flat, 
        edge_index_adj=edge_index_adj_batched,
        edge_index_od=edge_index_od_batched,
        edge_index_od_t=edge_index_od_t_batched
    )
    
    # 3. Forward Pass
    with torch.no_grad():
        # Encoder
        embeddings_flat = model.encoder(data) # (B*N, Hidden)
        
        # Classifier
        cls_logits_flat = model.classifier(embeddings_flat) # (B*N, 1)
        cls_probs_flat = torch.sigmoid(cls_logits_flat)
        
        # Regressor
        reg_preds_flat = model.regressor(embeddings_flat) # (B*N, Out)
        
        # Reshape back for viz
        embeddings = embeddings_flat.reshape(b_size, n_nodes, -1)
        cls_probs = cls_probs_flat.reshape(b_size, n_nodes, 1)
        reg_preds = reg_preds_flat.reshape(b_size, n_nodes, -1)
        
    # 4. Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Data Flow (Untrained Weights)', fontsize=16, fontweight='bold')
    
    # Input
    # (Batch, Nodes, Features) -> Average over batch for viz
    input_avg = batch_x.mean(dim=0).numpy()
    sns.heatmap(input_avg[:, :50], ax=axes[0, 0], cmap='viridis')
    axes[0, 0].set_title(f'1. Model Input (Avg over Batch)\nShape: {batch_x.shape}')
    axes[0, 0].set_xlabel('Features (First 50)')
    axes[0, 0].set_ylabel('Nodes')
    
    # Embeddings
    emb_avg = embeddings.mean(dim=0).numpy()
    sns.heatmap(emb_avg, ax=axes[0, 1], cmap='magma')
    axes[0, 1].set_title(f'2. GAT Embeddings (Latent Space)\nShape: {embeddings.shape}')
    axes[0, 1].set_xlabel('Hidden Dimensions')
    
    # Classifier Output
    sns.histplot(cls_probs.flatten().numpy(), bins=30, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title(f'3. Classifier Probabilities (Untrained)\nShape: {cls_probs.shape}')
    axes[1, 0].set_xlabel('Probability of Delay')
    axes[1, 0].set_xlim(0, 1)
    
    # Regressor Output
    sns.histplot(reg_preds.flatten().numpy(), bins=30, ax=axes[1, 1], color='orange')
    axes[1, 1].set_title(f'4. Regressor Predictions (Untrained)\nShape: {reg_preds.shape}')
    axes[1, 1].set_xlabel('Predicted Normalized Delay')
    
    plt.tight_layout()
    plt.savefig('viz_step5_model_flow.png')
    print("  ✓ Saved viz_step5_model_flow.png")
    plt.close()

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Data Pipeline")
    parser.add_argument('--data_source', type=str, default='cdata', choices=['cdata', 'udata'])
    parser.add_argument('--seq_len', type=int, default=8)
    parser.add_argument('--horizons', type=int, nargs=1, default=[12])
    parser.add_argument('--delay_threshold', type=float, default=5.0)
    parser.add_argument('--weather_file', type=str, default='weather_cn.npy')
    parser.add_argument('--period_hours', type=int, default=24)
    parser.add_argument('--use_node_level', action='store_true', default=True)
    return parser.parse_args()

def main():
    set_style()
    args = parse_args()
    
    if args.data_source == 'udata':
        args.weather_file = 'weather2016_2021.npy'
        
    print("="*60)
    print("DATA PIPELINE VISUALIZATION")
    print("="*60)
    
    # 1. Load Raw Data
    print("\n[STEP 1] Loading Flight Data...")
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
    
    # Visualize Raw Data
    visualize_raw_data(train_raw, val_raw, test_raw, args.delay_threshold)
    
    # Visualize Normalization Effects
    visualize_normalization_effects(train_raw, train_delay_scaled, scaler)
    
    # Visualize Graph Structure
    visualize_graph_structure(edge_index_adj, edge_index_od, num_nodes)
    
    # 2. Build Sequences
    print("\n[STEP 2] Building Sequences...")
    horizons = sorted({h for h in args.horizons if h > 0})
    max_horizon = horizons[0]
    
    build_fn = build_sequences_node_level if args.use_node_level else build_sequences
    
    train_x, train_y_reg, train_y_cls = build_fn(
        train_inputs, train_delay_scaled, train_raw,
        args.seq_len, max_horizon, args.delay_threshold, horizons
    )
    
    # Visualize Processed Sequences
    visualize_processed_sequences(train_x, train_y_cls, train_y_reg, args.seq_len, max_horizon)
    
    # Visualize Node Labels
    visualize_node_labels(train_y_cls, num_nodes)
    
    # Visualize Model Data Flow
    edge_indices = (edge_index_adj, edge_index_od, edge_index_od_t)
    feature_dim = train_inputs.shape[2]
    visualize_model_data_flow(train_x, edge_indices, num_nodes, args.seq_len, feature_dim)
    
    print("\n" + "="*60)
    print("Visualization Complete! Check the generated .png files.")
    print("="*60)

if __name__ == "__main__":
    main()
