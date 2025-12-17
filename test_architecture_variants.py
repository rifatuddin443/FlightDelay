"""Test architecture variants of the working threestagev3classsify.py model.

Tests modifications to hidden_channels, dropout rates, and fusion strategies.
Uses the PROVEN architecture that actually achieves results.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from classifykat import *
import time
import json
from typing import Dict

def test_variant(
    variant_name: str,
    hidden_channels: int,
    dropout_encoder: float,
    dropout_cls: float,
    epochs: int = 15,
) -> Dict[str, float]:
    """Test one architecture variant."""
    print(f"\n{'='*80}")
    print(f"Testing: {variant_name}")
    print(f"  hidden_channels={hidden_channels}, dropout_enc={dropout_encoder}, dropout_cls={dropout_cls}")
    print(f"{'='*80}")
    
    # Load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    (
        edge_index_adj, edge_index_od, edge_index_od_t,
        train_inputs, val_inputs, test_inputs,
        train_delay_scaled, val_delay_scaled, test_delay_scaled,
        train_raw, val_raw, test_raw,
        scaler, num_nodes,
    ) = load_flight_data('udata', 'weather2016_2021.npy', 24, 'udata')
    
    from classifykat_balanced import build_sequences_node_level
    
    train_x, train_y_reg, train_y_cls = build_sequences_node_level(
        train_inputs, train_delay_scaled, train_raw, 8, 3, 5.0, [3, 6, 12]
    )
    val_x, val_y_reg, val_y_cls = build_sequences_node_level(
        val_inputs, val_delay_scaled, val_raw, 8, 3, 5.0, [3, 6, 12]
    )
    
    # Subsample for speed
    train_x = train_x[:3000]
    train_y_cls = train_y_cls[:3000]
    val_x = val_x[:600]
    val_y_cls = val_y_cls[:600]
    
    feature_dim = train_inputs.shape[2]
    in_channels = 8 * feature_dim
    out_channels = 3  # 3 horizons
    
    # Modify dropout in model
    class ModifiedLightweightGATEncoder(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int, dropout: float):
            super().__init__()
            self.alpha_adj = nn.Parameter(torch.tensor(1.0))
            self.alpha_od = nn.Parameter(torch.tensor(1.0))
            self.alpha_od_t = nn.Parameter(torch.tensor(1.0))

            heads = 2
            self.gat_adj = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
            self.gat_od = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)
            self.gat_od_t = GATConv(in_channels, hidden_channels, heads=heads, concat=False, dropout=0.1)

            fusion_input_dim = hidden_channels * 3 + 3
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
            from kan import KAN
            self.fusion_kan = KAN(
                layers_hidden=[fusion_input_dim, hidden_channels, hidden_channels],
                grid_size=3,
                spline_order=2,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, data: Data) -> torch.Tensor:
            weights = F.softmax(torch.stack([self.alpha_adj, self.alpha_od, self.alpha_od_t]), dim=0)
            w_adj, w_od, w_od_t = weights

            x_adj = self.gat_adj(data.x, data.edge_index_adj)
            x_od = self.gat_od(data.x, data.edge_index_od)
            x_od_t = self.gat_od_t(data.x, data.edge_index_od_t)

            num_nodes = x_adj.size(0)
            scalars = torch.cat([
                w_adj.expand(num_nodes, 1),
                w_od.expand(num_nodes, 1),
                w_od_t.expand(num_nodes, 1),
            ], dim=1)

            x_concat = torch.cat([x_adj, x_od, x_od_t, scalars], dim=1)
            fused = F.relu(self.fusion_kan(x_concat))
            fused = self.dropout(fused)
            return fused
    
    class ModifiedPredictor(nn.Module):
        def __init__(self, encoder, in_channels: int, out_channels: int, hidden_channels: int, dropout_cls: float):
            super().__init__()
            self.encoder = encoder
            
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'efficient-kan', 'src'))
            from kan import KAN
            
            self.classifier = KAN(
                layers_hidden=[hidden_channels, hidden_channels // 2, 1],
                grid_size=3,
                spline_order=2,
            )
            
            self.regressor = KAN(
                layers_hidden=[hidden_channels, hidden_channels // 2, out_channels],
                grid_size=3,
                spline_order=2,
            )
            
            self.dropout_cls = nn.Dropout(dropout_cls)
            self.dropout_reg = nn.Dropout(dropout_cls)
        
        def forward_classifier(self, data: Data):
            hidden = self.encoder(data)
            hidden_dropped = self.dropout_cls(hidden)
            logits = self.classifier(hidden_dropped)
            return hidden, logits
        
        def forward_regressor(self, hidden: torch.Tensor):
            hidden_dropped = self.dropout_reg(hidden)
            return self.regressor(hidden_dropped)
        
        def forward(self, data: Data):
            hidden, logits = self.forward_classifier(data)
            reg_out = self.forward_regressor(hidden)
            return logits, reg_out
    
    # Build model
    encoder = ModifiedLightweightGATEncoder(in_channels, hidden_channels, dropout_encoder)
    model = ModifiedPredictor(encoder, in_channels, out_channels, hidden_channels, dropout_cls).to(device)
    
    # Train
    edge_index_adj = edge_index_adj.to(device)
    edge_index_od = edge_index_od.to(device)
    edge_index_od_t = edge_index_od_t.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    cls_loss_fn = nn.BCEWithLogitsLoss()
    
    start_time = time.time()
    best_f1 = 0.0
    best_metrics = {}
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        
        for i in range(len(train_x)):
            data = Data(
                x=train_x[i].to(device),
                edge_index_adj=edge_index_adj,
                edge_index_od=edge_index_od,
                edge_index_od_t=edge_index_od_t,
            )
            
            optimizer.zero_grad()
            _, logits = model.forward_classifier(data)
            loss = cls_loss_fn(logits, train_y_cls[i].to(device))
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Validation
        model.eval()
        val_logits_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_index_adj,
                    edge_index_od=edge_index_od,
                    edge_index_od_t=edge_index_od_t,
                )
                _, logits = model.forward_classifier(data)
                val_logits_list.append(torch.sigmoid(logits).cpu().numpy())
                val_targets_list.append(val_y_cls[i].numpy())
        
        val_probs = np.array(val_logits_list)
        val_targets = np.array(val_targets_list)
        val_metrics = classification_metrics(
            val_probs.reshape(-1, 1),
            val_targets.reshape(-1, 1),
        )
        
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_metrics = val_metrics
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {np.mean(epoch_losses):.4f} | "
                  f"Val F1: {val_metrics['f1']:.4f}")
    
    elapsed = time.time() - start_time
    
    print(f"\n[DONE] {variant_name} in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  F1: {best_metrics['f1']:.4f} | Precision: {best_metrics['precision']:.4f} | "
          f"Recall: {best_metrics['recall']:.4f}")
    
    return {
        'variant': variant_name,
        'f1': best_metrics['f1'],
        'precision': best_metrics['precision'],
        'recall': best_metrics['recall'],
        'accuracy': best_metrics['accuracy'],
        'time_seconds': elapsed,
        'hidden_channels': hidden_channels,
        'dropout_encoder': dropout_encoder,
        'dropout_cls': dropout_cls,
    }


def main():
    set_seed(42)
    
    print("="*80)
    print("ARCHITECTURE VARIANT COMPARISON")
    print("Testing proven KAN-GAT architecture with different hyperparameters")
    print("="*80)
    
    variants = [
        ("Current (h=64, d=0.3/0.2)", 64, 0.3, 0.2),
        ("Higher capacity (h=96)", 96, 0.3, 0.2),
        ("Lower dropout (d=0.1/0.1)", 64, 0.1, 0.1),
        ("Higher dropout (d=0.4/0.3)", 64, 0.4, 0.3),
        ("Largest (h=128, d=0.2)", 128, 0.2, 0.2),
    ]
    
    results = []
    for variant_name, hidden, dropout_enc, dropout_cls in variants:
        try:
            result = test_variant(variant_name, hidden, dropout_enc, dropout_cls)
            results.append(result)
        except Exception as e:
            print(f"ERROR testing {variant_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Variant':<35} {'F1':<10} {'Precision':<12} {'Recall':<12} {'Time(s)':<10}")
    print("-"*80)
    
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)
    for r in results_sorted:
        print(f"{r['variant']:<35} {r['f1']:<10.4f} {r['precision']:<12.4f} "
              f"{r['recall']:<12.4f} {r['time_seconds']:<10.1f}")
    
    if results_sorted:
        best = results_sorted[0]
        print("\n" + "="*80)
        print(f"WINNER: {best['variant']} with F1 = {best['f1']:.4f}")
        print(f"  Config: hidden={best['hidden_channels']}, "
              f"dropout_enc={best['dropout_encoder']}, dropout_cls={best['dropout_cls']}")
        print("="*80)
    
    # Save
    with open('variant_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to variant_comparison_results.json")


if __name__ == '__main__':
    main()
