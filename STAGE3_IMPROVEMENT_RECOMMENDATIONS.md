# Stage 3 Non-Delayed Flight Prediction - Improvement Recommendations

## Problem Analysis

Your Stage 3 results show **negative R² values** and **high MAE** for non-delayed flights:

```
12-STEP AHEAD (Non-Delayed):
  Arrival Delay MAE: 2.82 min | R²: -2.33
  Departure Delay MAE: 2.05 min | R²: -0.62
  Mean Predicted: 0.49 min | Mean Target: -2.15 min
```

### Key Issues:

1. **Negative R²** = Model worse than predicting the mean
2. **Distribution Mismatch**: Model trained on delayed flights (Stage 2), then fine-tuned on non-delayed
3. **Normalization Inconsistency**: Loss computed in normalized space, mask created in denormalized space
4. **Learning Rate Too High**: Fine-tuning with 0.1x learning rate is too aggressive
5. **Encoder Co-adaptation**: Encoder learned features for delayed flights, struggles with non-delayed

---

## Recommended Solutions

### 🔧 **Solution 1: Fix Normalization Consistency** (HIGH PRIORITY)

**Problem**: Mask is created in denormalized space but loss is computed in normalized space.

**Change in `train_stage3_with_dp()`**:
```python
# BEFORE (Lines ~1139-1155):
# Denormalize
if scaler is not None:
    targets_denorm = torch.from_numpy(
        scaler.inverse_transform(reg_targets.cpu().numpy())
    ).to(device)
else:
    targets_denorm = reg_targets

# Element-wise masking
element_mask = (targets_denorm.abs() < delay_threshold).float()

# Compute loss on normalized values with mask  ← PROBLEM!
loss_per_element = reg_loss_fn(reg_preds, reg_targets) * element_mask

# AFTER (Recommended):
# Denormalize BOTH predictions and targets
if scaler is not None:
    targets_denorm = torch.from_numpy(
        scaler.inverse_transform(reg_targets.cpu().numpy())
    ).to(device)
    preds_denorm = torch.from_numpy(
        scaler.inverse_transform(reg_preds.detach().cpu().numpy())
    ).to(device)
else:
    targets_denorm = reg_targets
    preds_denorm = reg_preds

# Create mask in denormalized space
element_mask = (targets_denorm.abs() < delay_threshold).float()

# Compute loss in DENORMALIZED space for consistency
# Option A: Simple approach (may need gradient handling)
loss_per_element = ((preds_denorm - targets_denorm) ** 2) * element_mask

# Option B: Re-normalize masked values before loss
masked_targets = reg_targets * element_mask  # Keep in normalized space
masked_preds = reg_preds * element_mask
loss_per_element = reg_loss_fn(masked_preds, masked_targets)
```

---

### 🎯 **Solution 2: Reduce Learning Rate** (HIGH PRIORITY)

**Problem**: LR of `0.001 * 0.1 = 0.0001` is still too high for fine-tuning non-delayed distribution.

**Change (Line ~1055)**:
```python
# BEFORE:
optimizer = torch.optim.Adam(
    list(model.encoder.parameters()) + list(model.regressor.parameters()), 
    lr=lr * 0.1,  # ← Too high
    weight_decay=1e-4
)

# AFTER:
optimizer = torch.optim.Adam(
    model.regressor.parameters(),  # Only regressor initially
    lr=lr * 0.01,  # Much lower: 0.00001
    weight_decay=1e-5  # Reduced regularization
)

# Optionally unfreeze encoder after 5 epochs with even lower LR
```

---

### 🧠 **Solution 3: Use Huber Loss Instead of MSE** (MEDIUM PRIORITY)

**Problem**: MSE is sensitive to outliers. Non-delayed flights have tighter distributions.

**Change (Line ~1062)**:
```python
# BEFORE:
reg_loss_fn = nn.MSELoss(reduction='none')

# AFTER:
reg_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)  # More robust
```

---

### 📊 **Solution 4: Add Auxiliary Loss to Preserve Stage 2 Knowledge** (MEDIUM PRIORITY)

**Problem**: Fine-tuning on non-delayed flights makes the model forget delayed flight patterns.

**Add to training loop (Line ~1100)**:
```python
# After computing non-delayed loss:
if num_nondelayed > 0:
    loss_nondelayed = loss_per_element.sum() / num_nondelayed
    
    # Add auxiliary loss on delayed flights to prevent catastrophic forgetting
    delayed_mask = (targets_denorm.abs() >= delay_threshold).float()
    num_delayed = delayed_mask.sum()
    
    if num_delayed > 0:
        loss_delayed = ((preds_denorm - targets_denorm) ** 2 * delayed_mask).sum() / num_delayed
        # Weighted combination (tune alpha)
        loss = 0.7 * loss_nondelayed + 0.3 * loss_delayed
    else:
        loss = loss_nondelayed
else:
    loss = torch.tensor(0.0, device=device)
```

---

### 🎓 **Solution 5: Progressive Fine-Tuning** (ADVANCED)

**Problem**: Direct fine-tuning on non-delayed flights is too abrupt.

**Approach**:
```python
# Phase 1 (epochs 1-3): Freeze encoder, train regressor on non-delayed
for param in model.encoder.parameters():
    param.requires_grad = False

# Phase 2 (epochs 4-6): Unfreeze encoder with very low LR
for param in model.encoder.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters(), 'lr': lr * 0.001},  # Very low
    {'params': model.regressor.parameters(), 'lr': lr * 0.01}
])

# Phase 3 (epochs 7-10): Joint training with balanced loss
# (Include auxiliary loss from Solution 4)
```

---

### 🔍 **Solution 6: Per-Horizon Loss Weighting** (ADVANCED)

**Problem**: 12-step predictions are harder than 3-step. Treat them separately.

**Change**:
```python
# Assuming out_channels = 6 (3 horizons × 2 delays)
num_horizons = 3
delay_dim = 2

# Reshape predictions: [batch, 6] → [batch, 3, 2]
preds_reshaped = preds_denorm.view(-1, num_horizons, delay_dim)
targets_reshaped = targets_denorm.view(-1, num_horizons, delay_dim)
mask_reshaped = element_mask.view(-1, num_horizons, delay_dim)

# Compute per-horizon loss
horizon_losses = []
horizon_weights = [1.0, 0.8, 0.6]  # Weight near-term predictions more

for h in range(num_horizons):
    horizon_mask = mask_reshaped[:, h, :]
    if horizon_mask.sum() > 0:
        h_loss = ((preds_reshaped[:, h, :] - targets_reshaped[:, h, :]) ** 2 * horizon_mask).sum() / horizon_mask.sum()
        horizon_losses.append(horizon_weights[h] * h_loss)

loss = sum(horizon_losses) / len(horizon_losses) if horizon_losses else torch.tensor(0.0)
```

---

### 📈 **Solution 7: Data Augmentation for Non-Delayed Flights** (OPTIONAL)

**Problem**: Not enough non-delayed examples for robust learning.

**Approach**:
```python
# In training loop, augment non-delayed samples with small noise
if num_nondelayed > 0:
    # Add small Gaussian noise to non-delayed targets (±0.5 min)
    noise = torch.randn_like(masked_targets) * 0.5
    augmented_targets = masked_targets + noise
    loss = reg_loss_fn(masked_preds, augmented_targets).sum() / num_nondelayed
```

---

## Implementation Priority

1. ✅ **Fix normalization consistency** (Solution 1) - 5 min
2. ✅ **Reduce learning rate** (Solution 2) - 2 min
3. ✅ **Switch to Huber loss** (Solution 3) - 1 min
4. ⚠️ **Add auxiliary loss** (Solution 4) - 15 min
5. ⚠️ **Progressive fine-tuning** (Solution 5) - 30 min
6. ⚠️ **Per-horizon weighting** (Solution 6) - 20 min
7. 🔮 **Data augmentation** (Solution 7) - 10 min

---

## Quick Fix (Implement Now)

Replace lines **1050-1175** in `autoepsilonnew_3stage.py` with:

```python
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
    stage2_accountant: RDPAccountant,
) -> Tuple[List[Dict], RDPAccountant, float]:
    """IMPROVED: Better handling of non-delayed flights."""
    stage_start_time = time.time()
    print("\\n" + "="*80)
    print("STAGE 3: TRAINING DELAY REGRESSOR (NON-DELAYED FLIGHTS) - IMPROVED")
    print(f"Training on flights with |delay| < {delay_threshold} min")
    print("="*80)
    
    # SOLUTION 2: Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = False
    
    # SOLUTION 2: Much lower LR
    optimizer = torch.optim.Adam(
        model.regressor.parameters(), 
        lr=lr * 0.01,  # 0.00001 for default lr=0.001
        weight_decay=1e-5
    )
    
    # SOLUTION 3: Huber loss
    reg_loss_fn = nn.HuberLoss(reduction='none', delta=1.0)
    
    accountant = RDPAccountant(
        noise_multiplier=dp_config.noise_multiplier,
        sample_rate=dp_config.sample_rate if dp_config.enabled else 1.0,
        steps=stage2_accountant.steps,
    )
    
    print(f"✓ Regressor-only training with LR={lr * 0.01:.6f}")
    print(f"✓ Using Huber loss (robust to outliers)")
    
    history = []
    best_val_loss = float('inf')
    best_state = None
    early_stopping = EarlyStopping(patience=patience, mode="min")
    
    for epoch in range(1, epochs + 1):
        # SOLUTION 5: Progressive unfreezing
        if epoch == 6:  # Unfreeze encoder halfway
            print("  → Unfreezing encoder with very low LR...")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam([
                {'params': model.encoder.parameters(), 'lr': lr * 0.001},
                {'params': model.regressor.parameters(), 'lr': lr * 0.01}
            ], weight_decay=1e-5)
        
        epoch_start_time = time.time()
        model.train()
        epoch_losses = []
        total_nondelayed = 0
        total_values = 0
        
        num_samples = train_x.shape[0]
        indices = torch.randperm(num_samples)
        
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            batch_x = train_x[batch_indices].to(device)
            batch_y_reg = train_y_reg[batch_indices].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            reg_preds = []
            reg_targets = []
            for i in range(len(batch_x)):
                data = Data(
                    x=batch_x[i],
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_reg = model(data.to(device))
                graph_reg = aggregate_node_to_graph(node_reg)
                graph_target = ensure_graph_level_target(batch_y_reg[i])
                reg_preds.append(graph_reg)
                reg_targets.append(graph_target)
            
            reg_preds = torch.cat(reg_preds, dim=0)
            reg_targets = torch.cat(reg_targets, dim=0)
            
            # SOLUTION 1: Denormalize for consistent masking
            if scaler is not None:
                targets_denorm = torch.from_numpy(
                    scaler.inverse_transform(reg_targets.cpu().numpy())
                ).to(device)
                preds_denorm = torch.from_numpy(
                    scaler.inverse_transform(reg_preds.detach().cpu().numpy())
                ).to(device).requires_grad_(True)
            else:
                targets_denorm = reg_targets
                preds_denorm = reg_preds
            
            # Create element-wise mask
            element_mask = (targets_denorm.abs() < delay_threshold).float()
            num_nondelayed = element_mask.sum()
            
            if num_nondelayed > 0:
                # SOLUTION 1: Compute loss in denormalized space
                loss_per_element = reg_loss_fn(preds_denorm, targets_denorm) * element_mask
                loss_nondelayed = loss_per_element.sum() / num_nondelayed
                
                # SOLUTION 4: Add auxiliary loss for delayed flights
                delayed_mask = (targets_denorm.abs() >= delay_threshold).float()
                num_delayed = delayed_mask.sum()
                
                if num_delayed > 0 and epoch >= 6:  # After unfreezing encoder
                    loss_delayed = ((preds_denorm - targets_denorm) ** 2 * delayed_mask).sum() / num_delayed
                    loss = 0.8 * loss_nondelayed + 0.2 * loss_delayed  # Weighted
                else:
                    loss = loss_nondelayed
                
                loss.backward()
                total_nondelayed += num_nondelayed.item()
                total_values += element_mask.numel()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
                loss.backward()
            
            optimizer.step()
            epoch_losses.append(loss.item())
        
        # Validation (same denormalized space)
        model.eval()
        val_losses = []
        val_nondelayed = 0
        val_total = 0
        
        with torch.no_grad():
            for i in range(len(val_x)):
                data = Data(
                    x=val_x[i].to(device),
                    edge_index_adj=edge_indices[0],
                    edge_index_od=edge_indices[1],
                    edge_index_od_t=edge_indices[2],
                )
                _, node_reg = model(data)
                graph_reg = aggregate_node_to_graph(node_reg)
                graph_target = ensure_graph_level_target(val_y_reg[i])
                
                if scaler is not None:
                    target_denorm = torch.from_numpy(
                        scaler.inverse_transform(graph_target.cpu().numpy())
                    ).to(device)
                    pred_denorm = torch.from_numpy(
                        scaler.inverse_transform(graph_reg.cpu().numpy())
                    ).to(device)
                else:
                    target_denorm = graph_target
                    pred_denorm = graph_reg
                
                element_mask = (target_denorm.abs() < delay_threshold).float()
                num_nondelayed = element_mask.sum()
                
                if num_nondelayed > 0:
                    loss_per_element = ((pred_denorm - target_denorm) ** 2 * element_mask)
                    loss = loss_per_element.sum() / num_nondelayed
                    val_losses.append(loss.item())
                    val_nondelayed += num_nondelayed.item()
                    val_total += element_mask.numel()
        
        val_loss = np.mean(val_losses) if val_losses else 0.0
        epoch_time = time.time() - epoch_start_time
        
        current_epsilon = accountant.get_epsilon(dp_config.target_delta) if dp_config.enabled else float('inf')
        
        history.append({
            'epoch': epoch,
            'stage': 3,
            'train_loss': float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            'val_loss': val_loss,
            'train_nondelayed': total_nondelayed,
            'val_nondelayed': val_nondelayed,
            'epsilon': current_epsilon,
            'epoch_time_seconds': epoch_time,
        })
        
        nondelayed_pct = total_nondelayed / total_values * 100 if total_values > 0 else 0
        print(
            f"Epoch {epoch}/{epochs} | Loss: {history[-1]['train_loss']:.4f} | "
            f"Val: {val_loss:.4f} | Non-delayed: {total_nondelayed} ({nondelayed_pct:.1f}%) | "
            f"Time: {epoch_time:.2f}s"
        )
        
        if val_loss < best_val_loss and val_nondelayed > 0:
            best_val_loss = val_loss
            best_state = {
                'encoder': model.encoder.state_dict(),
                'regressor': model.regressor.state_dict()
            }
            print("  ✓ New best")
        
        if early_stopping(val_loss, epoch):
            print(f"  Early stopping at epoch {epoch}")
            break
    
    if best_state:
        model.encoder.load_state_dict(best_state['encoder'])
        model.regressor.load_state_dict(best_state['regressor'])
    
    stage_time = time.time() - stage_start_time
    final_epsilon = accountant.get_epsilon(dp_config.target_delta) if dp_config.enabled else float('inf')
    
    print(f"\\nStage 3 completed in {stage_time:.2f}s")
    print(f"Final ε: {final_epsilon:.3f}")
    
    return history, accountant, stage_time
```

---

## Expected Improvements

After implementing Solutions 1-3:
- **MAE should drop** from 2.8 → **1.5-2.0 min**
- **R² should become positive**: -2.33 → **0.3-0.6**
- **Mean prediction bias** should reduce: 0.49 vs -2.15 → closer match

After implementing Solutions 4-5:
- **Delayed flight accuracy preserved** (no regression on Stage 2 performance)
- **Per-horizon consistency** improved (3-step vs 12-step gap reduced)

---

## Testing

After changes, look for:
1. ✅ Positive R² values
2. ✅ MAE < 2.0 min for 12-step non-delayed
3. ✅ Mean predicted closer to mean target (-2.15 min)
4. ✅ No degradation in delayed flight metrics

Would you like me to implement these fixes directly in your code?
