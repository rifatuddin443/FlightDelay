# Advanced Techniques for Severe Class Imbalance (97% Delayed)

## Problem Summary
Your flight delay dataset shows **97.62% delayed samples** even with different aggregation methods. This extreme imbalance makes classification trivial (model predicts "delayed" always → 97% accuracy but useless).

---

## ✅ ROOT CAUSE ANALYSIS

The 97% imbalance persists across all aggregation methods because:

1. **Temporal Correlation**: Sliding window with stride=1 creates highly overlapping samples
2. **Spatial Propagation**: Delays cascade through connected airports in graph
3. **Threshold Effect**: 5-minute threshold is too low for flight operations
4. **Data Filtering**: Preprocessing might filter out on-time samples
5. **Seasonal Bias**: Training period might be high-delay season

---

## 🛠️ SOLUTION TOOLKIT (8 Techniques)

### **TIER 1: Data-Level Fixes** ⭐⭐⭐ (Highest Priority)

#### 1. Temporal Sampling with Stride
**File**: `classifykat_advanced_balance.py::build_sequences_with_temporal_sampling()`

**Problem**: Stride=1 creates 99% correlated samples (window at t=0 vs t=1 are nearly identical)

**Solution**: Use stride=6-24 to skip hours and reduce redundancy

```python
from classifykat_advanced_balance import build_sequences_with_temporal_sampling

x_train, y_reg_train, y_cls_train = build_sequences_with_temporal_sampling(
    input_train, target_train, raw_train,
    seq_len=6, horizon=12, delay_threshold=5.0,
    target_horizons=[3, 6, 12],
    stride=12  # Skip 12 hours between samples
)
```

**Expected Impact**: Reduces samples by 12×, likely drops delayed% from 97% to 60-80%

---

#### 2. Class-Balanced Undersampling
**File**: `classifykat_advanced_balance.py::build_sequences_class_balanced()`

**Problem**: Too many delayed samples, model never learns "on-time" class

**Solution**: Keep all minority (on-time), undersample majority (delayed) to 40-50% balance

```python
from classifykat_advanced_balance import build_sequences_class_balanced

x_train, y_reg_train, y_cls_train = build_sequences_class_balanced(
    input_train, target_train, raw_train,
    seq_len=6, horizon=12, delay_threshold=5.0,
    target_horizons=[3, 6, 12],
    target_ratio=0.4  # 40% delayed, 60% on-time
)
```

**Expected Impact**: Forces 40/60 balance, model learns decision boundary

**⚠️ Warning**: After training on balanced data, you MUST tune threshold on validation set (see Tier 3)

---

#### 3. Hard Negative Mining
**File**: `classifykat_advanced_balance.py::build_sequences_with_hard_negatives()`

**Problem**: Easy positives (very delayed flights) dominate training

**Solution**: Keep all on-time + only marginally delayed flights (near threshold)

```python
from classifykat_advanced_balance import build_sequences_with_hard_negatives

x_train, y_reg_train, y_cls_train = build_sequences_with_hard_negatives(
    input_train, target_train, raw_train,
    seq_len=6, horizon=12, delay_threshold=5.0,
    target_horizons=[3, 6, 12],
    near_threshold_window=5.0  # Keep delays within [5, 10] minutes
)
```

**Expected Impact**: Focuses model on decision boundary, improves calibration

---

#### 4. Minority Class Augmentation
**File**: `classifykat_ensemble_calibration.py::augment_minority_class_temporal()`

**Problem**: Too few on-time samples for model to learn patterns

**Solution**: Create synthetic on-time samples via temporal jittering + Gaussian noise

```python
from classifykat_ensemble_calibration import augment_minority_class_temporal

x_train, y_reg_train, y_cls_train = augment_minority_class_temporal(
    x_train, y_reg_train, y_cls_train,
    augment_factor=5,  # 5× minority samples
    noise_std=0.05     # 5% Gaussian noise
)
```

**Expected Impact**: Balances dataset without discarding delayed samples

---

### **TIER 2: Loss Function Modifications** ⭐⭐

#### 5. Focal Loss
**File**: `classifykat_ensemble_calibration.py::FocalLoss`

**Problem**: BCEWithLogitsLoss treats all samples equally, wasted on easy examples

**Solution**: Focal Loss down-weights easy examples, focuses on hard ones

```python
from classifykat_ensemble_calibration import FocalLoss

# Replace BCEWithLogitsLoss with:
criterion_cls = FocalLoss(
    alpha=0.25,  # Weight for positive class (use 1 - minority_rate)
    gamma=2.0    # Focusing parameter (higher = more focus on hard)
)
```

**Expected Impact**: Better decision boundary learning, +5-10% recall

---

#### 6. Class-Balanced Loss
**File**: `classifykat_ensemble_calibration.py::ClassBalancedLoss`

**Problem**: Simple reweighting (pos_weight) insufficient for 97% imbalance

**Solution**: Effective number of samples reweighting (better than linear)

```python
from classifykat_ensemble_calibration import ClassBalancedLoss

n_ontime = (y_cls_train < 0.5).sum()
n_delayed = (y_cls_train >= 0.5).sum()

criterion_cls = ClassBalancedLoss(
    samples_per_class=[n_ontime, n_delayed],
    beta=0.9999,  # Decay factor (higher = more aggressive reweighting)
    loss_type='focal'  # Can use 'focal' or 'bce'
)
```

**Expected Impact**: Stronger than pos_weight, +10-15% recall

---

#### 7. Dynamic POS_WEIGHT Calculation
**File**: `classifykat_advanced_balance.py::compute_optimal_pos_weight()`

**Problem**: Manual pos_weight tuning is inefficient

**Solution**: Compute pos_weight based on desired recall

```python
from classifykat_advanced_balance import compute_optimal_pos_weight

# Automatically compute for 85% recall target
pos_weight = compute_optimal_pos_weight(y_cls_train, target_recall=0.85)

criterion_cls = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], device=device)
)
```

**Expected Impact**: Scientifically chosen pos_weight for recall target

---

### **TIER 3: Post-Training Calibration** ⭐⭐⭐

#### 8. Threshold Tuning + Temperature Scaling
**Files**: 
- `tune_threshold.py` (already created)
- `classifykat_ensemble_calibration.py::temperature_scaling()`

**Problem**: Model trained on imbalanced/balanced data is miscalibrated

**Solution**: 
1. Find optimal threshold on validation set
2. Apply temperature scaling to logits

```bash
# Step 1: Find optimal threshold
python tune_threshold.py \
    --model_path kan_gat_dp_proper.pth \
    --optimize recall_at_precision \
    --min_precision 0.90
```

```python
# Step 2: Temperature scaling (in evaluation code)
from classifykat_ensemble_calibration import temperature_scaling, compute_calibration_metrics

# Compute on validation set
optimal_T = temperature_scaling(val_logits, val_labels)

# Apply to test set
calibrated_test_logits = test_logits / optimal_T
calibrated_probs = torch.sigmoid(torch.tensor(calibrated_test_logits))
test_preds = (calibrated_probs >= tuned_threshold).float()
```

**Expected Impact**: +15-30% recall with maintained precision

---

## 🏆 RECOMMENDED 3-STAGE PIPELINE

### **Stage 1: Data Preparation** (Choose ONE strategy)

**Option A: Aggressive Balancing** (Best for learning)
```python
# Step 1: Temporal sampling
x, y_reg, y_cls = build_sequences_with_temporal_sampling(
    input_train, target_train, raw_train,
    seq_len=6, horizon=12, delay_threshold=5.0,
    target_horizons=[3, 6, 12],
    stride=12
)

# Step 2: Force balance
x, y_reg, y_cls = build_sequences_class_balanced(
    input_train, target_train, raw_train,
    seq_len=6, horizon=12, delay_threshold=5.0,
    target_horizons=[3, 6, 12],
    target_ratio=0.4
)

# Step 3: Augment minority
x, y_reg, y_cls = augment_minority_class_temporal(
    x, y_reg, y_cls,
    augment_factor=3,
    noise_std=0.05
)
```

**Option B: Conservative Approach** (Keep more data)
```python
# Step 1: Temporal sampling only
x, y_reg, y_cls = build_sequences_with_temporal_sampling(
    input_train, target_train, raw_train,
    seq_len=6, horizon=12, delay_threshold=5.0,
    target_horizons=[3, 6, 12],
    stride=6  # Less aggressive
)

# Step 2: Compute high pos_weight
pos_weight = compute_optimal_pos_weight(y_cls, target_recall=0.85)
```

---

### **Stage 2: Training with Better Loss**

```python
# Choose ONE:

# Option A: Focal Loss (recommended)
from classifykat_ensemble_calibration import FocalLoss
criterion_cls = FocalLoss(alpha=0.25, gamma=2.0)

# Option B: Class-Balanced Loss
from classifykat_ensemble_calibration import ClassBalancedLoss
n_ontime = (y_cls_train < 0.5).sum()
n_delayed = (y_cls_train >= 0.5).sum()
criterion_cls = ClassBalancedLoss(
    samples_per_class=[n_ontime, n_delayed],
    beta=0.9999,
    loss_type='focal'
)

# Option C: High POS_WEIGHT (simplest)
criterion_cls = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], device=device)
)
```

**Training Loop** (add MixUp if overfitting):
```python
for x, y_reg, y_cls in train_loader:
    # Optional: MixUp augmentation
    from classifykat_ensemble_calibration import apply_mixup_augmentation
    x, y_a, y_b, lam = apply_mixup_augmentation(x, y_cls, alpha=0.2)
    
    logits, reg_preds = model(x)
    
    # MixUp loss
    cls_loss = lam * criterion_cls(logits, y_a) + (1-lam) * criterion_cls(logits, y_b)
    reg_loss = criterion_reg(reg_preds, y_reg)
    
    total_loss = cls_loss + reg_loss
    total_loss.backward()
```

---

### **Stage 3: Post-Training Calibration**

```bash
# Step 1: Check calibration
python -c "
from classifykat_ensemble_calibration import compute_calibration_metrics
import numpy as np

val_logits = ...  # Load from validation inference
val_labels = ...

ece, _, _ = compute_calibration_metrics(val_logits, val_labels)
print(f'ECE: {ece:.4f} (lower is better, <0.05 is good)')
"

# Step 2: Temperature scaling
python -c "
from classifykat_ensemble_calibration import temperature_scaling

optimal_T = temperature_scaling(val_logits, val_labels)
# Use this T in test evaluation
"

# Step 3: Threshold tuning
python tune_threshold.py \
    --model_path MODEL.pth \
    --optimize recall_at_precision \
    --min_precision 0.90
```

---

## 📊 TESTING THE TECHNIQUES

Run the comparison script:
```bash
python test_advanced_balance.py
```

This will show:
- Impact of each technique on class distribution
- Number of samples after each transformation
- Recommended combinations

---

## 🎯 QUICK START (Copy-Paste Ready)

**Minimal Integration** (add to your training script):

```python
# At the top
from classifykat_advanced_balance import (
    build_sequences_with_temporal_sampling,
    build_sequences_class_balanced,
    compute_optimal_pos_weight,
)
from classifykat_ensemble_calibration import FocalLoss

# Replace build_sequences() calls with:

# For TRAINING set:
x_train, y_reg_train, y_cls_train = build_sequences_with_temporal_sampling(
    input_train, target_train, raw_train,
    seq_len, horizon, delay_threshold, target_horizons,
    stride=12
)

x_train, y_reg_train, y_cls_train = build_sequences_class_balanced(
    input_train, target_train, raw_train,
    seq_len, horizon, delay_threshold, target_horizons,
    target_ratio=0.4
)

# For VAL/TEST sets (NO balancing, use original):
x_val, y_reg_val, y_cls_val = build_sequences_with_temporal_sampling(
    input_val, target_val, raw_val,
    seq_len, horizon, delay_threshold, target_horizons,
    stride=1  # Keep all for accurate validation
)

# Replace loss function:
criterion_cls = FocalLoss(alpha=0.25, gamma=2.0)

# After training, run:
# python tune_threshold.py --model_path TRAINED_MODEL.pth --optimize recall_at_precision --min_precision 0.90
```

---

## 🔍 EXPECTED RESULTS

| Technique                      | Delayed% | Samples | Recall | Training Speed |
|-------------------------------|----------|---------|--------|---------------|
| Original (MAX aggregation)     | 97.6%    | 55,218  | 18.3%  | Baseline      |
| + Temporal (stride=12)         | ~70%     | 4,600   | ~50%   | 12× faster    |
| + Class Balance (40%)          | 40%      | 3,000   | ~75%   | 18× faster    |
| + Focal Loss                   | 40%      | 3,000   | ~80%   | 18× faster    |
| + Threshold Tuning             | 40%      | 3,000   | **85%**| 18× faster    |
| + Temperature Scaling          | 40%      | 3,000   | **87%**| 18× faster    |

---

## ❓ FAQ

**Q: Why not just adjust class_threshold?**
A: Threshold tuning ONLY works if model sees both classes during training. With 97% imbalance, model never learns "on-time" patterns. You need data-level fixes first.

**Q: Will this hurt precision?**
A: Yes, temporarily. But you can control the precision/recall trade-off via:
1. `target_ratio` in class balancing (lower = higher precision)
2. `--min_precision` in threshold tuning
3. Ensemble multiple models

**Q: Does this work with Differential Privacy?**
A: Yes! All techniques are compatible with DP-SGD. Smaller balanced datasets actually HELP privacy (less noise needed for same ε).

**Q: Which single technique has biggest impact?**
A: **Class-Balanced Undersampling** (Technique 2) → Forces model to learn both classes. Everything else is optimization.

---

## 📁 Files Created

1. **classifykat_advanced_balance.py** - Data-level techniques (1-4, 7)
2. **classifykat_ensemble_calibration.py** - Loss functions + calibration (5-6, 8)
3. **test_advanced_balance.py** - Comparison script
4. **tune_threshold.py** - Threshold optimization (already created)

---

## 🚀 NEXT STEPS

1. Run `python test_advanced_balance.py` to see impact of each technique
2. Choose Strategy: Aggressive (Option A) or Conservative (Option B)
3. Integrate into training script (copy-paste from Quick Start)
4. Train model with new data + loss function
5. Run threshold tuning: `python tune_threshold.py ...`
6. Evaluate on test set with tuned threshold
7. If still low recall → Try ensemble or increase augmentation

---

## 📞 TROUBLESHOOTING

**Problem: Still 90%+ delayed after temporal sampling**
- Try higher stride (24, 48)
- Check raw data distribution in original CSV
- Consider raising delay_threshold from 5 to 10-15 minutes

**Problem: Recall improved but precision dropped to <50%**
- Use `--min_precision 0.80` in threshold tuning
- Reduce `target_ratio` in class balancing (0.3 instead of 0.4)
- Try hard negative mining instead of full balancing

**Problem: Model overfits balanced training set**
- Add MixUp augmentation in training loop
- Increase DP noise (larger epsilon)
- Use more aggressive data augmentation (augment_factor=5-10)

**Problem: Training takes too long even with smaller dataset**
- Ghost Clipping already applied (from previous optimization)
- Consider reducing batch_size or simplifying model
- Differential Privacy is inherently 20-30× slower

---

Good luck! Start with **test_advanced_balance.py** to see which technique gives best balance.
