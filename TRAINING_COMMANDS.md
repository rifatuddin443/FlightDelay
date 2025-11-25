# Quick Reference: Training Commands After np.abs() Fix

## 🚀 RECOMMENDED: Train with Node-Level Labels

```bash
# Basic training (recommended settings)
python classifykatdpnew_auto_epsilon.py --use_node_level --target_epsilon 5.0

# With custom settings
python classifykatdpnew_auto_epsilon.py \
    --use_node_level \
    --target_epsilon 8.0 \
    --delay_threshold 5.0 \
    --batch_size 32 \
    --stage1_epochs 20 \
    --stage2_epochs 20 \
    --lr 0.001 \
    --max_grad_norm 1.5 \
    --model_path kan_gat_node_level_fixed.pth

# Higher delay threshold (15 min = "significant delay")
python classifykatdpnew_auto_epsilon.py \
    --use_node_level \
    --delay_threshold 15.0 \
    --target_epsilon 5.0 \
    --model_path kan_gat_node_15min.pth
```

---

## 📊 After Training: Tune Classification Threshold

```bash
# Find best F1 score
python tune_threshold.py \
    --model_path kan_gat_node_level_fixed.pth \
    --optimize f1

# Find best recall at 90% precision
python tune_threshold.py \
    --model_path kan_gat_node_level_fixed.pth \
    --optimize recall_at_precision \
    --min_precision 0.90

# Find best recall at 95% precision
python tune_threshold.py \
    --model_path kan_gat_node_level_fixed.pth \
    --optimize recall_at_precision \
    --min_precision 0.95
```

---

## 🔍 Diagnostic Commands

```bash
# Check actual class distributions
python diagnose_imbalance.py

# Verify np.abs() fix
python test_abs_fix.py

# Test advanced balancing techniques
python test_advanced_balance.py

# Compare labeling strategies
python test_class_balance.py
```

---

## 🎯 Different Training Strategies

### Strategy 1: Node-Level (Best Balance)
```bash
python classifykatdpnew_auto_epsilon.py --use_node_level --target_epsilon 5.0
# Result: ~26% delayed, per-airport predictions
```

### Strategy 2: Graph-Level (Original)
```bash
python classifykatdpnew_auto_epsilon.py --target_epsilon 5.0
# Result: ~65% delayed, network-wide predictions
```

### Strategy 3: Higher Threshold
```bash
python classifykatdpnew_auto_epsilon.py --delay_threshold 15.0 --target_epsilon 5.0
# Result: Focus on significant delays (15+ min)
```

### Strategy 4: More Privacy Budget
```bash
python classifykatdpnew_auto_epsilon.py --use_node_level --target_epsilon 10.0
# Result: Less noise, better model quality
```

### Strategy 5: Smaller Batch (More Privacy)
```bash
python classifykatdpnew_auto_epsilon.py --use_node_level --batch_size 16 --target_epsilon 5.0
# Result: Tighter privacy guarantee
```

---

## 📈 Expected Results

### Before Fix (old labels with np.abs):
- Train: 97% delayed (wrong!)
- Precision: 1.0, Recall: 0.18, F1: 0.31

### After Fix - Graph-Level:
- Train: ~65% delayed
- Expected: Precision 0.80-0.90, Recall 0.60-0.75, F1 0.70-0.80

### After Fix - Node-Level (Recommended):
- Train: ~26% delayed (varies by split)
- Expected: Precision 0.85-0.95, Recall 0.70-0.85, F1 0.75-0.90

---

## 🛠️ Troubleshooting

**Low recall after retraining?**
```bash
# 1. Tune threshold
python tune_threshold.py --model_path MODEL.pth --optimize recall_at_precision --min_precision 0.90

# 2. Try higher epsilon (less noise)
python classifykatdpnew_auto_epsilon.py --use_node_level --target_epsilon 10.0

# 3. Increase batch size (less noise per sample)
python classifykatdpnew_auto_epsilon.py --use_node_level --batch_size 64 --target_epsilon 5.0
```

**Still imbalanced?**
```bash
# Increase delay threshold
python classifykatdpnew_auto_epsilon.py --use_node_level --delay_threshold 15.0

# Use advanced balancing (see classifykat_advanced_balance.py)
```

**Training too slow?**
```bash
# Already optimized with Ghost Clipping
# But can reduce batch size for faster epochs
python classifykatdpnew_auto_epsilon.py --use_node_level --batch_size 16 --target_epsilon 5.0
```

---

## 📝 Quick Checklist

Before training:
- [ ] Verified np.abs() fix with `python test_abs_fix.py`
- [ ] Checked data distribution with `python diagnose_imbalance.py`
- [ ] Decided: node-level or graph-level?
- [ ] Chosen: delay_threshold (5, 10, or 15 min)?

After training:
- [ ] Run threshold tuning: `python tune_threshold.py ...`
- [ ] Note best threshold value
- [ ] Re-evaluate with tuned threshold
- [ ] Check precision/recall/F1 metrics

---

## 💡 Pro Tips

1. **Always use `--use_node_level`** for better balance and useful predictions
2. **Start with epsilon=5.0**, increase if model quality poor
3. **Always tune threshold** after training (never use default 0.5)
4. **Consider threshold=15.0** for operational "significant delay"
5. **Monitor epsilon** during training to ensure privacy budget maintained
