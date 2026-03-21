# Weather Data Impact Analysis - Complete Workflow

## Quick Summary

✓ **Weather IS currently being used** in the model pipeline  
The baseline `stacked_gru_transformer.py` already incorporates weather data along with delay features.

This workflow provides tools to:
1. **Quantify** the weather contribution to model performance
2. **Compare** models with/without weather
3. **Visualize** results and make recommendations

---

## File Overview

### Core Scripts

| File | Purpose |
|------|---------|
| `stacked_gru_transformer.py` | **Baseline**: Trains model WITH weather (existing) |
| `stacked_gru_transformer_weather_comparison.py` | **NEW**: Trains models WITH & WITHOUT weather side-by-side |
| `analyze_weather_comparison.py` | **NEW**: Analyzes comparison results & generates reports |

### Documentation

| File | Content |
|------|---------|
| `WEATHER_USAGE_GUIDE.md` | Detailed explanation of weather data integration |
| `README_WEATHER_WORKFLOW.md` | This file - complete workflow guide |

---

## Running the Comparison

### Step 1: Run Weather Comparison Experiment
```bash
# Train both variants (with & without weather)
python stacked_gru_transformer_weather_comparison.py \
    --epochs 50 \
    --classifier TSiTPlus \
    --batch_size 16
```

**Typical output directory**: `weather_comparison_20260321_120000/`

**This trains 2 models:**
- Model 1: Uses `[delay_features + weather_features]`
- Model 2: Uses `[delay_features only]`

### Step 2: Analyze Results
```bash
# Automatic analysis & recommendation
python analyze_weather_comparison.py weather_comparison_20260321_120000/

# With visualization (requires matplotlib)
python analyze_weather_comparison.py weather_comparison_20260321_120000/ --plot
```

---

## Understanding the Data

### Data Feature Stack

In the current pipeline, each node at each timestep has these features:

```
┌─────────────────────────────────────────────┐
│ delay_features (2)                          │
│ • Arrival_delay_t                           │
│ • Departure_delay_t                         │
├─────────────────────────────────────────────┤
│ weather_features (N)                        │
│ • Temperature, Humidity, Wind, Precip, etc. │
└─────────────────────────────────────────────┘
```

### Input Dimension Changes

- **WITH weather**: `c_in = 2 + weather_dim`
  - Example: `c_in = 2 + 5 = 7` if weather has 5 features
  
- **WITHOUT weather**: `c_in = 2`
  - Only delay features

The comparison experiments with both configs using identical architecture parameters.

---

## Example Results Interpretation

### Scenario 1: Positive Weather Impact
```
Classifier: TSiTPlus
  F1:  0.7234 → 0.7456  (+3.07%)  ← F1 of (without) to (with)
  Acc: 0.8123 → 0.8301  (+2.19%)

  Parameters (without):  145,230
  Parameters (with):     162,840  (+12.2%)

  Training time:
    Without weather: 342.5s
    With weather:    361.2s     (+5.5%)
```

**Recommendation:** ✓ KEEP WEATHER\
The 3% F1 improvement justifies the 12% parameter increase and minor speed overhead.

---

### Scenario 2: Negligible Impact
```
Classifier: ConvTranPlus
  F1:  0.7450 → 0.7461  (+0.15%)
  Acc: 0.8300 → 0.8305  (+0.06%)
```

**Recommendation:** ≈ OPTIONAL\
Weather provides minimal benefit. Can keep (if readily available) or remove (for simplicity).

---

### Scenario 3: Negative Impact
```
Classifier: TSiTPlus
  F1:  0.7456 → 0.7234  (-3.07%)
  Acc: 0.8301 → 0.8123  (-2.19%)
```

**Recommendation:** ✗ INVESTIGATE\
- Check weather data quality (missing values, outliers)
- Try alternative preprocessing/scaling
- Verify weather_cn.npy file is not corrupted
- Consider weather-independent features sufficient

---

## Output Files

The comparison script generates:

```
weather_comparison_20260321_120000/
├── WEATHER_COMPARISON_SUMMARY.csv          ← Main results table
├── TSiTPlus_WITH_weather_metrics.csv        ← Detailed metrics
├── TSiTPlus_NO_weather_metrics.csv
├── TSiTPlus_WITH_weather_best.pth           ← Model checkpoints
├── TSiTPlus_NO_weather_best.pth
├── CONFIG.txt                                ← Configuration log
└── weather_comparison_plots.png (optional)   ← Visualizations
```

### Key CSV Columns

`WEATHER_COMPARISON_SUMMARY.csv`:
- `classifier`: Model architecture used
- `f1_with_weather`: F1 score with weather
- `f1_no_weather`: F1 score without weather
- `f1_improvement`: Percent change in F1
- `accuracy_with_weather`, `accuracy_no_weather`: Accuracy metrics
- `accuracy_improvement`: Accuracy percent change
- `params_with`, `params_without`: Parameter counts
- `time_with`, `time_without`: Training time in seconds

---

## Advanced Usage

### Running with Different Classifiers

```bash
# Compare across both classifier types
python stacked_gru_transformer_weather_comparison.py \
    --classifier both \
    --epochs 50
```

This trains 4 models total:
- TSiTPlus + weather
- TSiTPlus + no weather
- ConvTranPlus + weather
- ConvTranPlus + no weather

### Adjusting Architecture Parameters

```bash
# Test weather impact with different model sizes
python stacked_gru_transformer_weather_comparison.py \
    --gru_dim 128 \
    --gat_hidden 96 \
    --gru_layers 3 \
    --epochs 50
```

### Using Different Weather File

```bash
python stacked_gru_transformer_weather_comparison.py \
    --weather_file weather2016_2021.npy \
    --data_source udata \
    --epochs 50
```

---

## Troubleshooting

### Q: Weather file not found
```
FileNotFoundError: Weather file not found: weather_cn.npy
```
**Solution:**
```bash
cd d:\flight\ delay\stpn\ paper\STPN-main
# Ensure weather_cn.npy exists in data directory
ls *.npy | grep weather
```

### Q: CUDA out of memory
```bash
# Reduce batch size and chunk size
python stacked_gru_transformer_weather_comparison.py \
    --batch_size 8 \
    --chunk_size 100 \
    --accumulation_steps 32
```

### Q: Training is very slow
```bash
# Reduce number of epochs for quick test
python stacked_gru_transformer_weather_comparison.py \
    --epochs 10 \
    --patience 5
```

### Q: Results show NaN metrics
**Possible causes:**
- Learning rate too high: `--lr 5e-5` (reduce by half)
- Gradient overflow: Check weather data range
- Insufficient data: Check data splits

---

## Integration into Production

Once you've confirmed weather impact:

### Option A: Keep Weather (if beneficial)
```bash
# Use original stacked_gru_transformer.py
python stacked_gru_transformer.py \
    --weather_file weather_cn.npy \
    --classifier TSiTPlus \
    --epochs 100
```

### Option B: Remove Weather (if not beneficial)
```python
# Edit stacked_gru_transformer.py:
# In main(), after loading:

train_inputs = train_inputs[:, :, :2]   # Keep only first 2 columns (delay)
val_inputs = val_inputs[:, :, :2]
test_inputs = test_inputs[:, :, :2]

feature_dim = 2  # Override
```

---

## Decision Matrix

Use this to decide on weather inclusion:

| F1 Improvement | Recommendation | Action |
|---|---|---|
| > 5% | Definitely keep | Use weather in production |
| 2-5% | Likely keep | Use weather; validate on holdout |
| 0.5-2% | Conditional | Keep if weather easily available; else remove |
| -0.5 to +0.5% | Optional | Remove for simplicity (no perf loss) |
| < -0.5% | Investigate | Debug weather preprocessing |

---

## Example Complete Workflow

```bash
# 1. Run comparison with default settings
python stacked_gru_transformer_weather_comparison.py --epochs 30

# 2. Analyze automatically (with plots)
python analyze_weather_comparison.py weather_comparison_20260321_120000/ --plot

# 3. Inspect detailed results
cat weather_comparison_20260321_120000/WEATHER_COMPARISON_SUMMARY.csv

# 4. Based on results, choose production config:

# If weather helped:
python stacked_gru_transformer.py --classifier TSiTPlus --epochs 100

# If weather didn't help:
# Edit stacked_gru_transformer.py to remove weather, then:
python stacked_gru_transformer.py --classifier TSiTPlus --epochs 100
```

---

## Key Metrics to Watch

### 1. F1 Score (Primary)
- Balanced metric for imbalanced data
- **Threshold**: >2% improvement to justify weather

### 2. Per-Channel F1
- Separate F1 for arrival vs. departure
- Weather may help one channel more than the other
- Example: Weather better predicts departures (+4%) than arrivals (+1%)

### 3. Accuracy
- Overall correct predictions
- Secondary metric (less important for imbalanced data)

### 4. Parameter Efficiency
- `params_ratio = params_with / params_without`
- Larger ratio = less parameter efficiency from weather
- If ratio > 1.2 and improvement < 2%, weather may not justify complexity

### 5. Training Time Ratio
- `time_ratio = time_with / time_without`
- Acceptable if < 1.1 (10% overhead)
- If > 1.2 (20% overhead) and improvement < 1%, favor simpler model

---

## Contributing / Extensions

If you modify these scripts, consider:

1. **Custom weather preprocessing**: Normalize by season, airport proximity
2. **Feature importance analysis**: Which weather features help most?
3. **Temporal patterns**: Does weather help more at certain times?
4. **Airport-specific models**: Does weather importance vary by airport?

---

## References

- Original paper: `stacked_gru_transformer.py` documentation
- Data loading: [classifykat.py](classifykat.py)
- Architecture: [hybrid_graph_tsai.py](hybrid_graph_tsai.py)

---

## Questions?

Check the detailed guide: [WEATHER_USAGE_GUIDE.md](WEATHER_USAGE_GUIDE.md)

