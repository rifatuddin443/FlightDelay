# QUICK REFERENCE - Weather Comparison

## TL;DR

**✓ Weather IS being used in the current pipeline**

To compare performance WITH vs WITHOUT weather:

```bash
run_weather_comparison.bat
```

Results appear in: `weather_comparison_YYYYMMDD_HHMMSS/`

Open: `WEATHER_COMPARISON_SUMMARY.csv` to see results

---

## One-Minute Workflow

```bash
# Step 1: Run comparison (10-60 min depending on epochs)
run_weather_comparison.bat --epochs 50

# Step 2: Check results
# Open: weather_comparison_20260321_120000/WEATHER_COMPARISON_SUMMARY.csv

# Step 3: Decide
# - F1 improvement > 2%?      → Keep weather ✓
# - F1 improvement -0.5 to 2% → Optional
# - F1 improvement < -0.5%?   → Investigate ✗
```

---

## Command Options

```bash
# Basic (uses defaults)
run_weather_comparison.bat

# Test all classifiers
run_weather_comparison.bat --classifier both --epochs 50

# Generate plots
run_weather_comparison.bat --epochs 50 --plot

# Quick test (10 epochs only)
run_weather_comparison.bat --epochs 10

# Adjust learning rate
run_weather_comparison.bat --lr 5e-5 --epochs 50

# Use different device
run_weather_comparison.bat --device cuda --epochs 50

# View help
run_weather_comparison.bat --help
```

---

## Interpreting Results

### Main CSV: `WEATHER_COMPARISON_SUMMARY.csv`

| Column | What It Means | Good Value |
|--------|---------------|-----------|
| `f1_improvement` | F1 score increase from weather | > 2% |
| `accuracy_improvement` | Accuracy increase from weather | > 1% |
| `f1_with_weather` | F1 score using weather data | Higher is better |
| `f1_no_weather` | F1 score without weather | ← No weather baseline |

### Decision Table

```
f1_improvement    Recommendation    Action
═══════════════════════════════════════════════════════════════
  > 5%           ✓ DEFINITELY ADD    Keep weather, validate results
  2-5%           ✓ LIKELY ADD        Use weather in production
  0.5-2%         ~ CONDITIONAL       Keep if available, OK to remove
 -0.5 +0.5%      ≈ NEGLIGIBLE        Safe to remove (no loss)
  < -0.5%        ✗ INVESTIGATE       Debug weather preprocessing
```

---

## Example Output

```
WEATHER_COMPARISON SUMMARY
═════════════════════════════

Classifier: TSiTPlus
  F1 WITHOUT weather:    0.7234
  F1 WITH weather:       0.7456  ← 3.07% improvement ✓
  Accuracy improvement:  +2.19%
  
Recommendation: ✓ KEEP WEATHER IN PRODUCTION
```

---

## Output Files

```
weather_comparison_20260321_120000/
├── WEATHER_COMPARISON_SUMMARY.csv     ← KEY RESULTS (read this!)
├── TSiTPlus_WITH_weather_metrics.csv
├── TSiTPlus_NO_weather_metrics.csv
├── TSiTPlus_WITH_weather_best.pth     ← Best model with weather
├── TSiTPlus_NO_weather_best.pth       ← Best model without weather
├── CONFIG.txt                          ← Configuration used
└── weather_comparison_plots.png        ← Plots (if --plot used)
```

---

## Data Pipeline

**Current inputs to model:**
```
[Delay_features (2) | Weather_features (N)]

Example:
  Column 0: Arrival delay
  Column 1: Departure delay  
  Column 2-6: Weather (temp, humidity, wind, pressure, precip)
  
  Total: 7 features per node per timestep
```

**Comparison tests:**
```
Model A: Uses columns 0-1 (delay only)
Model B: Uses columns 0-6 (delay + weather)

Compare performance → Impact of weather
```

---

## Common Scenarios

### Scenario 1: Weather Significantly Helps
```
f1_improvement: +3.5%
→ Decision: ✓ USE WITH WEATHER
→ Action: Done! stacked_gru_transformer.py already uses weather
```

### Scenario 2: Weather Slightly Helps
```
f1_improvement: +0.8%
→ Decision: ≈ Optional
→ Action: Use weather (already included); minimal complexity cost
```

### Scenario 3: Weather Makes No Difference
```
f1_improvement: +0.05%
→ Decision: ≈ NEGLIGIBLE
→ Action: Safe to remove for simplicity, or keep (your choice)
```

### Scenario 4: Weather Hurts Performance
```
f1_improvement: -2.1%
→ Decision: ✗ INVESTIGATE
→ Action: Check weather data quality; may need preprocessing
```

---

## Troubleshooting

### Error: "Weather file not found"
```bash
# Check if weather_cn.npy exists
dir cdata\weather*.npy

# If missing, check project setup
# Should be in: cdata\weather_cn.npy or udata\weather2016_2021.npy
```

### Error: "Python not found"
```bash
# Ensure Python is in PATH
python --version

# If not, run from Python directory or add to PATH
```

### Error: "CUDA out of memory"
```bash
# Reduce batch size
run_weather_comparison.bat --batch_size 8

# Or use CPU
run_weather_comparison.bat --device cpu --batch_size 16
```

### Warning: "Training very slow"
```bash
# Quick test with fewer epochs
run_weather_comparison.bat --epochs 5

# Then run full version after confirming it works
run_weather_comparison.bat --epochs 50
```

---

## Key Metrics at a Glance

| Metric | With Weather | Without Weather | Δ | % Δ |
|--------|---|---|---|---|
| **F1** | 0.7456 | 0.7234 | +0.0222 | **+3.07%** |
| **Accuracy** | 0.8301 | 0.8123 | +0.0178 | +2.19% |
| **F1_arrival** | 0.7320 | 0.7100 | +0.0220 | +3.10% |
| **F1_departure** | 0.7592 | 0.7368 | +0.0224 | +3.04% |
| **Params** | 162,840 | 145,230 | +17,610 | +12.1% |
| **Train time** | 361.2s | 342.5s | +18.7s | +5.5% |

→ **Decision: KEEP WEATHER** (3% gain worth 12% param increase)

---

## Next Steps

1. **Run comparison**: `run_weather_comparison.bat --epochs 50`
2. **Wait**: 20-60 minutes (depends on hardware)
3. **Check**: Open `WEATHER_COMPARISON_SUMMARY.csv`
4. **Decide**: Apply decision table above
5. **Implement**: Update production config if needed

---

## Files You Need to Know

```
MAIN SCRIPTS (run these):
  run_weather_comparison.bat     ← Start here (Windows)
  run_weather_comparison.sh      ← Start here (Linux/Mac)

ANALYSIS TOOLS:
  stacked_gru_transformer_weather_comparison.py  ← Comparison engine
  analyze_weather_comparison.py                  ← Result analyzer

DOCUMENTATION (read these):
  WEATHER_USAGE_GUIDE.md        ← Technical details
  README_WEATHER_WORKFLOW.md    ← Complete guide
  IMPLEMENTATION_SUMMARY.md     ← Implementation details
  QUICK_REFERENCE.md            ← This file
```

---

## Performance Expectations

| Component | Typical Value | Notes |
|-----------|---|---|
| **F1 improvement from weather** | 1-5% | Flight delays often weather-driven |
| **Training time per model** | 5-30 min | CPU: longer; GPU: faster |
| **Memory usage** | 2-8 GB | Depends on batch_size |
| **Model accuracy baseline** | 0.75-0.85 | Typical for flight prediction |

---

## For More Information

- **Technical details**: Read `WEATHER_USAGE_GUIDE.md`
- **Full workflow**: Read `README_WEATHER_WORKFLOW.md`
- **Implementation**: Read `IMPLEMENTATION_SUMMARY.md`
- **This guide**: You're reading it!

---

## Summary

✓ Weather **IS used** in current pipeline  
✓ To compare: Run `run_weather_comparison.bat`  
✓ Decision: Check `f1_improvement` in results CSV  
✓ Time: 30-60 minutes typically  
✓ Action: Keep or remove based on % improvement

