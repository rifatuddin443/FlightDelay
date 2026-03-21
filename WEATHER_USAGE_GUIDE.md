# Weather Data Usage & Comparison

## Current Status: **WEATHER IS BEING USED**

The `stacked_gru_transformer.py` script already incorporates weather data in the default pipeline. This document explains:
1. How weather data is currently integrated
2. How to compare models with/without weather
3. Interpretation of results

---

## Data Pipeline

### Data Loading (load_flight_data in classifykat.py)

The full feature stack per node at each timestep is:

```
[delay_features | weather_features | time_embeddings]
```

**Component breakdown:**
- **delay_features**: Shape `(num_nodes, timesteps, 2)` for arrival/departure delays
- **weather_features**: Shape `(num_nodes, timesteps, weather_dim)` loaded from `weather_cn.npy`
  - Scaled and clipped to [-1.5, 1.5] for numerical stability
  - Applied per-timestep scaling from training data statistics
- **time_embeddings**: Shape `(num_nodes, timesteps, 2)` - sinusoidal hour-of-day encoding
  - sin and cos components for periodic time representation

### Current Setup in stacked_gru_transformer.py

```python
# Feature stack: [delay (2) | weather (N) | time (2)]
train_inputs = np.concatenate([
    train_delay,                           # (nodes, steps, 2)
    weather_scaled[:, :train_end, :],      # (nodes, steps, weather_dim)
    time_embed[:, :train_end, :],          # (nodes, steps, 2)
], axis=2)

# Remove time embeddings (keep delay + weather)
train_inputs = train_inputs[:, :, :-2]   # Removes last 2 time columns
```

**Result:** Model input has shape `(nodes, steps, 2+weather_dim)`

---

## Running the Comparison

### Option 1: Original Script (WITH weather)

```bash
python stacked_gru_transformer.py --classifier TSiTPlus --epochs 50
```

This trains on: **delay_features + weather_features**

### Option 2: Comparison Script (WITH vs WITHOUT weather)

```bash
python stacked_gru_transformer_weather_comparison.py --classifier TSiTPlus --epochs 50
```

This trains TWO models:
1. **WITH weather**: Uses full feature set (delay + weather)
2. **WITHOUT weather**: Uses only delay features

Outputs a summary comparing:
- F1 score improvement: `(F1_with - F1_without) / F1_without * 100%`
- Accuracy improvement
- Per-channel (arrival/departure) delta
- Parameter count & training time differences

---

## Interpreting Results

### Expected Outcomes

**Case 1: Positive Weather Impact** (typical for flight delay prediction)
```
Classifier: TSiTPlus
  F1:  0.7234 → 0.7456  (+3.07%)
  Acc: 0.8123 → 0.8301  (+2.19%)
  
  F1_arrival: 0.7100 → 0.7320  (+3.10%)
  F1_departure: 0.7368 → 0.7592  (+3.04%)
```
→ Weather features contribute meaningfully to predictions

**Case 2: Negligible Weather Impact** (features already captured)
```
Classifier: ConvTranPlus
  F1:  0.7450 → 0.7461  (+0.15%)
  Acc: 0.8300 → 0.8305  (+0.06%)
```
→ Other features already encode weather information; little added value

**Case 3: Negative Weather Impact** (noise or overfitting)
```
Classifier: TSiTPlus
  F1:  0.7456 → 0.7234  (-3.07%)
  Acc: 0.8301 → 0.8123  (-2.19%)
```
→ Weather features add noise; consider removing or better preprocessing

---

## Available Weather Features

The weather data loaded depends on your `weather_cn.npy` or `weather2016_2021.npy` files.

Common flight-relevant weather metrics might include:
- Temperature (°C)
- Humidity (%)
- Wind speed (m/s)
- Wind direction (degrees)
- Precipitation (mm)
- Cloud cover (%)
- Visibility (km)
- Atmospheric pressure (hPa)

### Verify Weather Dimensions

Run quick diagnostic:
```python
import numpy as np

weather = np.load('weather_cn.npy')
print(f"Weather shape: {weather.shape}")          # Check dimensions
print(f"Weather dtype: {weather.dtype}")
print(f"Weather stats - mean: {np.nanmean(weather):.3f}, "
      f"std: {np.nanstd(weather):.3f}, "
      f"min: {np.nanmin(weather):.3f}, "
      f"max: {np.nanmax(weather):.3f}")
```

---

## Customization Options

### Option A: Remove Weather From Baseline

To train WITHOUT weather using the original script (not recommended - use comparison script instead):

```python
# In stacked_gru_transformer.py main()
# After loading data:
train_inputs = train_inputs[:, :, :2]   # Keep only delay features
val_inputs = val_inputs[:, :, :2]
test_inputs = test_inputs[:, :, :2]

feature_dim = 2  # Override
```

### Option B: Custom Weather Preprocessing

Pre-process weather data before running:

```python
def custom_weather_preprocess(weather_file):
    wdata = np.load(weather_file)
    
    # Example: Log-transform positive values
    wdata = np.log1p(np.abs(wdata))
    
    # Re-normalize
    wdata = (wdata - np.nanmean(wdata)) / (np.nanstd(wdata) + 1e-6)
    
    # Save modified version
    np.save('weather_cn_custom.npy', wdata)
    return 'weather_cn_custom.npy'

custom_weather_file = custom_weather_preprocess('weather_cn.npy')
# Then run with: --weather_file custom_weather_file
```

### Option C: Weighted Feature Contribution

To investigate feature importance, modify the GRU encoder to learn per-feature scaling:

```python
class FeatureScaledEncoder(GRUAttentionEncoder):
    def __init__(self, c_in, weather_dim=3, **kwargs):
        super().__init__(c_in, **kwargs)
        # Learnable per-feature importance
        self.feature_scales = nn.Parameter(torch.ones(c_in))
    
    def forward(self, x):
        # Scale input features
        x_scaled = x * self.feature_scales.view(1, -1, 1)
        return super().forward(x_scaled)
```

---

## Results Logging Location

Comparison script outputs to: `weather_comparison_YYYYMMDD_HHMMSS/`

Contents:
- `WEATHER_COMPARISON_SUMMARY.csv` - Side-by-side metrics
- `{classifier}_WITH_weather_metrics.csv` - Detailed with-weather results
- `{classifier}_NO_weather_metrics.csv` - Detailed without-weather results
- `{classifier}_WITH_weather_best.pth` - Saved model checkpoint (with weather)
- `{classifier}_NO_weather_best.pth` - Saved model checkpoint (without weather)
- `CONFIG.txt` - Configuration summary

---

## Next Steps

1. **Run baseline**: `python stacked_gru_transformer_weather_comparison.py --epochs 50`
2. **Analyze results**: Check `WEATHER_COMPARISON_SUMMARY.csv` in output directory
3. **Decision**:
   - If weather helps significantly (>2% improvement): Keep weather in production
   - If weather hurts performance: Investigate preprocessing or remove
   - If negligible: Either keep (minimal cost) or remove (reduce complexity)
4. **Fine-tune**: Adjust `--weather_file`, preprocessing, or feature engineering based on results

