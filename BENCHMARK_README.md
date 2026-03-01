# Comprehensive Model Benchmark Suite

A systematic comparison framework for flight delay prediction models, including classical baselines, machine learning methods, deep learning approaches, and state-of-the-art graph-based models.

## 📌 Quick Links

- 📖 **[Complete Model List & Comparison](MODELS_OVERVIEW.md)** - Detailed descriptions of all 15 models
- 📋 **[Quick Reference Card](BENCHMARK_QUICK_REFERENCE.txt)** - Command cheat sheet
- 🔧 **Setup Checker**: `python check_benchmark_setup.py`

## 📊 Models Compared (15 Total)

### Classical Baselines
1. **Historical Average (HA)**: Computes average delays over time periods
2. **VAR (Vector Auto-Regressive)**: Multivariate time series forecasting

### Machine Learning Models
3. **Random Forest**: Ensemble of decision trees
4. **XGBoost**: Extreme Gradient Boosting
5. **LightGBM**: Light Gradient Boosting Machine
6. **Gradient Boosting**: Traditional gradient boosting regressor

### Deep Learning Baselines
7. **LSTM**: Long Short-Term Memory recurrent network
8. **GRU**: Gated Recurrent Unit network
9. **Bi-LSTM**: Bidirectional LSTM
10. **CNN-LSTM**: Hybrid convolutional-recurrent architecture
11. **Attention-LSTM**: LSTM with attention mechanism
12. **Transformer**: Self-attention based sequential model

### Graph-Based State-of-the-Art
13. **STPN**: Spatio-Temporal Pattern Network
14. **DSAFNet**: Dual-Stream Attention Fusion Network
15. **Your Custom Model**: CNN-KAN with differential privacy

## 🚀 Quick Start

### Basic Usage

Run the full benchmark with all models:
```bash
python run_benchmark.py --data_dir cdata --epochs 50
```

### Quick Test Mode

Run a fast test with fewer epochs and limited models:
```bash
python run_benchmark.py --quick_test
```

This will:
- Use only 10 epochs
- Skip VAR and Transformer (slower models)
- Complete in ~10-15 minutes

## 📝 Command Line Options

### Data Parameters
```bash
--data_dir <path>        # Directory containing flight data (default: cdata)
--seq_length <int>       # Input sequence length (default: 12)
--pred_length <int>      # Prediction horizon (default: 4)
--test_ratio <float>     # Test set ratio (default: 0.2)
```

### Training Parameters
```bash
--epochs <int>           # Number of training epochs (default: 50)
--batch_size <int>       # Batch size (default: 32)
--lr <float>             # Learning rate (default: 0.001)
```

### Model Selection
```bash
--skip_classical         # Skip HA and VAR
--skip_dl                # Skip LSTM, GRU, Transformer
--skip_graph             # Skip STPN and DSAFNet
--quick_test             # Fast mode (10 epochs, limited models)
```

## 📖 Usage Examples

### Example 1: Compare Only Graph Models
```bash
python run_benchmark.py --skip_classical --skip_dl --epochs 30
```

### Example 2: Full Comparison with Custom Settings
```bash
python run_benchmark.py \
    --data_dir cdata \
    --epochs 100 \
    --batch_size 64 \
    --lr 0.0001 \
    --seq_length 18 \
    --pred_length 6
```

### Example 3: Classical vs Deep Learning Only
```bash
python run_benchmark.py --skip_graph --epochs 20
```

## 📊 Output Files

The benchmark creates a timestamped directory `benchmark_results_YYYYMMDD_HHMMSS/` containing:

### 1. Results Table (CSV)
```
benchmark_results_20260215_143022.csv
```
Contains MAE, RMSE, R², training time, and inference time for each model.

### 2. Detailed Results (JSON)
```
benchmark_results_20260215_143022.json
```
Complete results with all metrics and metadata.

### 3. Comparison Plots (PNG)
```
benchmark_comparison_20260215_143022.png
```
Visual comparison across 4 metrics:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- Training Time

## 📈 Evaluation Metrics

| Metric | Description | Better Value |
|--------|-------------|--------------|
| **MAE** | Mean Absolute Error | Lower ⬇️ |
| **RMSE** | Root Mean Squared Error | Lower ⬇️ |
| **R²** | Coefficient of Determination | Higher ⬆️ |
| **Training Time** | Time to train model (seconds) | - |
| **Inference Time** | Time to predict test set (seconds) | Lower ⬇️ |

## 🔧 Customization

### Adding Your Own Model

To benchmark your custom model:

1. Create a wrapper class in `benchmark_graph_models.py`:
```python
class YourModelWrapper(GraphModelWrapper):
    def __init__(self, ...):
        model = YourModel(...)
        super().__init__(model, 'YourModelName', device)
    
    def prepare_data_for_training(self, data_dict, batch_size):
        # Prepare data loaders
        return train_loader, test_loader
    
    def _compute_loss(self, batch):
        # Compute loss for training
        return loss
    
    def _forward_batch(self, batch):
        # Forward pass for evaluation
        return outputs, targets
```

2. Add to benchmark in `benchmark_graph_models.py`:
```python
def add_graph_models_to_benchmark(benchmark_runner):
    # ... existing code ...
    
    # Add your model
    your_model = YourModelWrapper(...)
    train_loader, test_loader = your_model.prepare_data_for_training(data)
    optimizer = torch.optim.Adam(your_model.model.parameters())
    training_time = your_model.train(train_loader, optimizer, epochs)
    results = your_model.evaluate(test_loader)
    benchmark_runner.results['YourModel'] = results
```

## 🎯 Expected Performance

Based on typical flight delay prediction benchmarks and recent literature (2020-2026):

| Model | Typical MAE Range | Relative Speed | Common Use Case |
|-------|-------------------|----------------|-----------------|
| Historical Average | 15-25 | ⚡⚡⚡ Very Fast | Simple baseline |
| VAR | 12-20 | ⚡⚡ Fast | Multivariate time series |
| Random Forest | 10-16 | ⚡⚡ Fast | Robust ensemble |
| XGBoost | 8-14 | ⚡⚡ Fast | High performance ML |
| LightGBM | 8-14 | ⚡⚡⚡ Very Fast | Large-scale data |
| Gradient Boosting | 9-15 | ⚡ Medium | Traditional ML |
| LSTM | 10-15 | ⚡ Medium | Sequential patterns |
| GRU | 10-15 | ⚡ Medium | Simpler RNN |
| Bi-LSTM | 9-14 | 🐌 Slow | Bidirectional context |
| CNN-LSTM | 8-13 | 🐌 Slow | Spatial-temporal features |
| Attention-LSTM | 7-13 | 🐌 Slow | Important timesteps |
| Transformer | 8-14 | 🐌 Slow | Self-attention |
| STPN | 7-12 | 🐌🐌 Very Slow | Graph structure |
| DSAFNet | 6-11 | 🐌🐌 Very Slow | Dual-stream fusion |
| CNN-KAN (Your Model) | **?** | 🐌🐌 Very Slow | Advanced architecture |

*Note: Actual performance depends on dataset characteristics*

## 🐛 Troubleshooting

### Issue: "DSAFNet not available"
**Solution**: Check if `DSAFnet.py` is in the same directory and all dependencies are installed.

### Issue: "STPN not available"
**Solution**: Ensure `model.py` contains the STPN class and torch_geometric is installed.

### Issue: Out of Memory
**Solution**: Reduce batch size or skip graph models:
```bash
python run_benchmark.py --batch_size 16 --skip_graph
```

### Issue: VAR taking too long
**Solution**: Use quick test mode or skip classical methods:
```bash
python run_benchmark.py --quick_test
# or
python run_benchmark.py --skip_classical
```

## 📚 Requirements

```bash
pip install torch numpy pandas matplotlib seaborn
pip install torch-geometric statsmodels
```

For differential privacy support:
```bash
pip install opacus
```

## 🤝 Contributing

To add new baseline methods:

1. Add model class to `comprehensive_benchmark.py`
2. Implement training and evaluation methods
3. Add to `BenchmarkRunner.run_all_baselines()`

## 📄 License

This benchmark suite is part of the STPN research project.

## 📧 Support

For issues or questions:
1. Check existing issues in the repository
2. Review the troubleshooting section above
3. Create a new issue with benchmark logs

---

**Happy Benchmarking! 🚀**
