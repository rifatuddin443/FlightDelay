# Flight Delay Prediction Models - Comprehensive List

This benchmark suite includes **15 different models** spanning classical statistics, machine learning, deep learning, and graph neural networks.

## 🎯 Complete Model List

### 📊 Category 1: Classical Statistical Methods (2 models)

| # | Model | Description | Key Features | Typical Use |
|---|-------|-------------|--------------|-------------|
| 1 | **Historical Average** | Periodic averaging | - No training required<br>- Fast inference<br>- Simple baseline | Quick baseline comparison |
| 2 | **VAR** | Vector Auto-Regressive | - Multivariate time series<br>- Linear relationships<br>- Statistical foundation | Traditional forecasting |

### 🌲 Category 2: Tree-Based Machine Learning (4 models)

| # | Model | Description | Key Features | Typical Use |
|---|-------|-------------|--------------|-------------|
| 3 | **Random Forest** | Ensemble of decision trees | - Robust to outliers<br>- Feature importance<br>- Non-linear patterns | General-purpose ML |
| 4 | **Gradient Boosting** | Sequential tree ensemble | - Iterative improvement<br>- Handles complex patterns<br>- sklearn implementation | Traditional ML benchmark |
| 5 | **XGBoost** | Extreme Gradient Boosting | - Regularization<br>- Parallel processing<br>- State-of-the-art performance | Competition-winning model |
| 6 | **LightGBM** | Light Gradient Boosting | - Fast training<br>- Low memory<br>- Large-scale datasets | High-performance ML |

### 🧠 Category 3: Deep Learning - Recurrent Models (6 models)

| # | Model | Description | Key Features | Typical Use |
|---|-------|-------------|--------------|-------------|
| 7 | **LSTM** | Long Short-Term Memory | - Sequential patterns<br>- Long-term dependencies<br>- Standard RNN | Temporal sequence modeling |
| 8 | **GRU** | Gated Recurrent Unit | - Simpler than LSTM<br>- Faster training<br>- Similar performance | Efficient sequential model |
| 9 | **Bi-LSTM** | Bidirectional LSTM | - Forward + backward context<br>- Better sequence understanding<br>- Double parameters | Enhanced temporal modeling |
| 10 | **CNN-LSTM** | Hybrid CNN + LSTM | - CNN for feature extraction<br>- LSTM for temporal modeling<br>- Spatial-temporal fusion | Multi-scale feature learning |
| 11 | **Attention-LSTM** | LSTM with attention | - Focus on important timesteps<br>- Weighted temporal aggregation<br>- Interpretable weights | Selective attention mechanism |
| 12 | **Transformer** | Self-attention model | - Parallel processing<br>- Global context<br>- State-of-the-art NLP | Advanced sequence modeling |

### 🕸️ Category 4: Graph Neural Networks (2 models)

| # | Model | Description | Key Features | Typical Use |
|---|-------|-------------|--------------|-------------|
| 13 | **STPN** | Spatio-Temporal Pattern Network | - Multiple graph views<br>- Spatial aggregation<br>- Temporal dynamics | Airport network modeling |
| 14 | **DSAFNet** | Dual-Stream Attention Fusion | - Spatial + temporal streams<br>- Cross-attention fusion<br>- Multi-graph encoding | Advanced graph modeling |

### 🌟 Category 5: Your Custom Model

| # | Model | Description | Key Features | Typical Use |
|---|-------|-------------|--------------|-------------|
| 15 | **CNN-KAN** | CNN with KAN + DP | - Kolmogorov-Arnold Networks<br>- Differential Privacy<br>- Three-stage pipeline | Privacy-preserving prediction |

---

## 📈 Model Selection Guide

### Speed vs. Accuracy Trade-off

```
Fast (Training Time)          │  Slow
─────────────────────────────────────────────
HA → VAR → LightGBM → XGBoost → RF → GB
     │                              
     └─→ GRU → LSTM → Transformer
                │
                └─→ Bi-LSTM → CNN-LSTM → Attention-LSTM
                                │
                                └─→ STPN → DSAFNet → CNN-KAN
```

### Complexity vs. Performance

```
Simple (Model Complexity)     │  Complex
─────────────────────────────────────────────
HA → VAR → RF → GB
          │
          └─→ XGBoost → LightGBM
                    │
                    └─→ LSTM → GRU → Transformer
                                    │
                                    └─→ Bi-LSTM → Attention-LSTM
                                                 │
                                                 └─→ CNN-LSTM → STPN → DSAFNet → CNN-KAN
```

---

## 🔍 Model Characteristics Comparison

### Data Requirements

| Model Type | Min. Samples | Feature Engineering | Graph Structure |
|------------|--------------|---------------------|-----------------|
| Classical (HA, VAR) | Low (100s) | Optional | Not needed |
| Tree-based (RF, XGB, LGB, GB) | Medium (1000s) | Helpful | Not needed |
| RNN-based (LSTM, GRU, etc.) | Medium-High (1000s) | Optional | Not needed |
| Transformer | High (10000s) | Optional | Not needed |
| Graph-based (STPN, DSAFNet) | Medium-High (1000s) | Optional | **Required** |
| CNN-KAN | Medium-High (1000s) | Integrated | **Required** |

### Computational Requirements

| Model Category | CPU Training | GPU Benefit | Memory Usage |
|----------------|--------------|-------------|--------------|
| Classical | ⚡ Fast | None | Low |
| Tree-based | ⚡ Fast | None | Medium |
| RNN (LSTM, GRU) | 🐌 Slow | High | Medium |
| Advanced RNN (Bi-LSTM, etc.) | 🐌🐌 Very Slow | Very High | High |
| Transformer | 🐌🐌 Very Slow | Very High | Very High |
| Graph Models | 🐌🐌🐌 Extremely Slow | Very High | High |

### Interpretability

| Model | Interpretability | Feature Importance | Visualization |
|-------|------------------|-------------------|---------------|
| Historical Average | ⭐⭐⭐⭐⭐ Excellent | ✅ Yes | ✅ Easy |
| VAR | ⭐⭐⭐⭐ Good | ✅ Yes | ✅ Easy |
| Random Forest | ⭐⭐⭐⭐ Good | ✅ Yes | ✅ Easy |
| XGBoost/LightGBM | ⭐⭐⭐ Moderate | ✅ Yes | ⚠️ Tools needed |
| Gradient Boosting | ⭐⭐⭐ Moderate | ✅ Yes | ⚠️ Tools needed |
| LSTM/GRU | ⭐⭐ Limited | ❌ No | ❌ Difficult |
| Bi-LSTM | ⭐⭐ Limited | ❌ No | ❌ Difficult |
| CNN-LSTM | ⭐⭐ Limited | Partial | ❌ Difficult |
| Attention-LSTM | ⭐⭐⭐ Moderate | ✅ Attention weights | ⚠️ Moderate |
| Transformer | ⭐⭐⭐ Moderate | ✅ Attention maps | ⚠️ Moderate |
| STPN | ⭐⭐ Limited | Partial | ❌ Difficult |
| DSAFNet | ⭐⭐ Limited | ✅ Attention weights | ⚠️ Moderate |
| CNN-KAN | ⭐ Poor | Partial | ❌ Difficult |

---

## 💡 Recommended Model Selection

### For Different Scenarios

#### 1. **Quick Baseline** (Need results in minutes)
- ✅ Historical Average
- ✅ LightGBM
- ✅ Random Forest

#### 2. **Best Performance** (Accuracy matters most)
- ✅ XGBoost
- ✅ LightGBM
- ✅ Attention-LSTM
- ✅ CNN-LSTM
- ✅ STPN
- ✅ DSAFNet

#### 3. **Interpretability** (Need to explain predictions)
- ✅ Random Forest
- ✅ XGBoost (with SHAP)
- ✅ Attention-LSTM (attention weights)
- ⚠️ Avoid: Deep graph models

#### 4. **Large-Scale Deployment** (Production environment)
- ✅ LightGBM (fast inference)
- ✅ XGBoost (good balance)
- ✅ GRU (if temporal patterns needed)
- ⚠️ Avoid: Heavy graph models, Transformer

#### 5. **Research & Development** (Exploring new methods)
- ✅ CNN-LSTM
- ✅ Attention-LSTM
- ✅ Transformer
- ✅ STPN
- ✅ DSAFNet
- ✅ CNN-KAN (your custom model)

#### 6. **Privacy-Sensitive Applications**
- ✅ CNN-KAN (differential privacy built-in)
- ⚠️ Classical methods can be made private
- ❌ Most others require additional privacy mechanisms

---

## 🚀 Running the Benchmark

### Test All Models
```bash
python run_benchmark.py --data_dir cdata --epochs 50
```

### Test Specific Categories
```bash
# Only classical + ML
python run_benchmark.py --skip_dl --skip_graph

# Only deep learning
python run_benchmark.py --skip_classical --skip_graph

# Only graph models
python run_benchmark.py --skip_classical --skip_dl
```

### Quick Test (Subset of models)
```bash
python run_benchmark.py --quick_test
```

---

## 📚 References & Citations

### Classical Methods
- Hamilton, J. D. (1994). Time series analysis. Princeton university press.
- Box, G. E., Jenkins, G. M., & Reinsel, G. C. (2015). Time series analysis: forecasting and control.

### Tree-Based Methods
- Breiman, L. (2001). Random forests. Machine learning, 45(1), 5-32.
- Chen, T., & Guestrin, C. (2016). Xgboost: A scalable tree boosting system. KDD.
- Ke, G., et al. (2017). Lightgbm: A highly efficient gradient boosting decision tree. NIPS.

### Deep Learning
- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. Neural computation.
- Cho, K., et al. (2014). Learning phrase representations using RNN encoder-decoder. EMNLP.
- Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.

### Graph Neural Networks
- Kipf, T. N., & Welling, M. (2016). Semi-supervised classification with graph convolutional networks.
- Veličković, P., et al. (2017). Graph attention networks. ICLR.

### Flight Delay Prediction
- [Your references from the review paper]
- Recent studies using machine learning for flight delay prediction (2020-2026)

---

**Last Updated:** February 15, 2026  
**Total Models:** 15  
**Benchmark Version:** 1.0
