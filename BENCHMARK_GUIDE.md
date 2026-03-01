# Quick Benchmark Guide

## Available Benchmark Scripts

### 1️⃣ Simple Benchmark (No Graph Models)
**File:** `simple_benchmark.py`  
**Models:** Historical Average, LSTM, GRU, Random Forest, XGBoost  
**Requirements:** Minimal (no graph dependencies)

```powershell
# Quick test
python simple_benchmark.py --quick_test

# Full run
python simple_benchmark.py --epochs 20
```

**Use when:**
- You don't need graph-based models
- Quick baseline comparison
- Testing basic models only

---

### 2️⃣ Benchmark with STPN
**File:** `benchmark_with_stpn.py`  
**Models:** Historical Average, LSTM, GRU, **STPN**  
**Requirements:** model.py must be available

```powershell
# Quick test
python benchmark_with_stpn.py --quick_test

# Full run
python benchmark_with_stpn.py --epochs 20
```

**Use when:**
- You want to test STPN specifically
- Compare graph model (STPN) vs simpler baselines
- No DSAFNet required

---

### 3️⃣ Comprehensive Benchmark (All Models)
**File:** `run_benchmark.py`  
**Models:** All 15 models including STPN, DSAFNet, tree methods, etc.  
**Requirements:** All dependencies (may have issues)

```powershell
python run_benchmark.py --quick_test
```

**Use when:**
- You have all dependencies installed
- Want complete comparison
- Research/publication purposes

---

## Quick Start Guide

### Step 1: Choose Your Benchmark

| Need | Use This |
|------|----------|
| **Just want quick results** | `simple_benchmark.py` |
| **Need STPN comparison** | `benchmark_with_stpn.py` |
| **Full research comparison** | `run_benchmark.py` |

### Step 2: Run It

```powershell
# Example: Benchmark with STPN (recommended)
cd "D:\flight delay\stpn paper\STPN-main"
python benchmark_with_stpn.py --quick_test
```

### Step 3: Check Results

Results saved to: `stpn_benchmark_YYYYMMDD_HHMMSS/`
- `results_*.csv` - Metrics table
- `results_*.json` - Detailed results
- `results_*.png` - Comparison plots

---

## Common Options

```powershell
--quick_test          # Fast mode (10 epochs)
--epochs 30           # Custom epoch count
--batch_size 32       # Custom batch size
--data_dir cdata      # Data directory
--lr 0.001            # Learning rate
```

---

## Example Commands

### Quick STPN Test (Recommended)
```powershell
python benchmark_with_stpn.py --quick_test
```
**Time:** ~5-10 minutes  
**Models:** 4 (HA, LSTM, GRU, STPN)

### Full STPN Benchmark
```powershell
python benchmark_with_stpn.py --epochs 30 --batch_size 16
```
**Time:** ~20-30 minutes  
**Models:** 4 (HA, LSTM, GRU, STPN)

### Simple Baseline Only
```powershell
python simple_benchmark.py --quick_test
```
**Time:** ~3-5 minutes  
**Models:** 5 (HA, LSTM, GRU, RF, XGB)

---

## Troubleshooting

### Issue: STPN not available
**Fix:** Check that `model.py` exists in the same directory

### Issue: Out of memory
**Fix:** Reduce batch size: `--batch_size 8`

### Issue: torch-sparse error
**Fix:** Use `simple_benchmark.py` instead (no graph dependencies)

---

## Output Files

All benchmarks create timestamped folders with:

- 📊 **CSV file** - Easy to open in Excel
- 📄 **JSON file** - Complete data for analysis  
- 📈 **PNG plot** - Visual comparison

---

**Recommended:** Start with `benchmark_with_stpn.py --quick_test` to get quick results with STPN!
