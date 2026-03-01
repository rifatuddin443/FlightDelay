"""
Benchmark Setup Checker

This script verifies that all required dependencies and data files are available
before running the full benchmark.

Usage:
    python check_benchmark_setup.py
"""

import sys
import os
from pathlib import Path


def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 7:
        print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"   ❌ Python {version.major}.{version.minor} (Need Python 3.7+)")
        return False


def check_imports():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'torch': 'PyTorch',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
    }
    
    optional_packages = {
        'torch_geometric': 'PyTorch Geometric (for graph models)',
        'opacus': 'Opacus (for differential privacy)',
        'statsmodels': 'StatsModels (for VAR)',
        'xgboost': 'XGBoost (for gradient boosting)',
        'lightgbm': 'LightGBM (for gradient boosting)',
    }
    
    all_ok = True
    
    # Check required packages
    for module, name in packages.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ❌ {name} - NOT FOUND")
            all_ok = False
    
    # Check optional packages
    print("\n   Optional packages:")
    for module, name in optional_packages.items():
        try:
            __import__(module)
            print(f"   ✅ {name}")
        except ImportError:
            print(f"   ⚠️  {name} - Not found (some features may be unavailable)")
    
    return all_ok


def check_data_files():
    """Check if data directory exists."""
    print("\n📊 Checking data files...")
    
    data_dirs = ['cdata', 'udata', 'data']
    found = False
    
    for data_dir in data_dirs:
        if os.path.exists(data_dir):
            print(f"   ✅ Found data directory: {data_dir}")
            
            # Check for typical files
            files = os.listdir(data_dir)
            if len(files) > 0:
                print(f"      Contains {len(files)} files")
                found = True
            else:
                print(f"      ⚠️  Directory is empty")
        
    if not found:
        print(f"   ⚠️  No data directories found. Looking for: {', '.join(data_dirs)}")
        print(f"      You may need to specify --data_dir when running the benchmark")
    
    return True  # Don't block on missing data


def check_model_files():
    """Check if model implementation files exist."""
    print("\n🤖 Checking model files...")
    
    model_files = {
        'classifykat.py': 'Core utilities',
        'baseline_methods.py': 'Classical baselines (HA, VAR)',
        'DSAFnet.py': 'DSAFNet model',
        'model.py': 'STPN model',
    }
    
    for file, description in model_files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} - {description}")
        else:
            print(f"   ⚠️  {file} - Not found ({description} will be unavailable)")
    
    return True


def check_benchmark_files():
    """Check if benchmark files were created correctly."""
    print("\n🔬 Checking benchmark files...")
    
    benchmark_files = {
        'comprehensive_benchmark.py': 'Main benchmark framework',
        'benchmark_graph_models.py': 'Graph model integration',
        'run_benchmark.py': 'Benchmark runner script',
        'BENCHMARK_README.md': 'Documentation',
    }
    
    all_ok = True
    for file, description in benchmark_files.items():
        if os.path.exists(file):
            print(f"   ✅ {file} - {description}")
        else:
            print(f"   ❌ {file} - NOT FOUND ({description})")
            all_ok = False
    
    return all_ok


def check_gpu():
    """Check if GPU is available."""
    print("\n🎮 Checking GPU availability...")
    
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   ✅ GPU available: {gpu_name}")
            print(f"      CUDA Version: {torch.version.cuda}")
        else:
            print(f"   ⚠️  No GPU available (will use CPU - slower)")
    except:
        print(f"   ⚠️  Could not check GPU status")
    
    return True


def print_summary(checks):
    """Print summary of checks."""
    print("\n" + "="*80)
    print("SETUP CHECK SUMMARY")
    print("="*80)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:12} - {check_name}")
    
    print("="*80)
    
    if all_passed:
        print("\n🎉 All critical checks passed! You're ready to run the benchmark.")
        print("\nQuick start:")
        print("  python run_benchmark.py --quick_test")
        print("\nFull benchmark:")
        print("  python run_benchmark.py --data_dir cdata --epochs 50")
    else:
        print("\n⚠️  Some checks failed. Please install missing dependencies.")
        print("\nInstall required packages:")
        print("  pip install torch numpy pandas matplotlib seaborn")
        print("\nInstall optional packages:")
        print("  pip install torch-geometric statsmodels opacus")
    
    print("\n" + "="*80 + "\n")


def main():
    """Run all checks."""
    print("\n" + "="*80)
    print(" " * 25 + "BENCHMARK SETUP CHECKER")
    print("="*80 + "\n")
    
    checks = {
        'Python Version': check_python_version(),
        'Required Packages': check_imports(),
        'Data Files': check_data_files(),
        'Model Files': check_model_files(),
        'Benchmark Files': check_benchmark_files(),
        'GPU': check_gpu(),
    }
    
    print_summary(checks)


if __name__ == '__main__':
    main()
