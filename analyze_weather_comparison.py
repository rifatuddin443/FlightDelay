"""
Weather Comparison Analysis Tool
=================================

Analyzes and visualizes results from the weather comparison experiment.

Usage:
    python analyze_weather_comparison.py weather_comparison_20260321_120000/
    python analyze_weather_comparison.py weather_comparison_20260321_120000/ --plot
"""

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_comparison_summary(results_dir: str) -> List[Dict]:
    """Load the main comparison CSV."""
    csv_path = os.path.join(results_dir, "WEATHER_COMPARISON_SUMMARY.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Summary CSV not found: {csv_path}")
    
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_percentage(val: str) -> float:
    """Parse percentage string like '+3.07%' to float."""
    if isinstance(val, str):
        return float(val.rstrip('%'))
    return val


def load_detailed_metrics(results_dir: str, label: str) -> Dict[str, float]:
    """Load detailed metrics for a specific model."""
    csv_path = os.path.join(results_dir, f"{label}_metrics.csv")
    if not os.path.exists(csv_path):
        return {}
    
    metrics = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                metrics[row['metric']] = float(row['value'])
            except (ValueError, KeyError):
                pass
    return metrics


def print_summary(comparison_data: List[Dict]) -> None:
    """Print formatted summary table."""
    if not comparison_data:
        print("No comparison data found.")
        return
    
    print("\n" + "="*100)
    print("WEATHER IMPACT SUMMARY")
    print("="*100 + "\n")
    
    print(f"{'Classifier':<20} {'F1 Improvement':>20} {'Acc Improvement':>20} "
          f"{'F1_arr Imp':>15} {'F1_dep Imp':>15}")
    print("-" * 100)
    
    for row in comparison_data:
        classifier = row['classifier']
        f1_imp = parse_percentage(row['f1_improvement'])
        acc_imp = parse_percentage(row['accuracy_improvement'])
        
        f1_with = float(row['f1_with_weather'])
        f1_no = float(row['f1_no_weather'])
        
        print(f"{classifier:<20} {f1_imp:>18.2f}% {acc_imp:>18.2f}%", end="")
        
        # Try to estimate per-channel improvement
        print(f"  {f1_imp:>13.2f}%  {acc_imp:>13.2f}%")


def print_detailed_analysis(results_dir: str, comparison_data: List[Dict]) -> None:
    """Print detailed analysis for each classifier."""
    print("\n" + "="*100)
    print("DETAILED PER-CLASSIFIER ANALYSIS")
    print("="*100 + "\n")
    
    for row in comparison_data:
        clf = row['classifier']
        print(f"\n{clf}")
        print("-" * 80)
        
        # Extract numbers
        f1_with = float(row['f1_with_weather'])
        f1_no = float(row['f1_no_weather'])
        acc_with = float(row['accuracy_with_weather'])
        acc_no = float(row['accuracy_no_weather'])
        params_with = int(row['params_with'])
        params_no = int(row['params_without'])
        time_with = float(row['time_with'].rstrip('s'))
        time_no = float(row['time_without'].rstrip('s'))
        
        f1_imp = parse_percentage(row['f1_improvement'])
        acc_imp = parse_percentage(row['accuracy_improvement'])
        
        print(f"  F1 Score:")
        print(f"    Without weather: {f1_no:.4f}")
        print(f"    With weather:    {f1_with:.4f}")
        print(f"    Improvement:     {f1_imp:+.2f}%")
        
        print(f"\n  Accuracy:")
        print(f"    Without weather: {acc_no:.4f}")
        print(f"    With weather:    {acc_with:.4f}")
        print(f"    Improvement:     {acc_imp:+.2f}%")
        
        print(f"\n  Model Complexity:")
        print(f"    Parameters (no weather):   {params_no:>12,}")
        print(f"    Parameters (with weather): {params_with:>12,}")
        print(f"    Increase:                  {params_with - params_no:>12,} ({(params_with - params_no) / params_no * 100:.1f}%)")
        
        print(f"\n  Training Time:")
        print(f"    Without weather: {time_no:>8.1f}s")
        print(f"    With weather:    {time_with:>8.1f}s")
        print(f"    Overhead:        {time_with - time_no:>8.1f}s ({(time_with - time_no) / time_no * 100:.1f}%)")


def make_recommendation(comparison_data: List[Dict]) -> str:
    """Generate recommendation based on results."""
    if not comparison_data:
        return "No data for recommendation."
    
    improvements = [parse_percentage(row['f1_improvement']) for row in comparison_data]
    avg_improvement = np.mean(improvements)
    max_improvement = np.max(improvements)
    min_improvement = np.min(improvements)
    
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100 + "\n")
    
    print(f"Average F1 improvement across classifiers: {avg_improvement:+.2f}%")
    print(f"Range: {min_improvement:+.2f}% to {max_improvement:+.2f}%\n")
    
    if avg_improvement > 2.0:
        recommendation = (
            "✓ KEEP WEATHER IN PRODUCTION\n"
            "  Weather features provide meaningful improvement to model performance.\n"
            f"  Average gain of {avg_improvement:.2f}% F1 justifies the added complexity."
        )
    elif avg_improvement > 0.5:
        recommendation = (
            "~ MARGINAL BENEFIT\n"
            "  Weather features provide small but consistent improvement.\n"
            f"  {avg_improvement:.2f}% gain may be worth keeping depending on:\n"
            "  • Target system complexity requirements\n"
            "  • Acceptable margin for model improvement\n"
            "  • Weather data acquisition/maintenance costs"
        )
    elif avg_improvement >= -0.5:
        recommendation = (
            "≈ NEGLIGIBLE IMPACT\n"
            f"  Weather features have minimal effect ({avg_improvement:+.2f}%).\n"
            "  Consider:\n"
            "  • Removing weather to simplify model (no performance loss)\n"
            "  • Keeping weather if it's readily available (near-zero cost)\n"
            "  • Investigating alternative weather preprocessing"
        )
    else:
        recommendation = (
            "✗ AVOID WEATHER IN CURRENT FORM\n"
            f"  Weather features degrade performance by {abs(avg_improvement):.2f}%.\n"
            "  Recommendations:\n"
            "  • Remove weather from production model\n"
            "  • Investigate preprocessing (scaling, outlier handling)\n"
            "  • Check for data quality issues in weather file\n"
            "  • Consider alternative weather metrics or sources"
        )
    
    print(recommendation)
    return recommendation


def print_comparison_table(results_dir: str, comparison_data: List[Dict]) -> None:
    """Print detailed comparison table."""
    print("\n" + "="*100)
    print("FULL COMPARISON TABLE")
    print("="*100 + "\n")
    
    header = (
        f"{'Classifier':<20} "
        f"{'F1_no_w':<12} {'F1_with_w':<12} {'F1_Δ':<8} {'F1_%Δ':<10} "
        f"{'Acc_no_w':<12} {'Acc_with_w':<12} {'Acc_%Δ':<10}"
    )
    print(header)
    print("-" * len(header))
    
    for row in comparison_data:
        clf = row['classifier']
        f1_no = float(row['f1_no_weather'])
        f1_w = float(row['f1_with_weather'])
        acc_no = float(row['accuracy_no_weather'])
        acc_w = float(row['accuracy_with_weather'])
        f1_imp = parse_percentage(row['f1_improvement'])
        acc_imp = parse_percentage(row['accuracy_improvement'])
        
        f1_delta = f1_w - f1_no
        acc_delta = acc_w - acc_no
        
        print(f"{clf:<20} "
              f"{f1_no:<12.4f} {f1_w:<12.4f} {f1_delta:<8.4f} {f1_imp:<9.2f}% "
              f"{acc_no:<12.4f} {acc_w:<12.4f} {acc_imp:<9.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Analyze weather comparison results")
    parser.add_argument("results_dir", help="Path to weather_comparison_* output directory")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib plots (requires matplotlib)")
    args = parser.parse_args()
    
    if not os.path.isdir(args.results_dir):
        print(f"ERROR: Directory not found: {args.results_dir}")
        return
    
    # Load data
    try:
        comparison_data = load_comparison_summary(args.results_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return
    
    # Print analyses
    print_summary(comparison_data)
    print_detailed_analysis(args.results_dir, comparison_data)
    print_comparison_table(args.results_dir, comparison_data)
    make_recommendation(comparison_data)
    
    # Optional plotting
    if args.plot:
        try:
            import matplotlib.pyplot as plt
            plot_comparison(comparison_data, args.results_dir)
        except ImportError:
            print("\nNote: matplotlib not available. Install with: pip install matplotlib")


def plot_comparison(comparison_data: List[Dict], results_dir: str) -> None:
    """Generate comparison plots."""
    import matplotlib.pyplot as plt
    
    classifiers = [row['classifier'] for row in comparison_data]
    f1_improvements = [parse_percentage(row['f1_improvement']) for row in comparison_data]
    acc_improvements = [parse_percentage(row['accuracy_improvement']) for row in comparison_data]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # F1 improvement
    axes[0].bar(classifiers, f1_improvements, color=['green' if x > 0 else 'red' for x in f1_improvements])
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[0].set_ylabel('F1 Improvement (%)')
    axes[0].set_title('F1 Score Impact of Weather')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_improvements):
        axes[0].text(i, v + 0.1, f'{v:+.2f}%', ha='center', fontsize=10)
    
    # Accuracy improvement
    axes[1].bar(classifiers, acc_improvements, color=['green' if x > 0 else 'red' for x in acc_improvements])
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].set_ylabel('Accuracy Improvement (%)')
    axes[1].set_title('Accuracy Impact of Weather')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(acc_improvements):
        axes[1].text(i, v + 0.05, f'{v:+.2f}%', ha='center', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "weather_comparison_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {plot_path}")


if __name__ == "__main__":
    main()
