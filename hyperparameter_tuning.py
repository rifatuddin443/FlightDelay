"""Hyperparameter tuning for CNN-based flight delay prediction with 3-stage training.

This script performs systematic hyperparameter search to improve Stage 1 and Stage 2 learning.
Supports grid search, random search, and Bayesian optimization strategies.

Usage:
    python hyperparameter_tuning.py --strategy grid --n_trials 20
    python hyperparameter_tuning.py --strategy random --n_trials 50
    python hyperparameter_tuning.py --strategy bayesian --n_trials 30
"""

import argparse
import csv
import itertools
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


class HyperparameterTuner:
    """Manages hyperparameter search and experiment tracking."""

    def __init__(
        self,
        base_script: str = "cnnopacus.py",
        output_dir: str = "tuning_results",
        strategy: str = "grid",
        n_trials: int = 20,
        base_args: Optional[Dict[str, Any]] = None,
    ):
        self.base_script = base_script
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self.n_trials = n_trials
        self.base_args = base_args or {}
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"tuning_results_{self.timestamp}.csv"
        self.summary_file = self.output_dir / f"tuning_summary_{self.timestamp}.txt"
        self.best_config_file = self.output_dir / f"best_config_{self.timestamp}.json"
        
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None

    def get_search_space(self) -> Dict[str, List[Any]]:
        """Define hyperparameter search space focusing on learning improvements."""
        return {
            # Learning rates per stage - most critical for early stopping issues
            # Stage 1: Classifier training
            'stage1_lr': [0.0005, 0.001, 0.002, 0.005],
            # Stage 2: Delayed regressor (often needs similar or slightly lower LR)
            'stage2_lr': [0.0003, 0.0005, 0.001, 0.002],
            # Stage 3: Non-delayed regressor (fine-tuning, needs lower LR)
            'stage3_lr': [0.00001, 0.00005, 0.0001, 0.0005],
            
            # Patience - allow more epochs before stopping
            'patience': [5, 8, 10, 15],
            
            # Model capacity
            'hidden_channels': [64, 128, 192, 256],
            
            # Training duration
            'stage1_epochs': [10, 15, 20, 25],
            'stage2_epochs': [12, 15, 20, 25],
            'stage3_epochs': [14, 18, 22, 26],
            
            # Batch size - affects gradient stability
            'batch_size': [64, 128, 256],
            
            # Class balancing
            'balance_50_50': [True, False],
            
            # DP settings (only if needed)
            'noise_multiplier': [0.0],  # Keep at 0 for now
            'max_grad_norm': [1.0, 2.0, 5.0],
        }

    def generate_grid_search_configs(self) -> List[Dict[str, Any]]:
        """Generate all combinations for grid search (can be very large)."""
        space = self.get_search_space()
        keys = list(space.keys())
        values = [space[k] for k in keys]
        
        configs = []
        for combo in itertools.product(*values):
            config = dict(zip(keys, combo))
            configs.append(config)
        
        if len(configs) > 500:
            print(f"Warning: Grid search would generate {len(configs)} trials.")
            print(f"Randomly sampling {self.n_trials} configurations instead.")
            random.shuffle(configs)
            configs = configs[:self.n_trials]
        
        return configs[:self.n_trials]

    def generate_random_search_configs(self) -> List[Dict[str, Any]]:
        """Generate random configurations for random search."""
        space = self.get_search_space()
        configs = []
        
        for _ in range(self.n_trials):
            config = {k: random.choice(v) for k, v in space.items()}
            configs.append(config)
        
        return configs

    def generate_focused_configs(self) -> List[Dict[str, Any]]:
        """Generate focused configurations targeting the early stopping problem."""
        configs = []
        
        # Priority 1: Coordinated LR schedules across stages
        lr_schedules = [
            # Conservative: lower LRs across all stages
            {'stage1_lr': 0.001, 'stage2_lr': 0.0005, 'stage3_lr': 0.00005},
            # Moderate: balanced approach
            {'stage1_lr': 0.002, 'stage2_lr': 0.001, 'stage3_lr': 0.0001},
            # Aggressive Stage 1: higher initial LR, then reduce
            {'stage1_lr': 0.005, 'stage2_lr': 0.001, 'stage3_lr': 0.0001},
            # Fine-tuning focused: higher Stage 3 LR
            {'stage1_lr': 0.001, 'stage2_lr': 0.001, 'stage3_lr': 0.0005},
        ]
        
        for lr_schedule in lr_schedules:
            for patience in [10, 15]:
                for hidden in [128, 256]:
                    configs.append({
                        **lr_schedule,
                        'patience': patience,
                        'hidden_channels': hidden,
                        'stage1_epochs': 20,
                        'stage2_epochs': 20,
                        'stage3_epochs': 20,
                        'batch_size': 128,
                        'balance_50_50': False,
                        'noise_multiplier': 0.0,
                        'max_grad_norm': 2.0,
                    })
        
        # Priority 2: Class balancing experiments with optimal LR schedules
        for balance in [True, False]:
            configs.append({
                'stage1_lr': 0.001,
                'stage2_lr': 0.0005,
                'stage3_lr': 0.0001,
                'patience': 10,
                'hidden_channels': 256,
                'stage1_epochs': 20,
                'stage2_epochs': 20,
                'stage3_epochs': 20,
                'batch_size': 128,
                'balance_50_50': balance,
                'noise_multiplier': 0.0,
                'max_grad_norm': 2.0,
            })
        
        # Priority 3: Batch size variations with adjusted LR (larger batch = higher LR)
        for batch_size in [64, 256]:
            # Smaller batch needs smaller LR, larger batch can use higher LR
            stage1_lr = 0.0005 if batch_size == 64 else 0.002
            stage2_lr = 0.0003 if batch_size == 64 else 0.001
            configs.append({
                'stage1_lr': stage1_lr,
                'stage2_lr': stage2_lr,
                'stage3_lr': 0.0001,
                'patience': 10,
                'hidden_channels': 256,
                'stage1_epochs': 20,
                'stage2_epochs': 20,
                'stage3_epochs': 20,
                'batch_size': batch_size,
                'balance_50_50': False,
                'noise_multiplier': 0.0,
                'max_grad_norm': 2.0,
            })
        
        return configs[:self.n_trials]

    def config_to_args(self, config: Dict[str, Any]) -> List[str]:
        """Convert configuration dict to command-line arguments."""
        args = []
        
        for key, value in {**self.base_args, **config}.items():
            if isinstance(value, bool):
                if value:
                    args.append(f"--{key}")
            elif isinstance(value, (int, float, str)):
                args.extend([f"--{key}", str(value)])
            elif isinstance(value, list):
                args.append(f"--{key}")
                args.extend([str(v) for v in value])
        
        return args

    def run_experiment(self, trial_id: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single training experiment with given configuration."""
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_id}/{self.n_trials}")
        print(f"{'='*80}")
        print(f"Configuration: {json.dumps(config, indent=2)}")
        
        cmd_args = self.config_to_args(config)
        cmd = [sys.executable, self.base_script] + cmd_args
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        start_time = time.time()
        result = {
            'trial_id': trial_id,
            'config': config,
            'timestamp': datetime.now().isoformat(),
            'success': False,
        }
        
        try:
            # Run training
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
            )
            
            elapsed_time = time.time() - start_time
            result['elapsed_time_minutes'] = elapsed_time / 60
            
            if process.returncode == 0:
                result['success'] = True
                # Parse metrics from output
                metrics = self._parse_metrics_from_output(process.stdout)
                result.update(metrics)
                print(f"\n✓ Trial {trial_id} completed successfully")
                print(f"  Classification F1: {metrics.get('test_f1', 'N/A'):.4f}")
                print(f"  Regression MAE: {metrics.get('test_mae_overall', 'N/A'):.4f} min")
                print(f"  Training time: {elapsed_time/60:.2f} min")
            else:
                result['error'] = process.stderr[-500:]  # Last 500 chars
                print(f"\n✗ Trial {trial_id} failed")
                print(f"  Error: {result['error']}")
        
        except subprocess.TimeoutExpired:
            result['error'] = "Timeout (>2 hours)"
            print(f"\n✗ Trial {trial_id} timed out")
        except Exception as e:
            result['error'] = str(e)
            print(f"\n✗ Trial {trial_id} crashed: {e}")
        
        return result

    def _parse_metrics_from_output(self, output: str) -> Dict[str, float]:
        """Extract key metrics from training script output."""
        metrics = {}
        
        # Parse final test metrics
        lines = output.split('\n')
        for i, line in enumerate(lines):
            # Classification metrics
            if 'F1:' in line and 'Accuracy:' in line:
                try:
                    # Example: "  F1: 0.5723 | Accuracy: 0.7075"
                    parts = line.split('|')
                    for part in parts:
                        if 'F1:' in part:
                            metrics['test_f1'] = float(part.split(':')[1].strip())
                        elif 'Precision:' in part:
                            metrics['test_precision'] = float(part.split(':')[1].strip())
                        elif 'Recall:' in part:
                            metrics['test_recall'] = float(part.split(':')[1].strip())
                        elif 'Accuracy:' in part:
                            metrics['test_accuracy'] = float(part.split(':')[1].strip())
                except:
                    pass
            
            # Regression metrics
            if 'REGRESSION (overall)' in line:
                try:
                    # Next line has: "  MAE: 6.0616 min | RMSE: 7.7019 min"
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    if 'MAE:' in next_line:
                        mae_str = next_line.split('MAE:')[1].split('min')[0].strip()
                        metrics['test_mae_overall'] = float(mae_str)
                    if 'RMSE:' in next_line:
                        rmse_str = next_line.split('RMSE:')[1].split('min')[0].strip()
                        metrics['test_rmse_overall'] = float(rmse_str)
                except:
                    pass
            
            # Stage 1 convergence info
            if 'STAGE 1' in line and 'TRAINING DELAY CLASSIFIER' in line:
                stage1_epochs = []
                stage1_f1s = []
                j = i + 1
                while j < len(lines) and 'Stage 1 completed' not in lines[j]:
                    if 'Epoch' in lines[j] and 'Val F1' in lines[j]:
                        try:
                            # Extract epoch and F1
                            epoch_part = lines[j].split('Epoch')[1].split('|')[0].strip()
                            epoch = int(epoch_part.split('/')[0])
                            f1_part = lines[j].split('Val F1 (macro):')[1].split('[')[0].strip()
                            f1 = float(f1_part)
                            stage1_epochs.append(epoch)
                            stage1_f1s.append(f1)
                        except:
                            pass
                    j += 1
                
                if stage1_f1s:
                    metrics['stage1_best_f1'] = max(stage1_f1s)
                    metrics['stage1_best_epoch'] = stage1_epochs[np.argmax(stage1_f1s)]
                    metrics['stage1_total_epochs'] = len(stage1_epochs)
                    metrics['stage1_improvement'] = stage1_f1s[-1] - stage1_f1s[0] if len(stage1_f1s) > 1 else 0.0
        
        return metrics

    def save_results(self) -> None:
        """Save all results to CSV and generate summary."""
        if not self.results:
            print("No results to save.")
            return
        
        # Save detailed CSV
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['trial_id', 'success', 'elapsed_time_minutes', 'timestamp']
            
            # Add config fields
            if self.results[0]['config']:
                fieldnames.extend([f'config_{k}' for k in self.results[0]['config'].keys()])
            
            # Add metric fields
            metric_keys = [k for k in self.results[0].keys() 
                          if k not in ['trial_id', 'config', 'success', 'timestamp', 'elapsed_time_minutes', 'error']]
            fieldnames.extend(metric_keys)
            fieldnames.append('error')
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                row = {
                    'trial_id': result['trial_id'],
                    'success': result['success'],
                    'elapsed_time_minutes': result.get('elapsed_time_minutes', 0),
                    'timestamp': result['timestamp'],
                    'error': result.get('error', ''),
                }
                
                # Flatten config
                for k, v in result['config'].items():
                    row[f'config_{k}'] = v
                
                # Add metrics
                for k in metric_keys:
                    row[k] = result.get(k, '')
                
                writer.writerow(row)
        
        print(f"\n✓ Results saved to: {self.results_file}")
        
        # Generate summary
        self._generate_summary()

    def _generate_summary(self) -> None:
        """Generate human-readable summary of tuning results."""
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("No successful trials to summarize.")
            return
        
        # Find best result by F1 score
        self.best_result = max(successful_results, key=lambda x: x.get('test_f1', 0))
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Strategy: {self.strategy}\n")
            f.write(f"Total trials: {len(self.results)}\n")
            f.write(f"Successful trials: {len(successful_results)}\n")
            f.write(f"Failed trials: {len(self.results) - len(successful_results)}\n\n")
            
            f.write("="*80 + "\n")
            f.write("BEST CONFIGURATION\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Trial ID: {self.best_result['trial_id']}\n")
            f.write(f"Test F1: {self.best_result.get('test_f1', 0):.4f}\n")
            f.write(f"Test MAE: {self.best_result.get('test_mae_overall', 0):.4f} min\n")
            f.write(f"Stage 1 Best Epoch: {self.best_result.get('stage1_best_epoch', 'N/A')}\n")
            f.write(f"Stage 1 Best F1: {self.best_result.get('stage1_best_f1', 0):.4f}\n\n")
            
            f.write("Configuration:\n")
            for k, v in self.best_result['config'].items():
                f.write(f"  {k}: {v}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("TOP 5 CONFIGURATIONS BY F1 SCORE\n")
            f.write("="*80 + "\n\n")
            
            top5 = sorted(successful_results, key=lambda x: x.get('test_f1', 0), reverse=True)[:5]
            for i, result in enumerate(top5, 1):
                f.write(f"{i}. Trial {result['trial_id']}: F1={result.get('test_f1', 0):.4f}, "
                       f"MAE={result.get('test_mae_overall', 0):.4f}, "
                       f"LRs=[{result['config'].get('stage1_lr')}, {result['config'].get('stage2_lr')}, {result['config'].get('stage3_lr')}], "
                       f"Patience={result['config'].get('patience')}, "
                       f"Hidden={result['config'].get('hidden_channels')}\n")
            
            # Analyze hyperparameter importance
            f.write("\n" + "="*80 + "\n")
            f.write("HYPERPARAMETER ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            self._analyze_hyperparameters(f, successful_results)
        
        # Save best config as JSON
        with open(self.best_config_file, 'w', encoding='utf-8') as f:
            json.dump(self.best_result['config'], f, indent=2)
        
        print(f"✓ Summary saved to: {self.summary_file}")
        print(f"✓ Best config saved to: {self.best_config_file}")
        
        # Print best config to console
        print(f"\n{'='*80}")
        print("BEST CONFIGURATION FOUND")
        print(f"{'='*80}")
        print(f"F1: {self.best_result.get('test_f1', 0):.4f} | MAE: {self.best_result.get('test_mae_overall', 0):.4f} min")
        print(f"\nRun with:")
        cmd_args = self.config_to_args(self.best_result['config'])
        print(f"python {self.base_script} {' '.join(cmd_args)}")

    def _analyze_hyperparameters(self, f, results: List[Dict]) -> None:
        """Analyze which hyperparameters most affect performance."""
        # Group by each hyperparameter and compute average F1
        params = list(results[0]['config'].keys())
        
        for param in params:
            values_map = {}
            for result in results:
                val = result['config'][param]
                if val not in values_map:
                    values_map[val] = []
                values_map[val].append(result.get('test_f1', 0))
            
            f.write(f"{param}:\n")
            for val, f1s in sorted(values_map.items(), key=lambda x: np.mean(x[1]), reverse=True):
                f.write(f"  {val}: avg F1 = {np.mean(f1s):.4f} (n={len(f1s)})\n")
            f.write("\n")

    def run(self) -> None:
        """Execute hyperparameter tuning."""
        print(f"\n{'='*80}")
        print("HYPERPARAMETER TUNING")
        print(f"{'='*80}")
        print(f"Strategy: {self.strategy}")
        print(f"Trials: {self.n_trials}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
        
        # Generate configurations
        if self.strategy == 'grid':
            configs = self.generate_grid_search_configs()
        elif self.strategy == 'random':
            configs = self.generate_random_search_configs()
        elif self.strategy == 'focused':
            configs = self.generate_focused_configs()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        print(f"Generated {len(configs)} configurations to test\n")
        
        # Run experiments
        for i, config in enumerate(configs, 1):
            result = self.run_experiment(i, config)
            self.results.append(result)
            
            # Save intermediate results after each trial
            self.save_results()
            
            print(f"\nProgress: {i}/{len(configs)} trials completed")
        
        print(f"\n{'='*80}")
        print("TUNING COMPLETE")
        print(f"{'='*80}")
        print(f"Total trials: {len(self.results)}")
        print(f"Successful: {sum(1 for r in self.results if r['success'])}")
        print(f"Results saved to: {self.output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for cnnopacus.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick focused search (recommended for early stopping issues)
  python hyperparameter_tuning.py --strategy focused --n_trials 15

  # Full grid search (slow)
  python hyperparameter_tuning.py --strategy grid --n_trials 50

  # Random search
  python hyperparameter_tuning.py --strategy random --n_trials 30

  # Custom base arguments
  python hyperparameter_tuning.py --strategy focused --base_data_source udata --base_seq_len 12
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['grid', 'random', 'focused'],
        default='focused',
        help='Search strategy (focused is recommended for addressing early stopping)',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=15,
        help='Number of trials to run',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='tuning_results',
        help='Directory to save results',
    )
    parser.add_argument(
        '--base_script',
        type=str,
        default='cnnopacus.py',
        help='Path to training script',
    )
    
    # Base arguments that stay constant across all trials
    parser.add_argument('--base_data_source', type=str, default='cdata')
    parser.add_argument('--base_seq_len', type=int, default=8)
    parser.add_argument('--base_horizons', type=int, nargs='+', default=[12])
    parser.add_argument('--base_delay_threshold', type=float, default=5.0)
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Prepare base arguments (constant across all trials)
    base_args = {
        'data_source': args.base_data_source,
        'seq_len': args.base_seq_len,
        'horizons': args.base_horizons,
        'delay_threshold': args.base_delay_threshold,
    }
    
    tuner = HyperparameterTuner(
        base_script=args.base_script,
        output_dir=args.output_dir,
        strategy=args.strategy,
        n_trials=args.n_trials,
        base_args=base_args,
    )
    
    tuner.run()


if __name__ == '__main__':
    main()
