"""Comprehensive learning rate tuning for 3-stage CNN training.

Focuses exclusively on finding optimal per-stage learning rates with a wide search space.
All other hyperparameters are fixed to reasonable defaults.

Usage:
    python tune_learning_rates.py --n_trials 50
    python tune_learning_rates.py --strategy exhaustive --n_trials 100
"""

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np


class LearningRateTuner:
    """Specialized tuner for learning rate optimization."""

    def __init__(
        self,
        base_script: str = "cnnopacus.py",
        output_dir: str = "lr_tuning_results",
        strategy: str = "comprehensive",
        n_trials: int = 50,
    ):
        self.base_script = base_script
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self.n_trials = n_trials
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_file = self.output_dir / f"lr_tuning_results_{self.timestamp}.csv"
        self.summary_file = self.output_dir / f"lr_tuning_summary_{self.timestamp}.txt"
        self.best_config_file = self.output_dir / f"best_lr_config_{self.timestamp}.json"
        
        self.results: List[Dict[str, Any]] = []
        self.best_result: Optional[Dict[str, Any]] = None
        
        # Fixed hyperparameters (not tuned)
        self.fixed_params = {
            'data_source': 'cdata',
            'seq_len': 24,
            'horizons': [12],
            'delay_threshold': 5.0,
            'patience': 10,  # Fixed at 10 for consistent comparison
            'hidden_channels': 256,  # Larger model
            'stage1_epochs': 20,
            'stage2_epochs': 20,
            'stage3_epochs': 20,
            'batch_size': 128,
            'balance_50_50': False,
            'noise_multiplier': 0.0,
            'max_grad_norm': 2.0,
            'skip_visualization': True,  # Disable visualization during tuning
        }

    def get_lr_search_space(self) -> Dict[str, List[float]]:
        """Define comprehensive learning rate search space.
        
        All stages get THE SAME range to fairly test if differentiation is needed.
        """
        if self.strategy == 'exhaustive':
            # Very dense grid - SAME RANGE FOR ALL STAGES
            return {
                'stage1_lr': [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01],
                'stage2_lr': [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01],
                'stage3_lr': [0.00001, 0.00005, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.005, 0.007, 0.01],
            }
        elif self.strategy == 'comprehensive':
            # Wide range - SAME FOR ALL STAGES
            return {
                'stage1_lr': [0.00005, 0.0001, 0.0003, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01],
                'stage2_lr': [0.00005, 0.0001, 0.0003, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01],
                'stage3_lr': [0.00005, 0.0001, 0.0003, 0.0008, 0.001, 0.003, 0.005, 0.008, 0.01],
            }
        elif self.strategy == 'coarse':
            # Quick exploration - SAME FOR ALL
            return {
                'stage1_lr': [0.0001, 0.0005, 0.001, 0.002, 0.005],
                'stage2_lr': [0.0001, 0.0005, 0.001, 0.002, 0.005],
                'stage3_lr': [0.0001, 0.0005, 0.001, 0.002, 0.005],
            }
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def generate_lr_configs(self) -> List[Dict[str, float]]:
        """Generate learning rate configurations exploring both unified and differentiated schedules.
        
        Explores:
        1. Unified: Same LR for all stages (tests if differentiation is needed)
        2. Differentiated: Different LRs per stage (various patterns)
        """
        space = self.get_lr_search_space()
        configs = []
        
        stage1_lrs = space['stage1_lr']
        stage2_lrs = space['stage2_lr']
        stage3_lrs = space['stage3_lr']
        
        if self.strategy == 'exhaustive':
            # Full grid search - test ALL combinations without constraints
            print("Generating exhaustive grid (no constraints)...")
            for s1_lr in stage1_lrs:
                for s2_lr in stage2_lrs:
                    for s3_lr in stage3_lrs:
                        configs.append({
                            'stage1_lr': s1_lr,
                            'stage2_lr': s2_lr,
                            'stage3_lr': s3_lr,
                        })
            print(f"Generated {len(configs)} exhaustive configurations")
            
        elif self.strategy in ['comprehensive', 'coarse']:
            # PATTERN 1: UNIFIED - Same LR for all stages (HIGH PRIORITY)
            print("\nGenerating unified LR configurations (same for all stages)...")
            unified_count = 0
            for lr in stage1_lrs:
                configs.append({
                    'stage1_lr': lr,
                    'stage2_lr': lr,
                    'stage3_lr': lr,
                })
                unified_count += 1
            print(f"  -> {unified_count} unified configs")
            
            # PATTERN 2: Decreasing schedules (S1 >= S2 >= S3)
            print("Generating decreasing LR schedules...")
            decreasing_count = 0
            for s1_lr in stage1_lrs:
                for s2_lr in stage2_lrs:
                    if s2_lr > s1_lr:
                        continue
                    for s3_lr in stage3_lrs:
                        if s3_lr > s2_lr:
                            continue
                        # Skip if already added as unified
                        if s1_lr == s2_lr == s3_lr:
                            continue
                        configs.append({
                            'stage1_lr': s1_lr,
                            'stage2_lr': s2_lr,
                            'stage3_lr': s3_lr,
                        })
                        decreasing_count += 1
            print(f"  -> {decreasing_count} decreasing configs")
            
            # PATTERN 3: Increasing schedules (S1 <= S2 <= S3)
            print("Generating increasing LR schedules...")
            increasing_count = 0
            for s1_lr in stage1_lrs:
                for s2_lr in stage2_lrs:
                    if s2_lr < s1_lr:
                        continue
                    for s3_lr in stage3_lrs:
                        if s3_lr < s2_lr:
                            continue
                        # Skip if already added
                        if s1_lr == s2_lr == s3_lr:
                            continue
                        configs.append({
                            'stage1_lr': s1_lr,
                            'stage2_lr': s2_lr,
                            'stage3_lr': s3_lr,
                        })
                        increasing_count += 1
            print(f"  -> {increasing_count} increasing configs")
            
            # PATTERN 4: V-shaped (high-low-high or low-high-low)
            print("Generating V-shaped schedules...")
            v_shaped_count = 0
            for s1_lr in stage1_lrs[::2]:  # Sample every other to reduce configs
                for s2_lr in stage2_lrs:
                    for s3_lr in stage3_lrs[::2]:
                        # V-shape: S2 is lowest or highest
                        is_v_down = (s1_lr >= s2_lr and s3_lr >= s2_lr and not (s1_lr == s2_lr == s3_lr))
                        is_v_up = (s1_lr <= s2_lr and s3_lr <= s2_lr and not (s1_lr == s2_lr == s3_lr))
                        if is_v_down or is_v_up:
                            configs.append({
                                'stage1_lr': s1_lr,
                                'stage2_lr': s2_lr,
                                'stage3_lr': s3_lr,
                            })
                            v_shaped_count += 1
            print(f"  -> {v_shaped_count} V-shaped configs")
        
        # Remove duplicates
        unique_configs = []
        seen = set()
        for config in configs:
            key = (config['stage1_lr'], config['stage2_lr'], config['stage3_lr'])
            if key not in seen:
                seen.add(key)
                unique_configs.append(config)
        
        print(f"Generated {len(unique_configs)} unique LR configurations")
        
        # Limit to n_trials if too many
        if len(unique_configs) > self.n_trials:
            print(f"Sampling {self.n_trials} from {len(unique_configs)} configurations")
            # Prioritize decreasing schedules
            import random
            random.seed(42)
            unique_configs.sort(key=lambda x: x['stage1_lr'], reverse=True)
            sampled = unique_configs[:self.n_trials // 2]
            sampled += random.sample(unique_configs[self.n_trials // 2:], 
                                    min(self.n_trials - len(sampled), len(unique_configs) - self.n_trials // 2))
            unique_configs = sampled
        
        return unique_configs[:self.n_trials]

    def run_experiment(self, trial_id: int, lr_config: Dict[str, float]) -> Dict[str, Any]:
        """Run a single training experiment with given LR configuration."""
        print(f"\n{'='*80}")
        print(f"TRIAL {trial_id}/{self.n_trials}")
        print(f"{'='*80}")
        print(f"Learning Rates: Stage1={lr_config['stage1_lr']:.6f}, "
              f"Stage2={lr_config['stage2_lr']:.6f}, Stage3={lr_config['stage3_lr']:.6f}")
        
        # Merge LR config with fixed params
        full_config = {**self.fixed_params, **lr_config}
        
        # Build command
        cmd = [sys.executable, self.base_script]
        for key, value in full_config.items():
            if isinstance(value, bool):
                if value:
                    cmd.append(f"--{key}")
            elif isinstance(value, list):
                cmd.append(f"--{key}")
                cmd.extend([str(v) for v in value])
            else:
                cmd.extend([f"--{key}", str(value)])
        
        print(f"\nFixed params: patience={self.fixed_params['patience']}, "
              f"hidden={self.fixed_params['hidden_channels']}, "
              f"batch={self.fixed_params['batch_size']}, "
              f"epochs=[{self.fixed_params['stage1_epochs']}, "
              f"{self.fixed_params['stage2_epochs']}, "
              f"{self.fixed_params['stage3_epochs']}]")
        
        start_time = time.time()
        result = {
            'trial_id': trial_id,
            'lr_config': lr_config,
            'timestamp': datetime.now().isoformat(),
            'success': False,
        }
        
        try:
            # Run training with real-time output streaming
            import subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                universal_newlines=True
            )
            
            # Stream output in real-time and collect for parsing
            output_lines = []
            for line in process.stdout:
                print(line, end='')  # Print in real-time
                output_lines.append(line)
            
            process.wait(timeout=7200)  # 2 hour timeout
            output = ''.join(output_lines)
            
            elapsed_time = time.time() - start_time
            result['elapsed_time_minutes'] = elapsed_time / 60
            
            if process.returncode == 0:
                result['success'] = True
                # Parse metrics from output
                metrics = self._parse_metrics_from_output(output)
                result.update(metrics)
                
                # Calculate LR schedule score (how well did convergence work?)
                schedule_score = self._calculate_schedule_score(metrics)
                result['schedule_score'] = schedule_score
                
                print(f"\n✓ Trial {trial_id} completed successfully")
                print(f"  Classification F1: {metrics.get('test_f1', 0):.4f}")
                print(f"  Regression MAE: {metrics.get('test_mae_overall', 0):.4f} min")
                print(f"  Stage 1 Best Epoch: {metrics.get('stage1_best_epoch', 'N/A')}")
                print(f"  Stage 1 Improvement: {metrics.get('stage1_improvement', 0):.4f}")
                print(f"  Schedule Score: {schedule_score:.4f}")
                print(f"  Training time: {elapsed_time/60:.2f} min")
            else:
                result['error'] = output[-500:] if output else "Unknown error"
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
        lines = output.split('\n')
        
        for i, line in enumerate(lines):
            # Classification metrics
            if 'F1:' in line and 'Accuracy:' in line and 'CLASSIFICATION' in output[max(0, i-5):i]:
                try:
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
                    next_line = lines[i + 1] if i + 1 < len(lines) else ""
                    if 'MAE:' in next_line:
                        mae_str = next_line.split('MAE:')[1].split('min')[0].strip()
                        metrics['test_mae_overall'] = float(mae_str)
                    if 'RMSE:' in next_line:
                        rmse_str = next_line.split('RMSE:')[1].split('min')[0].strip()
                        metrics['test_rmse_overall'] = float(rmse_str)
                except:
                    pass
            
            # Stage 1 convergence analysis
            if 'STAGE 1' in line and 'TRAINING DELAY CLASSIFIER' in line:
                stage1_epochs = []
                stage1_f1s = []
                stage1_losses = []
                j = i + 1
                while j < len(lines) and 'Stage 1 completed' not in lines[j]:
                    if 'Epoch' in lines[j] and 'Val F1' in lines[j]:
                        try:
                            epoch_part = lines[j].split('Epoch')[1].split('|')[0].strip()
                            epoch = int(epoch_part.split('/')[0])
                            
                            loss_part = lines[j].split('Loss:')[1].split('|')[0].strip()
                            loss = float(loss_part)
                            
                            f1_part = lines[j].split('Val F1 (macro):')[1].split('[')[0].strip()
                            f1 = float(f1_part)
                            
                            stage1_epochs.append(epoch)
                            stage1_losses.append(loss)
                            stage1_f1s.append(f1)
                        except:
                            pass
                    j += 1
                
                if stage1_f1s:
                    metrics['stage1_best_f1'] = max(stage1_f1s)
                    metrics['stage1_best_epoch'] = stage1_epochs[np.argmax(stage1_f1s)]
                    metrics['stage1_total_epochs'] = len(stage1_epochs)
                    metrics['stage1_final_f1'] = stage1_f1s[-1]
                    metrics['stage1_improvement'] = stage1_f1s[-1] - stage1_f1s[0] if len(stage1_f1s) > 1 else 0.0
                    
                    # Loss convergence
                    if stage1_losses:
                        metrics['stage1_initial_loss'] = stage1_losses[0]
                        metrics['stage1_final_loss'] = stage1_losses[-1]
                        metrics['stage1_loss_reduction'] = stage1_losses[0] - stage1_losses[-1]
            
            # Stage 2 convergence
            if 'STAGE 2' in line and 'DELAYED FLIGHTS' in line:
                stage2_losses = []
                j = i + 1
                while j < len(lines) and 'Stage 2 completed' not in lines[j]:
                    if 'Epoch' in lines[j] and 'Val Loss:' in lines[j]:
                        try:
                            val_loss_part = lines[j].split('Val Loss:')[1].split('|')[0].strip()
                            val_loss = float(val_loss_part)
                            stage2_losses.append(val_loss)
                        except:
                            pass
                    j += 1
                
                if stage2_losses:
                    metrics['stage2_best_loss'] = min(stage2_losses)
                    metrics['stage2_final_loss'] = stage2_losses[-1]
                    metrics['stage2_loss_reduction'] = stage2_losses[0] - stage2_losses[-1]
        
        return metrics

    def _calculate_schedule_score(self, metrics: Dict[str, float]) -> float:
        """Calculate a score indicating how good the LR schedule was for convergence."""
        score = 0.0
        
        # Stage 1: Did it improve beyond epoch 1?
        if metrics.get('stage1_best_epoch', 1) > 1:
            score += 1.0
        
        # Stage 1: Improvement magnitude
        improvement = metrics.get('stage1_improvement', 0)
        if improvement > 0.05:
            score += 2.0
        elif improvement > 0.01:
            score += 1.0
        
        # Stage 1: Loss reduction
        loss_reduction = metrics.get('stage1_loss_reduction', 0)
        if loss_reduction > 0.01:
            score += 1.0
        
        # Stage 2: Loss reduction
        stage2_reduction = metrics.get('stage2_loss_reduction', 0)
        if stage2_reduction > 0.01:
            score += 1.0
        
        # Overall performance
        f1 = metrics.get('test_f1', 0)
        mae = metrics.get('test_mae_overall', 100)
        
        # F1 score bonus
        if f1 > 0.60:
            score += 3.0
        elif f1 > 0.55:
            score += 2.0
        elif f1 > 0.50:
            score += 1.0
        
        # MAE bonus (lower is better)
        if mae < 5.5:
            score += 2.0
        elif mae < 6.0:
            score += 1.0
        
        return score

    def save_results(self) -> None:
        """Save results and generate analysis."""
        if not self.results:
            return
        
        # Save CSV
        with open(self.results_file, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['trial_id', 'success', 'elapsed_time_minutes', 'timestamp',
                         'stage1_lr', 'stage2_lr', 'stage3_lr']
            
            # Add metric fields
            metric_keys = [k for k in self.results[0].keys() 
                          if k not in ['trial_id', 'lr_config', 'success', 'timestamp', 
                                      'elapsed_time_minutes', 'error']]
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
                    'stage1_lr': result['lr_config']['stage1_lr'],
                    'stage2_lr': result['lr_config']['stage2_lr'],
                    'stage3_lr': result['lr_config']['stage3_lr'],
                    'error': result.get('error', ''),
                }
                
                for k in metric_keys:
                    row[k] = result.get(k, '')
                
                writer.writerow(row)
        
        print(f"\n✓ Results saved to: {self.results_file}")
        self._generate_summary()

    def _generate_summary(self) -> None:
        """Generate comprehensive summary and visualizations."""
        successful_results = [r for r in self.results if r['success']]
        
        if not successful_results:
            print("No successful trials to summarize.")
            return
        
        # Find best by multiple criteria
        best_f1 = max(successful_results, key=lambda x: x.get('test_f1', 0))
        best_mae = min(successful_results, key=lambda x: x.get('test_mae_overall', 100))
        best_schedule = max(successful_results, key=lambda x: x.get('schedule_score', 0))
        best_convergence = max(successful_results, key=lambda x: x.get('stage1_improvement', 0))
        
        self.best_result = best_f1  # Use F1 as primary metric
        
        with open(self.summary_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            f.write("LEARNING RATE TUNING SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Strategy: {self.strategy}\n")
            f.write(f"Total trials: {len(self.results)}\n")
            f.write(f"Successful trials: {len(successful_results)}\n\n")
            
            f.write("Fixed Parameters:\n")
            for k, v in self.fixed_params.items():
                f.write(f"  {k}: {v}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("BEST CONFIGURATIONS\n")
            f.write("="*80 + "\n\n")
            
            f.write("1. Best F1 Score:\n")
            self._write_config_details(f, best_f1)
            
            f.write("\n2. Best MAE (Lowest):\n")
            self._write_config_details(f, best_mae)
            
            f.write("\n3. Best Learning Schedule (Convergence):\n")
            self._write_config_details(f, best_schedule)
            
            f.write("\n4. Best Stage 1 Improvement:\n")
            self._write_config_details(f, best_convergence)
            
            f.write("\n" + "="*80 + "\n")
            f.write("TOP 10 BY F1 SCORE\n")
            f.write("="*80 + "\n\n")
            
            top10 = sorted(successful_results, key=lambda x: x.get('test_f1', 0), reverse=True)[:10]
            for i, result in enumerate(top10, 1):
                lr_cfg = result['lr_config']
                f.write(f"{i:2d}. F1={result.get('test_f1', 0):.4f} MAE={result.get('test_mae_overall', 0):.2f} "
                       f"| LRs: [{lr_cfg['stage1_lr']:.6f}, {lr_cfg['stage2_lr']:.6f}, {lr_cfg['stage3_lr']:.6f}] "
                       f"| S1_epoch={result.get('stage1_best_epoch', 'N/A')} "
                       f"| S1_impr={result.get('stage1_improvement', 0):.4f}\n")
            
            # Analyze LR patterns
            f.write("\n" + "="*80 + "\n")
            f.write("LEARNING RATE PATTERN ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            self._analyze_lr_patterns(f, successful_results)
        
        # Save best config
        with open(self.best_config_file, 'w', encoding='utf-8') as f:
            json.dump({**self.fixed_params, **self.best_result['lr_config']}, f, indent=2)
        
        print(f"✓ Summary saved to: {self.summary_file}")
        print(f"✓ Best config saved to: {self.best_config_file}")
        
        # Print to console
        print(f"\n{'='*80}")
        print("BEST LEARNING RATE CONFIGURATION")
        print(f"{'='*80}")
        lr_cfg = self.best_result['lr_config']
        print(f"F1: {self.best_result.get('test_f1', 0):.4f} | MAE: {self.best_result.get('test_mae_overall', 0):.4f} min")
        print(f"\nLearning Rates:")
        print(f"  Stage 1 (Classifier):       {lr_cfg['stage1_lr']:.6f}")
        print(f"  Stage 2 (Delayed Reg):      {lr_cfg['stage2_lr']:.6f}")
        print(f"  Stage 3 (Non-Delayed Reg):  {lr_cfg['stage3_lr']:.6f}")
        print(f"\nRun with:")
        print(f"python cnnopacus.py --stage1_lr {lr_cfg['stage1_lr']} "
              f"--stage2_lr {lr_cfg['stage2_lr']} --stage3_lr {lr_cfg['stage3_lr']} "
              f"--patience {self.fixed_params['patience']} "
              f"--hidden_channels {self.fixed_params['hidden_channels']}")

    def _write_config_details(self, f, result: Dict) -> None:
        """Write detailed configuration info to file."""
        lr_cfg = result['lr_config']
        f.write(f"  Trial ID: {result['trial_id']}\n")
        f.write(f"  Stage 1 LR: {lr_cfg['stage1_lr']:.6f}\n")
        f.write(f"  Stage 2 LR: {lr_cfg['stage2_lr']:.6f}\n")
        f.write(f"  Stage 3 LR: {lr_cfg['stage3_lr']:.6f}\n")
        f.write(f"  Test F1: {result.get('test_f1', 0):.4f}\n")
        f.write(f"  Test MAE: {result.get('test_mae_overall', 0):.4f} min\n")
        f.write(f"  Stage 1 Best Epoch: {result.get('stage1_best_epoch', 'N/A')}\n")
        f.write(f"  Stage 1 Improvement: {result.get('stage1_improvement', 0):.4f}\n")
        f.write(f"  Schedule Score: {result.get('schedule_score', 0):.2f}\n")

    def _analyze_lr_patterns(self, f, results: List[Dict]) -> None:
        """Analyze which LR patterns work best."""
        # Categorize patterns
        unified_schedules = []  # Same LR for all stages
        decreasing_schedules = []  # S1 >= S2 >= S3
        increasing_schedules = []  # S1 <= S2 <= S3
        v_shaped_schedules = []  # Other patterns
        
        for result in results:
            lr_cfg = result['lr_config']
            s1, s2, s3 = lr_cfg['stage1_lr'], lr_cfg['stage2_lr'], lr_cfg['stage3_lr']
            
            # Check if unified (same LR)
            if abs(s1 - s2) < 1e-7 and abs(s2 - s3) < 1e-7:
                unified_schedules.append(result)
            elif s1 >= s2 >= s3 and not (s1 == s2 == s3):
                decreasing_schedules.append(result)
            elif s1 <= s2 <= s3 and not (s1 == s2 == s3):
                increasing_schedules.append(result)
            else:
                v_shaped_schedules.append(result)
        
        # KEY QUESTION: Do we need different LRs per stage?
        f.write("=" * 80 + "\n")
        f.write("KEY FINDING: UNIFIED vs DIFFERENTIATED LEARNING RATES\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Pattern 1: UNIFIED (S1 = S2 = S3) - Same LR for all stages\n")
        if unified_schedules:
            avg_f1 = np.mean([r.get('test_f1', 0) for r in unified_schedules])
            avg_mae = np.mean([r.get('test_mae_overall', 0) for r in unified_schedules])
            f.write(f"  Count: {len(unified_schedules)}\n")
            f.write(f"  Avg F1: {avg_f1:.4f} | Avg MAE: {avg_mae:.2f}\n")
            best = max(unified_schedules, key=lambda x: x.get('test_f1', 0))
            lr = best['lr_config']
            f.write(f"  Best: LR={lr['stage1_lr']:.6f} (all stages) -> F1={best.get('test_f1', 0):.4f}, MAE={best.get('test_mae_overall', 0):.2f}\n")
        else:
            f.write("  No unified schedules tested\n")
        
        f.write("\nPattern 2: DECREASING (S1 >= S2 >= S3)\n")
        if decreasing_schedules:
            avg_f1 = np.mean([r.get('test_f1', 0) for r in decreasing_schedules])
            avg_mae = np.mean([r.get('test_mae_overall', 0) for r in decreasing_schedules])
            f.write(f"  Count: {len(decreasing_schedules)}\n")
            f.write(f"  Avg F1: {avg_f1:.4f} | Avg MAE: {avg_mae:.2f}\n")
            best = max(decreasing_schedules, key=lambda x: x.get('test_f1', 0))
            lr = best['lr_config']
            f.write(f"  Best: [{lr['stage1_lr']:.6f}, {lr['stage2_lr']:.6f}, {lr['stage3_lr']:.6f}] "
                   f"-> F1={best.get('test_f1', 0):.4f}, MAE={best.get('test_mae_overall', 0):.2f}\n")
        else:
            f.write("  No decreasing schedules tested\n")
        
        f.write("\nPattern 3: INCREASING (S1 <= S2 <= S3)\n")
        if increasing_schedules:
            avg_f1 = np.mean([r.get('test_f1', 0) for r in increasing_schedules])
            avg_mae = np.mean([r.get('test_mae_overall', 0) for r in increasing_schedules])
            f.write(f"  Count: {len(increasing_schedules)}\n")
            f.write(f"  Avg F1: {avg_f1:.4f} | Avg MAE: {avg_mae:.2f}\n")
            best = max(increasing_schedules, key=lambda x: x.get('test_f1', 0))
            lr = best['lr_config']
            f.write(f"  Best: [{lr['stage1_lr']:.6f}, {lr['stage2_lr']:.6f}, {lr['stage3_lr']:.6f}] "
                   f"-> F1={best.get('test_f1', 0):.4f}, MAE={best.get('test_mae_overall', 0):.2f}\n")
        else:
            f.write("  No increasing schedules tested\n")
        
        f.write("\nPattern 4: V-SHAPED (other patterns)\n")
        if v_shaped_schedules:
            avg_f1 = np.mean([r.get('test_f1', 0) for r in v_shaped_schedules])
            avg_mae = np.mean([r.get('test_mae_overall', 0) for r in v_shaped_schedules])
            f.write(f"  Count: {len(v_shaped_schedules)}\n")
            f.write(f"  Avg F1: {avg_f1:.4f} | Avg MAE: {avg_mae:.2f}\n")
            best = max(v_shaped_schedules, key=lambda x: x.get('test_f1', 0))
            lr = best['lr_config']
            f.write(f"  Best: [{lr['stage1_lr']:.6f}, {lr['stage2_lr']:.6f}, {lr['stage3_lr']:.6f}] "
                   f"-> F1={best.get('test_f1', 0):.4f}, MAE={best.get('test_mae_overall', 0):.2f}\n")
        else:
            f.write("  No V-shaped schedules tested\n")
        
        # Statistical comparison
        f.write("\n" + "=" * 80 + "\n")
        f.write("STATISTICAL COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        all_groups = [
            ("Unified", unified_schedules),
            ("Decreasing", decreasing_schedules),
            ("Increasing", increasing_schedules),
            ("V-shaped", v_shaped_schedules)
        ]
        
        # Rank by average F1
        ranked = sorted([(name, grp) for name, grp in all_groups if grp],
                       key=lambda x: np.mean([r.get('test_f1', 0) for r in x[1]]),
                       reverse=True)
        
        f.write("Ranking by Average F1:\n")
        for i, (name, group) in enumerate(ranked, 1):
            avg_f1 = np.mean([r.get('test_f1', 0) for r in group])
            f.write(f"  {i}. {name}: {avg_f1:.4f} (n={len(group)})\n")
        
        f.write("\nConclusion:\n")
        if ranked:
            winner = ranked[0]
            f.write(f"  Best pattern type: {winner[0]}\n")
            if winner[0] == "Unified":
                f.write("  -> SAME learning rate works best for all stages\n")
                f.write("  -> No need for per-stage LR differentiation\n")
            else:
                f.write("  -> DIFFERENT learning rates per stage is beneficial\n")
                f.write(f"  -> Use {winner[0].lower()} schedule pattern\n")
        
        # Optimal ranges
        f.write("\n" + "="*80 + "\n")
        f.write("OPTIMAL LR RANGES (Top 25% Performers)\n")
        f.write("="*80 + "\n\n")
        
        # Ensure at least 1 result in top quartile
        quartile_size = max(1, len(results) // 4)
        top_quartile = sorted(results, key=lambda x: x.get('test_f1', 0), reverse=True)[:quartile_size]
        
        s1_lrs = [r['lr_config']['stage1_lr'] for r in top_quartile]
        s2_lrs = [r['lr_config']['stage2_lr'] for r in top_quartile]
        s3_lrs = [r['lr_config']['stage3_lr'] for r in top_quartile]
        
        if s1_lrs:  # Only write if we have data
            f.write(f"Stage 1 LR: [{min(s1_lrs):.6f}, {max(s1_lrs):.6f}] (median: {np.median(s1_lrs):.6f})\n")
            f.write(f"Stage 2 LR: [{min(s2_lrs):.6f}, {max(s2_lrs):.6f}] (median: {np.median(s2_lrs):.6f})\n")
            f.write(f"Stage 3 LR: [{min(s3_lrs):.6f}, {max(s3_lrs):.6f}] (median: {np.median(s3_lrs):.6f})\n")

    def run(self) -> None:
        """Execute learning rate tuning."""
        print(f"\n{'='*80}")
        print("COMPREHENSIVE LEARNING RATE TUNING")
        print(f"{'='*80}")
        print(f"Strategy: {self.strategy}")
        print(f"Trials: {self.n_trials}")
        print(f"Output directory: {self.output_dir}")
        print(f"{'='*80}\n")
        
        print("Fixed parameters:")
        for k, v in self.fixed_params.items():
            print(f"  {k}: {v}")
        
        # Generate LR configurations
        configs = self.generate_lr_configs()
        print(f"\nTesting {len(configs)} learning rate configurations\n")
        
        # Run experiments
        for i, lr_config in enumerate(configs, 1):
            result = self.run_experiment(i, lr_config)
            self.results.append(result)
            
            # Save intermediate results
            self.save_results()
            
            print(f"\nProgress: {i}/{len(configs)} trials completed")
            if self.best_result:
                print(f"Current best F1: {self.best_result.get('test_f1', 0):.4f}")
        
        print(f"\n{'='*80}")
        print("TUNING COMPLETE")
        print(f"{'='*80}")
        print(f"Total trials: {len(self.results)}")
        print(f"Successful: {sum(1 for r in self.results if r['success'])}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Comprehensive learning rate tuning for cnnopacus.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Search Strategies:
  coarse        - Quick 4x4x3=48 config exploration
  comprehensive - Balanced 7x7x7 with smart sampling (~100 configs)
  exhaustive    - Dense grid 9x9x9 with constraints (~200-300 configs)

Examples:
  # Quick test (recommended first)
  python tune_learning_rates.py --strategy coarse --n_trials 20

  # Comprehensive search (recommended)
  python tune_learning_rates.py --strategy comprehensive --n_trials 50

  # Exhaustive search (very thorough, takes longer)
  python tune_learning_rates.py --strategy exhaustive --n_trials 100
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        choices=['coarse', 'comprehensive', 'exhaustive'],
        default='comprehensive',
        help='Search strategy granularity',
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=30,
        help='Maximum number of trials to run',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='lr_tuning_results',
        help='Directory to save results',
    )
    parser.add_argument(
        '--base_script',
        type=str,
        default='cnnopacus.py',
        help='Path to training script',
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    tuner = LearningRateTuner(
        base_script=args.base_script,
        output_dir=args.output_dir,
        strategy=args.strategy,
        n_trials=args.n_trials,
    )
    
    tuner.run()


if __name__ == '__main__':
    main()
