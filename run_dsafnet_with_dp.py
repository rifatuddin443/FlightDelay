"""Convenience entrypoint to launch DSAFNet training with differential privacy enabled.

This script simply builds a curated argument list and reuses the main routine from
`DSAFnet_optimized.py`, ensuring that DP-related hyperparameters are always provided.
Use it for quick DP experiments without remembering the long CLI.
"""
import argparse
import runpy
import sys
from pathlib import Path


def build_base_argv(args: argparse.Namespace) -> list[str]:
    base_args = [
        '--device', args.device,
        '--data', args.data,
        '--in_channels', str(args.in_channels),
        '--out_channels', str(args.out_channels),
        '--in_len', str(args.in_len),
        '--out_len', str(args.out_len),
        '--batch', str(args.batch),
        '--episode', str(args.episodes),
        '--lr', str(args.lr),
        '--hidden_dim', str(args.hidden_dim),
        '--support_len', str(args.support_len),
        '--period', str(args.period),
        '--target_delta', str(args.target_delta),
        '--max_grad_norm', str(args.max_grad_norm),
        '--output_dir', args.output_dir,
    ]

    # Force DP flag (base script defaults to DP already, but this keeps intent explicit)
    base_args.append('--dp')

    if args.mode == 'epsilon':
        base_args.extend(['--target_epsilon', str(args.target_epsilon)])
        # When targeting epsilon we still pass noise multiplier so Opacus can start from a sane value
        base_args.extend(['--noise_multiplier', str(args.noise_multiplier)])
    else:
        base_args.extend(['--target_epsilon', '0'])
        base_args.extend(['--noise_multiplier', str(args.noise_multiplier)])

    # Scheduler / early stopping knobs
    base_args.extend([
        '--early_stop_patience', str(args.early_stop_patience),
        '--lr_patience', str(args.lr_patience),
        '--lr_factor', str(args.lr_factor),
        '--min_lr', str(args.min_lr),
    ])

    return base_args


def main() -> None:
    parser = argparse.ArgumentParser(description='Run DSAFNet with DP defaults')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--data', default='US')
    parser.add_argument('--in_channels', type=int, default=2)
    parser.add_argument('--out_channels', type=int, default=2)
    parser.add_argument('--in_len', type=int, default=12)
    parser.add_argument('--out_len', type=int, default=12)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--support_len', type=int, default=3)
    parser.add_argument('--period', type=int, default=36)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--target_epsilon', type=float, default=4.0)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    parser.add_argument('--noise_multiplier', type=float, default=1.5)
    parser.add_argument('--max_grad_norm', type=float, default=1.5)
    parser.add_argument('--mode', choices=['epsilon', 'noise'], default='epsilon',
                        help='Choose whether to target epsilon or use a fixed noise multiplier')
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--lr_patience', type=int, default=3)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--min_lr', type=float, default=1e-7)
    parser.add_argument('--output_dir', default='./results_dp')
    parser.add_argument('--in_channels_weather', type=int, default=None,
                        help='Optional override for in_channels if weather features are appended')

    args, extra = parser.parse_known_args()

    if args.in_channels_weather is not None:
        args.in_channels = args.in_channels_weather

    base_script = Path(__file__).with_name('DSAFnet_optimized.py')
    if not base_script.exists():
        raise FileNotFoundError(f'Base script not found: {base_script}')

    base_argv = [str(base_script)] + build_base_argv(args) + extra
    sys.argv = base_argv
    runpy.run_path(str(base_script), run_name='__main__')


if __name__ == '__main__':
    main()
