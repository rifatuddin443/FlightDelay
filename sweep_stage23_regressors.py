"""Sweep regressor architectures for stage 2/3.

This script repeatedly runs `stacked_gru_transformer_three_stage.py` with different
`--regressor` settings, starting from a stage-1 checkpoint, and compares the final
metrics written to `three_stage_metrics.csv`.

Example (Windows PowerShell):
  python sweep_stage23_regressors.py \
    --resume_checkpoint "D:\\flight delay\\stpn paper\\STPN-main\\checkpoints\\stacked_gru_three_stage_20260324_003536\\checkpoint_stage1_epoch6.pt" \
    --start_stage 2 \
    --regressors mlp deep_mlp residual_mlp gru tsit convtran \
    --metric regression_rmse_overall

To forward extra args to the underlying training script, add `--` then args:
  python sweep_stage23_regressors.py --resume_checkpoint "...pt" -- --dp --epsilon 7.5
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


DEFAULT_REGRESSORS: Tuple[str, ...] = (
    # "mlp",
    # "deep_mlp",
    # "residual_mlp",
    # "gru",
    # "tsit",
    # "convtran",
    # "nbeats",
    "node_transformer",
    # "tft",
    # "graph_gat",
)


@dataclass
class RunResult:
    regressor: str
    out_dir: str
    metrics: Dict[str, Any]
    returncode: int


def _read_metrics_csv(metrics_path: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")

    with open(metrics_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            raise ValueError(f"unexpected metrics header in {metrics_path}: {header}")

        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                continue
            key = str(row[0]).strip()
            val_raw = row[1]
            if key == "__cli_args__" or key == "":
                continue
            # Try numeric conversion, fall back to string.
            try:
                val_num = float(val_raw)
                metrics[key] = val_num
            except Exception:
                metrics[key] = str(val_raw)

    return metrics


def _rank_key(metrics: Dict[str, Any], metric_name: str) -> float:
    val = metrics.get(metric_name)
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except Exception:
        return float("inf")


def _metric_higher_is_better(metric_name: str) -> bool:
    m = (metric_name or "").lower()
    # Common classification-style metrics where higher is better.
    return any(k in m for k in ("f1", "accuracy", "precision", "recall", "auc"))


def _find_checkpoint_to_copy(run_out_dir: str) -> Optional[str]:
    ckpt_dir = os.path.join(run_out_dir, "checkpoints")
    candidates = [
        os.path.join(ckpt_dir, "best_stage3.pt"),
        os.path.join(ckpt_dir, "best_stage2.pt"),
        os.path.join(ckpt_dir, "best_stage1.pt"),
        os.path.join(ckpt_dir, "latest_checkpoint.pt"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p

    # Fall back to the newest numbered checkpoint if present.
    if os.path.isdir(ckpt_dir):
        try:
            files = [
                os.path.join(ckpt_dir, f)
                for f in os.listdir(ckpt_dir)
                if f.startswith("checkpoint_stage") and f.endswith(".pt")
            ]
            files = [f for f in files if os.path.isfile(f)]
            if files:
                files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                return files[0]
        except Exception:
            return None
    return None


def _resolve_resume_checkpoint(path_or_dir: str, *, start_stage: int) -> str:
    """Resolve a user-provided checkpoint path.

    Accepts either:
    - a direct path to a .pt file, or
    - a directory that contains checkpoint files (e.g., a "checkpoints" folder).

    For directories, prefers stage-appropriate best checkpoints.
    """
    p = os.path.abspath(path_or_dir)
    if os.path.isfile(p):
        return p

    if not os.path.isdir(p):
        raise FileNotFoundError(f"checkpoint not found: {p}")

    # If a directory was given, try to pick the most appropriate checkpoint.
    preferred: List[str] = []
    if int(start_stage) >= 3:
        preferred.append(os.path.join(p, "best_stage3.pt"))
    if int(start_stage) >= 2:
        preferred.append(os.path.join(p, "best_stage2.pt"))
    preferred.extend(
        [
            os.path.join(p, "best_stage1.pt"),
            os.path.join(p, "latest_checkpoint.pt"),
        ]
    )
    for candidate in preferred:
        if os.path.isfile(candidate):
            return candidate

    # Fall back to the newest .pt file in the directory.
    try:
        pt_files = [
            os.path.join(p, f)
            for f in os.listdir(p)
            if f.lower().endswith(".pt") and os.path.isfile(os.path.join(p, f))
        ]
        if pt_files:
            pt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            return pt_files[0]
    except Exception:
        pass

    raise FileNotFoundError(
        "No checkpoint .pt files found under directory: " + p + "\n"
        "Expected one of: best_stage3.pt, best_stage2.pt, best_stage1.pt, latest_checkpoint.pt"
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _check_disk_space(path: str, *, min_free_bytes: int = 200 * 1024 * 1024) -> None:
    """Raise a friendly error if the target drive is out of space.

    Defaults to requiring at least 200MB free, since checkpoints can be large.
    """
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return
    if int(usage.free) < int(min_free_bytes):
        free_mb = usage.free / (1024 * 1024)
        raise RuntimeError(
            f"Not enough free disk space to write sweep outputs under: {path}\n"
            f"Free space: {free_mb:.1f} MB\n"
            "Free up space or run with --out_root pointing to a drive with more space."
        )


def _run_one(
    *,
    python_exe: str,
    base_script: str,
    resume_checkpoint: str,
    start_stage: int,
    regressor: str,
    out_dir: str,
    checkpoint_dir: str,
    passthrough_args: Sequence[str],
    fail_fast: bool,
) -> RunResult:
    _ensure_dir(out_dir)
    _ensure_dir(checkpoint_dir)
    log_path = os.path.join(out_dir, "run.log")

    cmd: List[str] = [
        python_exe,
        "-u",
        base_script,
        "--resume_checkpoint",
        resume_checkpoint,
        "--checkpoint_dir",
        checkpoint_dir,
        "--start_stage",
        str(start_stage),
        "--regressor",
        regressor,
        "--output_dir",
        out_dir,
    ]
    cmd.extend(list(passthrough_args))

    # Stream logs live to the terminal while also writing to file.
    with open(log_path, "w", encoding="utf-8", newline="") as logf:
        logf.write("COMMAND:\n")
        logf.write(" ".join(cmd) + "\n\n")
        logf.flush()

        print("COMMAND:")
        print("  " + " ".join(cmd))
        print(f"[log] {log_path}")

        env = dict(os.environ)
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("PYTHONUTF8", "1")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            # Write to log file
            logf.write(line)
            logf.flush()
            # Mirror to terminal
            sys.stdout.write(line)
            sys.stdout.flush()

        returncode = proc.wait()

    if returncode != 0 and fail_fast:
        raise RuntimeError(f"run failed (regressor={regressor}, rc={returncode}); see {log_path}")

    metrics_path = os.path.join(out_dir, "three_stage_metrics.csv")
    metrics: Dict[str, Any] = {}
    if returncode == 0 and os.path.isfile(metrics_path):
        metrics = _read_metrics_csv(metrics_path)

    return RunResult(regressor=regressor, out_dir=out_dir, metrics=metrics, returncode=int(returncode))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep stage2/3 regressor architectures")
    p.add_argument(
        "--resume_checkpoint",
        type=str,
        required=True,
        help="Path to stage-1 (or later) checkpoint .pt OR a directory containing checkpoints",
    )
    p.add_argument("--start_stage", type=int, default=2, choices=[1, 2, 3], help="Start stage for underlying script")
    p.add_argument("--regressors", nargs="+", default=list(DEFAULT_REGRESSORS), help="Regressor names to try")
    p.add_argument(
        "--metric",
        type=str,
        default="regression_mae_overall",
        help="Metric key in three_stage_metrics.csv to rank by",
    )
    p.add_argument("--higher_is_better", action="store_true", help="Rank metric with higher values as better")
    p.add_argument("--lower_is_better", action="store_true", help="Rank metric with lower values as better")
    p.add_argument(
        "--out_root",
        type=str,
        default="auto",
        help="Output root directory. 'auto' creates ./sweeps/stage23_regressors_<ts>",
    )
    p.add_argument("--python", type=str, default=sys.executable, help="Python executable to run the training script")
    p.add_argument("--fail_fast", action="store_true", help="Stop on first failed run")
    p.add_argument("--dry_run", action="store_true", help="Print commands but do not execute")

    # Forward any remaining args to the underlying script.
    p.add_argument("passthrough", nargs=argparse.REMAINDER, help="Args after -- are forwarded to the training script")

    args = p.parse_args(list(argv) if argv is not None else None)
    if args.passthrough and args.passthrough[0] == "--":
        args.passthrough = args.passthrough[1:]
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if bool(args.higher_is_better) and bool(args.lower_is_better):
        raise ValueError("Choose only one of --higher_is_better or --lower_is_better")
    higher_is_better = bool(args.higher_is_better) or (
        (not bool(args.lower_is_better)) and _metric_higher_is_better(str(args.metric))
    )

    here = os.path.dirname(os.path.abspath(__file__))
    base_script = os.path.join(here, "stacked_gru_transformer_three_stage.py")
    if not os.path.isfile(base_script):
        raise FileNotFoundError(f"base script not found: {base_script}")

    resume_checkpoint = _resolve_resume_checkpoint(str(args.resume_checkpoint), start_stage=int(args.start_stage))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = args.out_root
    if not out_root or str(out_root).lower() == "auto":
        out_root = os.path.join(here, "sweeps", f"stage23_regressors_{ts}")
    out_root = os.path.abspath(out_root)
    _ensure_dir(out_root)
    _check_disk_space(out_root)

    regressors: List[str] = list(args.regressors)

    print("=" * 80)
    print("STAGE 2/3 REGRESSOR SWEEP")
    print("=" * 80)
    print(f"checkpoint: {resume_checkpoint}")
    print(f"start_stage: {args.start_stage}")
    print(f"metric ({'higher' if higher_is_better else 'lower'} is better): {args.metric}")
    print(f"out_root: {out_root}")
    if args.passthrough:
        print("passthrough args:")
        print("  " + " ".join(args.passthrough))

    results: List[RunResult] = []
    for name in regressors:
        run_out_dir = os.path.join(out_root, f"regressor_{name}")
        run_ckpt_dir = os.path.join(run_out_dir, "checkpoints")
        cmd_preview = [
            args.python,
            "-u",
            base_script,
            "--resume_checkpoint",
            resume_checkpoint,
            "--checkpoint_dir",
            run_ckpt_dir,
            "--start_stage",
            str(args.start_stage),
            "--regressor",
            name,
            "--output_dir",
            run_out_dir,
            *list(args.passthrough),
        ]
        if args.dry_run:
            print("[dry-run] " + " ".join(cmd_preview))
            continue

        print(f"[run] regressor={name} -> {run_out_dir}")
        res = _run_one(
            python_exe=args.python,
            base_script=base_script,
            resume_checkpoint=resume_checkpoint,
            start_stage=int(args.start_stage),
            regressor=name,
            out_dir=run_out_dir,
            checkpoint_dir=run_ckpt_dir,
            passthrough_args=list(args.passthrough),
            fail_fast=bool(args.fail_fast),
        )
        results.append(res)

    if args.dry_run:
        return 0

    # Filter successful runs with the metric present.
    successful = [r for r in results if r.returncode == 0 and r.metrics]
    ranked = sorted(
        successful,
        key=lambda r: _rank_key(r.metrics, args.metric),
        reverse=bool(higher_is_better),
    )

    summary_path = os.path.join(out_root, "sweep_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "regressor", args.metric, "out_dir", "returncode"])
        for i, r in enumerate(ranked, start=1):
            w.writerow([i, r.regressor, _rank_key(r.metrics, args.metric), r.out_dir, r.returncode])
        for r in results:
            if r in successful:
                continue
            w.writerow(["", r.regressor, "", r.out_dir, r.returncode])

    print("\n" + "-" * 80)
    print("LEADERBOARD")
    print("-" * 80)
    if not ranked:
        print("No successful runs produced metrics.")
        print(f"See logs under: {out_root}")
        return 2

    for i, r in enumerate(ranked, start=1):
        score = _rank_key(r.metrics, args.metric)
        mae = r.metrics.get("regression_mae_overall", "")
        rmse = r.metrics.get("regression_rmse_overall", "")
        f1 = r.metrics.get("classification_f1_macro", "")
        print(f"#{i:02d} {r.regressor:12s} {args.metric}={score:.6f} | mae={mae} | rmse={rmse} | f1={f1}")

    best = ranked[0]
    print("\nBEST:")
    print(f"  regressor: {best.regressor}")
    print(f"  {args.metric}: {_rank_key(best.metrics, args.metric):.6f}")
    print(f"  out_dir: {best.out_dir}")

    ckpt_src = _find_checkpoint_to_copy(best.out_dir)
    if ckpt_src:
        ckpt_dst = os.path.join(out_root, "best_checkpoint.pt")
        shutil.copy2(ckpt_src, ckpt_dst)
        print(f"  checkpoint: {ckpt_dst}")
    else:
        print("  checkpoint: (not found to copy; check run's checkpoints directory)")
    print(f"\nWrote: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
