#!/usr/bin/env python
import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

def find_pths(pth_dir=None, pattern=None, list_file=None, recursive=False):
    files = []
    if list_file:
        with open(list_file, 'r', encoding='utf-8') as f:
            files = [line.strip() for line in f if line.strip()]
    elif pattern:
        glob_pattern = pattern
        files = glob.glob(glob_pattern, recursive=recursive)
    elif pth_dir:
        p = Path(pth_dir)
        files = [str(fp) for fp in (p.rglob("*.pth") if recursive else p.glob("*.pth"))]
    else:
        raise ValueError("Provide one of --glob, --pth-dir, or --list-file")
    # Deduplicate and sort for stable order
    unique = sorted({str(Path(f)) for f in files})
    return unique

def main():
    ap = argparse.ArgumentParser(description="Run evaluate_regression_v4.py on multiple .pth files.")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--glob", help="Glob pattern for .pth files")
    src.add_argument("--pth-dir", default=".", help="Directory containing .pth files (default: current directory)")
    src.add_argument("--list-file", help="Text file with one .pth path per line")
    ap.add_argument("--recursive", action="store_true", default=False, help="Search subdirectories")
    ap.add_argument("--log-dir", default="evaluation_logs", help="Directory for log files")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    ap.add_argument("--stop-on-error", action="store_true", help="Stop at first error")
    ap.add_argument("--limit", type=int, default=None, help="Max number of files to process")
    ap.add_argument("--start", type=int, default=0, help="Start index (for resuming)")
    args = ap.parse_args()

    # Default to current directory if no source specified
    if not (args.glob or args.pth_dir or args.list_file):
        args.pth_dir = "."

    pths = find_pths(pth_dir=args.pth_dir, pattern=args.glob, list_file=args.list_file, recursive=args.recursive)
    
    # Filter to only root-level .pth files if pth_dir is specified and not recursive
    if args.pth_dir and not args.recursive:
        base_dir = Path(args.pth_dir).resolve()
        pths = [p for p in pths if Path(p).resolve().parent == base_dir]
    
    if args.start:
        pths = pths[args.start:]
    if args.limit is not None:
        pths = pths[:args.limit]

    if not pths:
        print("No .pth files found for the given inputs.", file=sys.stderr)
        sys.exit(2)

    log_dir = None
    if args.log_dir:
        log_dir = Path(args.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(pths)} .pth files. Running sequentially...\n")
    failures = 0
    for i, pth in enumerate(pths):
        p = Path(pth).resolve()
        name = p.stem

        header = f"[{i+1}/{len(pths)}] {name}"
        print("=" * len(header))
        print(header)
        print("=" * len(header))

        if args.dry_run:
            print(f'python evaluate_regression_v4.py --model_path "{p}"\n')
            continue

        stdout_fh = None
        try:
            if log_dir:
                # sanitize filename
                safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
                run_log = log_dir / f"{i:03d}_{safe}.log"
                stdout_fh = open(run_log, "w", encoding="utf-8", newline="")
                stdout_fh.write(f"# {header}\n# Model: {p}\n\n")
                stdout_fh.flush()

            # Run with proper argument passing to avoid shell interpretation issues
            result = subprocess.run(
                ["python", "evaluate_regression_v4.py", "--model_path", str(p)],
                stdout=stdout_fh or sys.stdout,
                stderr=subprocess.STDOUT,
                cwd=os.getcwd()
            )
            rc = result.returncode
            if rc != 0:
                failures += 1
                print(f"-> FAILED with exit code {rc}\n")
                if stdout_fh:
                    stdout_fh.write(f"\n# FAILED with exit code {rc}\n")
                if args.stop_on_error:
                    break
            else:
                print("-> OK\n")
                if stdout_fh:
                    stdout_fh.write("\n# OK\n")
        except Exception as e:
            failures += 1
            print(f"-> ERROR: {e}\n")
            if stdout_fh:
                stdout_fh.write(f"\n# ERROR: {e}\n")
            if args.stop_on_error:
                break
        finally:
            if stdout_fh:
                stdout_fh.close()

    print(f"Done. Total: {len(pths)}, Failures: {failures}")
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()