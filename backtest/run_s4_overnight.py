"""The overnight job: record the detector's output once, then sweep it.

    python -m backtest.run_s4_overnight --workers 10

Stage 1 (~7h) runs Strategy 4's detector across the universe and writes
data/s4_contexts.pkl. Stage 2 (minutes) rebuilds trades under ~45 parameter
arms and prints the comparison table. Everything lands in
logs/s4_overnight_<stamp>.log as well as on stdout, so the morning read does
not depend on the terminal still being open.

Stage 2 is re-runnable on its own against the recorded contexts:

    python -m backtest.s4_sweep_construction

so adding arms tomorrow costs minutes, not another night.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

CONTEXTS = "data/s4_contexts.pkl"


def stage(name: str, cmd: list[str], log) -> bool:
    banner = f"\n{'=' * 70}\n{name}\n  $ {' '.join(cmd)}\n{'=' * 70}"
    print(banner, flush=True)
    log.write(banner + "\n")
    log.flush()
    started = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, env={**os.environ,
                                                       "PYTHONPATH": str(Path.cwd())})
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
    code = proc.wait()
    mins = (time.time() - started) / 60
    tail = f"-- {name}: {'OK' if code == 0 else f'FAILED ({code})'} in {mins:.1f}m"
    print(tail, flush=True)
    log.write(tail + "\n")
    log.flush()
    return code == 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--skip-record", action="store_true",
                    help="reuse an existing data/s4_contexts.pkl and sweep only")
    args = ap.parse_args()

    Path("logs").mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path("logs") / f"s4_overnight_{stamp}.log"
    py = sys.executable
    started = time.time()

    with open(path, "w", encoding="utf-8") as log:
        log.write(f"Strategy 4 overnight - started {datetime.now():%Y-%m-%d %H:%M:%S}\n")
        if not args.skip_record:
            ok = stage("Stage 1: record the detector's output",
                       [py, "-m", "backtest.s4_record",
                        "--workers", str(args.workers), "--out", CONTEXTS], log)
            if not ok:
                print("stage 1 failed - not sweeping a partial recording")
                return
        elif not Path(CONTEXTS).exists():
            print(f"--skip-record given but {CONTEXTS} does not exist")
            return

        stage("Stage 2: sweep the trade construction",
              [py, "-m", "backtest.s4_sweep_construction", "--contexts", CONTEXTS], log)

        done = f"\nTOTAL {(time.time() - started) / 60:.1f}m - log at {path}"
        print(done)
        log.write(done + "\n")


if __name__ == "__main__":
    main()
