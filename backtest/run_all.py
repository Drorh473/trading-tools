"""Run every generation and measurement script once, back to back.

  python -m backtest.run_all [--only NAME ...] [--skip NAME ...]

This is not a new caching layer - it is the thing that used to mean typing
one command, waiting for it, then typing the next, across an afternoon.
Each step below is a REAL subprocess, exactly the command you would type by
hand, so one script's process-wide state (generate_v2's module-level
threshold overrides onto notifier.strategies.ema_trend_v2 - see
tests/test_generators_forming_row.py for why that matters) can never leak
into another's, and a step that crashes does not take the rest down with it.

What makes a REPEAT run of this fast is each step's own cache - the
per-instance store portfolio.generate's instance_cache_path uses, or the
whole-output .hash sidecar generate_v2.py/generate_s4_deep.py check - not
this script, which only ever runs each step in order and reports what
happened. A step whose cache is fully warm still costs the time to notice
nothing changed and exit; that is seconds, not hours.

--only/--skip match by substring against the step name (case-insensitive),
so `--only s3` runs just the Strategy 3 swing step, `--skip s4` skips both
Strategy 4 steps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STEPS: list[tuple[str, list[str]]] = [
    ("portfolio (Strategies 1 / 2.1 / 3 / 4)", [sys.executable, "-m", "backtest.run"]),
    ("Strategy 3 swing, isolated", [sys.executable, "-m", "backtest.run_s3_swing"]),
    ("Strategy 2.1 v2 signals", [sys.executable, "-m", "backtest.generate_v2"]),
    ("Strategy 2.1 v2 sweep", [sys.executable, "-m", "backtest.sweep_v2"]),
    ("Strategy 4 deep signals", [sys.executable, "-m", "backtest.generate_s4_deep"]),
    ("Strategy 4 deep replay", [sys.executable, "-m", "backtest.replay_s4_deep"]),
]


def run_steps(steps: list[tuple[str, list[str]]]) -> list[tuple[str, int, float]]:
    """Run each step in order, recording its exit code and wall time. A
    failing step is reported and the loop moves on - one broken script must
    not hide whether the others still work."""
    results = []
    for name, cmd in steps:
        print(f"\n{'=' * 70}\n{name}\n  $ {' '.join(cmd)}\n{'=' * 70}", flush=True)
        t0 = time.time()
        proc = subprocess.run(cmd)
        elapsed = time.time() - t0
        results.append((name, proc.returncode, elapsed))
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"-- {name}: {status} in {elapsed / 60:.1f}m", flush=True)
    return results


def _filter_steps(steps, only: list[str], skip: list[str]) -> list[tuple[str, list[str]]]:
    if only:
        needles = [s.lower() for s in only]
        steps = [(name, cmd) for name, cmd in steps if any(n in name.lower() for n in needles)]
    if skip:
        needles = [s.lower() for s in skip]
        steps = [(name, cmd) for name, cmd in steps if not any(n in name.lower() for n in needles)]
    return steps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", default=[], help="run only steps whose name contains this (repeatable)")
    ap.add_argument("--skip", action="append", default=[], help="skip steps whose name contains this (repeatable)")
    args = ap.parse_args()

    steps = _filter_steps(STEPS, args.only, args.skip)

    if not steps:
        print("no steps match --only/--skip")
        return

    t0 = time.time()
    results = run_steps(steps)

    print(f"\n{'=' * 70}\nSUMMARY  ({(time.time() - t0) / 60:.1f}m total)\n{'=' * 70}")
    for name, code, elapsed in results:
        print(f"  {'OK  ' if code == 0 else 'FAIL'}  {elapsed / 60:6.1f}m  {name}")

    if any(code != 0 for _, code, _ in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
