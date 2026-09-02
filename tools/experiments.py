"""Has this already been tried? What did it find?

  python -m tools.experiments ls [--name SUBSTRING] [--dir experiments]

Lists every recorded experiment run - see backtest/experiments.py for why
each run is its own immutable file rather than a document a rerun can
silently overwrite. --name filters by substring against the experiment
name, case-insensitively, so "python -m tools.experiments ls --name btc"
shows every run of anything BTC-gate-related without knowing the exact name.
"""

from __future__ import annotations

import argparse
import sys

from backtest.experiments import list_experiments


def format_table(runs: list[dict]) -> str:
    if not runs:
        return "no experiments recorded"

    lines = []
    for r in runs:
        arms_summary = ", ".join(f"{a['name']}={a['net_expectancy']:+.3f}R" for a in r.get("arms", []))
        lines.append(
            f"{r.get('written_at', '?')[:19]}  {r.get('name', '?'):<30} "
            f"[{r.get('verdict', '?')}]  {arms_summary}"
        )
    return "\n".join(lines)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ls"])
    ap.add_argument("--name", default=None, help="substring filter, case-insensitive")
    ap.add_argument("--dir", default="experiments")
    args = ap.parse_args()

    runs = list_experiments(args.dir)
    if args.name:
        needle = args.name.lower()
        runs = [r for r in runs if needle in r.get("name", "").lower()]

    print(format_table(runs))


if __name__ == "__main__":
    main()
