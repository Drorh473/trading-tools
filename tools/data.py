"""What's actually in data/, and what produced it.

  python -m tools.data ls [--dir data]

Every generated pickle is opaque on its own - a filename, a size, a mtime.
For one that carries a manifest (see backtest/manifest.py), this also shows
the git commit that produced it and whatever else the generator recorded
(instance count, universe, row count...). One that doesn't is flagged
plainly rather than silently skipped - a file with "no manifest" is exactly
the state signals_v2_ship.pkl was in when its instance count went unnoticed.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from backtest.manifest import read_manifest


@dataclass
class DatasetInfo:
    path: str  # relative to the scanned directory, forward-slashed
    size_bytes: int
    modified: str
    manifest: dict | None


def list_datasets(data_dir: str = "data") -> list[DatasetInfo]:
    root = Path(data_dir)
    infos = []
    for pkl in sorted(root.rglob("*.pkl")):
        stat = pkl.stat()
        infos.append(
            DatasetInfo(
                path=str(pkl.relative_to(root)).replace("\\", "/"),
                size_bytes=stat.st_size,
                modified=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                manifest=read_manifest(str(pkl)),
            )
        )
    return infos


def format_table(infos: list[DatasetInfo]) -> str:
    if not infos:
        return "no .pkl files found"

    lines = []
    for info in infos:
        size_mb = info.size_bytes / (1024 * 1024)
        if info.manifest is None:
            origin = "no manifest"
        else:
            sha = info.manifest.get("git_sha")
            origin = sha[:10] if sha else "no git_sha recorded"
            extra = {k: v for k, v in info.manifest.items() if k not in ("git_sha", "written_at")}
            if extra:
                origin += "  " + ", ".join(f"{k}={v}" for k, v in sorted(extra.items()))
        lines.append(f"{info.path:<45} {size_mb:8.1f} MB  {origin}")
    return "\n".join(lines)


def main() -> None:
    # A real filename can hold anything a symbol's own ticker does - CJK,
    # accents, whatever Bitget lists - and the terminal's codepage is not
    # guaranteed to cover it (crashed outright on Windows cp1255 against
    # s1_bars/龙虾USDT.pkl, a real cached symbol). Replacing what a
    # given terminal can't render is the right failure mode for a listing
    # tool: one row looking slightly off beats the whole command dying
    # because of one file deep in a 773-file scan.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["ls"])
    ap.add_argument("--dir", default="data")
    args = ap.parse_args()

    print(format_table(list_datasets(args.dir)))


if __name__ == "__main__":
    main()
