"""Entrypoint: python -m journal.main
Generates an all-time statistical report from the trade table.
"""

import argparse
from pathlib import Path

from config import settings
from core.storage import Storage
from journal.report import render_markdown, save_charts
from journal.stats import compute_stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a trade statistics report")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    storage = Storage(settings.trades_db_path)
    stats = compute_stats(storage.read_all())

    report_text = render_markdown(stats, title="All-Time Trade Report")
    save_charts(stats, args.output_dir)

    report_path = Path(args.output_dir) / "report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")

    print(report_text)
    print(f"\nSaved to {report_path}")


if __name__ == "__main__":
    main()
