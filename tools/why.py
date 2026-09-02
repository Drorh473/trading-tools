"""Why did (or didn't) a strategy fire on a symbol, at a given moment?

  python -m tools.why SYMBOL --strategy "Strategy 3" [--at 2026-08-15T00:00]

STRICTLY READ-ONLY - fetches public candle data only, the same endpoint the
live scanner polls; nothing here places, amends or cancels anything.

WHY THIS EXISTS
  The rule funnel behind this ("of every box_len find_consolidation tried,
  which rule killed it") was a one-off script run once, by hand, against
  17.5M candidate box lengths, and its result lived only in a log file
  (logs/s3_rule_funnel.log) nobody could rerun against a different symbol or
  date without rebuilding the script from scratch. The instrumentation it
  used - find_consolidation's own `stats` parameter - was already IN the
  strategy; what was missing was a reusable way to ask it a question.

  This is that: Strategy.explain() (see notifier/strategies/base.py) is the
  general contract - every strategy gets a coarse fired/not-fired answer for
  free, and earns a richer ladder by overriding it, same pattern as
  chart_overlay(). Strategy 3 is the first to do so, wrapping the funnel
  that already existed rather than re-deriving it - see VolumeRun.explain's
  own docstring for why the box search gets the detailed ladder and the
  breakout/volume trigger does not, yet.

--at accepts anything pandas.Timestamp parses (e.g. "2026-08-15",
"2026-08-15T14:00"). Omit it to ask "why isn't this armed right now" against
live bars, exactly as the scanner would see them this instant.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from config import settings
from core.bitget_client import client_from_settings
from notifier.main import build_strategies
from notifier.scanner import _split_reference_key, bars_dataframe
from notifier.strategies.base import ExplainResult, Strategy

DEFAULT_CANDLE_LIMIT = 600


def find_strategy(name: str, strategies: list[Strategy]) -> Strategy:
    """The one live instance whose tag (or any tag it can emit, see
    Strategy.all_tags) contains `name` case-insensitively.

    Substring rather than exact match so "Strategy 3" or even "3 1D" both
    work without knowing the instance's full tag string ("Strategy 3
    1D/1H") up front - but ambiguous substrings are refused rather than
    silently picking one, since a wrong strategy answering "why" about the
    wrong rules is worse than an extra command-line character.
    """
    needle = name.lower()
    matches = [s for s in strategies if any(needle in tag.lower() for tag in s.all_tags())]
    available = ", ".join(sorted({tag for s in strategies for tag in s.all_tags()}))
    if not matches:
        raise ValueError(f"no strategy tag matches {name!r}. Available: {available}")
    if len(matches) > 1:
        matched = ", ".join(sorted({tag for s in matches for tag in s.all_tags()}))
        raise ValueError(f"{name!r} matches more than one strategy ({matched}) - be more specific")
    return matches[0]


def trim_to(bars: pd.DataFrame, at: pd.Timestamp | None) -> pd.DataFrame:
    """Bars up to and including `at`, re-indexed 0-based - or `bars`
    unchanged when `at` is None (the "right now" case).

    Re-indexing matters: find_consolidation, structure_context and friends
    all index positions from 0 against whatever frame they're handed, the
    same way scanner._bars() always returns a fresh 0-based frame per
    fetch - a trimmed-but-not-reset frame would carry the ORIGINAL row
    labels, and a strategy reading `.iloc[some_index]` would silently read
    the wrong row.
    """
    if at is None:
        return bars
    return bars[bars["ts"] <= at].reset_index(drop=True)


def fetch_bars(
    bitget, symbol: str, strategy: Strategy, at: pd.Timestamp | None = None, limit: int = DEFAULT_CANDLE_LIMIT
) -> dict[str, pd.DataFrame]:
    """bars_by_timeframe for every timeframe `strategy` declares, trimmed to
    `at` - the same dict shape evaluate()/explain() expect.

    A "SYMBOL@TF" declared timeframe (Strategy 1's market_trend_symbol,
    Strategy 4's dealing-range reference) fetches THAT symbol's bars, not
    the one being explained - see scanner._split_reference_key, reused here
    rather than re-implemented so the two can't silently drift apart.

    Fetches CLOSED bars only. That is not exactly what evaluate() sees for a
    strategy that reads a forming reference candle (forming_bar_timeframes) -
    a known, accepted gap: this tool answers "what did the last CLOSED
    picture say", which is what a chart reviewed after the fact shows anyway.
    """
    bars_by_tf: dict[str, pd.DataFrame] = {}
    for tf in strategy.timeframes:
        ref = _split_reference_key(tf)
        fetch_symbol, granularity = ref if ref else (symbol, tf)
        candles = bitget.get_candles(fetch_symbol, granularity=granularity, limit=limit, closed_only=True)
        bars_by_tf[tf] = trim_to(bars_dataframe(candles), at)
    return bars_by_tf


def format_report(symbol: str, strategy: Strategy, at: pd.Timestamp | None, result: ExplainResult) -> str:
    lines = [
        f"{symbol}  |  {strategy.tag}  |  as of {at if at is not None else 'now'}",
        "FIRED" if result.fired else "DID NOT FIRE",
        "",
    ]
    for check in result.checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"  [{status}] {check.name}: {check.detail}")

    if result.fired and result.signal is not None:
        s = result.signal
        lines += ["", f"  entry {s.entry_price:g}  stop {s.stop_loss:g}  direction {s.direction}"]

    if result.funnel:
        lines += ["", "funnel (candidates rejected at each rule):"]
        total = sum(result.funnel.values())
        for name, count in sorted(result.funnel.items(), key=lambda kv: kv[1], reverse=True):
            pct = 100 * count / total if total else 0.0
            lines.append(f"  {name:<40} {count:>10,}  ({pct:5.1f}%)")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("symbol")
    ap.add_argument("--strategy", required=True, help="substring of a live strategy's tag, e.g. 'Strategy 3'")
    ap.add_argument("--at", default=None, help="ISO timestamp to evaluate as of; omit for 'right now'")
    args = ap.parse_args()

    at = pd.Timestamp(args.at) if args.at else None

    strategies = build_strategies()
    try:
        strategy = find_strategy(args.strategy, strategies)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)

    bitget = client_from_settings(settings)
    bars_by_tf = fetch_bars(bitget, args.symbol, strategy, at=at)
    result = strategy.explain(args.symbol, bars_by_tf)

    print(format_report(args.symbol, strategy, at, result))


if __name__ == "__main__":
    main()
