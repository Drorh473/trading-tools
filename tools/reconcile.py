"""Does the live deficit come from the rules, or from everything AROUND them?

  python -m tools.reconcile [--strategy TAG] [--since YYYY-MM-DD]

Strategy 1 1H measures -0.061R in backtest and has real closed trades in the
journal - and nothing has ever compared the two. This runs backtest/score.py's
own simulator against the EXACT plan each real trade used (its original
Signal, its real fill price, the runner target the live tracker actually
committed to) and reports the gap between what score.simulate() says that
plan was worth and what the trade actually closed at. A live deficit that
matches the backtest's own number is a signal problem; one that doesn't is
slippage, fees, or fill timing - a different question with a different fix.

SCOPE - READ BEFORE TRUSTING A NUMBER THIS PRINTS
  score.simulate() models exactly one exit shape: a 50% partial at target 1,
  breakeven on the remainder, then a second target (or an open runner). That
  is the scanner's own default - what a signal gets when its strategy does
  not override partial_fraction at all (Strategy 1, most of Strategy 2.1).
  It is NOT what Strategy 3 (75% partial, a daily-resistance runner) or
  Strategy 4 (100%, no partial) actually do. A trade whose ORIGINAL signal
  (not the trade row's own bookkeeping - see resolve_remainder_target) used
  a different partial_fraction is SKIPPED with the reason printed, never
  silently scored against a shape it never used. A wrong number that looks
  measured is worse than an honest gap.

  Read-only: fetches the same public candle endpoint the live scanner polls
  and the trades database. Places nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import pandas as pd

from backtest.score import simulate
from config import settings
from core.bitget_client import client_from_settings
from core.storage import SignalRecord, Storage, Trade
from notifier.scanner import bars_dataframe
from notifier.strategies.base import signal_from_json


def is_eligible(signal) -> tuple[bool, str]:
    """Whether `signal`'s OWN exit shape is the one score.simulate() models.
    (eligible, reason) - reason is empty when eligible."""
    if signal.partial_fraction is None or signal.partial_fraction == 0.5:
        return True, ""
    return (
        False,
        f"strategy's own partial_fraction ({signal.partial_fraction}) is not "
        "the 50% score.simulate() models",
    )


def resolve_remainder_target(signal, trade) -> float | None:
    """The REAL runner target, preferring what the live tracker actually
    committed to (Trade.runner_target) over the strategy's own signal field.

    They differ for a real reason: the scanner computes the runner's actual
    target dynamically at dispatch time (see scanner._dispatch's own
    `remainder_target` local) and only the TRADE row remembers it (via
    set_exit_plan) - most strategies (Strategy 1 included) never write it
    onto the Signal object itself, so signal.remainder_target is often None
    even for a trade that had a real, tracked runner target.
    """
    return trade.runner_target if trade.runner_target is not None else signal.remainder_target


def find_start_index(bars: pd.DataFrame, dispatched_at: pd.Timestamp) -> int | None:
    """The last bar at or before `dispatched_at` - the candle the signal
    actually fired on, or None if the fetched history doesn't reach back
    that far."""
    eligible = bars.index[bars["ts"] <= dispatched_at]
    if len(eligible) == 0:
        return None
    return int(eligible[-1])


@dataclass
class ReconcileResult:
    trade_id: int
    symbol: str
    strategy_tag: str
    skipped: bool
    skip_reason: str = ""
    real_r: float | None = None
    backtest_r: float | None = None
    delta: float | None = None


def reconcile_trade(trade: Trade, record: SignalRecord, bars: pd.DataFrame) -> ReconcileResult:
    """One real trade vs. what score.simulate() says its own plan was worth."""
    base = dict(trade_id=trade.מספר_עסקה, symbol=trade.סימבול, strategy_tag=trade.תגית_אסטרטגיה or "")

    if not record.signal_json:
        return ReconcileResult(**base, skipped=True, skip_reason="no signal_json recorded for this trade")

    signal = signal_from_json(record.signal_json)
    eligible, reason = is_eligible(signal)
    if not eligible:
        return ReconcileResult(**base, skipped=True, skip_reason=reason)

    dispatched_at = pd.Timestamp(record.dispatched_at)
    start = find_start_index(bars, dispatched_at)
    if start is None:
        return ReconcileResult(**base, skipped=True, skip_reason="no bar at or before the signal's dispatch time")

    signal.remainder_target = resolve_remainder_target(signal, trade)
    scored = simulate(bars, start, signal, runner="target", fill_at=trade.מחיר_כניסה)

    real_r = trade.מכפיל_R
    return ReconcileResult(
        **base, skipped=False, real_r=real_r, backtest_r=scored.r_net,
        delta=(real_r - scored.r_net) if real_r is not None else None,
    )


def format_report(results: list[ReconcileResult]) -> str:
    scored = [r for r in results if not r.skipped]
    skipped = [r for r in results if r.skipped]

    lines = [f"{len(scored)} trade(s) reconciled, {len(skipped)} skipped"]
    if scored:
        mean_delta = sum(r.delta for r in scored) / len(scored)
        lines.append(f"mean delta (real R - backtest R): {mean_delta:+.3f}")
        lines.append("")
        for r in scored:
            lines.append(
                f"  #{r.trade_id:<5} {r.symbol:<12} {r.strategy_tag:<20} "
                f"real {r.real_r:+.2f}R  backtest {r.backtest_r:+.2f}R  delta {r.delta:+.2f}R"
            )
    if skipped:
        lines.append("")
        lines.append("skipped:")
        for r in skipped:
            lines.append(f"  #{r.trade_id:<5} {r.symbol:<12} {r.strategy_tag:<20} {r.skip_reason}")
    return "\n".join(lines)


def _signal_for_trade(signals: list[SignalRecord], trade_id: int) -> SignalRecord | None:
    for record in signals:
        if record.trade_id == trade_id:
            return record
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", default=None, help="substring filter on strategy tag")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD; only trades opened on/after this date")
    args = ap.parse_args()

    storage = Storage(settings.trades_db_path)
    since = pd.Timestamp(args.since).date() if args.since else None

    trades = [t for t in storage.read_all(start=since) if t.is_closed]
    if args.strategy:
        needle = args.strategy.lower()
        trades = [t for t in trades if needle in (t.תגית_אסטרטגיה or "").lower()]

    signals = storage.read_signals()
    bitget = client_from_settings(settings)
    bars_cache: dict[str, pd.DataFrame] = {}

    results = []
    for trade in trades:
        record = _signal_for_trade(signals, trade.מספר_עסקה)
        if record is None:
            results.append(
                ReconcileResult(
                    trade_id=trade.מספר_עסקה, symbol=trade.סימבול, strategy_tag=trade.תגית_אסטרטגיה or "",
                    skipped=True, skip_reason="no linked signal row",
                )
            )
            continue

        if trade.סימבול not in bars_cache:
            candles = bitget.get_candles(trade.סימבול, granularity="1H", limit=3000, closed_only=True)
            bars_cache[trade.סימבול] = bars_dataframe(candles)

        results.append(reconcile_trade(trade, record, bars_cache[trade.סימבול]))

    print(format_report(results))


if __name__ == "__main__":
    main()
