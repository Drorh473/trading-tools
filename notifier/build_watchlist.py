"""CLI command to refresh notifier/watchlist.py with the current top-N
Bitget USDT-margined futures symbols by 24h volume - filtered to symbols the
account can actually place split-entry trades on. Run manually when you
want to update the list:

    python -m notifier.build_watchlist
    python -m notifier.build_watchlist --top 50

Bitget enforces its $5-per-order minimum on EACH leg of a split entry
separately, not on the position total. At 1% risk on a ~$100 account, the
20% market leg (always the smaller one) needs the total position to be
worth at least $25 - which needs the stop to sit no more than about 4% from
entry. A symbol whose swings are typically much wider than that will
routinely produce trades where the total clears $5 but the market leg alone
does not: ADAUSDT did exactly this on 2026-08-03, rejected live with
"less than the minimum amount 5 USDT" on a $6.35 position. That was the
first live order this was actually attempted, so it had never been checked.

Ranking is still by volume - liquidity and spread are why the watchlist is
ranked that way at all - but a symbol whose OWN recent swings are too wide
for this account size to trade in split form is skipped in favour of the
next-highest-volume symbol that isn't, rather than silently producing
signals that fail on approval.

This is a property of the symbol's typical volatility relative to current
equity, not of any single signal - it is re-measured each time the list is
rebuilt so it tracks both as the account size changes and as symbols
themselves calm down or heat up.
"""

import argparse
import statistics
from pathlib import Path

from core.bitget_client import BitgetClient
from notifier.strategies.indicators import atr
from notifier.strategies.rsi_fib_reversal import ATR_PERIOD, FIB_ENTRY, FIB_STOP, MARKET_ENTRY_FRACTION, SWING_ATR_MULTIPLE
from notifier.strategies.structure import zigzag_pivots

WATCHLIST_PATH = Path(__file__).parent / "watchlist.py"

# How far Strategy 1's stop sits from its entry, as a share of the swing being
# retraced (78.6% - 61.8%). Fixed by the strategy, not a tunable here.
_FIB_GAP = FIB_STOP - FIB_ENTRY
_DEFAULT_RISK_PCT = 0.01  # the scanner's un-confirmed-signal risk; the common case
_MIN_NOTIONAL_PER_LEG = 5.0
_CANDLES_FOR_SWING_HISTORY = 300


def top_symbols_by_volume(bitget: BitgetClient, top: int) -> list[str]:
    tickers = bitget.get_all_tickers()
    ranked = sorted(tickers, key=lambda t: float(t["usdtVolume"]), reverse=True)
    return [t["symbol"] for t in ranked[:top]]


def _median_swing_pct(bitget: BitgetClient, symbol: str) -> float | None:
    """The symbol's own recent swings (the same ZigZag Strategy 1 anchors on),
    as a fraction of price - a per-symbol volatility measure, not a per-signal
    one, since any given day's actual signal is too sparse to measure this
    from directly."""
    candles = bitget.get_candles(symbol, granularity="1H", limit=_CANDLES_FOR_SWING_HISTORY, closed_only=True)
    if len(candles) < 60:
        return None

    import pandas as pd

    bars = pd.DataFrame(candles, columns=["ts", "open", "high", "low", "close", "base_vol", "quote_vol"])
    for col in ["high", "low", "close"]:
        bars[col] = bars[col].astype(float)

    thresholds = atr(bars, ATR_PERIOD) * SWING_ATR_MULTIPLE
    pivots = zigzag_pivots(bars, thresholds)
    pcts = []
    for a, b in zip(pivots, pivots[1:]):
        i0, i1 = a[0], b[0]
        lo = min(bars["low"].iloc[i0], bars["low"].iloc[i1])
        hi = max(bars["high"].iloc[i0], bars["high"].iloc[i1])
        price = bars["close"].iloc[i1]
        if price > 0:
            pcts.append((hi - lo) / price)

    return statistics.median(pcts) if pcts else None


def clears_split_entry_minimum(median_swing_pct: float, equity: float) -> bool:
    """Would a typical signal on this symbol clear Bitget's per-order minimum
    on its smaller (market) leg, at the default risk_pct?"""
    stop_pct = _FIB_GAP * median_swing_pct
    if stop_pct <= 0:
        return False
    notional = (_DEFAULT_RISK_PCT / stop_pct) * equity
    return notional * MARKET_ENTRY_FRACTION >= _MIN_NOTIONAL_PER_LEG


def executable_symbols_by_volume(bitget: BitgetClient, top: int, equity: float, pool_multiplier: int = 3) -> list[str]:
    """Top-`top` by volume, skipping any symbol whose own typical swing is too
    wide for a split entry to clear the per-leg minimum at this equity -
    filling the gap from further down the volume ranking instead of shrinking
    the list."""
    candidates = top_symbols_by_volume(bitget, top * pool_multiplier)
    selected: list[str] = []
    skipped: list[str] = []

    for symbol in candidates:
        if len(selected) >= top:
            break
        median_pct = _median_swing_pct(bitget, symbol)
        if median_pct is None:
            continue  # not enough history to judge; neither selected nor reported as skipped
        if clears_split_entry_minimum(median_pct, equity):
            selected.append(symbol)
        else:
            skipped.append(symbol)

    if skipped:
        print(f"Skipped {len(skipped)} symbol(s) whose typical swing is too wide to split at ${equity:.0f} equity:")
        print(f"  {skipped}")

    return selected


def write_watchlist(symbols: list[str]) -> None:
    lines = ['"""Crypto pairs to scan, as Bitget USDT-margined futures symbols."""', "", "WATCHLIST: list[str] = ["]
    lines += [f'    "{s}",' for s in symbols]
    lines += ["]", ""]
    WATCHLIST_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the notifier watchlist by 24h volume")
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument(
        "--no-executability-filter",
        action="store_true",
        help="Rank by volume only, ignoring whether split-entry trades would clear Bitget's per-leg minimum.",
    )
    args = parser.parse_args()

    bitget = BitgetClient()  # public market data only, no credentials needed

    if args.no_executability_filter:
        symbols = top_symbols_by_volume(bitget, args.top)
    else:
        equity = bitget.get_account_equity()
        symbols = executable_symbols_by_volume(bitget, args.top, equity)

    write_watchlist(symbols)
    print(f"Wrote {len(symbols)} symbols to {WATCHLIST_PATH}")


if __name__ == "__main__":
    main()
