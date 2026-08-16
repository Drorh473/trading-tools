"""Strategy 2 v1 against v2, replayed as ONE ACCOUNT rather than as trades.

    python -m backtest.replay_head_to_head

The trade-level head-to-head says v1 and v2 earn the same per trade (+0.381R
vs +0.383R) and that v2 simply fires 37x more often - 5,763 signals against
155, for 2,205R against 59R. That total is an upper bound no account can
collect: 5,763 trades a year is ~16 a day and the account holds about five
concurrent positions before margin binds, so the portfolio refuses the
overwhelming majority and arrival order decides which.

§44 measured that exact shape once already - loosening the cap took Strategy 1
1H from 368 trades at +0.12R to 796 at -0.02R. Frequency is not free, and only
a portfolio replay prices it.

So: same bars, same cap, same margin, same $5 floor, same one-position-per-
symbol rule. Only the rules generating the signals differ.

v2's signals were generated with its three thresholds OFF so they could be
swept; they are re-applied here as filters, so what is replayed is v2 AS
CONFIGURED. v1's gates live inside its own evaluate(), so its population is
already what it would trade.
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict

import pandas as pd

from backtest import engine as bt
from backtest import portfolio as pf
from backtest.generate_v2 import CACHE
from backtest.run import report


def _fill_preplaced(acct, symbol: str) -> None:
    """Fill a PRE-PLACED limit on the bar that touched it.

    The engine rests every limit leg and fills it only when a LATER bar trades
    through the price. That is right for v1: its signal fires on a candle that
    closed back above EMA9, so the live bot really does place a limit
    afterwards and wait for a retest.

    v2 is pre-placed. Its signal exists BECAUSE this bar's low reached
    ema9_prev - the order was already resting there and filled during the bar
    that triggered it. Making it wait for a second touch takes only the subset
    where price came back, which is the subset where price kept going against
    the trade, and holds the symbol slot for four bars in the meantime. Both
    penalise v2 for something it does not do.

    Charged MAKER, because a resting limit is what filled.
    """
    pos = acct.open_positions.get(symbol)
    if pos is None or pos.pending_size <= 0 or pos.size > 0:
        return
    pos.size = pos.pending_size
    pos.entry = pos.pending_price
    pos.pending_size = 0.0
    acct.equity -= bt._fee(pos.size * pos.entry, maker=True)


def replay(bars_1h, signals, cap: float, preplaced: bool):
    """portfolio._replay's ordering, with the pre-placed fill available.

    Kept here rather than pushed into portfolio.py because it is a property of
    how a STRATEGY places its entry, and portfolio.py's own instances do not
    have it. Ordering is copied deliberately: bars close, open positions
    advance, then new entries are considered, so a signal on the bar that
    closed a position can reuse the freed margin - the live loop's order.
    """
    previous_cap = bt.MAX_TOTAL_RISK_PCT
    bt.MAX_TOTAL_RISK_PCT = cap
    try:
        acct = bt.Account()
        ts_to_row = {s: {ts: i for i, ts in enumerate(f["ts"].values)} for s, f in bars_1h.items()}
        by_ts = defaultdict(list)
        for symbol, found in signals.items():
            for ts, i, close, pos_i, sig in found:
                by_ts[ts].append((symbol, i, close, pos_i, sig))

        timeline = sorted(set().union(*(set(f["ts"].values) for f in bars_1h.values())))
        for ts in timeline:
            for symbol in list(acct.open_positions):
                row = ts_to_row[symbol].get(ts)
                if row is None:
                    continue
                bt.step_position(acct, acct.open_positions[symbol], bars_1h[symbol].iloc[row], row, ts)

            # Alphabetical, then instance: arbitrary but stable, so the result
            # cannot depend on which worker finished first.
            for symbol, i, close, pos_i, sig in sorted(by_ts.get(ts, ()), key=lambda e: (e[0], e[3])):
                # The flat 4-hour cancel the live tracker actually applies.
                if not bt.try_open(acct, sig, close, i, pf.SPECS, 4) or not preplaced:
                    continue
                _fill_preplaced(acct, symbol)
                # AND advance it against its OWN bar. The engine's convention is
                # that a position opened at ts is first stepped at ts+1, which
                # is right when the fill happens on a later bar. A pre-placed
                # limit fills DURING the trigger bar, so skipping that bar hides
                # every trade that touched EMA9 and then ran to its stop inside
                # the same hour - deleting the worst trades and nothing else.
                pos = acct.open_positions.get(symbol)
                if pos is not None:
                    bt.step_position(acct, pos, bars_1h[symbol].iloc[i], i, ts)
                    if pos.size <= 0 and symbol in acct.open_positions:
                        acct.open_positions.pop(symbol, None)
        return acct
    finally:
        bt.MAX_TOTAL_RISK_PCT = previous_cap

# v2's shipped gates, re-applied to a population generated without them.
HOLD, MIN_STOP, MIN_NET = 5, 0.003, 1.5
MAKER, TAKER = 0.0002, 0.0006


def _passes_v2_gates(sig, hold_base: int, hold_ref: int, paired: bool) -> bool:
    entry, stop = sig.entry_price, sig.stop_loss
    risk = abs(entry - stop)
    if risk <= 0 or entry <= 0:
        return False
    if (hold_ref if paired else hold_base) < HOLD:
        return False
    stop_frac = risk / entry
    if stop_frac < MIN_STOP:
        return False
    net = (sig.reward_risk_ratio * risk - MAKER * entry) / (risk + (MAKER + TAKER) * entry)
    return net >= MIN_NET


def load(path: str, gated: bool) -> tuple[dict, tuple]:
    """{symbol: [(ts, i, close, pos, signal)]} in the shape _replay wants."""
    _key, instances, raw = pickle.load(open(path, "rb"))
    out: dict[str, list] = {}
    kept = dropped = 0
    for symbol, entries in raw.items():
        rows = []
        for ts, i, close, pos, sig, hold_base, hold_ref, _walk in entries:
            if gated:
                paired = instances[pos][1] is not None
                if not _passes_v2_gates(sig, hold_base, hold_ref, paired):
                    dropped += 1
                    continue
            kept += 1
            rows.append((ts, i, close, pos, sig))
        if rows:
            out[symbol] = rows
    if gated:
        print(f"  gates kept {kept}, dropped {dropped}")
    return out, instances


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=float, default=0.10, help="MAX_TOTAL_RISK_PCT, settled at 10%")
    args = ap.parse_args()

    _key, bars, _sig = pickle.load(open(CACHE, "rb"))
    bars_1h = {}
    for symbol, frame in bars.items():
        f = frame.copy()
        f["ts"] = pd.to_datetime(f["ts"])
        bars_1h[symbol] = f
    print(f"{len(bars_1h)} symbols of 1H bars, cap {args.cap*100:.0f}%\n")

    for label, path, gated, preplaced in (
        ("STRATEGY 2 v1 (live today) - limit rests, waits for a retest", "data/signals_v1.pkl", False, False),
        ("STRATEGY 2.1 v2 - limit pre-placed, fills on the trigger bar", "data/signals_v2.pkl", True, True),
        ("STRATEGY 2.1 v2 - forced to wait for a retest (NOT how it works)", "data/signals_v2.pkl", True, False),
    ):
        print(f"loading {label}")
        signals, _instances = load(path, gated)
        acct = replay(bars_1h, signals, args.cap, preplaced)
        report(acct, label)
        print()


if __name__ == "__main__":
    main()
