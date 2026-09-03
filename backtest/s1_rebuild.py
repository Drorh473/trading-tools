"""Rebuild Strategy 1's ENTRY RULE, rather than tune parameters around it.

Why a rebuild. Seven parameter blocks were swept on 2026-09-02 and five gates
on 2026-09-03; all were refuted out-of-sample, and gross expectancy - before a
cent of fees - is negative. What every one of those arms had in common is that
they change what happens AFTER a signal fires, never which bar fires it.

What the measurements point at. The current trigger goes long the moment
RSI(10) crosses DOWN through 30 (rsi_fib_reversal.evaluate) - it buys into
accelerating downward momentum - and then rests 80% of the position at a
FURTHER retracement, the 61.8% Fib, with the stop only 0.168 of the leg beyond
it. Two independent cuts say that is where the money goes:

  - trades whose resting leg actually filled score -0.26/-0.32 R, while trades
    that never filled it score -0.00/-0.04 (Closed.limit_filled, 1,160 trades)
  - setups whose price had ALREADY retraced past the entry when the cross fired
    score -0.38/-0.52 R at a 15-22% win rate (gap_pct)

Both describe one thing: the trade is entered while price is still moving
against it. So the rebuild is "wait for the turn instead of catching the
knife" - hold fire until RSI crosses BACK through the threshold, then enter at
market on that bar.

Why this does not need a 36-minute regeneration per variant. A new trigger
fires on different bars, which normally means regenerating. But the turn bar is
always FORWARD of a cached cross-down signal on the same symbol, so it can be
scanned to; and the Fib swing is recoverable exactly from any signal's
(entry, stop) - see s1_overnight.swing_of and memory:s1-fib-swing-recoverable.
So every variant here is a transform of the SAME cached signal set, one replay
each. The inherited precondition is honest but worth stating: these are the
turns that follow a cross-down, not every turn in the data.

CONTROL. "market at the signal bar" is replayed alongside, so the comparison
isolates TIMING. Without it a turn arm differs from the shipped baseline in two
ways at once (when it enters, and that it stops resting a limit leg), and the
market_fraction result on 2026-09-03 showed how badly that confounds - it
flipped 0.37R on the $5 floor alone. The balance is Dror's real $230 for the
same reason (memory:replay-start-equity).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from backtest import portfolio as pf
from backtest.s1_overnight import (INSTANCE_CACHE, HOURS_2Y, S1, frames_for,
                                   line, load_deep, map_s1, score, swing_of)
from backtest.score import confirmed_pivots
from notifier.strategies.indicators import atr, rsi
from notifier.watchlist import WATCHLIST

OTHER = {i for i in range(len(pf.INSTANCES)) if i != S1}
RSI_PERIOD, OVERSOLD, OVERBOUGHT = 10, 30.0, 70.0
SERIES: dict = {}


def build_series(bars):
    """RSI, close, ATR and ts per symbol as numpy - the transforms index these
    millions of times, and a DataFrame lookup is ~100x slower per access (the
    same cost that made score_too_small 97% of a replay)."""
    for sym, f in bars.items():
        SERIES[sym] = {
            "rsi": rsi(f["close"], RSI_PERIOD).to_numpy(),
            "close": f["close"].to_numpy(),
            "atr": atr(f).to_numpy(),
            "ts": f["ts"].to_numpy(),
        }


def turn_bar(sym, i, direction, max_wait):
    """First bar after `i` where RSI crosses BACK through its threshold - 30
    upward for a long, 70 downward for a short. None if it never does within
    max_wait, in which case the setup is simply not taken."""
    s = SERIES.get(sym)
    if s is None:
        return None
    r = s["rsi"]
    n = len(r)
    for j in range(i + 1, min(i + max_wait, n - 1) + 1):
        if np.isnan(r[j]) or np.isnan(r[j - 1]):
            continue
        if direction == "long":
            if r[j - 1] < OVERSOLD <= r[j]:
                return j
        elif r[j - 1] > OVERBOUGHT >= r[j]:
            return j
    return None


def _stop_for(mode, sig, sym, j, entry):
    lo, hi = swing_of(sig)
    a = SERIES[sym]["atr"][j]
    if mode == "fib786":
        return float(sig.stop_loss)
    if mode == "swing":
        return lo if sig.direction == "long" else hi
    if mode == "swing_atr":
        if np.isnan(a):
            return None
        return (lo - 0.5 * a) if sig.direction == "long" else (hi + 0.5 * a)
    raise ValueError(mode)


def _market_row(row, j, entry, stop):
    """Rewrite a signal as a single market fill at bar j. ts, bar index and
    close all move together - engine.try_open fills at the bar_close the replay
    hands it (entry_basis = bar_close), so leaving any of them at the old bar
    would price the trade on information it does not have yet."""
    _ts, _i, _c, pos_i, sig = row
    s = SERIES[sig.symbol]
    return (pd.Timestamp(s["ts"][j]), j, float(entry), pos_i,
            replace(sig, entry_price=float(entry), stop_loss=float(stop),
                    limit_entry=None, limit_note="market", market_fraction=1.0))


def _valid(direction, entry, stop):
    return stop is not None and (stop < entry if direction == "long" else stop > entry)


def market_now(sigs, stop_mode):
    """CONTROL: the same bar the cross fires on, but entered at market."""
    def f(row):
        _ts, i, close, _p, sig = row
        stop = _stop_for(stop_mode, sig, sig.symbol, i, close)
        if not _valid(sig.direction, close, stop):
            return None
        return _market_row(row, i, close, stop)
    return map_s1(sigs, f)


def wait_for_turn(sigs, max_wait, stop_mode):
    """TREATMENT: hold fire until RSI crosses back through, enter there."""
    def f(row):
        _ts, i, _c, _p, sig = row
        j = turn_bar(sig.symbol, i, sig.direction, max_wait)
        if j is None:
            return None
        entry = float(SERIES[sig.symbol]["close"][j])
        stop = _stop_for(stop_mode, sig, sig.symbol, j, entry)
        if not _valid(sig.direction, entry, stop):
            return None
        return _market_row(row, j, entry, stop)
    return map_s1(sigs, f)


def arms(sigs):
    a = [("baseline", "baseline (as shipped)", lambda: sigs)]
    for sm in ("fib786", "swing", "swing_atr"):
        a.append(("control", f"market at signal bar / stop {sm}",
                  lambda sm=sm: market_now(sigs, sm)))
    for w in (4, 8, 12, 24, 48):
        a.append(("turn_fib", f"turn within {w}h / stop fib786",
                  lambda w=w: wait_for_turn(sigs, w, "fib786")))
    for w in (8, 24, 48):
        for sm in ("swing", "swing_atr"):
            a.append((f"turn_{sm}", f"turn within {w}h / stop {sm}",
                      lambda w=w, sm=sm: wait_for_turn(sigs, w, sm)))
    return a


def run(bars, sigs, piv, out, label, universe, block, start, end, eq):
    acct = pf.replay(bars, sigs, skip_pos=OTHER, cancel_override=4,
                     max_total_risk=0.15, start_ts=start, end_ts=end,
                     pivots_cache=piv, score_refused=False, start_equity=eq)
    row = score(label, acct, universe, block)
    print(line(row), file=out, flush=True)
    return row


def fit_confirm(bars, sigs, piv, out, universe, mid, end, eq):
    print(f"\n{'=' * 78}\n=== {universe} @ ${eq:.0f}: YEAR 1 (fit) ===",
          file=out, flush=True)
    blocks: dict = {}
    for block, label, build in arms(sigs):
        row = run(bars, build(), piv, out, label, universe, f"{block}/year1",
                  None, mid, eq)
        blocks.setdefault(block, []).append((row, label, build))
    base_n = blocks["baseline"][0][0]["n"]
    floor = max(20, int(0.25 * base_n))
    print(f"\n=== {universe} @ ${eq:.0f}: YEAR 2 (blind) ===  [min n = {floor}]",
          file=out, flush=True)
    picks = {}
    for block, rows in blocks.items():
        ok = [r for r in rows if r[0]["n"] >= floor]
        if not ok:
            # NO fallback to the full row list. An earlier cut of this fell back
            # to `or rows`, and on DEEP2Y that let turn_swing_atr - whose best
            # year-1 arm held 191 trades against a floor of 402 - win its block
            # and confirm at +0.228R on 73 trades. That is precisely the
            # sample-shrinkage artifact the floor exists to exclude, arriving
            # through the fallback. A block with nothing big enough to trust
            # has no answer, and saying so is the answer.
            print(f"  [{block}] NO ARM QUALIFIES (best n="
                  f"{max(r[0]['n'] for r in rows)} < {floor}) - not confirmed",
                  file=out, flush=True)
            continue
        best = max(ok, key=lambda r: r[0]["exp_r_drop_top3"])
        print(f"  [{block}] year-1 pick: {best[1]}", file=out, flush=True)
        picks[block] = run(bars, best[2](), piv, out, best[1], universe,
                           f"{block}/year2", mid, end, eq)
    return picks


def main():
    raw = load_deep()
    live100 = list(WATCHLIST)
    deep2y = sorted(s for s, v in raw.items()
                    if len(v["ts"]) >= HOURS_2Y + pf.WARMUP["1H"] + 5)
    live2y = [s for s in live100 if s in set(deep2y)]
    every = sorted(set(live100) | set(deep2y))
    cache, usable = frames_for(raw, every)
    bars, sigs = pf.generate(usable, HOURS_2Y, workers=10, cache=cache,
                             instance_cache_path=INSTANCE_CACHE, only_pos=[S1])
    piv = {s: confirmed_pivots(f, atr(f)) for s, f in bars.items()}
    build_series(bars)

    dest = os.path.join(os.environ.get("TMPDIR", "."), "s1_rebuild_results.txt")
    out = open(dest, "w", encoding="utf-8")
    summary = {}
    try:
        n_sig = sum(1 for rows in sigs.values() for r in rows if r[3] == S1)
        for w in (4, 8, 12, 24, 48):
            k = sum(1 for rows in wait_for_turn(sigs, w, "fib786").values()
                    for r in rows if r[3] == S1)
            print(f"turn within {w:>2}h: {k:>6}/{n_sig} signals survive "
                  f"({100 * k / n_sig:.0f}%)", file=out, flush=True)
        for name, syms in (("LIVE2Y", live2y), ("DEEP2Y", deep2y)):
            keep = set(syms) & set(bars)
            b = {s: bars[s] for s in keep}
            g = {s: sigs.get(s, []) for s in keep}
            p = {s: piv[s] for s in keep}
            top = max(f["ts"].max() for f in b.values())
            mid = top - pd.Timedelta(hours=HOURS_2Y // 2)
            summary[name] = fit_confirm(
                b, g, p, out, name, mid, top + pd.Timedelta(seconds=1), 230.0)
        print(f"\n{'=' * 78}\n=== YEAR 2 (blind), $230 ===", file=out, flush=True)
        print(f"{'universe':<9}{'block':<16}{'expR':>9}{'top3':>9}{'n':>7}"
              f"{'win%':>7}{'maxDD':>8}{'equity':>9}", file=out, flush=True)
        for name, picks in summary.items():
            for block, r in picks.items():
                print(f"{name:<9}{block:<16}{r.get('exp_r', float('nan')):>+9.3f}"
                      f"{r.get('exp_r_drop_top3', float('nan')):>+9.3f}"
                      f"{r.get('n', 0):>7}{r.get('win_pct', 0):>7.1f}"
                      f"{r.get('max_dd_pct', 0):>7.1f}%"
                      f"{r.get('equity', 0):>9.0f}", file=out, flush=True)
    finally:
        out.close()
    print("wrote", dest)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done {time.time() - t:.0f}s")
