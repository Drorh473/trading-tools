"""Is the market-entry result about the ENTRY PRICE, or about TRADE WIDTH?

Market entry won 7 of 8 disjoint cells (backtest/s1_market_entry.py), but the
paired scorer showed it does NOT get a better price - on the trades where the
resting limit filled, the Fib fill beat the market fill by 0.095R. And the
portfolio takes only ~350 trades on the market arm against ~1,200 on the
baseline, from the same signals.

The explanation those two facts point at: entering at the bar close puts the
fill much FURTHER from the 78.6% Fib stop than the split entry's blended price
does, so risk-per-unit is wider, positions are smaller, trades last longer and
each holds its symbol slot and margin for longer. If that is the whole story,
the "edge" is a position-sizing and holding-time change wearing an entry
rule's clothes - and it would be honest to ship it as that, or not at all.

So: a 2x2 that varies the two factors independently.

              entry = blended Fib          entry = market close
  risk R_base   A  baseline (as shipped)     C  market, stop pulled in
  risk R_wide   D  split entry, stop pushed  B  market + 78.6% Fib stop

  A is what runs today. B is the 7-of-8 winner. C isolates ENTRY PRICE
  (market fill, baseline's width). D isolates WIDTH (shipped entry mechanic,
  the market arm's width).

  B ~ D >> A, C   -> it was width and holding time all along
  B ~ C >> A, D   -> it is genuinely the entry price
  B >> C, D       -> the two only pay together

Per signal, with c the bar close, e the 61.8% Fib and s the 78.6% Fib stop:

    b       = 0.2*c + 0.8*e          the split entry's blended fill price
    R_base  = |b - s|                what the account risks per unit today
    R_wide  = |c - s|                what the market arm risks per unit

Same four disjoint symbol folds and two disjoint years as s1_market_entry, at
Dror's real $230, so the cells are comparable across both scripts.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import replace
from math import comb

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from backtest import portfolio as pf
from backtest.s1_market_entry import EQUITY, CANCEL_H, FOLDS, fold_of
from backtest.s1_overnight import (INSTANCE_CACHE, HOURS_2Y, S1, drop_top3,
                                   frames_for, load_deep, map_s1)
from backtest.score import confirmed_pivots
from notifier.strategies.indicators import atr
from notifier.watchlist import WATCHLIST

OTHER = {i for i in range(len(pf.INSTANCES)) if i != S1}
MKT_FRACTION = 0.2    # rsi_fib_reversal.MARKET_ENTRY_FRACTION


def _parts(row):
    """(close, blended entry, stop, R_base, R_wide) for one signal."""
    _ts, _i, c, _p, sig = row
    c = float(c)
    e, s = float(sig.entry_price), float(sig.stop_loss)
    b = MKT_FRACTION * c + (1 - MKT_FRACTION) * e
    return c, b, s, abs(b - s), abs(c - s)


def _long(sig):
    return sig.direction == "long"


def arm_market_wide(sigs):
    """B: 100% at market, keep the 78.6% Fib stop. The 7-of-8 winner."""
    def f(row):
        c, _b, s, _rb, _rw = _parts(row)
        sig = row[4]
        if (s >= c) if _long(sig) else (s <= c):
            return None
        return (row[0], row[1], c, row[3],
                replace(sig, entry_price=c, limit_entry=None,
                        limit_note="market", market_fraction=1.0))
    return map_s1(sigs, f)


def arm_market_matched(sigs):
    """C: 100% at market, stop pulled IN so risk-per-unit equals the baseline's.
    Same fill as B, same position size as A - so a difference from B is width
    and a difference from A is the entry price."""
    def f(row):
        c, _b, _s, rb, _rw = _parts(row)
        sig = row[4]
        if not np.isfinite(rb) or rb <= 0:
            return None
        stop = c - rb if _long(sig) else c + rb
        return (row[0], row[1], c, row[3],
                replace(sig, entry_price=c, stop_loss=stop, limit_entry=None,
                        limit_note="market", market_fraction=1.0))
    return map_s1(sigs, f)


def arm_split_widened(sigs):
    """D: the shipped split entry untouched, stop PUSHED OUT so risk-per-unit
    equals the market arm's. Same trades and same fills as A, B's width."""
    def f(row):
        _c, b, _s, _rb, rw = _parts(row)
        sig = row[4]
        if not np.isfinite(rw) or rw <= 0:
            return None
        stop = b - rw if _long(sig) else b + rw
        return (row[0], row[1], row[2], row[3], replace(sig, stop_loss=stop))
    return map_s1(sigs, f)


ARMS = [
    ("A baseline (shipped)", lambda s: s),
    ("B market + fib stop", arm_market_wide),
    ("C market, width matched", arm_market_matched),
    ("D split, width widened", arm_split_widened),
]


def cell(bars, sigs, piv, start, end):
    acct = pf.replay(bars, sigs, skip_pos=OTHER, cancel_override=CANCEL_H,
                     max_total_risk=0.15, start_ts=start, end_ts=end,
                     pivots_cache=piv, score_refused=False, start_equity=EQUITY)
    rs = [c.r for c in acct.closed]
    if not rs:
        return None
    return {"n": len(rs), "expR": float(np.mean(rs)), "top3": drop_top3(rs),
            "dd": 100.0 * acct.max_dd, "eq": acct.equity,
            "refused": acct.declined_too_small}


def main():
    raw = load_deep()
    live100 = list(WATCHLIST)
    deep2y = sorted(s for s, v in raw.items()
                    if len(v["ts"]) >= HOURS_2Y + pf.WARMUP["1H"] + 5)
    every = sorted(set(live100) | set(deep2y))
    cache, usable = frames_for(raw, every)
    bars, sigs = pf.generate(usable, HOURS_2Y, workers=10, cache=cache,
                             instance_cache_path=INSTANCE_CACHE, only_pos=[S1])
    piv = {s: confirmed_pivots(f, atr(f)) for s, f in bars.items()}
    built = {name: fn(sigs) for name, fn in ARMS}

    dest = os.path.join(os.environ.get("TMPDIR", "."), "s1_width_vs_entry.txt")
    out = open(dest, "w", encoding="utf-8")
    try:
        # How different the two widths actually are - if R_wide is barely wider
        # than R_base the whole 2x2 has nothing to separate and the rest of
        # this file is noise.
        ratios = []
        for rows in sigs.values():
            for row in rows:
                if row[3] != S1:
                    continue
                _c, _b, _s, rb, rw = _parts(row)
                if rb > 0 and np.isfinite(rw / rb):
                    ratios.append(rw / rb)
        r = np.array(ratios)
        print(f"R_wide / R_base over {len(r)} signals: median {np.median(r):.2f}x"
              f"  quartiles {np.percentile(r, 25):.2f}x/{np.percentile(r, 75):.2f}x",
              file=out, flush=True)

        top = max(bars[s]["ts"].max() for s in deep2y if s in bars)
        mid = top - pd.Timedelta(hours=HOURS_2Y // 2)
        end = top + pd.Timedelta(seconds=1)

        cells, per_arm = [], {n: [] for n, _ in ARMS}
        print(f"\n{'cell':<14}" + "".join(f"{n.split()[0]:>26}" for n, _ in ARMS),
              file=out, flush=True)
        print(f"{'':<14}" + "".join(f"{'n':>7}{'expR':>9}{'top3':>10}"
                                    for _ in ARMS), file=out, flush=True)
        for k in range(FOLDS):
            syms = [s for s in deep2y if fold_of(s) == k and s in bars]
            b = {s: bars[s] for s in syms}
            p = {s: piv[s] for s in syms}
            for yr, (a, z) in (("year1", (None, mid)), ("year2", (mid, end))):
                line = f"fold{k}/{yr:<8}"
                res = {}
                for name, _ in ARMS:
                    g = {s: built[name].get(s, []) for s in syms}
                    rr = cell(b, g, p, a, z)
                    res[name] = rr
                    if rr is None:
                        line += f"{'-':>26}"
                        continue
                    per_arm[name].append(rr["expR"])
                    line += f"{rr['n']:>7}{rr['expR']:>+9.3f}{rr['top3']:>+10.3f}"
                cells.append(res)
                print(line, file=out, flush=True)

        print(f"\n{'arm':<26}{'mean expR':>11}{'mean top3':>11}"
              f"{'beats A':>9}{'p':>8}{'mean n':>8}", file=out, flush=True)
        base = per_arm["A baseline (shipped)"]
        for name, _ in ARMS:
            v = per_arm[name]
            if not v:
                continue
            tops = [c[name]["top3"] for c in cells if c[name]]
            ns = [c[name]["n"] for c in cells if c[name]]
            wins = sum(1 for c in cells
                       if c[name] and c["A baseline (shipped)"]
                       and c[name]["expR"] > c["A baseline (shipped)"]["expR"])
            tot = len(v)
            p = sum(comb(tot, x) for x in range(wins, tot + 1)) / 2 ** tot
            tag = "" if name.startswith("A") else f"{wins}/{tot}"
            print(f"{name:<26}{np.mean(v):>+11.3f}{np.mean(tops):>+11.3f}"
                  f"{tag:>9}{'' if name.startswith('A') else f'{p:>8.3f}'}"
                  f"{np.mean(ns):>8.0f}", file=out, flush=True)

        print("\nreading: B~D >> A,C = width and holding time. "
              "B~C >> A,D = the entry price. B >> C,D = only together.",
              file=out, flush=True)
    finally:
        out.close()
    print("wrote", dest)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done {time.time() - t:.0f}s")
