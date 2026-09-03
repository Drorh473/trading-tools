"""Settle ONE question: does Strategy 1 do better entering 100% at market on
the RSI cross than resting 80% of the position at the 61.8% Fib?

This is the only change that has ever produced a positive blind number on this
strategy, and it survived three different setups on 2026-09-03. It is also the
change most easily faked by the $5-per-leg floor, which is why it gets its own
test instead of another block in a sweep.

There is NOTHING TO FIT here - two fixed arms, no threshold, no grid. So this
is not a fit/confirm rig; it is a replication test, and the instrument is
agreement across cells that do not share data.

WHY THE OLD "two universes agree" CHECK WAS NOT INDEPENDENT. s1_overnight
builds live2y as `[s for s in live100 if s in set(deep2y)]` - LIVE2Y is a
SUBSET of DEEP2Y, so the two universes share every one of LIVE2Y's symbols.
Agreement between them was never independent evidence, and today's LIVE2Y-vs-
DEEP2Y disagreement is a disagreement between a set and its own superset.
Here DEEP2Y is partitioned into four DISJOINT folds by a hash of the symbol
(not alphabetical - that clusters 1000-prefixed and stock tickers together),
crossed with two disjoint years: eight cells sharing no symbol and no bar.

TWO INSTRUMENTS.

  1. PORTFOLIO (the account Dror actually runs): both arms replayed per cell
     at his real $230. Reported as a sign test - if the market arm were no
     better, it should win about half the cells; winning all eight is p=1/256.

  2. PAIRED SCORER (backtest/score.simulate, explicitly NOT a portfolio): the
     SAME signal scored both ways - filled at the bar close as a taker, versus
     resting at the Fib as a maker that only trades if price reaches it within
     the live 4h cancel. No capital competition, no $5 floor, no path
     dependence, and the two arms are matched trade for trade. This is what
     isolates the entry price itself; the portfolio test says what it is worth
     on a real account. They answer different questions and both are reported.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import replace

import numpy as np
import pandas as pd

sys.path.insert(0, os.getcwd())

from backtest import portfolio as pf
from backtest import score as sc
from backtest.s1_overnight import (INSTANCE_CACHE, HOURS_2Y, S1, frames_for,
                                   drop_top3, load_deep, map_s1)
from backtest.score import confirmed_pivots
from notifier.strategies.indicators import atr
from notifier.watchlist import WATCHLIST

OTHER = {i for i in range(len(pf.INSTANCES)) if i != S1}
FOLDS = 4
CANCEL_H = 4          # tracker.ENTRY_TIMEOUT_SECONDS, the live cancel
EQUITY = 230.0        # Dror's real balance (memory:replay-start-equity)


def fold_of(symbol: str) -> int:
    """Deterministic, and uncorrelated with anything about the symbol. md5 of
    the name rather than alphabetical order, which would put every 1000-prefixed
    token in one fold and the tokenised stocks in another - those differ in
    volatility and in how they gap, so an alphabetical split would compare
    populations rather than replicate a result."""
    return int(hashlib.md5(symbol.encode()).hexdigest(), 16) % FOLDS


def to_market(sigs, closes):
    """100% at market on the signal bar; the 78.6% Fib stop is unchanged."""
    def f(row):
        _ts, i, close, pos_i, sig = row
        stop = float(sig.stop_loss)
        if (stop < close) if sig.direction == "long" else (stop > close):
            return (_ts, i, close, pos_i,
                    replace(sig, entry_price=float(close), limit_entry=None,
                            limit_note="market", market_fraction=1.0))
        return None
    return map_s1(sigs, f)


def cell(bars, sigs, piv, start, end):
    acct = pf.replay(bars, sigs, skip_pos=OTHER, cancel_override=CANCEL_H,
                     max_total_risk=0.15, start_ts=start, end_ts=end,
                     pivots_cache=piv, score_refused=False, start_equity=EQUITY)
    rs = [c.r for c in acct.closed]
    if not rs:
        return None
    return {"n": len(rs), "expR": float(np.mean(rs)), "top3": drop_top3(rs),
            "win": 100.0 * sum(1 for c in acct.closed if c.pnl > 0) / len(rs),
            "dd": 100.0 * acct.max_dd, "eq": acct.equity,
            "refused": acct.declined_too_small}


def paired_scorer(bars, sigs):
    """Same signal, both entries, matched trade for trade. An unfilled resting
    limit contributes 0R to the Fib arm - it is a trade that never happened,
    not a loss - and the market arm's same signal still counts, which is the
    whole asymmetry being measured."""
    rows = []
    for sym, found in sigs.items():
        f = bars.get(sym)
        if f is None:
            continue
        closes = f["close"].to_numpy()
        for ts, i, _c, pos_i, sig in found:
            if pos_i != S1 or i + 1 >= len(closes):
                continue
            mk = sc.simulate(f, i, sig, fill_at=float(closes[i]))
            fb = sc.simulate(f, i, sig, fill_within=CANCEL_H)
            rows.append({"symbol": sym, "ts": ts,
                         "mkt_r": mk.r_net, "mkt_res": mk.result,
                         "fib_r": fb.r_net, "fib_res": fb.result})
    return pd.DataFrame(rows)


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
    mkt = to_market(sigs, None)

    dest = os.path.join(os.environ.get("TMPDIR", "."), "s1_market_entry.txt")
    out = open(dest, "w", encoding="utf-8")
    try:
        top = max(bars[s]["ts"].max() for s in deep2y if s in bars)
        mid = top - pd.Timedelta(hours=HOURS_2Y // 2)
        end = top + pd.Timedelta(seconds=1)

        print("=" * 78, file=out)
        print("INSTRUMENT 1 - PORTFOLIO, four disjoint symbol folds x two years,"
              f" @ ${EQUITY:.0f}", file=out, flush=True)
        print("=" * 78, file=out)
        print(f"{'cell':<16}{'n base':>7}{'n mkt':>7}{'expR base':>11}"
              f"{'expR mkt':>10}{'top3 mkt':>10}{'dd base':>9}{'dd mkt':>8}"
              f"{'  winner':<9}", file=out, flush=True)
        wins = total = 0
        for k in range(FOLDS):
            syms = [s for s in deep2y if fold_of(s) == k and s in bars]
            b = {s: bars[s] for s in syms}
            p = {s: piv[s] for s in syms}
            gb = {s: sigs.get(s, []) for s in syms}
            gm = {s: mkt.get(s, []) for s in syms}
            for yr, (a, z) in (("year1", (None, mid)), ("year2", (mid, end))):
                rb, rm = cell(b, gb, p, a, z), cell(b, gm, p, a, z)
                if rb is None or rm is None:
                    print(f"fold{k}/{yr:<10} no trades", file=out, flush=True)
                    continue
                total += 1
                better = rm["expR"] > rb["expR"]
                wins += better
                print(f"fold{k}/{yr:<10}{rb['n']:>7}{rm['n']:>7}"
                      f"{rb['expR']:>+11.3f}{rm['expR']:>+10.3f}"
                      f"{rm['top3']:>+10.3f}{rb['dd']:>8.1f}%{rm['dd']:>7.1f}%"
                      f"{'  market' if better else '  baseline':<9}",
                      file=out, flush=True)
        # One-sided sign test on the ACHIEVED count. Printing only p for a clean
        # sweep would quietly overstate a 7-of-8: P(>=7 of 8) is 0.035, not the
        # 0.0039 of P(8 of 8).
        from math import comb
        p = sum(comb(total, x) for x in range(wins, total + 1)) / 2 ** total
        print(f"\nmarket entry wins {wins} of {total} disjoint cells "
              f"(a coin flip gives {total / 2:.1f}); "
              f"P(>= {wins} of {total} by chance) = {p:.3f}",
              file=out, flush=True)

        print(f"\n{'=' * 78}", file=out)
        print("INSTRUMENT 2 - PAIRED SCORER, same signals, no portfolio",
              file=out, flush=True)
        print("=" * 78, file=out)
        df = paired_scorer(bars, sigs)
        df.to_pickle(os.path.join(os.environ.get("TMPDIR", "."),
                                  "s1_paired.pkl"))
        filled = df[df.fib_res != "unfilled"]
        print(f"signals scored          {len(df)}", file=out, flush=True)
        print(f"resting limit filled    {len(filled)} "
              f"({100*len(filled)/len(df):.1f}%) within the {CANCEL_H}h cancel",
              file=out, flush=True)
        print(f"\n{'population':<34}{'n':>7}{'mkt R':>9}{'fib R':>9}"
              f"{'diff':>9}", file=out, flush=True)
        print(f"{'all signals (fib unfilled = 0R)':<34}{len(df):>7}"
              f"{df.mkt_r.mean():>+9.3f}{df.fib_r.mean():>+9.3f}"
              f"{df.mkt_r.mean()-df.fib_r.mean():>+9.3f}", file=out, flush=True)
        print(f"{'only where the limit filled':<34}{len(filled):>7}"
              f"{filled.mkt_r.mean():>+9.3f}{filled.fib_r.mean():>+9.3f}"
              f"{filled.mkt_r.mean()-filled.fib_r.mean():>+9.3f}",
              file=out, flush=True)
        print(f"\ndrop-top-3, limit-filled subset:  market "
              f"{drop_top3(filled.mkt_r.to_numpy()):+.3f}   "
              f"fib {drop_top3(filled.fib_r.to_numpy()):+.3f}",
              file=out, flush=True)
        yr1 = df[pd.to_datetime(df.ts) < mid]
        yr2 = df[pd.to_datetime(df.ts) >= mid]
        for nm, d in (("year 1", yr1), ("year 2", yr2)):
            if len(d):
                print(f"{nm:<34}{len(d):>7}{d.mkt_r.mean():>+9.3f}"
                      f"{d.fib_r.mean():>+9.3f}"
                      f"{d.mkt_r.mean()-d.fib_r.mean():>+9.3f}",
                      file=out, flush=True)
    finally:
        out.close()
    print("wrote", dest)


if __name__ == "__main__":
    t = time.time()
    main()
    print(f"done {time.time() - t:.0f}s")
