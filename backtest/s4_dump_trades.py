"""Every closed trade under one parameter arm, listed.

    python -m backtest.s4_dump_trades --min-gap-atr 0.25

A sweep row is 77 trades collapsed into one number. This prints them: the
symbol, the levels the signal actually quoted, what the trade did and what it
paid. Use it to check that an arm's trades are real setups rather than an
artifact of the rebuild.

Entry/stop/target are recovered by matching each closed trade back to the
latest signal on that symbol at or before the fill, which is the signal that
placed the order.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import pickle

import numpy as np
import pandas as pd

from backtest.portfolio import replay
from backtest.s4_context import Params, build_signal
from backtest.s4_sweep_construction import (
    CANCEL, COLUMNS, CONFIRM_YEARS, FIT_YEARS, INSTANCE_POS, START_EQUITY,
    TAIL_DAYS, TIMEFRAME,
)


def _stamp(v) -> pd.Timestamp:
    """Engine timestamps are int64 MILLISECONDS; pd.Timestamp on a bare int
    reads NANOSECONDS and lands in 1970."""
    if isinstance(v, (int, np.integer)):
        return pd.Timestamp(int(v), unit="ms")
    return pd.Timestamp(v)


def _opened_ms(trade, frames) -> int | None:
    """Closed.opened_at is a BAR INDEX while Closed.closed_at is a timestamp -
    an inconsistency in engine._close, not a conversion. So the open time has
    to be resolved through the symbol's own frame."""
    frame = frames.get(trade.symbol)
    if frame is None or not 0 <= int(trade.opened_at) < len(frame):
        return None
    return int(frame["ts"].iloc[int(trade.opened_at)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="data/s4_contexts_partial.pkl")
    ap.add_argument("--bars", default="data/bars_1h_deep_np.pkl")
    ap.add_argument("--min-gap-atr", type=float, default=0.25)
    ap.add_argument("--min-reward-risk", type=float, default=None)
    ap.add_argument("--gap-target-fraction", type=float, default=None)
    ap.add_argument("--csv", default="reports/s4_trades.csv")
    args = ap.parse_args()

    kw = {"min_gap_atr": args.min_gap_atr}
    if args.min_reward_risk is not None:
        kw["min_reward_risk"] = args.min_reward_risk
    if args.gap_target_fraction is not None:
        kw["gap_target_fraction"] = args.gap_target_fraction
    p = Params(**kw)
    print(f"ARM: {p.label()}\n")

    with open(args.contexts, "rb") as fh:
        contexts = pickle.load(fh)
    with open(args.bars, "rb") as fh:
        raw = pickle.load(fh)
    frames = {s: pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})
              for s, cols in raw.items() if s in contexts}
    del raw

    signals, by_symbol = {}, {}
    for symbol, ctxs in contexts.items():
        bars = frames.get(symbol)
        rows = []
        for ctx in ctxs:
            sig = build_signal(ctx, p, TIMEFRAME, bars=bars)
            if sig is not None:
                rows.append((ctx.ts, ctx.bar_index, ctx.close, INSTANCE_POS, sig))
        if rows:
            signals[symbol] = rows
            rows.sort(key=lambda r: r[0])
            by_symbol[symbol] = ([r[0] for r in rows], [r[4] for r in rows])

    trades = []
    for label, years in (("fit", FIT_YEARS), ("confirm", CONFIRM_YEARS)):
        for year in years:
            lo = pd.Timestamp(f"{year}-01-01", tz="UTC")
            hi = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            ys = {s: [e for e in v if lo <= pd.Timestamp(e[0], unit="ms", tz="UTC") < hi]
                  for s, v in signals.items()}
            ys = {s: v for s, v in ys.items() if v}
            if not ys:
                continue
            cut = hi + pd.Timedelta(days=TAIL_DAYS)
            sub = {s: frames[s][pd.to_datetime(frames[s]["ts"], unit="ms", utc=True) < cut]
                   .reset_index(drop=True) for s in ys if s in frames}
            acct = replay(sub, ys, cancel_override=CANCEL, start_equity=START_EQUITY)
            for t in acct.closed:
                trades.append((label, t))

    trades.sort(key=lambda e: _stamp(e[1].closed_at))
    print(f"{len(trades)} closed trades\n")
    hdr = (f"{'win':<8}{'symbol':<14}{'dir':<6}{'opened':<17}{'entry':>11}"
           f"{'stop':>11}{'target':>11}{'R:R':>6}{'exit':>10}{'R':>7}")
    print(hdr)
    print("-" * len(hdr))

    out_rows = []
    for label, t in trades:
        stamps, sigs = by_symbol.get(t.symbol, ([], []))
        open_ms = _opened_ms(t, frames)
        sig = None
        if stamps and open_ms is not None:
            i = bisect.bisect_right(stamps, open_ms) - 1
            if i >= 0:
                sig = sigs[i]
        e = sig.entry_price if sig else float("nan")
        s_ = sig.stop_loss if sig else float("nan")
        rr = sig.reward_risk_ratio if sig else float("nan")
        tgt = (e + abs(e - s_) * rr if t.direction == "long"
               else e - abs(e - s_) * rr) if sig else float("nan")
        print(f"{label:<8}{t.symbol:<14}{t.direction:<6}"
              f"{_stamp(open_ms) if open_ms else _stamp(t.closed_at):%Y-%m-%d %H:%M} "
              f"{e:>10.6g}{s_:>11.6g}{tgt:>11.6g}{rr:>6.1f}"
              f"{t.reason:>10}{t.r:>+7.2f}")
        out_rows.append(dict(window=label, symbol=t.symbol, direction=t.direction,
                             opened=_stamp(open_ms) if open_ms else None,
                             closed=_stamp(t.closed_at),
                             entry=e, stop=s_, target=tgt, planned_rr=rr,
                             exit=t.reason, r=t.r, pnl=t.pnl))

    if out_rows:
        import os
        os.makedirs(os.path.dirname(args.csv), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
            w.writeheader()
            w.writerows(out_rows)
        print(f"\nwritten to {args.csv}")

    for label in ("fit", "confirm"):
        rs = [t.r for lb, t in trades if lb == label]
        if not rs:
            continue
        wins = [r for r in rs if r > 0]
        print(f"{label:<8} n={len(rs):<4} win={len(wins) / len(rs) * 100:>5.1f}%  "
              f"meanWin={sum(wins) / len(wins) if wins else 0:>5.2f}R  "
              f"exp={sum(rs) / len(rs):>+6.3f}R")


if __name__ == "__main__":
    main()
