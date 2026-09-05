"""Does the LTCUSDT near-miss generalize, and does fixing it help or hurt?

    python -m backtest.s4_gap_margin_check --contexts data/s4_contexts_partial.pkl

DELIBERATELY NOT part of s4_sweep_construction.py's arms(). That sweep reuses
GapCtx exactly as recorded; margin_pct > 0 needs to re-derive each setup's
structure_context window on demand (see s4_context.build_signal's docstring
for why GapCtx itself was NOT extended to carry this - the multi-hour full
recording running when this was built stores the old shape, and touching the
dataclass mid-flight risks breaking that pickle). So this script pays a real
extra cost per setup: recomputing structure_context once per (symbol,
bar_index) reached, shared across every margin value via `window_cache` so
sweeping 7 values costs ~1x the recompute, not 7x.

REPORTS TWO THINGS. First, the LTCUSDT trade itself: does its target change
(or vanish) once the near-miss counts as closed, at the margin the case
itself implied (~0.24)? Second, the aggregate: how many OTHER setups have a
gap sitting inside their own displacement leg the same way, and what margin
does to expectancy across the whole recorded set.
"""
from __future__ import annotations

import argparse
import pickle
import statistics as st

import pandas as pd

from backtest.portfolio import replay
from backtest.s4_context import Params, build_signal
from backtest.s4_sweep_construction import (
    CANCEL, COLUMNS, CONFIRM_YEARS, FIT_YEARS, INSTANCE_POS, START_EQUITY,
    TAIL_DAYS, TIMEFRAME,
)

MARGINS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0)
LTC_CASE = ("LTCUSDT", "2023-11-09 20:00:00")


def signals_for(contexts, frames, p, window_cache):
    out = {}
    for symbol, ctxs in contexts.items():
        bars = frames.get(symbol)
        rows = []
        for ctx in ctxs:
            sig = build_signal(ctx, p, TIMEFRAME, bars=bars, window_cache=window_cache)
            if sig is not None:
                rows.append((ctx.ts, ctx.bar_index, ctx.close, INSTANCE_POS, sig))
        if rows:
            out[symbol] = rows
    return out


def year_slices(signals, frames, years):
    out = []
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
        out.append((ys, sub))
    return out


def score(trades):
    if not trades:
        return dict(n=0, win=None, exp=None, drop3=None)
    rs = sorted(t.r for t in trades)
    wins = [r for r in rs if r > 0]
    return dict(n=len(rs), win=len(wins) / len(rs), exp=st.mean(rs),
                drop3=st.mean(rs[:-3]) if len(rs) > 3 else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="data/s4_contexts_partial.pkl")
    ap.add_argument("--bars", default="data/bars_1h_deep_np.pkl")
    args = ap.parse_args()

    with open(args.contexts, "rb") as fh:
        contexts = pickle.load(fh)
    with open(args.bars, "rb") as fh:
        raw = pickle.load(fh)
    frames = {s: pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})
              for s, cols in raw.items() if s in contexts}
    del raw

    total = sum(len(v) for v in contexts.values())
    print(f"{total} setups, {len(contexts)} symbols, {len(MARGINS)} margins to test\n")

    window_cache: dict = {}
    ltc_symbol, ltc_opened = LTC_CASE
    ltc_ms = int(pd.Timestamp(ltc_opened, tz="UTC").timestamp() * 1000)

    rows = []
    for m in MARGINS:
        p = Params(gap_close_margin_pct=m)
        signals = signals_for(contexts, frames, p, window_cache)

        ltc_sig = None
        for ts, _i, _c, _pos, sig in signals.get(ltc_symbol, []):
            if abs(ts - ltc_ms) <= 3600_000:
                ltc_sig = sig
        ltc_target = None
        if ltc_sig is not None:
            risk = abs(ltc_sig.entry_price - ltc_sig.stop_loss)
            ltc_target = (ltc_sig.entry_price - risk * ltc_sig.reward_risk_ratio
                          if ltc_sig.direction == "short"
                          else ltc_sig.entry_price + risk * ltc_sig.reward_risk_ratio)

        fit_closed, conf_closed = [], []
        for label, years, sink in (("fit", FIT_YEARS, fit_closed),
                                   ("confirm", CONFIRM_YEARS, conf_closed)):
            for ys, sub in year_slices(signals, frames, years):
                acct = replay(sub, ys, cancel_override=CANCEL, start_equity=START_EQUITY)
                sink.extend(acct.closed)

        n_sig = sum(len(v) for v in signals.values())
        rows.append((m, n_sig, ltc_sig, ltc_target, score(fit_closed), score(conf_closed)))
        print(f"  margin={m:<5g} signals={n_sig:<6} "
              f"LTC={'target ' + f'{ltc_target:.3f}' if ltc_sig else 'NO SIGNAL':<20} "
              f"cache_size={len(window_cache)}", flush=True)

    print(f"\n{'margin':>7} | {'LTC target':>14} | {'signals':>7} | {'FIT':^24} | {'CONFIRM':^24}")
    print(f"{'':>7} | {'':>14} | {'':>7} | {'n':>4} {'win':>6} {'exp':>7} | "
          f"{'n':>4} {'win':>6} {'exp':>7}")
    print("-" * 90)
    for m, n_sig, ltc_sig, ltc_target, fit, conf in rows:
        ltc_str = f"{ltc_target:.3f}" if ltc_target else "no signal"
        print(f"{m:>7g} | {ltc_str:>14} | {n_sig:>7} | "
              f"{fit['n']:>4} {_p(fit['win'])} {_f(fit['exp'])} | "
              f"{conf['n']:>4} {_p(conf['win'])} {_f(conf['exp'])}")

    print(f"\n{'window cache hits'}: {len(window_cache)} distinct (symbol, bar_index) "
          f"pairs recomputed once, reused across all {len(MARGINS)} margins")


def _f(x):
    return "     - " if x is None else f"{x:>+7.3f}"


def _p(x):
    return "     -" if x is None else f"{x * 100:>5.1f}%"


if __name__ == "__main__":
    main()
