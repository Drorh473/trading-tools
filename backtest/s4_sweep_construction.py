"""Tier B: sweep how the TRADE is built, not which setups are taken.

    python -m backtest.s4_sweep_construction

Reads data/s4_contexts.pkl (backtest/s4_record.py) and rebuilds every trade
under each parameter set, so no arm re-runs the detector. Tier A established
that selection alone cannot fix this strategy: the only two arms positive on
the fit window were within noise of zero and both collapsed on drop-top-3.
What Tier A DID establish is that hit rate is steeply monotonic in target
distance - 60% at R:R<=2.5 down to 10% at R:R>=5 - but the low-R region is
badly under-sampled because the shipped construction rarely produces a low-R
setup. This sweep populates that region on purpose.

THE BAR AN ARM HAS TO CLEAR. Expectancy is win% x meanWin - (1-win%). The
shipped construction sits at ~-0.30R with 13.1% x 4.31R. Shortening the target
raises win% and lowers meanWin at the same time, so an arm is only interesting
if the PRODUCT moves - and only if it survives the confirm window and
drop-top-3 there.

PROTOCOL. Fit 2023-2025, confirm blind on 2026. Picking on the confirm window
is how you manufacture an edge out of noise; Tier A produced a +0.93R arm that
was -1.00R on fit, on nine trades.
"""
from __future__ import annotations

import argparse
import pickle
import statistics as st

import pandas as pd

from backtest.portfolio import replay
from backtest.s4_context import Params, build_signal

CONTEXTS = "data/s4_contexts.pkl"
BARS = "data/bars_1h_deep_np.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")
INSTANCE_POS = 4          # OrderBlockStrategy's slot in portfolio.INSTANCES
TIMEFRAME = "1H"
TAIL_DAYS = 90
CANCEL = 30
START_EQUITY = 230.0
FIT_YEARS = (2023, 2024, 2025)
CONFIRM_YEARS = (2026,)
MIN_TRADES = 30


def arms() -> dict:
    """One-factor sweeps around the shipped baseline, plus the 2D grid over the
    two knobs Dror opened. Every range extends PAST the plausible optimum in
    both directions - an optimum sitting at the edge of a swept range is a
    statement about where the sweep stopped."""
    out = {"BASELINE (ships today)": Params()}

    # EVERY KNOB THAT SHORTENS THE TARGET IS GATED BY MIN_REWARD_RISK, so each
    # one has to be swept jointly with it or the arm measures the floor rather
    # than the target. Audited 2026-09-04 over 2.1M gaps: the nearest unclosed
    # gap of any size sits a median 1.59% of price away, the nearest clearing
    # MIN_GAP_ATR=1.0 sits 4.90% away. Against a stop of roughly 1.2% of price
    # the close one implies R:R near 1.3 - refused outright by the shipped 2.0
    # floor. Lower the gap floor alone and the same trades are simply declined
    # one gate later, which would read as "shorter targets do not help".
    for f in (0.0, 0.25, 0.5):
        for mrr in (0.5, 1.0, 1.5, 2.0):
            if f == 0.5 and mrr == 2.0:
                continue  # the baseline
            out[f"target {f:g}, minR:R {mrr:g}"] = Params(
                gap_target_fraction=f, min_reward_risk=mrr)
    for f in (0.15, 0.75, 1.0):
        out[f"target fraction {f:g}"] = Params(gap_target_fraction=f)
    # Moving the entry toward the block's NEAR edge moves it toward the stop,
    # so risk shrinks and the computed R:R balloons straight into
    # MAX_REWARD_RISK. Measured on the smoke run: entry 0.0 left 3 signals
    # where entry 0.25 left 42 - almost all of the difference was the cap
    # firing, not the entry being bad. Each entry arm therefore gets an
    # uncapped twin, so the sweep reads the entry rather than the ceiling.
    for f in (0.0, 0.25, 0.75, 1.0):
        out[f"entry fraction {f:g}"] = Params(entry_fraction=f)
        out[f"entry {f:g}, R:R uncapped"] = Params(entry_fraction=f, max_reward_risk=1e9)
    out["BASELINE, R:R uncapped"] = Params(max_reward_risk=1e9)
    # The gap floor is the single biggest determinant of target distance: it
    # discards 91.8% of gaps (median gap 0.27 ATR against a 1.0 floor) and in
    # 42% of windows leaves a live gap with nothing clearing the bar, so the
    # setup is declined for want of a target. Crossed with the R:R floor for
    # the same reason as above.
    for g in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        for mrr in (1.0, 2.0):
            if g == 1.0 and mrr == 2.0:
                continue  # the baseline
            out[f"gap {g:g} ATR, minR:R {mrr:g}"] = Params(
                min_gap_atr=g, min_reward_risk=mrr)
    # Both target knobs pulled together, at the floor that lets them through -
    # a near-edge target on a gap floor low enough to find a near gap at all.
    for g in (0.25, 0.5):
        for t in (0.0, 0.25):
            out[f"gap {g:g} + target {t:g}, minR:R 1"] = Params(
                min_gap_atr=g, gap_target_fraction=t, min_reward_risk=1.0)
    for s in (0.25, 0.75, 1.0, 1.5):
        out[f"stop buffer {s:g} ATR"] = Params(stop_atr_buffer=s)
    for m in (3, 4, 5, 6, 8):
        out[f"max R:R {m:g}"] = Params(max_reward_risk=m)
    for m in (1.0, 1.5, 3.0):
        out[f"min R:R {m:g}"] = Params(min_reward_risk=m)
    out["no Asia-session blocks"] = Params(asia_gated=True)
    # The joint grid: these two are the same trade-off pulled from opposite
    # ends, so their interaction is not the sum of the marginals.
    for e in (0.0, 0.25, 0.5):
        for t in (0.0, 0.25, 0.5):
            if e == 0.5 and t == 0.5:
                continue  # that is the baseline
            out[f"grid entry {e:g} / target {t:g}"] = Params(
                entry_fraction=e, gap_target_fraction=t)
    return out


def signals_for(contexts, p: Params) -> dict:
    """Rebuild the whole signal set under one parameter arm."""
    out = {}
    for symbol, ctxs in contexts.items():
        rows = []
        for ctx in ctxs:
            sig = build_signal(ctx, p, TIMEFRAME)
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
               .reset_index(drop=True)
               for s in ys if s in frames}
        out.append((ys, sub))
    return out


def score(trades, n_signals) -> dict:
    if not trades:
        return dict(n=0, sig=n_signals, win=None, exp=None, drop3=None, meanwin=None)
    rs = sorted(t.r for t in trades)
    wins = [r for r in rs if r > 0]
    return dict(n=len(rs), sig=n_signals,
                win=len(wins) / len(rs), exp=st.mean(rs),
                drop3=st.mean(rs[:-3]) if len(rs) > 3 else None,
                meanwin=st.mean(wins) if wins else None)


def run(contexts, frames, p: Params) -> tuple[dict, dict]:
    signals = signals_for(contexts, p)
    n_signals = sum(len(v) for v in signals.values())
    res = []
    for years in (FIT_YEARS, CONFIRM_YEARS):
        closed = []
        for ys, sub in year_slices(signals, frames, years):
            acct = replay(sub, ys, cancel_override=CANCEL, start_equity=START_EQUITY)
            closed.extend(acct.closed)
        res.append(score(closed, n_signals))
    return res[0], res[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default=CONTEXTS)
    ap.add_argument("--bars", default=BARS)
    args = ap.parse_args()

    with open(args.contexts, "rb") as fh:
        contexts = pickle.load(fh)
    with open(args.bars, "rb") as fh:
        raw = pickle.load(fh)
    frames = {s: pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})
              for s, cols in raw.items() if s in contexts}
    del raw

    total = sum(len(v) for v in contexts.values())
    print(f"{total} recorded setups across {len(contexts)} symbols")
    print(f"fit {FIT_YEARS} / confirm {CONFIRM_YEARS}, cancel={CANCEL}, "
          f"start ${START_EQUITY:g}\n")

    rows = []
    a = arms()
    for i, (name, p) in enumerate(a.items(), 1):
        fit, conf = run(contexts, frames, p)
        rows.append((name, p, fit, conf))
        print(f"  [{i:>2}/{len(a)}] {name:<28} fit n={fit['n']:<4} "
              f"exp={_f(fit['exp'])}  confirm n={conf['n']:<4} exp={_f(conf['exp'])}",
              flush=True)

    print(f"\n{'arm':<28} | {'signals':>7} | {'FIT 2023-25':^30} | {'CONFIRM 2026':^30}")
    print(f"{'':<28} | {'':>7} | {'n':>4} {'win':>6} {'mw':>6} {'exp':>7} | "
          f"{'n':>4} {'win':>6} {'mw':>6} {'exp':>7} {'drop3':>7}")
    print("-" * 112)
    for name, _params, fit, conf in sorted(
            rows, key=lambda r: (r[2]["exp"] is None, -(r[2]["exp"] or -99))):
        thin = "*" if (conf["n"] or 0) < MIN_TRADES else " "
        print(f"{name:<28} | {fit['sig']:>7} | {fit['n']:>4} {_p(fit['win'])} "
              f"{_m(fit['meanwin'])} {_f(fit['exp'])} | {conf['n']:>4} "
              f"{_p(conf['win'])} {_m(conf['meanwin'])} {_f(conf['exp'])} "
              f"{_f(conf['drop3'])} {thin}")
    print(f"\nSorted by FIT expectancy - that is the window an arm is allowed to be")
    print(f"picked on. Read CONFIRM only for the arms that already won on fit.")
    print(f"* fewer than {MIN_TRADES} closed trades in confirm - not a result.")
    print("mw = mean winning trade, in R. Expectancy needs win% x mw > (1 - win%).")


def _f(x) -> str:
    return "     - " if x is None else f"{x:>+7.3f}"


def _m(x) -> str:
    return "     -" if x is None else f"{x:>6.2f}"


def _p(x) -> str:
    return "     -" if x is None else f"{x * 100:>5.1f}%"


if __name__ == "__main__":
    main()
