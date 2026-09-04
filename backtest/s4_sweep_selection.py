"""Tier A: can SELECTION alone make Strategy 4 clear zero?

    python -m backtest.s4_sweep_selection

WHY THIS RUNS IN MINUTES AND THE REST DOES NOT. Every arm here is a SUBSET of
the signals already generated in data/s4_signals_deep.pkl - it declines
setups, it never constructs a different trade from one. So no regeneration is
needed. Anything that moves the entry, the stop or the target builds a
different trade out of the same block and therefore cannot be done this way;
that is Tier B (s4_sweep_construction.py).

WHAT IS BEING TESTED. Measured over 198 closed trades from 2023: 13.1% win
rate, wins pay +4.31R, losses -1.00R, expectancy -0.302R. Break-even needs
either 18.8% at the current win size or 6.62R at the current hit rate, and the
PLANNED median is 6.50R - so the construction is marginally negative even when
it performs exactly as designed. Winners average 4.31R against 6.44R across
all signals, i.e. the far-target setups are the ones that do not arrive.

If that is a selection effect, declining the far-target setups should raise
expectancy. If it is not, nothing here will help and the answer lives in the
entry/target construction instead. Either result is worth the eight minutes.

PROTOCOL. Fit on 2023-2025, confirm blind on 2026. An arm is only interesting
if it survives the confirm window AND survives dropping its best three trades
there - a result that needs its top 3 is a result about three trades.

Start equity is Dror's real ~$230, not the $100 default: Bitget's flat $5
minimum notional selects on stop WIDTH, so the starting balance changes which
trades exist rather than merely scaling them.
"""
from __future__ import annotations

import argparse
import pickle
import statistics as st

import pandas as pd

from backtest.portfolio import replay

BARS = "data/bars_1h_deep_np.pkl"
SIGNALS = "data/s4_signals_deep.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")
TAIL_DAYS = 90          # room for a December trade to resolve; never a signal source
CANCEL = 30             # the live window (UNFILLED_CANDLES), not the stale default of 4
START_EQUITY = 230.0
FIT_YEARS = (2023, 2024, 2025)
CONFIRM_YEARS = (2026,)
MIN_TRADES = 30
ASIA_NOTE = "Asia session"


def _asia(sig) -> bool:
    return any(ASIA_NOTE in n for n in (sig.extra_notes or ()))


def build_arms() -> dict:
    """name -> predicate over the Signal. Every arm is a pure decline rule.

    The R:R bands sweep PAST the interesting region in both directions on
    purpose - an optimum sitting at the edge of the swept range is an artifact
    of where the sweep stopped, not a finding.
    """
    arms: dict = {"baseline (everything)": lambda s: True}
    for cap in (2.5, 3, 3.5, 4, 5, 6, 8, 10):
        arms[f"max R:R <= {cap}"] = lambda s, c=cap: s.reward_risk_ratio <= c
    for floor in (3, 4, 5):
        arms[f"min R:R >= {floor}"] = lambda s, f=floor: s.reward_risk_ratio >= f
    for lo, hi in ((2, 4), (3, 5), (4, 6), (2, 5)):
        arms[f"R:R in [{lo}, {hi}]"] = (
            lambda s, a=lo, b=hi: a <= s.reward_risk_ratio <= b)
    arms["OB1.0 only"] = lambda s: "OB1.0" in s.strategy_tag
    arms["OB2.0 only"] = lambda s: "OB2.0" in s.strategy_tag
    arms["exclude Asia-session blocks"] = lambda s: not _asia(s)
    arms["Asia-session blocks ONLY"] = _asia
    arms["longs only"] = lambda s: s.direction == "long"
    arms["shorts only"] = lambda s: s.direction == "short"
    # The two most promising single rules, combined - a joint effect is not
    # the sum of two marginal ones.
    arms["max R:R <= 5 + no Asia"] = lambda s: s.reward_risk_ratio <= 5 and not _asia(s)
    arms["max R:R <= 4 + OB2.0"] = (
        lambda s: s.reward_risk_ratio <= 4 and "OB2.0" in s.strategy_tag)
    return arms


def year_slices(signals, frames, years):
    """Per-year signal subsets plus the bars each needs, tail included."""
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
        sub = {}
        for s in ys:
            f = frames.get(s)
            if f is None:
                continue
            sub[s] = f[pd.to_datetime(f["ts"], unit="ms", utc=True) < cut].reset_index(drop=True)
        out.append((ys, sub))
    return out


def score(trades) -> dict:
    """Expectancy, and the same number with its best three trades removed."""
    if not trades:
        return dict(n=0, win=None, exp=None, drop3=None)
    rs = sorted(t.r for t in trades)
    exp = st.mean(rs)
    drop3 = st.mean(rs[:-3]) if len(rs) > 3 else None
    wins = sum(1 for r in rs if r > 0)
    return dict(n=len(rs), win=wins / len(rs), exp=exp, drop3=drop3)


def run_arm(pred, fit_slices, confirm_slices) -> tuple[dict, dict]:
    out = []
    for slices in (fit_slices, confirm_slices):
        closed = []
        for ys, sub in slices:
            filtered = {s: [e for e in v if pred(e[4])] for s, v in ys.items()}
            filtered = {s: v for s, v in filtered.items() if v}
            if not filtered:
                continue
            acct = replay(sub, filtered, cancel_override=CANCEL,
                          start_equity=START_EQUITY)
            closed.extend(acct.closed)
        out.append(score(closed))
    return out[0], out[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default=BARS)
    ap.add_argument("--signals", default=SIGNALS)
    args = ap.parse_args()

    with open(args.signals, "rb") as fh:
        signals = pickle.load(fh)
    with open(args.bars, "rb") as fh:
        raw = pickle.load(fh)
    frames = {s: pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})
              for s, cols in raw.items() if s in signals}
    del raw

    total = sum(len(v) for v in signals.values())
    print(f"{total} signals; fit {FIT_YEARS} / confirm {CONFIRM_YEARS}; "
          f"cancel={CANCEL} bars, start equity ${START_EQUITY:g}\n")

    fit_slices = year_slices(signals, frames, FIT_YEARS)
    confirm_slices = year_slices(signals, frames, CONFIRM_YEARS)

    rows = []
    arms = build_arms()
    for i, (name, pred) in enumerate(arms.items(), 1):
        fit, conf = run_arm(pred, fit_slices, confirm_slices)
        rows.append((name, fit, conf))
        print(f"  [{i:>2}/{len(arms)}] {name:<30} "
              f"fit n={fit['n']:<4} exp={_f(fit['exp'])}   "
              f"confirm n={conf['n']:<4} exp={_f(conf['exp'])}", flush=True)

    print(f"\n{'arm':<30} | {'FIT 2023-25':^28} | {'CONFIRM 2026':^28}")
    print(f"{'':<30} | {'n':>4} {'win':>6} {'exp':>7} {'drop3':>7} | "
          f"{'n':>4} {'win':>6} {'exp':>7} {'drop3':>7}")
    print("-" * 94)
    for name, fit, conf in sorted(
            rows, key=lambda r: (r[2]["exp"] is None, -(r[2]["exp"] or 0))):
        thin = "*" if (conf["n"] or 0) < MIN_TRADES else " "
        print(f"{name:<30} | {fit['n']:>4} {_p(fit['win'])} {_f(fit['exp'])} "
              f"{_f(fit['drop3'])} | {conf['n']:>4} {_p(conf['win'])} "
              f"{_f(conf['exp'])} {_f(conf['drop3'])} {thin}")
    print(f"\n* fewer than {MIN_TRADES} closed trades in the confirm window "
          f"- not a result, however good the number looks")


def _f(x) -> str:
    return "     - " if x is None else f"{x:>+7.3f}"


def _p(x) -> str:
    return "     - " if x is None else f"{x * 100:>5.1f}%"


if __name__ == "__main__":
    main()
