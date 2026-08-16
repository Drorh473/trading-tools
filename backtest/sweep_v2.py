"""Sweep Strategy 2 v2's three unevidenced thresholds over one generated population.

    python -m backtest.sweep_v2 --signals data/signals_v2.pkl

EMA9_HOLD_BARS, MIN_STOP_PCT and MIN_NET_REWARD_RISK were all picked rather
than measured - the hold length in particular came from v1 ("10 is just a
number I threw"). generate_v2 runs with all three OFF and records what each
setup actually had, so every candidate value is a filter over the same
population and the whole sweep costs seconds.

WHAT THIS IS NOT: a portfolio. There is no competition for capital, no
aggregate cap, no margin, and no slippage. Trades overlap freely, so the
effective sample is smaller than n suggests and the t-statistics are
optimistic. It compares RULE VARIANTS against one another; it does not say what
the strategy earns. §42 is the standing lesson - a harness can be correct in
every part and wrong as a whole.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter

import numpy as np

MAKER, TAKER = 0.0002, 0.0006
HOLD_GRID = (0, 1, 2, 3, 5, 8, 10, 15, 20, 30)
STOP_GRID = (0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.02)
NET_GRID = (0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)


def rows(signals: dict, instances, slippage: float = 0.0) -> list[dict]:
    """One dict per generated setup, with everything a filter might key on.

    `slippage` is a fraction of price and is charged ONLY where the exit is a
    market order. v2 enters on a resting limit and takes profit on a limit, and
    a resting limit cannot fill worse than its own price - so neither of those
    slips. The STOP does: it is a trigger followed by a market fill, and it is
    where most of these trades end.

    Charged in R it is slippage / stop_fraction, so it scales inversely with
    how tight the stop is - which is why it matters most to exactly the trades
    the design is built around.
    """
    out = []
    for sym, entries in signals.items():
        for ts, i, close, pos, sig, hold_base, hold_ref, (result, bars) in entries:
            entry, stop = sig.entry_price, sig.stop_loss
            risk = abs(entry - stop)
            if risk <= 0 or entry <= 0:
                continue
            stop_frac = risk / entry
            r1 = sig.reward_risk_ratio
            r2 = abs(sig.remainder_target - entry) / risk
            net_rr = (r1 * risk - MAKER * entry) / (risk + (MAKER + TAKER) * entry)
            fee_in, fee_stop, fee_tgt = MAKER / stop_frac, TAKER / stop_frac, MAKER / stop_frac

            slip = slippage / stop_frac  # in R, and worse the tighter the stop

            if result == "both targets":
                # both tiers are resting limits: no market order, no slippage
                gross, fees = 0.5 * r1 + 0.5 * r2, fee_in + fee_tgt
            elif result == "target1 then stop":
                # first half on a limit, the runner stopped out at breakeven
                gross, fees = 0.5 * r1 - 0.5 * slip, fee_in + 0.5 * fee_tgt + 0.5 * fee_stop
            elif result == "target1, runner open":
                gross, fees = 0.5 * r1, fee_in + 0.5 * fee_tgt
            elif result == "stop":
                gross, fees = -1.0 - slip, fee_in + fee_stop
            else:
                gross, fees = 0.0, fee_in + fee_stop

            out.append(
                dict(
                    symbol=sym, instance=instances[pos], paired=instances[pos][1] is not None,
                    # The hold is proved by the TREND-SETTING timeframe: the
                    # reference for a pair, the only timeframe for a standalone.
                    # Using min() here would be the both-timeframes rule, which
                    # takes paired to n=1 at this threshold and was replaced
                    # precisely because it did that.
                    hold=hold_ref if instances[pos][1] else hold_base,
                    hold_base=hold_base, hold_ref=hold_ref,
                    stop_frac=stop_frac, net_rr=net_rr, r1=r1,
                    gross=gross, net=gross - fees, result=result, bars=bars,
                )
            )
    return out


def stat(sel: list[dict], key: str = "net") -> tuple:
    if not sel:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")
    a = np.array([r[key] for r in sel], dtype=float)
    g = np.array([r["gross"] for r in sel], dtype=float)
    se = a.std(ddof=1) / len(a) ** 0.5 if len(a) > 1 else float("nan")
    return len(a), a.mean(), se, (a.mean() / se if se else float("nan")), 100 * (g > 0).mean()


def table(name: str, sel: list[dict], grid, field, label) -> None:
    print(f"\n{name}")
    print(f"  {label:>10s} {'n':>7s} {'win%':>6s} {'grossR':>8s} {'netR':>8s} {'SE':>7s} {'t':>6s}")
    for v in grid:
        keep = [r for r in sel if r[field] >= v]
        n, mean, se, t, win = stat(keep)
        gm = np.mean([r["gross"] for r in keep]) if keep else float("nan")
        show = f"{v:.4g}" if isinstance(v, float) else str(v)
        print(f"  {show:>10s} {n:7d} {win:6.1f} {gm:8.3f} {mean:8.3f} {se:7.3f} {t:6.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", default="data/signals_v2.pkl")
    ap.add_argument("--min-hold", type=int, default=0)
    args = ap.parse_args()

    key, instances, signals = pickle.load(open(args.signals, "rb"))
    data = rows(signals, instances)
    print(f"{len(data)} setups across {len(signals)} symbols, {len(instances)} instances")
    print("NO portfolio, NO slippage, overlapping trades counted independently.\n")

    print("outcomes:", dict(Counter(r["result"] for r in data).most_common()))
    print("by instance:", dict(Counter(str(r["instance"]) for r in data).most_common()))

    h = np.array([r["hold"] for r in data])
    print(
        f"\nhold-run distribution (bars the EMA9 actually held): "
        f"p10 {np.percentile(h,10):.0f}  p25 {np.percentile(h,25):.0f}  "
        f"median {np.median(h):.0f}  p75 {np.percentile(h,75):.0f}  p90 {np.percentile(h,90):.0f}  max {h.max():.0f}"
    )

    base = [r for r in data if r["hold"] >= args.min_hold]
    table("EMA9_HOLD_BARS  (the number to pick)", base, HOLD_GRID, "hold", "hold >=")
    table("MIN_STOP_PCT", base, STOP_GRID, "stop_frac", "stop >=")
    table("MIN_NET_REWARD_RISK", base, NET_GRID, "net_rr", "net R:R >=")

    print("\n--- hold swept separately for paired and standalone ---")
    for paired, name in ((False, "standalone"), (True, "paired")):
        table(f"  {name}", [r for r in base if r["paired"] is paired], HOLD_GRID, "hold", "hold >=")

    print("\n--- hold x stop, net R (the two gates interact) ---")
    print(f"  {'hold':>6s} " + "".join(f"{s*100:>9.2f}%" for s in STOP_GRID[1:]))
    for hv in HOLD_GRID:
        cells = []
        for sv in STOP_GRID[1:]:
            keep = [r for r in base if r["hold"] >= hv and r["stop_frac"] >= sv]
            cells.append(f"{np.mean([r['net'] for r in keep]):9.2f}" if len(keep) >= 30 else f"{'-':>9s}")
        print(f"  {hv:>6d} " + "".join(cells))
    print("\n  (cells with fewer than 30 trades left blank)")


if __name__ == "__main__":
    main()
