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
# The stop as a multiple of the ATR it is buffered with. MIN_STOP_PCT is
# scale-free in percent and so cannot tell a wide stop from a wide market:
# LABUSDT's stop was 4.3% of price and 1.13 ATR simultaneously, and every one
# of the eight alerts Dror reviewed sat under 1.7 ATR (median 0.85). The stop
# is EMA20 +/- 0.10 x ATR, so its distance is just whatever the EMA9-EMA20 gap
# happens to be - there is no floor on it anywhere in the strategy.
STOP_ATR_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
# The three scale conditions on the trend read, all shipping at their disabled
# defaults. AVAXUSDT 1D passed with swing drift of -0.11 ATR on the highs and
# twelve EMA9 crossings in thirty bars.
DRIFT_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
CROSS_GRID = (999, 12, 10, 8, 6, 5, 4, 3, 2)
SPAN_GRID = (0, 10, 15, 20, 25, 30, 40, 60)


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
        for row in entries:
            ts, i, close, pos, sig, hold_base, hold_ref, (result, bars) = row[:8]
            metrics = row[8] if len(row) > 8 else {}
            # THE FILL, not the level that selected it. sig.entry_price is the
            # EMA9; ENTRY_MODE="next_open" opens at market on the next candle,
            # which by construction has closed back on the trend side - so the
            # fill is always on the far side of the level and the stop is always
            # further away than sig.entry_price says. score.simulate has scored
            # at the fill since the one-scorer change; filtering here on
            # sig.entry_price kept the sweep selecting on a trade nobody gets.
            entry = float(row[11]) if len(row) > 11 else sig.entry_price
            stop = sig.stop_loss
            atr_prev = float(row[12]) if len(row) > 12 else float("nan")
            risk = abs(entry - stop)
            if risk <= 0 or entry <= 0:
                continue
            stop_frac = risk / entry
            # NaN sorts False against every comparison, so a setup with no ATR
            # is dropped by any floor above 0 rather than silently kept.
            stop_atr = risk / atr_prev if atr_prev and atr_prev > 0 else float("nan")
            r1 = sig.reward_risk_ratio
            net_rr = (r1 * risk - MAKER * entry) / (risk + (MAKER + TAKER) * entry)

            # The R comes from score.simulate, recorded at generation. This
            # used to be re-derived HERE from the result label, with its own
            # fee arithmetic and its own idea of what the runner did - a second
            # scorer in all but name, and the reason a fixed-target model
            # survived long after the strategy stopped using one.
            gross, net = row[9], row[10]

            # Slippage, charged exactly where score.simulate charges it: on the
            # market-order legs only. A resting limit cannot fill worse than
            # its own price, so entry and target 1 never slip; the stop is a
            # trigger followed by a market fill and always does. In R that is
            # slippage / stop_fraction, so it bites hardest on the tight stops
            # this strategy is built around.
            slip_r = slippage / stop_frac
            if result == "stop":
                net -= slip_r
            elif result in ("target1 then stop", "runner open"):
                net -= 0.5 * slip_r

            out.append(
                dict(
                    symbol=sym, instance=instances[pos], paired=instances[pos][1] is not None,
                    hold=hold_ref if instances[pos][1] else hold_base,
                    hold_base=hold_base, hold_ref=hold_ref,
                    # Worst across the timeframes that must show the condition,
                    # so a filter on these is "both timeframes pass".
                    span=min([m["span"] for m in metrics.values()], default=10**6),
                    drift=min([min(m["drift_high"], m["drift_low"]) for m in metrics.values()],
                              default=10**6),
                    crossings=max([m["crossings"] for m in metrics.values()], default=0),
                    stop_frac=stop_frac, stop_atr=stop_atr, net_rr=net_rr, r1=r1,
                    gross=gross, net=net, result=result, bars=bars,
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


def table(name: str, sel: list[dict], grid, field, label, at_most: bool = False) -> None:
    """`at_most` flips the filter to `field <= v`, for a measure where MORE is
    worse. EMA9 crossings is the only one, and sweeping it with >= would have
    read it exactly backwards - selecting the choppiest setups as survivors."""
    print(f"\n{name}")
    print(f"  {label:>10s} {'n':>7s} {'win%':>6s} {'grossR':>8s} {'netR':>8s} {'SE':>7s} {'t':>6s}")
    for v in grid:
        keep = [r for r in sel if (r[field] <= v if at_most else r[field] >= v)]
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
    table("MIN_STOP_ATR  (the floor MIN_STOP_PCT cannot express)", base, STOP_ATR_GRID, "stop_atr", "stop ATR >=")
    table("MIN_NET_REWARD_RISK", base, NET_GRID, "net_rr", "net R:R >=")

    print("\n--- the three scale conditions, all shipping OFF ---")
    table("MIN_SWING_DRIFT_ATR", base, DRIFT_GRID, "drift", "drift >=")
    table("MAX_EMA9_CROSSINGS", base, CROSS_GRID, "crossings", "cross <=", at_most=True)
    table("MIN_PIVOT_SPAN_BARS", base, SPAN_GRID, "span", "span >=")

    a = np.array([r["stop_atr"] for r in base], dtype=float)
    a = a[np.isfinite(a)]
    if len(a):
        print(
            f"\nstop-in-ATR distribution: p10 {np.percentile(a,10):.2f}  p25 {np.percentile(a,25):.2f}  "
            f"median {np.median(a):.2f}  p75 {np.percentile(a,75):.2f}  p90 {np.percentile(a,90):.2f}"
        )

    print("\n--- stop-ATR floor x drift floor, net R ---")
    print(f"  {'stopATR':>8s} " + "".join(f"{d:>9.1f}" for d in DRIFT_GRID))
    for sv in STOP_ATR_GRID:
        cells = []
        for dv in DRIFT_GRID:
            keep = [r for r in base if r["stop_atr"] >= sv and r["drift"] >= dv]
            cells.append(f"{np.mean([r['net'] for r in keep]):9.2f}" if len(keep) >= 30 else f"{'-':>9s}")
        print(f"  {sv:>8.2f} " + "".join(cells))

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
