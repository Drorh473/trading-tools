"""Check the R:R arithmetic I gave with two approximations in it.

I claimed a near gap implies R:R near 1.3 and therefore needs a >43% win rate.
Two things in that were loose and are checked here against real numbers:

  1. the ~1.2% stop width came from an older 57-signal sample, not from the
     11,243-signal set the current rules produce
  2. the 1.59% / 4.90% target distances were measured for SHORTS only

Reports the real stop-width distribution, target distance for both
directions, and the implied R:R and break-even win rate at each.
"""
from __future__ import annotations

import argparse
import pickle
import statistics as st

import pandas as pd

from notifier.strategies.indicators import atr
from notifier.strategies.order_block import (
    ATR_PERIOD, MIN_GAP_ATR, find_gaps, gap_is_closed,
)

BARS = "data/bars_1h_deep_np.pkl"
SIGNALS = "data/s4_signals_deep.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")
WINDOW = 601
STRIDE = 50


def q(xs, p):
    return st.quantiles(xs, n=100)[p - 1] if len(xs) > 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=60)
    args = ap.parse_args()

    # ---- 1. the real stop width, from the signals the current rules produce
    with open(SIGNALS, "rb") as fh:
        sigs = pickle.load(fh)
    widths, ratios = [], []
    for v in sigs.values():
        for e in v:
            s = e[4]
            widths.append(abs(s.entry_price - s.stop_loss) / s.entry_price * 100)
            ratios.append(s.reward_risk_ratio)
    print(f"STOP WIDTH over {len(widths)} real signals, as % of entry")
    for p in (10, 25, 50, 75, 90):
        print(f"  p{p:<3} {q(widths, p):>6.2f}%")
    stop_med = q(widths, 50)
    print(f"  (the figure I quoted earlier was ~1.20% from an older sample)")
    print(f"\nPLANNED R:R over the same signals: median {q(ratios, 50):.2f}, "
          f"mean {st.mean(ratios):.2f}")

    # ---- 2. target distance, BOTH directions
    with open(BARS, "rb") as fh:
        raw = pickle.load(fh)
    out = {"short": {"any": [], "pass": [], "none": 0},
           "long": {"any": [], "pass": [], "none": 0}}

    for si, (symbol, cols) in enumerate(list(raw.items())[: args.symbols], 1):
        frame = pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})
        if len(frame) < WINDOW + 100:
            continue
        for end in range(WINDOW, len(frame), STRIDE):
            window = frame.iloc[end + 1 - WINDOW: end + 1].reset_index(drop=True)
            a = atr(window, ATR_PERIOD)
            gaps = find_gaps(window)
            if not gaps:
                continue
            price = float(window["close"].iloc[-1])
            for direction in ("short", "long"):
                want = "up" if direction == "short" else "down"
                cands = [g for g in gaps if g.direction == want
                         and (g.high < price if direction == "short" else g.low > price)
                         and not gap_is_closed(g, window)]
                if not cands:
                    continue
                if direction == "short":
                    nearest = max(g.high for g in cands)
                    dist = (price - nearest) / price * 100
                else:
                    nearest = min(g.low for g in cands)
                    dist = (nearest - price) / price * 100
                out[direction]["any"].append(dist)
                ok = [g for g in cands
                      if g.size >= MIN_GAP_ATR * float(a.iloc[g.end_index])]
                if not ok:
                    out[direction]["none"] += 1
                    continue
                if direction == "short":
                    d2 = (price - max(g.high for g in ok)) / price * 100
                else:
                    d2 = (min(g.low for g in ok) - price) / price * 100
                out[direction]["pass"].append(d2)
        if si % 20 == 0:
            print(f"  ..{si} symbols", flush=True)

    print(f"\nTARGET DISTANCE, median % of price, and what it implies at a "
          f"{stop_med:.2f}% stop")
    print(f"  {'':<8} {'nearest any':>12} {'R:R':>6} {'BE win':>8} | "
          f"{'clears floor':>13} {'R:R':>6} {'BE win':>8} | {'no target':>10}")
    for d in ("short", "long"):
        a_med = q(out[d]["any"], 50)
        p_med = q(out[d]["pass"], 50)
        rr_a, rr_p = a_med / stop_med, p_med / stop_med
        be_a, be_p = 1 / (1 + rr_a) * 100, 1 / (1 + rr_p) * 100
        share = out[d]["none"] / max(len(out[d]["any"]), 1) * 100
        print(f"  {d:<8} {a_med:>11.2f}% {rr_a:>6.2f} {be_a:>7.1f}% | "
              f"{p_med:>12.2f}% {rr_p:>6.2f} {be_p:>7.1f}% | {share:>9.1f}%")
    print(f"\n  n(short)={len(out['short']['any'])}  n(long)={len(out['long']['any'])}")
    print("  BE win = win rate needed to break even at that R:R, ignoring fees.")


if __name__ == "__main__":
    main()
