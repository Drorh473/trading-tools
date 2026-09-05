"""What does Strategy 4 actually call a gap, and is that the bottleneck?

    python -m backtest.s4_gap_audit --symbols 40

find_gaps is six lines: for every bar i, if bar i+2's LOW is above bar i's
HIGH there is an up-gap spanning [high_i, low_i+2]; the mirror makes a
down-gap. Nothing about the middle candle is checked, gaps are never deduped,
and there is no recency limit.

Then MIN_GAP_ATR = 1.0 throws most of them away, and the target becomes the
nearest SURVIVOR. Since winners average 4.31R while all signals average 6.44R,
how far that floor pushes the target is the question this audit answers:

  1. the size distribution of every gap found, in ATR at its own end bar
  2. what share clear the 1.0 floor
  3. how many are duplicates - the same imbalance emitted from overlapping
     three-bar windows
  4. THE CRUX: distance from price to the nearest unclosed gap of ANY size,
     versus the nearest that clears the floor. The difference is what the
     floor costs in target distance, and therefore in hit rate.
"""
from __future__ import annotations

import argparse
import pickle
import statistics as st

import pandas as pd

from notifier.strategies.order_block import (
    ATR_PERIOD, MIN_GAP_ATR, find_gaps, gap_is_closed,
)
from notifier.strategies.indicators import atr

BARS = "data/bars_1h_deep_np.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")
WINDOW = 601
STRIDE = 50


def pct(xs, p):
    return st.quantiles(xs, n=100)[p - 1] if len(xs) > 2 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default=BARS)
    ap.add_argument("--symbols", type=int, default=40)
    args = ap.parse_args()

    with open(args.bars, "rb") as fh:
        raw = pickle.load(fh)

    sizes_atr, overlaps, total_gaps = [], 0, 0
    body_ratio = []
    near_any, near_pass, no_pass = [], [], 0
    windows = 0

    picked = [(s, c) for s, c in raw.items()][: args.symbols]
    for si, (symbol, cols) in enumerate(picked, 1):
        frame = pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})
        if len(frame) < WINDOW + 100:
            continue
        for end in range(WINDOW, len(frame), STRIDE):
            window = frame.iloc[end + 1 - WINDOW: end + 1].reset_index(drop=True)
            a = atr(window, ATR_PERIOD)
            gaps = find_gaps(window)
            if not gaps:
                continue
            windows += 1
            price = float(window["close"].iloc[-1])

            prev = None
            for g in gaps:
                total_gaps += 1
                at = float(a.iloc[g.end_index])
                if at > 0:
                    sizes_atr.append(g.size / at)
                # Same imbalance seen twice: consecutive start bars whose
                # price ranges overlap.
                if prev is not None and g.start_index == prev.start_index + 1 \
                        and g.low < prev.high and g.high > prev.low:
                    overlaps += 1
                prev = g
                # How much of the wick-bounded zone a BODY-bounded one would keep.
                o1, c1 = float(window["open"].iloc[g.start_index]), float(window["close"].iloc[g.start_index])
                o3, c3 = float(window["open"].iloc[g.end_index]), float(window["close"].iloc[g.end_index])
                if g.direction == "up":
                    b_low, b_high = max(o1, c1), min(o3, c3)
                else:
                    b_low, b_high = max(o3, c3), min(o1, c1)
                if g.size > 0:
                    body_ratio.append(max(0.0, b_high - b_low) / g.size)

            # The crux, measured for a SHORT (targets below price) - the more
            # common direction in this set. Distance to the nearest unclosed
            # up-gap entirely below price, with and without the ATR floor.
            cands = [g for g in gaps
                     if g.direction == "up" and g.high < price and not gap_is_closed(g, window)]
            if not cands:
                continue
            anyg = max(g.high for g in cands)
            passing = [g for g in cands if g.size >= MIN_GAP_ATR * float(a.iloc[g.end_index])]
            near_any.append((price - anyg) / price * 100)
            if passing:
                near_pass.append((price - max(g.high for g in passing)) / price * 100)
            else:
                no_pass += 1
        if si % 10 == 0:
            print(f"  {si}/{len(picked)} symbols", flush=True)

    print(f"\n{total_gaps} gaps found across {windows} windows\n")

    print("GAP SIZE, in ATR at the gap's own end bar")
    for p in (10, 25, 50, 75, 90, 95):
        print(f"  p{p:<3} {pct(sizes_atr, p):>6.2f} ATR")
    passing = sum(1 for s in sizes_atr if s >= MIN_GAP_ATR)
    print(f"  clear the {MIN_GAP_ATR:g} ATR floor: {passing}/{len(sizes_atr)} "
          f"= {passing / max(len(sizes_atr), 1) * 100:.1f}%")
    for floor in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        n = sum(1 for s in sizes_atr if s >= floor)
        print(f"    at {floor:>4}: {n / max(len(sizes_atr), 1) * 100:>5.1f}% survive")

    print(f"\nDUPLICATES: {overlaps}/{total_gaps} = "
          f"{overlaps / max(total_gaps, 1) * 100:.1f}% of gaps overlap their "
          f"immediate predecessor")
    print(f"  (the same imbalance emitted again from the next three-bar window)")

    print(f"\nWICK vs BODY zone: a body-bounded gap keeps a median "
          f"{pct(body_ratio, 50) * 100:.0f}% of the wick-bounded one")
    collapse = sum(1 for r in body_ratio if r <= 0)
    print(f"  {collapse}/{len(body_ratio)} = {collapse / max(len(body_ratio), 1) * 100:.1f}% "
          f"would VANISH entirely on bodies")

    print(f"\nTHE CRUX - distance from price to the nearest target (shorts)")
    print(f"  nearest gap of ANY size     : median {pct(near_any, 50):>5.2f}% "
          f"of price   (n={len(near_any)})")
    print(f"  nearest clearing {MIN_GAP_ATR:g} ATR   : median {pct(near_pass, 50):>5.2f}% "
          f"of price   (n={len(near_pass)})")
    print(f"  windows with a gap but NONE clearing the floor: {no_pass}"
          f"  ({no_pass / max(len(near_any), 1) * 100:.1f}%)")


if __name__ == "__main__":
    main()
