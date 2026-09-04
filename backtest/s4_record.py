"""Run Strategy 4's detector once over the universe and record its raw output.

    python -m backtest.s4_record --workers 10

Writes data/s4_contexts.pkl as {symbol: [SetupCtx, ...]} - every bar at which
at least one order block was live, with the gaps that were candidate targets.
backtest/s4_sweep_construction.py then rebuilds trades from these under many
parameter sets without re-running the detector, which is the whole point: the
detector is ~all of the 6.5h and none of it depends on the swept parameters.

WINDOW CAPPING is copied from generate_s4_deep and matters: the live scanner
fetches candle_limit (600) bars and never sees more, so an uncapped window
lets structure_context search past anything the bot could do - and at 5,000
bars one evaluate costs 3.4s against 29ms at the live size.

TS IS INT64 MILLISECONDS in these caches. pd.Timestamp on a bare int reads
NANOSECONDS and lands silently in 1970 - which is exactly the bug that made
the Asia-session read uniformly False across all 11,243 deep signals.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from multiprocessing import Pool

import pandas as pd

from backtest.portfolio import WARMUP
from backtest.s4_context import BlockCtx, GapCtx, SetupCtx
from notifier.strategies import order_block as ob
from notifier.strategies.order_block import OrderBlockStrategy
from notifier.strategies.structure import zigzag_pivots

BARS_DEFAULT = "data/bars_1h_deep_np.pkl"
OUT_DEFAULT = "data/s4_contexts.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")
LIVE_WINDOW = 601
TIMEFRAME = "1H"
RECORD_STEEPNESS = 0.0  # record everything; the floor is swept at replay


def _frame(cols: dict) -> pd.DataFrame:
    return pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})


def record_symbol(task):
    symbol, cols = task
    # PERMISSIVE ON PURPOSE. The steepness floor is the last gate in
    # _qualifies and the expansion loop advances unconditionally, so recording
    # at 0.0 strictly ADDS expansions and every block still carries its own
    # measured steepness. The shipped floor is then one filter among the sweep
    # arms rather than something baked into the recording.
    strat = OrderBlockStrategy(TIMEFRAME, session_gated=False, min_steepness=RECORD_STEEPNESS)
    frame = _frame(cols)
    n = len(frame)
    if n <= WARMUP[TIMEFRAME] + 100:
        return symbol, []

    stamps = frame["ts"].to_numpy()
    closes = frame["close"].to_numpy()
    out = []
    for i in range(WARMUP[TIMEFRAME] + 1, n):
        lo = max(0, i + 1 - LIVE_WINDOW)
        bars = frame.iloc[lo: i + 1]
        try:
            ctx = context_for(strat, symbol, bars, int(stamps[i]), i, float(closes[i]))
        except Exception:
            continue
        if ctx is not None:
            out.append(ctx)
    return symbol, out


def context_for(strat, symbol, bars, ts, bar_index, close):
    """evaluate()'s detector half, stopping before any parametric decision."""
    window, structure = ob.structure_context(
        bars, atr_multiple=ob.STRUCTURE_ATR_MULTIPLE, atr_period=ob.ATR_PERIOD)
    if structure.trend is None or structure.anchor_index is None:
        return None
    direction = "long" if structure.trend == "up" else "short"

    anchor = structure.anchor_index
    range_start = max(0, anchor - ob.DEALING_RANGE_LOOKBACK)
    leg_high = float(window["high"].iloc[range_start: anchor + 1].max())
    leg_low = float(window["low"].iloc[range_start: anchor + 1].min())
    if leg_high <= leg_low:
        return None
    equilibrium = (leg_high + leg_low) / 2

    atr_series = ob.atr(window, ob.ATR_PERIOD)
    pivots = zigzag_pivots(window, atr_series * ob.STRUCTURE_ATR_MULTIPLE)
    gaps = ob.find_gaps(window)
    blocks = strat._find_blocks(window, atr_series, pivots, gaps, direction)
    if not blocks:
        return None

    price = float(window["close"].iloc[-1])
    # Only gaps that could ever be a target under SOME parameter set: the right
    # direction, still open, and beyond price. MIN_GAP_ATR is deliberately NOT
    # applied here - it is the thing being swept - so size and the ATR it is
    # measured against are both recorded instead.
    want = "down" if direction == "long" else "up"
    gap_ctxs = []
    for gap in gaps:
        if gap.direction != want:
            continue
        if ob.gap_is_closed(gap, window):
            continue
        if direction == "long" and not gap.low > price:
            continue
        if direction == "short" and not gap.high < price:
            continue
        gap_ctxs.append(GapCtx(
            direction=gap.direction, low=float(gap.low), high=float(gap.high),
            size=float(gap.size), atr_at_end=float(atr_series.iloc[gap.end_index]),
            start_index=int(gap.start_index), end_index=int(gap.end_index)))
    if not gap_ctxs:
        return None

    block_ctxs = [
        BlockCtx(low=float(b.low), high=float(b.high), direction=b.direction,
                 variant=b.variant, index=int(b.index),
                 displacement_end=int(b.displacement_end),
                 sweep_level=float(b.sweep_level), sweep_extreme=float(b.sweep_extreme),
                 in_asia=bool(strat._in_asia_session(window, b.index)),
                 steepness=b.steepness)
        for b in sorted(blocks, key=lambda b: b.index, reverse=True)
    ]
    return SetupCtx(symbol=symbol, ts=int(ts), bar_index=int(bar_index), close=close,
                    price=price, atr_now=float(atr_series.iloc[-1]),
                    equilibrium=equilibrium, blocks=block_ctxs, gaps=gap_ctxs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default=BARS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit-symbols", type=int, default=None,
                    help="scan only the first N symbols - for a smoke run")
    args = ap.parse_args()

    with open(args.bars, "rb") as fh:
        raw = pickle.load(fh)
    tasks = list(raw.items())
    if args.limit_symbols:
        tasks = tasks[: args.limit_symbols]
    print(f"{len(tasks)} symbols, {args.workers} workers", flush=True)

    started = time.time()
    result, total = {}, 0
    with Pool(args.workers) as pool:
        for k, (symbol, ctxs) in enumerate(pool.imap_unordered(record_symbol, tasks), 1):
            if ctxs:
                result[symbol] = ctxs
                total += len(ctxs)
            if k % 25 == 0 or k == len(tasks):
                el = time.time() - started
                eta = el / k * (len(tasks) - k)
                print(f"  {k}/{len(tasks)} symbols - {total} setups - "
                      f"{el / 60:.1f}m elapsed - ETA {eta / 60:.1f}m", flush=True)

    with open(args.out, "wb") as fh:
        pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
    span = sorted(c.ts for v in result.values() for c in v)
    print(f"\n{total} setups across {len(result)} symbols -> {args.out}")
    if span:
        print(f"span {pd.Timestamp(span[0], unit='ms'):%Y-%m-%d} .. "
              f"{pd.Timestamp(span[-1], unit='ms'):%Y-%m-%d}")
    print(f"total {(time.time() - started) / 60:.1f}m")


if __name__ == "__main__":
    main()
