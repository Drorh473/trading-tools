"""Signal generation for Strategy 2.1's 15m instances, on a 15m spine.

generate_v2 iterates 1H bars and derives 4H and 1D from them. This does the same
one level down: the spine is 15m and 1H is derived from it, so both timeframes
come from ONE fetch and align exactly. Mixing the 15m fetch with the separate 1H
cache would pair two datasets with different spans and different symbols.

The base-versus-reference rule is carried over unchanged, because it is where
the lookahead lives:

    base       partial candle INCLUDES the entry bar - its candle is where the
               fill is tested, and a fill may use the bar it happens on
    reference  partial candle stops at the PREVIOUS bar - everything it
               contributes is a decision, and the order rests before the entry
               bar opens

Getting that backwards is what made v2 look 3,614x better than it was.

    python -m backtest.generate_15m --workers 10 --min-days 365
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from backtest import checkpoint
from backtest.score import confirmed_pivots, simulate
from notifier.strategies import ema_trend_v2 as v2
from notifier.strategies.ema_trend_v2 import EmaTrendV2, hold_run, structure_metrics
from notifier.strategies.indicators import atr

# Same discipline as generate_v2: the widest population, every threshold OFF and
# recorded, so all of them are swept over one population afterwards.
v2.EMA9_HOLD_BARS = 0
v2.MIN_STOP_PCT = 0.0
v2.MIN_NET_REWARD_RISK = 0.0
v2.MIN_PIVOT_SPAN_BARS = 0
v2.MIN_SWING_DRIFT_ATR = 0.0
v2.MAX_EMA9_CROSSINGS = 999
# The structure gate went ON 2026-08-21. It has to come off HERE, or generation
# pre-filters the population and every sweep over the result compares a subset
# against a whole - including the sweep that turned it on. structure_metrics is
# recorded per setup either way, so "gate on" stays a filter applied afterwards.
v2.REQUIRE_STRUCTURE_TREND = False

BARS_15M = "data/bars_15m.pkl"
# Paired 15m/1H dropped with the paired instances themselves - it cannot fire
# live, so measuring it would cost half this run for nothing.
INSTANCES: tuple[tuple[str, str | None], ...] = (("15m", None),)
RULE = {"1H": "1h"}
WARMUP_HIGHER = 205  # SMA200 on the reference, plus room
START = 1000  # 15m bars before the first evaluation: 205 closed 1H bars needs 820
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "base_vol": "sum"}
CHECKPOINT_EVERY = 5


def _forming_row(spine: pd.DataFrame, lo: int, hi: int) -> dict:
    w = spine.iloc[lo : hi + 1]
    return {
        "ts": w["ts"].iloc[0],
        "open": float(w["open"].iloc[0]),
        "high": float(w["high"].max()),
        "low": float(w["low"].min()),
        "close": float(w["close"].iloc[-1]),
        "base_vol": float(w["base_vol"].sum()),
    }


def scan_symbol(args):
    symbol, spine = args
    if spine is None or len(spine) < START + 500:
        return symbol, []
    spine = spine.copy()
    spine["ts"] = pd.to_datetime(spine["ts"], unit="ms")
    idx = spine.set_index("ts")
    closed = {tf: idx.resample(rule).agg(AGG).dropna().reset_index() for tf, rule in RULE.items()}
    bucket = {tf: spine["ts"].dt.floor(rule) for tf, rule in RULE.items()}
    first_in = {tf: spine.groupby(bucket[tf]).cumcount() for tf in RULE}
    n_closed = {tf: closed[tf]["ts"].searchsorted(bucket[tf].values, side="left") for tf in RULE}

    instances = [(EmaTrendV2(b, r), b, r) for b, r in INSTANCES]
    pivots = confirmed_pivots(spine, atr(spine, 14))
    out, seen = [], set()

    def frame(tf: str, i: int, upto: int):
        k = int(n_closed[tf][i])
        if k < WARMUP_HIGHER:
            return None
        lo = i - int(first_in[tf].iloc[i])
        if upto < lo:
            return None
        return pd.concat(
            [closed[tf].iloc[:k], pd.DataFrame([_forming_row(spine, lo, upto)])], ignore_index=True
        )

    for i in range(START, len(spine) - 5):
        base_view = spine.iloc[: i + 1]
        for pos, (inst, base, ref) in enumerate(instances):
            views = {base: base_view}
            if ref is not None:
                r = frame(ref, i, i - 1)  # reference stops at the PREVIOUS bar
                if r is None:
                    continue
                views[ref] = r
            try:
                sig = inst.evaluate(symbol, views)
            except Exception:
                continue
            if sig is None:
                continue
            key = sig.dedupe_key or (symbol, sig.strategy_tag, sig.direction, sig.entry_price)
            if key in seen:
                continue
            seen.add(key)
            trend = "up" if sig.direction == "long" else "down"
            ref_view = views[ref] if ref else base_view
            # The fill is the NEXT candle's open, not sig.entry_price - see
            # generate_v2 and score.simulate's `fill_at`. entry_price is the
            # EMA9 that selected the setup and the market has already left it.
            if i + 1 >= len(spine):
                continue
            fill = float(spine["open"].iloc[i + 1])
            scored = simulate(spine, i, sig, runner="choch", pivots=pivots, fill_at=fill)
            if scored.result == "invalid":
                continue
            # The ATR the STOP is measured against - same bar and same period
            # _trigger buffers with. MIN_STOP_ATR is swept over this, and it is
            # the one number the 1H/4H/1D population could not supply for 15m.
            atr_prev = atr(base_view, v2.ATR_PERIOD).iloc[-2]
            atr_prev = float(atr_prev) if pd.notna(atr_prev) else float("nan")
            # THE SAME LAYOUT generate_v2 writes, so backtest.sweep_v2 reads
            # this population directly instead of needing a second adapter.
            # Two row shapes for one strategy is how two scorers happened.
            out.append((
                spine["ts"].iloc[i], i, float(spine["close"].iloc[i]), pos, sig,
                hold_run(base_view.iloc[:-1], trend),
                hold_run(ref_view.iloc[:-1], trend),
                (scored.result, scored.bars),
                {"base": structure_metrics(base_view.iloc[:-1]),
                 "ref": structure_metrics(ref_view.iloc[:-1])},
                scored.r_gross, scored.r_net, fill, atr_prev,
            ))
    return symbol, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--min-days", type=int, default=365)
    ap.add_argument("--out", default="data/signals_15m.pkl")
    ap.add_argument("--checkpoint", default="data/signals_15m_partial.pkl")
    args = ap.parse_args()

    bars = pickle.load(open(BARS_15M, "rb"))
    keep = {
        s: f for s, f in bars.items()
        if (pd.to_datetime(f.ts.max(), unit="ms") - pd.to_datetime(f.ts.min(), unit="ms")).days >= args.min_days
    }
    print(f"{len(keep)} of {len(bars)} symbols have >= {args.min_days} days of 15m", flush=True)

    done = {}
    if os.path.exists(args.checkpoint):
        try:
            # The checkpoint stores INSTANCES precisely so a resume cannot mix
            # populations, but this used to read [1] and drop the key on the
            # floor. Dropping the paired 15m/1H instance would then have
            # resumed 50 symbols generated under the OLD instance list into a
            # run reporting the new one.
            saved_instances, saved_done = pickle.load(open(args.checkpoint, "rb"))
            if tuple(saved_instances) != INSTANCES:
                print(
                    f"checkpoint was built for {saved_instances}, this run is "
                    f"{INSTANCES} - regenerating from scratch",
                    flush=True,
                )
            else:
                done = saved_done
                print(f"resuming: {len(done)} symbols", flush=True)
        except Exception:
            done = {}

    todo = [(s, f) for s, f in keep.items() if s not in done]
    t0, n = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for sym, res in ex.map(scan_symbol, todo, chunksize=1):
            done[sym] = res
            n += 1
            el = (time.time() - t0) / 60
            print(f"[{n}/{len(todo)}] {sym:16s} {len(res):5d} signals  {el:5.1f}m elapsed, "
                  f"~{el/max(n,1)*(len(todo)-n):5.1f}m left", flush=True)
            if n % CHECKPOINT_EVERY == 0:
                checkpoint.write(args.checkpoint, (INSTANCES, done))

    with open(args.out, "wb") as fh:
        pickle.dump((INSTANCES, done), fh)
    total = sum(len(v) for v in done.values())
    print(f"\nDONE: {total} signals across {len(done)} symbols in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
