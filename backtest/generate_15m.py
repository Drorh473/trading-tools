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

BARS_15M = "data/bars_15m.pkl"
INSTANCES: tuple[tuple[str, str | None], ...] = (("15m", None), ("15m", "1H"))
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
            scored = simulate(spine, i, sig, runner="choch", pivots=pivots)
            out.append((
                spine["ts"].iloc[i], i, pos, sig,
                hold_run(base_view.iloc[:-1], trend),
                hold_run(ref_view.iloc[:-1], trend),
                {"base": structure_metrics(base_view.iloc[:-1]),
                 "ref": structure_metrics(ref_view.iloc[:-1])},
                scored.result, scored.r_gross, scored.r_net,
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
            done = pickle.load(open(args.checkpoint, "rb"))[1]
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
                tmp = args.checkpoint + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump((INSTANCES, done), fh)
                os.replace(tmp, args.checkpoint)

    with open(args.out, "wb") as fh:
        pickle.dump((INSTANCES, done), fh)
    total = sum(len(v) for v in done.values())
    print(f"\nDONE: {total} signals across {len(done)} symbols in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
