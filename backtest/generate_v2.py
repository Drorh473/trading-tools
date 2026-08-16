"""Signal generation for Strategy 2 v2, across every instance the cache can reach.

Separate from portfolio.generate on purpose. That generator hands a strategy a
SLICE of a precomputed higher-timeframe frame, whose current row already holds
the completed OHLC of a candle that has not finished yet at the 1H bar being
evaluated. v1 never noticed because it reads closed bars only. v2 reads the
FORMING candle by design, so it would have inherited hours of lookahead
silently - the exact shape of defect §42 and §45 are both about.

Here the partial candle is built from the 1H bars that have actually printed:

    closed higher-TF bars  ...  +  one synthetic row aggregated from the 1H
                                  bars since that timeframe last closed

so the strategy sees precisely what it would see live, and its own
`closed = forming.iloc[:-1]` still lands on genuinely closed data.

Checkpointed every CHECKPOINT_EVERY symbols. Interrupting it costs minutes, not
the run - the first v1 generation ran fifteen hours and wrote nothing until the
end.

    python -m backtest.generate_v2 --workers 10 --out data/signals_v2.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from notifier.strategies import ema_trend_v2 as v2
from notifier.strategies.ema_trend import EmaTrendFollowing
from notifier.strategies.ema_trend_v2 import EmaTrendV2, hold_run

# What the scanner does for a strategy that sets no partial_fraction: half at
# the signal's own reward:risk, the rest at REMAINDER_TARGET_RATIO of its risk.
# v1 is such a strategy; v2 states both prices itself.
SCANNER_REMAINDER_RATIO = 3.0

# GENERATE THE WIDEST POPULATION, FILTER AFTERWARDS.
#
# Three of v2's constants are thresholds nobody has evidence for -
# EMA9_HOLD_BARS ("10 is just a number I threw", and it came from v1 rather
# than from any measurement), MIN_STOP_PCT and MIN_NET_REWARD_RISK. Generating
# at a chosen value discards exactly the setups needed to tell whether that
# value is right, and re-generating per candidate costs two hours each.
#
# So generation runs with all three OFF and records what each setup actually
# had: how many bars its EMA9 held, how wide its stop was, what it netted.
# Every candidate threshold is then a filter over one population - the
# generate-once-replay-cheaply structure §42 established.
#
# MAX_STOP_PCT stays on: it is a crash-regime guard, not a tuning parameter.
v2.EMA9_HOLD_BARS = 0
v2.MIN_STOP_PCT = 0.0
v2.MIN_NET_REWARD_RISK = 0.0

WALK_BARS = 200  # how far forward an unresolved trade is followed

CACHE = os.getenv(
    "BACKTEST_SIGNALS",
    r"C:/Users/dror/AppData/Local/Temp/claude/C--Users-dror-study-projects-trading-tools"
    r"/09d1f0c4-21e2-409d-8d79-c9fb73a4f6bc/scratchpad/signals.pkl",
)
CHECKPOINT_EVERY = 5

# Only what the cache can actually reach. It holds 1H bars; 4H and 1D resample
# from them cleanly, 15m cannot be derived at any price. The two 15m instances
# are therefore absent here and can only ever be measured on the ~22 days
# Bitget serves - which is a separate, much weaker exercise.
MEASURABLE: tuple[tuple[str, str | None], ...] = (
    ("1H", None),
    ("4H", None),
    ("1D", None),
    ("1H", "4H"),
    ("4H", "1D"),
)

# The same three v1 instances the live bot runs, minus 15m/1H which no cache
# can reach. Generated through THIS builder rather than portfolio.generate, so
# the head-to-head is fees, fills, exits and bars all held identical and only
# the rules differing - the one comparison that says whether replacing v1 with
# v2 is an improvement rather than a change.
#
# v1 is generated as it stands AFTER this session's two fixes (its own
# reward:risk of 2.0, and the maker fee), because that is what is running now
# and therefore what v2 has to beat.
#
# v1 reads CLOSED bars only. It is handed closed frames with no forming row -
# passing it the partial candle v2 wants would put an unfinished bar in
# .iloc[-1], which is the bug its own prior-bar idiom exists to prevent.
V1_MEASURABLE: tuple[tuple[str, str | None], ...] = (
    ("1H", "4H"),
    ("4H", "1D"),
    ("1D", None),
)

RULE = {"4H": "4h", "1D": "1D"}
WARMUP_HIGHER = 205  # SMA200 plus room
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "base_vol": "sum"}


def _closed_frames(h1: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Fully closed higher-timeframe bars, labelled by their OPEN time."""
    out = {}
    idx = h1.set_index("ts")
    for tf, rule in RULE.items():
        f = idx.resample(rule).agg(AGG).dropna()
        out[tf] = f.reset_index()
    return out


def _forming_row(h1: pd.DataFrame, lo: int, hi: int) -> dict:
    """One synthetic candle from h1[lo:hi+1] - the bars that have printed since
    this timeframe last closed."""
    w = h1.iloc[lo : hi + 1]
    return {
        "ts": w["ts"].iloc[0],
        "open": float(w["open"].iloc[0]),
        "high": float(w["high"].max()),
        "low": float(w["low"].min()),
        "close": float(w["close"].iloc[-1]),
        "base_vol": float(w["base_vol"].sum()),
    }


def scan_symbol_v1(args):
    """v1 over the same bars, closed frames only."""
    symbol, h1 = args
    if h1 is None or len(h1) < 1500:
        return symbol, []
    h1 = h1.copy()
    h1["ts"] = pd.to_datetime(h1["ts"])
    closed = _closed_frames(h1)
    bucket = {tf: h1["ts"].dt.floor(rule) for tf, rule in RULE.items()}
    first_in = {tf: h1.groupby(bucket[tf]).cumcount() for tf in RULE}
    n_closed = {tf: closed[tf]["ts"].searchsorted(bucket[tf].values, side="left") for tf in RULE}

    instances = [(EmaTrendFollowing(b, r), b, r) for b, r in V1_MEASURABLE]
    highs, lows = h1["high"].to_numpy(), h1["low"].to_numpy()
    out = []
    for i in range(1200, len(h1)):
        views: dict[str, pd.DataFrame] = {"1H": h1.iloc[: i + 1]}
        for tf in RULE:
            k = int(n_closed[tf][i])
            if k >= WARMUP_HIGHER:
                views[tf] = closed[tf].iloc[:k]

        for pos, (inst, base, ref) in enumerate(instances):
            if base not in views or (ref is not None and ref not in views):
                continue
            if base != "1H" and int(first_in[base].iloc[i]) != 0:
                continue
            try:
                sig = inst.evaluate(symbol, {tf: views[tf] for tf in ([base] + ([ref] if ref else []))})
            except Exception:
                continue
            if sig is not None:
                # v1 has no hold concept of its own to record; its hold is
                # implicit in its 10-candle filter, so both columns are marked
                # -1 and the sweep's hold filter is a no-op for these rows.
                out.append((h1["ts"].iloc[i], i, float(h1["close"].iloc[i]), pos, sig, -1, -1, _walk(highs, lows, i, sig)))
    return symbol, out


def scan_symbol(args):
    symbol, h1 = args
    if h1 is None or len(h1) < 1500:
        return symbol, []
    h1 = h1.copy()
    h1["ts"] = pd.to_datetime(h1["ts"])
    closed = _closed_frames(h1)
    bucket = {tf: h1["ts"].dt.floor(rule) for tf, rule in RULE.items()}
    # index of the first 1H bar in each bar's own higher-TF bucket
    first_in = {tf: h1.groupby(bucket[tf]).cumcount() for tf in RULE}
    # how many higher-TF bars have fully closed before each 1H bar
    n_closed = {tf: closed[tf]["ts"].searchsorted(bucket[tf].values, side="left") for tf in RULE}

    instances = [(EmaTrendV2(b, r), b, r) for b, r in MEASURABLE]
    highs, lows = h1["high"].to_numpy(), h1["low"].to_numpy()
    out = []
    for i in range(1200, len(h1)):
        views: dict[str, pd.DataFrame] = {"1H": h1.iloc[: i + 1]}
        for tf in RULE:
            k = int(n_closed[tf][i])
            if k < WARMUP_HIGHER:
                continue
            lo = i - int(first_in[tf].iloc[i])
            frame = pd.concat(
                [closed[tf].iloc[:k], pd.DataFrame([_forming_row(h1, lo, i)])],
                ignore_index=True,
            )
            views[tf] = frame

        for pos, (inst, base, ref) in enumerate(instances):
            if base not in views or (ref is not None and ref not in views):
                continue
            # Evaluate an instance only when its OWN base bar has just closed.
            # The live scanner aligns to candle closes; a 4H instance re-reading
            # identical data on each of the four 1H bars inside its candle is
            # both wrong and four times the work.
            if base != "1H" and int(first_in[base].iloc[i]) != 0:
                continue
            try:
                sig = inst.evaluate(symbol, {tf: views[tf] for tf in ([base] + ([ref] if ref else []))})
            except Exception:
                continue
            if sig is not None:
                trend = "up" if sig.direction == "long" else "down"
                ref_view = views[ref] if ref else views[base]
                out.append(
                    (
                        h1["ts"].iloc[i],
                        i,
                        float(h1["close"].iloc[i]),
                        pos,
                        sig,
                        # What this setup ACTUALLY had, so the thresholds can be
                        # swept instead of assumed.
                        hold_run(views[base].iloc[:-1], trend),
                        hold_run(ref_view.iloc[:-1], trend),
                        _walk(highs, lows, i, sig),
                    )
                )
    return symbol, out


def _walk(highs, lows, i: int, sig) -> tuple[str, int]:
    """Which of stop / first target / runner target this trade reached first.

    Trade-level only - no portfolio, no competition for capital, no fees. It
    exists to compare RULE VARIANTS against one population, not to state what
    the strategy earns. A bar that touches the stop resolves as a stop even if
    it also touched a target, which is the conservative reading of an
    ambiguous candle.
    """
    entry, stop = sig.entry_price, sig.stop_loss
    risk = abs(entry - stop)
    long = sig.direction == "long"
    sign = 1 if long else -1
    t1 = entry + sig.reward_risk_ratio * risk * sign
    # v2 states its runner price; v1 sets no partial_fraction and so takes the
    # scanner's ratio target. Modelling v1's runner as v2's would compare the
    # rules against an exit v1 never gets.
    t2 = sig.remainder_target if sig.remainder_target is not None else entry + SCANNER_REMAINDER_RATIO * risk * sign
    hi, lo = highs[i + 1 : i + 1 + WALK_BARS], lows[i + 1 : i + 1 + WALK_BARS]
    if len(hi) == 0:
        return "unresolved", 0
    if long:
        stopped = np.flatnonzero(lo <= stop)
        first = np.flatnonzero(hi >= t1)
        second = np.flatnonzero(hi >= t2)
    else:
        stopped = np.flatnonzero(hi >= stop)
        first = np.flatnonzero(lo <= t1)
        second = np.flatnonzero(lo <= t2)
    s = stopped[0] if stopped.size else len(hi)
    f = first[0] if first.size else len(hi)
    g = second[0] if second.size else len(hi)
    if g < s:
        return "both targets", int(g) + 1
    if f < s:
        return ("target1 then stop", int(s) + 1) if s < len(hi) else ("target1, runner open", int(f) + 1)
    if s < len(hi):
        return "stop", int(s) + 1
    return "unresolved", len(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join("data", "signals_v2.pkl"))
    ap.add_argument("--checkpoint", default=os.path.join("data", "signals_v2_partial.pkl"))
    ap.add_argument("--symbols", type=int, default=0, help="0 = all")
    ap.add_argument("--variant", choices=("v2", "v1"), default="v2")
    args = ap.parse_args()

    scanner = scan_symbol if args.variant == "v2" else scan_symbol_v1
    measurable = MEASURABLE if args.variant == "v2" else V1_MEASURABLE

    key, bars, _ = pickle.load(open(CACHE, "rb"))
    syms = list(bars)
    if args.symbols:
        syms = syms[: args.symbols]

    done: dict = {}
    if os.path.exists(args.checkpoint):
        try:
            saved_key, done = pickle.load(open(args.checkpoint, "rb"))
            if saved_key != (args.variant, len(measurable)):
                done = {}
            else:
                print(f"resuming: {len(done)} symbols already generated")
        except Exception:
            done = {}

    todo = [(s, bars[s]) for s in syms if s not in done]
    print(f"{len(todo)} symbols to scan, {args.variant}, {len(measurable)} instances, {args.workers} workers")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    t0 = time.time()
    n = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for sym, res in ex.map(scanner, todo, chunksize=1):
            done[sym] = res
            n += 1
            el = time.time() - t0
            rate = el / max(n, 1)
            print(
                f"[{n}/{len(todo)}] {sym:14s} {len(res):5d} signals   "
                f"{el/60:5.1f}m elapsed, ~{rate*(len(todo)-n)/60:5.1f}m left",
                flush=True,
            )
            if n % CHECKPOINT_EVERY == 0:
                tmp = args.checkpoint + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(((args.variant, len(measurable)), done), fh)
                os.replace(tmp, args.checkpoint)

    with open(args.out, "wb") as fh:
        pickle.dump(((args.variant, len(measurable)), measurable, done), fh)
    total = sum(len(v) for v in done.values())
    print(f"\nDONE: {total} signals across {len(done)} symbols in {(time.time()-t0)/60:.1f} min -> {args.out}")
    import collections

    c = collections.Counter()
    for v in done.values():
        for row in v:
            c[measurable[row[3]]] += 1
    for inst, cnt in c.most_common():
        print(f"   {str(inst):20s} {cnt}")


if __name__ == "__main__":
    main()
