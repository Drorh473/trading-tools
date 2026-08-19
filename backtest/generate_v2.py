"""Signal generation for Strategy 2 v2, across every instance the cache can reach.

Separate from portfolio.generate on purpose. That generator hands a strategy a
SLICE of a precomputed higher-timeframe frame, whose current row already holds
the completed OHLC of a candle that has not finished yet at the 1H bar being
evaluated. v1 never noticed because it reads closed bars only. v2 reads the
FORMING candle by design, so it would have inherited hours of lookahead
silently - the exact shape of defect ֲ§42 and ֲ§45 are both about.

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

from backtest.score import confirmed_pivots, simulate
from notifier.strategies import ema_trend_v2 as v2
from notifier.strategies.ema_trend_v2 import EmaTrendV2, hold_run, structure_metrics
from notifier.strategies.indicators import atr

# v2 states its first target and sets remainder_target=None on purpose, which
# live means the remainder is TRAILED on structure by poll_trailing_stops
# rather than aimed at a second price. score.simulate(runner="choch") is that
# exit; nothing here invents a fixed multiple any more.

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
# generate-once-replay-cheaply structure ֲ§42 established.
#
# MAX_STOP_PCT stays on: it is a crash-regime guard, not a tuning parameter.
v2.EMA9_HOLD_BARS = 0
v2.MIN_STOP_PCT = 0.0
v2.MIN_NET_REWARD_RISK = 0.0
# The three scale conditions on the trend read, likewise recorded and swept
# rather than chosen. See ema_trend_v2.MIN_PIVOT_SPAN_BARS.
v2.MIN_PIVOT_SPAN_BARS = 0
v2.MIN_SWING_DRIFT_ATR = 0.0
v2.MAX_EMA9_CROSSINGS = 999


CACHE = os.getenv(
    "BACKTEST_SIGNALS",
    r"C:/Users/dror/AppData/Local/Temp/claude/C--Users-dror-study-projects-trading-tools"
    r"/09d1f0c4-21e2-409d-8d79-c9fb73a4f6bc/scratchpad/signals.pkl",
)
CHECKPOINT_EVERY = int(os.getenv("BACKTEST_CHECKPOINT_EVERY", "5"))

# Only what the cache can actually reach. It holds 1H bars; 4H and 1D resample
# from them cleanly, 15m cannot be derived at any price. The 15m instance is
# therefore absent here - a limit of THIS CACHE, not of the exchange - and is
# measured by backtest/generate_15m.py against a real 15m fetch.
#
# This used to read "can only ever be measured on the ~22 days Bitget serves".
# That was false. ~22 days is what /api/v2/mix/market/candles returns for a
# plain limit; history-candles is anchored by endTime and pages back to the
# symbol's listing date - 249,112 15m bars on BTCUSDT reaching 2019-07-10,
# measured 2026-08-17.
#
# The two paired instances ("1H"/"4H" and "4H"/"1D") were dropped from this
# list when they were dropped from the strategy. Keeping them here would have
# cost two thirds of the generation time to re-measure a variant that cannot
# fire live. Their last measurement stands: -0.171R at t -5.79 over 5,094
# trades for 4H/1H. This list now mirrors ema_trend_v2.INSTANCES exactly,
# minus 15m; test_generate_v2_measures_what_ships pins that.
MEASURABLE: tuple[tuple[str, str | None], ...] = (
    ("1H", None),
    ("4H", None),
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


def _pivots_on_1h(tf: str, frame: pd.DataFrame, n_closed_tf, n_h1: int) -> list:
    """This timeframe's confirmed swings, re-indexed onto the 1H spine.

    The runner is trailed on the structure of the timeframe the setup was taken
    on - a 4H trade trails 4H swings - but this generator walks trades forward
    on 1H bars, because 1H is the finest price it has and a coarser walk would
    miss stops that a 4H candle only shows as a wick.

    So each pivot is dated by the 1H bar on which its own timeframe's candle
    CLOSED, which is the first moment the level could have been acted on.
    Dating it by the 1H bar it printed on would trail behind a level nobody
    could see yet - the same lookahead that confirmed_pivots exists to avoid,
    reintroduced one level up.
    """
    a = atr(frame)
    piv = confirmed_pivots(frame, a)
    if tf == "1H":
        return piv
    out = []
    for c, price, is_high in piv:
        j = int(np.searchsorted(n_closed_tf, c + 1, side="left"))
        if j < n_h1:
            out.append((j, price, is_high))
    out.sort()
    return out


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
    # One set per timeframe, built once: confirmed_pivots walks the whole frame
    # and the trades on a symbol number in the hundreds.
    pivots = {"1H": _pivots_on_1h("1H", h1, None, len(h1))}
    for tf in RULE:
        pivots[tf] = _pivots_on_1h(tf, closed[tf], n_closed[tf], len(h1))
    out = []
    # The scanner dedupes live; nothing did here, so one setup was recorded once
    # per bar. WLDUSDT produced the same 4H long on eight consecutive bars. That
    # does not merely inflate counts - eight copies of one setup are not eight
    # observations, so every standard error and t-statistic computed over this
    # population was overstated.
    seen: set = set()
    def frame(tf: str, i: int, upto: int):
        """Closed bars plus the partial candle built from 1H bars up to `upto`.

        `upto` is the whole correctness question. A timeframe acting as the
        BASE gets the entry bar included, because its partial candle is where
        the fill is tested and a fill may use the bar it happens on. A
        timeframe acting as the REFERENCE gets bars only up to the PREVIOUS
        close, because everything it contributes is a decision, and the order
        is placed before the entry bar opens.
        """
        k = int(n_closed[tf][i])
        if k < WARMUP_HIGHER:
            return None
        lo = i - int(first_in[tf].iloc[i])
        if upto < lo:
            return None  # this candle has no bars yet at that point in time
        return pd.concat(
            [closed[tf].iloc[:k], pd.DataFrame([_forming_row(h1, lo, upto)])],
            ignore_index=True,
        )

    for i in range(1200, len(h1)):
        base_views: dict[str, pd.DataFrame] = {"1H": h1.iloc[: i + 1]}
        ref_views: dict[str, pd.DataFrame] = {"1H": h1.iloc[:i]}
        for tf in RULE:
            b = frame(tf, i, i)
            if b is not None:
                base_views[tf] = b
            r = frame(tf, i, i - 1)
            if r is not None:
                ref_views[tf] = r

        for pos, (inst, base, ref) in enumerate(instances):
            if base not in base_views or (ref is not None and ref not in ref_views):
                continue
            views = {base: base_views[base]}
            if ref is not None:
                views[ref] = ref_views[ref]
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
                key = sig.dedupe_key or (symbol, sig.strategy_tag, sig.direction, sig.entry_price)
                if key in seen:
                    continue
                seen.add(key)
                trend = "up" if sig.direction == "long" else "down"
                ref_view = views[ref] if ref else views[base]
                # THE scorer - the same call generate_15m makes. This used to be
                # a local _walk that gave the runner a fixed second target and
                # kept the ORIGINAL stop for the whole trade, so it credited
                # "both targets" to runners the live bot would already have
                # closed at breakeven. v2 ships remainder_target=None, which
                # live means poll_trailing_stops trails it on structure.
                # ENTRY_MODE="next_open" enters at MARKET on the candle after
                # the rejection, so the fill is that candle's open - not
                # sig.entry_price, which is the EMA9 that selected the setup and
                # sits on the other side of it by construction. Scoring at the
                # EMA9 measured a fill nobody gets: 1.89x understated risk over
                # 591 setups, worse on 100% of them.
                if i + 1 >= len(h1):
                    continue
                fill = float(h1["open"].iloc[i + 1])
                scored = simulate(h1, i, sig, runner="choch", pivots=pivots[base],
                                  fill_at=fill)
                if scored.result == "invalid":
                    continue
                # The ATR the STOP is measured against - the same bar and the
                # same period _trigger uses for its 0.10 x ATR buffer. Recorded
                # because MIN_STOP_PCT is scale-free in percent and therefore
                # cannot see volatility: LABUSDT's stop was 4.3% of price and
                # 1.13 ATR at once. Sweeping a floor in ATR needs the
                # denominator, and nothing was keeping it.
                atr_prev = atr(views[base], v2.ATR_PERIOD).iloc[-2]
                atr_prev = float(atr_prev) if pd.notna(atr_prev) else float("nan")
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
                        (scored.result, scored.bars),
                        # The scale measures for BOTH timeframes, so every
                        # candidate threshold is a filter over one population.
                        {"base": structure_metrics(views[base].iloc[:-1]),
                         "ref": structure_metrics(ref_view.iloc[:-1])},
                        # At zero slippage. The sweep charges it afterwards,
                        # since only the market-stop leg slips and how much is
                        # a parameter worth varying.
                        scored.r_gross,
                        scored.r_net,
                        # The price the trade is actually OPENED at, and the ATR
                        # its stop distance should be judged against. sig.entry_price
                        # is the EMA9 that SELECTED the setup and sits on the far
                        # side of the fill by construction, so every quantity
                        # derived from it - stop fraction, net R:R - describes a
                        # trade nobody gets. simulate() has scored at `fill`
                        # since the one-scorer change; these two make the SWEEP
                        # able to filter on the same basis.
                        fill,
                        atr_prev,
                    )
                )
    return symbol, out


# _walk lived here: a second scorer, deleted rather than fixed. It resolved a
# trade by comparing the first bar to touch the stop, target 1 and target 2 -
# with the ORIGINAL stop for the whole trade, and a fixed second target the
# shipping config does not have. backtest/score.simulate is the one scorer now,
# and tests/test_score.py pins the case the two disagreed on.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=os.path.join("data", "signals_v2.pkl"))
    ap.add_argument("--checkpoint", default=os.path.join("data", "signals_v2_partial.pkl"))
    ap.add_argument("--symbols", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    scanner, measurable = scan_symbol, MEASURABLE

    key, bars, _ = pickle.load(open(CACHE, "rb"))
    syms = list(bars)
    if args.symbols:
        syms = syms[: args.symbols]

    done: dict = {}
    if os.path.exists(args.checkpoint):
        try:
            saved_key, done = pickle.load(open(args.checkpoint, "rb"))
            if saved_key != ("v2", len(measurable)):
                done = {}
            else:
                print(f"resuming: {len(done)} symbols already generated")
        except Exception:
            done = {}

    todo = [(s, bars[s]) for s in syms if s not in done]
    print(f"{len(todo)} symbols to scan, {len(measurable)} instances, {args.workers} workers")
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
                    pickle.dump((("v2", len(measurable)), done), fh)
                os.replace(tmp, args.checkpoint)

    with open(args.out, "wb") as fh:
        pickle.dump((("v2", len(measurable)), measurable, done), fh)
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
