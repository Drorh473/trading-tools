"""A real portfolio backtest: time on the OUTSIDE, symbols on the inside.

WHY THIS EXISTS
  run_backtest.py loops symbols on the outside and time on the inside, so it
  runs one symbol's entire year to completion before starting the next. At most
  one position is ever open. Every constraint it advertises as modelled - the
  6% aggregate risk cap, the 2-slot swing pool, one-position-per-symbol, and
  the margin budget that free equity implies - therefore never binds. Measured
  on an 8-symbol smoke run: risk_cap 0, already_in_symbol 0, swing_slots 0
  across 41 trades. The -30.6% headline it produced is a concatenation of 98
  sequential single-symbol years, not a portfolio, and its max-drawdown figure
  and decline counters are artifacts.

THE SPLIT THAT MAKES IT CHEAP
  Strategy instances hold only __init__ configuration - no mutable state - so
  evaluate() is a pure function of bars. Signal GENERATION (expensive, ~6h,
  embarrassingly parallel across symbols) is therefore separable from portfolio
  REPLAY (cheap, seconds, inherently sequential because equity compounds).
  Generate once, cache, then replay every rule variant against the identical
  signal set. That is what makes an honest baseline/limit/market A/B possible
  at all; under the old driver each arm cost its own full run.

  It also removes a bias. The old driver skipped evaluate() while a position
  was open on that symbol, so the signal population depended on account
  history - a different fill rule silently changed which signals existed. Here
  the signal set is canonical and only the portfolio rules vary.

WHAT IS DELIBERATELY UNCHANGED
  Account, try_open and step_position are imported from backtest.py, so fees,
  the $5 per-leg floor, split fills, the two-tier exit and the stop-first
  convention on a bar spanning both are identical. The only thing that changes
  is WHEN each symbol is looked at.

KNOWN DIVERGENCE FROM LIVE, inherited and not introduced here
  Instances are called directly, so the scanner's ARMING layer is bypassed.
  Strategy 3's day instance is recorded in the handoff as arming 0 times in
  practice; here it will signal freely. Backtest signal counts for armed
  strategies are therefore an upper bound on what the live bot would produce.

  _replay threads confirmed pivots into try_open only for instances whose OWN
  base timeframe is "1H" (this is the only spine it steps bar-by-bar). Any
  remainder_target=None position on a 4H- or 1D-base instance (e.g.
  RsiFibReversal("4H")/("1D")) still trails on nothing and sits pinned at
  breakeven - understated exactly as every instance was before this was added.
"""
import logging
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from backtest import checkpoint
from backtest import engine as bt
from backtest.score import confirmed_pivots
from notifier.strategies.base import TIMEFRAME_SECONDS as TF_SECONDS
from notifier.strategies.indicators import atr
from notifier.strategies.order_block import OrderBlockStrategy
from notifier.strategies.rsi_fib_reversal import RsiFibReversal
from notifier.strategies.volume_run import VolumeRun
from notifier.watchlist import WATCHLIST

logger = logging.getLogger(__name__)

SIGNALS = os.getenv("BACKTEST_SIGNALS", os.path.join("data", "signals.pkl"))
# Partial progress, written as symbols complete and removed once the finished
# signal set lands. Separate from SIGNALS so an interrupted run can never be
# mistaken for a complete one.
CHECKPOINT = os.getenv("BACKTEST_CHECKPOINT", os.path.join("data", "signals_partial.pkl"))

# Excluded, with the reason:
#   Strategy 2 1H/15m  - needs 15m, which this 1H cache does not hold
#   Strategy 3 1D/5m   - needs 5m, same
#   Strategy 4 15m     - same
# Approximating them would be inventing the result. Two of the nine LIVE
# instances are therefore still unmeasured after this run.
#
# NOT because the exchange is short of history. This list used to read "Bitget
# serves 22 days of 15m, ~2 days of 5m", and on that basis three live
# instances were written off as permanently unmeasurable. That figure was one
# get_candles call's result recorded as a property of the exchange: it is what
# /api/v2/mix/market/candles returns for a plain limit. history-candles is
# anchored by endTime and pages back to the symbol's LISTING date - measured
# 2026-08-17 on BTCUSDT, 249,112 15m bars and 747,336 5m bars, both reaching
# 2019-07-10 (2,595 days), contiguous and price-correct. What these instances
# need is a 15m/5m fetch, not a different exchange.
# NOTE: Strategy 2's three entries were removed when it was retired, so every
# POSITION in this list shifted. A signals cache generated before 2026-08-16
# stores instance positions and cannot be replayed against this list.
INSTANCES = [
    (RsiFibReversal("1H"), ["1H"], 24 * 4),
    (RsiFibReversal("4H"), ["4H"], 6 * 4),
    (RsiFibReversal("1D"), ["1D"], 30),
    (VolumeRun("1D", "1H", time_exit_days=3), ["1D", "1H"], 24 * 3),
    (OrderBlockStrategy("1H", session_gated=False), ["1H"], 30),
]

WARMUP = {"1H": 260, "4H": 260, "1D": 230}
BARS = {"1H": 9000, "4H": 2500, "1D": 700}
SPECS = {"step": 0.001}  # permissive; the $5 leg floor is what bites


def _ts_ms(ts_array):
    """A `ts` column as int64 epoch milliseconds, whichever of the two
    conventions this repo's bar frames actually use: bt.load_bars' own cache
    stores it as datetime64 (bars_dataframe converts on fetch), while a frame
    built straight from data/bars_1h_deep.pkl - or resampled from it, as
    Strategy 3's derived daily cache is - keeps it as the raw int64 ms the
    exchange returned. Both need to land in the same space before bucket
    arithmetic can compare across them."""
    if np.issubdtype(ts_array.dtype, np.datetime64):
        return ts_array.astype("datetime64[ms]").astype("int64")
    return ts_array.astype("int64")


# --------------------------------------------------------------------------
# Phase 1: generate every signal, in parallel across symbols.
# --------------------------------------------------------------------------

def scan_symbol(args):
    """Every signal one symbol would have produced across the window.

    Returns (symbol, [(ts, local_bar_index, bar_close, instance_pos, signal)]).
    Runs in a worker process; bars for this symbol only are passed in, so no
    worker ever loads the whole 51MB cache.

    A 4th element - [(pos_i, strategy, needs, cancel), ...] - restricts
    evaluation to just those instances, addressed by the SAME pos_i the full
    INSTANCES list would use, so their rows slot into a portfolio replay
    unchanged. This is what lets the per-instance cache ask for a rescan of
    only the strategies whose hash went stale, without touching the rest.
    Omitting it (every existing call site) scans the whole module-global
    INSTANCES list, exactly as before.
    """
    symbol, bars, hours, *rest = args
    instances = rest[0] if rest else [(i, s, n, c) for i, (s, n, c) in enumerate(INSTANCES)]
    h1 = bars.get("1H")
    if h1 is None or len(h1) <= WARMUP["1H"] + 100:
        return symbol, []

    idx = {}
    for tf, frame in bars.items():
        if tf == "1H" or frame is None:
            continue
        # How many of this timeframe's bars have CLOSED as of each 1H bar -
        # not "how many have a ts <= this 1H bar's ts". Those agree for a real
        # exchange feed (closed_only=True never delivers a candle before it has
        # actually closed, so every row's ts is already strictly behind ANY 1H
        # bar that could reference it) but diverge hard for a frame built by
        # resampling 1H bars ahead of time (as Strategy 3's derived daily cache
        # is): that frame holds each day's FINAL close even for the 1H bars
        # from early that same day, so comparing raw ts let a 3am bar see its
        # own day's eventual high/close - a same-day lookahead leak. Flooring
        # each 1H bar's own ts to ITS bucket start and searching side="left"
        # excludes that bucket's own row regardless of source; for genuinely
        # closed-only data this is a no-op (verified: same result both ways),
        # since a closed candle's ts is already provably before its own bucket
        # would even start. Found via Strategy 3 swing reading 10,008 valid
        # setups and firing 0 breakout crossings on real data - the crossing
        # threshold was silently baked from the future every single time.
        bucket_ms = TF_SECONDS[tf] * 1000
        h1_ms = _ts_ms(h1["ts"].values)
        own_bucket_start = (h1_ms // bucket_ms) * bucket_ms
        idx[tf] = _ts_ms(frame["ts"].values).searchsorted(own_bucket_start, side="left")

    out = []
    armed_cache: dict = {}
    start = max(WARMUP["1H"] + 1, len(h1) - hours)
    for i in range(start, len(h1)):
        view = {"1H": h1.iloc[: i + 1]}
        ok = True
        for tf in bars:
            if tf == "1H":
                continue
            if bars[tf] is None:
                continue
            k = int(idx[tf][i])
            if k < WARMUP.get(tf, 230):
                continue
            view[tf] = bars[tf].iloc[:k]

        for pos_i, strategy, needs, _cancel in instances:
            if any(view.get(tf) is None for tf in needs):
                continue
            # Only when this instance's OWN base bar has just closed. The live
            # scanner aligns to candle closes; a 4H strategy re-reading
            # identical data on each of the four 1H bars inside its candle is
            # both wrong and four times the work.
            base = min(needs, key=lambda tf: TF_SECONDS[tf])
            if base != "1H" and int(idx[base][i]) == int(idx[base][i - 1]):
                continue
            # THE ARMING LAYER, which this harness used to skip entirely.
            #
            # Live, a strategy that declares armed_timeframes is evaluated only
            # on symbols its arms() accepted, and arms() is recomputed on each
            # REGULAR scan - i.e. off the non-armed timeframes, at their
            # cadence. Calling evaluate() on every bar instead lets an armed
            # instance signal on setups the live bot would never have looked
            # at, so its backtest counts were an upper bound and known to be
            # one. Strategy 3's 1D/5m is the instance this matters for, and it
            # is the one about to be backtested for the first time.
            if strategy.armed_timeframes:
                unarmed = [tf for tf in needs if tf not in strategy.armed_timeframes]
                if not unarmed:
                    continue  # nothing to arm FROM; live it would never be polled
                # Recompute only when a non-armed bar closed, as live does.
                slowest = max(unarmed, key=lambda tf: TF_SECONDS[tf])
                key = (pos_i, int(idx[slowest][i]) if slowest != "1H" else i)
                if key not in armed_cache:
                    try:
                        armed_cache[key] = strategy.arms(
                            symbol, {tf: view[tf] for tf in unarmed if view.get(tf) is not None}
                        )
                    except Exception:
                        armed_cache[key] = False
                if not armed_cache[key]:
                    continue
            try:
                sig = strategy.evaluate(symbol, {tf: view[tf] for tf in needs})
            except Exception:
                continue
            if sig is not None:
                out.append((h1["ts"].iloc[i], i, float(h1["close"].iloc[i]), pos_i, sig))
    return symbol, out


# A completed symbol is a finished, independent piece of work - there is no
# reason for one to be lost because a later one was interrupted. The first full
# run held every result in memory and wrote nothing until all 98 symbols were
# done, so killing it at symbol 30 would have discarded roughly fifteen hours.
# That is not a hypothetical: the machine slept overnight mid-run, and the only
# reason the work survived is that nobody touched it.
CHECKPOINT_EVERY = 5


def _load_checkpoint(path: str, key) -> dict:
    """Symbols already generated for THIS scope, or nothing.

    The key guards against resuming into a different question: a checkpoint
    from a 40-symbol run must not silently supply 40 of the 98 symbols another
    run needs, because the result would look complete and be wrong.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        saved_key, signals = pickle.load(open(path, "rb"))
    except Exception:
        return {}  # a truncated checkpoint costs time, never correctness
    return signals if saved_key == key else {}


def _save_checkpoint(path: str, key, signals: dict) -> None:
    """Write via a temp file and replace, so an interrupt during the write
    cannot leave a half-written checkpoint where a whole one used to be."""
    if not path:
        return
    try:
        checkpoint.write(path, (key, signals))
    except Exception:
        logger.exception("Could not write the signal checkpoint")


def generate(symbols, hours, workers, checkpoint: str | None = None, key=None, cache=None,
             instance_cache_path: str | None = None, only_pos=None):
    """cache, when given, replaces the bt.load_bars fetch-or-load with a
    caller-supplied {(symbol, tf): DataFrame} dict - e.g. one built from
    data/bars_1h_deep.pkl, for a deep multi-year universe bt.load_bars' own
    cache was never fetched at.

    instance_cache_path switches to the per-(symbol, instance_hash) cache
    (see instance_cache.py): a strategy's own hash only changes when its
    rule actually changed, so editing ONE instance rescans only that
    instance, for every symbol - not the ~6h full re-scan every edit costs
    today. checkpoint/key are ignored in this mode; the per-instance store
    is its own checkpoint (every entry it holds is already complete).

    only_pos restricts generation to those INSTANCES positions. A caller that
    replays just one instance (run_s3_swing.py drops the other four through
    skip_pos) otherwise pays to generate signals it then throws away -
    measured on this list, that is ~66% of the work, because Strategy 4
    alone costs 56% of a full scan and is one of the discarded ones."""
    if cache is None:
        cache = bt.load_bars(symbols, ["1D", "1H", "4H"], BARS)
    usable = [s for s in symbols
              if cache.get((s, "1H")) is not None
              and len(cache[(s, "1H")]) > WARMUP["1H"] + 100]
    bars_1h = {s: cache[(s, "1H")] for s in usable}

    if instance_cache_path is not None:
        return _generate_per_instance(bars_1h, cache, hours, workers, instance_cache_path,
                                       only_pos=only_pos)

    signals = _load_checkpoint(checkpoint, key)
    signals = {s: v for s, v in signals.items() if s in bars_1h}
    todo = [s for s in usable if s not in signals]
    if signals:
        print(f"resuming: {len(signals)} symbols already generated, {len(todo)} to go", flush=True)
    print(f"{len(usable)} symbols usable · generating signals on {workers} workers", flush=True)
    if not todo:
        return bars_1h, signals

    subset = (None if only_pos is None
              else [(p,) + tuple(INSTANCES[p]) for p in sorted(set(only_pos))])
    tasks = [
        ((s, {tf: cache.get((s, tf)) for tf in ("1D", "1H", "4H")}, hours) if subset is None
         else (s, {tf: cache.get((s, tf)) for tf in ("1D", "1H", "4H")}, hours, subset))
        for s in todo
    ]

    t0, done = time.time(), 0
    with Pool(workers) as pool:
        for symbol, found in pool.imap_unordered(scan_symbol, tasks):
            signals[symbol] = found
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                _save_checkpoint(checkpoint, key, signals)
            if done % 10 == 0:
                total = sum(len(v) for v in signals.values())
                print(f"  {len(signals)}/{len(usable)} symbols · {total} signals · "
                      f"{time.time()-t0:.0f}s elapsed", flush=True)
    _save_checkpoint(checkpoint, key, signals)
    return bars_1h, signals


def _rows_at_pos(rows, pos_i):
    """Cached rows re-pointed at `pos_i`.

    Each row carries the INSTANCES position it was GENERATED under (row[3]),
    but the store is keyed by the instance's HASH - so reordering INSTANCES
    leaves a cached row's index pointing at whatever strategy now occupies
    that slot. This repo has done exactly that reorder before: Strategy 2's
    retirement shifted every later position (see this module's docstring on
    why a pre-2026-08-16 signals cache cannot be replayed against the current
    list). Nothing raises when it happens - _replay reads INSTANCES[pos_i]
    for the cancel window and skip_pos tests against it, so the only symptom
    is a trade quietly attributed to the wrong strategy.
    """
    return [(ts, i, close, pos_i, sig) for ts, i, close, _generated_at, sig in rows]


def _merge_and_store(store, hashes, signals, symbol, found, fresh_rows, stale):
    """Fold one symbol's freshly-scanned rows into `store` (keyed by the
    hash that produced them) and into `signals` (the flat per-symbol list
    scan_symbol/replay have always used).

    Every stale position gets a store entry even when it fired NOTHING for
    this symbol - otherwise an instance with zero signals on a symbol would
    look perpetually stale and be rescanned forever."""
    by_pos = defaultdict(list)
    for row in found:
        by_pos[row[3]].append(row)
    for pos_i in stale:
        store[(symbol, hashes[pos_i])] = by_pos.get(pos_i, [])
    signals[symbol] = fresh_rows + found


def _run_subset(task):
    symbol, bars_for_symbol, hours, subset, fresh_rows, stale = task
    _sym, found = scan_symbol((symbol, bars_for_symbol, hours, subset))
    return symbol, found, fresh_rows, stale


def _generate_per_instance(bars_1h, cache, hours, workers, path, only_pos=None):
    """generate()'s instance_cache_path branch: hash every current INSTANCES
    entry, load what is already cached under those hashes, and scan only the
    (symbol, instance) pairs the cache does not already hold.

    only_pos narrows the whole operation to those INSTANCES positions - both
    what gets scanned and what comes back. Entries for the positions left
    out are neither read nor evicted, so asking for them later still finds
    them cached.

    `hours` is folded into the store key alongside the instance hash. A
    cache entry built by scanning the last 24 hours must never be handed
    back for a request asking for the last 200 - that would silently
    describe a wider window with a narrower window's answer. This means
    widening the window pays a full re-scan of that instance rather than an
    incremental one; only a strategy's own hash changing, or the window
    staying fixed, is fast - the same scope test-driven-development pinned
    up front (growing the window is not yet incremental, only correct)."""
    from backtest import instance_cache as ic

    wanted = list(range(len(INSTANCES))) if only_pos is None else sorted(set(only_pos))
    hashes = {pos_i: (ic.instance_hash(INSTANCES[pos_i][0]), hours) for pos_i in wanted}
    store = ic.load_store(path)

    tasks = []
    signals = {}
    for symbol in bars_1h:
        stale = [pos_i for pos_i, h in hashes.items() if (symbol, h) not in store]
        fresh_rows = []
        for pos_i, h in hashes.items():
            if pos_i not in stale:
                fresh_rows.extend(_rows_at_pos(store.get((symbol, h), []), pos_i))
        if not stale:
            signals[symbol] = fresh_rows
            continue
        subset = [(pos_i,) + INSTANCES[pos_i] for pos_i in stale]
        bars_for_symbol = {tf: cache.get((symbol, tf)) for tf in ("1D", "1H", "4H")}
        tasks.append((symbol, bars_for_symbol, hours, subset, fresh_rows, stale))

    print(f"{len(bars_1h)} symbols usable · {len(bars_1h) - len(tasks)} fully cached · "
          f"{len(tasks)} have a stale instance", flush=True)
    if not tasks:
        return bars_1h, signals

    t0, done = time.time(), 0
    if workers and workers > 1:
        with Pool(workers) as pool:
            for symbol, found, fresh_rows, stale in pool.imap_unordered(_run_subset, tasks):
                _merge_and_store(store, hashes, signals, symbol, found, fresh_rows, stale)
                done += 1
                if done % CHECKPOINT_EVERY == 0:
                    ic.save_store(path, store)
    else:
        for task in tasks:
            symbol, found, fresh_rows, stale = _run_subset(task)
            _merge_and_store(store, hashes, signals, symbol, found, fresh_rows, stale)
            done += 1
            if done % CHECKPOINT_EVERY == 0:
                ic.save_store(path, store)

    ic.save_store(path, store)
    print(f"per-instance generation done in {time.time()-t0:.0f}s "
          f"({len(tasks)} symbols rescanned)", flush=True)
    return bars_1h, signals


# --------------------------------------------------------------------------
# Phase 2: replay the portfolio in timestamp order. Seconds, not hours.
# --------------------------------------------------------------------------

def replay(bars_1h, signals, skip_pos=(), cancel_override=None, max_total_risk=None,
           start_ts=None, end_ts=None, pivots_cache=None, score_refused=True,
           start_equity=None):
    """One account, one clock, every symbol competing for it.

    Ordering within a timestamp mirrors the live loop: bars close, open
    positions are advanced, and only then does the scanner look for new
    entries. A signal on the same bar that closed a position can therefore
    reuse the freed margin, which is what the live bot does too.

    skip_pos drops instances by their INSTANCES position. It exists because
    Strategy 4 is in DRY_RUN_TAGS live - it places nothing - so including it
    here lets it consume margin and one-position-per-symbol slots that the real
    account still has free, crowding out live strategies. Both readings are
    wanted: with it, what the account WOULD do if Strategy 4 graduated;
    without, what it does today.

    start_ts/end_ts bound the timeline to [start_ts, end_ts) - a fresh
    bt.Account() is always created here, so calling this once per calendar
    year is what gives each year its own $100 account rather than one
    continuously-compounding run that can decay below Bitget's $5 floor in a
    bad early stretch and never recover (see memory:
    flat-strategy-equity-death-spiral). bars_1h is passed in FULL (unsliced)
    even when windowing - signal bar_index values were assigned against the
    full frame at generation time, and slicing bars_1h separately per call
    would desync them.

    pivots_cache is an optional {symbol: confirmed_pivots(...)} built once by
    the caller and shared across arms. Confirmed pivots are a pure function of
    bars, and bars do not change between the arms of a sweep, so recomputing
    them per arm is pure waste - measured at ~0.1s per symbol, i.e. ~20s an arm
    over a 200-symbol universe, paid again by every one of a few dozen arms.
    None keeps the old behaviour of computing them lazily per call. The lists
    are only ever read (each Position walks its own cursor), so sharing is safe.

    score_refused=False skips scoring the signals the $5 floor refused. That
    scoring is 97% of this function's runtime - profiled at 464s of a 477s
    DEEP2Y replay, 6.9M pandas row lookups across 8,912 score_too_small calls -
    because every refused signal is walked bar by bar exactly like a taken one,
    and the floor refuses an order of magnitude more than it passes. It answers
    "what did the small account miss?", which a PARAMETER SWEEP never asks: the
    sweep reads acct.declined_too_small, a counter try_open increments for free,
    and never acct.scored_too_small. Only backtest/run.py consumes the scored
    list, so it keeps the default. Nothing else about the replay changes -
    acct.too_small is still drained so memory stays flat, and every field the
    sweep reads is computed identically either way.

    start_equity overrides the $100 the account opens with. It is a real
    parameter rather than a bt.START_EQUITY monkeypatch because Account's
    equity and peak are DATACLASS FIELD DEFAULTS, bound when the class is
    created - rebinding the module global afterwards silently does nothing,
    and a sweep would report $100 numbers while believing otherwise.

    It is not cosmetic. Position notional is (risk% / stop%) x equity, so the
    $5-per-leg MIN_NOTIONAL floor bites in inverse proportion to the balance:
    doubling equity halves the stop width at which a leg gets refused. Since
    the floor selects on stop WIDTH, every arm that moves stop width or leg
    count is confounded by it - which is what made a market_fraction sweep on
    $100 look like an edge on 2026-09-03 when it was the floor reshaping the
    trade population. Model the balance actually being traded.
    """
    skip_pos = set(skip_pos)
    # The aggregate open-risk ceiling was raised 6% -> 15% in production on
    # 2026-08-14, ahead of this measurement rather than because of it. Per-trade
    # risk is unchanged, so what the cap governs is how many trades may run at
    # once - a drawdown decision. Both are replayed so the change has a number.
    previous_cap = bt.MAX_TOTAL_RISK_PCT
    if max_total_risk is not None:
        bt.MAX_TOTAL_RISK_PCT = max_total_risk
    try:
        return _replay(bars_1h, signals, skip_pos, cancel_override, start_ts, end_ts,
                       pivots_cache, score_refused, start_equity)
    finally:
        bt.MAX_TOTAL_RISK_PCT = previous_cap


def _replay(bars_1h, signals, skip_pos, cancel_override, start_ts=None, end_ts=None,
            pivots_cache=None, score_refused=True, start_equity=None):
    acct = (bt.Account() if start_equity is None
            else bt.Account(equity=start_equity, peak=start_equity))

    # A position with no stated remainder_target trails on confirmed swings of
    # the timeframe its own instance entered on (engine.try_open's pivots
    # param) - without them the remainder just sits at breakeven with an
    # infinite target, understating every such runner (see try_open's
    # docstring). Only instances whose OWN base timeframe is "1H" get a real
    # answer here, because that is the only spine this replay walks bar-by-bar;
    # a 4H- or 1D-based instance's pivots would need re-indexing onto the 1H
    # spine (as backtest/generate_v2.py's _pivots_on_1h does) and still don't
    # get one - those remain understated exactly as before this fix.
    pivots_by_symbol: dict = {} if pivots_cache is None else pivots_cache

    def pivots_for(symbol):
        if symbol not in pivots_by_symbol:
            f = bars_1h[symbol]
            pivots_by_symbol[symbol] = confirmed_pivots(f, atr(f))
        return pivots_by_symbol[symbol]

    ts_to_row = {s: {ts: i for i, ts in enumerate(f["ts"].values)}
                 for s, f in bars_1h.items()}
    by_ts = defaultdict(list)
    for symbol, found in signals.items():
        for ts, i, close, pos_i, sig in found:
            if pos_i in skip_pos:
                continue
            if start_ts is not None and ts < start_ts:
                continue
            if end_ts is not None and ts >= end_ts:
                continue
            by_ts[ts].append((symbol, i, close, pos_i, sig))

    timeline = sorted(set().union(*(set(f["ts"].values) for f in bars_1h.values())))
    if start_ts is not None:
        timeline = [ts for ts in timeline if ts >= start_ts]
    if end_ts is not None:
        timeline = [ts for ts in timeline if ts < end_ts]
    scored_too_small = []

    for ts in timeline:
        for symbol in list(acct.open_positions):
            row = ts_to_row[symbol].get(ts)
            if row is None:
                continue
            pos = acct.open_positions[symbol]
            bt.step_position(acct, pos, bars_1h[symbol].iloc[row], row, ts)

        # Deterministic order when several symbols signal on the same bar.
        # Whoever is first gets the margin; alphabetical is arbitrary but
        # stable, and the alternative - letting dict order decide - would make
        # the result depend on which worker finished first.
        for symbol, i, close, pos_i, sig in sorted(by_ts.get(ts, ()), key=lambda e: (e[0], e[3])):
            _strategy, needs, cancel_after = INSTANCES[pos_i]
            # Live cancels EVERY unfilled entry at a flat 4 hours
            # (tracker.ENTRY_TIMEOUT_SECONDS), while these per-instance windows
            # run to 96 bars. The divergence only bites when nothing filled
            # immediately - which is precisely the single-leg limit fallback -
            # so the arm that looks best is the one the longer window flatters.
            # cancel_override=4 is the honest model of the bot as it is today.
            base = min(needs, key=lambda tf: TF_SECONDS[tf])
            pivots = pivots_for(symbol) if base == "1H" else None
            bt.try_open(acct, sig, close, i, SPECS,
                        cancel_after if cancel_override is None else cancel_override,
                        pivots)

        # Score and drain what the $5 floor refused, so memory stays flat.
        # Draining happens either way - it is what bounds memory; only the
        # scoring is optional (see replay's score_refused).
        if len(acct.too_small) > 400:
            if score_refused:
                for symbol in {e[0] for e in acct.too_small}:
                    mine = [e for e in acct.too_small if e[0] == symbol]
                    scored_too_small.extend(bt.score_too_small(mine, bars_1h[symbol]))
            acct.too_small = []

    if score_refused:
        for symbol in {e[0] for e in acct.too_small}:
            mine = [e for e in acct.too_small if e[0] == symbol]
            scored_too_small.extend(bt.score_too_small(mine, bars_1h[symbol]))
    acct.too_small = []
    acct.scored_too_small = scored_too_small
    return acct
