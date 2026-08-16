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
"""
import logging
import os
import pickle
import time
from collections import defaultdict
from multiprocessing import Pool

from backtest import engine as bt
from notifier.strategies.base import TIMEFRAME_SECONDS as TF_SECONDS
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
#   Strategy 2 1H/15m  - needs 15m, Bitget serves 22 days
#   Strategy 3 1D/5m   - needs 5m, ~2 days
#   Strategy 4 15m     - same 15m limit
# Approximating them would be inventing the result. Two of the nine LIVE
# instances are therefore still unmeasured after this run.
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


# --------------------------------------------------------------------------
# Phase 1: generate every signal, in parallel across symbols.
# --------------------------------------------------------------------------

def scan_symbol(args):
    """Every signal one symbol would have produced across the window.

    Returns (symbol, [(ts, local_bar_index, bar_close, instance_pos, signal)]).
    Runs in a worker process; bars for this symbol only are passed in, so no
    worker ever loads the whole 51MB cache.
    """
    symbol, bars, hours = args
    h1 = bars.get("1H")
    if h1 is None or len(h1) <= WARMUP["1H"] + 100:
        return symbol, []

    idx = {}
    for tf, frame in bars.items():
        if tf == "1H" or frame is None:
            continue
        idx[tf] = frame["ts"].searchsorted(h1["ts"].values, side="right")

    out = []
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

        for pos_i, (strategy, needs, _cancel) in enumerate(INSTANCES):
            if any(view.get(tf) is None for tf in needs):
                continue
            # Only when this instance's OWN base bar has just closed. The live
            # scanner aligns to candle closes; a 4H strategy re-reading
            # identical data on each of the four 1H bars inside its candle is
            # both wrong and four times the work.
            base = min(needs, key=lambda tf: TF_SECONDS[tf])
            if base != "1H" and int(idx[base][i]) == int(idx[base][i - 1]):
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
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(tmp, "wb") as fh:
            pickle.dump((key, signals), fh)
        os.replace(tmp, path)
    except Exception:
        logger.exception("Could not write the signal checkpoint")


def generate(symbols, hours, workers, checkpoint: str | None = None, key=None):
    cache = bt.load_bars(symbols, ["1D", "1H", "4H"], BARS)
    usable = [s for s in symbols
              if cache.get((s, "1H")) is not None
              and len(cache[(s, "1H")]) > WARMUP["1H"] + 100]
    bars_1h = {s: cache[(s, "1H")] for s in usable}

    signals = _load_checkpoint(checkpoint, key)
    signals = {s: v for s, v in signals.items() if s in bars_1h}
    todo = [s for s in usable if s not in signals]
    if signals:
        print(f"resuming: {len(signals)} symbols already generated, {len(todo)} to go", flush=True)
    print(f"{len(usable)} symbols usable · generating signals on {workers} workers", flush=True)
    if not todo:
        return bars_1h, signals

    tasks = [(s, {tf: cache.get((s, tf)) for tf in ("1D", "1H", "4H")}, hours) for s in todo]

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


# --------------------------------------------------------------------------
# Phase 2: replay the portfolio in timestamp order. Seconds, not hours.
# --------------------------------------------------------------------------

def replay(bars_1h, signals, skip_pos=(), cancel_override=None, max_total_risk=None):
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
        return _replay(bars_1h, signals, skip_pos, cancel_override)
    finally:
        bt.MAX_TOTAL_RISK_PCT = previous_cap


def _replay(bars_1h, signals, skip_pos, cancel_override):
    acct = bt.Account()

    ts_to_row = {s: {ts: i for i, ts in enumerate(f["ts"].values)}
                 for s, f in bars_1h.items()}
    by_ts = defaultdict(list)
    for symbol, found in signals.items():
        for ts, i, close, pos_i, sig in found:
            if pos_i in skip_pos:
                continue
            by_ts[ts].append((symbol, i, close, pos_i, sig))

    timeline = sorted(set().union(*(set(f["ts"].values) for f in bars_1h.values())))
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
            _strategy, _needs, cancel_after = INSTANCES[pos_i]
            # Live cancels EVERY unfilled entry at a flat 4 hours
            # (tracker.ENTRY_TIMEOUT_SECONDS), while these per-instance windows
            # run to 96 bars. The divergence only bites when nothing filled
            # immediately - which is precisely the single-leg limit fallback -
            # so the arm that looks best is the one the longer window flatters.
            # cancel_override=4 is the honest model of the bot as it is today.
            bt.try_open(acct, sig, close, i, SPECS,
                        cancel_after if cancel_override is None else cancel_override)

        # Score and drain what the $5 floor refused, so memory stays flat.
        if len(acct.too_small) > 400:
            for symbol in {e[0] for e in acct.too_small}:
                mine = [e for e in acct.too_small if e[0] == symbol]
                scored_too_small.extend(bt.score_too_small(mine, bars_1h[symbol]))
            acct.too_small = []

    for symbol in {e[0] for e in acct.too_small}:
        mine = [e for e in acct.too_small if e[0] == symbol]
        scored_too_small.extend(bt.score_too_small(mine, bars_1h[symbol]))
    acct.too_small = []
    acct.scored_too_small = scored_too_small
    return acct
