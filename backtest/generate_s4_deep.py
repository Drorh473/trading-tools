"""Generate every Strategy 4 (order block) 1H signal across the full universe.

Strategy 4 has never had a backtest. The only sample that existed was
data/s4_signals_1H.pkl - 57 signals across 75 symbols over ~7 years, roughly
one signal per symbol every two years - which is far too thin to measure
anything. The first deliverable here is not an expectancy number, it is the
answer to "what sample is achievable at all", which nobody currently knows.

    python -m backtest.generate_s4_deep --workers 10

Reads data/bars_1h_deep_np.pkl (758 symbols, 9.44M bars, back to 2019-07-09)
and writes data/s4_signals_deep.pkl as {symbol: [(ts, bar_index, close,
INSTANCE_POS, signal), ...]}, the shape backtest.portfolio.replay() consumes.

TWO THINGS THIS DOES DIFFERENTLY FROM portfolio.scan_symbol, both deliberate.

The window is CAPPED at LIVE_WINDOW bars. scan_symbol passes h1.iloc[:i+1] -
every bar of history seen so far - but the live scanner fetches candle_limit
(600) and never sees more, so an unbounded window lets structure_context grow
its search past anything the bot could actually do. It is also what made a
deep run impractical: cost scales with what you hand it, and at 5,000 bars one
evaluate took 3.4 SECONDS against 29ms at the live size.

Only the Strategy 4 instance is evaluated. The other four in INSTANCES would
re-measure work already done and, at this depth, dominate the runtime.

The cache stores `ts` as int64 MILLISECONDS. pd.Timestamp(x) reads a bare int
as NANOSECONDS and lands silently in 1970, so anything that formats a date
must pass unit='ms'.

CACHING. data/s4_signals_deep_store.pkl holds every (symbol, rule-hash) pair
this script has ever fully scanned (see instance_cache.py and _current_hash),
and never evicts one. A genuinely new rule still costs a full rescan of all
734 symbols - there is no shortcut around evaluating new logic against real
bars - but switching BACK to a rule version already generated before is free,
because its results are still sitting in the store under their own hash.
args.out is a flat {symbol: [...]} export of just the current hash's view,
rebuilt from the store every run, in the shape backtest.portfolio.replay()
consumes.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import time
from multiprocessing import Pool

import pandas as pd

from backtest import instance_cache as ic
from backtest.portfolio import INSTANCES, WARMUP
from backtest.sampling import stratified_sample

BARS_DEFAULT = "data/bars_1h_deep_np.pkl"
OUT_DEFAULT = "data/s4_signals_deep.pkl"
# Every (symbol, rule-hash) this script has ever fully scanned, kept forever
# rather than overwritten - args.out below is a flat EXPORT of just the
# current hash's view, rebuilt from this store every run.
STORE_DEFAULT = "data/s4_signals_deep_store.pkl"
COLUMNS = ("ts", "open", "high", "low", "close", "base_vol", "quote_vol")

# The scanner's own candle_limit. Everything the live strategy can see.
LIVE_WINDOW = 601

# Position of OrderBlockStrategy("1H") inside portfolio.INSTANCES. Recorded in
# each signal tuple so portfolio.replay() can look up the cancel window, and
# asserted at import so a reordering of INSTANCES fails loudly here instead of
# silently replaying Strategy 4's signals as some other strategy's.
INSTANCE_POS = 4
assert type(INSTANCES[INSTANCE_POS][0]).__name__ == "OrderBlockStrategy", (
    "INSTANCES reordered; INSTANCE_POS is stale"
)


def _frame(cols: dict) -> pd.DataFrame:
    return pd.DataFrame({c: cols[c] for c in COLUMNS if c in cols})


def scan_symbol(task):
    """Every Strategy 4 signal one symbol would have produced."""
    symbol, cols = task
    strategy = INSTANCES[INSTANCE_POS][0]
    frame = _frame(cols)
    n = len(frame)
    if n <= WARMUP["1H"] + 100:
        return symbol, []

    found = []
    closes = frame["close"].to_numpy()
    stamps = frame["ts"].to_numpy()
    for i in range(WARMUP["1H"] + 1, n):
        lo = max(0, i + 1 - LIVE_WINDOW)
        try:
            signal = strategy.evaluate(symbol, {"1H": frame.iloc[lo : i + 1]})
        except Exception:
            continue
        if signal is not None:
            found.append((int(stamps[i]), i, float(closes[i]), INSTANCE_POS, signal))
    return symbol, found


def _current_hash() -> str:
    """No per-symbol structure to key a rescan by, so the only question this
    script can answer cheaply is "did the strategy (or this script's own
    window-capping logic) change at all" - covers what instance_hash alone
    would miss, since LIVE_WINDOW capping lives in THIS file, not in
    OrderBlockStrategy itself."""
    strategy = INSTANCES[INSTANCE_POS][0]
    parts = [ic.instance_hash(strategy)]
    with open(__file__, "rb") as fh:
        parts.append(hashlib.sha256(fh.read()).hexdigest())
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()[:16]


def _plan(symbols: list, store: dict, current_hash: str) -> tuple[dict, list]:
    """Which symbols already have a cache entry under the CURRENT hash
    (reused verbatim, from `store`), and which still need scanning.

    A symbol cached only under a DIFFERENT hash is treated as needing a
    rescan - trusting it would silently describe a rule nobody runs any
    more. But that old entry is never touched: `store` keeps every hash it
    has ever seen, so switching the strategy back to a version already
    generated before finds its results still here, and costs nothing."""
    todo = [s for s in symbols if (s, current_hash) not in store]
    todo_set = set(todo)
    signals = {s: store[(s, current_hash)] for s in symbols if s not in todo_set}
    return signals, todo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", default=BARS_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--store", default=STORE_DEFAULT,
                     help="per-(symbol, rule-hash) cache; every rule version this "
                          "has ever fully scanned is kept, never evicted")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit", type=int, default=0, help="first N symbols only, for a smoke run")
    # A stratified random subset. The full universe costs ~75 core-hours, and
    # the question it answers first - what signal RATE do the current rules
    # produce - is an average over bars, not something only exhaustion can
    # reach. Sampling across the bar-count deciles keeps deep and shallow
    # listings represented in proportion, so the rate scales up honestly.
    ap.add_argument("--sample", type=int, default=0, help="stratified random N symbols")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    current_hash = _current_hash()

    with open(args.bars, "rb") as fh:
        bars = pickle.load(fh)
    symbols = [s for s, c in bars.items() if len(c.get("ts", ())) > WARMUP["1H"] + 100]
    symbols.sort()
    if args.limit:
        symbols = symbols[: args.limit]
    if args.sample:
        symbols = stratified_sample(symbols, args.sample,
                                     lambda s: len(bars[s]["ts"]), seed=args.seed)

    # store keeps every (symbol, rule-hash) this script has ever scanned,
    # never evicted - so switching the strategy back to a version already
    # generated before is free, and only a genuinely new rule pays for a
    # real scan. args.out below is a flat export of just this run's hash.
    store = ic.load_store(args.store)
    signals, todo = _plan(symbols, store, current_hash)

    total_bars = sum(len(bars[s]["ts"]) for s in todo)
    print(f"{len(symbols)} symbols usable · {len(signals)} already cached under this rule · "
          f"{len(todo)} to go ({total_bars:,} bars) on {args.workers} workers", flush=True)
    if not todo:
        print("nothing to do")
        with open(args.out, "wb") as fh:
            pickle.dump(signals, fh, protocol=4)
        return

    tasks = [(s, bars[s]) for s in todo]
    t0, done = time.time(), 0
    with Pool(args.workers) as pool:
        for symbol, found in pool.imap_unordered(scan_symbol, tasks):
            signals[symbol] = found
            store[(symbol, current_hash)] = found
            done += 1
            if done % 10 == 0 or done == len(todo):
                ic.save_store(args.store, store)
                with open(args.out, "wb") as fh:
                    pickle.dump(signals, fh, protocol=4)
                got = sum(len(v) for v in signals.values())
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(todo) - done) / rate / 3600 if rate else 0
                print(f"  {len(signals)}/{len(symbols)} symbols · {got} signals · "
                      f"{time.time()-t0:.0f}s · ETA {eta:.1f}h", flush=True)

    ic.save_store(args.store, store)
    with open(args.out, "wb") as fh:
        pickle.dump(signals, fh, protocol=4)

    got = sum(len(v) for v in signals.values())
    with_any = sum(1 for v in signals.values() if v)
    print(f"\n{got} signals across {with_any}/{len(signals)} symbols "
          f"in {(time.time()-t0)/3600:.2f}h")
    if got:
        stamps = sorted(e[0] for v in signals.values() for e in v)
        print(f"span {pd.Timestamp(stamps[0], unit='ms'):%Y-%m-%d} .. "
              f"{pd.Timestamp(stamps[-1], unit='ms'):%Y-%m-%d}")


if __name__ == "__main__":
    main()
