"""Baseline expectancy for Strategy 3's swing instance (VolumeRun("1D","1H")),
isolated from the rest of the portfolio, against the full deep universe.

No P&L validation existed for this strategy before this script - it was
rebuilt around a horizontal-level detector on 2026-08-23 and shipped
DRY_RUN-only for exactly that reason (see notifier/main.py). This answers the
first, most basic question: is it profitable at all.

    python -m backtest.run_s3_swing [hours] [--sample N] [--workers N]

Data
  data/bars_1h_deep.pkl (758 symbols, 1H, back to each symbol's listing date)
  - not backtest.engine.load_bars' own cache, which only reaches back ~380
  days. data/bars_1d_tail.pkl looked like its daily companion by name but
  isn't: none of BTCUSDT/ETHUSDT/SOLUSDT/... are in it (561 symbols, none of
  them the majors) - it is some other, unrelated long-tail universe, not this
  one. So the 1D frame VolumeRun's trend_timeframe needs is DERIVED here by
  resampling each symbol's own 1H bars, which is also strictly more correct
  (identical source, no separate-fetch alignment risk).

  `ts` throughout is raw int64 MILLISECONDS, exactly as bars_1h_deep.pkl
  stores it and exactly what VolumeRun.weekly_trend_levels expects
  (pd.to_datetime(daily["ts"], unit="ms")) - this script converts it only
  once, deliberately, to resample daily bars, and immediately converts back;
  everywhere else (year boundaries, sorting, dict keys) it stays a raw int64
  ms epoch, specifically to avoid the ns-misread trap that has bit this
  measurement effort twice already.

Isolation
  Only Strategy 3 swing (INSTANCES[3]) is GENERATED, and only it is replayed:
  every other instance is passed in skip_pos, so its signals could never open
  a position or compete for margin here anyway.

  This used to generate the full INSTANCES list and throw four fifths of it
  away at replay, because "skipping it per-instance is not reliably possible
  across Windows' spawn-based Pool - a mutated module global does not reach a
  freshly spawned worker". True of a module global; the subset now travels
  inside the pickled task tuple (portfolio.scan_symbol's 4th element), which
  reaches a spawned worker fine. Measured on this INSTANCES list, generating
  all five costs ~3x generating just this one - Strategy 4 alone is 56% of a
  full scan, and it was one of the four being discarded.

Fresh account per calendar year
  Each year gets its own bt.Account() (portfolio.replay's start_ts/end_ts
  window), not one continuously-compounding run - a bad early stretch can
  decay equity below Bitget's $5 floor and lock the account out of trading
  for the rest of the window (memory: flat-strategy-equity-death-spiral).
"""
import argparse
import pickle
import time

import pandas as pd

from backtest import engine as bt
from backtest import portfolio as pf
from backtest import stats
from backtest.sampling import stratified_sample

S3_SWING_POS = 3  # INSTANCES[3] == VolumeRun("1D", "1H", time_exit_days=3)
# What is generated, and what is excluded from the replay. Kept as one pair
# because they must partition INSTANCES exactly: generating an instance the
# replay skips is wasted hours, and skipping one that was never generated
# would silently measure nothing. test_generated_and_skipped_partition_instances
# pins that.
GENERATE_POS = [S3_SWING_POS]
SKIP_POS = frozenset(i for i in range(len(pf.INSTANCES)) if i not in GENERATE_POS)
BARS_1H = "data/bars_1h_deep.pkl"
# Its own file, separate from run.py's data/instance_signals.pkl: the store
# key is (symbol, instance_hash, hours) with no fingerprint of the BARS
# content, and this script's daily bars are DERIVED (resampled from 1H)
# rather than fetched - the same symbol could hash-collide across the two
# scripts' different bars sources otherwise.
INSTANCE_CACHE = "data/s3_swing_instance_signals.pkl"

DAILY_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "base_vol": "sum"}


def _daily_from_1h(h1: pd.DataFrame) -> pd.DataFrame:
    """Resample this symbol's own 1H bars into daily candles, ts kept as raw
    ms int64. The last row is dropped unconditionally - it is the still-
    forming "today" bucket (1H data never ends exactly at a UTC day
    boundary), and keeping it would let a partial day be read as a closed
    one, the same class of lookahead confirmed_pivots exists to avoid."""
    idx = pd.to_datetime(h1["ts"], unit="ms")
    daily = h1.set_index(idx).resample("1D").agg(DAILY_AGG).dropna()
    daily = daily.iloc[:-1].rename_axis("ts").reset_index()  # drop the still-forming last day
    # NOT `.astype("int64") // 1_000_000`: that assumes int64 always reads out
    # in nanoseconds, true on pandas < 2 but not here - pd.to_datetime(unit="ms")
    # on this pandas (3.0.5) keeps datetime64[ms] resolution straight through
    # resample/reset_index, so a bare int64 view is ALREADY milliseconds and
    # dividing again corrupts every ts down near the 1970 epoch. Casting to
    # datetime64[ms] explicitly first makes the int64 view correct regardless
    # of whatever resolution resample happened to produce.
    daily["ts"] = daily["ts"].astype("datetime64[ms]").astype("int64")
    return daily


def _year_bounds_ms(ts_values) -> list[tuple[int, int, int]]:
    """[(year, start_ms, end_ms)] covering every calendar year present, in
    raw epoch milliseconds (UTC) - computed by hand, not via pd.to_datetime
    on the data itself, so the ns-misread trap never gets a chance to fire."""
    import datetime as dt

    lo, hi = int(min(ts_values)), int(max(ts_values))
    first_year = dt.datetime.utcfromtimestamp(lo / 1000).year
    last_year = dt.datetime.utcfromtimestamp(hi / 1000).year
    out = []
    for y in range(first_year, last_year + 1):
        start = int(dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
        end = int(dt.datetime(y + 1, 1, 1, tzinfo=dt.timezone.utc).timestamp() * 1000)
        out.append((y, start, end))
    return out


def _load_universe():
    print(f"loading {BARS_1H} ...", flush=True)
    with open(BARS_1H, "rb") as f:
        h1 = pickle.load(f)
    print(f"{len(h1)} symbols with 1H · deriving daily bars by resampling", flush=True)
    cache = {}
    symbols = []
    for s, f1h in h1.items():
        if f1h is None or len(f1h) < 400:
            continue
        daily = _daily_from_1h(f1h)
        if len(daily) < 60:  # VolumeRun.min_daily_bars() floor, roughly
            continue
        cache[(s, "1H")] = f1h
        cache[(s, "1D")] = daily
        symbols.append(s)
    symbols.sort()
    print(f"{len(symbols)} symbols usable after the daily-history floor", flush=True)
    return symbols, cache


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else float("nan")


def report_arm(label, trades):
    s = stats.summarize(trades)
    if s.n == 0:
        print(f"{label}: no closed trades")
        return
    print(f"{label}: n={s.n} win={s.win_rate*100:.0f}% totalR={s.total_r:+.1f} "
          f"expectancy={s.expectancy:+.3f}R")
    if s.drop_top3_n:
        print(f"  drop-top-3: n={s.drop_top3_n} totalR={s.drop_top3_total_r:+.1f} "
              f"expectancy={s.drop_top3_expectancy:+.3f}R")
    print(f"  exit reasons: {s.exit_reasons}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("hours", nargs="?", type=int, default=17520,
                     help="1H bars back from the end (default ~2 years)")
    ap.add_argument("--workers", type=int, default=10)
    # A stratified subset, for iterating on the rule without paying the full
    # universe. Because the signal store is keyed (symbol, instance_hash,
    # hours), the symbols a sampled run scans are cached under exactly the key
    # a later full run looks them up by - so a sample is a down payment, not
    # throwaway work.
    ap.add_argument("--sample", type=int, default=0,
                     help="stratified random N symbols (0 = the whole universe)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    hours = args.hours

    symbols, cache = _load_universe()
    if args.sample:
        symbols = stratified_sample(symbols, args.sample,
                                     lambda s: len(cache[(s, "1H")]), seed=args.seed)
        print(f"stratified sample: {len(symbols)} symbols (seed {args.seed})", flush=True)

    t0 = time.time()
    bars_1h, signals = pf.generate(symbols, hours, workers=args.workers, cache=cache,
                                   instance_cache_path=INSTANCE_CACHE,
                                   only_pos=GENERATE_POS)
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    n_s3 = sum(1 for found in signals.values() for e in found if e[3] == S3_SWING_POS)
    print(f"\n{n_s3} Strategy 3 swing signals across {len(bars_1h)} symbols")
    if n_s3 == 0:
        print("No signals at all - stopping before replay.")
        return

    all_ts = [ts for f in bars_1h.values() for ts in f["ts"].values]
    years = _year_bounds_ms(all_ts)
    print(f"years present: {[y for y, _, _ in years]}")

    all_trades = []
    print("\n================ Strategy 3 swing, isolated, fresh account/year ================")
    for year, start_ms, end_ms in years:
        acct = pf.replay(bars_1h, signals, skip_pos=SKIP_POS,
                         start_ts=start_ms, end_ts=end_ms)
        report_arm(f"{year}", acct.closed)
        all_trades.extend(acct.closed)

    print("\n================ pooled across every year ================")
    report_arm("ALL YEARS", all_trades)


if __name__ == "__main__":
    main()
