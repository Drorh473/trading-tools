"""Baseline expectancy for Strategy 3's swing instance (VolumeRun("1D","1H")),
isolated from the rest of the portfolio, against the full deep universe.

No P&L validation existed for this strategy before this script - it was
rebuilt around a horizontal-level detector on 2026-08-23 and shipped
DRY_RUN-only for exactly that reason (see notifier/main.py). This answers the
first, most basic question: is it profitable at all.

    python -m backtest.run_s3_swing [hours]

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
  Generation runs the full INSTANCES list (skipping it per-instance is not
  reliably possible across Windows' spawn-based Pool - a mutated module
  global does not reach a freshly spawned worker), but replay only counts
  Strategy 3 swing (INSTANCES[3]): every other instance is passed in
  skip_pos, so its signals never open a position or compete for margin here.

Fresh account per calendar year
  Each year gets its own bt.Account() (portfolio.replay's start_ts/end_ts
  window), not one continuously-compounding run - a bad early stretch can
  decay equity below Bitget's $5 floor and lock the account out of trading
  for the rest of the window (memory: flat-strategy-equity-death-spiral).
"""
import pickle
import sys
import time

import pandas as pd

from backtest import engine as bt
from backtest import portfolio as pf

S3_SWING_POS = 3  # INSTANCES[3] == VolumeRun("1D", "1H", time_exit_days=3)
BARS_1H = "data/bars_1h_deep.pkl"
SIGNAL_CACHE = "data/s3_swing_signals.pkl"
CHECKPOINT = "data/s3_swing_signals_partial.pkl"

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
    n = len(trades)
    if n == 0:
        print(f"{label}: no closed trades")
        return
    wins = [t for t in trades if t.pnl > 0]
    tot_r = sum(t.r for t in trades)
    print(f"{label}: n={n} win={len(wins)/n*100:.0f}% totalR={tot_r:+.1f} "
          f"expectancy={tot_r/n:+.3f}R")
    if n > 3:
        dropped = sorted(trades, key=lambda t: -t.r)[3:]
        dn = len(dropped)
        dr = sum(t.r for t in dropped)
        print(f"  drop-top-3: n={dn} totalR={dr:+.1f} expectancy={dr/dn:+.3f}R")
    reasons = {r: sum(1 for t in trades if t.reason == r) for r in ("stop", "target", "runner")}
    print(f"  exit reasons: {reasons}")


def main():
    hours = int(sys.argv[1]) if len(sys.argv) > 1 else 17520  # ~2 years of 1H bars

    symbols, cache = _load_universe()

    key = ("s3_swing", len(symbols), hours)
    t0 = time.time()
    bars_1h, signals = pf.generate(symbols, hours, workers=10,
                                   checkpoint=CHECKPOINT, key=key, cache=cache)
    with open(SIGNAL_CACHE, "wb") as f:
        pickle.dump((key, bars_1h, signals), f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"generation done in {time.time()-t0:.0f}s", flush=True)

    n_s3 = sum(1 for found in signals.values() for e in found if e[3] == S3_SWING_POS)
    print(f"\n{n_s3} Strategy 3 swing signals across {len(bars_1h)} symbols")
    if n_s3 == 0:
        print("No signals at all - stopping before replay.")
        return

    all_ts = [ts for f in bars_1h.values() for ts in f["ts"].values]
    years = _year_bounds_ms(all_ts)
    print(f"years present: {[y for y, _, _ in years]}")

    skip_pos = {i for i in range(len(pf.INSTANCES)) if i != S3_SWING_POS}

    all_trades = []
    print("\n================ Strategy 3 swing, isolated, fresh account/year ================")
    for year, start_ms, end_ms in years:
        acct = pf.replay(bars_1h, signals, skip_pos=skip_pos,
                         start_ts=start_ms, end_ts=end_ms)
        report_arm(f"{year}", acct.closed)
        all_trades.extend(acct.closed)

    print("\n================ pooled across every year ================")
    report_arm("ALL YEARS", all_trades)


if __name__ == "__main__":
    main()
