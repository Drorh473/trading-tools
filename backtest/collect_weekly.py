"""Weekly incremental top-up of data/bars_1h_deep.pkl.

1H is the only timeframe fetched: backtest.generate_v2._closed_frames already
resamples 4H and 1D from it (RULE = {"4H": "4h", "1D": "1D"}), so topping up
1H alone keeps all three current without a second fetch path.

    python -m backtest.collect_weekly

Meant to run once a week (cron on the VM, Saturday 00:00 Asia/Jerusalem) so a
year of runs never needs another full historical backfill - each run fetches
only the bars newer than what is already cached per symbol, rather than
fetch_1h_deep's full from-listing-date pull. A symbol newly in the top-200
that isn't in the cache yet still gets that full pull, once.

Bootstraps nothing: run backtest.fetch_1h_deep once first if data/bars_1h_deep.pkl
doesn't exist yet. This script only ever grows an existing cache.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

import pandas as pd

from backtest.fetch_1h_deep import COLUMNS, OUT_DEFAULT
from backtest.fetch_1h_deep import fetch_symbol as fetch_symbol_full
from config import settings
from core.bitget_client import client_from_settings
from core.run_guard import notify_on_completion

# Added on top of the raw gap when sizing a top-up request, so a run that
# fires a little early or late (or a symbol whose clock drifts slightly)
# still covers the whole gap in one request rather than leaving a bar or two
# stranded until next week.
TOPUP_BUFFER_BARS = 48

# Past this many hours of gap, fall back to the full paged fetch instead of
# asking get_candles to top up. get_candles's own history-candles loop has no
# attempt cap and calls _request directly - none of fetch_symbol's backoff on
# 429 - which is fine for a week's worth of pages and not something to trust
# for a cache stale enough to mean several missed runs.
TOPUP_FALLBACK_HOURS = 24 * 30

HOUR_MS = 3_600_000


def top_up_symbol(client, symbol: str, existing: pd.DataFrame, now_ms: int | None = None) -> pd.DataFrame:
    """Only the bars newer than what's already cached for this symbol."""
    if existing.empty:
        return fetch_symbol_full(client, symbol)

    last_ts = int(existing["ts"].iloc[-1])
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    gap_hours = max((now_ms - last_ts) / HOUR_MS, 0.0)

    if gap_hours > TOPUP_FALLBACK_HOURS:
        return fetch_symbol_full(client, symbol)

    wanted = int(gap_hours) + TOPUP_BUFFER_BARS
    data = client.get_candles(symbol, granularity="1H", limit=wanted, closed_only=True)
    if not data:
        return existing

    frame = pd.DataFrame(data, columns=COLUMNS[: len(data[0])])
    frame = frame.astype({c: float for c in frame.columns})
    frame["ts"] = frame["ts"].astype("int64")

    # Pages can overlap the cache's own last bar; dedupe on ts rather than
    # trusting the seam, same as fetch_1h_deep's own stitching.
    combined = pd.concat([existing, frame], ignore_index=True)
    combined = combined.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return combined


def main() -> None:
    # Bitget lists CJK symbol names and Windows redirects stdout through the
    # locale codepage - see fetch_1h_deep.py's own note. Same fix here so a
    # print never gets to decide whether a week's top-up survives.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--out", default=OUT_DEFAULT)
    args = parser.parse_args()

    with notify_on_completion("Weekly bar top-up") as note:
        if not os.path.exists(args.out):
            raise FileNotFoundError(
                f"{args.out} doesn't exist - run `python -m backtest.fetch_1h_deep` once "
                "to bootstrap it before this incremental top-up can run."
            )

        client = client_from_settings(settings)
        tickers = client.get_all_tickers()
        ranked = sorted(tickers, key=lambda t: float(t.get("usdtVolume") or 0), reverse=True)
        symbols = [t["symbol"] for t in ranked[: args.top]]

        with open(args.out, "rb") as handle:
            bars: dict[str, pd.DataFrame] = pickle.load(handle)

        topped_up = 0
        new_symbols = 0
        new_bars = 0
        failed: list[str] = []

        for symbol in symbols:
            before = len(bars[symbol]) if symbol in bars else 0
            try:
                if symbol in bars:
                    bars[symbol] = top_up_symbol(client, symbol, bars[symbol])
                    topped_up += 1
                else:
                    bars[symbol] = fetch_symbol_full(client, symbol)
                    new_symbols += 1
            except Exception as exc:
                failed.append(symbol)
                print(f"  {symbol:<14} FAILED  {str(exc)[:90]}")
                continue
            new_bars += len(bars[symbol]) - before

        tmp = args.out + ".tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(bars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, args.out)

        total = sum(len(f) for f in bars.values())
        note.headline = (
            f"{topped_up} topped up, {new_symbols} new symbols, +{new_bars:,} bars "
            f"({total:,} total across {len(bars)} symbols)"
        )
        if failed:
            note.headline += f"; {len(failed)} failed: {', '.join(failed[:5])}"
        print(f"\n{note.headline}")


if __name__ == "__main__":
    main()
