"""Fetch 1H bars back to each symbol's LISTING DATE for the top-200 symbols.

Why this exists: every result measured so far sat on a 1H cache of 9,199 bars
per symbol (2025-07-24 -> 2026-08-11, 383 days). That window is a single
regime - BTC fell 45.6% across it - so a parameter whose sample dies just past
its "optimum" (the >=40-bar pivot-span rule: 144 trades at 40, 18 at 60, 4 at
80) cannot be told apart from a boundary artifact.

The 383 days were never the exchange's limit. /api/v2/mix/market/candles caps
its lookback, but /api/v2/mix/market/history-candles is anchored by endTime and
reaches back to the symbol's first bar: BTCUSDT 1H returns 62,278 bars from
2019-07-10, measured 2026-08-17. Depth varies per symbol only because listing
dates do.

    python -m backtest.fetch_1h_deep --workers 6 --top 200

Checkpointed per symbol into data/bars_1h_deep.pkl (gitignored), so an
interrupt costs one symbol rather than the run. Re-running resumes: symbols
already present are skipped unless --refresh.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import settings
from core.bitget_client import client_from_settings

OUT_DEFAULT = "data/bars_1h_deep.pkl"
COLUMNS = ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]
# One history-candles request returns at most 200 rows; asking for more is a
# hard 40053, not a silent clamp. Verified 2026-08-17 at 300/500/1000/1500.
PAGE = 200
# A page that returns fewer than this many rows means we have reached the
# symbol's listing date, not a hiccup - the endpoint fills a full page while
# history remains.
MAX_PAGES = 2000  # 400k bars; far past any 1H listing depth


def _fetch_page(client, symbol: str, end_ms: int | None) -> list[list[str]]:
    """One page of 1H history, with backoff on 429 rather than a hard retry.

    Bitget answers a burst with HTTP 429 and no Retry-After; hammering it just
    extends the throttle, so each attempt waits longer than the last.
    """
    params = {
        "symbol": symbol,
        "productType": "USDT-FUTURES",
        "granularity": "1H",
        "limit": str(PAGE),
    }
    if end_ms is not None:
        params["endTime"] = str(int(end_ms))

    for attempt in range(6):
        try:
            return client._request(
                "GET", "/api/v2/mix/market/history-candles", params=params, signed=False
            )
        except RuntimeError as exc:
            transient = "429" in str(exc) or "Too Many Requests" in str(exc)
            if not transient or attempt == 5:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{symbol}: rate limited out")


def fetch_symbol(client, symbol: str) -> pd.DataFrame:
    """Every 1H bar this symbol has, oldest first.

    Pages backwards by endTime. The loop stops when a page comes back empty or
    stops moving the anchor - a page whose oldest bar equals the previous
    anchor would otherwise spin forever on a symbol at its floor.
    """
    rows: list[list[str]] = []
    end_ms: int | None = None
    for _ in range(MAX_PAGES):
        page = _fetch_page(client, symbol, end_ms)
        if not page:
            break
        oldest = int(page[0][0])
        if end_ms is not None and oldest >= end_ms:
            break
        rows = page + rows
        end_ms = oldest
        if len(page) < PAGE:
            break

    if not rows:
        return pd.DataFrame(columns=COLUMNS)

    frame = pd.DataFrame(rows, columns=COLUMNS[: len(rows[0])])
    frame = frame.astype({c: float for c in frame.columns})
    frame["ts"] = frame["ts"].astype("int64")
    # Pages are stitched at their boundaries, so the same bar can arrive twice;
    # dedupe on ts rather than trusting the seam.
    frame = frame.drop_duplicates(subset="ts").sort_values("ts").reset_index(drop=True)
    return frame


def main() -> None:
    # Bitget lists symbols with CJK names - 龙虾USDT is in the top 200 by
    # volume - and Windows encodes redirected stdout with the locale codepage,
    # cp1255 (Hebrew) on this machine. Printing that symbol's progress line
    # raised UnicodeEncodeError and killed a run 78 symbols deep: the fetch
    # was fine, REPORTING it was not. Never let logging decide whether hours
    # of fetching survive.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--top", type=int, default=200)
    parser.add_argument("--out", default=OUT_DEFAULT)
    parser.add_argument("--refresh", action="store_true", help="refetch symbols already saved")
    args = parser.parse_args()

    client = client_from_settings(settings)

    tickers = client.get_all_tickers()
    ranked = sorted(tickers, key=lambda t: float(t.get("usdtVolume") or 0), reverse=True)
    symbols = [t["symbol"] for t in ranked[: args.top]]
    print(f"{len(tickers)} symbols live; taking top {len(symbols)} by 24h usdtVolume")

    bars: dict[str, pd.DataFrame] = {}
    if os.path.exists(args.out) and not args.refresh:
        with open(args.out, "rb") as handle:
            bars = pickle.load(handle)
        print(f"resuming: {len(bars)} symbols already in {args.out}")

    todo = [s for s in symbols if s not in bars]
    print(f"{len(todo)} to fetch, {args.workers} workers\n")

    lock = threading.Lock()
    started = time.time()
    done = 0

    def save() -> None:
        """Atomic: a crash mid-write must not leave a truncated pickle where a
        good one was."""
        tmp = args.out + ".tmp"
        with open(tmp, "wb") as handle:
            pickle.dump(bars, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, args.out)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_symbol, client, s): s for s in todo}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception as exc:
                print(f"  {symbol:<14} FAILED  {str(exc)[:90]}")
                continue
            with lock:
                bars[symbol] = frame
                done += 1
                span_days = (
                    (frame["ts"].iloc[-1] - frame["ts"].iloc[0]) / 86_400_000 if len(frame) else 0
                )
                first = (
                    pd.to_datetime(frame["ts"].iloc[0], unit="ms").strftime("%Y-%m-%d")
                    if len(frame)
                    else "-"
                )
                rate = done / max(time.time() - started, 1e-9)
                eta = (len(todo) - done) / rate / 60 if rate else 0
                print(
                    f"  [{done:>3}/{len(todo)}] {symbol:<14} {len(frame):>7,} bars  "
                    f"{span_days:>7.1f}d  from {first}   eta {eta:>5.1f}m"
                )
                if done % 10 == 0:
                    save()

    save()
    elapsed = time.time() - started
    total = sum(len(f) for f in bars.values())
    print(f"\nwrote {args.out}: {len(bars)} symbols, {total:,} bars, {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
