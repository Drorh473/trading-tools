"""Fetch deep 15m history for the top-volume symbols.

THE 22-DAY LIMIT WAS NEVER REAL. §47, generate_v2's docstring and the v2 spec
all record that "Bitget serves ~22 days of 15m", so both 15m instances were
written off as permanently unmeasurable. That came from calling get_candles with
a plain limit. Paging through history-candles reaches 418 DAYS on 15m - further
back than the 1H cache this project has been measuring on all along.

Checkpointed per symbol, so an interrupted fetch resumes.
"""
import argparse, os, pickle, time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from config import settings
from core.bitget_client import client_from_settings

OUT = "data/bars_15m.pkl"
BARS = 40000  # ~418 days
COLS = ["ts", "open", "high", "low", "close", "base_vol", "quote_vol"]


def frame(rows):
    df = pd.DataFrame([r[:7] for r in rows], columns=COLS)
    for c in COLS[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    return df.dropna().sort_values("ts").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--bars", type=int, default=BARS)
    a = ap.parse_args()

    client = client_from_settings(settings)
    tickers = client.get_all_tickers()
    ranked = sorted(
        (t for t in tickers if t["symbol"].endswith("USDT")),
        key=lambda t: float(t.get("usdtVolume") or 0),
        reverse=True,
    )[: a.top]
    syms = [t["symbol"] for t in ranked]
    print(f"top {len(syms)} by 24h USDT volume; deepest {ranked[0]['symbol']}, "
          f"thinnest {ranked[-1]['symbol']} at ${float(ranked[-1]['usdtVolume'])/1e6:.1f}M", flush=True)

    done = {}
    if os.path.exists(OUT):
        try:
            done = pickle.load(open(OUT, "rb"))
            print(f"resuming: {len(done)} symbols already fetched", flush=True)
        except Exception:
            done = {}

    todo = [s for s in syms if s not in done]
    t0 = time.time()
    lock_n = [0]

    def one(sym):
        """Retry 429s with exponential backoff rather than dropping the symbol.

        The first pass lost 115 of 200 to rate limits because a 429 anywhere in
        the paging loop aborts the whole symbol - and paging 418 days is ~35
        requests, so the deepest symbols are the likeliest to be cut off.
        """
        delay = 5.0
        for attempt in range(5):
            try:
                return sym, frame(client.get_candles(sym, granularity="15m", limit=a.bars))
            except Exception as exc:
                if "429" not in str(exc) or attempt == 4:
                    return sym, exc
                time.sleep(delay)
                delay *= 2
        return sym, RuntimeError("unreachable")

    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for sym, res in ex.map(one, todo):
            lock_n[0] += 1
            if isinstance(res, Exception):
                print(f"  [{lock_n[0]}/{len(todo)}] {sym:14s} FAILED {res}", flush=True)
                continue
            done[sym] = res
            if lock_n[0] % 10 == 0 or lock_n[0] == len(todo):
                span = (pd.to_datetime(res.ts.max(), unit="ms") - pd.to_datetime(res.ts.min(), unit="ms")).days
                print(f"  [{lock_n[0]}/{len(todo)}] {sym:14s} {len(res):6d} bars, {span}d, "
                      f"{(time.time()-t0)/60:.1f}m elapsed", flush=True)
                tmp = OUT + ".tmp"
                with open(tmp, "wb") as fh:
                    pickle.dump(done, fh)
                os.replace(tmp, OUT)

    with open(OUT, "wb") as fh:
        pickle.dump(done, fh)
    n = [len(v) for v in done.values()]
    print(f"\nDONE: {len(done)} symbols, median {int(pd.Series(n).median())} bars, "
          f"{(time.time()-t0)/60:.1f} min -> {OUT}")


if __name__ == "__main__":
    main()
