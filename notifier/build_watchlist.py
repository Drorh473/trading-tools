"""CLI command to refresh notifier/watchlist.py with the current top-N
Bitget USDT-margined futures symbols by 24h volume. Run manually when you
want to update the list:

    python -m notifier.build_watchlist
    python -m notifier.build_watchlist --top 50
"""

import argparse
from pathlib import Path

from core.bitget_client import BitgetClient

WATCHLIST_PATH = Path(__file__).parent / "watchlist.py"


def top_symbols_by_volume(bitget: BitgetClient, top: int) -> list[str]:
    tickers = bitget.get_all_tickers()
    ranked = sorted(tickers, key=lambda t: float(t["usdtVolume"]), reverse=True)
    return [t["symbol"] for t in ranked[:top]]


def write_watchlist(symbols: list[str]) -> None:
    lines = ['"""Crypto pairs to scan, as Bitget USDT-margined futures symbols."""', "", "WATCHLIST: list[str] = ["]
    lines += [f'    "{s}",' for s in symbols]
    lines += ["]", ""]
    WATCHLIST_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh the notifier watchlist by 24h volume")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    bitget = BitgetClient()  # public market data only, no credentials needed
    symbols = top_symbols_by_volume(bitget, args.top)
    write_watchlist(symbols)
    print(f"Wrote {len(symbols)} symbols to {WATCHLIST_PATH}")


if __name__ == "__main__":
    main()
