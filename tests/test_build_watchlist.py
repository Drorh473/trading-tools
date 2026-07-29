from notifier.build_watchlist import top_symbols_by_volume


class FakeBitget:
    def __init__(self, tickers):
        self._tickers = tickers

    def get_all_tickers(self):
        return self._tickers


def _ticker(symbol, volume):
    return {"symbol": symbol, "usdtVolume": str(volume)}


def test_ranks_by_volume_descending():
    bitget = FakeBitget(
        [
            _ticker("LOWUSDT", 100),
            _ticker("BTCUSDT", 5_000_000),
            _ticker("MIDUSDT", 10_000),
        ]
    )

    symbols = top_symbols_by_volume(bitget, top=2)

    assert symbols == ["BTCUSDT", "MIDUSDT"]


def test_respects_top_n():
    bitget = FakeBitget([_ticker(f"SYM{i}USDT", i) for i in range(10)])
    assert len(top_symbols_by_volume(bitget, top=3)) == 3
