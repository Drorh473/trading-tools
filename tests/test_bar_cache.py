from notifier.bar_cache import BarCache


class FakeBitget:
    def __init__(self):
        self.calls = []

    def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
        self.calls.append((symbol, granularity, limit, closed_only))
        return [[str(1000 * (i + 1)), "100", "101", "99", "100", "1", "1"] for i in range(5)]


def test_refetches_only_once_the_candle_turns_over():
    bitget = FakeBitget()
    cache = BarCache(bitget)
    hour = 3600.0

    cache.get("BTCUSDT", "1H", now=hour)
    cache.get("BTCUSDT", "1H", now=hour + 900)  # same hourly candle
    cache.get("BTCUSDT", "1H", now=hour + 3599)  # still the same one

    assert len(bitget.calls) == 1

    cache.get("BTCUSDT", "1H", now=hour + 3600)  # next candle

    assert len(bitget.calls) == 2


def test_different_symbols_and_timeframes_are_cached_independently():
    bitget = FakeBitget()
    cache = BarCache(bitget)

    cache.get("BTCUSDT", "1H", now=0.0)
    cache.get("ETHUSDT", "1H", now=0.0)
    cache.get("BTCUSDT", "4H", now=0.0)

    assert len(bitget.calls) == 3


def test_deep_history_override_fetches_a_larger_limit_for_just_that_key():
    bitget = FakeBitget()
    cache = BarCache(bitget, candle_limit=600, deep_history={("BTCUSDT", "1H"): 20000})

    cache.get("BTCUSDT", "1H", now=0.0)
    cache.get("ETHUSDT", "1H", now=0.0)

    btc_limit = next(c[2] for c in bitget.calls if c[0] == "BTCUSDT")
    eth_limit = next(c[2] for c in bitget.calls if c[0] == "ETHUSDT")
    assert btc_limit == 20001
    assert eth_limit == 601


def test_fetches_include_the_forming_candle():
    bitget = FakeBitget()
    cache = BarCache(bitget)

    cache.get("BTCUSDT", "1H", now=0.0)

    assert bitget.calls[0][3] is False  # closed_only=False
