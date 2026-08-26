import importlib.util

from notifier.build_watchlist import top_symbols_by_volume, write_watchlist


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


def test_write_watchlist_produces_a_module_that_imports_back_the_same_symbols(tmp_path, monkeypatch):
    """notifier/scanner.py imports WATCHLIST from the file this writes, live -
    a malformed literal (an unescaped symbol, a dropped comma) would not fail
    until the next deploy tried to import it."""
    import notifier.build_watchlist as bw

    out_path = tmp_path / "watchlist.py"
    monkeypatch.setattr(bw, "WATCHLIST_PATH", out_path)
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    write_watchlist(symbols)

    spec = importlib.util.spec_from_file_location("written_watchlist", out_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.WATCHLIST == symbols
