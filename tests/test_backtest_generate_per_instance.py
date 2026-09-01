"""portfolio.generate's per-instance cache: the actual point of this whole
effort. A strategy-rule edit costs ~6h today because generate() re-scans
every instance for every symbol whenever ANYTHING changes - there is no way
to ask for just the one instance that actually changed. These tests pin that
a hash change rescans ONLY the instance whose hash changed, and that its
result is identical to what a full cold scan would have produced - a cache
that quietly reuses the wrong thing would corrupt every measurement built on
it without ever raising an error.
"""

import pandas as pd

from backtest import portfolio as pf
from notifier.strategies.base import Strategy


def _bars(n=400, start="2025-01-01"):
    ts = pd.date_range(start, periods=n, freq="h")
    return pd.DataFrame({
        "ts": ts, "open": [100.0] * n, "high": [100.0] * n,
        "low": [100.0] * n, "close": [100.0] * n, "volume": [1.0] * n,
    })


class _CountingA(Strategy):
    tag = "a"
    timeframes = ["1H"]

    def __init__(self):
        self.calls = 0

    def evaluate(self, symbol, bars_by_timeframe):
        self.calls += 1
        return None


class _CountingB(Strategy):
    tag = "b"
    timeframes = ["1H"]

    def __init__(self, threshold):
        self.threshold = threshold
        self.calls = 0

    def evaluate(self, symbol, bars_by_timeframe):
        self.calls += 1
        return None


def _cache_for(symbols, bars_by_symbol):
    cache = {}
    for s in symbols:
        cache[(s, "1H")] = bars_by_symbol[s]
        cache[(s, "1D")] = None
        cache[(s, "4H")] = None
    return cache


def test_an_unchanged_instance_is_never_rescanned_after_another_instances_hash_changes(tmp_path, monkeypatch):
    symbols = ["AAAUSDT", "BBBUSDT"]
    bars_by_symbol = {s: _bars() for s in symbols}
    cache_path = str(tmp_path / "instance_signals.pkl")

    a1, b1 = _CountingA(), _CountingB(threshold=1)
    monkeypatch.setattr(pf, "INSTANCES", [(a1, ["1H"], 24), (b1, ["1H"], 24)])
    pf.generate(symbols, hours=24, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    assert a1.calls > 0 and b1.calls > 0, "the cold run must have scanned both instances"

    # Only B's config changes - a different threshold means a different hash.
    a2, b2 = _CountingA(), _CountingB(threshold=2)
    monkeypatch.setattr(pf, "INSTANCES", [(a2, ["1H"], 24), (b2, ["1H"], 24)])
    pf.generate(symbols, hours=24, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    assert a2.calls == 0, "A's hash is unchanged - it must be served from cache, not re-evaluated"
    assert b2.calls > 0, "B's hash changed - it must actually be rescanned"


def test_the_reused_signals_are_identical_to_a_fresh_cold_scan(tmp_path, monkeypatch):
    """The cache's whole value proposition depends on this: a partially
    cached run must produce EXACTLY what a full regeneration would have,
    not merely something plausible."""
    from notifier.strategies.base import Signal

    symbols = ["AAAUSDT"]
    bars_by_symbol = {s: _bars() for s in symbols}
    cache_path = str(tmp_path / "instance_signals.pkl")

    class _Firing(Strategy):
        tag = "firing"
        timeframes = ["1H"]

        def evaluate(self, symbol, bars_by_timeframe):
            if len(bars_by_timeframe["1H"]) == 400:  # fires once, on the very last bar
                return Signal(symbol=symbol, direction="long", entry_price=100.0,
                               stop_loss=95.0, strategy_tag="firing")
            return None

    other1 = _CountingB(threshold=1)
    monkeypatch.setattr(pf, "INSTANCES", [(_Firing(), ["1H"], 24), (other1, ["1H"], 24)])
    _bars1h, signals_partial = pf.generate(
        symbols, hours=50, workers=1,
        cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    # Change ONLY the other instance, so _Firing is served from cache this time.
    other2 = _CountingB(threshold=2)
    monkeypatch.setattr(pf, "INSTANCES", [(_Firing(), ["1H"], 24), (other2, ["1H"], 24)])
    _bars1h, signals_cached = pf.generate(
        symbols, hours=50, workers=1,
        cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    # And a totally fresh cold scan, no cache at all, for ground truth.
    monkeypatch.setattr(pf, "INSTANCES", [(_Firing(), ["1H"], 24), (_CountingB(threshold=2), ["1H"], 24)])
    _bars1h, signals_cold = pf.generate(
        symbols, hours=50, workers=1,
        cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=str(tmp_path / "fresh.pkl"))

    firing_rows_cached = [r for r in signals_cached["AAAUSDT"] if r[3] == 0]
    firing_rows_cold = [r for r in signals_cold["AAAUSDT"] if r[3] == 0]
    assert firing_rows_cached == firing_rows_cold
    assert len(firing_rows_cached) == 1


def test_a_brand_new_instance_only_scans_itself_not_the_existing_ones(tmp_path, monkeypatch):
    """Adding a strategy must not force every existing one to rescan."""
    symbols = ["AAAUSDT"]
    bars_by_symbol = {s: _bars() for s in symbols}
    cache_path = str(tmp_path / "instance_signals.pkl")

    a1 = _CountingA()
    monkeypatch.setattr(pf, "INSTANCES", [(a1, ["1H"], 24)])
    pf.generate(symbols, hours=24, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    a2, brand_new = _CountingA(), _CountingB(threshold=1)
    monkeypatch.setattr(pf, "INSTANCES", [(a2, ["1H"], 24), (brand_new, ["1H"], 24)])
    pf.generate(symbols, hours=24, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    assert a2.calls == 0, "the pre-existing instance must be served from cache"
    assert brand_new.calls > 0, "the newly added instance has no cache entry yet"


def test_generation_survives_being_interrupted_between_symbols(tmp_path, monkeypatch):
    """The whole reason checkpointing exists elsewhere in this package: an
    interrupted run must not lose the symbols it already finished."""
    symbols = ["AAAUSDT", "BBBUSDT", "CCCUSDT"]
    bars_by_symbol = {s: _bars() for s in symbols}
    cache_path = str(tmp_path / "instance_signals.pkl")

    monkeypatch.setattr(pf, "CHECKPOINT_EVERY", 1)
    monkeypatch.setattr(pf, "INSTANCES", [(_CountingA(), ["1H"], 24)])
    pf.generate(symbols, hours=24, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    from backtest import instance_cache as ic
    saved = ic.load_store(cache_path)
    assert len(saved) == 3, "all three symbols' entries must have actually landed on disk"


def test_widening_the_hours_window_actually_rescans(tmp_path, monkeypatch):
    """A cache entry built by scanning the last 24 hours must never be handed
    back for a request asking for the last 200 hours - that would silently
    describe a wider window with a narrower window's answer, which is worse
    than not caching at all: the number would look complete and be wrong,
    exactly the failure the checkpoint scoping (test_backtest_checkpoint.py)
    already guards against on the OLD cache path."""
    symbols = ["AAAUSDT"]
    bars_by_symbol = {s: _bars() for s in symbols}
    cache_path = str(tmp_path / "instance_signals.pkl")

    narrow = _CountingA()
    monkeypatch.setattr(pf, "INSTANCES", [(narrow, ["1H"], 24)])
    pf.generate(symbols, hours=24, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)
    narrow_calls = narrow.calls

    wide = _CountingA()
    monkeypatch.setattr(pf, "INSTANCES", [(wide, ["1H"], 24)])
    pf.generate(symbols, hours=200, workers=1,
                cache=_cache_for(symbols, bars_by_symbol), instance_cache_path=cache_path)

    assert wide.calls > narrow_calls, (
        "a wider window must actually be scanned, not served from the narrower window's cache entry"
    )
