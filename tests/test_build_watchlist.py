import pytest

from notifier.build_watchlist import (
    _FIB_GAP,
    clears_split_entry_minimum,
    executable_symbols_by_volume,
    top_symbols_by_volume,
)


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


# ---- per-leg minimum vs a symbol's own typical swing ----


def test_fib_gap_matches_the_strategys_own_stop_distance():
    # 78.6% - 61.8%, fixed by Strategy 1. If this drifts from the strategy's
    # own constants, the measurement is checking the wrong thing.
    assert _FIB_GAP == pytest.approx(0.168)


def test_clears_minimum_at_a_typical_swing():
    # ADAUSDT's own typical swing (not the outlier day that actually failed
    # live) clears the minimum at $100 equity - matches the live measurement
    # that ADAUSDT itself wasn't on the "would drop" list.
    assert clears_split_entry_minimum(median_swing_pct=0.15, equity=100.0) is True


def test_rejects_a_symbol_whose_typical_swing_is_too_wide():
    assert clears_split_entry_minimum(median_swing_pct=0.60, equity=100.0) is False


def test_the_threshold_scales_with_equity():
    # A bigger account can afford wider swings and still clear $5 on the
    # market leg - this must not be a fixed swing-pct cutoff.
    swing = 0.30
    assert clears_split_entry_minimum(swing, equity=50.0) is False
    assert clears_split_entry_minimum(swing, equity=500.0) is True


def test_zero_swing_never_clears():
    assert clears_split_entry_minimum(0.0, equity=1_000_000.0) is False


# ---- selection: skip-and-backfill rather than shrink the list ----


class FakeBitgetWithSwings(FakeBitget):
    """Deterministic ZigZag swing per symbol, so selection is tested without
    hitting the network."""

    def __init__(self, tickers, swings):
        super().__init__(tickers)
        self._swings = swings

    def get_candles(self, symbol, granularity="1H", limit=300, closed_only=True):
        # Flat warmup keeps ATR low, then clean ramps whose amplitude clears
        # the ZigZag threshold by a wide margin - a tight oscillation (tried
        # first) never confirmed a single pivot, since ATR scaled with the
        # same amplitude the threshold was measured against.
        swing_pct = self._swings.get(symbol, 0.15)
        base = 100.0
        peak = base * (1 + swing_pct)

        def leg(start, stop, n):
            step = (stop - start) / n
            return [start + step * (i + 1) for i in range(n)]

        closes = [base] * 30 + leg(base, peak, 10) + leg(peak, base, 10) + leg(base, peak, 10) + leg(peak, base, 10)
        return [
            [str(1000 + i), str(c), str(c * 1.001), str(c * 0.999), str(c), "1", "1"]
            for i, c in enumerate(closes)
        ]


def test_skips_a_wide_swing_symbol_in_favour_of_the_next_ranked_one():
    # BADUSDT outranks GOODUSDT by volume but its swing is too wide to clear
    # the minimum; GOODUSDT should be selected in its place rather than the
    # list simply coming up one short.
    bitget = FakeBitgetWithSwings(
        tickers=[_ticker("BADUSDT", 300), _ticker("GOODUSDT", 200), _ticker("OKUSDT", 100)],
        swings={"BADUSDT": 0.9, "GOODUSDT": 0.1, "OKUSDT": 0.1},
    )

    selected = executable_symbols_by_volume(bitget, top=2, equity=100.0)

    assert "BADUSDT" not in selected
    assert selected == ["GOODUSDT", "OKUSDT"]


def test_still_ranks_by_volume_among_qualifiers():
    bitget = FakeBitgetWithSwings(
        tickers=[_ticker("FIRSTUSDT", 200), _ticker("SECONDUSDT", 100)],
        swings={"FIRSTUSDT": 0.1, "SECONDUSDT": 0.1},
    )

    assert executable_symbols_by_volume(bitget, top=2, equity=100.0) == ["FIRSTUSDT", "SECONDUSDT"]
