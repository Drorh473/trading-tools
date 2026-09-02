import pandas as pd
import pytest

from notifier.strategies.base import Signal, Strategy
from tools.why import fetch_bars, find_strategy, format_report, trim_to


class _Strategy(Strategy):
    def __init__(self, tag, tags=(), timeframes=None):
        self.tag = tag
        self.tags = tags
        self.timeframes = timeframes or ["1H"]

    def evaluate(self, symbol, bars_by_timeframe):
        return None


def _bars(n=10, start_ts="2026-01-01"):
    return pd.DataFrame(
        {
            "ts": pd.date_range(start_ts, periods=n, freq="h"),
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.0] * n,
            "base_vol": [1.0] * n,
            "quote_vol": [1.0] * n,
        }
    )


# ---------------------------------------------------------------------------
# find_strategy
# ---------------------------------------------------------------------------


def test_find_strategy_matches_a_unique_substring_case_insensitively():
    strategies = [_Strategy("Strategy 1 1H"), _Strategy("Strategy 3 1D/1H")]

    assert find_strategy("strategy 1", strategies).tag == "Strategy 1 1H"


def test_find_strategy_matches_against_any_declared_tag_not_just_the_primary_one():
    strategies = [_Strategy("Strategy 4 1H OB2.0", tags=("Strategy 4 1H OB1.0", "Strategy 4 1H OB2.0"))]

    assert find_strategy("OB1.0", strategies) is strategies[0]


def test_find_strategy_raises_with_the_available_list_when_nothing_matches():
    strategies = [_Strategy("Strategy 1 1H")]

    with pytest.raises(ValueError, match="Strategy 1 1H"):
        find_strategy("nonexistent", strategies)


def test_find_strategy_raises_when_more_than_one_matches():
    strategies = [_Strategy("Strategy 2.1 1H"), _Strategy("Strategy 2.1 4H")]

    with pytest.raises(ValueError, match="more than one"):
        find_strategy("Strategy 2.1", strategies)


# ---------------------------------------------------------------------------
# trim_to
# ---------------------------------------------------------------------------


def test_trim_to_none_returns_the_frame_unchanged():
    bars = _bars()
    assert trim_to(bars, None) is bars


def test_trim_to_a_timestamp_drops_everything_after_it():
    bars = _bars(n=10)
    at = pd.Timestamp("2026-01-01 05:00:00")

    trimmed = trim_to(bars, at)

    assert len(trimmed) == 6  # hours 00:00 through 05:00 inclusive
    assert trimmed["ts"].max() == at


def test_trim_to_resets_the_index_so_downstream_position_math_stays_0_based():
    bars = _bars(n=10)
    at = pd.Timestamp("2026-01-01 05:00:00")

    trimmed = trim_to(bars, at)

    assert list(trimmed.index) == list(range(len(trimmed)))


# ---------------------------------------------------------------------------
# fetch_bars
# ---------------------------------------------------------------------------


class _FakeBitget:
    def __init__(self, candles_by_symbol):
        self._candles = candles_by_symbol

    def get_candles(self, symbol, granularity="1H", limit=100, closed_only=True):
        return self._candles[(symbol, granularity)]


def _candles(n=5):
    return [[str(1000 * (i + 1)), "100", "101", "99", "100", "1", "1"] for i in range(n)]


def test_fetch_bars_fetches_every_declared_timeframe():
    strategy = _Strategy("Strategy 1 1H", timeframes=["1H"])
    bitget = _FakeBitget({("BTCUSDT", "1H"): _candles()})

    bars_by_tf = fetch_bars(bitget, "BTCUSDT", strategy)

    assert "1H" in bars_by_tf
    assert len(bars_by_tf["1H"]) == 5


def test_fetch_bars_resolves_a_cross_symbol_reference_timeframe():
    """A "SYMBOL@TF" declared timeframe (Strategy 1's market_trend_symbol,
    Strategy 4's dealing-range reference) fetches THAT symbol's own bars, not
    the one being explained - matching scanner._split_reference_key."""
    strategy = _Strategy("Strategy 1 1H", timeframes=["1H", "BTCUSDT@1D"])
    bitget = _FakeBitget({
        ("ETHUSDT", "1H"): _candles(),
        ("BTCUSDT", "1D"): _candles(n=3),
    })

    bars_by_tf = fetch_bars(bitget, "ETHUSDT", strategy)

    assert len(bars_by_tf["1H"]) == 5
    assert len(bars_by_tf["BTCUSDT@1D"]) == 3


def test_fetch_bars_trims_every_timeframe_to_the_given_timestamp():
    strategy = _Strategy("Strategy 1 1H", timeframes=["1H"])
    bitget = _FakeBitget({("BTCUSDT", "1H"): _candles(n=5)})
    # candle timestamps are 1000ms, 2000ms, ... 5000ms
    at = pd.Timestamp(3000, unit="ms")

    bars_by_tf = fetch_bars(bitget, "BTCUSDT", strategy, at=at)

    assert len(bars_by_tf["1H"]) == 3


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_shows_fired_and_each_check():
    from notifier.strategies.base import ExplainResult, RuleCheck

    result = ExplainResult(
        fired=True,
        signal=Signal(symbol="BTCUSDT", direction="long", entry_price=100.0, stop_loss=90.0, strategy_tag="t"),
        checks=(RuleCheck("consolidation_found", True, "box 90-110"),),
    )

    text = format_report("BTCUSDT", _Strategy("Strategy 3 1D/1H"), None, result)

    assert "FIRED" in text
    assert "consolidation_found" in text
    assert "box 90-110" in text


def test_format_report_shows_did_not_fire_and_the_failing_check():
    from notifier.strategies.base import ExplainResult, RuleCheck

    result = ExplainResult(
        fired=False, signal=None,
        checks=(RuleCheck("consolidation_found", False, "no qualifying box"),),
    )

    text = format_report("BTCUSDT", _Strategy("Strategy 3 1D/1H"), None, result)

    assert "DID NOT FIRE" in text
    assert "no qualifying box" in text


def test_format_report_renders_the_funnel_sorted_by_count_descending():
    from notifier.strategies.base import ExplainResult

    result = ExplainResult(
        fired=False, signal=None, checks=(),
        funnel={"rule_a": 3, "rule_b": 10, "rule_c": 1},
    )

    text = format_report("BTCUSDT", _Strategy("Strategy 3 1D/1H"), None, result)

    a, b, c = text.index("rule_a"), text.index("rule_b"), text.index("rule_c")
    assert b < a < c  # rule_b (10) first, then rule_a (3), then rule_c (1)
