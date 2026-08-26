"""The partial-candle builder both offline generators use to see exactly what
the live scanner would see mid-candle - both files' docstrings call this
"where the lookahead lives". A bug here (wrong bound, wrong aggregation)
would silently corrupt every backtest number built from it, with no error to
surface it.
"""

import pandas as pd
import pytest

SWEPT_V2_CONSTANTS = (
    "EMA9_HOLD_BARS", "MIN_STOP_PCT", "MIN_NET_REWARD_RISK",
    "MIN_PIVOT_SPAN_BARS", "MIN_SWING_DRIFT_ATR", "MAX_EMA9_CROSSINGS",
    "REQUIRE_STRUCTURE_TREND",
)


@pytest.fixture(autouse=True)
def _restore_v2_thresholds():
    """Importing either generator module below zeroes out
    notifier.strategies.ema_trend_v2's swept thresholds as an import-time side
    effect (see backtest/generate_v2.py's module docstring), and that mutation
    outlives the import. Restore afterward so this file doesn't leave every
    later strategy test running against a strategy with no thresholds - the
    exact hazard test_both_generators_disable_the_same_thresholds
    (tests/test_score.py) exists to catch, reproduced here because this file
    also imports the generators.
    """
    import notifier.strategies.ema_trend_v2 as v2

    before = {name: getattr(v2, name) for name in SWEPT_V2_CONSTANTS}
    yield
    for name, value in before.items():
        setattr(v2, name, value)


def _spine():
    return pd.DataFrame(
        {
            "ts": [1, 2, 3, 4, 5],
            "open": [10.0, 20.0, 30.0, 40.0, 50.0],
            "high": [15.0, 25.0, 35.0, 45.0, 55.0],
            "low": [9.0, 19.0, 5.0, 39.0, 49.0],
            "close": [12.0, 22.0, 32.0, 42.0, 52.0],
            "base_vol": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )


def _assert_aggregates_only_the_window(forming_row):
    row = forming_row(_spine(), 1, 3)  # rows at index 1, 2, 3 (ts 2, 3, 4)

    assert row["ts"] == 2, "labelled by the window's FIRST bar, not the whole frame's"
    assert row["open"] == 20.0, "open is the window's first open, not the earlier bar's"
    assert row["high"] == 45.0, "high must not reach past the window into bar 5's 55.0"
    assert row["low"] == 5.0, "low must not reach past the window into bar 1's 9.0"
    assert row["close"] == 42.0, "close is the window's LAST close, mid-candle so far"
    assert row["base_vol"] == 9.0, "volume sums only rows 2-4 (2+3+4), not the whole frame"


def test_generate_15m_forming_row_aggregates_only_its_own_window():
    from backtest.generate_15m import _forming_row

    _assert_aggregates_only_the_window(_forming_row)


def test_generate_v2_forming_row_aggregates_only_its_own_window():
    from backtest.generate_v2 import _forming_row

    _assert_aggregates_only_the_window(_forming_row)
