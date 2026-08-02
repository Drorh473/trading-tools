import pandas as pd

from notifier.strategies.volume_run import VolumeRun, find_consolidation


def _bars(closes: list[float], volumes: list[float] | None = None, freq: str = "D") -> pd.DataFrame:
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq=freq),
            "open": s,
            "high": s * 1.005,
            "low": s * 0.995,
            "close": s,
            "base_vol": pd.Series(volumes if volumes is not None else [1.0] * len(s), dtype=float),
            "quote_vol": 1.0,
        }
    )


def _leg(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


def daily_setup() -> pd.DataFrame:
    """A long rally, a pivot high printed on a volume spike, a pivot low on
    raised volume, then volume drying up while price coils between them."""
    closes = _leg(50, 150, 240) + _leg(150, 163, 6) + _leg(163, 151, 6) + _leg(151, 158, 6)
    daily = _bars(closes)
    daily.loc[245, "high"] = 164.0
    daily.loc[245, "base_vol"] = 8.0  # the spike that made the level
    daily.loc[251, "low"] = 149.0
    daily.loc[251, "base_vol"] = 3.0
    daily.loc[252:, "base_vol"] = 0.3  # volume falls away inside the range
    return daily


def entry_bars(last_close: float, count: int = 21) -> pd.DataFrame:
    return _bars([160.0] * (count - 1) + [last_close], freq="h")


def test_finds_the_consolidation():
    setup = find_consolidation(daily_setup())

    assert setup is not None
    assert setup.top == 164.0
    assert setup.bottom == 149.0


def test_no_consolidation_without_a_volume_spike_at_the_top():
    daily = daily_setup()
    daily.loc[245, "base_vol"] = 1.0  # the level was never defended

    assert find_consolidation(daily) is None


def test_no_consolidation_when_volume_is_not_drying_up():
    daily = daily_setup()
    daily.loc[252:, "base_vol"] = 5.0  # volume rising inside the range instead

    assert find_consolidation(daily) is None


def test_no_consolidation_without_an_uptrend():
    closes = _leg(200, 60, 240) + _leg(60, 66, 6) + _leg(66, 61, 6) + _leg(61, 64, 6)
    daily = _bars(closes)
    daily.loc[245, "base_vol"] = 8.0
    daily.loc[251, "base_vol"] = 3.0
    daily.loc[252:, "base_vol"] = 0.3

    assert find_consolidation(daily) is None


def test_no_consolidation_when_the_range_is_far_too_wide():
    # Bracketing pivots alone say nothing about distance: on live data a
    # 17-ATR span got called a consolidation purely because price sat between
    # two distant levels.
    daily = daily_setup()
    daily.loc[245, "high"] = 400.0

    assert find_consolidation(daily) is None


def test_fires_on_a_breakout_above_the_range():
    signal = VolumeRun("1D", "1H").evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(164.5)})

    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 3 1D/1H"
    assert signal.entry_price == 164.5  # market at the breakout close
    assert signal.stop_loss < signal.entry_price
    assert signal.reward_risk_ratio == 2.0
    assert signal.partial_fraction == 0.75


def test_no_signal_while_price_stays_inside_the_range():
    assert VolumeRun("1D", "1H").evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(162.0)}) is None


def test_only_the_first_close_above_the_range_fires():
    # Every later candle is also above the level; re-firing on each is how one
    # stale TSLAUSDT short went out four times in eleven hours.
    already_broken = _bars([160.0] * 18 + [165.0, 166.0, 167.0], freq="h")

    assert VolumeRun("1D", "1H").evaluate("TESTUSDT", {"1D": daily_setup(), "1H": already_broken}) is None


def test_at_all_time_highs_the_runner_has_no_price_and_trails_instead():
    signal = VolumeRun("1D", "1H").evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(164.5)})

    assert signal.remainder_target is None  # nothing overhead to exit into
    assert "3 trading days" in signal.remainder_note
    assert any("trail" in note.lower() for note in signal.extra_notes)


def test_resistance_between_the_break_and_the_target_rejects_the_trade():
    # An old high sitting just above the breakout is what stops price reaching
    # a 1:2 target, so the setup is not worth taking.
    daily = daily_setup()
    daily.loc[100, "high"] = 166.0  # a prior peak just overhead
    daily.loc[100, "base_vol"] = 9.0

    signal = VolumeRun("1D", "1H").evaluate("TESTUSDT", {"1D": daily, "1H": entry_bars(164.5)})

    assert signal is None


def test_the_day_version_has_no_three_day_clock():
    swing = VolumeRun("1D", "1H", time_exit_days=3)
    day = VolumeRun("1D", "5m", time_exit_days=None, armed_only=True)

    assert day.armed_timeframes == ("5m",)
    assert swing.armed_timeframes == ()
    assert day.tag == "Strategy 3 1D/5m"

    signal = day.evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(164.5)})
    assert signal is not None
    assert "trading days" not in signal.remainder_note


def test_arms_only_when_price_presses_the_top_of_the_range():
    day = VolumeRun("1D", "5m", armed_only=True)
    daily = daily_setup()

    # close sits at 158 in a 149-164 range -> 60% up, nowhere near the top tenth
    assert day.arms("TESTUSDT", {"1D": daily}) is False

    pressing = daily.copy()
    pressing.loc[pressing.index[-1], "close"] = 163.5  # ~97% of the way up
    assert day.arms("TESTUSDT", {"1D": pressing}) is True
