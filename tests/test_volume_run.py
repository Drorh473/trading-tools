import pandas as pd

import pytest

from notifier.strategies.indicators import atr
from notifier.strategies.volume_run import (
    DAILY_PARAMS,
    HOURLY_PARAMS,
    ConsolidationParams,
    VolumeRun,
    _dominant_pivots,
    find_consolidation,
)


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
    """An upward IMPULSE that leaves a wick, then a pause that dries up.

    Strategy 3 only goes long, so the range has to follow a move up - Dror, on
    BNBUSDT and SOLUSDT: "this method is only for long so it must be after a
    big move up not down". Bar 245 is that move: a 7.5-point span on a ~2.1 ATR
    with more than half of it upper wick, on a volume spike. The wick top is
    the level the trade later breaks.

    The floor is the FIRST LOW after it (bar 247), not the nearest pivot low
    anywhere below price - "it should count from the first low so the 3 candle
    after the big one". Volume then falls away across the coil's own bars.

    This fixture used to be a rally with a manually-raised high and no impulse
    at all, paired with a pivot low six bars later; it passed every check while
    describing nothing the strategy is about.
    """
    closes = [50 + 100 * (i + 1) / 240 for i in range(240)] + [151, 153, 155, 156, 157]
    closes += [160, 155.5, 153.5, 155, 156.5, 155, 157, 155.5, 156.8, 156, 157.2, 156.4, 157.5]
    closes += [156, 157.3, 155.8, 156.9, 156.2, 157.4, 155.9, 157.1, 156.3, 157.2, 156.5, 157.3]
    daily = _bars(closes)
    daily.loc[245, ["open", "high", "low", "close"]] = [157.0, 164.0, 156.5, 160.0]
    daily.loc[245, "base_vol"] = 8.0  # the spike that made the level
    daily.loc[246, ["high", "low"]] = [160.0, 155.0]
    daily.loc[247, ["high", "low"]] = [156.0, 153.0]  # the first low: the range floor
    daily.loc[247, "base_vol"] = 3.0
    daily.loc[248:258, "base_vol"] = 1.0  # busy early in the coil...
    daily.loc[259:, "base_vol"] = 0.3     # ...and quiet later: a real dry-up
    return daily


def pressing_setup() -> pd.DataFrame:
    """The same setup with price pressing the level, which is what arming
    looks for: 163.2 in a 153-164 range is ~93% of the way up."""
    daily = daily_setup()
    daily.loc[266:, "close"] = [161.0, 162.0, 161.5, 163.2]
    daily.loc[266:, "high"] = [162.0, 163.0, 162.5, 163.6]
    daily.loc[266:, "low"] = [160.0, 161.0, 161.0, 162.5]
    return daily


def entry_bars(last_close: float, count: int = 21) -> pd.DataFrame:
    return _bars([160.0] * (count - 1) + [last_close], freq="h")


def _flat_series(n: int, default: float, overrides: dict) -> pd.Series:
    s = pd.Series([default] * n, dtype=float)
    for i, v in overrides.items():
        s.iloc[i] = v
    return s


def test_dominance_overrides_recency():
    """The core fix. A smaller, more RECENT candidate must not beat an older
    one that dominates on body, wick, and volume together - that is exactly
    what silently anchored the level on a bar the market never treated as
    resistance."""
    n = 50
    highs = _flat_series(n, 100.0, {10: 125.0, 40: 104.0})
    opens = _flat_series(n, 100.0, {10: 100.0, 40: 100.0})
    closes = _flat_series(n, 100.0, {10: 110.0, 40: 102.0})  # body: 10 vs 2
    volumes = _flat_series(n, 1.0, {10: 10.0, 40: 2.0})  # ratio: 10x vs 2x
    a = pd.Series([1.0] * n)

    tier1, tier2 = _dominant_pivots(highs, opens, closes, volumes, [10, 40], a, baseline_bars=5)

    assert tier1 == 10, "the dominant, older bar must win - not bar 40 for being newer"
    assert tier2 == 40, "the excluded-tier1 fallback is the genuinely different remaining bar"


def test_tier_two_when_body_disagrees_with_wick_and_volume():
    n = 50
    highs = _flat_series(n, 100.0, {10: 142.0, 20: 130.0})
    opens = _flat_series(n, 100.0, {10: 100.0, 20: 100.0})
    closes = _flat_series(n, 100.0, {10: 140.0, 20: 105.0})  # bar 10 has the huge body
    volumes = _flat_series(n, 1.0, {10: 2.0, 20: 15.0})  # bar 20 has the huge volume
    a = pd.Series([1.0] * n)
    # bar 20 also wins wick: 130-105=25 vs bar 10's 142-140=2

    tier1, tier2 = _dominant_pivots(highs, opens, closes, volumes, [10, 20], a, baseline_bars=5)

    assert tier1 is None, "body (bar 10) disagrees with wick+volume (bar 20) - no 3-way consensus"
    assert tier2 == 20, "wick and volume alone agree on bar 20"


def test_no_signal_when_all_three_disagree():
    n = 50
    highs = _flat_series(n, 100.0, {10: 140.0, 20: 130.0, 30: 120.0})
    opens = _flat_series(n, 100.0, {10: 100.0, 20: 100.0, 30: 100.0})
    closes = _flat_series(n, 100.0, {10: 138.0, 20: 105.0, 30: 108.0})  # body winner: 10
    volumes = _flat_series(n, 1.0, {10: 2.0, 20: 3.0, 30: 20.0})  # volume winner: 30
    a = pd.Series([1.0] * n)
    # wick: bar10=138-138=0(body=close so wick~0), bar20=130-105=25, bar30=120-108=12 -> wick winner: 20

    tier1, tier2 = _dominant_pivots(highs, opens, closes, volumes, [10, 20, 30], a, baseline_bars=5)

    assert tier1 is None
    assert tier2 is None, "wick winner (20) and volume winner (30) do not agree either"


def test_a_too_wide_dominant_pairing_falls_through_to_the_next_candidate():
    """A dominant impulse whose own first low leaves too wide a range must not
    kill the setup - a nearer, still qualifying impulse should be tried.

    The two are spaced well apart deliberately. An impulse has to out-top the
    preceding RALLY_LOOKBACK bars AND have a real rally behind it, so a second
    one inside that window can never qualify while the first one's high is
    still in view. The dip to 144 before the second impulse is what gives it
    that rally; without it the rise into 164 measures 3.0 ATR and is correctly
    refused.
    """
    closes = [50 + 100 * (i + 1) / 240 for i in range(240)] + [151, 153, 155, 156, 158]
    closes += [172, 160, 152] + [150, 148, 146, 145, 144, 147, 150, 153, 156, 158]
    closes += [159, 158, 159] + [160]
    closes += [154, 151.5, 154, 152, 153.5, 152.5, 154, 152, 153.5, 153]
    closes += [154, 152.2, 153.8, 152.6, 153.9, 152.4, 154.1, 152.8, 153.6, 153.1, 153.9, 152.9]
    daily = _bars(closes)

    # The dominant, OLDER impulse: its own first low leaves a 26-point range.
    daily.loc[245, ["open", "high", "low", "close"]] = [158.0, 176.0, 157.0, 172.0]
    daily.loc[245, "base_vol"] = 20.0
    daily.loc[246, ["high", "low"]] = [172.0, 158.0]
    daily.loc[247, ["high", "low"]] = [160.0, 140.0]  # dips below its own rally start
    daily.loc[247, "base_vol"] = 3.0

    # The nearer impulse, with its own rally up from 144.
    daily.loc[261, ["open", "high", "low", "close"]] = [157.0, 164.0, 156.0, 160.0]
    daily.loc[261, "base_vol"] = 8.0
    daily.loc[262, ["high", "low"]] = [158.0, 153.0]
    daily.loc[263, ["high", "low"]] = [154.0, 150.0]
    daily.loc[263, "base_vol"] = 3.0
    daily.loc[264:274, "base_vol"] = 1.0
    daily.loc[275:, "base_vol"] = 0.3

    setup = find_consolidation(daily, DAILY_PARAMS)

    assert setup is not None, "must fall through to the later impulse, not return None"
    assert setup.top == 164.0
    assert setup.top_index == 261


def test_a_level_retested_after_forming_is_disqualified():
    """The INTCUSDT case: a dominant candle sets the level, but the market
    pokes back up to it a few bars later - no genuine convergence ever
    happened, just a spike and an immediate retest."""
    control = find_consolidation(daily_setup(), DAILY_PARAMS)
    assert control is not None and control.top == 164.0, "control: the clean setup is found"

    poked = daily_setup()
    poked.loc[250, "high"] = 164.5  # back above the level before any coil formed

    assert find_consolidation(poked, DAILY_PARAMS) is None


def test_a_dip_that_erases_the_rally_is_not_a_consolidation():
    """There is no cap on the SPAN or on how long the coil runs - Dror: "there
    is no limit to the width the opposite the longer the consolidation the
    better". What is capped is the DIP: a pullback that gives the whole rally
    back has left nothing to break out of.
    """
    assert find_consolidation(daily_setup(), DAILY_PARAMS) is not None, "control"

    deep = daily_setup()
    deep.loc[247, "low"] = 145.0  # below where the rally into 164 began (146.76)

    assert find_consolidation(deep, DAILY_PARAMS) is None


def test_a_coil_that_trends_is_not_a_coil():
    """"the consolidation is measured that there is no trend not up or down".
    A steady drift fits a straight line; chop does not."""
    trending = daily_setup()
    trending.loc[248:, "close"] = [153.0 + 0.45 * i for i in range(len(trending) - 248)]

    assert find_consolidation(trending, DAILY_PARAMS) is None


def test_finds_the_consolidation():
    setup = find_consolidation(daily_setup())

    assert setup is not None
    assert setup.top == 164.0
    assert setup.bottom == 153.0


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
    daily.loc[252:254, "base_vol"] = 1.0
    daily.loc[255:, "base_vol"] = 0.3  # a real dry-up inside the range

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
    day = VolumeRun("1H", "5m", time_exit_days=None, armed_only=True, params=HOURLY_PARAMS)

    assert day.armed_timeframes == ("5m",)
    assert swing.armed_timeframes == ()
    assert day.tag == "Strategy 3 1H/5m"
    assert day.params is HOURLY_PARAMS

    # The algorithm is scale-agnostic - the same shape of consolidation reads
    # the same whether the "structure" bars are daily or hourly, only the
    # tuning (here HOURLY_PARAMS) differs.
    signal = day.evaluate("TESTUSDT", {"1H": daily_setup(), "5m": entry_bars(164.5)})
    assert signal is not None
    assert "trading days" not in signal.remainder_note


def test_arms_only_when_price_presses_the_top_of_the_range():
    day = VolumeRun("1H", "5m", armed_only=True, params=HOURLY_PARAMS)
    daily = daily_setup()

    # close sits at 158 in a 149-164 range -> 60% up, nowhere near the top tenth
    assert day.arms("TESTUSDT", {"1H": daily}) is False

    pressing = daily.copy()
    pressing.loc[pressing.index[-1], "close"] = 163.5  # ~97% of the way up
    assert day.arms("TESTUSDT", {"1H": pressing}) is True


def test_find_consolidation_defaults_to_daily_params():
    assert find_consolidation(daily_setup()) == find_consolidation(daily_setup(), DAILY_PARAMS)


def test_find_consolidation_actually_uses_the_given_params():
    # An absurdly strict pivot threshold means no reversal in the fixture can
    # ever confirm a pivot at all - proof the params passed in are what
    # zigzag_pivots is actually thresholded on, not a decorative argument.
    impossible = ConsolidationParams(
        pivot_atr_multiple=1000.0,
        volume_baseline_bars=DAILY_PARAMS.volume_baseline_bars,
        volume_spike_multiple=DAILY_PARAMS.volume_spike_multiple,
        volume_increase_multiple=DAILY_PARAMS.volume_increase_multiple,
        volume_decline_max=DAILY_PARAMS.volume_decline_max,
        min_consolidation_bars=DAILY_PARAMS.min_consolidation_bars,
        max_range_atr=DAILY_PARAMS.max_range_atr,
        zigzag_lookback=DAILY_PARAMS.zigzag_lookback,
    )
    assert find_consolidation(daily_setup(), DAILY_PARAMS) is not None
    assert find_consolidation(daily_setup(), impossible) is None


# ---- the rebuild: penetration, absolute width, range-keyed dedupe ----


def test_a_graze_past_the_range_top_is_not_a_breakout():
    # TSLAUSDT closed 0.012% past the line - four cents on a $324 stock - and
    # that counted, because the line is the pivot bar's own high and merely
    # touching it qualified. A break now has to clear it by a margin.
    day = VolumeRun("1H", "5m", params=HOURLY_PARAMS)

    grazed = day.evaluate("TESTUSDT", {"1H": daily_setup(), "5m": entry_bars(164.1)})
    assert grazed is None, "a close a fraction above the top is not a breakout"

    cleared = day.evaluate("TESTUSDT", {"1H": daily_setup(), "5m": entry_bars(164.5)})
    assert cleared is not None, "a genuine break must still fire"


def test_the_signal_is_deduped_on_the_range_not_the_entry_price():
    # TSLAUSDT alerted twice ten minutes apart off an IDENTICAL range, because
    # the default dedupe key includes the entry price and the two 5m closes
    # differed by two cents. Keying on the level claims the range once.
    setup = find_consolidation(daily_setup())
    day = VolumeRun("1D", "1H")

    first = day.evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(164.5)})
    second = day.evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(164.9)})

    assert first.entry_price != second.entry_price  # different closes...
    assert first.dedupe_key == second.dedupe_key  # ...but the same trade
    assert first.dedupe_key[-1] == round(setup.top, 10)


def test_only_the_intraday_instance_is_session_gated():
    swing = VolumeRun("1D", "1H", time_exit_days=3)
    intraday = VolumeRun("1H", "5m", armed_only=True, params=HOURLY_PARAMS, session_gated=True)

    # A daily bar spans a whole session, so the question does not arise for it.
    assert swing.session_gated is False
    assert intraday.session_gated is True


def test_the_stop_sits_under_the_recent_low_not_under_the_breakout_bar():
    """Dror: "the stop should be the recent low before the breakout".

    The breakout candle's own low is wherever that one bar happened to open
    from and says nothing about where the move would be wrong. The last low
    the market actually turned at does. Here the breakout bar's low is 158
    while the swing low three bars earlier is 155, so a stop derived from the
    breakout bar would sit far too tight.
    """
    day = VolumeRun("1H", "5m", time_exit_days=None, params=HOURLY_PARAMS)
    daily = daily_setup()
    entry = _bars([160.0] * 14 + [159, 155, 157, 159, 160, 158, 166.0], freq="h")
    entry.loc[15, "low"] = 155.0   # the recent swing low the market turned at
    entry.loc[20, "low"] = 158.0   # the breakout bar's own, much higher, low

    signal = day.evaluate("TESTUSDT", {"1H": daily, "5m": entry})

    assert signal is not None
    # Anchored on 155, not on the 158 of the bar that broke out.
    assert signal.stop_loss < 155.0, f"stop {signal.stop_loss} is not under the recent low"
    assert signal.stop_loss > 150.0, "and not miles below it either"
