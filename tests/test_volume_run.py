import pandas as pd

import pytest

from notifier.strategies.indicators import atr
from notifier.strategies.volume_run import (
    DAY_PARAMS,
    DAY_PARTIAL_FRACTION,
    STOP_AT_RECENT_LOW,
    SWING_PARAMS,
    ConsolidationParams,
    VolumeRun,
    _dominant_pivots,
    find_consolidation,
)


def swing() -> VolumeRun:
    """The 1D/1H instance as main.py builds it."""
    return VolumeRun("1D", "1H", time_exit_days=3)


def day() -> VolumeRun:
    """The 1D/5m instance as main.py builds it. Same daily consolidation as
    the swing version - the cheatsheet identifies it on the daily chart for
    both - with a faster trigger and a flat exit at 1:2."""
    return VolumeRun(
        "1D", "5m",
        time_exit_days=None,
        armed_only=True,
        params=DAY_PARAMS,
        session_gated=True,
        partial_fraction=DAY_PARTIAL_FRACTION,
        stop_anchor=STOP_AT_RECENT_LOW,
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


def short_coil_setup() -> pd.DataFrame:
    """The same impulse and level, with the coil cut to FIVE bars.

    The day sheet allows a consolidation of "just a few single days"; the swing
    version's 20-bar floor does not. Truncating after bar 252 leaves 5 bars of
    pause, and the dry-up is compressed into that window so every other rule
    still passes on its own merits - the only thing separating the two
    versions here is min_consolidation_bars.
    """
    daily = daily_setup().iloc[:253].copy()
    daily.loc[248:250, "base_vol"] = 1.0  # busy early...
    daily.loc[251:252, "base_vol"] = 0.3  # ...quiet late, inside five bars
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

    setup = find_consolidation(daily, SWING_PARAMS)

    assert setup is not None, "must fall through to the later impulse, not return None"
    assert setup.top == 164.0
    assert setup.top_index == 261


def test_a_level_retested_after_forming_is_disqualified():
    """The INTCUSDT case: a dominant candle sets the level, but the market
    pokes back up to it a few bars later - no genuine convergence ever
    happened, just a spike and an immediate retest."""
    control = find_consolidation(daily_setup(), SWING_PARAMS)
    assert control is not None and control.top == 164.0, "control: the clean setup is found"

    poked = daily_setup()
    poked.loc[250, "high"] = 164.5  # back above the level before any coil formed

    assert find_consolidation(poked, SWING_PARAMS) is None


def test_a_dip_that_erases_the_rally_is_not_a_consolidation():
    """There is no cap on the SPAN or on how long the coil runs - Dror: "there
    is no limit to the width the opposite the longer the consolidation the
    better". What is capped is the DIP: a pullback that gives the whole rally
    back has left nothing to break out of.
    """
    assert find_consolidation(daily_setup(), SWING_PARAMS) is not None, "control"

    deep = daily_setup()
    deep.loc[247, "low"] = 145.0  # below where the rally into 164 began (146.76)

    assert find_consolidation(deep, SWING_PARAMS) is None


def test_the_trend_gate_reads_price_against_sma200_and_nothing_else():
    """The gate now qualifies the IMPULSE candle, and that candle is often the
    move that starts the trend - so the EMA50-above-SMA200 half had to go. It
    is a lagging confirmation that cannot be true at the start of a move, and
    it rejected EPICUSDT (close 0.4654 > SMA200 0.4126, EMA50 only 0.3482).
    """
    from notifier.strategies.volume_run import in_uptrend_at

    closes = pd.Series([0.4654] * 3)
    sma200 = pd.Series([0.4126] * 3)
    assert in_uptrend_at(closes, sma200, 2) is True, "close above the SMA200 is the whole test"

    assert in_uptrend_at(pd.Series([0.30]), pd.Series([0.41]), 0) is False

    # An un-warmed MA cannot confirm anything, so it must not pass.
    assert in_uptrend_at(pd.Series([1.0]), pd.Series([float("nan")]), 0) is False
    assert in_uptrend_at(pd.Series([1.0]), pd.Series([0.5]), 9) is False, "out of range"


def test_the_weekly_average_only_fills_in_where_the_200_day_one_is_missing():
    """The coverage change must be provably non-regressive: on any bar with 200
    days behind it, the answer has to be exactly the 200-day average it always
    was. Only bars without one get the 10-week fallback.
    """
    from notifier.strategies.volume_run import TREND_MA_PERIOD, trend_levels
    from notifier.strategies.indicators import sma

    daily = daily_setup()
    levels = trend_levels(daily)
    daily_only = sma(daily["close"], TREND_MA_PERIOD)

    warmed = daily_only.notna()
    assert warmed.any() and (~warmed).any(), "fixture must span both regimes"
    pd.testing.assert_series_equal(
        levels[warmed], daily_only[warmed], check_names=False,
        obj="bars with a warmed 200-day average must be untouched",
    )
    # And the early bars, which used to have NO answer at all, now do.
    assert levels[~warmed].notna().any(), "the fallback must actually fill in"


def test_the_weekly_average_is_built_from_completed_weeks_only():
    """No lookahead: a bar's level may only use weeks that closed before it."""
    from notifier.strategies.volume_run import weekly_trend_levels

    daily = daily_setup()
    levels = weekly_trend_levels(daily, weeks=4)

    # Truncating the frame must not change the level on any bar that survives -
    # if a later week leaked in, the two would disagree.
    cut = 200
    truncated = weekly_trend_levels(daily.iloc[:cut].copy(), weeks=4)
    pd.testing.assert_series_equal(
        levels.iloc[:cut], truncated, check_names=False,
        obj="a bar's weekly level must not depend on bars after it",
    )


def test_the_history_floor_drops_now_that_the_200_day_average_is_optional():
    """The old 230-bar floor was only ever "warm up the SMA200", and it is why
    30 of the 100 watchlist symbols could never signal at all."""
    assert swing().min_daily_bars() == 127
    assert day().min_daily_bars() == 110
    assert swing().min_daily_bars() < 230 and day().min_daily_bars() < 230

    # A symbol with too little history is still refused rather than guessed at.
    short = daily_setup().iloc[-60:].reset_index(drop=True)
    assert swing().evaluate("TESTUSDT", {"1D": short, "1H": entry_bars(164.5)}) is None


def test_a_coil_that_climbs_steadily_is_not_a_coil():
    """An UPWARD drift is still a leg with a box drawn round it: price is
    running, not pausing. A steady climb fits a straight line; chop does not."""
    trending = daily_setup()
    trending.loc[248:, "close"] = [153.0 + 0.45 * i for i in range(len(trending) - 248)]

    assert find_consolidation(trending, SWING_PARAMS) is None


def test_a_coil_that_drifts_DOWN_is_still_a_coil():
    """Dror, revising the earlier "no trend not up or down": "inside the box
    can be a small downtrend or no trend at all".

    Giving part of the impulse back is what a pause looks like. Measured across
    every setup the detector finds, all but one drift down - and the old
    direction-blind rule was discarding the steadiest of them for sloping.
    """
    drifting = daily_setup()
    # The mirror image of the test above: same tight fit, downward, and ending
    # INSIDE the 153.0-164.0 box. The slope was 0.45/bar, which walked the last
    # close to 152.55 - through the floor and out of the range entirely. That
    # made the fixture assert something its own docstring does not claim:
    # price leaving the box was being accepted as a coil, which is the defect
    # that let Strategy 3 pair a live price with a range it had fallen out of.
    drifting.loc[248:, "close"] = [162.0 - 0.35 * i for i in range(len(drifting) - 248)]

    from notifier.strategies.volume_run import _coil_fit

    r_squared, slope = _coil_fit(drifting["close"].iloc[248:])
    assert slope < 0 and r_squared > 0.5, "fixture must be a tight DOWNWARD fit"

    assert find_consolidation(drifting, SWING_PARAMS) is not None, (
        "a steady downward drift inside the box must be accepted"
    )


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


def test_both_versions_read_their_consolidation_off_the_daily_chart():
    """The refactor. Both cheatsheets identify the consolidation on the DAILY
    chart and differ only in the trigger; the day version used to read its
    whole structure - range, spike, dry-up, resistance - off hourly bars.
    """
    assert swing().trend_timeframe == "1D"
    assert day().trend_timeframe == "1D"
    assert day().entry_timeframe == "5m"
    assert day().tag == "Strategy 3 1D/5m"
    assert day().armed_timeframes == ("5m",)
    assert swing().armed_timeframes == ()

    # Same daily fixture, same range, both fire off it.
    assert day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(164.5)}) is not None
    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(164.5)}) is not None


def test_the_day_version_exits_flat_at_one_to_two_with_no_runner():
    """The day sheet names one exit - "profit at a 1:2 ratio" - and nothing
    after it. The instance used to take 75% and open a runner it then had no
    rule to close: time_exit_days was None, so the note read "at your
    discretion" and the remainder just sat there.
    """
    signal = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(164.5)})

    assert signal is not None
    assert signal.reward_risk_ratio == 2.0
    assert signal.partial_fraction == 1.0, "the whole position leaves at the target"
    assert signal.remainder_target is None
    assert signal.remainder_note == "", "there is no remainder to describe"
    assert not any("runner" in note.lower() for note in signal.extra_notes)
    # The trailing rule survives: both sheets end on it, and a flat 1:2 exit
    # still wants its stop dragged up on the way there.
    assert any("trail" in note.lower() for note in signal.extra_notes)


def test_the_swing_version_still_takes_seventy_five_percent_and_runs_the_rest():
    signal = swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(164.5)})

    assert signal.partial_fraction == 0.75
    assert "3 trading days" in signal.remainder_note


def test_the_day_version_accepts_a_coil_of_only_a_few_days():
    """"The consolidation can be just a few single days" - the day sheet, in
    as many words. The 20-bar floor came from Dror's FIGHTUSDT call and the
    swing sheet is silent on the question, so it stays there; applying it to
    the day version would make the sheet's own allowance unreachable.
    """
    short = short_coil_setup()

    assert find_consolidation(short, SWING_PARAMS) is None, "5 bars is under the swing floor of 20"
    assert find_consolidation(short, DAY_PARAMS) is not None, "but it is what the day sheet allows"

    assert day().evaluate("TESTUSDT", {"1D": short, "5m": entry_bars(164.5)}) is not None
    assert swing().evaluate("TESTUSDT", {"1D": short, "1H": entry_bars(164.5)}) is None


def test_arms_wherever_a_live_consolidation_exists():
    """The 0.10 band was removed on measurement, not preference.

    It required price within a tenth of the range top. Across 62,353 daily bars
    spanning 2021-2026 - 313 small-cap coins and 195 majors - that armed ONCE,
    and the small caps never got closer than 0.79 of the way up. Worse, it was
    filtering on the wrong thing: on the eight bars actually followed by a
    break, position ran from 0.153 to 0.770, median 0.612. Price jumps from a
    standing start rather than creeping up to the level, so the band caught 0
    of 8 breaks while "no band" caught all eight.

    A qualifying consolidation is itself rare enough to be the filter - 1.9% of
    small-cap symbol-days - which keeps the 5m poll affordable.
    """
    instance = day()
    daily = daily_setup()

    # 158 in a 149-164 range: 60% up, which the old band refused and the break
    # data says is squarely where breaks come from.
    assert instance.arms("TESTUSDT", {"1D": daily}) is True

    pressing = daily.copy()
    pressing.loc[pressing.index[-1], "close"] = 163.5  # ~97% of the way up
    assert instance.arms("TESTUSDT", {"1D": pressing}) is True


def test_arming_still_refuses_a_symbol_with_no_consolidation():
    """Removing the band must not arm the whole watchlist: that is 100 symbols
    x 4 timeframes x 288 polls = 115,200 fetches a day against the bot's ~3,100,
    on an API that answers bursts with 429. The consolidation requirement is
    what keeps it to ~2 symbols a day."""
    instance = day()
    flat = daily_setup()
    # No impulse, no level, no range - just a line.
    flat["close"] = 100.0
    flat["high"] = 100.5
    flat["low"] = 99.5
    flat["open"] = 100.0

    assert find_consolidation(flat, DAY_PARAMS) is None
    assert instance.arms("TESTUSDT", {"1D": flat}) is False


def test_find_consolidation_defaults_to_daily_params():
    assert find_consolidation(daily_setup()) == find_consolidation(daily_setup(), SWING_PARAMS)


def test_find_consolidation_actually_uses_the_given_params():
    # An absurdly strict pivot threshold means no reversal in the fixture can
    # ever confirm a pivot at all - proof the params passed in are what
    # zigzag_pivots is actually thresholded on, not a decorative argument.
    impossible = ConsolidationParams(
        pivot_atr_multiple=1000.0,
        volume_baseline_bars=SWING_PARAMS.volume_baseline_bars,
        volume_spike_multiple=SWING_PARAMS.volume_spike_multiple,
        volume_increase_multiple=SWING_PARAMS.volume_increase_multiple,
        volume_decline_max=SWING_PARAMS.volume_decline_max,
        min_consolidation_bars=SWING_PARAMS.min_consolidation_bars,
        max_range_atr=SWING_PARAMS.max_range_atr,
        max_range_pct=SWING_PARAMS.max_range_pct,
        zigzag_lookback=SWING_PARAMS.zigzag_lookback,
    )
    assert find_consolidation(daily_setup(), SWING_PARAMS) is not None
    assert find_consolidation(daily_setup(), impossible) is None


def test_the_width_cap_is_actually_enforced():
    """max_range_atr and max_range_pct were both declared, documented at
    length, set on every params object - and read by nothing. The rule they
    described did not exist, which is how BANKUSDT's 199.9%-wide, 20.5-ATR
    "consolidation" produced a live signal on 2026-07-19.

    So this asserts the FIELD changes the ANSWER, not merely that some wide
    range is refused: a cap below the fixture's own span must reject it while
    everything else stays identical.
    """
    setup = find_consolidation(daily_setup(), SWING_PARAMS)
    assert setup is not None, "control: the fixture is a valid consolidation"

    from notifier.strategies.indicators import atr as _atr
    span_atr = (setup.top - setup.bottom) / _atr(daily_setup(), 14).iloc[-1]
    assert span_atr < SWING_PARAMS.max_range_atr, "the fixture must pass the shipped cap"

    from dataclasses import replace

    too_tight = replace(SWING_PARAMS, max_range_atr=span_atr / 2)
    assert find_consolidation(daily_setup(), too_tight) is None, (
        "a cap under the fixture's own span must reject it - if this passes, "
        "max_range_atr is being ignored again"
    )

    generous = replace(SWING_PARAMS, max_range_atr=span_atr * 2)
    assert find_consolidation(daily_setup(), generous) is not None

    span_pct = (setup.top - setup.bottom) / setup.bottom
    assert span_pct < SWING_PARAMS.max_range_pct, "the fixture must pass the shipped cap"
    assert find_consolidation(daily_setup(), replace(SWING_PARAMS, max_range_pct=span_pct / 2)) is None
    assert find_consolidation(daily_setup(), replace(SWING_PARAMS, max_range_pct=span_pct * 2)) is not None


def test_the_percentage_cap_survives_the_atr_inflation_that_defeats_the_atr_cap():
    """BANKUSDT's proportions, and why one cap is not enough.

    Its range was 0.07856-0.2356 - 199.9% wide - and it signalled live on
    2026-07-19. Measured in ATR that same range read 40.2 five days earlier,
    20.5 two days earlier and 9.0 on the signal day, because the breakout was
    inflating ATR as it ran. A 12-ATR cap is consulted only at the break, by
    which time the range has "narrowed" under it. The percentage does not move.
    """
    # The real numbers off the chart, so this cannot drift with a fixture.
    low, high = 0.07856, 0.2356
    atr_on_signal_day = 0.01737  # 0.00391 five days earlier: it QUADRUPLED
    width = high - low

    assert width / atr_on_signal_day < SWING_PARAMS.max_range_atr, (
        "the ATR cap alone does NOT catch BANKUSDT - by the signal day the "
        "breakout had inflated ATR until the range measured 9.0 ATR, under the "
        "12 cap. Any fix that relies on the ATR cap here is not a fix."
    )
    assert width / low > SWING_PARAMS.max_range_pct, (
        "the percentage cap is what refuses it: 199.9% against a 100% ceiling"
    )


# ---- the rebuild: penetration and range-keyed dedupe ----


def test_a_graze_past_the_range_top_is_not_a_breakout():
    # TSLAUSDT closed 0.012% past the line - four cents on a $324 stock - and
    # that counted, because the line is the pivot bar's own high and merely
    # touching it qualified. A break now has to clear it by a margin.
    #
    # Kept through the move back to daily structure: this guard was about the
    # TRIGGER being fast, not about the consolidation being on the wrong
    # chart, and a 5m close can still graze a daily level by a hair.
    grazed = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(164.1)})
    assert grazed is None, "a close a fraction above the top is not a breakout"

    cleared = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(164.5)})
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
    # The 5m TRIGGER is what needs a live market, and it still does even though
    # the structure it triggers against is now daily. The 1H trigger reads a
    # whole hour of trade and is left alone.
    assert swing().session_gated is False
    assert day().session_gated is True


def _split_anchor_entry_bars():
    """Entry bars where the two sheet rules give clearly different stops: the
    breakout bar's own low is 158, the last low before it is 155."""
    entry = _bars([160.0] * 14 + [159, 155, 157, 159, 160, 158, 166.0], freq="h")
    entry.loc[15, "low"] = 155.0   # the recent swing low the market turned at
    entry.loc[20, "low"] = 158.0   # the breakout bar's own, much higher, low
    return entry


def test_the_day_version_stops_under_the_last_low_BEFORE_the_breakout():
    """Day sheet: "stop below the last low before the breakout"."""
    signal = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": _split_anchor_entry_bars()})

    assert signal is not None
    # Anchored on 155, not on the 158 of the bar that broke out.
    assert signal.stop_loss < 155.0, f"stop {signal.stop_loss} is not under the recent low"
    assert signal.stop_loss > 150.0, "and not miles below it either"


def test_the_swing_version_stops_under_the_BREAKOUT_CANDLES_low():
    """Swing sheet: "stop below the last low of the breakout candle".

    The two sheets genuinely differ here, and the code used to apply the day
    rule to both. Dror's earlier call had been the reverse - the breakout bar's
    low "is wherever that particular candle happened to open from" - and on
    reading the sheets back he chose the sheets.
    """
    signal = swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": _split_anchor_entry_bars()})

    assert signal is not None
    # Anchored on the breakout bar's own 158, so it sits ABOVE where the day
    # version's stop would land - the two rules must not collapse together.
    assert 155.0 < signal.stop_loss < 158.0, (
        f"stop {signal.stop_loss} is not a buffer below the breakout candle's 158 low"
    )

    day_signal = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": _split_anchor_entry_bars()})
    assert signal.stop_loss > day_signal.stop_loss, "the two anchors must give different stops"


def test_an_unknown_stop_anchor_is_refused_at_construction():
    """A typo'd anchor silently falling back to one of the two rules would put
    real money behind the wrong sheet."""
    with pytest.raises(ValueError, match="stop_anchor"):
        VolumeRun("1D", "1H", stop_anchor="under_the_coil")


def test_price_that_has_left_the_range_is_not_a_consolidation():
    """The gate that Strategy 3 never had, and the reason it never signalled.

    Every other rule bounds the range against the IMPULSE - is the level real,
    was it left untested, is the coil long enough and narrow enough - and none
    of them looks at where price stands now. `highs > price` is trivially true
    once price has collapsed, the floor was never compared to price at all, and
    "untested since it formed" is satisfied most easily by a level price fell
    away from and never came back to: pristine precisely because it is
    irrelevant.

    Measured on ADAUSDT 2026-02-23: price 0.2621 against a 0.9010-1.0204 range
    set on 2025-08-14 - 5.3 range-widths below the floor, needing a 289% candle
    to break. Across 4,224 daily bars price sat in the top 10% of its own range
    exactly 0 times.
    """
    fallen = daily_setup()
    # The box is 153.0-164.0; drop price far beneath it, as ADAUSDT did.
    fallen.loc[248:, "close"] = [60.0] * (len(fallen) - 248)
    fallen.loc[248:, "low"] = [59.0] * (len(fallen) - 248)
    fallen.loc[248:, "high"] = [61.0] * (len(fallen) - 248)

    assert find_consolidation(fallen, SWING_PARAMS) is None, (
        "a range price has fallen out of is not a consolidation"
    )
