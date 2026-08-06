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
    """An old, dominant candle is not penalised for being far away in price -
    the time the market spent failing to break it reads as MORE validation,
    not less. But when pairing that specific bar with the current bottom
    pivot makes a range too wide to be a coil, the whole setup must not be
    thrown away on that one bad pairing - a nearer, still-agreeing candidate
    should be tried instead.

    `dominant` is placed BEFORE `weaker`, deliberately. Old-vs-new divergence
    on recency is already covered by test_dominance_overrides_recency; this
    ordering exists so the fixture stays valid alongside the later "not tested
    since formation" rule. If dominant came AFTER weaker instead, dominant's
    own (higher) high would sit inside weaker's post-formation window and
    disqualify weaker outright - not a fixture bug, a real interaction: a
    later spike above an earlier candidate's high genuinely means that
    candidate's level has been tested, regardless of width.
    """
    closes = (
        _leg(50, 150, 240) + _leg(150, 158, 10) + _leg(158, 155, 4) + [155.5] * 6
        + _leg(155.5, 160, 8) + _leg(160, 156, 4) + [156] * 18
    )
    daily = _bars(closes)
    dominant, low, weaker = 250, 256, 267

    # The dominant, OLDER candidate: wins body, wick, and volume outright, but
    # pairing it with `low` makes a 13.1x-ATR range - too wide.
    daily.loc[dominant, "high"] = 172.0
    daily.loc[dominant, "open"] = 155.0
    daily.loc[dominant, "close"] = 168.0
    daily.loc[dominant, "base_vol"] = 15.0

    daily.loc[low, "low"] = 149.0
    daily.loc[low, "base_vol"] = 3.0

    # The weaker, LATER candidate: pairing it with `low` makes a 7.97x-ATR
    # range - inside DAILY_PARAMS' 8x cap, and nothing after it re-tests it.
    daily.loc[weaker, "high"] = 163.0
    daily.loc[weaker, "open"] = 156.0
    daily.loc[weaker, "close"] = 161.0
    daily.loc[weaker, "base_vol"] = 5.0

    daily.loc[weaker + 2 :, "base_vol"] = 0.3  # volume dries up inside the accepted range

    # Control: confirm the DOMINANT bar really does win the argmax contest -
    # if it didn't, this test would pass for the wrong reason.
    tier1, tier2 = _dominant_pivots(
        daily["high"], daily["open"], daily["close"], daily["base_vol"],
        [dominant, weaker], atr(daily, 14), DAILY_PARAMS.volume_baseline_bars,
    )
    assert tier1 == dominant
    assert tier2 == weaker

    setup = find_consolidation(daily, DAILY_PARAMS)

    assert setup is not None
    assert setup.top == 163.0, "must fall through to the later bar, not return None"
    assert setup.top_index == weaker


def test_a_level_retested_after_forming_is_disqualified():
    """The INTCUSDT case: a dominant candle sets the level, but the market
    pokes back up to it a few bars later - no genuine convergence ever
    happened. That candidate must be dropped, even though on its own (absent
    the poke) it is exactly daily_setup()'s known-good consolidation.

    The poke sits deliberately AFTER bar 248, where 245 confirms as a pivot
    (164.0 - low[248] finally clears the reversal threshold) - not between
    245 and 248, where the zigzag is still accumulating the decline needed to
    confirm 245 at all. A poke placed too early does not test an established
    level; it just becomes the new pending high itself, since the original
    peak was never confirmed yet to begin with. Verified against the
    zigzag's actual bar-by-bar trace, not assumed.
    """
    closes = _leg(50, 150, 240) + _leg(150, 163, 6) + _leg(163, 151, 6) + _leg(151, 158, 6)
    daily = _bars(closes)
    daily.loc[245, "high"] = 164.0
    daily.loc[245, "open"] = 150.0
    daily.loc[245, "close"] = 158.0
    daily.loc[245, "base_vol"] = 8.0
    daily.loc[251, "low"] = 149.0
    daily.loc[251, "base_vol"] = 3.0
    daily.loc[252:, "base_vol"] = 0.3

    # Sanity check: this is daily_setup()'s exact shape, so absent the poke it
    # must succeed - if it didn't, the test below would pass for the wrong
    # reason (some other check failing, not the retest rule).
    assert find_consolidation(daily.copy(), DAILY_PARAMS) is not None

    daily.loc[249, "high"] = 164.5  # pokes back up to the level, post-confirmation

    setup = find_consolidation(daily, DAILY_PARAMS)

    assert setup is None, "a retested level must not be selected, not even via fallback"


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


def test_the_absolute_width_cap_rejects_what_atr_alone_would_allow():
    # AXTIUSDT passed a 28.22%-wide "consolidation" because a violent move had
    # inflated hourly ATR, and the width test was ATR-relative only. A plain
    # percentage ceiling cannot be argued around by recent volatility.
    from dataclasses import replace

    assert find_consolidation(daily_setup(), DAILY_PARAMS) is not None
    too_tight_to_allow_anything = replace(DAILY_PARAMS, max_range_pct=0.01)
    assert find_consolidation(daily_setup(), too_tight_to_allow_anything) is None


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
