import numpy as np
import pandas as pd

import pytest

from notifier.strategies.volume_run import (
    DAY_PARAMS,
    DAY_PARTIAL_FRACTION,
    STOP_AT_RECENT_LOW,
    SWING_PARAMS,
    VolumeRun,
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


def _ramp(a: float, b: float, n: int) -> list[float]:
    step = (b - a) / n
    return [a + step * (i + 1) for i in range(n)]


def _bars(closes, highs=None, lows=None, vols=None, freq: str = "D") -> pd.DataFrame:
    s = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(s), freq=freq),
            "open": s,
            "high": pd.Series(highs, dtype=float) if highs is not None else s + 1.0,
            "low": pd.Series(lows, dtype=float) if lows is not None else s - 1.0,
            "close": s,
            "base_vol": pd.Series(vols, dtype=float) if vols is not None else pd.Series([1.0] * len(s)),
            "quote_vol": 1.0,
        }
    )


def entry_bars(last_close: float, count: int = 35, last_vol: float = 50.0) -> pd.DataFrame:
    """A 1H/5m proxy: flat below the level, then the breakout candle. `count`
    must clear evaluate()'s length guard (volume_baseline_bars + 1 = 31); the
    default gives margin. `last_vol` is high enough to clear the breakout-
    volume rule (>=1.3x a median of mostly-1.0 bars) by default."""
    return _bars(
        [160.0] * (count - 1) + [last_close],
        highs=[160.2] * (count - 1) + [last_close + 0.2],
        lows=[159.8] * (count - 1) + [last_close - 0.2],
        vols=[1.0] * (count - 1) + [last_vol],
        freq="h",
    )


# --- The control fixture -----------------------------------------------
#
# A genuine break-of-structure uptrend (bootstraps DOWN, then an observed
# CHoCH flips it UP - verified directly against structure_context, not
# merely trend_structure's bootstrap, which a plain monotonic climb only
# ever produces via the bootstrap path and structure_context refuses to
# credit), followed by a real rally, a level held for 12 bars on declining
# volume, and a breakout on real volume. Every threshold is cleared with
# comfortable margin - see the derivation notes below each block for the
# exact numbers this was checked against before being written down.

def _structural_uptrend_prefix() -> list[float]:
    """115 closes. Bootstraps DOWN (110 undercuts the first low, protected
    becomes the 140 high), then a genuine CHoCH to UP when price clears that
    140 protected level, then two more legs of confirmed continuation.
    Verified directly against structure_context (not just trend_structure):
    trend="up", choch_count=1."""
    return (
        _ramp(100, 150, 15)   # H0 = 150
        + _ramp(150, 110, 15)  # L1 = 110
        + _ramp(110, 140, 15)  # H2 = 140 (a lower high - irrelevant to what follows)
        + _ramp(140, 90, 15)   # L3 = 90, undercuts L1(110) -> bootstrap DOWN, protected=H2(140)
        + _ramp(90, 170, 20)   # H4 = 170, clears protected(140) -> CHoCH UP, choch_count=1
        + _ramp(170, 130, 15)  # L5 = 130, a higher low (above the new protected, L3=90)
        + _ramp(130, 200, 20)  # H6 = 200, confirms the continuation
    )


def daily_setup() -> pd.DataFrame:
    """143 bars: prefix (115) + rally (15) + level bar (1) + 12-bar box.

    Rule 1 (uptrend): structure reads up with choch=1 at the box's own start;
    price also clears the 8-week fallback average there (166.7) by a wide
    margin (258).
    Rule 1 (min rally): 12.1 ATR into the level, against a 3.5 floor.
    Rule 2 (resistance): nothing overhead in this fixture at all.
    Rule 3 (box starts at level): the level bar IS the box's first bar.
    Rule 3 (ceiling holds): 12 bars, against a 5-bar floor.
    Rule 3 (width): 3.05 ATR, against a 4.0 cap.
    Rule 3 (drift): -2.37 total, 14% of box height, against a 40% cap.
    Rule 4 (ceiling volume): 6.0x the baseline median, against 0.75x.
    Rule 4 (declining): late-half average is 21% of the early half.
    """
    closes = list(_structural_uptrend_prefix())
    highs = [c + 1.0 for c in closes]
    lows = [c - 1.0 for c in closes]

    # Ends at 255, not 260: the last rally bar's high (256) must sit clearly
    # BELOW the tolerance band under the level's ceiling (263 - 0.5 ATR), or
    # it also qualifies as "at the level" and the widest-box-wins rule picks
    # it instead - which is correct algorithm behaviour, just not what this
    # fixture's derivation notes claim if left ambiguous.
    rally = _ramp(200, 175, 5) + _ramp(175, 255, 10)
    closes += rally
    highs += [c + 1.0 for c in rally]
    lows += [c - 1.0 for c in rally]

    # The level bar: a real high with a wick.
    closes.append(258.0)
    highs.append(263.0)
    lows.append(255.0)
    level_index = len(closes) - 1

    box_closes = [252, 250, 248, 251, 249, 253, 250, 252, 249, 251, 250, 252]
    closes += box_closes
    highs += [c + 1.5 for c in box_closes]
    lows += [c - 1.5 for c in box_closes]

    vols = [1.0] * len(closes)
    vols[level_index] = 6.0  # real volume defended the level
    for i in range(level_index + 1, level_index + 7):
        vols[i] = 1.2  # busy early in the box...
    for i in range(level_index + 7, len(closes)):
        vols[i] = 0.3  # ...quiet late: a real dry-up

    return _bars(closes, highs, lows, vols)


def breakout_bar_close() -> float:
    """The close daily_setup()'s ceiling (263.0) needs cleared, for tests
    that append a real daily breakout bar rather than using entry_bars()."""
    return 268.0


def daily_setup_with_breakout() -> pd.DataFrame:
    daily = daily_setup()
    extra = _bars([breakout_bar_close()], highs=[269.0], lows=[265.0], vols=[8.0])
    return pd.concat([daily, extra], ignore_index=True)


def test_structural_uptrend_reads_true_for_a_genuine_break_of_structure():
    """RULE 1. structure_context must read an OBSERVED change of character -
    not merely a bootstrap guess, which a plain monotonic climb only ever
    produces and which structure_context itself refuses to credit."""
    from notifier.strategies.volume_run import structural_uptrend, trend_levels
    from notifier.strategies.indicators import atr

    daily = daily_setup()
    box_start = 130  # the level bar's own index; see daily_setup()'s derivation
    assert daily["close"].iloc[box_start] == 258.0, "fixture index drifted"

    levels = trend_levels(daily)
    atr_series = atr(daily, 14)

    assert structural_uptrend(daily, box_start, daily["close"], levels, atr_series) is True


def test_structural_uptrend_is_false_when_the_long_term_average_fails_despite_structure():
    """ADAUSDT and BCHUSDT: a locally rising two-swing structure while sitting
    30-45% under the symbol's own 200-day average, at the bottom of its
    range - a bounce inside a larger downtrend. Structure alone would read
    this as "up"; the long-term-average half is what refuses it."""
    from notifier.strategies.volume_run import structural_uptrend, trend_levels
    from notifier.strategies.indicators import atr

    daily = daily_setup()
    box_start = 130
    closes = daily["close"].copy()
    # Push every close, from the box start onward, far under a level the
    # long-term average will read as high - the fixture's own structure
    # legs are untouched, only what happens AFTER them changes.
    levels = trend_levels(daily)
    depressed_level = levels.iloc[box_start] * 3  # comfortably above anything closes reach
    levels = levels.copy()
    levels.iloc[box_start] = depressed_level

    atr_series = atr(daily, 14)
    assert structural_uptrend(daily, box_start, closes, levels, atr_series) is False


def test_structural_uptrend_never_overrides_a_down_structure_reading():
    """A structure reading of DOWN must never be rescued by the fallback
    rally - only a genuine "no verdict at all" is eligible for that."""
    from notifier.strategies.volume_run import structural_uptrend, trend_levels
    from notifier.strategies.indicators import atr

    # A clean downtrend: bootstraps up then CHoCHs down, mirroring the
    # up-fixture's shape in reverse.
    closes = (
        _ramp(100, 50, 15)     # L0 = 50
        + _ramp(50, 90, 15)    # H1 = 90
        + _ramp(90, 60, 15)    # L2 = 60 (a higher low - irrelevant to what follows)
        + _ramp(60, 110, 15)   # H3 = 110, exceeds H1(90) -> bootstrap UP, protected=L2(60)
        + _ramp(110, 40, 20)   # L4 = 40, undercuts protected(60) -> CHoCH DOWN, choch_count=1
        + _ramp(40, 80, 15)    # H5 = 80, a lower high
        + _ramp(80, 20, 20)    # L6 = 20, confirms the continuation down
    )
    daily = _bars(closes, highs=[c + 1.0 for c in closes], lows=[c - 1.0 for c in closes])
    start = len(daily)

    from notifier.strategies.structure import structure_context
    _w, structure = structure_context(daily, atr_multiple=2.0, min_lookback=min(200, start))
    assert structure.trend == "down" and structure.choch_count >= 1, "fixture must genuinely read down"

    # Even with an enormous fallback rally available and price far above any
    # long-term average, a DOWN verdict must still refuse.
    padded = pd.concat(
        [daily, _bars([500.0] * 25, highs=[600.0] * 25, lows=[400.0] * 25)],
        ignore_index=True,
    )
    from notifier.strategies.volume_run import trend_levels
    from notifier.strategies.indicators import atr as _atr

    levels = trend_levels(padded)
    atr_series = _atr(padded, 14)
    box_start = len(padded) - 1

    assert structural_uptrend(padded, box_start, padded["close"], levels, atr_series) is False


def test_structural_uptrend_falls_back_to_a_rally_when_structure_has_no_verdict():
    """ALCHUSDT: 60-76 bars of listing history, containing one crash and one
    rally - too little for any confirmed change of character. Every BOS
    threshold from 1.0 to 3.0 ATR reads no trend there. A rally into the
    level stands in, but only because structure found nothing to say."""
    from notifier.strategies.volume_run import structural_uptrend, trend_levels
    from notifier.strategies.indicators import atr
    from notifier.strategies.structure import structure_context

    # One crash, one rally - short enough that no CHoCH can be observed, but
    # long enough (70 days, 10 weeks) for the 8-week long-term average to
    # have resolved and for price to sit above it. `start` (like
    # daily_setup()'s box_start) is the INDEX of the level bar itself, one
    # past the crash+rally history it is judged against - not one-past-the-
    # end of the whole frame, which in_uptrend_at correctly refuses as
    # out of range.
    history = _ramp(100, 20, 35) + _ramp(20, 90, 35)
    closes = history + [92.0]  # the level bar
    daily = _bars(closes, highs=[c + 1.0 for c in closes], lows=[c - 1.0 for c in closes])
    start = len(daily) - 1

    _w, structure = structure_context(
        daily.iloc[:start].reset_index(drop=True), atr_multiple=2.0, min_lookback=min(200, start)
    )
    assert structure.trend is None, "fixture must genuinely have no structural verdict"

    levels = trend_levels(daily)
    atr_series = atr(daily, 14)
    result_with_rally = structural_uptrend(daily, start, daily["close"], levels, atr_series)

    rally_atr = (max(history[-20:]) - min(history[-20:])) / atr_series.iloc[start]
    if rally_atr >= 4.5:
        assert result_with_rally is True, f"rally={rally_atr:.2f} ATR should clear the fallback"
    else:
        assert result_with_rally is False


def test_find_consolidation_locates_the_control_boxs_exact_boundaries():
    """The box the fixture was built around: ceiling at the level bar's own
    high (263.0), floor the lowest low inside the box (246.5), starting AT
    the level bar (index 130)."""
    from notifier.strategies.volume_run import SWING_PARAMS, find_consolidation

    setup = find_consolidation(daily_setup(), SWING_PARAMS)

    assert setup is not None
    assert setup.top == pytest.approx(263.0)
    assert setup.bottom == pytest.approx(246.5)
    assert setup.started_at == 130


def test_a_level_set_late_inside_the_window_is_not_a_box_start():
    """WLDUSDT: the level was set 3 bars before the break - a fresh high, not
    a held ceiling. Mutate the control fixture so the true maximum sits a few
    bars INTO what would otherwise be a valid box, with nothing at the box's
    own first bar reaching anywhere near it."""
    daily = daily_setup()
    # Push the level bar's high even further up, and drop the box's OWN
    # first two bars' highs so they can never anchor a box at the new level.
    daily.loc[130, "high"] = 240.0  # was 263 - now well inside the rally's own range
    daily.loc[135, "high"] = 270.0  # a fresh high 5 bars into the old box
    daily.loc[135, "base_vol"] = 6.0

    setup = find_consolidation(daily, SWING_PARAMS)
    assert setup is None or setup.top != 270.0, "a level set late inside the window must not win"


def test_a_ceiling_that_breaks_too_soon_is_not_a_box():
    """TAGUSDT: the true high broke again only 3 bars before the frame ends -
    not a level the market paused at.

    Simply truncating the fixture does not isolate this: with MIN_BOX_BARS=10
    and CEILING_HOLD_BARS=5, any box whose OWN start bar sets the ceiling
    (the normal case) already has hold=box_len-1>=9, so shortening the frame
    just removes every candidate box entirely (MIN_BOX_BARS itself rejects
    it) rather than exercising the hold check specifically. What genuinely
    isolates it: a SECOND, slightly higher high late in the box, within
    BOX_LEVEL_TOLERANCE_ATR of the first (so the box still "starts at the
    level" via the near-tie) but recent enough that IT has held only 3 bars.
    """
    from notifier.strategies.indicators import atr

    daily = daily_setup()
    a = atr(daily, 14).iloc[-1]
    daily.loc[139, "high"] = 263.0 + 0.3 * a  # a hair above 263, still within tolerance of it
    daily.loc[139, "base_vol"] = 6.0

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_a_level_set_on_thin_volume_is_not_defended():
    """SUIUSDT: the bar that set the ceiling printed BELOW its own coil's
    quiet average - a test on the coil alone cannot see that, so this checks
    the level bar's OWN volume directly."""
    daily = daily_setup()
    daily.loc[130, "base_vol"] = 0.2  # was 6.0 - now quieter than the coil itself

    # A large synthetic fixture can coincidentally contain some OTHER valid
    # box once the intended one is refused; what matters is that the SPECIFIC
    # 263-level box (set on thin volume) is not what comes back.
    setup = find_consolidation(daily, SWING_PARAMS)
    assert setup is None or setup.top != 263.0, "a level set on thin volume must not win"


def test_volume_rising_late_in_the_box_is_not_a_dry_up():
    daily = daily_setup()
    daily.loc[137:141, "base_vol"] = 5.0  # was 0.3 - now busier late than early

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_a_box_that_climbs_steadily_is_not_a_pause():
    """An upward drift is still the move, not a break from it. The climb must
    stay UNDER the fixed ceiling (263) throughout, or it sets a fresh high
    late in the box and gets caught by "starts at the level" / "ceiling
    holds" instead of by the drift rule this test means to isolate."""
    daily = daily_setup()
    climb = [252.0 + 0.7 * i for i in range(12)]
    assert climb[-1] + 1.5 < 263.0, "must not exceed the fixed ceiling"
    daily.loc[131:142, "close"] = climb
    daily.loc[131:142, "high"] = [c + 1.5 for c in climb]
    daily.loc[131:142, "low"] = [c - 1.5 for c in climb]

    from notifier.strategies.volume_run import _coil_fit
    r2, slope = _coil_fit(daily["close"].iloc[130:143])
    assert slope > 0 and r2 > 0.5, "fixture must be a genuine upward drift"

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_a_box_that_gives_back_too_much_of_its_own_height_is_not_a_pause():
    """A downward drift is allowed but bounded - see MAX_COIL_DOWN_DRIFT_SHARE.
    The decline must stay gentle enough that box width stays under MAX_BOX_ATR
    (4.0), or the width cap catches it instead of the drift rule this test
    means to isolate - a steep-enough decline to violate drift share also
    tends to blow the width cap, since both are measured against the same
    box height."""
    daily = daily_setup()
    fall = [258.0 - 1.05 * i for i in range(12)]
    daily.loc[131:142, "close"] = fall
    daily.loc[131:142, "high"] = [c + 1.5 for c in fall]
    daily.loc[131:142, "low"] = [c - 1.5 for c in fall]

    from notifier.strategies.volume_run import MAX_BOX_ATR, MAX_COIL_DOWN_DRIFT_SHARE, _coil_fit
    from notifier.strategies.indicators import atr

    a = atr(daily, 14).iloc[-1]
    width = 263.0 - (min(fall) - 1.5)
    assert width / a < MAX_BOX_ATR, "fixture must pass the width cap on its own"
    r2, slope = _coil_fit(daily["close"].iloc[130:143])
    assert abs(slope * 12) / width > MAX_COIL_DOWN_DRIFT_SHARE, "fixture must violate the drift share"

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_a_box_wider_than_the_cap_is_not_a_coil():
    from notifier.strategies.volume_run import MAX_BOX_ATR
    from notifier.strategies.indicators import atr as _atr

    daily = daily_setup()
    a = _atr(daily, 14).iloc[-1]
    assert (263.0 - 246.5) / a < MAX_BOX_ATR, "control fixture must pass the shipped cap"

    # Blow the floor out far enough that the span exceeds the cap.
    daily.loc[133, "low"] = 246.5 - a * (MAX_BOX_ATR + 1)

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_too_small_a_rally_into_the_level_is_not_a_setup():
    """The rule's own lookback is 20 bars before the box start (110-129 for
    this fixture), not just the "rally" section (115-129) - the window must
    be flattened in full, and CLOSE to the fixed ceiling (263), or the low
    ATR a flat-but-distant window produces inflates the ratio right back up."""
    from notifier.strategies.volume_run import MIN_RALLY_INTO_LEVEL_ATR
    from notifier.strategies.indicators import atr as _atr

    daily = daily_setup()
    flat = [255.0 + 0.02 * i for i in range(20)]
    daily.loc[110:129, "close"] = flat
    daily.loc[110:129, "high"] = [c + 0.3 for c in flat]
    daily.loc[110:129, "low"] = [c - 0.3 for c in flat]

    a = _atr(daily, 14).iloc[130]
    rally_atr = (263.0 - min(c - 0.3 for c in flat)) / a
    assert rally_atr < MIN_RALLY_INTO_LEVEL_ATR, "fixture must fall under the floor"

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_a_collapse_away_from_the_box_is_caught_by_the_width_cap():
    """ADAUSDT's real failure mode, replayed here: price 5.3 range-widths
    below a range set five months earlier - the reason Strategy 3's earlier,
    impulse-candle detector produced zero signals across two live instances
    in its entire life. That detector anchored its range to a specific PAST
    candle and never re-examined it against where price currently stood, so
    a stale range survived a total collapse.

    This detector cannot carry that vulnerability - it has no persisted
    state, and every box is re-derived fresh from a window ending at the
    current bar, so a collapsed bar is either inside the box it defines (by
    construction) or blows the box wide enough to fail the width cap. No
    separate "is price still inside" check is possible to write here that
    isn't algebraically vacuous - see find_consolidation's own comment at
    the point an earlier, dead version of that check used to live.
    """
    daily = daily_setup()
    daily.loc[142, "close"] = 60.0
    daily.loc[142, "high"] = 61.0
    daily.loc[142, "low"] = 59.0

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_no_consolidation_without_a_structural_uptrend():
    """ADAUSDT/BCHUSDT, end to end through find_consolidation rather than
    structural_uptrend directly. A straight crash-only prefix would leave NO
    verdict at all (no reversals to confirm), which is eligible for the
    fallback rally and would pass for the wrong reason - here the prefix
    genuinely bootstraps up then CHoCHs DOWN (the mirror image of
    _structural_uptrend_prefix), so this actually exercises "a down verdict
    is never overridden", not merely "no verdict was available".
    """
    down_prefix = (
        _ramp(100, 50, 15) + _ramp(50, 90, 15) + _ramp(90, 60, 15)
        + _ramp(60, 110, 15) + _ramp(110, 40, 20) + _ramp(40, 80, 15) + _ramp(80, 20, 20)
    )
    assert len(down_prefix) == len(_structural_uptrend_prefix())

    daily = daily_setup()
    n = len(down_prefix)
    daily.loc[: n - 1, "close"] = down_prefix
    daily.loc[: n - 1, "high"] = [c + 1.0 for c in down_prefix]
    daily.loc[: n - 1, "low"] = [c - 1.0 for c in down_prefix]

    from notifier.strategies.structure import structure_context
    _w, structure = structure_context(
        daily.iloc[:130].reset_index(drop=True), atr_multiple=2.0, min_lookback=min(200, 130)
    )
    assert structure.trend == "down" and structure.choch_count >= 1, "fixture must genuinely read down"

    assert find_consolidation(daily, SWING_PARAMS) is None


def test_fires_on_a_breakout_above_the_level():
    signal = swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(268.5)})

    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 3 1D/1H"
    assert signal.entry_price == 268.5
    assert signal.stop_loss < signal.entry_price
    assert signal.reward_risk_ratio == 2.0
    assert signal.partial_fraction == 0.75


def test_no_signal_while_price_stays_inside_the_level():
    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(260.0)}) is None


def test_min_daily_bars_is_sized_for_the_narrowest_box_not_the_widest():
    """ALCHUSDT, one of the two reference setups, has 76 days of history at
    its box start. A floor sized for MAX_BOX_BARS(60) would refuse it
    outright even though find_consolidation finds its box correctly when
    given those bars directly - the search already narrows its own upper
    bound to whatever history is actually available
    (`max_len = min(MAX_BOX_BARS, n - baseline)`), so gating on the widest
    conceivable box here would refuse a symbol that has enough history for
    every box it might actually find, just not the widest one.
    """
    from notifier.strategies.volume_run import MAX_BOX_BARS, MIN_BOX_BARS

    floor = swing().min_daily_bars()
    assert floor < MAX_BOX_BARS + swing().params.volume_baseline_bars, (
        "the floor must not scale with the widest possible box"
    )
    assert floor >= MIN_BOX_BARS + swing().params.volume_baseline_bars, (
        "but it does need enough for the narrowest box plus its own volume baseline"
    )
    assert floor <= 76, "must actually admit ALCHUSDT's 76-bar history"


def test_a_breakout_on_thin_volume_does_not_fire():
    """RULE 5: volume must rise on the break itself. Both reference setups
    cleared 1.3x the median with real margin (1.54x, 2.94x, measured on
    daily volume in the backtest this was validated against) - this is
    applied to the ENTRY TIMEFRAME's own closed trigger bar instead, since
    the daily bar for "today" is not closed yet when the trigger fires
    intraday. See evaluate()'s own comment for that seam."""
    thin = entry_bars(268.5, last_vol=1.1)  # a close above the level, but no real volume behind it

    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": thin}) is None


def test_a_breakout_on_real_volume_still_fires():
    strong = entry_bars(268.5, last_vol=50.0)

    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": strong}) is not None


def test_too_short_an_entry_frame_for_the_volume_baseline_is_refused():
    """_volume_ratio's baseline window needs volume_baseline_bars(30) bars of
    history before the trigger bar; ATR_PERIOD+2(16) alone is not enough for
    it, only for the ATR call. A too-short entry frame must be refused rather
    than let _volume_ratio silently see a thin baseline."""
    short_entry = entry_bars(268.5, count=20)  # well under volume_baseline_bars+1

    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": short_entry}) is None


def test_only_the_first_close_above_the_level_fires():
    # Every later candle is also above the level; re-firing on each is how one
    # stale TSLAUSDT short went out four times in eleven hours.
    already_broken = _bars(
        [160.0] * 30 + [265.0, 266.0, 267.0],
        highs=[160.2] * 30 + [265.2, 266.2, 267.2],
        lows=[159.8] * 30 + [264.8, 265.8, 266.8],
        vols=[1.0] * 30 + [50.0, 50.0, 50.0],
        freq="h",
    )
    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": already_broken}) is None


def test_at_all_time_highs_the_runner_has_no_price_and_trails_instead():
    signal = swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(268.5)})

    assert signal.remainder_target is None  # nothing overhead to exit into
    assert "3 trading days" in signal.remainder_note
    assert any("trail" in note.lower() for note in signal.extra_notes)


def test_resistance_between_the_break_and_the_target_rejects_the_trade():
    # An old high sitting just above the breakout is what stops price reaching
    # a 1:2 target, so the setup is not worth taking.
    daily = daily_setup()
    daily.loc[100, "high"] = 275.0  # a prior peak just overhead (must exceed the target)
    daily.loc[100, "base_vol"] = 9.0

    signal = swing().evaluate("TESTUSDT", {"1D": daily, "1H": entry_bars(268.5)})

    assert signal is None


def test_both_versions_read_their_consolidation_off_the_daily_chart():
    assert swing().trend_timeframe == "1D"
    assert day().trend_timeframe == "1D"
    assert day().entry_timeframe == "5m"
    assert day().tag == "Strategy 3 1D/5m"
    assert day().armed_timeframes == ("5m",)
    assert swing().armed_timeframes == ()

    assert day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(268.5)}) is not None
    assert swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(268.5)}) is not None


def test_the_day_version_exits_flat_at_one_to_two_with_no_runner():
    signal = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(268.5)})

    assert signal is not None
    assert signal.reward_risk_ratio == 2.0
    assert signal.partial_fraction == 1.0, "the whole position leaves at the target"
    assert signal.remainder_target is None
    assert signal.remainder_note == "", "there is no remainder to describe"
    assert not any("runner" in note.lower() for note in signal.extra_notes)
    assert any("trail" in note.lower() for note in signal.extra_notes)


def test_the_swing_version_still_takes_seventy_five_percent_and_runs_the_rest():
    signal = swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(268.5)})

    assert signal.partial_fraction == 0.75
    assert "3 trading days" in signal.remainder_note


def test_arms_wherever_a_live_consolidation_exists():
    instance = day()
    assert instance.arms("TESTUSDT", {"1D": daily_setup()}) is True


def test_arming_refuses_a_symbol_with_no_consolidation():
    instance = day()
    flat = daily_setup()
    flat["close"] = 100.0
    flat["high"] = 100.5
    flat["low"] = 99.5
    flat["open"] = 100.0
    flat["base_vol"] = 1.0

    assert find_consolidation(flat, DAY_PARAMS) is None
    assert instance.arms("TESTUSDT", {"1D": flat}) is False


def test_find_consolidation_defaults_to_swing_params():
    assert find_consolidation(daily_setup()) == find_consolidation(daily_setup(), SWING_PARAMS)


def test_the_signal_is_deduped_on_the_level_not_the_entry_price():
    # TSLAUSDT alerted twice ten minutes apart off an IDENTICAL range, because
    # the default dedupe key includes the entry price and the two 5m closes
    # differed by two cents. Keying on the level claims the range once.
    setup = find_consolidation(daily_setup())
    instance = VolumeRun("1D", "1H")

    first = instance.evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(268.5)})
    second = instance.evaluate("TESTUSDT", {"1D": daily_setup(), "1H": entry_bars(268.9)})

    assert first.entry_price != second.entry_price  # different closes...
    assert first.dedupe_key == second.dedupe_key  # ...but the same trade
    assert first.dedupe_key[-1] == round(setup.top, 10)


def test_only_the_intraday_instance_is_session_gated():
    assert swing().session_gated is False
    assert day().session_gated is True


def test_an_unknown_stop_anchor_is_refused_at_construction():
    with pytest.raises(ValueError, match="stop_anchor"):
        VolumeRun("1D", "1H", stop_anchor="under_the_coil")


def _split_anchor_entry_bars():
    """Entry bars where the two sheet rules give clearly different stops: the
    breakout bar's own low is 261.5, the last low the market actually turned
    at before it is 254.0 - well separated, with no coincidental ties among
    the intervening bars."""
    return _bars(
        [260.0] * 30 + [259, 254, 257, 259, 260, 262, 268.5],
        highs=[260.2] * 30 + [259.2, 254.2, 257.2, 259.2, 260.2, 262.2, 268.7],
        lows=[259.8] * 30 + [258.8, 254.0, 256.8, 258.8, 259.8, 260.5, 261.5],
        vols=[1.0] * 30 + [1, 1, 1, 1, 1, 1, 50.0],
        freq="h",
    )


def test_the_day_version_stops_under_the_last_low_BEFORE_the_breakout():
    signal = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": _split_anchor_entry_bars()})

    assert signal is not None
    assert signal.stop_loss < 254.0, f"stop {signal.stop_loss} is not under the recent low"
    assert signal.stop_loss > 245.0, "and not miles below it either"


def test_the_swing_version_stops_under_the_BREAKOUT_CANDLES_low():
    signal = swing().evaluate("TESTUSDT", {"1D": daily_setup(), "1H": _split_anchor_entry_bars()})

    assert signal is not None
    assert 254.0 < signal.stop_loss < 261.5, (
        f"stop {signal.stop_loss} is not a buffer below the breakout candle's 261.5 low"
    )

    day_signal = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": _split_anchor_entry_bars()})
    assert signal.stop_loss > day_signal.stop_loss, "the two anchors must give different stops"


def test_a_graze_past_the_level_is_not_a_breakout():
    # The day sheet's 5m trigger needs to clear the level by a margin, not
    # merely touch it - see min_penetration_atr.
    grazed = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(263.05, last_vol=50.0)})
    assert grazed is None, "a close a fraction above the level is not a breakout"

    cleared = day().evaluate("TESTUSDT", {"1D": daily_setup(), "5m": entry_bars(268.5, last_vol=50.0)})
    assert cleared is not None, "a genuine break must still fire"
