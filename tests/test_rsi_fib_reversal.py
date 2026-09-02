import pandas as pd
import pytest

from notifier.strategies import rsi_fib_reversal
from notifier.strategies.indicators import atr
from notifier.strategies.structure import nearest_level_beyond, trend_structure
from notifier.strategies.rsi_fib_reversal import (
    MARKET_TREND_MA_PERIOD,
    SWING_MIN_LOOKBACK,
    _structure_context,
    FIB_ENTRY,
    FIB_STOP,
    RsiFibReversal,
    _downtrend_leg,
    _uptrend_leg,
)


# Every bar carries a high/low range, so ATR reflects the market's own
# volatility. With range-less bars ATR is set entirely by the pullback that
# triggers the signal, which makes any pullback deep enough to move RSI
# automatically deep enough to confirm a reversal pivot and end the leg -
# a fixture artifact rather than anything a real chart does.
WICK = 3.0


def _bars_from_closes(closes: list[float], highs=None, lows=None) -> pd.DataFrame:
    series = pd.Series(closes)
    high = pd.Series(highs) if highs is not None else series + WICK
    low = pd.Series(lows) if lows is not None else series - WICK
    return pd.DataFrame(
        {
            "ts": pd.date_range("2020-01-01", periods=len(series), freq="h"),
            "open": series,
            "high": high,
            "low": low,
            "close": series,
            "base_vol": 1.0,
            "quote_vol": 1.0,
        }
    )


def _evaluate(symbol, closes, highs=None, lows=None):
    return RsiFibReversal().evaluate(symbol, {"1H": _bars_from_closes(closes, highs, lows)})


def _ramp(start: float, stop: float, bars: int) -> list[float]:
    step = (stop - start) / bars
    return [start + step * (i + 1) for i in range(bars)]


# A rising market with real structure: a first leg up, a genuine interim
# pullback that forms a swing low at 140, then the leg being retraced up to
# 300, then the sharp drop that pushes RSI(10) below 30. The swing detector
# needs an actual reversal to anchor on - a straight line has no pivots.
#
# The interim pullback is deliberately much larger than the one that triggers
# the signal. That gap is the whole point: a reversal big enough to end the
# previous leg has to be distinguishable from a retracement within the current
# one, and a threshold can only tell them apart if the chart itself does.
# The rally flattens before it rolls over, so RSI is already coming off the
# boil when the pullback starts. Without that, only a pullback steep enough to
# also count as a reversal could drag RSI(10) under 30, and the fixture could
# not express "still inside the leg" and "oversold" at the same time.
#
# THE FIXTURE MUST CONTAIN A REAL CHANGE OF CHARACTER. The previous version was
# a ramp up, one pullback, and a ramp on - which produced a trend the detector
# only ever GUESSED: choch_count was 0, so the "uptrend" came from comparing
# the two oldest pivots in the window rather than from any turn the market
# made. Every Strategy 1 test here passed on that basis until the CHoCH
# requirement exposed it. So the market now establishes a DOWNTREND first
# (lower highs at 285 and 240, lower lows), bottoms at 140, and rallies
# through the protected 240 high - a genuine turn, anchored on 140.
UPTREND_SWING_LOW = 140.0
UPTREND_PEAK = 300.0
UPTREND = [
    300.0,
    *_ramp(300, 250, 25),
    *_ramp(250, 285, 12),                     # lower high than 300
    *_ramp(285, 195, 25),                     # lower low - downtrend established
    *_ramp(195, 240, 12),                     # lower high again
    *_ramp(240, UPTREND_SWING_LOW, 25),       # the low the trend turns from
    *_ramp(UPTREND_SWING_LOW, 290, 150),      # breaks back above 240: CHoCH to up
    *_ramp(290, UPTREND_PEAK, 20),
]
# Ends on the bar RSI(10) actually crosses 30, since evaluate() looks for the
# crossing itself rather than for RSI merely sitting below it.
UPTREND_PULLBACK = [UPTREND_PEAK - i * 3 for i in range(1, 5)]

# The exact mirror: an UPTREND established first (higher lows at 118 and 165,
# higher highs), a peak at 260, then a decline through the protected 165 low.
DOWNTREND_SWING_HIGH = 260.0
DOWNTREND_TROUGH = 100.0
DOWNTREND = [
    100.0,
    *_ramp(100, 150, 25),
    *_ramp(150, 118, 12),                     # higher low than 100
    *_ramp(118, 205, 25),                     # higher high - uptrend established
    *_ramp(205, 165, 12),                     # higher low again
    *_ramp(165, DOWNTREND_SWING_HIGH, 25),    # the high the trend turns from
    *_ramp(DOWNTREND_SWING_HIGH, 112, 150),   # breaks back below 165: CHoCH to down
    *_ramp(112, DOWNTREND_TROUGH, 20),
]
DOWNTREND_BOUNCE = [DOWNTREND_TROUGH + i * 3 for i in range(1, 5)]


def test_fires_long_on_oversold_rsi_cross_above_200ma():
    signal = _evaluate("BTCUSDT", UPTREND + UPTREND_PULLBACK)

    assert signal is not None
    assert signal.symbol == "BTCUSDT"
    assert signal.direction == "long"
    assert signal.strategy_tag == "Strategy 1 1H"
    assert signal.reward_risk_ratio == 2.0
    assert signal.stop_loss < signal.entry_price < UPTREND_PULLBACK[-1]


def test_fires_short_on_overbought_rsi_cross_below_200ma():
    signal = _evaluate("ETHUSDT", DOWNTREND + DOWNTREND_BOUNCE)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.reward_risk_ratio == 2.0
    assert DOWNTREND_BOUNCE[-1] < signal.entry_price < signal.stop_loss


def test_the_long_signal_carries_a_fee_domination_gate():
    """MIN_LEG_PCT already guards against fee-domination as a one-time
    calibrated proxy; this is the live equivalent Strategy 2.1 runs with,
    computed fresh from the actual fee formula every signal rather than
    tuned once. Dror, 2026-08-27: "add it for the other [strategies]"."""
    from notifier.risk_sizing import round_trip_fee_for
    from notifier.strategies.rsi_fib_reversal import ENTRY_FEE_PCT, MARKET_ENTRY_FRACTION, MIN_NET_REWARD_RISK

    signal = _evaluate("BTCUSDT", UPTREND + UPTREND_PULLBACK)

    assert signal.fill_guard is not None
    assert signal.fill_guard.min_net_reward_risk == MIN_NET_REWARD_RISK
    assert signal.fill_guard.maker_fee_pct == pytest.approx(ENTRY_FEE_PCT)
    assert signal.fill_guard.round_trip_fee_pct == pytest.approx(round_trip_fee_for(MARKET_ENTRY_FRACTION))


def test_the_short_signal_carries_a_fee_domination_gate_too():
    from notifier.risk_sizing import round_trip_fee_for
    from notifier.strategies.rsi_fib_reversal import ENTRY_FEE_PCT, MARKET_ENTRY_FRACTION, MIN_NET_REWARD_RISK

    signal = _evaluate("ETHUSDT", DOWNTREND + DOWNTREND_BOUNCE)

    assert signal.fill_guard is not None
    assert signal.fill_guard.min_net_reward_risk == MIN_NET_REWARD_RISK
    assert signal.fill_guard.maker_fee_pct == pytest.approx(ENTRY_FEE_PCT)
    assert signal.fill_guard.round_trip_fee_pct == pytest.approx(round_trip_fee_for(MARKET_ENTRY_FRACTION))


def test_a_fee_dominated_stop_is_refused():
    """Direct arithmetic check on the gate's own formula (mirrors how
    ema_trend_v2's equivalent gate is pinned): a stop tight enough for the
    round-trip fee to eat a large share of it must fail the net floor even
    at the gross 2:1 this strategy always targets."""
    from notifier.risk_sizing import round_trip_fee_for
    from notifier.strategies.base import FillGuard
    from notifier.strategies.rsi_fib_reversal import ENTRY_FEE_PCT, MARKET_ENTRY_FRACTION

    entry = 100.0
    stop = 99.7  # a 0.3% stop - tight enough that fees start to matter
    guard = FillGuard(
        min_net_reward_risk=1.5,
        maker_fee_pct=ENTRY_FEE_PCT,
        round_trip_fee_pct=round_trip_fee_for(MARKET_ENTRY_FRACTION),
    )
    refusal = guard.refuses(entry, stop, reward_risk_ratio=2.0)
    assert refusal is not None
    assert "net reward:risk" in refusal


def _reference_bars(direction: str) -> pd.DataFrame:
    """A reference symbol's bars reading a clean, unambiguous trend by the
    same price-vs-200MA convention the strategy already uses on itself: a
    monotonic ramp so the final price sits well above (up) or below (down)
    its own 200-period average."""
    n = MARKET_TREND_MA_PERIOD + 10
    closes = _ramp(100.0, 200.0, n) if direction == "up" else _ramp(200.0, 100.0, n)
    return _bars_from_closes(closes)


def test_default_instance_has_no_market_trend_gate():
    strat = RsiFibReversal("1H")
    assert strat.market_trend_symbol is None
    assert strat.timeframes == ["1H"]


def test_instance_with_a_reference_symbol_declares_its_compound_timeframe():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    assert strat.timeframes == ["1H", "BTCUSDT@1H"]


def test_long_fires_when_the_reference_agrees_uptrend():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    signal = strat.evaluate("BTCUSDT", {"1H": bars, "BTCUSDT@1H": _reference_bars("up")})
    assert signal is not None
    assert signal.direction == "long"


def test_long_is_gated_when_the_reference_disagrees_downtrend():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    signal = strat.evaluate("BTCUSDT", {"1H": bars, "BTCUSDT@1H": _reference_bars("down")})
    assert signal is None


def test_short_fires_when_the_reference_agrees_downtrend():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    signal = strat.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": _reference_bars("down")})
    assert signal is not None
    assert signal.direction == "short"


def test_short_is_gated_when_the_reference_disagrees_uptrend():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    signal = strat.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": _reference_bars("up")})
    assert signal is None


def test_missing_reference_data_fails_open_not_closed():
    """A transient fetch gap on the reference symbol is not evidence the
    market disagrees - the trade still fires, matching every other
    best-effort gate in this codebase (e.g. Scanner._session_allows)."""
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    signal = strat.evaluate("BTCUSDT", {"1H": bars})  # no "BTCUSDT@1H" key at all
    assert signal is not None


def test_short_reference_history_fails_open_too():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    thin_reference = _bars_from_closes([100.0] * 50)  # well under MARKET_TREND_MA_PERIOD + 1
    signal = strat.evaluate("BTCUSDT", {"1H": bars, "BTCUSDT@1H": thin_reference})
    assert signal is not None


# --------------------------------------------------------------------------
# btc_levels_symbol - the levels-based BTC gate (mtf_regime_read_timing):
# daily sets direction, the 1H chart's OWN significant levels set timing.
# A DIFFERENT gate from market_trend_symbol above (price-vs-MA) - the two
# can coexist on one instance, each with its own compound key(s).
# --------------------------------------------------------------------------

def _staircase_closes(direction: str) -> list[float]:
    """A clean structure_trend read via a staircase of confirmed swings - a
    plain monotonic ramp confirms no pivots at all (zigzag_pivots needs an
    actual reversal), unlike the simple price-vs-MA convention
    _market_trend_agrees uses, so _reference_bars' ramp fixture doesn't work
    here."""
    sign = 1.0 if direction == "up" else -1.0
    c, price = [100.0], 100.0
    for _ in range(6):
        for _ in range(14):
            price += 3.0 * sign
            c.append(price)
        for _ in range(6):
            price -= 1.5 * sign
            c.append(price)
    return c


def _btc_levels_bars(direction: str) -> pd.DataFrame:
    """Reference bars for the BTC-levels gate: a clean staircase trend with
    no significant level anywhere near the final price, so
    mtf_regime_read_timing reads the trend straight through unmodified."""
    return _bars_from_closes(_staircase_closes(direction))


def test_default_instance_has_no_btc_levels_gate():
    strat = RsiFibReversal("1H")
    assert strat.btc_levels_symbol is None
    assert strat.timeframes == ["1H"]


def test_instance_with_btc_levels_declares_both_compound_timeframes():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    assert strat.timeframes == ["1H", "BTCUSDT@1D", "BTCUSDT@1H"]


def test_long_fires_when_the_btc_levels_gate_agrees_uptrend():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    ref = _btc_levels_bars("up")
    signal = strat.evaluate("BTCUSDT", {"1H": bars, "BTCUSDT@1D": ref, "BTCUSDT@1H": ref})
    assert signal is not None
    assert signal.direction == "long"


def test_long_is_gated_when_the_btc_levels_gate_disagrees_downtrend():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    ref = _btc_levels_bars("down")
    signal = strat.evaluate("BTCUSDT", {"1H": bars, "BTCUSDT@1D": ref, "BTCUSDT@1H": ref})
    assert signal is None


def test_short_fires_when_the_btc_levels_gate_agrees_downtrend():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    ref = _btc_levels_bars("down")
    signal = strat.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1D": ref, "BTCUSDT@1H": ref})
    assert signal is not None
    assert signal.direction == "short"


def test_short_is_gated_when_the_btc_levels_gate_disagrees_uptrend():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    ref = _btc_levels_bars("up")
    signal = strat.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1D": ref, "BTCUSDT@1H": ref})
    assert signal is None


def test_btc_levels_gate_fails_open_with_missing_reference_data():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    signal = strat.evaluate("BTCUSDT", {"1H": bars})  # no compound keys at all
    assert signal is not None


def test_btc_levels_gate_fails_open_with_thin_reference_history():
    strat = RsiFibReversal("1H", btc_levels_symbol="BTCUSDT")
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    thin = _bars_from_closes([100.0] * 5)
    signal = strat.evaluate("BTCUSDT", {"1H": bars, "BTCUSDT@1D": thin, "BTCUSDT@1H": thin})
    assert signal is not None


def _reference_bars_with_a_confirmed_level(peak: float, trough: float, final: float) -> pd.DataFrame:
    """A higher-timeframe series with one clean, confirmed swing high at
    `peak`: ramps up to it, back down to `trough` (the reversal that
    confirms the pivot), then to `final` - big enough amplitude that the
    ATR-based zigzag threshold confirms the peak as a real pivot regardless
    of how close `peak` itself sits to whatever entry price a test uses."""
    closes = _ramp(50.0, peak, 20) + _ramp(peak, trough, 20) + _ramp(trough, final, 10)
    return _bars_from_closes(closes)


def test_default_instance_has_no_paired_target():
    strat = RsiFibReversal("1H")
    assert strat.target_timeframe is None
    assert strat.timeframes == ["1H"]


def test_paired_instance_declares_its_target_timeframe():
    strat = RsiFibReversal("1H", target_timeframe="4H")
    assert strat.timeframes == ["1H", "4H"]
    assert strat.tag == "Strategy 1 1H/4H"


def test_paired_target_uses_the_higher_timeframes_confirmed_level():
    strat = RsiFibReversal("1H", target_timeframe="4H")
    ref = _reference_bars_with_a_confirmed_level(peak=140.0, trough=20.0, final=60.0)
    ratio = strat._paired_reward_risk_ratio({"4H": ref}, "long", entry=100.0, risk=2.0)
    # nearest_level_beyond reads the WICK high (peak + WICK = 143), 43 above
    # the 100 entry; risk is 2, so the level implies a 21.5:1 reward:risk -
    # nothing like the flat 2.0.
    assert ratio == pytest.approx((140.0 + WICK - 100.0) / 2.0)


def test_paired_target_falls_back_when_the_level_is_too_close():
    strat = RsiFibReversal("1H", target_timeframe="4H")
    # Same huge surrounding amplitude (so the pivot still confirms and IS
    # found, at 101 + WICK - a real level, just a bad one to aim at) - a
    # large risk (10) makes its ratio land under PAIRED_TARGET_MIN_RATIO
    # rather than this accidentally testing the "no level found" path instead.
    ref = _reference_bars_with_a_confirmed_level(peak=101.0, trough=20.0, final=60.0)
    level = nearest_level_beyond(ref, atr(ref, rsi_fib_reversal.ATR_PERIOD) * rsi_fib_reversal.STRUCTURE_ATR_MULTIPLE,
                                  100.0, "long")
    assert level is not None, "fixture must actually confirm a level, or this test proves nothing"
    ratio = strat._paired_reward_risk_ratio({"4H": ref}, "long", entry=100.0, risk=10.0)
    assert (level - 100.0) / 10.0 < rsi_fib_reversal.PAIRED_TARGET_MIN_RATIO
    assert ratio == rsi_fib_reversal.REWARD_RISK_RATIO


def test_paired_target_falls_back_when_the_level_is_too_far():
    strat = RsiFibReversal("1H", target_timeframe="4H")
    ref = _reference_bars_with_a_confirmed_level(peak=5000.0, trough=20.0, final=60.0)
    ratio = strat._paired_reward_risk_ratio({"4H": ref}, "long", entry=100.0, risk=2.0)
    assert ratio == rsi_fib_reversal.REWARD_RISK_RATIO


def test_paired_target_falls_back_with_no_reference_data():
    strat = RsiFibReversal("1H", target_timeframe="4H")
    ratio = strat._paired_reward_risk_ratio({}, "long", entry=100.0, risk=2.0)
    assert ratio == rsi_fib_reversal.REWARD_RISK_RATIO


def test_paired_target_falls_back_with_no_confirmed_level_at_all():
    strat = RsiFibReversal("1H", target_timeframe="4H")
    flat = _bars_from_closes([100.0] * 30)  # no swings, nothing to confirm
    ratio = strat._paired_reward_risk_ratio({"4H": flat}, "long", entry=100.0, risk=2.0)
    assert ratio == rsi_fib_reversal.REWARD_RISK_RATIO


def test_no_signal_without_enough_history():
    assert _evaluate("BTCUSDT", [100.0] * 50) is None


def test_no_signal_when_rsi_not_crossing():
    # Flat price series: RSI stays near 50, never crosses 30/70.
    assert _evaluate("BTCUSDT", [100.0] * 210) is None


def test_uptrend_leg_anchors_on_the_pivot_that_started_the_leg():
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)

    swing_low, swing_high = _uptrend_leg(bars)

    # The interim pullback low, not the 100.0 the whole series started from,
    # and read off the wicks rather than the closes.
    assert swing_low == pytest.approx(UPTREND_SWING_LOW - WICK)
    assert swing_high == pytest.approx(UPTREND_PEAK + WICK)


def test_downtrend_leg_anchors_on_the_pivot_that_started_the_leg():
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)

    swing_low, swing_high = _downtrend_leg(bars)

    assert swing_high == pytest.approx(DOWNTREND_SWING_HIGH + WICK)
    assert swing_low == pytest.approx(DOWNTREND_TROUGH - WICK)


def test_swing_uses_wick_extremes_not_closes():
    # The pivot candle wicks below its close and the peak candle wicks above
    # its close; both wicks are the real extremes and must anchor the Fib.
    closes = UPTREND + UPTREND_PULLBACK
    highs = [c + WICK for c in closes]
    lows = [c - WICK for c in closes]
    # Both located by MEANING rather than by index arithmetic, which pointed
    # at the wrong bars the moment the fixture was reshaped to contain a real
    # CHoCH: the old slice missed the structural low, and index(max(...))
    # returned bar 0, because the fixture now OPENS at 300 - the same value as
    # its peak - so it modified the opening candle instead of the top.
    pivot_idx = closes.index(min(closes))
    peak_idx = max(range(pivot_idx, len(closes)), key=lambda i: closes[i])
    assert closes[pivot_idx] == pytest.approx(UPTREND_SWING_LOW), "the fixture's structural low moved"
    assert closes[peak_idx] == pytest.approx(UPTREND_PEAK), "the fixture's peak moved"
    highs[peak_idx] = closes[peak_idx] + 8
    lows[pivot_idx] = closes[pivot_idx] - 15

    signal = _evaluate("BTCUSDT", closes, highs=highs, lows=lows)

    assert signal is not None
    expected_range = highs[peak_idx] - lows[pivot_idx]
    assert signal.entry_price == pytest.approx(highs[peak_idx] - expected_range * FIB_ENTRY)
    assert signal.stop_loss == pytest.approx(highs[peak_idx] - expected_range * FIB_STOP)


def test_swing_ignores_deeper_low_after_the_peak():
    # A deeper low during the pullback that triggers the signal is part of the
    # retracement, not the start of the leg being retraced - the anchor stays
    # the structural pivot. The wick is kept shallow enough not to clear the
    # reversal threshold, because one that did would genuinely end the leg.
    closes = UPTREND + UPTREND_PULLBACK
    lows = [c - WICK for c in closes]
    lows[-1] = closes[-1] - 12

    swing_low, swing_high = _uptrend_leg(_bars_from_closes(closes, lows=lows))

    assert swing_low == pytest.approx(UPTREND_SWING_LOW - WICK)
    assert swing_high == pytest.approx(UPTREND_PEAK + WICK)


def test_leg_stops_at_a_price_regime_break():
    # The SNDKUSDT case: an old high-price regime, a one-session ~35% crash,
    # then a new lower regime. Taking the global max and global min over the
    # window draws a Fib straddling the crash, putting entry far above any
    # price the market has traded since. The leg must start after the crash.
    old_regime = [1400.0, *_ramp(1400, 1650, 99)]
    crash = _ramp(1650, 1000, 8)
    # Rallies to 1270, breaks back under the 1050 swing low - which turns the
    # trend down and makes 1270 the anchor - then bounces enough for that low to
    # confirm as a swing. The bounce matters: the trend turns on a CONFIRMED
    # swing, not on price poking through intraday, so a fixture that ends
    # mid-break would be testing a state the detector deliberately ignores.
    new_regime = [
        *_ramp(1000, 1130, 40), *_ramp(1130, 1050, 20),
        *_ramp(1050, 1270, 40), *_ramp(1270, 1010, 30), *_ramp(1010, 1120, 14),
    ]
    bars = _bars_from_closes(old_regime + crash + new_regime)

    swing_low, swing_high = _downtrend_leg(bars)

    assert swing_high < 1400  # not the pre-crash 1650
    assert swing_low > 1000  # not the crash bottom either


def test_a_pullback_that_holds_structure_is_still_a_tradeable_leg():
    """Deliberate reversal of the old rule, and the reason this was rebuilt.

    The previous guard ended a leg the moment an opposite ZigZag pivot
    confirmed. That is what silently blocked AAPLUSDT: the rally off 302.04
    confirmed a swing low, so the setup vanished - while the leg it had
    selected instead was 90% retraced and should have been the dead one. A
    pullback that has NOT broken structure is a retracement, which is exactly
    what this strategy exists to sell into.
    """
    deep = UPTREND + [UPTREND_PEAK - i * 12 for i in range(1, 8)]
    shallow = UPTREND + UPTREND_PULLBACK

    assert _uptrend_leg(_bars_from_closes(shallow)) is not None
    assert _uptrend_leg(_bars_from_closes(deep)) is not None


def test_no_leg_once_price_is_past_its_own_stop():
    """The TSLAUSDT failure the old guard was really protecting against.

    The code kept quoting entry 303.64 / stop 306.31 off a finished leg while
    price walked to 312, so every alert arrived already past its own stop. That
    is now caught by the retracement test rather than by the pivot rule: beyond
    FIB_STOP the trade cannot be entered, whatever the structure says.
    """
    leg = _uptrend_leg(_bars_from_closes(UPTREND + UPTREND_PULLBACK))
    assert leg is not None
    swing_low, swing_high = leg
    past_stop = swing_high - (swing_high - swing_low) * FIB_STOP

    # Walk price just under that level without breaking the structure low.
    beyond = UPTREND + _ramp(UPTREND_PEAK, past_stop - 2, 6)

    assert _uptrend_leg(_bars_from_closes(beyond)) is None


def test_a_symbol_offers_a_leg_in_only_one_direction_at_a_time():
    # A market is either retracing an up-leg or a down-leg. Before the guard
    # both legs resolved at once off unrelated stale pivots, which is what let
    # a short fire on a symbol that had been rallying for days.
    for closes in (UPTREND + UPTREND_PULLBACK, DOWNTREND + DOWNTREND_BOUNCE):
        bars = _bars_from_closes(closes)

        assert (_uptrend_leg(bars) is None) != (_downtrend_leg(bars) is None)


def test_no_leg_without_an_identifiable_reversal():
    # A perfectly monotonic series never reverses, so there is no pivot marking
    # where the current leg began and no Fib can honestly be drawn.
    bars = _bars_from_closes(_ramp(100, 300, 210))

    assert _uptrend_leg(bars) is None
    assert _downtrend_leg(bars) is None


def test_a_leg_too_small_to_clear_the_fee_is_rejected(monkeypatch):
    """MIN_LEG_PCT, the price of the finer pivot threshold.

    Dropping STRUCTURE_ATR_MULTIPLE to 1.25 doubled the rate of fee-dominated
    legs from 8% to 16% across a 41-day replay. APTUSDT 1H was Dror's live
    example: a 4.2% leg giving a 0.70% stop, where the 0.12% round trip eats
    17% of 1R before the market moves. The Fib gap between entry and stop is
    16.8% of the leg, so a 6% leg is about a 1% stop - the boundary.

    Asserted by lifting the constant rather than shrinking a fixture: what
    matters is that the check is wired and binding, and a synthetic leg small
    enough to trip it would also be too small to form pivots at all.
    """
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    assert _uptrend_leg(bars) is not None, "this leg is normally tradeable"

    monkeypatch.setattr(rsi_fib_reversal, "MIN_LEG_PCT", 5.0)  # demand a 500% leg

    assert _uptrend_leg(bars) is None


def test_the_minimum_leg_applies_to_shorts_too(monkeypatch):
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    assert _downtrend_leg(bars) is not None

    monkeypatch.setattr(rsi_fib_reversal, "MIN_LEG_PCT", 5.0)

    assert _downtrend_leg(bars) is None


def test_no_leg_once_price_is_past_halfway_to_its_stop():
    """MMTUSDT at 75% retraced and GOOGLUSDT at 74%, both live.

    Past the midpoint of entry and stop the trade has under a third of its risk
    distance left, and the resting limit sits on the WRONG SIDE of the market -
    a sell limit below price, a buy limit above it - so it fills instantly
    instead of resting and the alert's blended entry is fiction. Measured at 9%
    win, -0.73R over 41 days.
    """
    leg = _uptrend_leg(_bars_from_closes(UPTREND + UPTREND_PULLBACK))
    assert leg is not None
    swing_low, swing_high = leg
    rng = swing_high - swing_low
    midpoint = (FIB_ENTRY + FIB_STOP) / 2

    # Just inside the bound is still tradeable; just past it is not.
    inside = swing_high - rng * (midpoint - 0.05)
    beyond = swing_high - rng * (midpoint + 0.05)

    assert _uptrend_leg(_bars_from_closes(UPTREND + _ramp(UPTREND_PEAK, inside, 6))) is not None
    assert _uptrend_leg(_bars_from_closes(UPTREND + _ramp(UPTREND_PEAK, beyond, 6))) is None


def test_the_band_between_entry_and_the_cut_is_still_allowed():
    """Cutting at FIB_ENTRY would be wrong.

    The stretch between 61.8% and the midpoint measured comfortably profitable;
    only the last part before the stop collapses. A guard at the entry level
    would throw away the good band with the bad.
    """
    leg = _uptrend_leg(_bars_from_closes(UPTREND + UPTREND_PULLBACK))
    swing_low, swing_high = leg
    rng = swing_high - swing_low
    just_past_entry = swing_high - rng * (FIB_ENTRY + 0.02)

    assert _uptrend_leg(_bars_from_closes(UPTREND + _ramp(UPTREND_PEAK, just_past_entry, 6))) is not None


# ---- the trend must have been observed to turn ----
#
# Sweeping the old fixed 200-bar lookback from 200 to 600 bars over identical
# price data flipped 43% of symbols' trend direction and made 27% flip two or
# more times - BTCUSDT 1H ran up-up-up-up-down-down-down-up-down. The answer
# was an artifact of where the window started, so no choice of window fixes
# it. Dror's rule: require an observed CHoCH, and grow the window while there
# isn't one.


def test_a_bootstrap_only_read_is_not_a_trend():
    """A market that only ever rises has no change of character in it. The
    detector must say so rather than inferring a trend from whichever two
    pivots happen to be oldest.

    Higher highs and higher lows the whole way, so the bootstrap has plenty of
    pivots to work with and confidently reports "up" - it simply never
    observes anything break. An earlier version of this test used a plain
    ramp, which returned None because the window held fewer than three pivots
    at all, so it passed against the reverted code too and proved nothing.
    """
    closes = (
        [100.0] * 20
        + _ramp(100, 150, 25) + _ramp(150, 130, 15)
        + _ramp(130, 200, 25) + _ramp(200, 180, 15)
        + _ramp(180, 260, 25) + _ramp(260, 240, 15)
        + _ramp(240, 320, 25) + _ramp(320, 300, 15)
        + _ramp(300, 380, 25)
    )
    bars = _bars_from_closes(closes)

    # The raw read really does claim a trend - that is what makes this a test.
    thresholds = atr(bars, rsi_fib_reversal.ATR_PERIOD) * rsi_fib_reversal.STRUCTURE_ATR_MULTIPLE
    raw = trend_structure(bars.reset_index(drop=True), thresholds.reset_index(drop=True))
    assert raw.trend == "up" and raw.choch_count == 0, "fixture must bootstrap without ever turning"

    _, structure = _structure_context(bars)

    assert structure.trend is None, "no observed turn means no reading"


def test_a_real_turn_is_reported():
    """The control, and the fixture this file now uses: a downtrend that
    genuinely breaks upward."""
    _, structure = _structure_context(_bars_from_closes(UPTREND + UPTREND_PULLBACK))

    assert structure.choch_count >= 1
    assert structure.trend == "up"


def test_the_window_grows_until_it_finds_a_turn():
    """The growing half of the rule. UPTREND's turn sits outside the first 200
    bars, so a fixed window would miss it entirely - the accepted window has
    to be wider than the floor.
    """
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)

    window, structure = _structure_context(bars)

    assert structure.choch_count >= 1
    assert len(window) > SWING_MIN_LOOKBACK, "the window had to grow past the floor to find the turn"


def test_the_window_stops_growing_at_the_first_turn_found():
    """It takes the SMALLEST window containing a turn, not the largest
    available - a wider view drags in older, already-resolved structure whose
    bootstrap can change the answer again."""
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)

    window, _ = _structure_context(bars)

    assert len(window) < len(bars), "it should stop at the first window with a turn, not consume everything"


# --------------------------------------------------------------------------
# market_trend_timeframe / market_trend_ma_period / market_trend_confirm_bars
# - three optional knobs on the same market_trend_symbol gate, added
# 2026-08-28 to test whether the gate's REFERENCE definition can be improved
# (the shipped 1H BTCUSDT gate measured negative in year 1 on every check).
# All default to today's exact behaviour; see the class docstring.
# --------------------------------------------------------------------------

def test_default_gate_is_unaffected_by_the_new_knobs():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    assert strat.tag == "Strategy 1 1H +BTCUSDT"
    assert strat._market_key == "BTCUSDT@1H"
    assert strat.market_trend_timeframe == "1H"
    assert strat.market_trend_ma_period == MARKET_TREND_MA_PERIOD
    assert strat.market_trend_confirm_bars == 0


def test_market_trend_timeframe_reads_a_different_compound_key():
    strat = RsiFibReversal("1H", market_trend_symbol="BTCUSDT", market_trend_timeframe="4H")
    assert strat._market_key == "BTCUSDT@4H"
    assert strat.timeframes == ["1H", "BTCUSDT@4H"]
    assert "(@4H)" in strat.tag

    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    # The 1H key is present and would BLOCK a long if it were consulted; the
    # instance is configured to read the 4H key instead, which agrees.
    signal = strat.evaluate(
        "BTCUSDT",
        {"1H": bars, "BTCUSDT@1H": _reference_bars("down"), "BTCUSDT@4H": _reference_bars("up")},
    )
    assert signal is not None
    assert signal.direction == "long"


def test_market_trend_ma_period_override_reads_where_the_default_cannot():
    """60 bars is below MARKET_TREND_MA_PERIOD + 1 (201), so the DEFAULT
    period has no reading and fails open even though the reference clearly
    disagrees. The same 60 bars are enough for an explicit period=50, which
    gives a real reading and gates the trade."""
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    thin_uptrend = _bars_from_closes([100.0 + i for i in range(60)])  # 60 bars, clearly "up"

    default_period = RsiFibReversal("1H", market_trend_symbol="BTCUSDT")
    signal = default_period.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": thin_uptrend})
    assert signal is not None, "60 bars < 201 - default period has no reading, fails open"

    overridden = RsiFibReversal("1H", market_trend_symbol="BTCUSDT", market_trend_ma_period=50)
    assert "(ma50)" in overridden.tag
    signal = overridden.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": thin_uptrend})
    assert signal is None, "60 bars >= 51 - period=50 reads 'up', gates the short"


def test_confirm_bars_fails_open_on_a_flip_inside_the_window():
    """Numerically verified fixture: sma(period=5) on these closes reads
    above/below/above/below/above across the last 5 bars - a flip inside
    any 2- or 3-bar trailing window."""
    flipping = _bars_from_closes([100.0] * 20 + [100.5, 99.0, 100.8, 99.2, 100.6])
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)

    unconfirmed = RsiFibReversal("1H", market_trend_symbol="BTCUSDT", market_trend_ma_period=5)
    signal = unconfirmed.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": flipping})
    assert signal is None, "single-bar read at the final bar is 'up' (100.6 > ma), gates the short"

    confirmed = RsiFibReversal(
        "1H", market_trend_symbol="BTCUSDT", market_trend_ma_period=5, market_trend_confirm_bars=3
    )
    assert "(ma5,c3)" in confirmed.tag
    signal = confirmed.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": flipping})
    assert signal is not None, "the last 3 bars disagree with each other - fails open, not gated"


def test_confirm_bars_gates_normally_once_the_window_agrees():
    """Companion fixture: sma(period=5) reads below/above/above/above/above
    across the last 5 bars - the last 3 all agree, so confirm_bars=3 should
    gate exactly like an unconfirmed read would."""
    settled = _bars_from_closes([100.0] * 20 + [104.0, 105.0, 106.0, 107.0, 108.0])
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)

    confirmed = RsiFibReversal(
        "1H", market_trend_symbol="BTCUSDT", market_trend_ma_period=5, market_trend_confirm_bars=3
    )
    signal = confirmed.evaluate("ETHUSDT", {"1H": bars, "BTCUSDT@1H": settled})
    assert signal is None, "last 3 bars all read 'up' - a confirmed disagreement, gates the short"


# ---------------------------------------------------------------------------
# chart_overlay: the Fib swing itself - the evidence entry/stop were measured
# from, not just their resulting prices (which chart.render already draws).
# ---------------------------------------------------------------------------


def test_chart_overlay_marks_the_long_fib_swing_high_and_low():
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    strategy = RsiFibReversal()
    signal = strategy.evaluate("BTCUSDT", {"1H": bars})
    assert signal is not None
    swing_low, swing_high = _uptrend_leg(bars)

    overlay = strategy.chart_overlay({"1H": bars}, signal)

    prices = [price for price, _label, _color in overlay.levels]
    assert any(abs(p - swing_low) < 1e-6 for p in prices)
    assert any(abs(p - swing_high) < 1e-6 for p in prices)


def test_chart_overlay_marks_the_short_fib_swing_high_and_low():
    bars = _bars_from_closes(DOWNTREND + DOWNTREND_BOUNCE)
    strategy = RsiFibReversal()
    signal = strategy.evaluate("ETHUSDT", {"1H": bars})
    assert signal is not None
    swing_low, swing_high = _downtrend_leg(bars)

    overlay = strategy.chart_overlay({"1H": bars}, signal)

    prices = [price for price, _label, _color in overlay.levels]
    assert any(abs(p - swing_low) < 1e-6 for p in prices)
    assert any(abs(p - swing_high) < 1e-6 for p in prices)


def test_chart_overlay_marks_the_anchor_pivot_position_and_price():
    """The anchor is the pivot the whole leg is measured from - for a long,
    that's the swing LOW the uptrend turned up from (see _leg's docstring)."""
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    strategy = RsiFibReversal()
    signal = strategy.evaluate("BTCUSDT", {"1H": bars})
    swing_low, _swing_high = _uptrend_leg(bars)

    overlay = strategy.chart_overlay({"1H": bars}, signal)

    assert len(overlay.markers) == 1
    position, price, _label = overlay.markers[0]
    assert 0 <= position < len(bars)
    assert abs(price - swing_low) < 1e-6


def test_chart_overlay_returns_none_when_its_own_timeframe_is_missing():
    bars = _bars_from_closes(UPTREND + UPTREND_PULLBACK)
    strategy = RsiFibReversal()
    signal = strategy.evaluate("BTCUSDT", {"1H": bars})

    assert strategy.chart_overlay({}, signal) is None
