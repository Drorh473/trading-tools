"""Strategy 4 - order blocks.

The fixtures here have to carry a GENUINE prior trend and an observed change of
character, because structure_context refuses any read whose trend rests only on
the bootstrap guess. Strategy 1's whole fixture set was silently encoding that
bug before it was found (13 tests, every one a ramp-pullback-ramp with
choch_count == 0), so the builder below is asserted to produce a real turn
rather than assumed to.
"""

import pandas as pd
import pytest

from notifier.strategies import order_block
from notifier.strategies.order_block import (
    MIN_REWARD_RISK,
    Gap,
    OrderBlockStrategy,
    find_expansions,
    find_gaps,
    gap_is_closed,
)
from notifier.strategies.indicators import atr
from notifier.strategies.structure import structure_context

ATR_MULTIPLE = 1.25


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """rows are (open, high, low, close); volume and timestamps are filler."""
    return pd.DataFrame(
        {
            "ts": pd.to_datetime([1_600_000_000_000 + i * 3_600_000 for i in range(len(rows))], unit="ms"),
            "open": [r[0] for r in rows],
            "high": [r[1] for r in rows],
            "low": [r[2] for r in rows],
            "close": [r[3] for r in rows],
            "base_vol": [100.0] * len(rows),
            "quote_vol": [100.0] * len(rows),
        }
    )


def _ramp(rows: list, start: float, end: float, steps: int, wick: float = 0.4) -> float:
    """Linear leg from `start` to `end`, one candle per step."""
    step = (end - start) / steps
    price = start
    for _ in range(steps):
        nxt = price + step
        rows.append((price, max(price, nxt) + wick, min(price, nxt) - wick, nxt))
        price = nxt
    return price


def _bullish_setup() -> pd.DataFrame:
    """A downtrend, a real CHoCH up, then an untested bullish order block.

    Laid out so every number the strategy reads is deliberate:
      - the block wicks below a prior swing low at 96 and closes back above it
      - the expansion leaving it gaps from the block's high 98 up to 100
      - a later leg leaves a SECOND gap (105 -> 112) which is the target, and
        the retracement stops at 108 so that gap stays unclosed
      - price ends at 108, exactly halfway back from the 118 extreme to the
        block's 98 top
    """
    rows: list[tuple[float, float, float, float]] = []
    # Downtrend with lower highs and lower lows - enough swings for pivots.
    price = _ramp(rows, 120.0, 112.0, 8)
    price = _ramp(rows, price, 116.0, 5)
    price = _ramp(rows, price, 104.0, 10)
    price = _ramp(rows, price, 109.0, 5)
    price = _ramp(rows, price, 96.0, 10)
    price = _ramp(rows, price, 100.0, 4)
    price = _ramp(rows, price, 84.0, 10)
    # The rally that breaks structure upward - this is the CHoCH.
    price = _ramp(rows, price, 105.0, 14)
    # Pullback to 97, then a bounce big enough to CONFIRM 97 as a pivot low.
    # That confirmed low is the liquidity the block goes on to take; without
    # the bounce it is not a pivot and there is nothing to sweep.
    price = _ramp(rows, price, 97.0, 5)
    price = _ramp(rows, price, 102.0, 3)
    price = _ramp(rows, price, 98.2, 2)

    # The order block: a down candle whose wick runs the 97 low and whose
    # close comes back above it. Range 96.0 - 98.5, so entry is 97.25.
    rows.append((98.2, 98.5, 96.0, 98.0))
    # Displacement away. The gap is [98.5, 106]: the block's high to the low
    # of the candle two bars later.
    rows.append((98.6, 108.0, 98.55, 107.8))
    rows.append((107.9, 109.0, 106.0, 108.5))
    # Two counter candles end the run.
    rows.append((108.5, 108.7, 107.5, 107.8))
    rows.append((107.8, 108.0, 106.8, 107.0))
    # Second leg up, leaving the TARGET gap [107.5, 113].
    rows.append((107.0, 107.5, 106.5, 107.2))
    rows.append((107.5, 116.0, 107.4, 115.8))
    rows.append((115.9, 118.0, 113.0, 117.6))
    # Retracement back toward the block, leaving a BEARISH gap [109.2, 115]
    # overhead on the way down. That is the target: an up-gap would sit below
    # the market and be filled by the very move down to the entry. It has to
    # clear MIN_GAP_ATR, which is a full ATR since Dror's FIGHTUSDT read.
    rows.append((117.6, 117.8, 115.0, 115.2))
    rows.append((115.2, 115.3, 109.0, 109.2))
    # Ends at 104, still outside the block and below the gap's floor so it
    # stays unfilled.
    rows.append((109.2, 109.2, 104.0, 104.0))
    return _bars(rows)


def test_the_fee_constant_is_sourced_not_duplicated():
    """Was a local hardcoded 0.0008 - numerically right for this strategy
    (market_fraction=0.0, maker-in-taker-out) but a second source of truth
    for the same fee Dror's "one shared fee constant, use it everywhere"
    was meant to prevent. Deduped 2026-08-27.

    A plain equality check on the VALUE cannot catch a drift back to a
    hardcoded literal - `ROUND_TRIP_FEE = 0.0008` and
    `ROUND_TRIP_FEE = ROUND_TRIP_FEE_PCT` read identically at runtime today.
    So this reads the SOURCE, matching how test_score.py's
    test_both_generators_disable_the_same_thresholds pins similar
    module-constant assignments: the RHS must be a name reference (an
    import), not a numeric literal.
    """
    import ast
    import pathlib

    src = ast.parse(pathlib.Path(order_block.__file__).read_text(encoding="utf-8"))
    assignments = [
        node for node in src.body
        if isinstance(node, ast.Assign)
        for t in node.targets
        if isinstance(t, ast.Name) and t.id == "ROUND_TRIP_FEE"
    ]
    assert len(assignments) == 1, "ROUND_TRIP_FEE must be assigned exactly once at module level"
    assert isinstance(assignments[0].value, ast.Name), (
        "ROUND_TRIP_FEE must be sourced from an import (e.g. ROUND_TRIP_FEE_PCT), "
        "not a hardcoded numeric literal"
    )
    assert order_block.ROUND_TRIP_FEE == pytest.approx(0.0008)


def test_the_fixture_really_contains_an_observed_change_of_character():
    """Proves the fixture's intent before anything is asserted about behaviour.

    Without this, a fixture whose trend rests on the bootstrap guess would make
    every test below pass or fail for reasons unrelated to order blocks - which
    is exactly what Strategy 1's suite turned out to be doing.
    """
    bars = _bullish_setup()
    _, structure = structure_context(bars, atr_multiple=ATR_MULTIPLE)

    assert structure.trend == "up"
    assert structure.choch_count > 0, "fixture's uptrend is only the bootstrap guess, not an observed turn"


def test_a_three_candle_imbalance_is_found_with_wick_boundaries():
    """Dror's marking rule: the wick end of the candle before the gap and the
    end of the candle after it."""
    bars = _bars([(10, 11, 9, 10.5), (10.5, 14, 10.4, 13.8), (13.9, 15, 12, 14.5)])

    gaps = find_gaps(bars)

    assert len(gaps) == 1
    assert gaps[0].direction == "up"
    assert (gaps[0].low, gaps[0].high) == (11, 12)  # before.high -> after.low
    assert gaps[0].midpoint == 11.5


def test_a_gap_is_open_until_price_traverses_it_completely():
    """"טרם נסגר" is weaker than "טרם נבדק": poking in does not close it.

    This is load-bearing - Dror's own rule is that a gap tested once will
    probably fill completely next time, which makes a partly-filled gap a
    better target than a virgin one, not a disqualified one.
    """
    gap = Gap("up", low=11.0, high=12.0, start_index=0, end_index=2)
    poked = _bars([(10, 11, 9, 10.5), (10.5, 14, 10.4, 13.8), (13.9, 15, 12, 14.5), (14, 14.5, 11.5, 12.0)])
    filled = _bars([(10, 11, 9, 10.5), (10.5, 14, 10.4, 13.8), (13.9, 15, 12, 14.5), (14, 14.5, 10.8, 11.0)])

    assert not gap_is_closed(gap, poked)
    assert gap_is_closed(gap, filled)


def test_a_slow_grind_is_not_an_expansion_however_far_it_travels():
    """The QQQUSDT failure, in the form it takes once the bar cap is gone.

    A 4x ATR move over two bars cannot be slow; over ten it is a slope. With
    the length cap removed, size alone stops being evidence of displacement and
    the steepness floor is the only thing separating the two.
    """
    rows = []
    _ramp(rows, 100.0, 112.0, 12, wick=0.1)  # 12 up candles, 1.0 each
    grind = _bars(rows)
    atr_series = grind["high"].sub(grind["low"]).rolling(14, min_periods=1).mean()

    burst = _bars([(100, 100.2, 99.8, 100.1), (100.2, 112.5, 100.1, 112.0), (112, 112.5, 111, 112.2)])
    burst_atr = burst["high"].sub(burst["low"]).rolling(14, min_periods=1).mean()

    assert find_expansions(grind, atr_series) == []
    assert find_expansions(burst, burst_atr), "a single 12-point candle on a 4-point ATR must qualify"


def test_one_counter_candle_is_tolerated_inside_an_expansion_but_not_two():
    """Dror's call - a multi-candle expansion with one small red bar in it is
    still one move. Two is a fight, not displacement."""
    # Quiet leading bars so ATR is measured on ordinary volatility rather than
    # on the burst itself - see _qualifies. ATR settles at 1.0.
    #
    # No single candle here clears 4x ATR on its own, deliberately: with the
    # shortest-qualifying-burst rule a lone big candle would be trimmed to
    # itself and the counter-candle tolerance would never be reached, which is
    # what the first version of this fixture did.
    quiet = [(100.0, 100.5, 99.5, 100.0)] * 15
    one = _bars(quiet + [
        (100.0, 102.2, 99.9, 102.0),    # 2.0 body - not enough alone
        (102.0, 102.2, 101.5, 101.7),   # small counter, 0.3
        (101.7, 104.5, 101.6, 104.3),   # 2.6 body; the run totals 4.3
    ])
    two = _bars(quiet + [
        (100.0, 102.2, 99.9, 102.0),
        (102.0, 102.2, 101.5, 101.7),
        (101.7, 101.9, 101.2, 101.4),   # second counter, back to back
        (101.4, 104.5, 101.3, 104.3),
    ])

    spans = [(e.start, e.end) for e in find_expansions(one, atr(one, 14))]
    assert (15, 17) in spans, "one tolerated counter candle should not split the run"

    for exp in find_expansions(two, atr(two, 14)):
        assert not (exp.start == 15 and exp.end >= 17), "two counter candles must end the run"


def test_a_bullish_order_block_produces_a_signal_priced_off_the_block():
    bars = _bullish_setup()
    strategy = OrderBlockStrategy("1H", session_gated=False)

    signal = strategy.evaluate("TESTUSDT", {"1H": bars})

    assert signal is not None
    assert signal.direction == "long"
    assert signal.strategy_tag.endswith("OB2.0")
    # Entry is the midpoint of the block's own range (96.0 - 98.5).
    assert signal.entry_price == pytest.approx(97.25)
    # The stop sits below the price the SWEEP REACHED (the block's 96.0 wick),
    # not below the 96.6 level it ran - a stop on the level would sit inside
    # the block, where the wick that made the setup would have taken it out.
    assert signal.stop_loss < 96.0
    # 100% limit: a block entry has no market portion by construction.
    assert signal.market_fraction == 0.0
    assert signal.limit_entry == pytest.approx(97.25)
    # One exit, the whole position, at the gap.
    assert signal.partial_fraction == 1.0
    assert signal.reward_risk_ratio >= MIN_REWARD_RISK


def test_the_target_is_a_gap_the_move_to_the_entry_cannot_fill():
    """Dror, on the first rendered batch: "if the price will go to the ob it
    will fill the gap so this gap is irrelevant".

    A long's limit sits below the market, so every UP-gap overhead is closed
    by the descent to it - and an unfilled up-gap is below the market by
    construction anyway. The gap still open ahead of a long is the one a DOWN
    move left overhead, which the approach moves away from.
    """
    bars = _bullish_setup()
    strategy = OrderBlockStrategy("1H", session_gated=False)

    signal = strategy.evaluate("TESTUSDT", {"1H": bars})

    assert signal is not None
    target = signal.entry_price + (signal.entry_price - signal.stop_loss) * signal.reward_risk_ratio
    # The setup's own displacement gap is the up-gap [98.5, 106] - price falls
    # straight through it to reach 97.25 and it is not a target. The bearish
    # gap [109.2, 115] left during the retracement is, at its midpoint 112.1.
    assert target == pytest.approx(112.1)


def test_a_sliver_of_a_gap_is_not_a_target():
    """Dror on the first rendered batch: "in nvda there isnt a gap there".

    The code had picked a 0.1-wide imbalance on a ~3.0 ATR - 0.03 ATR, a line
    rather than a zone. Here the overhead bearish gap is shrunk below the floor
    and the setup must decline instead of quoting it.
    """
    rows = _bullish_setup()
    thin = rows.iloc[:-3].reset_index(drop=True)
    thin = pd.concat([thin, _bars([
        (117.6, 117.8, 115.0, 115.2),
        (115.2, 115.3, 109.0, 109.2),
        (109.2, 114.95, 104.0, 104.0),  # gap is now [114.95, 115] - 0.05 wide
    ])], ignore_index=True)
    strategy = OrderBlockStrategy("1H", session_gated=False)

    assert strategy.evaluate("TESTUSDT", {"1H": rows}) is not None, "the wide-gap fixture must still signal"
    assert strategy.evaluate("TESTUSDT", {"1H": thin}) is None


def test_the_limit_goes_out_as_soon_as_the_setup_is_complete():
    """There is no wait-for-price condition, deliberately.

    Two proximity rules were built and measured; the ATR one removed 99.1% of
    otherwise-complete setups on 1H, because price crosses from "near" to
    "inside the block" within a candle and there is no near-but-outside state
    to catch. A resting limit does not need price nearby to be placed.
    """
    bars = _bullish_setup()
    strategy = OrderBlockStrategy("1H", session_gated=False)

    signal = strategy.evaluate("TESTUSDT", {"1H": bars})

    assert signal is not None
    distance = abs(float(bars["close"].iloc[-1]) - signal.entry_price) / float(
        atr(bars, order_block.ATR_PERIOD).iloc[-1]
    )
    # The fixture has to place the limit beyond what any tight proximity cap
    # would have admitted, or this test would pass with the gate still in.
    assert distance > 1.0, f"limit is only {distance:.2f} ATR away; fixture proves nothing"


def test_an_unreachably_distant_target_is_declined(monkeypatch):
    """A 49R plan is not a good trade, it is a gap a long way off. Dror's cap
    after reading the first replay."""
    bars = _bullish_setup()
    strategy = OrderBlockStrategy("1H", session_gated=False)

    signal = strategy.evaluate("TESTUSDT", {"1H": bars})
    assert signal is not None
    assert signal.reward_risk_ratio <= order_block.MAX_REWARD_RISK

    monkeypatch.setattr(order_block, "MAX_REWARD_RISK", signal.reward_risk_ratio - 0.1)
    assert strategy.evaluate("TESTUSDT", {"1H": bars}) is None


def _tested_afterwards() -> pd.DataFrame:
    """The same setup, but price dips back into the block and then leaves again.

    The dip has to be followed by a fresh leg that leaves its OWN gap,
    otherwise the setup fails for a reason that has nothing to do with being
    tested: returning to the block necessarily fills every gap above it, so a
    naive "add one candle into the block" fixture is rejected for having no
    target and the untested rule is never reached. That was the first version
    of this test, and it passed against code with the rule removed.
    """
    rows = _bullish_setup()
    tail = _bars([
        (108.0, 108.2, 96.5, 99.0),  # trades back into the 96.0 - 98.5 block
        (99.0, 99.5, 98.8, 99.4),
        (99.5, 106.0, 99.4, 105.8),  # new leg, leaving a fresh [99.5, 105.9] gap
        (106.1, 110.0, 105.9, 109.5),
        (109.5, 109.7, 103.0, 103.2),  # retraces back toward the block again
    ])
    return pd.concat([rows, tail], ignore_index=True)


def test_a_block_price_has_already_returned_into_is_dead():
    """Untested is the whole premise: the orders that make a block work are
    the ones still sitting in it. Price coming back once has already consumed
    them."""
    strategy = OrderBlockStrategy("1H", session_gated=False)

    assert strategy.evaluate("TESTUSDT", {"1H": _tested_afterwards()}) is None


def test_a_block_in_premium_is_refused_for_a_long():
    """"האורדר בלוק חייב להימצא באזור דיסקאונט" - below the dealing range's
    own equilibrium, measured at the price actually transacted."""
    bars = _bullish_setup()
    strategy = OrderBlockStrategy("1H", session_gated=False)
    window, structure = structure_context(bars, atr_multiple=ATR_MULTIPLE)
    leg_low = float(window["low"].iloc[structure.anchor_index])
    leg_high = float(window["high"].iloc[structure.anchor_index :].max())

    signal = strategy.evaluate("TESTUSDT", {"1H": bars})

    assert signal is not None
    assert signal.entry_price < (leg_high + leg_low) / 2


def test_no_trend_no_trade():
    """structure_context returns no trend when nothing in the available history
    turned. Without a trend there is no dealing range, so premium/discount
    cannot even be evaluated - and Strategy 2 shorting MUUUSDT into rising
    highs is what an ungated version looks like."""
    rows: list[tuple[float, float, float, float]] = []
    _ramp(rows, 100.0, 160.0, 70)  # one uninterrupted ramp: no CHoCH anywhere
    bars = _bars(rows)
    strategy = OrderBlockStrategy("1H", session_gated=False)

    _, structure = structure_context(bars, atr_multiple=ATR_MULTIPLE)
    assert structure.choch_count == 0, "fixture must contain no observed turn for this test to mean anything"
    assert strategy.evaluate("TESTUSDT", {"1H": bars}) is None


def test_the_premium_discount_range_is_not_anchored_on_the_block(monkeypatch):
    """WIFUSDT 2026-08-12: the CHoCH anchor WAS the block's own candle, so the
    dealing range was the eleven bars the block itself opened and the block sat
    at the top of it. Premium was then guaranteed for every short of that
    shape - the test could never fail.

    Dror: "if it ob 2.0 and it for short it should be in the expensive area not
    in the cheap one as this block". Measured over a window ending AT the
    block, the range is the one price had built BEFORE the block existed, and
    nothing the block does can move it.
    """
    bars = _bullish_setup()
    strategy = OrderBlockStrategy("1H", session_gated=False)
    window, structure = structure_context(bars, atr_multiple=ATR_MULTIPLE)
    anchor = structure.anchor_index

    start = max(0, anchor - order_block.DEALING_RANGE_LOOKBACK)
    assert start < anchor, "the range must span bars BEFORE the anchor, not after it"
    high = float(window["high"].iloc[start : anchor + 1].max())
    low = float(window["low"].iloc[start : anchor + 1].min())

    # Nothing after the anchor may contribute - that is what made it circular.
    assert high == float(window["high"].iloc[start : anchor + 1].max())
    assert low <= float(window["low"].iloc[anchor])
    signal = strategy.evaluate("TESTUSDT", {"1H": bars})
    assert signal is not None
    assert signal.entry_price < (high + low) / 2, "a long must sit in discount of that range"


def test_the_entry_is_a_maker_fill_so_the_fee_is_not_taker_both_ways():
    """Guards the constant against being 'corrected' back to 0.0012. Strategy 4
    sets market_fraction = 0.0, so the whole entry rests as a limit and fills as
    a maker at 0.02%; the exit a risk gate should price is the taker stop at
    0.06%. Same correction, same reasoning, as Strategy 2's ROUND_TRIP_FEE_PCT.
    """
    assert order_block.ROUND_TRIP_FEE == pytest.approx(0.0008)
    # fee/risk reaches the 0.25 ceiling at a 0.32% stop, not the old 0.48%.
    assert order_block._fee_fraction_of_risk(100.0, 99.68) == pytest.approx(
        order_block.MAX_FEE_FRACTION_OF_RISK, rel=1e-2
    )
