"""The portfolio backtest must actually be a portfolio.

The driver this replaces looped symbols on the OUTSIDE and time on the inside,
so it ran one symbol's entire year to completion before starting the next. At
most one position was ever open, and every constraint it advertised as
modelled - the 6% aggregate risk cap, the 2-slot swing pool,
one-position-per-symbol, the margin budget - therefore never fired once.
Measured on an 8-symbol run: risk_cap 0, already_in_symbol 0, swing_slots 0
across 41 trades. It still reported a max drawdown and four decline counters,
all of them meaningless, and a -30.6% headline was read off it for two days.

Nothing in that harness was WRONG in a way a unit test would have caught - the
fees were right, the fills were right, the exits were right. Only the loop
order was wrong, and the output looked entirely plausible. So these tests
assert the one thing that distinguishes a portfolio from a sequence: that
positions in different symbols can be open AT THE SAME TIME, and that the caps
bind when they are.
"""

import pandas as pd
import pytest

from backtest import engine as bt
from backtest import portfolio as pf
from notifier.strategies.base import Signal


def _bars(n=40, start="2026-01-01", price=100.0, freq="h"):
    """A flat series, so nothing resolves unless a test moves price itself."""
    ts = pd.date_range(start, periods=n, freq=freq)
    return pd.DataFrame({
        "ts": ts,
        "open": [price] * n,
        "high": [price] * n,
        "low": [price] * n,
        "close": [price] * n,
        "volume": [1000.0] * n,
    })


def _signal(symbol, entry=100.0, stop=95.0, tag="Strategy 1 1H"):
    """All at market: market_fraction 1.0 keeps the $5 floor out of the way,
    so these tests measure concurrency rather than sizing."""
    return Signal(
        symbol=symbol, direction="long", entry_price=entry, stop_loss=stop,
        strategy_tag=tag, limit_entry=None, market_fraction=1.0,
    )


def _replay(signals_by_symbol, bars=None):
    bars = bars or {s: _bars() for s in signals_by_symbol}
    return pf.replay(bars, signals_by_symbol)


def test_two_symbols_signalling_on_the_same_bar_are_open_together():
    """The defect in one assertion.

    Under the old symbols-outside loop AAAUSDT's whole series ran first, so by
    the time BBBUSDT was looked at its position had long since resolved and the
    two were never open at once - which is why no cap ever bound.
    """
    bars = {"AAAUSDT": _bars(), "BBBUSDT": _bars()}
    ts = bars["AAAUSDT"]["ts"].iloc[10]
    signals = {
        "AAAUSDT": [(ts, 10, 100.0, 0, _signal("AAAUSDT"))],
        "BBBUSDT": [(ts, 10, 100.0, 0, _signal("BBBUSDT"))],
    }

    acct = pf.replay(bars, signals)

    assert acct.taken == 2
    assert set(acct.open_positions) == {"AAAUSDT", "BBBUSDT"}


def test_the_6pct_aggregate_risk_cap_actually_binds():
    """Seven simultaneous 1% trades cannot all be taken against a 6% ceiling.

    acct.declined_risk_cap stayed at 0 for the whole life of the old harness,
    which is what made the cap look modelled when it was merely present.

    It binds at FIVE, not the six the arithmetic suggests, and that is correct:
    each entry's taker fee comes out of equity immediately, so both the risk
    budget and the ceiling shrink as positions open. The sixth trade misses by
    about a thousandth of a dollar. Pinned at five deliberately - if a change
    to the fee model ever makes this six, that is worth being told about.
    """
    symbols = [f"S{i}USDT" for i in range(7)]
    bars = {s: _bars() for s in symbols}
    ts = bars[symbols[0]]["ts"].iloc[10]
    signals = {s: [(ts, 10, 100.0, 0, _signal(s))] for s in symbols}

    acct = pf.replay(bars, signals)

    assert acct.taken == 5
    assert acct.declined_risk_cap == 2
    assert len(acct.open_positions) == 5, "and all five are open at once"


def test_a_clean_stop_out_reads_close_to_minus_1r_not_exactly():
    """Dror, 2026-08-26, after plan_position's own sizing was made
    fee-inclusive: "check if it changed something in the statistics". It did
    - _close() used to feed R the RAW price-move pnl, never netting either
    fee against it (only acct.equity's dollar total absorbed them), so a
    backtested clean stop-out always read EXACTLY -1.00R regardless of fees,
    a different convention from the live bot's own fee-inclusive מכפיל_R.

    Both entry and exit fees are netted into R now, matching live. This
    fixture enters fully at MARKET (both legs pay TAKER, 0.0006 each - true
    round-trip 0.12%) against plan_position's shared 0.08% assumption
    (maker-in/taker-out, correct for Strategy 2.1's own limit-only entries,
    not this one) - so the real cost still exceeds what sizing reserved for
    it, and R lands a hair WORSE than -1.00, not exactly on it. That gap is
    the caveat already flagged when the sizing fix shipped, not a new bug.
    """
    bars = {"XUSDT": _bars()}
    ts = bars["XUSDT"]["ts"]
    signals = {"XUSDT": [(ts.iloc[10], 10, 100.0, 0, _signal("XUSDT"))]}
    bars["XUSDT"].loc[11:, ["open", "high", "low", "close"]] = 95.0  # straight to the stop

    acct = pf.replay(bars, signals)

    assert len(acct.closed) == 1
    assert acct.closed[0].reason == "stop"
    assert acct.closed[0].r < -1.0, "both fees must now cost something in R, not just in equity"
    assert acct.closed[0].r == pytest.approx(-1.0, abs=0.01), "still close - only real fees, no slippage modelled"


def test_a_second_signal_on_a_symbol_already_held_is_refused():
    bars = {"AAAUSDT": _bars()}
    ts = bars["AAAUSDT"]["ts"]
    signals = {"AAAUSDT": [
        (ts.iloc[10], 10, 100.0, 0, _signal("AAAUSDT")),
        (ts.iloc[12], 12, 100.0, 0, _signal("AAAUSDT")),
    ]}

    acct = pf.replay(bars, signals)

    assert acct.taken == 1
    assert acct.declined_exposed == 1


def test_skip_pos_removes_an_instance_without_disturbing_the_rest():
    """Strategy 4 is dry-run live, so it must be possible to ask what the
    account does WITHOUT it competing for margin and symbol slots."""
    bars = {"AAAUSDT": _bars(), "BBBUSDT": _bars()}
    ts = bars["AAAUSDT"]["ts"].iloc[10]
    signals = {
        "AAAUSDT": [(ts, 10, 100.0, 0, _signal("AAAUSDT"))],
        "BBBUSDT": [(ts, 10, 100.0, 7, _signal("BBBUSDT", tag="Strategy 4 1H OB2.0"))],
    }

    acct = pf.replay(bars, signals, skip_pos={7})

    assert set(acct.open_positions) == {"AAAUSDT"}


class TestTheSplitFallback:
    """The $5 per-leg floor refuses a whole trade when EITHER leg is under it,
    and since notional is (risk% / stop%) x equity it is the WIDE-stop trades
    that get cut - the ones whose stops survive noise. The fallback collapses
    such a trade onto one leg at identical risk. These tests pin the three
    behaviours apart; which one is right is a measurement, not a unit test.
    """

    def _wide_stop_split(self):
        # 12% stop at $100 equity and 1% risk: notional ~$8.33, so a 20/80
        # split leaves a $1.67 market leg - well under the $5 floor.
        return {"AAAUSDT": [(_bars()["ts"].iloc[10], 10, 100.0, 0, Signal(
            symbol="AAAUSDT", direction="long", entry_price=100.0, stop_loss=88.0,
            strategy_tag="Strategy 1 1H", limit_entry=99.0, market_fraction=0.2,
        ))]}

    def test_baseline_refuses_it(self, monkeypatch):
        monkeypatch.setattr(bt, "SPLIT_FALLBACK", "")
        acct = _replay(self._wide_stop_split())
        assert acct.taken == 0
        assert acct.declined_too_small == 1

    def test_the_limit_fallback_takes_it_as_one_resting_leg(self, monkeypatch):
        monkeypatch.setattr(bt, "SPLIT_FALLBACK", "limit")
        acct = _replay(self._wide_stop_split())
        assert acct.taken == 1
        assert acct.rescued == 1
        pos = acct.open_positions["AAAUSDT"]
        assert pos.size == 0, "nothing fills at market; the whole leg rests"
        assert pos.pending_price == pytest.approx(99.0)

    def test_the_market_fallback_takes_it_filled(self, monkeypatch):
        monkeypatch.setattr(bt, "SPLIT_FALLBACK", "market")
        acct = _replay(self._wide_stop_split())
        assert acct.taken == 1
        assert acct.rescued == 1
        pos = acct.open_positions["AAAUSDT"]
        assert pos.size > 0
        assert pos.pending_size == 0

    def test_risk_is_unchanged_by_which_fallback_is_used(self, monkeypatch):
        """The point of collapsing legs is to clear the floor WITHOUT changing
        what is at stake. If a fallback quietly risked more it would be buying
        its extra trades with leverage rather than with arithmetic."""
        risks = {}
        for mode in ("limit", "market"):
            monkeypatch.setattr(bt, "SPLIT_FALLBACK", mode)
            acct = _replay(self._wide_stop_split())
            risks[mode] = acct.open_positions["AAAUSDT"].risk_amount
        assert risks["limit"] == pytest.approx(risks["market"], rel=1e-9)
        assert risks["limit"] == pytest.approx(1.0), "1% of $100"


def test_an_unfilled_resting_leg_is_cancelled_at_the_window_and_the_trade_vanishes(monkeypatch):
    """cancel_override models live's flat 4-hour entry timeout
    (tracker.ENTRY_TIMEOUT_SECONDS) against the per-instance windows the
    harness otherwise uses, which run to 96 bars. The gap only bites when
    nothing filled immediately - which is exactly the limit fallback.
    """
    monkeypatch.setattr(bt, "SPLIT_FALLBACK", "limit")
    bars = {"AAAUSDT": _bars()}  # flat at 100.0, so the 99.0 limit never fills
    signal = Signal(
        symbol="AAAUSDT", direction="long", entry_price=100.0, stop_loss=88.0,
        strategy_tag="Strategy 1 1H", limit_entry=99.0, market_fraction=0.2,
    )
    signals = {"AAAUSDT": [(bars["AAAUSDT"]["ts"].iloc[10], 10, 100.0, 0, signal)]}

    acct = pf.replay(bars, signals, cancel_override=4)

    assert acct.taken == 1
    assert acct.open_positions == {}, "cancelled 4 bars later, never having filled"
    assert acct.closed == [], "and it is not a trade - nothing was ever at risk"


# ---------------------------------------------------------------------------
# The remainder with no stated target: trailed on structure, not aimed at 3R.
# ---------------------------------------------------------------------------

def _pos(remainder_target=None, pivots=(), r1=2.0, entry=100.0, stop=99.0):
    acct = bt.Account()
    p = bt.Position(
        symbol="X", tag="t", direction="long", entry=entry, size=1.0, stop=stop,
        target=entry + r1 * abs(entry - stop), remainder_target=remainder_target,
        partial_fraction=0.5, opened_at=0, risk_amount=abs(entry - stop), margin=10.0,
        pivots=list(pivots),
    )
    acct.open_positions["X"] = p
    return acct, p


def _bar(high, low, close=None):
    return pd.Series({"high": high, "low": low, "close": close if close is not None else low})


def test_a_remainder_with_no_target_trails_instead_of_aiming_at_a_multiple():
    """It used to be sent to REMAINDER_RATIO x r1 x risk, which is not a rule.

    `risk` was read as `abs(pos.entry - pos.stop)` three lines after the stop
    had been set to the entry, so it was always 0 and the `or` fallback - the
    distance to target 1, i.e. r1 x risk - always won. At the shipping r1 of
    2.0 the runner was aimed at 6R rather than 3R, and mostly walked back to
    the breakeven stop instead of paying. This is the config Strategy 2.1
    ships, so the error was not hypothetical.
    """
    acct, p = _pos(remainder_target=None, r1=2.0)
    bt.step_position(acct, p, _bar(102.001, 100.0), 1, 1)

    assert p.took_partial
    assert p.stop == 100.0, "the partial must move the stop to breakeven"
    assert p.trailing, "no stated target means trail on structure"
    assert p.target == float("inf"), "there is no second price to aim at"


def test_a_stated_remainder_target_is_still_honoured_exactly():
    acct, p = _pos(remainder_target=104.0, r1=2.0)
    bt.step_position(acct, p, _bar(102.001, 100.0), 1, 1)
    assert p.target == 104.0
    assert not p.trailing


def test_the_trailing_stop_ratchets_onto_confirmed_swings_and_never_loosens():
    """A swing BELOW the breakeven floor must not drag the stop back down."""
    acct, p = _pos(pivots=[(2, 99.5, False), (3, 101.0, False), (4, 100.2, False)])
    bt.step_position(acct, p, _bar(102.001, 100.0), 1, 1)
    assert p.stop == 100.0

    bt.step_position(acct, p, _bar(103.0, 101.5), 2, 2)   # 99.5 is below breakeven
    assert p.stop == 100.0, "a swing under the floor must not loosen the stop"

    bt.step_position(acct, p, _bar(104.0, 101.5), 3, 3)   # 101.0 confirmed
    assert p.stop == 101.0

    bt.step_position(acct, p, _bar(105.0, 102.0), 4, 4)   # 100.2 is worse, ignore
    assert p.stop == 101.0, "the ratchet is one-way"


def test_a_swing_is_not_used_before_the_bar_that_confirms_it():
    """Trailing behind a level nobody could see yet is lookahead."""
    acct, p = _pos(pivots=[(9, 101.0, False)])
    bt.step_position(acct, p, _bar(102.001, 100.0), 1, 1)
    for i in range(2, 9):
        bt.step_position(acct, p, _bar(103.0, 101.5), i, i)
        assert p.stop == 100.0, f"pivot confirmed at 9 must not act on bar {i}"
    bt.step_position(acct, p, _bar(103.0, 101.5), 9, 9)
    assert p.stop == 101.0


def test_short_side_trails_the_other_way():
    acct = bt.Account()
    p = bt.Position(
        symbol="X", tag="t", direction="short", entry=100.0, size=1.0, stop=101.0,
        target=98.0, remainder_target=None, partial_fraction=0.5, opened_at=0,
        risk_amount=1.0, margin=10.0, pivots=[(2, 99.0, True), (3, 100.5, True)],
    )
    acct.open_positions["X"] = p
    bt.step_position(acct, p, _bar(100.0, 97.999), 1, 1)
    assert p.stop == 100.0 and p.trailing
    bt.step_position(acct, p, _bar(99.5, 98.0), 2, 2)
    assert p.stop == 99.0, "a short trails DOWN onto confirmed highs"
    bt.step_position(acct, p, _bar(99.5, 98.0), 3, 3)
    assert p.stop == 99.0, "100.5 is worse than 99.0 - one-way"


def test_engine_and_score_agree_on_the_same_trailed_trade():
    """The two implementations must not drift apart again.

    §the-one-scorer: there were two scorers and they disagreed by 0.06R. The
    portfolio engine is necessarily a third implementation - it has equity,
    margin and fees that score.py does not - but the EXIT it models has to be
    the same one, or the portfolio and the sweep describe different strategies.
    """
    from backtest.score import simulate

    sig = Signal(
        symbol="X", direction="long", entry_price=100.0, stop_loss=99.0,
        reward_risk_ratio=2.0, strategy_tag="t", partial_fraction=0.5,
        remainder_target=None,
    )
    rows = [(102.5, 100.0, 101.0), (103.0, 101.2, 102.5), (102.0, 100.9, 101.0)]
    frame = pd.DataFrame(
        [{"high": h, "low": lo, "close": c} for h, lo, c in [(100.0, 100.0, 100.0)] + rows]
    )
    pivots = [(2, 101.0, False)]

    scored = simulate(frame, 0, sig, runner="choch", pivots=pivots)

    acct, p = _pos(remainder_target=None, pivots=pivots)
    for i, (h, lo, c) in enumerate(rows, start=1):
        if bt.step_position(acct, p, _bar(h, lo, c), i, i):
            break

    assert scored.result == "target1 then stop"
    assert acct.closed and acct.closed[0].reason == "stop"
    # Both banked half at +2R and were trailed out at 101.0, i.e. +1R on the
    # remainder: 0.5*2 + 0.5*1 = 1.5R gross before their different fee models.
    assert scored.r_gross == pytest.approx(1.5)
    # The engine's R now nets BOTH exit fees (the partial's maker fee and the
    # final leg's taker fee - risk_amount=1.0 here, so R and $ coincide):
    # 0.5*102.0*MAKER + 0.5*101.0*TAKER = 0.0102 + 0.0303 = 0.0405 off gross.
    # Was tolerated at abs=0.02 when only one leg's fee (sometimes neither)
    # reached R at all; both do now, same convention as live's מכפיל_R.
    assert acct.closed[0].r == pytest.approx(1.5 - 0.0405, abs=1e-3)


def test_the_trail_ignores_swings_confirmed_before_the_trade_opened():
    """Otherwise the ratchet takes max() over the symbol's entire history.

    A long's stop then jumps to the HIGHEST swing low ever printed, which is
    routinely above the current market - and step_position closes a position
    whose stop is above market at that stop, booking the gap as profit. On a
    30-day Strategy 1 replay this read +27565% with a single +164R trade.

    score.simulate has always skipped them; this is the engine agreeing.
    """
    # A swing low at 140.0 confirmed at bar 2, long before the trade opens at
    # bar 50 with a 100.0 entry. Acting on it would set the stop 40% above the
    # market and close the trade instantly for a fabricated +40R.
    history = [(2, 140.0, False), (3, 130.0, False)]
    acct = bt.Account()
    sig = Signal(
        symbol="X", direction="long", entry_price=100.0, stop_loss=99.0,
        reward_risk_ratio=2.0, strategy_tag="t", partial_fraction=0.5,
        remainder_target=None, market_fraction=1.0, limit_entry=None,
    )
    bt.try_open(acct, sig, 100.0, 50, pf.SPECS, 4, pivots=history)
    p = acct.open_positions["X"]
    assert p.pivot_cursor == len(history), "both swings predate the entry"

    bt.step_position(acct, p, _bar(102.001, 100.0), 51, 51)
    assert p.stop == 100.0, "breakeven, not the 140.0 swing from bar 2"
    assert not acct.closed, "nothing should have closed"


def test_a_swing_confirmed_after_entry_is_still_used():
    """The guard above must not disarm trailing altogether."""
    acct = bt.Account()
    sig = Signal(
        symbol="X", direction="long", entry_price=100.0, stop_loss=99.0,
        reward_risk_ratio=2.0, strategy_tag="t", partial_fraction=0.5,
        remainder_target=None, market_fraction=1.0, limit_entry=None,
    )
    bt.try_open(acct, sig, 100.0, 50, pf.SPECS, 4, pivots=[(2, 140.0, False), (52, 101.0, False)])
    p = acct.open_positions["X"]
    assert p.pivot_cursor == 1

    bt.step_position(acct, p, _bar(102.001, 100.0), 51, 51)
    assert p.stop == 100.0
    bt.step_position(acct, p, _bar(103.0, 101.5), 52, 52)
    assert p.stop == 101.0, "the swing confirmed after entry still ratchets"


# ---------------------------------------------------------------------------
# The arming layer, which the harness used to skip.
# ---------------------------------------------------------------------------

def test_the_harness_only_evaluates_an_armed_strategy_where_it_armed():
    """Live, an armed instance is polled only on symbols arms() accepted.

    The harness called evaluate() on every bar regardless, so an armed
    strategy signalled on setups the live bot would never have looked at and
    its backtest counts were an upper bound. Strategy 3's 1D/5m is the
    instance this matters for, and it is the one about to be measured for the
    first time now that 5m turned out to be fetchable.
    """
    from notifier.strategies.base import Strategy

    calls = {"arms": 0, "evaluate": 0}

    class Armed(Strategy):
        tag = "armed"
        timeframes = ["1D", "1H"]
        armed_timeframes = ("1H",)

        def arms(self, symbol, bars_by_timeframe):
            calls["arms"] += 1
            # Never close enough - live this symbol is never polled at all.
            assert "1H" not in bars_by_timeframe, "arms() sees only the non-armed timeframes"
            return False

        def evaluate(self, symbol, bars_by_timeframe):
            calls["evaluate"] += 1
            raise AssertionError("evaluate must not run on an unarmed symbol")

    # The daily series has to START earlier than the 1H window, or the 230-bar
    # daily warmup is never satisfied and the loop skips the instance before
    # arming is ever consulted.
    bars = {"1D": _bars(500, start="2025-01-01", freq="D"), "1H": _bars(3000)}
    original = pf.INSTANCES
    pf.INSTANCES = [(Armed(), ["1D", "1H"], 24)]
    try:
        _symbol, found = pf.scan_symbol(("BTCUSDT", bars, 500))
    finally:
        pf.INSTANCES = original

    assert found == []
    assert calls["evaluate"] == 0
    assert calls["arms"] > 0, "arming must actually have been consulted"


def test_an_unarmed_strategy_is_untouched_by_the_gate():
    """The gate must not change what every other instance does."""
    from notifier.strategies.base import Strategy

    seen = {"evaluate": 0}

    class Plain(Strategy):
        tag = "plain"
        timeframes = ["1H"]

        def evaluate(self, symbol, bars_by_timeframe):
            seen["evaluate"] += 1
            return None

    original = pf.INSTANCES
    pf.INSTANCES = [(Plain(), ["1H"], 24)]
    try:
        pf.scan_symbol(("BTCUSDT", {"1H": _bars(3000)}, 50))
    finally:
        pf.INSTANCES = original

    assert seen["evaluate"] == 50, "an unarmed strategy is still evaluated every bar"
