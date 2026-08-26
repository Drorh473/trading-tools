import pytest

from notifier.risk_sizing import plan_position


def test_matches_worked_example_from_the_rule():
    # From the user's rule: stop 0.8%, risk 2%, $1000 equity -> Y=2.5, notional $2500.
    # Isolated from the fee correction below via round_trip_fee_pct=0 - this
    # is the base multiplier rule on its own, not the sizing plan_position
    # actually returns by default.
    entry = 100.0
    stop = entry - entry * 0.008  # 0.8% below entry
    plan = plan_position(
        equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long", round_trip_fee_pct=0.0,
    )

    assert plan.notional_value == pytest.approx(2500, rel=1e-3)
    assert plan.risk_amount == pytest.approx(20)  # 2% of $1000


def test_position_is_sized_smaller_to_absorb_the_round_trip_fee():
    """AIOUSDT #68 and 26 other closed losers, 2026-08-26: a CLEAN stop-out -
    zero price slippage - still read worse than -1.00R, because sizing
    ignored fees entirely and only the price move was weighed against
    risk_pct. Dror: "we can compute the stop include the fees and make it
    1r". Position size now shrinks just enough that price-move dollars PLUS
    the round-trip fee, not the price move alone, equals the intended risk -
    the real payoff below is that a clean stop-out now costs almost exactly
    risk_pct of equity, not risk_pct plus whatever the fee silently added.
    """
    entry, stop = 100.0, 99.0  # 1% price risk
    plan = plan_position(equity=10_000, risk_pct=0.01, entry_price=entry, stop_loss=stop, direction="long")

    fee_pct = 0.0008  # ROUND_TRIP_FEE_PCT, the shared default
    effective_risk_pct = 0.01 + fee_pct
    expected_notional = (0.01 / effective_risk_pct) * 10_000
    assert plan.notional_value == pytest.approx(expected_notional, rel=1e-6)
    assert plan.notional_value < 100_000, "smaller than the fee-blind 100,000 the old formula gave"

    price_loss = plan.position_size * (entry - stop)
    fee_loss = plan.notional_value * fee_pct
    assert (price_loss + fee_loss) == pytest.approx(plan.risk_amount, rel=1e-6)


def test_required_margin_scales_with_leverage():
    entry = 100.0
    stop = entry - entry * 0.008
    plan_10x = plan_position(
        equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long",
        leverage=10, round_trip_fee_pct=0.0,
    )
    plan_5x = plan_position(
        equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long",
        leverage=5, round_trip_fee_pct=0.0,
    )

    assert plan_10x.required_margin == pytest.approx(250, rel=1e-3)
    assert plan_5x.required_margin == pytest.approx(500, rel=1e-3)
    assert plan_10x.notional_value == pytest.approx(plan_5x.notional_value)


def test_long_position_take_profit_uses_default_1_to_3_ratio():
    plan = plan_position(equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=95, direction="long")
    assert plan.take_profit == pytest.approx(115)  # 5 risk * 3 reward:risk


def test_short_position_take_profit():
    plan = plan_position(
        equity=10_000, risk_pct=0.02, entry_price=100, stop_loss=105, direction="short", reward_risk_ratio=2.0
    )
    assert plan.take_profit == pytest.approx(90)


def test_invalid_direction_raises():
    with pytest.raises(ValueError):
        plan_position(equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=95, direction="sideways")


def test_zero_risk_raises():
    with pytest.raises(ValueError):
        plan_position(equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=100, direction="long")


def test_risk_pct_above_2_percent_raises():
    with pytest.raises(ValueError):
        plan_position(equity=10_000, risk_pct=0.03, entry_price=100, stop_loss=95, direction="long")


def test_risk_pct_zero_or_negative_raises():
    with pytest.raises(ValueError):
        plan_position(equity=10_000, risk_pct=0, entry_price=100, stop_loss=95, direction="long")


def test_non_positive_leverage_raises():
    with pytest.raises(ValueError):
        plan_position(equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=95, direction="long", leverage=0)


def test_dynamic_leverage_fits_within_full_equity_by_default():
    # notional = 2500 (per the worked example), equity = 1000 -> would only
    # need 2.5x to fit within available capital, but the 10x floor wins.
    entry = 100.0
    stop = entry - entry * 0.008
    plan = plan_position(
        equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long", round_trip_fee_pct=0.0,
    )

    assert plan.leverage == pytest.approx(10.0, rel=1e-3)
    assert plan.required_margin == pytest.approx(250, rel=1e-3)


def test_dynamic_leverage_accounts_for_committed_margin():
    entry = 100.0
    stop = entry - entry * 0.008
    # only 200 of the 1000 equity is still free (800 tied up in other trades)
    plan = plan_position(
        equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long",
        available_budget=200, round_trip_fee_pct=0.0,
    )

    assert plan.leverage == pytest.approx(2500 / 200, rel=1e-3)
    assert plan.required_margin == pytest.approx(200, rel=1e-3)


def test_dynamic_leverage_never_goes_below_10x():
    # plenty of budget available -> would need no leverage at all, but the
    # 10x floor still applies so capital stays free for other trades.
    plan = plan_position(
        equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=95, direction="long", round_trip_fee_pct=0.0,
    )
    assert plan.leverage == 10.0
    assert plan.required_margin == pytest.approx(200, rel=1e-3)


def test_dynamic_leverage_capped_and_raises_if_still_insufficient():
    entry = 100.0
    stop = entry - entry * 0.008  # notional will be 2500 given 2% risk on 1000 equity
    with pytest.raises(ValueError):
        # only 50 available, needs 50x leverage to fit -> exceeds the 20x cap
        plan_position(
            equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long", available_budget=50
        )


def test_dynamic_leverage_raises_when_no_budget_left():
    with pytest.raises(ValueError):
        plan_position(equity=1000, risk_pct=0.01, entry_price=100, stop_loss=95, direction="long", available_budget=0)


def test_a_symbol_capped_below_the_leverage_floor_is_sized_to_its_cap():
    """BTWUSDT, live 2026-08-21: "Exceeded the maximum settable leverage"
    (Bitget 40797). Its maxLever is 5 and the bot asked for 10.

    MIN_LEVERAGE is a FLOOR of 10, so on any symbol capped below that the plan
    could only ever ask for something the exchange refuses - the trade could
    never be placed, however much free margin the account had. 17 of the 759
    contracts are under 10x, mostly tokenized stocks (XIAOMI, MEITUAN,
    NETEASE, KUAISHOU, SMIC) plus HUSDT at 4x.

    The cap has to win over the floor: ask for 5 on a 5x symbol.
    """
    plan = plan_position(
        equity=1000.0,
        risk_pct=0.01,
        entry_price=0.449761,
        stop_loss=0.454541,
        direction="short",
        available_budget=1000.0,
        max_leverage=5.0,
    )
    assert plan.leverage <= 5.0, "the exchange cap must beat the MIN_LEVERAGE floor"
    assert plan.leverage > 0
    # and the margin has to be priced at the leverage actually used, or the
    # account commits less than the position really needs
    assert plan.required_margin == pytest.approx(plan.notional_value / plan.leverage)
