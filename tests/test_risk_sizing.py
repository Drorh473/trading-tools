import pytest

from notifier.risk_sizing import plan_position


def test_matches_worked_example_from_the_rule():
    # From the user's rule: stop 0.8%, risk 2%, $1000 equity -> Y=2.5, notional $2500
    entry = 100.0
    stop = entry - entry * 0.008  # 0.8% below entry
    plan = plan_position(equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long")

    assert plan.notional_value == pytest.approx(2500, rel=1e-3)
    assert plan.risk_amount == pytest.approx(20)  # 2% of $1000


def test_required_margin_scales_with_leverage():
    entry = 100.0
    stop = entry - entry * 0.008
    plan_10x = plan_position(equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long", leverage=10)
    plan_5x = plan_position(equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long", leverage=5)

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
    # notional = 2500 (per the worked example), equity = 1000 -> needs 2.5x
    # to fit the whole notional value within available capital.
    entry = 100.0
    stop = entry - entry * 0.008
    plan = plan_position(equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long")

    assert plan.leverage == pytest.approx(2.5, rel=1e-3)
    assert plan.required_margin == pytest.approx(1000, rel=1e-3)


def test_dynamic_leverage_accounts_for_committed_margin():
    entry = 100.0
    stop = entry - entry * 0.008
    # only 200 of the 1000 equity is still free (800 tied up in other trades)
    plan = plan_position(
        equity=1000, risk_pct=0.02, entry_price=entry, stop_loss=stop, direction="long", available_budget=200
    )

    assert plan.leverage == pytest.approx(2500 / 200, rel=1e-3)
    assert plan.required_margin == pytest.approx(200, rel=1e-3)


def test_dynamic_leverage_never_goes_below_1x():
    # plenty of budget available -> no need for leverage at all
    plan = plan_position(equity=10_000, risk_pct=0.01, entry_price=100, stop_loss=95, direction="long")
    assert plan.leverage == 1.0


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
