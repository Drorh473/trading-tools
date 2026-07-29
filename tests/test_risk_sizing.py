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
