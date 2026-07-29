"""Position sizing, following the user's own risk rules:

- Risk only 1-2% of the portfolio per trade (hard cap — never more than 2%).
- Don't take a trade at worse than a 1:3 risk-reward ratio starting out
  (default here; the user may lower it deliberately later once they've
  accumulated enough statistics to justify it).
- Position size: divide the risk % by the stop-loss %-distance to get a
  multiplier Y, then Y * equity gives the notional position value. Actual
  capital required is that notional value divided by leverage.

This is embedded directly into each signal alert rather than exposed as a
separate command.
"""

from dataclasses import dataclass

MAX_RISK_PCT = 0.02
DEFAULT_REWARD_RISK_RATIO = 3.0


@dataclass
class PositionPlan:
    position_size: float  # in units of the asset
    notional_value: float  # position size expressed in quote currency ($)
    required_margin: float  # actual capital needed, after leverage
    risk_amount: float  # $ actually at risk if the stop is hit
    take_profit: float


def plan_position(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    direction: str,
    reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO,
    leverage: float = 1.0,
) -> PositionPlan:
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if not (0 < risk_pct <= MAX_RISK_PCT):
        raise ValueError(
            f"risk_pct must be between 0 and {MAX_RISK_PCT:.0%} (risk only 1-2% per trade), got {risk_pct:.2%}"
        )

    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")

    price_risk = abs(entry_price - stop_loss)
    if price_risk == 0:
        raise ValueError("stop_loss cannot equal entry_price")

    stop_pct = price_risk / entry_price
    multiplier = risk_pct / stop_pct  # "Y" in the user's rule

    notional_value = multiplier * equity
    position_size = notional_value / entry_price
    required_margin = notional_value / leverage
    risk_amount = equity * risk_pct

    reward_per_unit = price_risk * reward_risk_ratio
    take_profit = entry_price + reward_per_unit if direction == "long" else entry_price - reward_per_unit

    return PositionPlan(
        position_size=position_size,
        notional_value=notional_value,
        required_margin=required_margin,
        risk_amount=risk_amount,
        take_profit=take_profit,
    )
