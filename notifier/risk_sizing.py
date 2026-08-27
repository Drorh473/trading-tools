"""Position sizing, following the user's own risk rules:

- Risk only 1-2% of the portfolio per trade (hard cap — never more than 2%).
- Don't take a trade at worse than a 1:3 risk-reward ratio starting out
  (default here; the user may lower it deliberately later once they've
  accumulated enough statistics to justify it).
- Position size: divide the risk % by the stop-loss %-distance to get a
  multiplier Y, then Y * equity gives the notional position value. Actual
  capital required is that notional value divided by leverage.

Leverage is dynamic by default: rather than a fixed multiplier, it's solved
for so the required margin fits within whatever capital isn't already tied
up in other open trades (equity - committed_margin), so several trades can
run simultaneously without running out of margin. It's never let to drop
below MIN_LEVERAGE, though — even when the account has enough free capital
to cover a trade at 1x, the plan still uses at least 10x so that little
margin is tied up per position and more of it stays free for other trades.
Risk itself (risk_amount) is unaffected by leverage — it's always
equity * risk_pct regardless of how many trades are open, per the user's
per-trade risk rule.

This is embedded directly into each signal alert rather than exposed as a
separate command.
"""

from dataclasses import dataclass

MAX_RISK_PCT = 0.02
DEFAULT_REWARD_RISK_RATIO = 3.0
MIN_LEVERAGE = 10.0
DEFAULT_MAX_LEVERAGE = 20.0
# Bitget's maker/taker fees, shared across every strategy rather than each
# defining its own - they're an account-level fact, not a per-strategy
# choice. Previously only ema_trend_v2.py had a copy, used just for its own
# FillGuard's net reward:risk gate; plan_position never accounted for fees at
# all, which is why a CLEAN stop-out (zero slippage) still read worse than
# -1.00R - the size being risked was solved against pure price distance, and
# fees ate into the realized loss on top of that. Dror, 2026-08-26, after
# checking AIOUSDT #68 and 26 other closed losers: "we can compute the stop
# include the fees and make it 1r".
MAKER_FEE_PCT = 0.0002
TAKER_FEE_PCT = 0.0006
# The exit leg is ALWAYS taker, for every strategy: a stop-loss triggers a
# market order once hit, regardless of how the position was entered. Only the
# entry leg varies by strategy - see round_trip_fee_for.
ROUND_TRIP_FEE_PCT = MAKER_FEE_PCT + TAKER_FEE_PCT  # the pure-limit-entry case (Strategy 4)


def round_trip_fee_for(market_fraction: float) -> float:
    """The true round-trip fee for a signal whose entry is this fraction market.

    Sizing every strategy against the same flat ROUND_TRIP_FEE_PCT (maker in,
    taker out) fixed the systemic bias for strategies close to that mix, but
    left Strategy 2.1 - entered 100% at market under ENTRY_MODE="next_open" -
    still under-sized: its true fee is taker in AND taker out, 0.12%, 50% more
    than the 0.08% shared constant assumed. This computes the entry leg from
    the ACTUAL split the signal carries (0.0 = Strategy 4's pure limit, 1.0 =
    Strategy 2.1's pure market, 0.2 = the default split entry) instead of
    guessing at one number for every strategy.
    """
    entry_fee_pct = market_fraction * TAKER_FEE_PCT + (1 - market_fraction) * MAKER_FEE_PCT
    return entry_fee_pct + TAKER_FEE_PCT


@dataclass
class PositionPlan:
    position_size: float  # in units of the asset
    notional_value: float  # position size expressed in quote currency ($)
    required_margin: float  # actual capital needed, after leverage
    leverage: float  # the leverage this plan used (given or auto-computed)
    risk_amount: float  # $ actually at risk if the stop is hit
    take_profit: float


def plan_position(
    equity: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    direction: str,
    reward_risk_ratio: float = DEFAULT_REWARD_RISK_RATIO,
    leverage: float | None = None,
    available_budget: float | None = None,
    max_leverage: float = DEFAULT_MAX_LEVERAGE,
    round_trip_fee_pct: float = ROUND_TRIP_FEE_PCT,
) -> PositionPlan:
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    if not (0 < risk_pct <= MAX_RISK_PCT):
        raise ValueError(
            f"risk_pct must be between 0 and {MAX_RISK_PCT:.0%} (risk only 1-2% per trade), got {risk_pct:.2%}"
        )

    price_risk = abs(entry_price - stop_loss)
    if price_risk == 0:
        raise ValueError("stop_loss cannot equal entry_price")

    # Sized against price_risk PLUS the round-trip fee expressed in price
    # terms, so a CLEAN stop-out - no slippage - actually costs risk_pct of
    # equity instead of risk_pct plus whatever the fee quietly added on top.
    # Before this, every stop-out read worse than -1.00R even with a perfect
    # fill; see ROUND_TRIP_FEE_PCT's own comment for the trades that showed
    # it. take_profit below still measures its reward off the raw
    # price_risk - this is a sizing correction, not a change to where the
    # strategy's own reward:risk ratio is targeted.
    effective_risk = price_risk + entry_price * round_trip_fee_pct
    stop_pct = effective_risk / entry_price
    multiplier = risk_pct / stop_pct  # "Y" in the user's rule

    notional_value = multiplier * equity
    position_size = notional_value / entry_price
    risk_amount = equity * risk_pct

    budget = equity if available_budget is None else available_budget

    if leverage is None:
        if budget <= 0:
            raise ValueError("No available margin budget for a new trade (equity already committed elsewhere)")
        leverage = min(max(notional_value / budget, MIN_LEVERAGE), max_leverage)
    elif leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")

    required_margin = notional_value / leverage
    if required_margin > budget + 1e-9:
        raise ValueError(
            f"Required margin {required_margin:.2f} exceeds available budget {budget:.2f} "
            f"even at max leverage {max_leverage:g}x — not enough free capital for this trade"
        )

    reward_per_unit = price_risk * reward_risk_ratio
    take_profit = entry_price + reward_per_unit if direction == "long" else entry_price - reward_per_unit

    return PositionPlan(
        position_size=position_size,
        notional_value=notional_value,
        required_margin=required_margin,
        leverage=leverage,
        risk_amount=risk_amount,
        take_profit=take_profit,
    )
