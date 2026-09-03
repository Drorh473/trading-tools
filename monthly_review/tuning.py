"""Slow leaks: costs the backtest assumes away, measured against what the
account actually paid.

Both numbers here were modelled as constants and never once checked against a
live month. That is the exact shape of the Strategy 2.1 bug - sized against a
flat 0.08% while really paying 0.12% - which survived for months because
nothing compared the assumption to the bill.

Neither can be seen weekly. A week holds a handful of fills, and the spread on
per-fill slippage is far wider than the drift being looked for.
"""

from dataclasses import dataclass

from core.storage import SignalRecord, Trade
from monthly_review.noise import Finding, check_mean

# The backtest charges NO entry slippage - backtest/engine.py, "No slippage
# beyond the fee model". So the baseline every live fill is tested against is
# a flat zero, and any finding that fires is the backtest being optimistic by
# a measurable amount rather than two estimates disagreeing.
MODELLED_ENTRY_SLIPPAGE_BP = 0.0

# How far the real fee bill may sit from the modelled one before the report
# says so. Not a noise-band test: this compares two TOTALS over the same
# fills, not two samples, so there is nothing to be uncertain about beyond
# rounding and the handful of trades whose tag the estimator cannot map.
FEE_MODEL_TOLERANCE = 0.15


@dataclass(frozen=True)
class FeeModelCheck:
    actual: float
    modelled: float
    tolerance: float = FEE_MODEL_TOLERANCE

    @property
    def ratio(self) -> float | None:
        """actual / modelled. None when nothing was modelled - a month with no
        closed trades, where the honest answer is not 1.0."""
        return self.actual / self.modelled if self.modelled else None

    @property
    def fires(self) -> bool:
        ratio = self.ratio
        return ratio is not None and abs(ratio - 1.0) > self.tolerance


def entry_slippage_bp(signal: SignalRecord, trade: Trade) -> float | None:
    """How far the fill landed from the price the signal planned, in basis
    points, signed as a COST: positive means the fill was worse.

    Worse is direction-dependent, which is the whole reason this is not a bare
    subtraction - filling ABOVE plan is a cost to a long and a gift to a
    short, and averaging the unsigned difference would report a strategy that
    consistently fills favourably as though it were bleeding.
    """
    planned = signal.entry_price
    actual = trade.מחיר_כניסה
    if not planned or not actual:
        return None
    direction = (trade.כיוון or signal.direction or "").lower()
    sign = 1.0 if direction.startswith("long") else -1.0
    return sign * (actual - planned) / planned * 10_000.0


def slippage_findings(
    signals: list[SignalRecord],
    trades_by_id: dict[int, Trade],
    num_tests: int,
) -> dict[str, Finding]:
    """Per instance: is the month's average entry slippage distinguishable
    from the zero the backtest assumes?

    Fires only outside the band, per Dror 2026-09-03. An instance whose
    slippage is real but whose sample is small comes back quiet, carrying the
    smallest cost it could have detected - which is the number that says
    whether waiting another month is worth it.
    """
    by_tag: dict[str, list[float]] = {}
    for signal in signals:
        trade = trades_by_id.get(signal.trade_id) if signal.trade_id else None
        if trade is None:
            continue
        slip = entry_slippage_bp(signal, trade)
        if slip is not None:
            by_tag.setdefault(signal.strategy_tag, []).append(slip)

    return {
        tag: check_mean(f"{tag} entry slippage (bp)", values, MODELLED_ENTRY_SLIPPAGE_BP, num_tests)
        for tag, values in sorted(by_tag.items())
    }
