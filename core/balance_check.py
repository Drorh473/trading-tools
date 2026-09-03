"""Does the exchange balance still match what the bot thinks it did?

THE IDEA: reconcile FLAT TO FLAT.

The monthly report reconciles over a calendar month, which means positions are
usually open at one end or both, and their unrealized P&L lands in the
residual legitimately. That makes a small residual unreadable - it could be a
real leak or it could be an open trade - so the monthly version can only speak
confidently when the books happen to be flat at the boundary.

Between two moments when the account holds NOTHING, that ambiguity disappears.
Every dollar that moved has been realized into a trade row, or is a fee, or is
funding. Anything left over is money that moved for a reason nothing in this
codebase modelled, and it can be said so immediately rather than on the 1st.

WHY THE CHECKPOINT CARRIES CUMULATIVE P&L RATHER THAN A DATE

There is no exit timestamp on a trade row - תאריך is the ENTRY date - so
"realized since the checkpoint" cannot be summed by date without inventing
one. Carrying the running total instead makes the delta a subtraction, and
sidesteps the missing column entirely. Fees and funding do come from
time-windowed API calls, which is why the checkpoint keeps a timestamp too.

WHAT THIS CANNOT SEE

If the account is never flat, this never runs. Nine live instances can keep
something open for a long time, and the check simply waits - it does not
degrade into a guess. The monthly reconciliation stays the backstop for
exactly that case.
"""

import json
from dataclasses import dataclass
from pathlib import Path

# How far the books may disagree with the exchange before it is worth saying.
#
# A JUDGEMENT, not a measurement, and deliberately in one place so it can be
# argued with. Two things push it up from zero: Bitget reports balances to
# more decimal places than the trade rows carry, and a fee or funding row can
# settle either side of the instant equity is read. Two things push it down:
# this account is around $100, so a dollar is a percent of everything, and the
# whole reason for the check is that a small persistent leak is invisible in
# any single week.
#
# Absolute rather than proportional: the errors being looked for - a fee that
# is not modelled, a transfer nobody recorded - do not scale with equity.
DIVERGENCE_TOLERANCE = 0.25

# Minimum gap between two checkpoints. Without it, a quiet flat account would
# re-run the fee and funding calls on every upkeep cycle to reconcile a window
# containing nothing.
MIN_INTERVAL_SECONDS = 3600.0


@dataclass(frozen=True)
class Checkpoint:
    at_ms: int
    equity: float
    cumulative_realized: float


@dataclass(frozen=True)
class Divergence:
    """The result of one flat-to-flat reconciliation."""

    previous: Checkpoint
    equity_now: float
    cumulative_realized_now: float
    fees: float
    funding: float

    @property
    def realized(self) -> float:
        return self.cumulative_realized_now - self.previous.cumulative_realized

    @property
    def explained(self) -> float:
        return self.realized - self.fees - self.funding

    @property
    def actual(self) -> float:
        return self.equity_now - self.previous.equity

    @property
    def residual(self) -> float:
        return self.actual - self.explained

    def alarming(self, tolerance: float = DIVERGENCE_TOLERANCE) -> bool:
        return abs(self.residual) > tolerance


def _path(db_path: str) -> Path:
    return Path(db_path).parent / "balance_checkpoint"


def load(db_path: str) -> Checkpoint | None:
    """The last flat checkpoint, or None if there has never been one.

    A corrupt file reads as None: losing one interval of reconciliation is a
    smaller harm than a scanner loop that raises on every cycle.
    """
    try:
        payload = json.loads(_path(db_path).read_text(encoding="utf-8"))
        return Checkpoint(
            at_ms=int(payload["at_ms"]),
            equity=float(payload["equity"]),
            cumulative_realized=float(payload["cumulative_realized"]),
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save(db_path: str, checkpoint: Checkpoint) -> None:
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "at_ms": checkpoint.at_ms,
                "equity": checkpoint.equity,
                "cumulative_realized": checkpoint.cumulative_realized,
            }
        ),
        encoding="utf-8",
    )


def cumulative_realized(trades) -> float:
    """Running total of realized P&L across every closed trade ever.

    Every closed trade, not the window's - the point is that subtracting two
    of these gives the window's realized P&L without needing an exit date.
    """
    return sum(t.רווח_הפסד or 0.0 for t in trades if t.is_closed)


def describe(divergence: Divergence) -> str:
    d = divergence
    return (
        f"BALANCE DIVERGENCE of {d.residual:+.2f} USDT.\n\n"
        f"The account was flat, then flat again, so nothing is unrealized and "
        f"every move should be accounted for:\n"
        f"  realized P&L   {d.realized:+.2f}\n"
        f"  fees           {-d.fees:+.2f}\n"
        f"  funding        {-d.funding:+.2f}\n"
        f"  = expected     {d.explained:+.2f}\n"
        f"  actual equity  {d.actual:+.2f}  (${d.previous.equity:.2f} -> ${d.equity_now:.2f})\n\n"
        f"Something moved the balance that the bot did not model."
    )
