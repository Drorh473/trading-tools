"""Did each instance keep signalling all month, or did it go quiet?

The scanner ALREADY watches for silence, continuously, per instance, against
thresholds calibrated to each instance's observed gaps - see
notifier.main.signal_silence_days and core.ledger. Nothing here duplicates
that, and the monthly report must not re-alert on what the scanner already
alerted on at the time.

What the ledger cannot do is history. It stores one timestamp per capability -
the last success - so it answers "is this instance alive right now" and
nothing else. Two very different months look identical to it:

    an instance that ran normally for 30 days
    an instance that went dark for 3 weeks and woke up yesterday

This reads the signals table instead, which is a real time series, and
reconstructs every gap in the month. Same thresholds, so the two never
disagree about what "too quiet" means for a given instance - only the window
differs.
"""

from dataclasses import dataclass
from datetime import date, datetime

from core.storage import SignalRecord
from monthly_review.window import LOCAL_TZ


@dataclass(frozen=True)
class Cadence:
    tag: str
    signals: int
    longest_silence_days: float
    breaches: int  # separate stretches that exceeded this instance's threshold
    threshold_days: float

    @property
    def silent_all_month(self) -> bool:
        """Never produced a signal in the whole window. Strategy 3 shipped
        live and has produced zero signals in the entire signals table - the
        standing example of a capability that is live, believed working, and
        has never once fired."""
        return self.signals == 0

    @property
    def alarming(self) -> bool:
        return self.silent_all_month or self.breaches > 0


def _parsed(at: str) -> datetime:
    parsed = datetime.fromisoformat(at)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=LOCAL_TZ)


def cadence_for(
    tag: str,
    signals: list[SignalRecord],
    start: date,
    end: date,
    threshold_days: float,
) -> Cadence:
    """Gaps are measured from the window's own edges, not just between
    signals: an instance whose last signal was on the 2nd and whose next came
    after the month ended has a 28-day silence, and measuring only
    signal-to-signal would score that as no gap at all.
    """
    window_start = datetime(start.year, start.month, start.day, tzinfo=LOCAL_TZ)
    window_end = datetime(end.year, end.month, end.day, tzinfo=LOCAL_TZ)

    mine = sorted(_parsed(s.dispatched_at) for s in signals if s.strategy_tag == tag)
    marks = [window_start] + mine + [window_end]

    gaps = [
        (later - earlier).total_seconds() / 86400.0
        for earlier, later in zip(marks, marks[1:])
    ]
    return Cadence(
        tag=tag,
        signals=len(mine),
        longest_silence_days=max(gaps) if gaps else 0.0,
        breaches=sum(1 for gap in gaps if gap > threshold_days),
        threshold_days=threshold_days,
    )
