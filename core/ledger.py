"""When each capability last actually worked.

Three features shipped and never worked once, and all three were found the same
way - Dror noticed. Not a test, not an alert, not a log line anybody read.

  - The partial take-profit, from 2026-08-03 to 2026-08-13. Every attempt was
    rejected, most with 22002 against positions that demonstrably existed. Not
    one automated target ever reached the exchange, and he had been setting
    every one by hand without either of us realising the bot had never managed
    it. Each individual failure looked transient; only the whole log at once
    showed a 100% failure rate.
  - The weekly report, from 2026-08-02. The cron fired every Sunday and died
    before sending, every time. He found it by missing it.
  - The untracked-position dedupe, held in memory, so every restart forgot.

They have one shape: a capability that has NEVER succeeded looks exactly like a
capability with nothing to do. A quiet week and a broken report are the same
observation. No amount of error handling inside a job closes that, because the
failure mode includes the job not running at all.

So success is recorded, and two questions can then be asked separately:

  ever()      has this EVER worked? - the five-month class
  overdue()   has it worked LATELY? - the stopped-working class

"Never" is a first-class state rather than an error, because it is what the
first two weeks of any new capability look like. Callers decide what to make of
it; see heartbeat.overdue_by, whose reasoning this follows.

Storage is one plain file per capability under data/ledger/, matching
weekly_review.heartbeat and for its stated reason: it must be readable even if
the database is the thing that is broken. One file per capability rather than a
single keyed document is deliberate too - the scanner and the weekly cron are
different processes, and a shared document is a read-modify-write race that
could silently lose exactly the update this module exists to keep.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# Capability names carry strategy tags, which contain spaces and slashes
# ("Strategy 3 1D/5m"). Anything outside this set becomes an underscore, so
# the name is still legible in a directory listing - the point is that a human
# scanning data/ledger/ can see what is and is not being recorded.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")

# The capability names the bot records under. Kept here rather than spelled at
# each call site so the weekly report can ask about a capability that has never
# fired even once - which is the entire point, and impossible if the name only
# comes into existence the first time it succeeds.
TAKE_PROFIT_PLACED = "take_profit_placed"
BREAKEVEN_STOP_MOVED = "breakeven_stop_moved"
ENTRY_ORDER_PLACED = "entry_order_placed"
WEEKLY_REPORT = "weekly_report"
# The monthly review. It needs its own watcher for a sharper version of the
# reason the weekly one has three: a weekly report that stops arriving is
# noticed within a fortnight, and that already went unnoticed for two weeks. A
# MONTHLY report that stops arriving looks exactly like an ordinary gap between
# months, so nothing about its absence is remarkable until a second one fails
# to show - two months later.
MONTHLY_REPORT = "monthly_report"
# The yearly review. Sharper again than the monthly watcher, for the same
# reason the monthly one is sharper than the weekly: a YEARLY report that
# stops arriving looks exactly like an ordinary gap between years, and
# nothing about its absence is remarkable until a second one fails to show -
# two years later.
YEARLY_REPORT = "yearly_report"
# The runner's stop being ratcheted to a confirmed swing. Its own capability
# rather than folded into BREAKEVEN_STOP_MOVED, because the two fail
# differently: a breakeven happens ONCE per trade and its absence is visible,
# while a trail that silently stops working looks exactly like a trade that
# never trended. The trail also only reached the exchange for the first time on
# 2026-08-17 - before that it sent no size and Bitget rejected it with 40019,
# so "has this EVER worked" is a live question for it rather than a formality.
TRAILING_STOP_MOVED = "trailing_stop_moved"


def signal_seen(strategy_tag: str) -> str:
    return f"signal.{strategy_tag}"


def order_placed(strategy_tag: str) -> str:
    return f"order.{strategy_tag}"


def armed(strategy_tag: str) -> str:
    return f"armed.{strategy_tag}"


def _dir(db_path: str) -> Path:
    return Path(db_path).parent / "ledger"


def _path(db_path: str, capability: str) -> Path:
    return _dir(db_path) / _UNSAFE.sub("_", capability)


def record(db_path: str, capability: str, now: datetime | None = None) -> None:
    """Note that this capability just worked.

    Deliberately swallows nothing: if the ledger cannot be written the caller
    should hear about it, because a ledger that silently stops recording
    recreates the very blindness it was built to remove.
    """
    now = now or datetime.now(timezone.utc)
    path = _path(db_path, capability)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(now.isoformat(), encoding="utf-8")


def try_record(db_path: str, capability: str, now: datetime | None = None) -> None:
    """record(), for use INSIDE the trading path.

    A stop that could not be moved to breakeven because a bookkeeping file was
    unwritable would be an absurd way to lose money, so here the write is
    allowed to fail. It is logged rather than silenced: the failure mode is
    then a capability that looks stale while actually working, which nags the
    weekly report - the safe direction. The opposite arrangement, where a
    ledger error can abort an exit, is not.
    """
    try:
        record(db_path, capability, now)
    except OSError:
        logger.exception("Could not record ledger success for %s", capability)


def last_success(db_path: str, capability: str) -> datetime | None:
    """None means no record - which is a real state, not an error."""
    try:
        return datetime.fromisoformat(_path(db_path, capability).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def ever(db_path: str, capability: str) -> bool:
    return last_success(db_path, capability) is not None


def age_days(db_path: str, capability: str, now: datetime | None = None) -> float | None:
    now = now or datetime.now(timezone.utc)
    seen = last_success(db_path, capability)
    if seen is None:
        return None
    return (now - seen).total_seconds() / 86400


_WATCH_STARTED = ".watch_started"


def began_watching(db_path: str, now: datetime | None = None) -> datetime:
    """When this ledger started watching, established on first use.

    Without it the first report after any deploy declares every capability
    broken, because none of them has had a chance to work yet - thirteen
    NEVER-worked lines on day one, which is how a report stops being read.
    That is the same failure as the five-minute Telegram expiry: an alert so
    noisy it trains you to ignore it is not a working alert.

    heartbeat.overdue_by solved this for one job by refusing to call a report
    that has never run "late". This is the general form.
    """
    path = _dir(db_path) / _WATCH_STARTED
    try:
        return datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        started = now or datetime.now(timezone.utc)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(started.isoformat(), encoding="utf-8")
        except OSError:
            logger.exception("Could not record when ledger watching began")
        return started


@dataclass(frozen=True)
class Status:
    capability: str
    last: datetime | None
    age: float | None       # days since it last worked, None if never
    max_age: float          # what was expected of it
    never: bool             # has not worked once since it was first watched
    overdue: bool           # worked before, but not lately
    watched_for: float      # days this ledger has been watching at all

    @property
    def alarming(self) -> bool:
        """A capability that has never worked is only news once it has had
        long enough to work. Before that, "never" is just "new"."""
        if self.never:
            return self.watched_for > self.max_age
        return self.overdue


def survey(db_path: str, expectations: dict[str, float],
           now: datetime | None = None) -> list[Status]:
    """Every watched capability, worst first.

    expectations maps a capability to how many days may pass before its silence
    means something. That number is a judgement about the CADENCE of the thing
    being watched, not a technical constant: a weekly report is late after
    eight days, while a 1D strategy going ten days without a setup may be
    perfectly ordinary. Getting it wrong in the tight direction produces alert
    fatigue, which is its own failure - the five-minute Telegram expiry taught
    that already.
    """
    now = now or datetime.now(timezone.utc)
    watched_for = (now - began_watching(db_path, now)).total_seconds() / 86400
    out = []
    for capability, max_age in expectations.items():
        age = age_days(db_path, capability, now)
        out.append(Status(
            capability=capability,
            last=last_success(db_path, capability),
            age=age,
            max_age=max_age,
            never=age is None,
            overdue=age is not None and age > max_age,
            watched_for=watched_for,
        ))
    # Never-worked first: five months of a silently broken take-profit is a
    # different and worse thing than a report that is a day late.
    return sorted(out, key=lambda s: (not s.never, -(s.age or 0.0)))


def format_survey(statuses: list[Status]) -> str:
    """The weekly report's liveness section.

    Says the quiet part out loud - a capability with no record at all gets the
    words "never worked", because "no data" is how this hid for five months.
    """
    alarming = [s for s in statuses if s.alarming]
    if not alarming:
        return "All watched capabilities have worked recently."

    lines = ["*Capabilities that need looking at*"]
    for s in alarming:
        if s.never:
            lines.append(f"- `{s.capability}` — has NEVER worked, in {s.watched_for:.0f} days of watching")
        else:
            lines.append(f"- `{s.capability}` — last worked {s.age:.1f} days ago "
                         f"(expected within {s.max_age:g})")
    return "\n".join(lines)
