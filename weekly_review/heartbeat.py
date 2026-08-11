"""When the weekly report last succeeded.

The report ran on schedule every Sunday from 2026-08-02 and crashed before
sending, every time, for two weeks. Nothing noticed: cron appends to a log file
nobody reads, and a report that never arrives looks exactly like a quiet week.
Dror found it by missing it.

So success is recorded, and the always-on scanner checks the record. That
covers the two failures separately - the report shouting when it breaks covers
a crash, and this covers the case where it never runs at all, which no amount
of error handling inside the job can catch.

Deliberately a plain file next to the trades DB rather than a table: it must
be readable even if the database is the thing that is broken.
"""

from datetime import datetime, timezone
from pathlib import Path


def _path(db_path: str) -> Path:
    return Path(db_path).parent / "weekly_review_last_success"


def record_success(db_path: str, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(now.isoformat(), encoding="utf-8")


def last_success(db_path: str) -> datetime | None:
    """None when it has never succeeded - which is a real state, not an error.
    That was true for the first two weeks this report existed.
    """
    path = _path(db_path)
    try:
        return datetime.fromisoformat(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def overdue_by(db_path: str, max_age_days: float, now: datetime | None = None) -> float | None:
    """How many days late the report is, or None if it is not late.

    The window is deliberately wider than the seven-day cadence: a report that
    ran an hour late, or a clock that drifted, is not worth an alert. Only a
    genuinely missed run is.
    """
    now = now or datetime.now(timezone.utc)
    seen = last_success(db_path)
    if seen is None:
        return None  # never run - the caller decides what to make of that
    age_days = (now - seen).total_seconds() / 86400
    return age_days - max_age_days if age_days > max_age_days else None
