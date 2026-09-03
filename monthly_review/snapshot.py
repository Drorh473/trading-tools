"""Account equity as it stood when a monthly report last ran.

Bitget's API answers "what is equity NOW". There is no endpoint for "what was
it on the 1st", so a month-over-month balance line cannot be reconstructed
after the fact - it has to be recorded as it happens. This is that record.

Deliberately a plain file next to the trades DB, for the same reason
weekly_review.heartbeat is: it must survive, and be readable, even when the
database is the broken thing the report is trying to describe.

The first run has nothing to compare against, and says so. That is a real
state, not an error - exactly as with the weekly heartbeat's never-run case.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


def _path(db_path: str) -> Path:
    return Path(db_path).parent / "monthly_equity_snapshot"


def record(db_path: str, equity: float, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"at": now.isoformat(), "equity": equity}), encoding="utf-8")


def previous(db_path: str) -> tuple[datetime, float] | None:
    """The last recorded (when, equity), or None if nothing has been recorded.

    A corrupt or half-written file reads as None rather than raising: a balance
    line the report cannot produce is worth saying out loud, but it is not
    worth losing the other sections over.
    """
    try:
        payload = json.loads(_path(db_path).read_text(encoding="utf-8"))
        return datetime.fromisoformat(payload["at"]), float(payload["equity"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
