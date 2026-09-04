"""Account equity as it stood each time a monthly report ran.

Bitget's API answers "what is equity NOW". There is no endpoint for "what was
it on the 1st", so a month-over-month balance line cannot be reconstructed
after the fact - it has to be recorded as it happens. This is that record.

Kept as a growing history, not just the latest value: monthly_review itself
only ever needs `previous()` (the last entry), but the same recording is the
only source yearly_review has for a monthly equity curve - Bitget cannot
answer "what was equity in March" any more than it can answer "what was it
on the 1st". A value that gets overwritten every month would have thrown that
away before a yearly report ever got to read it. `history()` is that read.

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


def _read_entries(path: Path) -> list[dict]:
    """Every recorded (at, equity) pair, oldest first - or [] if the file is
    missing, corrupt, or half-written. A report that cannot read this is
    worth losing a balance line over, not worth crashing over.

    Reads the file this module wrote before history existed too: that file
    was a single {"at", "equity"} object rather than {"history": [...]}, and
    every one already on the VM is in that shape. Treating it as a one-entry
    history rather than raising means the very first run after this changed
    still has SOMETHING to compare against, and the next record() upgrades
    the file to the new shape on its own.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(payload, dict) and isinstance(payload.get("history"), list):
        return payload["history"]
    if isinstance(payload, dict) and "at" in payload and "equity" in payload:
        return [payload]
    return []


def record(db_path: str, equity: float, now: datetime | None = None) -> None:
    now = now or datetime.now(timezone.utc)
    path = _path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = _read_entries(path)
    entries.append({"at": now.isoformat(), "equity": equity})
    path.write_text(json.dumps({"history": entries}), encoding="utf-8")


def previous(db_path: str) -> tuple[datetime, float] | None:
    """The last recorded (when, equity), or None if nothing has been recorded.

    A corrupt or half-written file reads as None rather than raising: a
    balance line the report cannot produce is worth saying out loud, but it
    is not worth losing the other sections over.
    """
    entries = _read_entries(_path(db_path))
    if not entries:
        return None
    try:
        last = entries[-1]
        return datetime.fromisoformat(last["at"]), float(last["equity"])
    except (KeyError, ValueError, TypeError):
        return None


def history(db_path: str) -> list[tuple[datetime, float]]:
    """Every (when, equity) this module has ever recorded, oldest first.

    One entry per monthly report that successfully read equity - so roughly
    one point a month, at whatever cadence the monthly cron actually ran at,
    with no entry for a month the job failed before reaching snapshot.record.
    A caller wanting a specific window (yearly_review, for one) filters this
    itself; nothing here assumes what range it will be read over.
    """
    out = []
    for entry in _read_entries(_path(db_path)):
        try:
            out.append((datetime.fromisoformat(entry["at"]), float(entry["equity"])))
        except (KeyError, ValueError, TypeError):
            continue
    return out
