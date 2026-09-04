"""monthly_review.snapshot: does the equity history actually accumulate, and
does a pre-history file (the format every monthly_equity_snapshot on the VM
was written in before history() existed) still read correctly?
"""

import json
from datetime import datetime
from pathlib import Path

from monthly_review import snapshot


def test_record_appends_rather_than_overwrites(tmp_path):
    db_path = str(tmp_path / "trades.db")
    snapshot.record(db_path, 100.0, now=datetime(2026, 1, 1, 9, 0))
    snapshot.record(db_path, 110.0, now=datetime(2026, 2, 1, 9, 0))
    snapshot.record(db_path, 108.0, now=datetime(2026, 3, 1, 9, 0))

    history = snapshot.history(db_path)

    assert [equity for _at, equity in history] == [100.0, 110.0, 108.0]
    assert snapshot.previous(db_path) == (datetime(2026, 3, 1, 9, 0), 108.0)


def test_a_legacy_single_value_file_still_reads(tmp_path):
    """Every file this module has ever written before history() existed was
    a bare {"at", "equity"} object, not {"history": [...]}. A read that
    cannot handle that would make the first run after this changed report
    'no snapshot from a previous run' for an account that plainly has one.
    """
    db_path = str(tmp_path / "trades.db")
    legacy_path = Path(db_path).parent / "monthly_equity_snapshot"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({"at": "2026-01-01T09:00:00", "equity": 100.0}), encoding="utf-8"
    )

    assert snapshot.previous(db_path) == (datetime(2026, 1, 1, 9, 0), 100.0)
    assert snapshot.history(db_path) == [(datetime(2026, 1, 1, 9, 0), 100.0)]


def test_recording_after_a_legacy_file_upgrades_it_in_place(tmp_path):
    db_path = str(tmp_path / "trades.db")
    legacy_path = Path(db_path).parent / "monthly_equity_snapshot"
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({"at": "2026-01-01T09:00:00", "equity": 100.0}), encoding="utf-8"
    )

    snapshot.record(db_path, 105.0, now=datetime(2026, 2, 1, 9, 0))

    history = snapshot.history(db_path)
    assert [equity for _at, equity in history] == [100.0, 105.0]


def test_a_corrupt_file_reads_as_empty_history(tmp_path):
    db_path = str(tmp_path / "trades.db")
    bad_path = Path(db_path).parent / "monthly_equity_snapshot"
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    bad_path.write_text("not json", encoding="utf-8")

    assert snapshot.history(db_path) == []
    assert snapshot.previous(db_path) is None
