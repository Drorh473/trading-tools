from core.reset_db import reset_db
from core.storage import Storage


def test_reset_db_archives_existing_file(tmp_path):
    db_path = tmp_path / "trades.db"
    Storage(str(db_path))  # creates the file
    assert db_path.exists()

    archive_path = reset_db(str(db_path))

    assert archive_path is not None
    assert archive_path.exists()
    assert not db_path.exists()
    assert "archive" in archive_path.name


def test_reset_db_noop_when_missing(tmp_path):
    db_path = tmp_path / "does_not_exist.db"
    result = reset_db(str(db_path))
    assert result is None
