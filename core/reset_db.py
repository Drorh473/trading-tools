"""CLI command for the demo -> real-money transition: archives the current
trade database (renamed with a timestamp, never deleted) and leaves a fresh
one to be created the next time the app runs. Run manually via SSH:

    python -m core.reset_db
"""

import shutil
from datetime import datetime
from pathlib import Path

from config import settings


def reset_db(db_path: str | None = None) -> Path | None:
    path = Path(db_path or settings.trades_db_path)

    if not path.exists():
        print(f"No database found at {path} — nothing to archive.")
        return None

    archive_path = path.with_name(f"{path.stem}_archive_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}{path.suffix}")
    shutil.move(str(path), str(archive_path))
    print(f"Archived {path} -> {archive_path}")
    print("A fresh database will be created automatically the next time the app runs.")
    return archive_path


def main() -> None:
    path = Path(settings.trades_db_path)
    answer = input(
        f"Type YES to archive the current database ({path}) and start fresh: "
    )
    if answer.strip() != "YES":
        print("Aborted — no changes made.")
        return
    reset_db(str(path))


if __name__ == "__main__":
    main()
