"""SQLite-backed trade table using the locked Hebrew schema.

A trade goes through three states, inferable from which columns are filled:
  - pending:  row created on approval/`/add`, waiting for a matching Bitget
              position to actually appear (מחיר_כניסה IS NULL)
  - open:     position confirmed on Bitget (מחיר_כניסה set, מחיר_יציאה NULL)
  - closed:   position went flat on Bitget (מחיר_יציאה set)

Stop/target are split into "מקורי" (what the bot originally proposed, or
NULL for a manually-added trade with no bot plan) and "בפועל" (whatever is
actually set on the live Bitget position, which the tracker keeps in sync in
case the user adjusts it manually). סכום_סיכון, רווח_הפסד, and מכפיל_R are
always computed, never entered by hand.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    מספר_עסקה        INTEGER PRIMARY KEY AUTOINCREMENT,
    תאריך            TEXT NOT NULL,
    שעת_כניסה         TEXT,
    סימבול            TEXT NOT NULL,
    כיוון             TEXT NOT NULL,
    מחיר_כניסה         REAL,
    מחיר_יציאה         REAL,
    גודל_פוזיציה       REAL,
    סטופ_לוס_מקורי     REAL,
    יעד_רווח_מקורי     REAL,
    סטופ_לוס_בפועל     REAL,
    יעד_רווח_בפועל     REAL,
    סכום_סיכון         REAL,
    רווח_הפסד          REAL,
    מכפיל_R            REAL,
    תגית_אסטרטגיה      TEXT,
    הערות             TEXT
);
"""

_PRICE_TOLERANCE = 1e-9


@dataclass
class Trade:
    מספר_עסקה: int
    תאריך: str
    שעת_כניסה: str | None
    סימבול: str
    כיוון: str
    מחיר_כניסה: float | None
    מחיר_יציאה: float | None
    גודל_פוזיציה: float | None
    סטופ_לוס_מקורי: float | None
    יעד_רווח_מקורי: float | None
    סטופ_לוס_בפועל: float | None
    יעד_רווח_בפועל: float | None
    סכום_סיכון: float | None
    רווח_הפסד: float | None
    מכפיל_R: float | None
    תגית_אסטרטגיה: str | None
    הערות: str | None

    @property
    def is_pending(self) -> bool:
        return self.מחיר_כניסה is None

    @property
    def is_open(self) -> bool:
        return self.מחיר_כניסה is not None and self.מחיר_יציאה is None

    @property
    def is_closed(self) -> bool:
        return self.מחיר_יציאה is not None

    @property
    def changed_from_plan(self) -> bool:
        """True if the actual stop/target ever differed from the bot's
        original proposal. Always False for trades with no original plan
        (e.g. added manually via /add)."""
        if self.סטופ_לוס_מקורי is None and self.יעד_רווח_מקורי is None:
            return False
        return not (
            _approx_equal(self.סטופ_לוס_מקורי, self.סטופ_לוס_בפועל)
            and _approx_equal(self.יעד_רווח_מקורי, self.יעד_רווח_בפועל)
        )


def _approx_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < _PRICE_TOLERANCE


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def create_pending(
        self,
        symbol: str,
        direction: str,
        proposed_stop: float | None = None,
        proposed_target: float | None = None,
        strategy_tag: str | None = None,
        notes: str | None = None,
    ) -> int:
        """A signal was approved (or /add invoked): waiting for a matching
        Bitget position to confirm the trade actually happened."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (תאריך, סימבול, כיוון, סטופ_לוס_מקורי, יעד_רווח_מקורי, תגית_אסטרטגיה, הערות)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (date.today().isoformat(), symbol, direction, proposed_stop, proposed_target, strategy_tag, notes),
            )
            return cursor.lastrowid

    def confirm_entry(
        self,
        trade_id: int,
        entry_price: float,
        position_size: float,
        actual_stop: float | None,
        actual_target: float | None,
    ) -> None:
        """A matching Bitget position was found: the trade is now live."""
        risk_amount = abs(entry_price - actual_stop) * position_size if actual_stop is not None else None
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET שעת_כניסה = ?, מחיר_כניסה = ?, גודל_פוזיציה = ?,
                    סטופ_לוס_בפועל = ?, יעד_רווח_בפועל = ?, סכום_סיכון = ?
                WHERE מספר_עסקה = ?
                """,
                (
                    datetime.now().time().isoformat(timespec="seconds"),
                    entry_price,
                    position_size,
                    actual_stop,
                    actual_target,
                    risk_amount,
                    trade_id,
                ),
            )

    def update_actual_stop_target(self, trade_id: int, stop: float | None, target: float | None) -> None:
        """The tracker detected the live position's stop/target changed
        (e.g. the user adjusted it manually on Bitget)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET סטופ_לוס_בפועל = ?, יעד_רווח_בפועל = ? WHERE מספר_עסקה = ?",
                (stop, target, trade_id),
            )

    def set_strategy_tag(self, trade_id: int, tag: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE trades SET תגית_אסטרטגיה = ? WHERE מספר_עסקה = ?", (tag, trade_id))

    def close_trade(self, trade_id: int, exit_price: float, realized_pnl: float | None = None) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT כיוון, מחיר_כניסה, גודל_פוזיציה, סכום_סיכון FROM trades WHERE מספר_עסקה = ?",
                (trade_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No trade with id {trade_id}")
            direction, entry_price, position_size, risk_amount = row

            if realized_pnl is None:
                sign = 1 if direction == "long" else -1
                realized_pnl = sign * (exit_price - entry_price) * position_size

            r_multiple = realized_pnl / risk_amount if risk_amount else None

            conn.execute(
                "UPDATE trades SET מחיר_יציאה = ?, רווח_הפסד = ?, מכפיל_R = ? WHERE מספר_עסקה = ?",
                (exit_price, realized_pnl, r_multiple, trade_id),
            )

    def pending_trades(self) -> list[Trade]:
        return self._select("WHERE מחיר_כניסה IS NULL")

    def open_trades(self) -> list[Trade]:
        return self._select("WHERE מחיר_כניסה IS NOT NULL AND מחיר_יציאה IS NULL")

    def has_open_or_pending(self, symbol: str) -> bool:
        rows = self._select("WHERE סימבול = ? AND מחיר_יציאה IS NULL", [symbol])
        return len(rows) > 0

    def get_trade(self, trade_id: int) -> Trade:
        rows = self._select("WHERE מספר_עסקה = ?", [trade_id])
        if not rows:
            raise ValueError(f"No trade with id {trade_id}")
        return rows[0]

    def read_all(self, start: date | None = None, end: date | None = None) -> list[Trade]:
        clauses, params = [], []
        if start is not None:
            clauses.append("תאריך >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("תאריך <= ?")
            params.append(end.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._select(where, params)

    def _select(self, where: str = "", params: list | None = None) -> list[Trade]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM trades {where} ORDER BY מספר_עסקה", params or []).fetchall()
            return [Trade(**dict(row)) for row in rows]
