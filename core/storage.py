"""SQLite-backed trade table using the locked Hebrew schema.

A trade goes through these states, inferable from which columns are filled:
  - pending:   row created on approval/`/add`, waiting for a matching Bitget
               position to appear (מחיר_כניסה IS NULL, בוטלה = 0)
  - cancelled: no position ever appeared before the confirmation timeout
               (בוטלה = 1). Kept rather than deleted so "approved but never
               executed" stays visible as discipline data, and so the symbol
               is freed for future signals.
  - open:      position confirmed on Bitget (מחיר_כניסה set, מחיר_יציאה NULL)
  - closed:    position went flat on Bitget (מחיר_יציאה set)

Stop/target are split into "מקורי" (what the bot originally proposed, or NULL
for a manually-added trade with no bot plan) and "בפועל" (whatever actually
protects the live position, which the tracker keeps in sync). סכום_סיכון,
רווח_הפסד, and מכפיל_R are always computed, never entered by hand.

Partial exits are aggregated onto the same row: גודל_שנסגר accumulates the
size closed so far, and on the final close מחיר_יציאה/רווח_הפסד are set from
Bitget's own position history, whose closeAvgPrice and netProfit already
aggregate every partial. R is always measured against the ORIGINAL risk, so a
trade that takes half off at +2R and stops the runner at break-even correctly
reads as +1R.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
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
    גודל_שנסגר        REAL DEFAULT 0,
    סטופ_לוס_מקורי     REAL,
    יעד_רווח_מקורי     REAL,
    סטופ_לוס_בפועל     REAL,
    יעד_רווח_בפועל     REAL,
    סכום_סיכון         REAL,
    רווח_הפסד          REAL,
    מכפיל_R            REAL,
    מינוף             REAL,
    בוטלה             INTEGER DEFAULT 0,
    תגית_אסטרטגיה      TEXT,
    הערות             TEXT
);
"""

# Every signal the scanner dispatches, independent of what happened to it -
# approved, rejected, or never touched. Nothing wrote this down before: the
# trades table only ever gained a row once a signal was approved AND a
# matching Bitget position appeared, so a rejected or ignored signal left no
# trace anywhere, and there was no way to score a strategy's own output
# separately from what got approved. decision is NULL until a button is
# pressed; a report treats a still-NULL row older than its own window as
# "ignored" rather than needing an explicit timeout mechanism. paper_r is
# filled in later by replaying the signal forward against real candles (see
# journal/paper_sim.py) once its stop or target has actually been hit.
SIGNALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dispatched_at   TEXT NOT NULL,
    symbol          TEXT NOT NULL,
    direction       TEXT NOT NULL,
    entry_price     REAL NOT NULL,
    stop_loss       REAL NOT NULL,
    take_profit     REAL NOT NULL,
    strategy_tag    TEXT NOT NULL,
    confluence      TEXT,
    decision        TEXT,
    trade_id        INTEGER,
    paper_r         REAL,
    paper_resolved_at TEXT
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
    גודל_שנסגר: float | None
    סטופ_לוס_מקורי: float | None
    יעד_רווח_מקורי: float | None
    סטופ_לוס_בפועל: float | None
    יעד_רווח_בפועל: float | None
    סכום_סיכון: float | None
    רווח_הפסד: float | None
    מכפיל_R: float | None
    מינוף: float | None
    בוטלה: int
    תגית_אסטרטגיה: str | None
    הערות: str | None

    @property
    def is_cancelled(self) -> bool:
        return bool(self.בוטלה)

    @property
    def is_pending(self) -> bool:
        return self.מחיר_כניסה is None and not self.is_cancelled

    @property
    def is_open(self) -> bool:
        return self.מחיר_כניסה is not None and self.מחיר_יציאה is None

    @property
    def is_closed(self) -> bool:
        return self.מחיר_יציאה is not None

    @property
    def changed_from_plan(self) -> bool:
        """True if what actually protected the position differed from the
        bot's original proposal. Always False for trades with no original
        plan (e.g. added manually via /add)."""
        if self.סטופ_לוס_מקורי is None and self.יעד_רווח_מקורי is None:
            return False
        return not (
            _approx_equal(self.סטופ_לוס_מקורי, self.סטופ_לוס_בפועל)
            and _approx_equal(self.יעד_רווח_מקורי, self.יעד_רווח_בפועל)
        )

    @property
    def had_partial_exit(self) -> bool:
        if not self.גודל_שנסגר or not self.גודל_פוזיציה:
            return False
        return self.גודל_שנסגר < self.גודל_פוזיציה - _PRICE_TOLERANCE


def _approx_equal(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) < _PRICE_TOLERANCE


def _risk_amount(entry_price: float, stop: float | None, size: float) -> float | None:
    if stop is None:
        return None
    return abs(entry_price - stop) * size


@dataclass
class SignalRecord:
    id: int
    dispatched_at: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy_tag: str
    confluence: str | None
    decision: str | None  # None (not yet acted on), "approved", or "rejected"
    trade_id: int | None
    paper_r: float | None
    paper_resolved_at: str | None


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.execute(SIGNALS_SCHEMA)

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

    def cancel_pending(self, trade_id: int, reason: str = "no position detected before timeout") -> None:
        """No position ever showed up: keep the row for the record, but free
        the symbol so future signals on it aren't blocked forever."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET בוטלה = 1, הערות = COALESCE(הערות || ' | ', '') || ? WHERE מספר_עסקה = ?",
                (reason, trade_id),
            )

    def confirm_entry(
        self,
        trade_id: int,
        entry_price: float,
        position_size: float,
        actual_stop: float | None,
        actual_target: float | None,
        leverage: float,
    ) -> None:
        """A matching Bitget position was found: the trade is now live.
        leverage is whatever Bitget reports — the source of truth for
        committed_margin(), not just what was originally planned."""
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET שעת_כניסה = ?, מחיר_כניסה = ?, גודל_פוזיציה = ?,
                    סטופ_לוס_בפועל = ?, יעד_רווח_בפועל = ?, סכום_סיכון = ?, מינוף = ?
                WHERE מספר_עסקה = ?
                """,
                (
                    datetime.now().time().isoformat(timespec="seconds"),
                    entry_price,
                    position_size,
                    actual_stop,
                    actual_target,
                    _risk_amount(entry_price, actual_stop, position_size),
                    leverage,
                    trade_id,
                ),
            )

    def resync_position(self, trade_id: int, entry_price: float, position_size: float) -> None:
        """Scale-ins change the average entry and total size; keep the row in
        step and recompute risk off the new numbers."""
        with self._connect() as conn:
            stop = conn.execute(
                "SELECT סטופ_לוס_בפועל FROM trades WHERE מספר_עסקה = ?", (trade_id,)
            ).fetchone()[0]
            conn.execute(
                "UPDATE trades SET מחיר_כניסה = ?, גודל_פוזיציה = ?, סכום_סיכון = ? WHERE מספר_עסקה = ?",
                (entry_price, position_size, _risk_amount(entry_price, stop, position_size), trade_id),
            )

    def update_actual_stop_target(self, trade_id: int, stop: float | None, target: float | None) -> None:
        """The live stop/target changed (moved manually, or set after entry).
        Recompute risk so a stop added later still yields a usable R."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT מחיר_כניסה, גודל_פוזיציה FROM trades WHERE מספר_עסקה = ?", (trade_id,)
            ).fetchone()
            entry_price, size = row if row else (None, None)
            risk = _risk_amount(entry_price, stop, size) if entry_price and size else None
            conn.execute(
                "UPDATE trades SET סטופ_לוס_בפועל = ?, יעד_רווח_בפועל = ?, סכום_סיכון = ? WHERE מספר_עסקה = ?",
                (stop, target, risk, trade_id),
            )

    def record_partial(self, trade_id: int, closed_size: float, realized_pnl: float | None) -> None:
        """A scale-out: part of the position closed, the rest keeps running."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET גודל_שנסגר = ?, רווח_הפסד = ? WHERE מספר_עסקה = ?",
                (closed_size, realized_pnl, trade_id),
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
                """
                UPDATE trades
                SET מחיר_יציאה = ?, רווח_הפסד = ?, מכפיל_R = ?, גודל_שנסגר = גודל_פוזיציה
                WHERE מספר_עסקה = ?
                """,
                (exit_price, realized_pnl, r_multiple, trade_id),
            )

    def pending_trades(self) -> list[Trade]:
        return self._select("WHERE מחיר_כניסה IS NULL AND בוטלה = 0")

    def open_trades(self) -> list[Trade]:
        return self._select("WHERE מחיר_כניסה IS NOT NULL AND מחיר_יציאה IS NULL")

    def has_open_or_pending(self, symbol: str) -> bool:
        rows = self._select("WHERE סימבול = ? AND מחיר_יציאה IS NULL AND בוטלה = 0", [symbol])
        return len(rows) > 0

    def committed_margin(self) -> float:
        """Margin currently tied up across open trades, from each trade's real
        entry/size/leverage — how much capital a new trade can't use."""
        total = 0.0
        for trade in self.open_trades():
            if trade.מינוף:
                total += (trade.מחיר_כניסה * trade.גודל_פוזיציה) / trade.מינוף
        return total

    def total_open_risk(self) -> float:
        """Sum of money at risk across open trades, for the aggregate risk cap."""
        return sum(t.סכום_סיכון for t in self.open_trades() if t.סכום_סיכון)

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

    # ---- signal log: every dispatched signal, independent of the trades table ----

    def log_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        strategy_tag: str,
        confluence: str | None = None,
    ) -> int:
        """Recorded the moment a signal is dispatched, before Approve/Reject is
        even seen - so a rejected or ignored signal is measurable too."""
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals
                    (dispatched_at, symbol, direction, entry_price, stop_loss, take_profit, strategy_tag, confluence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    symbol,
                    direction,
                    entry_price,
                    stop_loss,
                    take_profit,
                    strategy_tag,
                    confluence,
                ),
            )
            return cursor.lastrowid

    def mark_signal_decision(self, signal_id: int, decision: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE signals SET decision = ? WHERE id = ?", (decision, signal_id))

    def link_signal_trade(self, signal_id: int, trade_id: int) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE signals SET trade_id = ? WHERE id = ?", (trade_id, signal_id))

    def record_paper_outcome(self, signal_id: int, r_multiple: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE signals SET paper_r = ?, paper_resolved_at = ? WHERE id = ?",
                (r_multiple, datetime.now(timezone.utc).isoformat(), signal_id),
            )

    def unresolved_signals(self) -> list[SignalRecord]:
        """Signals whose stop/target hasn't been confirmed yet against real
        candles - the paper_sim replay's work queue."""
        return self._select_signals("WHERE paper_r IS NULL")

    def read_signals(self, start: date | None = None, end: date | None = None) -> list[SignalRecord]:
        clauses, params = [], []
        if start is not None:
            clauses.append("dispatched_at >= ?")
            params.append(start.isoformat())
        if end is not None:
            clauses.append("dispatched_at <= ?")
            params.append(end.isoformat())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._select_signals(where, params)

    def _select_signals(self, where: str = "", params: list | None = None) -> list[SignalRecord]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM signals {where} ORDER BY id", params or []).fetchall()
            return [SignalRecord(**dict(row)) for row in rows]
