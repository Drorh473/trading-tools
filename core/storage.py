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

The last three columns are deliberately NOT part of the Hebrew journal: they
are the bot's own exit plan (where the stop goes once the partial fills, and
where the runner's target goes), not something Dror records or reads. They
exist because that plan used to live only in a closure inside the running
asyncio task, so restarting the service silently dropped it - the APTUSDT
short of 2026-08-13 took its partial and kept its original stop, while the
notification said the stop "should already be at entry". Written to the row,
the plan survives a restart and both the live and re-attached trackers read
the same one.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from core import clock

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
    הערות             TEXT,
    breakeven_stop    REAL,
    runner_target     REAL,
    partial_fraction  REAL,
    exit_managed      INTEGER DEFAULT 0,
    initial_risk      REAL
);
"""

# Columns added after the first databases were created. SCHEMA only runs as
# CREATE TABLE IF NOT EXISTS, so it does nothing to the live journal - and
# _select() builds Trade(**row), which raises the moment the dataclass and the
# real table disagree. Applied on every open, cheaply and idempotently.
_ADDED_COLUMNS = {
    "trades": {
        "breakeven_stop": "REAL",
        "runner_target": "REAL",
        # "no runner target" as a DECISION rather than an absence. See
        # Scanner._exit_plan_signal: a rebuilt signal with runner_target NULL
        # and nothing to say it was deliberate falls through to the daily
        # level and invents one, which turns the trail off for good.
        "runner_target_is_final": "INTEGER DEFAULT 0",
        "partial_fraction": "REAL",
        "exit_managed": "INTEGER DEFAULT 0",
        "initial_risk": "REAL",
    },
    # This map used to cover `trades` alone, and adding signal_json to the
    # SIGNALS schema would then have done nothing at all to the live journal:
    # CREATE TABLE IF NOT EXISTS leaves an existing table untouched, so every
    # log_signal naming the new column would have failed with "no such column"
    # - on the one write that happens for every alert the bot sends.
    "signals": {
        "signal_json": "TEXT",
    },
}

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
    paper_resolved_at TEXT,
    signal_json     TEXT
);
"""

# WHEN EACH SYMBOL/INSTANCE PAIR LAST PROMPTED, so the once-a-day throttle
# survives a restart.
#
# It did not. `Scanner._alerted` was a plain dict on the object, so every
# restart - a deploy, a crash, a systemd bounce - reset the whole thing to
# empty and the next scan re-prompted everything it had already asked about.
# MMTUSDT went out twice on 2026-08-18 inside ONE 4H candle: identical stop,
# identical target, identical breakeven, differing only in the market price
# quoted, because both the dedupe set and the throttle had been cleared
# between the two scans.
#
# The scarce resource this protects is Dror's attention - nothing executes
# without him pressing Approve - so state that silently empties itself is the
# same failure as no throttle at all.
#
# Keyed by (symbol, strategy_tag) rather than by symbol: the same pullback on
# 15m and on 4H are different trades with different stops, and collapsing them
# would hide which timeframe found it.
ALERT_THROTTLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS alert_throttle (
    symbol       TEXT NOT NULL,
    strategy_tag TEXT NOT NULL,
    alerted_at   REAL NOT NULL,
    PRIMARY KEY (symbol, strategy_tag)
);
"""

# WHEN EACH SCAN CYCLE BEGAN, and when the next one was due.
#
# The capability ledger answers "has this worked LATELY" by storing each
# capability's most recent success, which cannot answer "was the bot ever
# down" - a gap that has since recovered leaves no trace in a last-seen
# timestamp. Dror asked for the other question: report it in the weekly review
# "if the bot was down for even a small time".
#
# `due_at` is what makes a gap provable rather than guessed. The scan cadence
# is not fixed - it is min(timeframes, seconds_until_next_close), so it varies
# with which candle closes next - and comparing against an assumed 15 minutes
# would call an ordinary 4H-driven wait an outage. Recording when the bot
# itself expected to be back means a gap is measured against its own intent.
#
# Rows are ~96/day at the 15m cadence, so a year is ~35k rows of two floats.
# Pruned past PRUNE_DAYS anyway, since nothing reads further back than the
# weekly window.
SCAN_HEARTBEAT_SCHEMA = """
CREATE TABLE IF NOT EXISTS scan_heartbeat (
    ts      REAL PRIMARY KEY,
    due_at  REAL NOT NULL
);
"""

# How late the next heartbeat may be before it counts as downtime. A scan takes
# time to run - the 100-symbol sweep is 40-60s - and the loop only sleeps until
# the NEXT candle close after that, so a small overshoot is normal operation.
HEARTBEAT_GRACE_SECONDS = 180.0
HEARTBEAT_PRUNE_DAYS = 90

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
    # The bot's exit plan, set once at entry confirmation and read back when
    # the partial fills - including by a tracker re-attached after a restart,
    # which has no other way to learn it. breakeven_stop is None when the bot
    # is not managing this trade's exits, and that is what the partial-exit
    # notification keys on to say "move it by hand" rather than claiming a
    # move that nothing was ever going to make.
    breakeven_stop: float | None = None
    runner_target: float | None = None
    runner_target_is_final: bool = False
    partial_fraction: float | None = None
    # Set by /manage: this specific trade's exits may be managed even though
    # its strategy tag is not one the router knows. A hand-added trade's tag
    # is free text typed at the /add prompt ("strategy 1"), so it will never
    # match an instance tag like "Strategy 1 1H" - and rewriting the tag to
    # force a match would corrupt what the weekly review groups by, so the
    # permission is carried here instead of being inferred from the tag.
    exit_managed: int = 0
    # The 1R this trade was sized against, frozen against stop MOVES.
    # סכום_סיכון tracks risk as it stands right now, which is what the
    # aggregate cap wants - once a stop reaches breakeven the money at risk
    # really is ~0 and that headroom should be freed. R wants the opposite,
    # and sharing one column meant a trade that did exactly what it was
    # supposed to divided by ~0: APTUSDT #11 took its partial, went to
    # breakeven, closed +4.18 and reported 4653.25R.
    initial_risk: float | None = None

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
    signal_json: str | None = None  # the whole Signal, for /add <n>


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        parent = Path(db_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)
            conn.execute(SIGNALS_SCHEMA)
            conn.execute(ALERT_THROTTLE_SCHEMA)
            conn.execute(SCAN_HEARTBEAT_SCHEMA)
            for table, columns in _ADDED_COLUMNS.items():
                existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
                for column, column_type in columns.items():
                    if column not in existing:
                        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

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
                (clock.today().isoformat(), symbol, direction, proposed_stop, proposed_target, strategy_tag, notes),
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
                    סטופ_לוס_בפועל = ?, יעד_רווח_בפועל = ?, סכום_סיכון = ?, מינוף = ?,
                    initial_risk = ?
                WHERE מספר_עסקה = ?
                """,
                (
                    clock.now().time().isoformat(timespec="seconds"),
                    entry_price,
                    position_size,
                    actual_stop,
                    actual_target,
                    _risk_amount(entry_price, actual_stop, position_size),
                    leverage,
                    _risk_amount(entry_price, actual_stop, position_size),
                    trade_id,
                ),
            )

    def resync_position(self, trade_id: int, entry_price: float, position_size: float,
                        stop: float | None = None, leverage: float | None = None) -> None:
        """Scale-ins change the average entry and total size; keep the row in
        step and recompute risk off the new numbers.

        initial_risk moves too, because the POSITION changed: a split entry
        confirms on its market leg alone, so 1R computed there would be a
        fifth of the trade's real risk and every R off it five times too big.
        Only a stop MOVE leaves it alone.

        `stop` and `leverage` are for the staged confluence add-on, which is
        the other way a live position grows. It differs from a limit leg
        filling in two ways: it moves the stop on the WHOLE position to the
        broken pattern's own level, and it can open at a different leverage.
        Omit them and the existing stop is kept, which is the scale-in case.

        Until this was called from `_offer_add_on`, that path wrote nothing
        back at all - it read committed_margin() and total_open_risk() and then
        doubled a position without touching a column. total_open_risk() is what
        enforces the aggregate risk cap, so the cap undercounted exactly the
        trades carrying the most risk.
        """
        with self._connect() as conn:
            if stop is None:
                stop = conn.execute(
                    "SELECT סטופ_לוס_בפועל FROM trades WHERE מספר_עסקה = ?", (trade_id,)
                ).fetchone()[0]
            risk = _risk_amount(entry_price, stop, position_size)
            sets = ["מחיר_כניסה = ?", "גודל_פוזיציה = ?", "סכום_סיכון = ?",
                    "initial_risk = ?", "סטופ_לוס_בפועל = ?"]
            params: list = [entry_price, position_size, risk, risk, stop]
            if leverage:
                sets.append("מינוף = ?")
                params.append(leverage)
            params.append(trade_id)
            conn.execute(f"UPDATE trades SET {', '.join(sets)} WHERE מספר_עסקה = ?", params)

    def update_actual_stop_target(self, trade_id: int, stop: float | None, target: float | None) -> None:
        """The live stop/target changed (moved manually, or set after entry).

        סכום_סיכון follows the new stop, because that IS the money at risk
        now and total_open_risk() sizes the aggregate cap off it - a stop at
        breakeven genuinely risks nothing and should free that headroom.

        initial_risk deliberately does not follow: it is the 1R the trade was
        sized against, and a trade that reaches breakeven has not stopped
        having had a 1R. Sharing the one column is what made APTUSDT #11
        report 4653.25R - +4.18 divided by the 0.0009 left between its entry
        and a stop sitting on top of it. It is still WRITTEN here when unset,
        for the trade whose stop only appears after entry; that is the case
        this method's risk recomputation originally existed for.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT מחיר_כניסה, גודל_פוזיציה, initial_risk FROM trades WHERE מספר_עסקה = ?", (trade_id,)
            ).fetchone()
            entry_price, size, initial_risk = row if row else (None, None, None)
            risk = _risk_amount(entry_price, stop, size) if entry_price and size else None
            conn.execute(
                """
                UPDATE trades
                SET סטופ_לוס_בפועל = ?, יעד_רווח_בפועל = ?, סכום_סיכון = ?,
                    initial_risk = COALESCE(initial_risk, ?)
                WHERE מספר_עסקה = ?
                """,
                (stop, target, risk, risk, trade_id),
            )

    def set_exit_plan(
        self,
        trade_id: int,
        breakeven_stop: float | None,
        runner_target: float | None,
        partial_fraction: float | None,
        runner_target_is_final: bool = False,
    ) -> None:
        """What the bot commits to doing when the partial fills.

        Written only for a trade whose exits the bot actually manages, so
        breakeven_stop being set means "the stop WILL be moved here", not
        "here is where breakeven happens to be". Everything downstream - the
        partial-fill handler and the notification's wording - reads it that
        way, and a trade the bot only watches keeps it NULL.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE trades
                SET breakeven_stop = ?, runner_target = ?, partial_fraction = ?,
                    runner_target_is_final = ?
                WHERE מספר_עסקה = ?
                """,
                (breakeven_stop, runner_target, partial_fraction,
                 1 if runner_target_is_final else 0, trade_id),
            )

    def set_exit_managed(self, trade_id: int, managed: bool = True) -> None:
        """Grant (or revoke) exit management for one specific trade.

        Written AFTER set_exit_plan by the /manage flow, deliberately: if the
        plan lands and this does not, the trade is simply unmanaged, which is
        the safe direction to fail in. The reverse order would leave a trade
        marked managed with no plan to act on.
        """
        with self._connect() as conn:
            conn.execute(
                "UPDATE trades SET exit_managed = ? WHERE מספר_עסקה = ?",
                (1 if managed else 0, trade_id),
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
                """
                SELECT כיוון, מחיר_כניסה, גודל_פוזיציה, סכום_סיכון, initial_risk
                FROM trades WHERE מספר_עסקה = ?
                """,
                (trade_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f"No trade with id {trade_id}")
            direction, entry_price, position_size, risk_amount, initial_risk = row

            if realized_pnl is None:
                sign = 1 if direction == "long" else -1
                realized_pnl = sign * (exit_price - entry_price) * position_size

            # Against the 1R the trade was SIZED to, not whatever distance was
            # left to the stop at the end - see update_actual_stop_target.
            # Falls back for rows written before initial_risk existed.
            risk = initial_risk if initial_risk else risk_amount
            r_multiple = realized_pnl / risk if risk else None

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
        signal_json: str | None = None,
    ) -> int:
        """Recorded the moment a signal is dispatched, before Approve/Reject is
        even seen - so a rejected or ignored signal is measurable too.

        `signal_json` is the whole Signal, kept so an expired one can be
        re-offered later by its number. The seven columns beside it describe a
        signal well enough to SCORE it and not well enough to REBUILD it: they
        carry no partial_fraction, no remainder_target, no limit_entry and no
        fill guard, so a trade reconstructed from them alone would silently
        take the scanner's defaults instead of the strategy's own exit.
        """
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals
                    (dispatched_at, symbol, direction, entry_price, stop_loss, take_profit,
                     strategy_tag, confluence, signal_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    signal_json,
                ),
            )
            return cursor.lastrowid

    def signal_payload(self, signal_id: int) -> str | None:
        """The stored Signal for one dispatched alert, or None.

        None means the row predates signal_json, or the signal was logged
        without one - a too_small refusal, say. The caller must say so rather
        than rebuild a Signal from the columns and pretend it is the same
        trade: the columns carry no exit shape at all.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT signal_json FROM signals WHERE id = ?", (signal_id,)
            ).fetchone()
        return row[0] if row else None

    # ---- alert throttle: one prompt per symbol per instance per rolling day ----

    def last_alerted(self, symbol: str, strategy_tag: str) -> float | None:
        """When this symbol/instance pair last produced a prompt, or None.

        A unix timestamp rather than an ISO string because the caller compares
        it against a rolling window, not against a calendar day - a boundary
        would let a 23:50 signal silence the whole of the next day.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT alerted_at FROM alert_throttle WHERE symbol = ? AND strategy_tag = ?",
                (symbol, strategy_tag),
            ).fetchone()
            return float(row[0]) if row else None

    def record_alerted(self, symbol: str, strategy_tag: str, at: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alert_throttle (symbol, strategy_tag, alerted_at) VALUES (?, ?, ?)
                ON CONFLICT(symbol, strategy_tag) DO UPDATE SET alerted_at = excluded.alerted_at
                """,
                (symbol, strategy_tag, float(at)),
            )

    def clear_alert_throttle(self, symbol: str) -> None:
        """Release every instance's throttle on one symbol, because it has gone
        flat and is tradeable again."""
        with self._connect() as conn:
            conn.execute("DELETE FROM alert_throttle WHERE symbol = ?", (symbol,))

    # ---- availability: was the bot ever down, even briefly? -----------------

    def record_heartbeat(self, ts: float, due_at: float) -> None:
        """One scan cycle began at `ts`, and expects the next at `due_at`.

        REPLACE rather than INSERT: ts is the primary key, and a restart inside
        the same second must not raise into the scan loop. Availability
        bookkeeping is never worth ending a scan over.
        """
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO scan_heartbeat (ts, due_at) VALUES (?, ?)",
                (float(ts), float(due_at)),
            )

    def downtime_gaps(self, since: float | None = None,
                      grace: float = HEARTBEAT_GRACE_SECONDS) -> list[tuple[float, float, float]]:
        """(last_seen, back_at, seconds_missing) for every gap worth reporting.

        A gap counts when the next heartbeat arrives later than the previous
        cycle SAID it would, plus a grace. Measuring against due_at rather than
        a fixed interval is the point: the cadence is
        min(timeframes, seconds_until_next_close), so an ordinary wait for a 4H
        close is hours long and is not an outage.

        `seconds_missing` is measured from when the bot was DUE back, not from
        when it was last seen - the wait before that was expected and is not
        downtime.
        """
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, due_at FROM scan_heartbeat WHERE ts >= ? ORDER BY ts",
                (float(since) if since is not None else 0.0,),
            ).fetchall()
        gaps = []
        for (ts, due_at), (next_ts, _next_due) in zip(rows, rows[1:]):
            if next_ts > due_at + grace:
                gaps.append((ts, next_ts, next_ts - due_at))
        return gaps

    def heartbeat_span(self, since: float | None = None) -> tuple[float, float] | None:
        """(first, last) heartbeat in the window, or None if there are none.

        Needed to state downtime as a share of a period that was actually being
        watched. A table that only started recording on Tuesday cannot claim
        100% uptime for the week.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(ts), MAX(ts) FROM scan_heartbeat WHERE ts >= ?",
                (float(since) if since is not None else 0.0,),
            ).fetchone()
        return (row[0], row[1]) if row and row[0] is not None else None

    def prune_heartbeats(self, before: float) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM scan_heartbeat WHERE ts < ?", (float(before),))

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
