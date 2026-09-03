"""Says so when the account and the journal have diverged, or when the
weekly health report has gone quiet - the two ways a real problem hides
behind silence rather than an error.

Extracted from Scanner, which used to own this alongside a dozen unrelated
responsibilities.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from core.storage import Storage
from weekly_review import heartbeat as weekly_heartbeat

logger = logging.getLogger(__name__)

# The weekly report is due every 7 days; this is the age past which its
# absence is a fault rather than a late run. Wide enough that a delayed cron
# or a clock drift is not an alert, tight enough that one missed Sunday is.
WEEKLY_REPORT_MAX_AGE_DAYS = 8.0


def _reported_path(db_path: str) -> Path:
    return Path(db_path).parent / "reported_untracked"


def _load_reported(db_path: str) -> set[tuple]:
    """Which untracked positions have already been announced.

    A plain file beside the trades DB rather than a table: it must survive a
    restart, and it must not be able to take the scanner down if it is
    unreadable - a corrupt file simply means the next alert repeats once.
    """
    try:
        rows = _reported_path(db_path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return set()
    return {tuple(r.split("\t")) for r in rows if r.strip()}


def _save_reported(db_path: str, keys: set[tuple]) -> None:
    try:
        path = _reported_path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joined = "\n".join("\t".join(str(part) for part in k) for k in keys)
        path.write_text(joined, encoding="utf-8")
    except OSError:
        logger.exception("Could not persist the reported-untracked set; it will repeat after a restart")


class PositionHealthMonitor:
    def __init__(self, bitget, storage: Storage, bot):
        self.bitget = bitget
        self.storage = storage
        self.bot = bot
        # Positions already reported as untracked, keyed (symbol, direction,
        # open time). In memory on purpose: a restart re-reports, which is
        # the right behaviour for something that needs a decision from Dror.
        # Loaded from disk, not started empty - held only in memory it forgets
        # on every restart, so Dror got a fresh alert every deploy for a
        # position deliberately left alone.
        self._reported_untracked: set[tuple] = _load_reported(storage.db_path) if storage is not None else set()
        self._reported_weekly_overdue = None

    def already_exposed(self, symbol: str) -> bool:
        """Whether this symbol already has a trade on it - per the ACCOUNT as
        well as our own records.

        The database alone was the whole check, and it is only as good as
        its own bookkeeping. Live on 2026-08-08 a real APTUSDT short of
        9.035 @ 0.592, open since the 5th, was recorded here as closed;
        nothing suppressed a fresh Strategy 1 LONG on the same symbol. On a
        hedge-mode account that does not add to the position, it opens an
        opposing one - so the failure mode of trusting our own records is
        not a duplicate trade but an accidental hedge nobody chose.

        The exchange is the only thing that actually knows what is open, so
        it is asked too. Resting ENTRY orders count as well: an unfilled
        limit is a trade in flight, which is exactly the state the database
        calls "pending" and can lose the same way.

        A failed read falls back to the database answer rather than muting
        the watchlist - same call as session-gating makes about session
        data. It is no worse than the behaviour this replaces, and an
        outage should not silence every symbol.
        """
        if self.storage.has_open_or_pending(symbol):
            return True
        try:
            if self.bitget.get_positions(symbol):
                logger.warning(
                    "%s has a live position the trades DB does not know about - suppressing the signal. "
                    "The records and the account have diverged; reconcile them.",
                    symbol,
                )
                return True
            resting_entries = [
                o for o in self.bitget.get_open_orders(symbol)
                if (o.get("tradeSide") or "").lower() == "open"
            ]
            if resting_entries:
                logger.warning("%s has a resting entry order the trades DB does not know about", symbol)
                return True
        except Exception:
            logger.exception("Could not check %s against the account; falling back to the DB alone", symbol)
        return False

    async def poll_untracked_positions(self) -> None:
        """Say so when the account holds something the bot is not tracking.

        The APTUSDT short of 9.035 @ 0.592 was opened by hand on 2026-08-05
        at 15:05:54 UTC - 57 seconds after trade #9 closed - and never
        registered. Nothing was wrong with the records; the bot simply was
        never told. It surfaced three days later only because a Strategy 1
        long fired on the same symbol, and by then it had sat without a
        stop or a target the whole time.

        Silence is the actual failure here. already_exposed() now
        suppresses signals for such a symbol, which is correct but
        invisible - without this, a forgotten position quietly mutes its
        own symbol forever.

        Reported once per position, keyed on when it was opened, so a
        position that is deliberately left alone does not nag every hour. A
        new position on the same symbol and side gets its own alert
        because its open time differs.
        """
        try:
            positions = self.bitget.get_all_positions()
        except Exception:
            logger.exception("Could not read open positions to check for untracked ones")
            return

        tracked = {
            (t.סימבול, t.כיוון)
            for t in (*self.storage.open_trades(), *self.storage.pending_trades())
        }
        for position in positions:
            symbol, direction = position["symbol"], position["direction"]
            if (symbol, direction) in tracked:
                continue
            key = (symbol, direction, str(position["raw"].get("cTime")))
            if key in self._reported_untracked:
                continue
            self._reported_untracked.add(key)
            _save_reported(self.storage.db_path, self._reported_untracked)

            stop, target = None, None
            try:
                stop, target = self.bitget.get_stop_target(symbol, direction)
            except Exception:
                logger.exception("Could not read stop/target for the untracked %s position", symbol)
            missing = [name for name, value in (("stop", stop), ("target", target)) if value is None]
            warning = f"\nIt has no {' and no '.join(missing)} on the exchange." if missing else ""
            await self.bot.send_message(
                f"UNTRACKED position: {symbol} {direction} {position['size']:g} @ {position['entry_price']:g}.\n"
                f"The bot is not managing it, and it is blocking new {symbol} signals."
                f"{warning}\nUse /add to track it, or close it."
            )

    async def poll_weekly_report_overdue(self) -> None:
        """Say so when the weekly report has stopped arriving.

        The report itself now alerts when it crashes, but that cannot cover
        the case where it never runs - a removed crontab, a VM that was
        down on a Sunday, a venv broken by a bad deploy. In all of those the
        job produces no output at all, and an absent report is
        indistinguishable from a quiet week. It was absent for two weeks
        straight and only surfaced because Dror asked where it had gone.

        Reported once a day rather than hourly: it is a "look at this when
        you can" fact, not something that gets more true by repeating.
        """
        overdue = weekly_heartbeat.overdue_by(self.storage.db_path, WEEKLY_REPORT_MAX_AGE_DAYS)
        if overdue is None:
            return
        today = datetime.now(timezone.utc).date()
        if self._reported_weekly_overdue == today:
            return
        self._reported_weekly_overdue = today
        await self.bot.send_message(
            f"WEEKLY REPORT OVERDUE by {overdue:.1f} days - the last one that reached you was "
            f"{weekly_heartbeat.last_success(self.storage.db_path):%Y-%m-%d %H:%M} UTC.\n"
            f"The Sunday cron is not producing a report. Check ~/weekly_review.log on the VM."
        )
