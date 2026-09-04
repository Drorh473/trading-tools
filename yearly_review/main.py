"""Entrypoint: python -m yearly_review.main

Runs once a year (cron on the VM, Jan 1st at 09:00 Asia/Jerusalem) and
reports on the year that just ENDED — see yearly_review.window.

IT MUST SAY SO WHEN IT BREAKS, for the same reason weekly_review.main and
monthly_review.main must - only sharper. A monthly report that stops arriving
is noticed within two months; a YEARLY one that stops arriving looks exactly
like an ordinary gap between years, and nothing about its absence is
remarkable until a second one fails to show - two years later.

THE EQUITY SNAPSHOT IS WRITTEN EVEN WHEN THE REPORT FAILS, for the same
reason as monthly_review.main: Bitget cannot be asked what equity was on
Jan 1 after Jan 1 has passed, so a crash between reading equity and sending
the report would cost the next report its balance line for a full year.
"""

import traceback
from datetime import datetime

from config import settings
from core import clock, ledger
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import send_message, send_photo
from yearly_review import chart, snapshot
from yearly_review.analyze import analyze
from yearly_review.render import render


def _alert(text: str) -> None:
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, text)


def _alert_photo(photo: bytes, caption: str) -> None:
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_photo(settings.telegram_bot_token, settings.telegram_chat_id, photo, caption)


def main() -> None:
    try:
        storage = Storage(settings.trades_db_path)
        bitget = client_from_settings(settings)

        report = analyze(storage, bitget=bitget)

        # Recorded before the send, deliberately - see the module docstring.
        if report.reconciliation.equity_end is not None:
            snapshot.record(
                settings.trades_db_path,
                report.reconciliation.equity_end,
                now=datetime.now(clock.LOCAL_TZ),
            )

        _alert(render(report))

        # The charts are sent as follow-up messages, after the text report,
        # and never block it: either one drawing nothing (too few equity
        # points, or no strategy with a closed trade) is a normal state for
        # the first year or two of this account's life, not a failure worth
        # losing the report over.
        equity_image = chart.equity_curve(report.monthly_equity, report.label)
        if equity_image is not None:
            _alert_photo(equity_image, f"Monthly equity — {report.label}")

        pnl_image = chart.strategy_pnl(report.stats.by_strategy, report.label)
        if pnl_image is not None:
            _alert_photo(pnl_image, f"P&L by strategy — {report.label}")

        # Recorded only after the report has actually been SENT - see
        # monthly_review.main for why that ordering matters.
        ledger.try_record(settings.trades_db_path, ledger.YEARLY_REPORT)
    except Exception:
        _alert(f"Yearly review FAILED:\n```\n{traceback.format_exc()[-1200:]}\n```")
        raise


if __name__ == "__main__":
    main()
