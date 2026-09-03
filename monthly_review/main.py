"""Entrypoint: python -m monthly_review.main

Runs once a month (cron on the VM, 1st at 09:00 Asia/Jerusalem) and reports on
the month that just ENDED — see monthly_review.window.

IT MUST SAY SO WHEN IT BREAKS, for the same reason weekly_review.main must:
that job ran on schedule every Sunday for two weeks, died before sending every
time, and left nothing but a traceback in a log file nobody reads. A report
that never arrives is indistinguishable from a quiet month, and a monthly
cadence means the silence would last four times as long before anyone noticed.

THE EQUITY SNAPSHOT IS WRITTEN EVEN WHEN THE REPORT FAILS. Bitget cannot be
asked what equity was on the 1st after the 1st has passed, so a crash between
reading equity and sending the report would cost the next report its balance
line permanently. The snapshot is therefore recorded as soon as it is read,
before anything that can fail.
"""

import traceback
from datetime import datetime, timedelta

from config import settings
from core import clock, ledger
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import send_message
from monthly_review import snapshot
from monthly_review.analyze import analyze
from monthly_review.render import render
from monthly_review.window import last_full_month
from notifier.main import LIVE_TAGS, signal_silence_days

# The modelled fee bill comes from the SAME estimator the weekly report uses -
# each strategy's own declared entry mix - rather than a second copy that
# could drift from it. The whole point of the check is to catch that estimator
# disagreeing with the exchange, so re-deriving it here would test nothing.
from weekly_review.analyze import _fee_by_strategy


def _alert(text: str) -> None:
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, text)


def main() -> None:
    try:
        storage = Storage(settings.trades_db_path)
        bitget = client_from_settings(settings)

        start, end = last_full_month()
        closed = [t for t in storage.read_all(start=start, end=end - timedelta(days=1)) if t.is_closed]
        modelled_fees = sum(fee for _count, fee in _fee_by_strategy(closed).values())

        report = analyze(
            storage,
            live_tags=set(LIVE_TAGS),
            silence_days=signal_silence_days,
            bitget=bitget,
            modelled_fees=modelled_fees,
        )

        # Recorded before the send, deliberately - see the module docstring.
        if report.reconciliation.equity_end is not None:
            snapshot.record(
                settings.trades_db_path,
                report.reconciliation.equity_end,
                now=datetime.now(clock.LOCAL_TZ),
            )

        _alert(render(report))

        # Recorded only after the report has actually been SENT. A run that
        # built a report and failed to deliver it has not done its job, and
        # marking it successful here would tell the silence watcher everything
        # is fine while nothing reaches Dror - the precise failure this whole
        # ledger exists to catch.
        ledger.try_record(settings.trades_db_path, ledger.MONTHLY_REPORT)
    except Exception:
        _alert(f"Monthly review FAILED:\n```\n{traceback.format_exc()[-1200:]}\n```")
        raise


if __name__ == "__main__":
    main()
