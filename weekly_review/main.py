"""Entrypoint: python -m weekly_review.main
Runs once a week (cron on the VM, Sunday 20:00 Asia/Jerusalem) to report
real trade activity plus paper-simulated signal outcomes for the week,
against the all-time baseline, and send the result via Telegram.

Resolving as many pending paper signals as possible happens here, right
before the report is built, rather than as a separate always-on job -
weekly is the only cadence anything currently reads this data on, so
there's no reason to resolve it more often than that.

IT MUST SAY SO WHEN IT BREAKS. This job ran on schedule every Sunday from
2026-08-02 and died before sending, every time, on a Bitget 400 - and the only
trace was a traceback appended to a log file nobody reads. Two weeks of
reports were simply absent, which is indistinguishable from a quiet week. The
failure is now reported to Telegram on the way out, and a successful run
leaves a marker the scanner watches (see heartbeat.py) so a job that stops
running at all is noticed too.
"""

import traceback

from config import settings
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import send_message
from journal.paper_sim import resolve_pending
from weekly_review.analyze import analyze, render
from weekly_review.heartbeat import record_success


def _alert(text: str) -> None:
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, text)


def main() -> None:
    try:
        storage = Storage(settings.trades_db_path)
        bitget = client_from_settings(settings)

        resolved = resolve_pending(storage, bitget)
        print(f"Resolved {resolved} paper signal(s) this run.")

        report = render(analyze(storage))
        print(report)
        _alert(report)
    except Exception as exc:
        # Reported before re-raising, so the traceback still reaches the log
        # and the exit code is still non-zero. Only the last frames are sent:
        # the point is to know it broke and roughly where, not to read a full
        # traceback on a phone.
        tail = "".join(traceback.format_exc().strip().splitlines(keepends=True)[-6:])
        try:
            _alert(f"WEEKLY REPORT FAILED: {type(exc).__name__}: {exc}\n\n{tail}")
        except Exception:
            # A broken alert path must not replace the original error with a
            # different one - the log is the last resort and it keeps both.
            traceback.print_exc()
        raise

    # Only after the report has actually been sent. Recording it earlier would
    # make the heartbeat certify runs that produced nothing.
    record_success(settings.trades_db_path)


if __name__ == "__main__":
    main()
