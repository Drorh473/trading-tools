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
from datetime import date, datetime, timedelta

from config import settings
from core import clock, ledger
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import send_message, send_photo
from journal.paper_sim import resolve_pending
from notifier.main import LEDGER_EXPECTATIONS
from weekly_review.analyze import WeeklyReport, analyze, prune_stale_heartbeats, render
from weekly_review.charts import daily_pnl_chart, strategy_breakdown_chart
from weekly_review.heartbeat import record_success


def _alert(text: str) -> None:
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, text)


def _send_charts(report: WeeklyReport) -> None:
    """Best-effort: a chart failing to build or send must not fail the run -
    the substantive numbers already went out in the text report above it.
    Same reasoning as prune_stale_heartbeats' own "never let tidying up fail
    a run whose report already went out", just for a picture instead of
    housekeeping.
    """
    if not (settings.telegram_bot_token and settings.telegram_chat_id):
        return
    charts = (
        (lambda: strategy_breakdown_chart(report.real_this_week.by_strategy), "Trades and avg R by strategy this week"),
        (lambda: daily_pnl_chart(report.daily_pnl), "Daily PnL this week"),
    )
    for build, caption in charts:
        try:
            png = build()
            if png is not None:
                send_photo(settings.telegram_bot_token, settings.telegram_chat_id, png, caption=caption)
        except Exception:
            traceback.print_exc()


def _report_today() -> date:
    """The week analyze() should summarize is the one that just ENDED, not
    the one that started today. The cron fires Sunday 20:00 Jerusalem, at
    which point "today" is day zero of a NEW week - start_of_week(today)
    would resolve to itself, reporting on however many hours had elapsed
    since midnight instead of the full week that just finished.

    Proven wrong against a real production log, not hypothetically: a past
    Sunday run logged "Service started 1x this week: Sun 11:32... Watched
    continuously for 20.0h" - a 20-hour "week". Subtracting a day makes a
    Sunday-evening run resolve back to the Sunday that started the week
    which just ended, matching what "weekly performance review" actually
    means. Caught 2026-08-27 while previewing what a real Sunday send would
    look like.
    """
    return datetime.now(clock.LOCAL_TZ).date() - timedelta(days=1)


def main() -> None:
    try:
        storage = Storage(settings.trades_db_path)
        bitget = client_from_settings(settings)

        resolved = resolve_pending(storage, bitget)
        print(f"Resolved {resolved} paper signal(s) this run.")

        # Appended to the report rather than sent separately: the whole failing
        # of the three never-worked features was that their silence had no
        # place to show up. This puts it in the one message that is read.
        #
        # bitget passed through so the fees section gets the REAL total from
        # Bitget's own fills, not an estimate - if that call fails, it fails
        # the whole run (caught by the except below), matching this file's
        # own "IT MUST SAY SO WHEN IT BREAKS" design rather than silently
        # reporting "not available" for what would actually be a real API
        # problem worth knowing about.
        weekly = analyze(storage, today=_report_today(), bitget=bitget)
        report = render(weekly)
        report = f"{report}\n\n{ledger.format_survey(ledger.survey(settings.trades_db_path, LEDGER_EXPECTATIONS))}"
        print(report)
        _alert(report)
        _send_charts(weekly)
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

    # Housekeeping, after the report is out: the scan heartbeat grows ~96 rows
    # a day and nothing reads further back than one week.
    try:
        prune_stale_heartbeats(storage)
    except Exception:
        # Never let tidying up fail a run whose report already went out.
        traceback.print_exc()

    # Only after the report has actually been sent. Recording it earlier would
    # make the heartbeat certify runs that produced nothing.
    record_success(settings.trades_db_path)
    # The heartbeat file and the ledger are two different watchers and only the
    # heartbeat was being written. The ledger therefore held WEEKLY_REPORT as
    # "never worked" while the report was in fact running every Sunday - which
    # would have made the report announce its own death, in itself, on the
    # eighth day of watching. An alarm that fires when nothing is wrong is the
    # failure this whole mechanism was built to avoid.
    ledger.try_record(settings.trades_db_path, ledger.WEEKLY_REPORT)


if __name__ == "__main__":
    main()
