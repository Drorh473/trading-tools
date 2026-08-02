"""Entrypoint: python -m weekly_review.main
Runs once a week (cron on the VM, Sunday 20:00 Asia/Jerusalem) to report
real trade activity plus paper-simulated signal outcomes for the week,
against the all-time baseline, and send the result via Telegram.

Resolving as many pending paper signals as possible happens here, right
before the report is built, rather than as a separate always-on job -
weekly is the only cadence anything currently reads this data on, so
there's no reason to resolve it more often than that.
"""

from config import settings
from core.bitget_client import client_from_settings
from core.storage import Storage
from core.telegram_bot import send_message
from journal.paper_sim import resolve_pending
from weekly_review.analyze import analyze, render


def main() -> None:
    storage = Storage(settings.trades_db_path)
    bitget = client_from_settings(settings)

    resolved = resolve_pending(storage, bitget)
    print(f"Resolved {resolved} paper signal(s) this run.")

    report = render(analyze(storage))

    print(report)
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, report)


if __name__ == "__main__":
    main()
