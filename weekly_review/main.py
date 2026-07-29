"""Entrypoint: python -m weekly_review.main
Runs once a week (via a scheduled job — cron on the eventual always-on VM,
Task Scheduler locally) to compare this week's performance against the
all-time baseline and send the result via Telegram.
"""

from config import settings
from core.storage import Storage
from core.telegram_bot import send_message
from weekly_review.analyze import analyze, render_comparison


def main() -> None:
    storage = Storage(settings.trades_db_path)
    comparison = analyze(storage)
    report = render_comparison(comparison)

    print(report)
    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, report)


if __name__ == "__main__":
    main()
