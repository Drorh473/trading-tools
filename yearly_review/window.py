"""The calendar year a yearly report covers.

Same reasoning as monthly_review's last_full_month: the job runs just after
the year turns over, and the year it must summarise is the one that just
ENDED. Resolving "this year" on Jan 1 would report on a few hours.

Jerusalem throughout - trade rows are dated in local time (core.clock), so a
UTC year boundary would push the last evening of December into the wrong
report.
"""

from datetime import date, datetime

from core import clock

LOCAL_TZ = clock.LOCAL_TZ


def last_full_year(today: date | None = None) -> tuple[date, date]:
    """(Jan 1 of the year that just ended, Jan 1 of the current one).

    Half-open: [start, end). The end is next Jan 1 rather than Dec 31 of the
    reported year, so every "is this row in the window" check is a plain
    start <= d < end.
    """
    today = today or datetime.now(LOCAL_TZ).date()
    this_year_start = date(today.year, 1, 1)
    prev_year_start = date(today.year - 1, 1, 1)
    return prev_year_start, this_year_start


def to_ms(day: date) -> int:
    """Local midnight on `day`, as the epoch milliseconds Bitget wants."""
    return int(datetime(day.year, day.month, day.day, tzinfo=LOCAL_TZ).timestamp() * 1000)


def year_name(start: date) -> str:
    return str(start.year)
