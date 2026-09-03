"""The month a monthly report covers.

Same reasoning as weekly_review's _report_today: the job runs just after the
month turns over, and the month it must summarise is the one that just ENDED.
Resolving "this month" on the 1st would report on a few hours.

Jerusalem throughout - trade rows are dated in local time (core.clock), so a
UTC month boundary would push the last evening's trades into the wrong report.
"""

from datetime import date, datetime, timedelta

from core import clock

LOCAL_TZ = clock.LOCAL_TZ


def last_full_month(today: date | None = None) -> tuple[date, date]:
    """(first day of the month that just ended, first day of the current one).

    Half-open: [start, end). The end is the NEXT month's first day rather than
    the last day of the reported month, so every "is this row in the window"
    check is a plain start <= d < end with no last-day-of-February special
    case.
    """
    today = today or datetime.now(LOCAL_TZ).date()
    this_month_start = today.replace(day=1)
    prev_month_end = this_month_start
    prev_month_start = (this_month_start - timedelta(days=1)).replace(day=1)
    return prev_month_start, prev_month_end


def to_ms(day: date) -> int:
    """Local midnight on `day`, as the epoch milliseconds Bitget wants."""
    return int(datetime(day.year, day.month, day.day, tzinfo=LOCAL_TZ).timestamp() * 1000)


def month_name(start: date) -> str:
    return start.strftime("%B %Y")
