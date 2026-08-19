"""The one clock a trade is dated by.

The VM runs UTC (`timedatectl`: Local time = Universal time), so `date.today()`
and `datetime.now()` there return UTC while Dror reads every date as Israeli
time. That is not only a display difference. `weekly_review.start_of_week`
already defines the week in Asia/Jerusalem, deliberately - "the week he means
is Sunday-Saturday there" - and then filters rows whose `תאריך` was written in
UTC. Jerusalem runs UTC+2/+3, so a trade opened between midnight and 03:00
local carries the PREVIOUS UTC date and lands in the previous week's report.

One trade quietly attributed to the wrong week is the kind of error that never
announces itself: the totals still add up, they just add up somewhere else.

Market sessions are a separate matter and correctly use their own zones -
notifier.sessions on America/New_York, order_block on Asia/Tokyo. Those are
facts about exchanges. This is a fact about the person reading the report.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Asia/Jerusalem")


def now() -> datetime:
    """Timezone-aware local time."""
    return datetime.now(LOCAL_TZ)


def today() -> date:
    """The local calendar date, which is what a trade is filed under."""
    return now().date()
