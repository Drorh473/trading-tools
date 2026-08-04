"""When a tokenized instrument may be signalled on.

Intraday strategies read volume and structure off bars that assume a market
which is always open. That holds for crypto and breaks badly for tokenized
equities: Strategy 3's 1H/5m instance fired on AXTIUSDT at 21:50 ET and on
TSLAUSDT twice past midnight ET, hours after those shares stopped trading, on
"consolidations" that were really just the overnight tape going flat.

Bitget's contracts endpoint carries `isRwa`, which is a reliable
discriminator - present on all 100 watchlist symbols, YES on 32 - so no
maintained symbol list is needed to tell a tokenized instrument from a coin.

What isRwa does NOT tell you is which market it tracks, and those keep
different hours:

  - US equities (the large majority): a cash session, gated here.
  - Metals and energy: near-24h futures markets, so gating them to US equity
    hours would silence them for most of the day for no reason.
  - Asian listings: their real session runs while New York sleeps, so a US
    gate would be exactly backwards - blocking them when they are liquid and
    admitting them when they are not. Excluded from intraday entirely rather
    than gated wrongly; trading them properly needs real KRX/HKEX calendars,
    which is a different piece of work.

Anything tokenized that is not named below is treated as a US equity, which
is the safe default: the failure mode is missing a signal, not taking one
into a market that is shut.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")

# Near-24h futures markets. Tokenized, but not on an equity calendar.
CONTINUOUS_RWA = frozenset(
    {
        "XAUUSDT",  # gold
        "XAGUSDT",  # silver
        "XAUTUSDT",  # Tether Gold
        "CLUSDT",  # WTI crude
        "BZUSDT",  # Brent crude
    }
)

# Listed in Seoul or Hong Kong: a US gate would admit them at exactly the
# wrong hours, so intraday strategies skip them altogether.
ASIAN_LISTED_RWA = frozenset(
    {
        "SAMSUNGUSDT",
        "SKHYUSDT",
        "SKHYNIXUSDT",
        "ZHIPUUSDT",
        "ZHIPUHKDUSDT",
    }
)

MARKET_OPEN_MINUTES = 9 * 60 + 30  # 09:30 ET
MARKET_CLOSE_MINUTES = 16 * 60  # 16:00 ET
# No new intraday entry inside the last half hour. INTCUSDT fired at 15:55 ET
# - five minutes before the bell - which then had to hold a 0.6% stop through
# 17.5 hours of illiquid overnight tape before the market could resolve it.
CLOSE_BUFFER_MINUTES = 30


def may_signal_now(symbol: str, is_rwa: bool, now: datetime | None = None) -> bool:
    """Whether an intraday strategy may fire on this symbol right now.

    Crypto is always True. Tokenized instruments are judged against the market
    they actually track.
    """
    if not is_rwa:
        return True
    if symbol in ASIAN_LISTED_RWA:
        return False
    if symbol in CONTINUOUS_RWA:
        return True
    return _us_cash_session_open(now)


def _us_cash_session_open(now: datetime | None = None) -> bool:
    """True during the US cash session, stopping CLOSE_BUFFER_MINUTES early.

    Uses the America/New_York zone rather than a fixed UTC offset so the
    session tracks daylight saving without a table: 09:30 ET is 13:30 UTC in
    summer and 14:30 UTC in winter, and hardcoding either is wrong for half
    the year.

    Holidays are deliberately not modelled. On a market holiday the bars are
    flat rather than absent, and a flat tape produces no consolidation
    breakout to fire on - so the cost of ignoring them is far smaller than the
    cost of carrying a calendar that silently goes stale.
    """
    now = datetime.now(NEW_YORK) if now is None else now.astimezone(NEW_YORK)
    if now.weekday() >= 5:  # Saturday, Sunday
        return False
    minutes = now.hour * 60 + now.minute
    return MARKET_OPEN_MINUTES <= minutes < (MARKET_CLOSE_MINUTES - CLOSE_BUFFER_MINUTES)
