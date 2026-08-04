from datetime import datetime
from zoneinfo import ZoneInfo

from notifier.sessions import NEW_YORK, may_signal_now

UTC = ZoneInfo("UTC")


def _at(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=UTC)


def test_crypto_is_never_gated():
    # 3am ET on a Sunday - crypto does not care.
    assert may_signal_now("BTCUSDT", is_rwa=False, now=_at("2026-08-02 07:00")) is True


def test_us_equity_blocked_outside_cash_hours():
    # The three signals that got the intraday instance disabled, all of them
    # hours after their shares stopped trading.
    assert may_signal_now("AXTIUSDT", True, _at("2026-08-04 01:50")) is False  # 21:50 ET
    assert may_signal_now("TSLAUSDT", True, _at("2026-08-04 04:35")) is False  # 00:35 ET
    assert may_signal_now("TSLAUSDT", True, _at("2026-08-04 04:45")) is False  # 00:45 ET


def test_us_equity_allowed_mid_session():
    assert may_signal_now("INTCUSDT", True, _at("2026-08-03 15:00")) is True  # 11:00 ET


def test_no_new_entry_in_the_last_half_hour():
    # INTCUSDT fired at 15:55 ET, five minutes before the bell, and then had to
    # hold a 0.6% stop through 17.5 hours of illiquid overnight tape.
    assert may_signal_now("INTCUSDT", True, _at("2026-08-03 19:55")) is False  # 15:55 ET
    assert may_signal_now("INTCUSDT", True, _at("2026-08-03 19:29")) is True  # 15:29 ET
    assert may_signal_now("INTCUSDT", True, _at("2026-08-03 19:31")) is False  # 15:31 ET


def test_weekends_are_closed():
    assert may_signal_now("TSLAUSDT", True, _at("2026-08-01 17:00")) is False  # Saturday


def test_metals_and_energy_are_never_gated():
    # Tokenized, but tracking near-24h futures markets - a US equity gate
    # would silence them for most of the day for no reason.
    for symbol in ("XAUUSDT", "XAGUSDT", "XAUTUSDT", "CLUSDT", "BZUSDT"):
        assert may_signal_now(symbol, True, _at("2026-08-04 03:00")) is True


def test_asian_listings_are_excluded_entirely():
    # A US gate would be exactly backwards for these - admitting them when
    # their home market is shut - so intraday skips them altogether.
    for symbol in ("SAMSUNGUSDT", "ZHIPUUSDT", "ZHIPUHKDUSDT", "SKHYNIXUSDT"):
        assert may_signal_now(symbol, True, _at("2026-08-03 15:00")) is False
        assert may_signal_now(symbol, True, _at("2026-08-04 03:00")) is False


def test_the_session_follows_daylight_saving_rather_than_a_fixed_offset():
    # 09:30 ET is 13:30 UTC in summer and 14:30 UTC in winter. Hardcoding
    # either is wrong for half the year, so the zone does the work.
    summer_open = datetime(2026, 8, 3, 9, 45, tzinfo=NEW_YORK)
    winter_open = datetime(2026, 12, 3, 9, 45, tzinfo=NEW_YORK)
    assert may_signal_now("INTCUSDT", True, summer_open) is True
    assert may_signal_now("INTCUSDT", True, winter_open) is True
    assert summer_open.utcoffset() != winter_open.utcoffset()
