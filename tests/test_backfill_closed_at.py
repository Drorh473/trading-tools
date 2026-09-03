"""The matching in the backfill tool is inference, not a lookup: Bitget's
position history shares no id with our trade rows. A wrong close time is worse
than a missing one - a missing one is visibly missing, a wrong one silently
corrupts every excursion measurement built on it later.
"""

from core.storage import Trade
from tools.backfill_closed_at import PRICE_TOLERANCE, _matches


def _trade(direction="long", exit_price=110.0):
    return Trade(
        מספר_עסקה=1, תאריך="2026-08-05", שעת_כניסה="10:00", סימבול="BTCUSDT",
        כיוון=direction, מחיר_כניסה=100.0, מחיר_יציאה=exit_price, גודל_פוזיציה=1.0,
        גודל_שנסגר=1.0, סטופ_לוס_מקורי=95.0, יעד_רווח_מקורי=110.0,
        סטופ_לוס_בפועל=None, יעד_רווח_בפועל=None, סכום_סיכון=5.0,
        רווח_הפסד=10.0, מכפיל_R=2.0, מינוף=10.0, בוטלה=0,
        תגית_אסטרטגיה="Strategy 1 1H", הערות=None,
    )


def _row(direction="long", exit_price=110.0):
    return {"direction": direction, "exit_price": exit_price, "close_time_ms": 1}


def test_the_same_trade_matches_despite_averaging_differences():
    """Our exit price is the fill we recorded; Bitget's is an average across
    the closing legs. Exact equality would reject nearly every real match."""
    assert _matches(_trade(exit_price=110.0), _row(exit_price=110.05))


def test_a_different_trade_on_the_same_symbol_does_not_match():
    assert not _matches(_trade(exit_price=110.0), _row(exit_price=118.0))


def test_the_opposite_side_never_matches():
    """The account runs hedge mode, so a symbol can hold a long and a short at
    once and both can close near the same price."""
    assert not _matches(_trade(direction="long"), _row(direction="short"))


def test_a_trade_with_no_exit_price_is_never_matched():
    """Nothing to match on. Guessing here would attach a close time to a trade
    whose exit was never recorded."""
    assert not _matches(_trade(exit_price=None), _row())


def test_the_tolerance_boundary_is_where_it_says_it_is():
    inside = 110.0 * (1 + PRICE_TOLERANCE * 0.9)
    outside = 110.0 * (1 + PRICE_TOLERANCE * 1.5)

    assert _matches(_trade(exit_price=inside), _row(exit_price=110.0))
    assert not _matches(_trade(exit_price=outside), _row(exit_price=110.0))
