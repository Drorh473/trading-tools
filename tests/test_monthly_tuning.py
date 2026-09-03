from datetime import date

import pytest

from core.storage import SignalRecord, Trade
from monthly_review.cadence import cadence_for
from monthly_review.tuning import FeeModelCheck, entry_slippage_bp


def _signal(sid=1, tag="Strategy 1 1H", entry=100.0, direction="long", trade_id=None, at="2026-08-05T10:00:00"):
    return SignalRecord(
        id=sid, dispatched_at=at, symbol="BTCUSDT", direction=direction,
        entry_price=entry, stop_loss=95.0, take_profit=110.0, strategy_tag=tag,
        confluence=None, decision="approved", trade_id=trade_id,
        paper_r=None, paper_resolved_at=None,
    )


def _trade(direction="long", fill=100.0):
    return Trade(
        מספר_עסקה=1, תאריך="2026-08-05", שעת_כניסה="10:00", סימבול="BTCUSDT",
        כיוון=direction, מחיר_כניסה=fill, מחיר_יציאה=None, גודל_פוזיציה=1.0,
        גודל_שנסגר=None, סטופ_לוס_מקורי=95.0, יעד_רווח_מקורי=110.0,
        סטופ_לוס_בפועל=None, יעד_רווח_בפועל=None, סכום_סיכון=2.0,
        רווח_הפסד=None, מכפיל_R=None, מינוף=10.0, בוטלה=0,
        תגית_אסטרטגיה="Strategy 1 1H", הערות=None,
    )


def test_a_long_filling_above_plan_is_a_cost():
    slip = entry_slippage_bp(_signal(entry=100.0), _trade("long", fill=100.1))
    assert slip == pytest.approx(10.0)


def test_the_same_fill_is_a_GAIN_for_a_short():
    """Signing by direction is the point: averaging unsigned differences would
    report an instance that consistently fills favourably as bleeding."""
    slip = entry_slippage_bp(_signal(entry=100.0, direction="short"), _trade("short", fill=100.1))
    assert slip == pytest.approx(-10.0)


def test_slippage_is_none_when_the_trade_never_filled():
    assert entry_slippage_bp(_signal(), _trade(fill=None)) is None


def test_fee_model_check_fires_on_the_strategy_21_shaped_error():
    """Sized against 0.08% while really paying 0.12% - a 50% under-estimate."""
    check = FeeModelCheck(actual=1.20, modelled=0.80)
    assert check.fires
    assert check.ratio == pytest.approx(1.5)


def test_fee_model_check_tolerates_ordinary_estimation_error():
    assert not FeeModelCheck(actual=1.05, modelled=1.00).fires


def test_fee_model_check_has_no_opinion_when_nothing_was_modelled():
    """A month with no closed trades must not report a perfect fee model."""
    check = FeeModelCheck(actual=0.0, modelled=0.0)
    assert check.ratio is None
    assert not check.fires


def test_an_instance_that_never_signalled_is_flagged_not_scored_as_perfect():
    """Strategy 3 is live and has produced zero signals ever. A gap-based
    score with no gaps to measure must not read as healthy."""
    c = cadence_for("Strategy 3 1D/1H", [], date(2026, 8, 1), date(2026, 9, 1), threshold_days=4.0)

    assert c.silent_all_month and c.alarming
    assert c.longest_silence_days == pytest.approx(31.0)


def test_silence_is_measured_from_the_window_edges_not_only_between_signals():
    """Two signals, the last on the 2nd at 09:00, then nothing until the month
    ends: a 29.6-day silence. Measuring only signal-to-signal would score the
    same month as a 1-day gap and call it healthy."""
    signals = [_signal(1, at="2026-08-01T09:00:00"), _signal(2, at="2026-08-02T09:00:00")]
    c = cadence_for("Strategy 1 1H", signals, date(2026, 8, 1), date(2026, 9, 1), threshold_days=4.0)

    assert c.longest_silence_days == pytest.approx(29.625, abs=0.01)
    assert c.breaches == 1


def test_a_steadily_signalling_instance_is_quiet():
    signals = [_signal(i, at=f"2026-08-{day:02d}T09:00:00") for i, day in enumerate(range(1, 32, 2))]
    c = cadence_for("Strategy 1 1H", signals, date(2026, 8, 1), date(2026, 9, 1), threshold_days=4.0)

    assert not c.alarming
    assert c.signals == 16
