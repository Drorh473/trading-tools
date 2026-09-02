from core.storage import Trade
from journal.report import render_markdown
from journal.stats import Stats, SymbolBreakdown


def _trade() -> Trade:
    return Trade(
        מספר_עסקה=1, תאריך="2026-01-01", שעת_כניסה="10:00", סימבול="BTCUSDT", כיוון="long",
        מחיר_כניסה=100.0, מחיר_יציאה=110.0, גודל_פוזיציה=1.0, גודל_שנסגר=1.0,
        סטופ_לוס_מקורי=90.0, יעד_רווח_מקורי=120.0, סטופ_לוס_בפועל=90.0, יעד_רווח_בפועל=120.0,
        סכום_סיכון=10.0, רווח_הפסד=10.0, מכפיל_R=1.0, מינוף=1.0, בוטלה=0,
        תגית_אסטרטגיה="A", הערות=None,
    )


def _stats_with_symbols(by_symbol: dict[str, SymbolBreakdown]) -> Stats:
    return Stats(
        total_closed=sum(b.count for b in by_symbol.values()),
        win_rate=0.5,
        expectancy=0.1,
        total_pnl=100.0,
        max_drawdown=10.0,
        r_multiples=[1.0],
        equity_curve=[100.0],
        best_trade=_trade(),
        worst_trade=_trade(),
        by_symbol=by_symbol,
    )


def test_by_symbol_section_lists_every_symbol():
    stats = _stats_with_symbols({
        "BTCUSDT": SymbolBreakdown(count=5, win_rate=0.6, expectancy=0.3, total_pnl=50.0, expectancy_drop_top3=0.1),
        "ETHUSDT": SymbolBreakdown(count=2, win_rate=0.5, expectancy=-0.2, total_pnl=-10.0, expectancy_drop_top3=None),
    })

    text = render_markdown(stats)

    assert "BTCUSDT" in text
    assert "ETHUSDT" in text
    assert "By symbol" in text


def test_by_symbol_section_is_sorted_by_expectancy_descending():
    stats = _stats_with_symbols({
        "WORST": SymbolBreakdown(count=4, win_rate=0.2, expectancy=-0.5, total_pnl=-40.0, expectancy_drop_top3=-0.6),
        "BEST": SymbolBreakdown(count=4, win_rate=0.8, expectancy=0.9, total_pnl=90.0, expectancy_drop_top3=0.7),
        "MIDDLE": SymbolBreakdown(count=4, win_rate=0.5, expectancy=0.1, total_pnl=10.0, expectancy_drop_top3=0.05),
    })

    text = render_markdown(stats)

    assert text.index("BEST") < text.index("MIDDLE") < text.index("WORST")


def test_by_symbol_section_shows_drop_top3_when_available_and_flags_when_not():
    stats = _stats_with_symbols({
        "BTCUSDT": SymbolBreakdown(count=5, win_rate=0.6, expectancy=0.3, total_pnl=50.0, expectancy_drop_top3=0.15),
        "ETHUSDT": SymbolBreakdown(count=2, win_rate=0.5, expectancy=-0.2, total_pnl=-10.0, expectancy_drop_top3=None),
    })

    text = render_markdown(stats)

    assert "0.15" in text
    assert "too few trades" in text.lower()
