"""One shared place for win rate / total R / expectancy / drop-top-3 / per-tag
/ exit-reason stats, instead of the copy-pasted variants in run.py's report()
and run_s3_swing.py's report_arm(). The point (memory: sweep-past-the-optimum)
is that drop-top-3 - does the edge survive losing its three biggest winners -
must be computed the same way everywhere, not re-derived (or silently
skipped) each time a new script needs a report.
"""

from backtest.engine import Closed
from backtest.stats import summarize


def _trade(r, tag="t", reason="stop", pnl=None):
    return Closed(symbol="XUSDT", tag=tag, direction="long", opened_at=0, closed_at=1,
                  r=r, pnl=pnl if pnl is not None else r, reason=reason)


def test_summarizing_no_trades_gives_zero_n():
    s = summarize([])

    assert s.n == 0


def test_win_rate_total_r_and_expectancy():
    trades = [_trade(2.0), _trade(-1.0), _trade(-1.0), _trade(3.0)]

    s = summarize(trades)

    assert s.n == 4
    assert s.win_rate == 0.5
    assert s.total_r == 3.0
    assert s.expectancy == 0.75


def test_win_is_judged_by_pnl_not_by_r_being_positive():
    """Matches the existing convention in run.py/run_s3_swing.py: `c.pnl > 0`,
    not `c.r > 0` - they agree for every real trade, but a summary must not
    quietly redefine what a win is."""
    trades = [_trade(r=0.1, pnl=-0.5)]

    s = summarize(trades)

    assert s.wins == 0


def test_drop_top3_is_none_with_three_or_fewer_trades():
    s = summarize([_trade(1.0), _trade(2.0), _trade(3.0)])

    assert s.drop_top3_n is None
    assert s.drop_top3_expectancy is None


def test_drop_top3_drops_the_three_biggest_winners_by_r_not_the_three_worst_losers():
    # Sorted by R descending: 10, 5, 4, then the kept tail: -1, -2, -3.
    trades = [_trade(10.0), _trade(5.0), _trade(4.0), _trade(-1.0), _trade(-2.0), _trade(-3.0)]

    s = summarize(trades)

    assert s.drop_top3_n == 3
    assert s.drop_top3_total_r == -6.0
    assert s.drop_top3_expectancy == -2.0


def test_by_tag_breakdown():
    trades = [_trade(2.0, tag="A", pnl=20.0), _trade(-1.0, tag="A", pnl=-10.0),
              _trade(3.0, tag="B", pnl=30.0)]

    s = summarize(trades)

    assert set(s.by_tag) == {"A", "B"}
    assert s.by_tag["A"].n == 2
    assert s.by_tag["A"].wins == 1
    assert s.by_tag["A"].total_r == 1.0
    assert s.by_tag["A"].pnl == 10.0
    assert s.by_tag["B"].n == 1
    assert s.by_tag["B"].expectancy == 3.0


def test_exit_reason_counts():
    trades = [_trade(1.0, reason="stop"), _trade(1.0, reason="stop"), _trade(1.0, reason="target")]

    s = summarize(trades)

    assert s.exit_reasons == {"stop": 2, "target": 1}
