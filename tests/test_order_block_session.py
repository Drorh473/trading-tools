"""The Asia-session read, and the gate that was never wired.

Both bugs were found while trying to answer "does avoiding Asia-session blocks
improve the odds" and discovering the question could not be asked: every one
of the 11,243 signals in the deep backtest set came back "not Asia".
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from notifier.strategies.order_block import ASIA_TZ, OrderBlockStrategy


def _window(ts_values):
    return pd.DataFrame({"ts": ts_values})


def test_a_bare_int_timestamp_is_read_as_milliseconds():
    """The deep backtest caches keep raw int64 ms; live frames hold datetime64.

    pd.Timestamp on a bare int reads NANOSECONDS, which put every backtest bar
    on 1970-01-01 at hour 9 - one hour outside ASIA_SESSION_HOURS - so the
    session read was uniformly False in backtest while being correct live.
    """
    strat = OrderBlockStrategy("1H")
    # 2026-07-30 03:00 UTC = 12:00 in Tokyo, comfortably OUTSIDE Asia hours.
    ms = 1785380400000
    assert pd.Timestamp(ms, unit="ms").tz_localize("UTC").astimezone(ASIA_TZ).hour == 12
    assert strat._in_asia_session(_window([np.int64(ms)]), 0) is False

    # 2026-07-29 21:00 UTC = 06:00 in Tokyo, INSIDE Asia hours. Read as
    # nanoseconds this lands in 1970 and reports hour 9 - not Asia - which is
    # the failure this test exists for.
    ms_asia = 1785358800000
    assert pd.Timestamp(ms_asia, unit="ms").tz_localize("UTC").astimezone(ASIA_TZ).hour == 6
    assert strat._in_asia_session(_window([np.int64(ms_asia)]), 0) is True


def test_datetime64_timestamps_still_read_correctly():
    """The live path must not regress - bars_dataframe hands over datetime64."""
    strat = OrderBlockStrategy("1H")
    asia = pd.Timestamp("2026-07-29T21:00:00")      # 06:00 Tokyo
    not_asia = pd.Timestamp("2026-07-30T03:00:00")  # 12:00 Tokyo
    assert strat._in_asia_session(_window([asia]), 0) is True
    assert strat._in_asia_session(_window([not_asia]), 0) is False


def test_the_session_gate_defaults_off_because_that_is_what_runs_today():
    """session_gated was stored and never read, so every live instance has in
    fact been ungated. Honouring the flag without flipping the default would
    have started refusing Asia blocks in production as a side effect."""
    assert OrderBlockStrategy("1H").session_gated is False
    assert OrderBlockStrategy("15m").session_gated is False
    assert OrderBlockStrategy("1H", session_gated=True).session_gated is True


def test_the_gate_is_actually_consulted():
    """Guards the wiring itself: _signal_for must read the flag. Asserted on
    the source rather than a full setup, because constructing a real Asia
    order block end-to-end would test the detector, not the gate."""
    import inspect

    src = inspect.getsource(OrderBlockStrategy._signal_for)
    assert "self.session_gated" in src, "_signal_for no longer consults the gate"
