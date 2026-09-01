"""generate_v2.py has no per-symbol/per-instance structure of its own to key
a cache by (it measures one EmaTrendV2 configuration across the whole
watchlist), so the only thing worth checking cheaply is "did anything that
shapes this output change at all" - a whole-output hash, not the per-instance
cache the multi-strategy portfolio.py path uses.
"""

import sys

import pytest

SWEPT_V2_CONSTANTS = (
    "EMA9_HOLD_BARS", "MIN_STOP_PCT", "MIN_NET_REWARD_RISK",
    "MIN_PIVOT_SPAN_BARS", "MIN_SWING_DRIFT_ATR", "MAX_EMA9_CROSSINGS",
    "REQUIRE_STRUCTURE_TREND",
)


@pytest.fixture(autouse=True)
def _restore_v2_thresholds():
    """Importing generate_v2 zeroes out notifier.strategies.ema_trend_v2's
    swept thresholds as a deliberate import-time side effect (see
    generate_v2.py's own comment on those assignments), and that mutation
    outlives the import for the rest of the process. Same fixture as
    tests/test_generators_forming_row.py - required in every file that
    imports this module, or a later, unrelated test file (test_ema_trend_v2.py,
    which assumes the real defaults) fails for a reason that has nothing to
    do with what it's testing.

    The capture below must run BEFORE generate_v2 is ever imported in this
    process, so every test in this file imports it lazily (inside the test
    body, not at module level) - a module-level import happens at collection
    time, before any fixture gets a chance to run, and would have this
    fixture "restore" the already-polluted values it started with."""
    import notifier.strategies.ema_trend_v2 as v2

    before = {name: getattr(v2, name) for name in SWEPT_V2_CONSTANTS}
    yield
    for name, value in before.items():
        setattr(v2, name, value)


def test_current_hash_is_deterministic():
    from backtest import generate_v2 as gv2

    assert gv2._current_hash(gv2.MEASURABLE) == gv2._current_hash(gv2.MEASURABLE)


def test_main_skips_generation_entirely_when_the_output_is_already_fresh(tmp_path, monkeypatch):
    from backtest import generate_v2 as gv2
    from backtest import instance_cache as ic

    out = tmp_path / "signals_v2.pkl"
    out.write_bytes(b"already generated")
    current = gv2._current_hash(gv2.MEASURABLE)
    ic.write_sidecar_hash(str(out), current)

    # CACHE points somewhere that does not exist - if main() tried to
    # actually generate, this read would raise and fail the test.
    monkeypatch.setattr(gv2, "CACHE", str(tmp_path / "does_not_exist.pkl"))
    monkeypatch.setattr(sys, "argv", ["generate_v2", "--out", str(out)])

    gv2.main()  # must return early, never touching CACHE

    assert out.read_bytes() == b"already generated", "the stale-looking output must be left untouched"
