"""run_all's core loop: run every step, keep going even when one fails, and
report each step's outcome. Steps are exercised as REAL subprocesses (trivial
ones, not the actual backtests) rather than mocked, so this proves the loop
actually shells out and actually observes a real exit code - not that it
calls a mock the expected number of times.
"""

import sys

from backtest.run_all import _filter_steps, run_steps


def _step(name, code):
    return (name, [sys.executable, "-c", f"import sys; sys.exit({code})"])


def test_a_successful_step_records_exit_code_zero():
    results = run_steps([_step("ok", 0)])

    assert results == [("ok", 0, results[0][2])]


def test_a_failing_step_does_not_abort_the_remaining_steps():
    results = run_steps([_step("first", 1), _step("second", 0), _step("third", 2)])

    assert [(name, code) for name, code, _elapsed in results] == [
        ("first", 1), ("second", 0), ("third", 2),
    ]


def test_elapsed_time_is_recorded_and_non_negative():
    results = run_steps([_step("ok", 0)])

    assert results[0][2] >= 0.0


def test_only_keeps_steps_matching_the_substring_case_insensitively():
    steps = [("Strategy 3 swing", []), ("Strategy 4 deep signals", []), ("portfolio", [])]

    kept = _filter_steps(steps, only=["STRATEGY 3"], skip=[])

    assert [name for name, _ in kept] == ["Strategy 3 swing"]


def test_skip_removes_steps_matching_the_substring():
    steps = [("Strategy 3 swing", []), ("Strategy 4 deep signals", []), ("Strategy 4 deep replay", [])]

    kept = _filter_steps(steps, only=[], skip=["strategy 4"])

    assert [name for name, _ in kept] == ["Strategy 3 swing"]


def test_no_filters_keeps_every_step():
    steps = [("a", []), ("b", [])]

    assert _filter_steps(steps, only=[], skip=[]) == steps
