"""A stratified symbol sample, so iterating on a rule does not cost the full
universe every time.

generate_s4_deep.py already had this logic inline; run.py and run_s3_swing.py
did not, so the only way to ask "did my edit help?" was a full 637-symbol,
19-hour scan. Stratifying by bar count (rather than sampling uniformly) keeps
deep and shallow listings represented in proportion, so a signal RATE measured
on the sample scales up honestly.
"""

import pytest

from backtest.sampling import stratified_sample


def _sizes(symbols):
    """SYM0 has 0 bars, SYM1 has 1, ... - so a sample's spread is visible."""
    return {s: int(s[3:]) for s in symbols}


def _universe(n=100):
    return [f"SYM{i}" for i in range(n)]


def test_asking_for_everything_returns_everything():
    symbols = _universe(10)

    picked = stratified_sample(symbols, 10, _sizes(symbols).get)

    assert sorted(picked) == sorted(symbols)


def test_asking_for_more_than_exists_returns_everything_not_an_error():
    symbols = _universe(5)

    picked = stratified_sample(symbols, 999, _sizes(symbols).get)

    assert sorted(picked) == sorted(symbols)


def test_it_returns_exactly_the_requested_count():
    symbols = _universe(100)

    picked = stratified_sample(symbols, 20, _sizes(symbols).get)

    assert len(picked) == 20


def test_the_same_seed_gives_the_same_sample():
    """A measurement nobody can reproduce is not evidence - two runs quoting
    'the 100-symbol sample' must mean the same 100 symbols."""
    symbols = _universe(100)

    first = stratified_sample(symbols, 20, _sizes(symbols).get, seed=7)
    second = stratified_sample(symbols, 20, _sizes(symbols).get, seed=7)

    assert first == second


def test_a_different_seed_gives_a_different_sample():
    symbols = _universe(100)

    first = stratified_sample(symbols, 20, _sizes(symbols).get, seed=7)
    second = stratified_sample(symbols, 20, _sizes(symbols).get, seed=8)

    assert first != second


def test_the_sample_spans_the_size_range_rather_than_clustering():
    """The whole point of stratifying: a uniform draw would over-represent
    whatever band happens to hold most symbols, and signal rate per symbol
    depends strongly on how many bars that symbol has."""
    symbols = _universe(100)
    sizes = _sizes(symbols)

    picked = stratified_sample(symbols, 20, sizes.get, seed=7)
    drawn = sorted(sizes[s] for s in picked)

    assert drawn[0] < 15, f"nothing from the shallow end: {drawn}"
    assert drawn[-1] > 85, f"nothing from the deep end: {drawn}"


def test_an_empty_universe_is_not_an_error():
    assert stratified_sample([], 10, lambda s: 0) == []


def test_zero_or_negative_n_means_no_sampling_take_everything():
    """0 is the 'off' value every generator's --sample flag already uses."""
    symbols = _universe(10)

    assert sorted(stratified_sample(symbols, 0, _sizes(symbols).get)) == sorted(symbols)


def test_the_result_is_sorted_so_downstream_order_is_stable():
    symbols = _universe(100)

    picked = stratified_sample(symbols, 20, _sizes(symbols).get, seed=7)

    assert picked == sorted(picked)
