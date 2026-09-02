"""Pick a representative subset of symbols, so iterating on a rule does not
cost the whole universe every time.

WHY STRATIFY RATHER THAN DRAW UNIFORMLY. The question a sample answers first
is what signal RATE the current rules produce, and that rate depends strongly
on how much history a symbol has - a listing with 60,000 1H bars and one with
1,500 are not interchangeable draws. A uniform sample over-represents whichever
band happens to hold the most symbols; sampling proportionally across bar-count
deciles keeps deep and shallow listings in the same proportion as the full
universe, so the rate scales up honestly.

WHY THIS PAYS FOR ITSELF TWICE. portfolio.generate's cache is keyed by
(symbol, instance_hash, hours), so a sampled run is a DOWN PAYMENT on the full
one rather than throwaway work: the symbols it scanned are already cached under
exactly the key the full run will look them up by, and the full run only pays
for the remainder.

Lifted out of backtest/generate_s4_deep.py, which had it inline and was the
only script that could sample.
"""

from __future__ import annotations

import random
from typing import Callable, Sequence

STRATA = 10


def stratified_sample(
    symbols: Sequence[str],
    n: int,
    size_of: Callable[[str], int],
    seed: int = 7,
) -> list[str]:
    """`n` symbols drawn across `size_of`'s deciles, sorted.

    n <= 0, or n >= len(symbols), means "no sampling" and returns everything -
    0 is the off-value every generator's own --sample flag already uses.

    `size_of` maps a symbol to its bar count. It is a callable rather than a
    dict because the bar caches disagree on shape: a DataFrame per symbol in
    backtest_bars.pkl, a dict of numpy columns in bars_1h_deep_np.pkl.
    """
    symbols = list(symbols)
    if n <= 0 or n >= len(symbols):
        return sorted(symbols)

    by_size = sorted(symbols, key=size_of)
    rng = random.Random(seed)
    step = max(1, len(by_size) // STRATA)

    picked: list[str] = []
    for start in range(0, len(by_size), step):
        band = by_size[start : start + step]
        take = max(1, round(n * len(band) / len(by_size)))
        picked.extend(rng.sample(band, min(take, len(band))))

    # Rounding each band up to at least one can overshoot n; trim from a
    # shuffled copy rather than the head, so the trim does not preferentially
    # drop the deepest band (picked is built in ascending size order).
    rng.shuffle(picked)
    return sorted(picked[:n])
