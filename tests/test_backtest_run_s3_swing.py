"""run_s3_swing.py generates ONLY Strategy 3 swing and replays only it.

Those two lists have to partition INSTANCES exactly. Generating an instance
the replay then skips is what this script used to do, and it cost ~66% of a
19.5-hour run. Skipping one that was never generated is worse - it measures
nothing and says so nowhere.
"""

from backtest import portfolio as pf
from backtest import run_s3_swing as s3


def test_generated_and_skipped_partition_instances():
    """Adding an instance to INSTANCES without deciding which side it falls
    on should fail here rather than silently cost hours or measure nothing."""
    covered = set(s3.GENERATE_POS) | set(s3.SKIP_POS)

    assert covered == set(range(len(pf.INSTANCES)))
    assert not (set(s3.GENERATE_POS) & set(s3.SKIP_POS)), "an instance cannot be both"


def test_it_generates_exactly_the_strategy_3_swing_instance():
    assert s3.GENERATE_POS == [s3.S3_SWING_POS]
    strategy = pf.INSTANCES[s3.S3_SWING_POS][0]
    assert type(strategy).__name__ == "VolumeRun", (
        "INSTANCES reordered; S3_SWING_POS is stale and this script would "
        "generate and measure some other strategy under Strategy 3's name"
    )
