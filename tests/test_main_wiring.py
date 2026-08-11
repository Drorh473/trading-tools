"""The execution whitelist controls whether real money moves, and until now
nothing tested it. These are cheap invariants, not behaviour tests: they
guard against a strategy silently ending up in the wrong set.
"""

from notifier.main import (
    AUTO_EXECUTE_TAGS,
    DRY_RUN_TAGS,
    EXIT_MANAGED_TAGS,
    LIVE_TAGS,
    MAX_LEVERAGE,
    build_strategies,
)


def test_every_registered_strategy_is_routed_somewhere():
    """A strategy absent from both sets executes nothing and reports nothing -
    it just quietly never trades, which is the failure mode hardest to notice.
    Strategy 3 sat exactly there until it was graduated.
    """
    registered = {tag for s in build_strategies() for tag in s.all_tags()}
    unrouted = registered - AUTO_EXECUTE_TAGS

    assert not unrouted, f"registered but neither live nor dry run: {sorted(unrouted)}"


def test_no_tag_is_both_live_and_dry_run():
    """RoutingExecutor builds one dict from both, so an overlap would resolve
    silently by dict ordering rather than by anything deliberate."""
    assert not (LIVE_TAGS & DRY_RUN_TAGS)


def test_every_live_strategy_may_manage_its_own_exits():
    """Exit management is the WEAKER permission - reduce-only take-profits and
    protective stop moves. A strategy allowed to open a position but not to
    place its exits would leave trades it opened without a target."""
    assert LIVE_TAGS <= EXIT_MANAGED_TAGS


def test_no_tag_is_routed_that_no_strategy_produces():
    """A stale tag in the whitelist is dead config that reads as intent -
    worse, a typo'd tag means the strategy it was meant for is NOT live while
    the list looks complete."""
    registered = {tag for s in build_strategies() for tag in s.all_tags()}
    orphans = AUTO_EXECUTE_TAGS - registered

    assert not orphans, f"whitelisted but no strategy produces them: {sorted(orphans)}"


def test_a_multi_variant_strategy_routes_every_variant_it_can_emit():
    """A strategy that classifies its own signals emits more than one tag, and
    checking only `tag` would leave the others unrouted while the whitelist
    looked complete - the exact silent failure the tests above exist for.
    Strategy 4 tags each block OB1.0 or OB2.0 from one instance.
    """
    multi = [s for s in build_strategies() if len(s.all_tags()) > 1]
    assert multi, "expected at least one multi-tag strategy; this test is inert without one"

    for strategy in multi:
        assert strategy.tag in strategy.all_tags()
        assert set(strategy.all_tags()) <= AUTO_EXECUTE_TAGS


def test_leverage_cap_is_sane():
    """Sizing solves leverage per trade; this only bounds it. A cap set absurdly
    high would let one tight-stop signal consume the whole account."""
    assert 1.0 < MAX_LEVERAGE <= 20.0
