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
    format_trade_dump,
    parse_manage_args,
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


def test_manage_args_parse_the_useful_forms():
    assert parse_manage_args(["11", "0.6081"]) == (11, 0.6081, None)
    assert parse_manage_args(["11", "0.6081", "0.5373"]) == (11, 0.6081, 0.5373)


def test_manage_args_explain_themselves_rather_than_throwing():
    """A mistyped command reaches a live trader, so every rejection has to
    say what to type instead - and none of it may raise into the handler."""
    assert "Usage:" in parse_manage_args([])
    assert "Usage:" in parse_manage_args(["11"])
    assert "Usage:" in parse_manage_args(["11", "0.6", "0.5", "extra"])
    assert "isn't a trade id" in parse_manage_args(["APTUSDT", "0.6081"])
    assert "have to be prices" in parse_manage_args(["11", "cheap"])
    assert "have to be prices" in parse_manage_args(["11", "0.6081", "later"])


def test_trade_dump_no_longer_carries_the_dispatching_signals_reasoning(tmp_path):
    """/trade used to append the strategy's own `reason` string and its
    Confluence/limit/runner notes (deliberately excluded from every alert,
    see Signal.reason) - Dror asked for that removed 2026-09-04, since the
    trade's own recorded fields already answer what happened. Still checks
    the fields that DO belong here: whether the shipped stop/target diverged
    from the bot's original plan.
    """
    from core.storage import Storage
    from notifier.strategies.base import Signal, signal_to_json

    storage = Storage(str(tmp_path / "trades.db"))
    trade_id = storage.create_pending(
        symbol="APTUSDT", direction="long", proposed_stop=0.60, proposed_target=0.70,
        strategy_tag="Strategy 1 1H",
    )
    storage.confirm_entry(
        trade_id, entry_price=0.6081, position_size=100,
        actual_stop=0.605, actual_target=0.70, leverage=5.0,
    )
    signal = Signal(
        symbol="APTUSDT", direction="long", entry_price=0.6081, stop_loss=0.60,
        strategy_tag="Strategy 1 1H", reason="61.8% Fib retrace with BTC levels agreeing",
    )
    sid = storage.log_signal(
        symbol="APTUSDT", direction="long", entry_price=0.6081, stop_loss=0.60,
        take_profit=0.70, strategy_tag="Strategy 1 1H", confluence="BTC levels timing",
        signal_json=signal_to_json(signal),
    )
    storage.link_signal_trade(sid, trade_id)

    trade = storage.get_trade(trade_id)
    dump = format_trade_dump(trade)

    assert "61.8% Fib retrace with BTC levels agreeing" not in dump
    assert "BTC levels timing" not in dump
    assert "strategy reasoning" not in dump
    assert "diverged from the bot's original plan" in dump, "stop moved 0.60 -> 0.605"
    assert "open" in dump


async def test_cancelling_the_running_task_from_within_it_must_not_escape_uncaught():
    """Reproduces async_main()'s SIGTERM shutdown pattern in isolation - a
    nested coroutine cancels the CURRENT task to unwind scanner.run_forever()
    after a graceful cleanup. Confirmed live on the VM, 2026-08-26: without
    catching CancelledError around the awaited call, the cancellation
    completes the cleanup correctly (cancel_all_pending ran, bot.stop()
    completed) and then still escapes asyncio.run() as an uncaught error -
    systemd logged "Failed with result 'exit-code'" for a shutdown that had
    in fact gone perfectly.
    """
    import asyncio as asyncio_module

    async def run_forever():
        await asyncio_module.sleep(10)

    async def main():
        task = asyncio_module.current_task()

        async def on_sigterm():
            task.cancel()

        asyncio_module.create_task(on_sigterm())
        try:
            await run_forever()
        except asyncio_module.CancelledError:
            return "clean shutdown"
        finally:
            pass  # stands in for async_main()'s `finally: await bot.stop()`

    result = await asyncio_module.wait_for(main(), timeout=2.0)
    assert result == "clean shutdown"


# ---------------------------------------------------------------------------
# /risk: a live readout of open risk against the aggregate cap, so "how much
# room is left" doesn't require doing the arithmetic by hand from /status
# and a mental model of the cap.
# ---------------------------------------------------------------------------


def test_format_risk_readout_shows_equity_open_risk_and_cap():
    from notifier.main import format_risk_readout

    text = format_risk_readout(equity=100.0, open_risk=8.0, committed_margin=40.0, cap=10.0)

    assert "$100.00" in text
    assert "$8.00" in text
    assert "8.0%" in text  # open risk as % of equity
    assert "$10.00" in text  # the cap itself
    assert "$40.00" in text  # committed margin


def test_format_risk_readout_shows_remaining_headroom():
    from notifier.main import format_risk_readout

    text = format_risk_readout(equity=100.0, open_risk=6.0, committed_margin=0.0, cap=10.0)

    assert "$4.00" in text  # 10 - 6 headroom left


def test_format_risk_readout_handles_zero_equity_without_dividing_by_zero():
    from notifier.main import format_risk_readout

    text = format_risk_readout(equity=0.0, open_risk=0.0, committed_margin=0.0, cap=0.0)

    assert "$0.00" in text  # must not raise


def _resume_open_trades_call_kwargs() -> dict[str, str]:
    """Statically finds async_main's resume_open_trades(...) call and returns
    {kwarg_name: source of the value}, without actually running async_main -
    which needs a live Bitget/Telegram connection to get that far."""
    import ast
    import inspect
    import textwrap

    import notifier.main as main_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.async_main)))
    calls = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "resume_open_trades"
    ]
    assert len(calls) == 1, "expected exactly one resume_open_trades(...) call in async_main"
    return {kw.arg: ast.unparse(kw.value) for kw in calls[0].keywords}


def test_every_resume_open_trades_kwarg_is_real():
    """2026-09-03: an on_resize=scanner._on_resize argument built and tested
    on a different branch (against its own Scanner) leaked into a commit
    here via an unstaged edit sitting in the working tree at commit time.
    Neither resume_open_trades nor Scanner on THIS branch had ever gained a
    matching piece, so it deployed clean - pytest never calls async_main -
    and crashed the live service in a restart loop within seconds of
    startup. Checks both halves statically instead: every keyword this call
    passes must be a parameter resume_open_trades actually accepts, and
    every scanner.<name> callback must be a real Scanner attribute.
    """
    import inspect

    from execution.tracker import resume_open_trades
    from notifier.scanner import Scanner

    kwargs = _resume_open_trades_call_kwargs()
    accepted = set(inspect.signature(resume_open_trades).parameters)
    for name, rhs in kwargs.items():
        assert name in accepted, f"resume_open_trades has no parameter '{name}'"
        if rhs.startswith("scanner."):
            method_name = rhs.split(".", 1)[1]
            assert hasattr(Scanner, method_name), (
                f"resume_open_trades(...) passes {rhs}, but Scanner has no '{method_name}'"
            )


def _scanner_callback_kwargs_in_async_main() -> list[tuple[str, str, str]]:
    """Every (call target, kwarg name, source) in async_main where the value
    passed is a `scanner.<name>` attribute access - the shape of the
    2026-09-03 incident above, generalized past the one call site
    (resume_open_trades) that got a guard for it. A second, structurally
    identical site - make_add_conversation's on_partial=scanner._on_partial_exit -
    sat unguarded until this existed: same "deploys clean because nothing in
    the suite calls async_main" failure mode, just a different function."""
    import ast
    import inspect
    import textwrap

    import notifier.main as main_module

    tree = ast.parse(textwrap.dedent(inspect.getsource(main_module.async_main)))
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = ast.unparse(node.func)
        for kw in node.keywords:
            if kw.arg is not None and ast.unparse(kw.value).startswith("scanner."):
                found.append((target, kw.arg, ast.unparse(kw.value)))
    return found


def test_every_scanner_callback_passed_in_async_main_is_a_real_attribute():
    """Generalizes test_every_resume_open_trades_kwarg_is_real past its one
    call site: ANY `scanner.<name>` passed as a kwarg anywhere in async_main
    is the same failure shape, not just the one to resume_open_trades."""
    from notifier.scanner import Scanner

    callbacks = _scanner_callback_kwargs_in_async_main()
    assert callbacks, "expected at least one scanner.<name> callback in async_main; this test is inert without one"
    for target, kwarg, rhs in callbacks:
        method_name = rhs.split(".", 1)[1]
        assert hasattr(Scanner, method_name), (
            f"{target}(..., {kwarg}={rhs}, ...) references Scanner.{method_name}, which doesn't exist"
        )
