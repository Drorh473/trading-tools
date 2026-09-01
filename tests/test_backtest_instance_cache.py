"""instance_hash identifies a strategy instance's BEHAVIOR: its own module's
source plus its constructor state. Two instances built from identical code
and identical params must be indistinguishable to the cache, or every
re-generation would look "stale" for no reason and the cache would never pay
for itself.
"""

from backtest.instance_cache import instance_hash, is_output_fresh, load_store, save_store, write_sidecar_hash


class _Fake:
    def __init__(self, timeframe="1H"):
        self.timeframe = timeframe


def test_two_instances_with_identical_class_and_params_hash_the_same():
    a = _Fake("1H")
    b = _Fake("1H")

    assert instance_hash(a, shared_files=()) == instance_hash(b, shared_files=())


def test_different_params_on_the_same_class_hash_differently():
    a = _Fake("1H")
    b = _Fake("4H")

    assert instance_hash(a, shared_files=()) != instance_hash(b, shared_files=())


def test_different_classes_with_identical_state_hash_differently():
    """Two unrelated strategies that happen to store the same attribute must
    not collide just because their __dict__ looks the same."""

    class _OtherFake:
        def __init__(self, timeframe="1H"):
            self.timeframe = timeframe

    assert instance_hash(_Fake("1H"), shared_files=()) != instance_hash(_OtherFake("1H"), shared_files=())


def test_editing_the_instances_own_source_file_changes_the_hash(tmp_path):
    """The whole point: a strategy-rule edit must invalidate its own cache
    entry without anyone bumping a version number."""
    import importlib.util

    path = tmp_path / "fake_strategy.py"
    path.write_text("class Fake:\n    def __init__(self):\n        self.x = 1\n")
    before = instance_hash(_load(path).Fake(), shared_files=())

    path.write_text("class Fake:\n    def __init__(self):\n        self.x = 1\n        self.y = 2  # new rule\n")
    after = instance_hash(_load(path).Fake(), shared_files=())

    assert before != after


def test_editing_a_shared_file_changes_the_hash_even_though_the_instance_is_unchanged(tmp_path):
    """A fix to a shared indicator (ATR, EMA, a swing detector) must
    invalidate every strategy that relies on it, not just the one someone
    remembered to touch."""
    shared = tmp_path / "shared_indicator.py"
    shared.write_text("def atr(): return 1\n")
    before = instance_hash(_Fake("1H"), shared_files=(str(shared),))

    shared.write_text("def atr(): return 2  # threshold changed\n")
    after = instance_hash(_Fake("1H"), shared_files=(str(shared),))

    assert before != after


def test_editing_a_file_not_in_shared_files_leaves_the_hash_unchanged(tmp_path):
    """Only the files actually named as shared are folded in - an edit
    anywhere else in the repo must not force every strategy to rescan."""
    tracked = tmp_path / "tracked.py"
    tracked.write_text("A = 1\n")
    untracked = tmp_path / "untracked.py"
    untracked.write_text("B = 1\n")

    before = instance_hash(_Fake("1H"), shared_files=(str(tracked),))
    untracked.write_text("B = 999\n")
    after = instance_hash(_Fake("1H"), shared_files=(str(tracked),))

    assert before == after


def _load(path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("fake_strategy_module", str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # inspect.getfile resolves a class via sys.modules
    spec.loader.exec_module(module)
    return module


def test_real_strategy_instances_with_the_same_config_hash_the_same():
    """Integration check against a real strategy, not just the fixture."""
    from notifier.strategies.rsi_fib_reversal import RsiFibReversal

    assert instance_hash(RsiFibReversal("1H")) == instance_hash(RsiFibReversal("1H"))


def test_real_strategy_instances_with_different_timeframes_hash_differently():
    from notifier.strategies.rsi_fib_reversal import RsiFibReversal

    assert instance_hash(RsiFibReversal("1H")) != instance_hash(RsiFibReversal("4H"))


# ---------------------------------------------------------------------------
# The per-(symbol, instance_hash) store.
# ---------------------------------------------------------------------------

def test_a_saved_store_round_trips(tmp_path):
    path = str(tmp_path / "instance_signals.pkl")
    store = {("AAAUSDT", "h1"): [("row",)], ("AAAUSDT", "h2"): []}

    save_store(path, store)

    assert load_store(path) == store


def test_loading_a_missing_store_is_not_an_error(tmp_path):
    assert load_store(str(tmp_path / "nothing.pkl")) == {}


def test_a_truncated_store_costs_time_not_correctness(tmp_path):
    path = tmp_path / "instance_signals.pkl"
    path.write_bytes(b"\x80\x04 truncated garbage")

    assert load_store(str(path)) == {}


def test_saving_replaces_atomically_and_leaves_no_temp_file(tmp_path):
    path = str(tmp_path / "instance_signals.pkl")
    save_store(path, {("AAAUSDT", "h1"): []})
    save_store(path, {("AAAUSDT", "h1"): [], ("BBBUSDT", "h1"): []})

    assert set(load_store(path)) == {("AAAUSDT", "h1"), ("BBBUSDT", "h1")}
    assert [p.name for p in tmp_path.iterdir()] == ["instance_signals.pkl"]


# ---------------------------------------------------------------------------
# The sidecar hash: whole-script staleness for a single-strategy generator
# (generate_v2.py, generate_s4_deep.py) that has no per-instance granularity
# of its own - "did THIS script's own instance change at all" is enough to
# decide "reuse the existing output" vs "regenerate everything".
# ---------------------------------------------------------------------------

def test_output_is_not_fresh_when_the_file_does_not_exist_yet(tmp_path):
    out = str(tmp_path / "signals_v2.pkl")

    assert is_output_fresh(out, "somehash") is False


def test_output_is_not_fresh_when_no_hash_was_ever_recorded(tmp_path):
    out = tmp_path / "signals_v2.pkl"
    out.write_bytes(b"some pickled output")

    assert is_output_fresh(str(out), "somehash") is False


def test_output_is_fresh_when_the_recorded_hash_matches(tmp_path):
    out = tmp_path / "signals_v2.pkl"
    out.write_bytes(b"some pickled output")
    write_sidecar_hash(str(out), "somehash")

    assert is_output_fresh(str(out), "somehash") is True


def test_output_is_not_fresh_when_the_recorded_hash_differs(tmp_path):
    out = tmp_path / "signals_v2.pkl"
    out.write_bytes(b"some pickled output")
    write_sidecar_hash(str(out), "old_hash")

    assert is_output_fresh(str(out), "new_hash") is False
