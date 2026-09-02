"""generate_s4_deep.py only ever scans ONE strategy, so a genuinely new rule
still costs a full rescan of all 734 symbols - there is no shortcut around
evaluating new logic against real bars. What per-(symbol, hash) caching buys
is that SWITCHING BACK to a rule version already generated before becomes
instant, instead of the old whole-output-hash design, which discarded every
symbol's cached signals the moment the hash changed - even if that hash had
been seen (and fully scanned) before.
"""

from backtest import generate_s4_deep as gs4


def test_current_hash_is_deterministic():
    assert gs4._current_hash() == gs4._current_hash()


def test_plan_scans_every_symbol_when_the_store_is_empty():
    signals, todo = gs4._plan(["AAAUSDT", "BBBUSDT"], store={}, current_hash="h1")

    assert signals == {}
    assert todo == ["AAAUSDT", "BBBUSDT"]


def test_plan_reuses_symbols_already_cached_under_the_current_hash():
    store = {("AAAUSDT", "h1"): ["signal"], ("BBBUSDT", "h1"): []}

    signals, todo = gs4._plan(["AAAUSDT", "BBBUSDT"], store, current_hash="h1")

    assert signals == {"AAAUSDT": ["signal"], "BBBUSDT": []}
    assert todo == []


def test_plan_rescans_a_symbol_cached_only_under_a_different_hash():
    """The bug the old design had: a stale entry under an OLD hash must
    never be silently treated as this run's answer."""
    store = {("AAAUSDT", "old_hash"): ["stale signal"]}

    signals, todo = gs4._plan(["AAAUSDT"], store, current_hash="new_hash")

    assert signals == {}
    assert todo == ["AAAUSDT"]


def test_plan_never_evicts_entries_for_other_hashes():
    """The whole point: an old rule version's results stay in the store,
    unused but intact, so reverting to it later needs no rescan."""
    store = {("AAAUSDT", "old_hash"): ["old signal"]}

    gs4._plan(["AAAUSDT"], store, current_hash="new_hash")

    assert store == {("AAAUSDT", "old_hash"): ["old signal"]}, "must not mutate the store"


def test_switching_back_to_a_previously_seen_hash_needs_no_rescan():
    """End to end: hash A generates and populates the store, hash B (a rule
    edit) generates fresh entries alongside it, then reverting to hash A
    finds its old results still there - nothing to scan."""
    store = {}
    signals_a, todo_a = gs4._plan(["AAAUSDT", "BBBUSDT"], store, current_hash="A")
    assert todo_a == ["AAAUSDT", "BBBUSDT"]
    for s in todo_a:
        store[(s, "A")] = [f"{s}-signal-under-A"]

    signals_b, todo_b = gs4._plan(["AAAUSDT", "BBBUSDT"], store, current_hash="B")
    assert todo_b == ["AAAUSDT", "BBBUSDT"], "a genuinely new rule still needs a real scan"
    for s in todo_b:
        store[(s, "B")] = [f"{s}-signal-under-B"]

    signals_a_again, todo_a_again = gs4._plan(["AAAUSDT", "BBBUSDT"], store, current_hash="A")

    assert todo_a_again == [], "reverting to A must not rescan anything"
    assert signals_a_again == {"AAAUSDT": ["AAAUSDT-signal-under-A"], "BBBUSDT": ["BBBUSDT-signal-under-A"]}
