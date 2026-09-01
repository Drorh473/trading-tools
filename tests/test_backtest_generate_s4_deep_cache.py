"""generate_s4_deep.py already resumes per-symbol from its own output file,
but had no way to notice when the strategy itself changed - it would trust a
stale per-symbol entry forever, silently describing a rule nobody runs any
more. _resume_signals is the fix: reuse the existing output only when it
still reflects today's strategy; otherwise discard it and rescan everything.
"""

from backtest import generate_s4_deep as gs4
from backtest import instance_cache as ic


def test_current_hash_is_deterministic():
    assert gs4._current_hash() == gs4._current_hash()


def test_resume_signals_is_empty_when_no_output_exists_yet(tmp_path):
    out = str(tmp_path / "s4_signals_deep.pkl")

    assert gs4._resume_signals(out, ["AAAUSDT"], "somehash") == {}


def test_resume_signals_reuses_the_output_when_the_hash_still_matches(tmp_path):
    import pickle

    out = tmp_path / "s4_signals_deep.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"AAAUSDT": ["signal"]}, fh)
    ic.write_sidecar_hash(str(out), "current")

    result = gs4._resume_signals(str(out), ["AAAUSDT"], "current")

    assert result == {"AAAUSDT": ["signal"]}


def test_resume_signals_discards_the_output_when_the_strategy_changed(tmp_path):
    """The bug this fixes: without this check, a stale per-symbol entry from
    before a rule edit would be trusted as 'already done' forever."""
    import pickle

    out = tmp_path / "s4_signals_deep.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"AAAUSDT": ["stale signal from the old rule"]}, fh)
    ic.write_sidecar_hash(str(out), "old_hash")

    result = gs4._resume_signals(str(out), ["AAAUSDT"], "new_hash")

    assert result == {}, "a stale entry must be discarded, not reused, when the strategy has moved on"


def test_resume_signals_drops_symbols_no_longer_in_the_requested_universe(tmp_path):
    import pickle

    out = tmp_path / "s4_signals_deep.pkl"
    with open(out, "wb") as fh:
        pickle.dump({"AAAUSDT": ["signal"], "ZZZUSDT": ["signal"]}, fh)
    ic.write_sidecar_hash(str(out), "current")

    result = gs4._resume_signals(str(out), ["AAAUSDT"], "current")

    assert result == {"AAAUSDT": ["signal"]}
