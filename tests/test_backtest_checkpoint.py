"""Generating signals must survive being interrupted.

The first full run held every symbol in memory and wrote nothing until all 98
had finished. It took roughly fifteen hours of elapsed time - the machine slept
overnight partway through - and killing it at symbol 30 would have discarded
all of it. Nothing was lost, but only because nobody touched it.

A completed symbol is a finished, independent piece of work. These tests pin
the two properties that make resuming safe: partial work is kept, and it is
never reused for a different question.
"""

import pickle

from backtest import portfolio as pf


def test_a_checkpoint_from_the_same_run_is_resumed(tmp_path):
    path = str(tmp_path / "partial.pkl")
    pf._save_checkpoint(path, (98, 8760), {"AAAUSDT": ["signal"]})

    assert pf._load_checkpoint(path, (98, 8760)) == {"AAAUSDT": ["signal"]}


def test_a_checkpoint_from_a_different_scope_is_refused(tmp_path):
    """A 40-symbol checkpoint must not silently supply 40 of the 98 symbols a
    later run needs. The result would look complete and be wrong, which is
    worse than regenerating."""
    path = str(tmp_path / "partial.pkl")
    pf._save_checkpoint(path, (40, 8760), {"AAAUSDT": ["signal"]})

    assert pf._load_checkpoint(path, (98, 8760)) == {}


def test_a_missing_checkpoint_is_not_an_error(tmp_path):
    assert pf._load_checkpoint(str(tmp_path / "nothing.pkl"), (98, 8760)) == {}


def test_a_truncated_checkpoint_costs_time_not_correctness(tmp_path):
    """An interrupt during a write is exactly when this file is most likely to
    be damaged, so unreadable must mean "start over", never "crash"."""
    path = tmp_path / "partial.pkl"
    path.write_bytes(b"\x80\x04 truncated garbage")

    assert pf._load_checkpoint(str(path), (98, 8760)) == {}


def test_a_write_replaces_atomically_and_leaves_no_temp_file(tmp_path):
    """Written via a temp file and replaced, so an interrupt mid-write cannot
    leave a half-written checkpoint where a whole one used to be."""
    path = str(tmp_path / "partial.pkl")
    pf._save_checkpoint(path, (98, 8760), {"AAAUSDT": []})
    pf._save_checkpoint(path, (98, 8760), {"AAAUSDT": [], "BBBUSDT": []})

    assert set(pf._load_checkpoint(path, (98, 8760))) == {"AAAUSDT", "BBBUSDT"}
    assert [p.name for p in tmp_path.iterdir()] == ["partial.pkl"]


def test_the_checkpoint_holds_signals_only_not_bars(tmp_path):
    """Bars are already cached separately and are most of the 51MB. Writing
    them again every five symbols would make checkpointing cost more than the
    work it protects."""
    path = str(tmp_path / "partial.pkl")
    pf._save_checkpoint(path, (98, 8760), {"AAAUSDT": ["signal"]})

    key, payload = pickle.load(open(path, "rb"))

    assert key == (98, 8760)
    assert payload == {"AAAUSDT": ["signal"]}
