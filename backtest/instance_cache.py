"""Stable identity for a strategy instance, so a signal cache can tell "this
strategy's rule changed" from "this strategy is unchanged" without a human
remembering to bump a version number.

A strategy instance is __init__ configuration only (see portfolio.py's own
docstring: "no mutable state - evaluate() is a pure function of bars"), so
its behaviour is fully determined by two things: the code in its own module,
and whatever it was constructed with. Hash both and a rule edit changes the
hash automatically; an unrelated strategy, untouched.

Shared modules (base.py's FillGuard/Signal, indicators.py's ATR/EMA/swing
math, risk_sizing.py's position planning) are folded into EVERY instance's
hash unconditionally rather than traced per-strategy import graph - simpler,
and correct in the direction that matters: it can only over-invalidate
(an edit to indicators.py rescans every strategy, even ones that don't use
the function that changed), never under-invalidate.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import pickle

from backtest import checkpoint


def _shared_module_files() -> tuple[str, ...]:
    from notifier import risk_sizing
    from notifier.strategies import base, indicators

    return tuple(inspect.getfile(m) for m in (base, indicators, risk_sizing))


def _read(path: str) -> bytes:
    with open(path, "rb") as fh:
        return fh.read()


def instance_hash(instance, shared_files: tuple[str, ...] | None = None) -> str:
    """A short, stable digest of `instance`'s class source, the shared
    strategy modules, and its constructed state (every attribute __init__
    set on it - the constructor params themselves plus anything derived
    from them, which changes together with them anyway).

    `shared_files` is overridable for testing; production callers should
    leave it at the default (base.py, indicators.py, risk_sizing.py).
    """
    if shared_files is None:
        shared_files = _shared_module_files()

    own_source = _read(inspect.getfile(type(instance)))
    shared_source = b"\0".join(_read(p) for p in shared_files)
    state = repr(sorted(instance.__dict__.items(), key=lambda kv: kv[0])).encode()

    payload = b"\0".join((type(instance).__qualname__.encode(), own_source, shared_source, state))
    return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# The store: {(symbol, instance_hash): [signal rows]}, one entry per strategy
# instance per symbol - so a rule edit only evicts the (symbol, hash) pairs
# whose hash no longer matches, not the whole cache.
# ---------------------------------------------------------------------------


def load_store(path: str) -> dict:
    """Same resilience as checkpoint._load_checkpoint: a missing or truncated
    store costs a re-scan, never a crash - an interrupted write is exactly
    when this file is most likely to be damaged."""
    try:
        with open(path, "rb") as fh:
            return pickle.load(fh)
    except Exception:
        return {}


def save_store(path: str, store: dict) -> None:
    """Atomic replace, via the same primitive the signal checkpoint uses -
    an interrupt mid-write must never leave a half-written store where a
    whole one used to be."""
    checkpoint.write(path, store)


# ---------------------------------------------------------------------------
# Whole-output staleness for a single-strategy generator (generate_v2.py,
# generate_s4_deep.py) that has no per-symbol/per-instance structure of its
# own to key a store by: "did the instance(s) this script covers change at
# all" is the only question such a script can answer cheaply, so a match
# means reuse the WHOLE existing output, and a miss means a full regenerate
# exactly as today - no incremental rescan within the script.
#
# Recorded in a sidecar file next to the output rather than inside its
# pickle, so this can be bolted onto an existing output format without
# changing the tuple shape every downstream reader unpacks positionally.
# ---------------------------------------------------------------------------


def _sidecar_path(out_path: str) -> str:
    return out_path + ".hash"


def write_sidecar_hash(out_path: str, value: str) -> None:
    with open(_sidecar_path(out_path), "w") as fh:
        fh.write(value)


def _read_sidecar_hash(out_path: str) -> str | None:
    try:
        with open(_sidecar_path(out_path)) as fh:
            return fh.read().strip()
    except OSError:
        return None


def is_output_fresh(out_path: str, current_hash: str) -> bool:
    """Whether `out_path` already reflects `current_hash` - i.e. the output
    file exists, a hash was recorded for it, and that hash still matches."""
    if not os.path.exists(out_path):
        return False
    return _read_sidecar_hash(out_path) == current_hash
