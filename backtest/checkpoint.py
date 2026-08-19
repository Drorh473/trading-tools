"""Writing a checkpoint must never be the thing that loses the work.

A checkpoint exists so that a killed run costs one symbol instead of all of
them. Three generators had written theirs the same way - dump to `.tmp`, then
os.replace onto the real path - and on Windows that replace raises

    PermissionError: [WinError 5] Access is denied

whenever anything else holds a handle on the DESTINATION: an indexer, a virus
scanner, or a shell that opened the file to check how far along the run was.
It is transient, and it is not an error about the data.

It took out a 27-symbol run at symbol 25, after 54 minutes, by raising out of
the worker loop. An hour of finished work was discarded because a bookkeeping
write failed - the exact outcome the checkpoint was there to prevent.

So the rule here: try, retry briefly, and if it still will not land, say so and
carry on. A missing checkpoint costs a re-scan of the last few symbols. A
raised one costs every symbol.
"""

from __future__ import annotations

import os
import pickle
import time

ATTEMPTS = 5


def write(path: str, payload) -> bool:
    """Atomically replace `path` with `payload`. True if it landed.

    The temporary file is left in place on failure: it is a complete, valid
    pickle, so a run that dies anyway can still be recovered by hand from it.
    """
    tmp = path + ".tmp"
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(tmp, "wb") as fh:
        pickle.dump(payload, fh)
    for attempt in range(ATTEMPTS):
        try:
            os.replace(tmp, path)
            return True
        except PermissionError:
            time.sleep(0.5 * (attempt + 1))
    print(f"  (could not write {path}; continuing without it)", flush=True)
    return False
