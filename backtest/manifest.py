"""Provenance for a generated dataset, written beside the file it describes.

WHY THIS EXISTS
  data/ holds ~60 generated pickles and nothing records what produced any of
  them. That has already cost a real mistake: signals_v2_ship.pkl turned out
  to bundle 3 strategy instances, and a wrong number shipped on 2026-08-21
  because nothing visible said so - the instance count was there, but only
  inside the pickle itself, unreadable without writing code to unpickle it.

  A manifest is a plain-text sidecar (foo.pkl.manifest.json next to foo.pkl)
  written at generation time - human-readable with `cat`, diffable, and
  readable even by someone who has no interest in unpickling a numpy/pandas
  blob just to answer "what is this file".

WHY A SEPARATE MECHANISM FROM instance_cache's .hash SIDECAR
  The .hash sidecar answers one question - "is this output stale" - and is
  deliberately as small as an answer to that question needs to be (see its
  own module docstring). A manifest answers a different one - "what IS this
  file" - and grows with whatever a generator finds worth recording:
  instance count, universe, row count, wall time. Bundling the two would
  make every manifest field part of the freshness check, which is not what
  either was built for.

CALLING CONVENTION
  write_manifest(out_path, **fields) always records `git_sha` and
  `written_at` itself; the caller supplies whatever ELSE is worth knowing
  about this particular file - there is no fixed schema, because a signal
  cache and a bars cache have nothing in common to standardize on beyond
  those two.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone


def git_sha(cwd: str | None = None) -> str | None:
    """The repo's current commit, or None if there isn't one to read (not a
    git checkout, git unavailable) - a manifest is a diagnostic nicety, and
    a generation run must never fail because of it."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _manifest_path(out_path: str) -> str:
    return out_path + ".manifest.json"


def write_manifest(out_path: str, **fields) -> None:
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        **fields,
    }
    with open(_manifest_path(out_path), "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)


def read_manifest(out_path: str) -> dict | None:
    try:
        with open(_manifest_path(out_path)) as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
