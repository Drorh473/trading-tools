"""A permanent record of what was tested, what it found, and whether it
shipped - so the answer to "has this already been tried" is a file, not a
memory of who happened to remember MEMORY.md.

WHY THIS EXISTS
  The BTC trend gate was enabled, then reverted the same day the reading was
  redone properly: "year 2 separation does not reproduce (-0.002 vs +0.053
  claimed); the original arms came from two different runs." Nothing forced
  the two arms of that comparison to have come from the SAME run, because
  nothing recorded which run either arm came from at all - the numbers lived
  in a chat transcript and a docstring, not in anything checkable.

  Every other "tested, null" result in the project's own memory - divergence,
  HTF agreement, confirmed-rejection, the regime-straddle filter - has the
  same shape: a real measurement, made once, whose only trace is prose.
  Re-running any of them costs nothing but the discipline of remembering
  they were already tried.

EVERY ARM CARRIES drop_top3_expectancy, NOT AS AN OPTION.
  See feedback: an edge that dies without its three best trades is not an
  edge - that reading rule is worthless if it is optional to record, since
  the whole point is that a result can't be trusted without it. Recording an
  arm without it is refused outright rather than silently accepted with the
  field blank.

ONE FILE PER RUN, NEVER OVERWRITTEN.
  A second run of the same-named experiment is a second, independent
  measurement - conflating it with the first is exactly the failure the BTC
  gate postmortem describes. Filenames carry a microsecond timestamp plus a
  short random suffix so two runs seconds (or microseconds) apart can never
  collide onto the same file.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

from backtest.manifest import git_sha

REQUIRED_ARM_FIELDS = ("name", "n", "gross_expectancy", "net_expectancy", "drop_top3_expectancy")


def _validate_arms(arms: list[dict]) -> None:
    for arm in arms:
        missing = [f for f in REQUIRED_ARM_FIELDS if f not in arm]
        if missing:
            raise ValueError(
                f"arm {arm.get('name', '?')!r} is missing required field(s): {', '.join(missing)} "
                "- every arm must carry gross, net AND drop-top-3 expectancy, or the result "
                "cannot be read the way this project reads results (see feedback-sweep-past-the-optimum)"
            )


def record_experiment(
    name: str,
    *,
    hypothesis: str,
    universe: str,
    fit_period: str,
    confirm_period: str | None,
    arms: list[dict],
    verdict: str,
    notes: str = "",
    experiments_dir: str = "experiments",
) -> str:
    """Write one immutable record of one experiment run. Returns the path
    written.

    `verdict` is free text by design (e.g. "confirmed", "null", "reverted",
    "inconclusive") rather than an enum - see the divergence/HTF result,
    which needed "CONCLUSIVE... null" as a single distinguishable state a
    fixed set of choices would have forced apart or conflated.
    """
    _validate_arms(arms)
    os.makedirs(experiments_dir, exist_ok=True)

    now = datetime.now(timezone.utc)
    payload = {
        "name": name,
        "written_at": now.isoformat(),
        "git_sha": git_sha(),
        "hypothesis": hypothesis,
        "universe": universe,
        "fit_period": fit_period,
        "confirm_period": confirm_period,
        "arms": arms,
        "verdict": verdict,
        "notes": notes,
    }

    filename = f"{name}_{now.strftime('%Y%m%dT%H%M%S%f')}_{uuid.uuid4().hex[:8]}.json"
    path = os.path.join(experiments_dir, filename)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
    return path


def list_experiments(experiments_dir: str = "experiments") -> list[dict]:
    """Every recorded run under `experiments_dir`, oldest first. Empty (not
    an error) when the directory doesn't exist yet - a repo that has never
    recorded an experiment is a normal, valid state, not a broken one."""
    if not os.path.isdir(experiments_dir):
        return []
    runs = []
    for entry in sorted(os.listdir(experiments_dir)):
        if not entry.endswith(".json"):
            continue
        with open(os.path.join(experiments_dir, entry)) as fh:
            runs.append(json.load(fh))
    runs.sort(key=lambda r: r.get("written_at", ""))
    return runs
