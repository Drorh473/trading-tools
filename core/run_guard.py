"""A long-running script's outcome, reported the moment it happens - not
only as durable as the terminal session that started it.

WHY THIS EXISTS
  backtest/s1_overnight.py runs for hours with nobody attached, and
  s1_report.py exists specifically because "the run finishes in the middle
  of the night... a crashed or closed session loses the whole night." The
  weekly review job learned the same lesson the hard way: it "ran on
  schedule every Sunday from 2026-08-02 and died before sending, every time,
  on a Bitget 400 - and the only trace was a traceback appended to a log
  file nobody reads." That fix - alert before re-raising, log as the last
  resort, never let a broken alert path replace the real error - is
  generalized here rather than left to be hand-rolled by every long-running
  script that needs it.

WHY A CONTEXT MANAGER, NOT A DECORATOR
  The interesting content of a success message is usually only known AFTER
  the work runs (the sweep's best arm, the total signal count) - so the
  block gets a mutable `note` to fill in as it goes, and whatever's left on
  it when the block exits becomes the message. A decorator wrapping a
  function's return value would force every caller to thread that content
  back out through a return type instead of just setting an attribute where
  the work already is.
"""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable


@dataclass
class _Note:
    headline: str | None = None


def _default_alert(text: str) -> None:
    from config import settings
    from core.telegram_bot import send_message

    if settings.telegram_bot_token and settings.telegram_chat_id:
        send_message(settings.telegram_bot_token, settings.telegram_chat_id, text)


@contextmanager
def notify_on_completion(label: str, alert_fn: Callable[[str], None] | None = None):
    """Sends exactly one message when the wrapped block finishes, success or
    failure. `alert_fn` defaults to Telegram via config.settings; inject a
    fake for testing, or to redirect this somewhere else entirely.

    Yields a `note` whose `.headline` the block may set to anything worth
    saying about the run (a best arm, a total count) - included in the
    success message if set, omitted if not.
    """
    alert_fn = alert_fn or _default_alert
    note = _Note()
    t0 = time.time()
    try:
        yield note
    except Exception as exc:
        elapsed = time.time() - t0
        # Last frames only, same as weekly_review/main.py's own reasoning:
        # the point is to know it broke and roughly where, not to read a
        # full traceback on a phone.
        tail = "".join(traceback.format_exc().strip().splitlines(keepends=True)[-6:])
        try:
            alert_fn(f"{label} FAILED after {elapsed / 60:.1f}m: {type(exc).__name__}: {exc}\n\n{tail}")
        except Exception:
            # A broken alert path must not replace the original error with a
            # different one - the log is the last resort and it keeps both.
            traceback.print_exc()
        raise
    else:
        elapsed = time.time() - t0
        text = f"{label} DONE in {elapsed / 60:.1f}m"
        if note.headline:
            text += f"\n{note.headline}"
        try:
            alert_fn(text)
        except Exception:
            traceback.print_exc()
