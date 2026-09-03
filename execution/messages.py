"""Renders a Trade's state into the Telegram text Dror actually reads: a
close report, a scale-in notice, a partial-fill report.

Split out of execution/tracker.py, which mixed this with position-polling
logic proper. These take already-known data (a Trade row, numbers already
read from Bitget) and return a string - no network calls, no bitget
client, no storage. tracker.py imports nothing from here; this module
imports one thing back (breakeven_price) because it's trade-state logic
several exit-placement callers ALSO depend on, not a formatting concern
that happens to live in the wrong file.
"""

from __future__ import annotations

from core.storage import Trade
from execution.tracker import breakeven_price


def _qty(value: float | None) -> str:
    """A size without six trailing zeros. "Closed 33.000000 of 66.000000"
    was the old line; sizes span 0.07 contracts and 3267, so this trims
    rather than fixing the decimals."""
    if value is None:
        return "n/a"
    return f"{value:.6f}".rstrip("0").rstrip(".") or "0"


def _px(value: float | None) -> str:
    """Price at a readable precision without needing the symbol's specs.

    Significant figures rather than fixed decimals: the watchlist spans
    PEPEUSDT at 0.0000028 and SNDKUSDT at 1428, and the fixed .2f the older
    formatters use prints the former as "0.00".
    """
    return "n/a" if value is None else f"{value:.6g}"


def format_close_message(trade: Trade, exits: list[dict] | None = None) -> str:
    """The close report.

    Prices print at _px precision rather than .2f, which rendered APTUSDT
    #11's 0.608113 entry as "0.61" and its 0.5485 exit as "0.55" - the same
    fault the partial message had.

    `exits` breaks out what each close went off at. Bitget's closeAvgPrice is
    one size-weighted average across every close, so a trade that took half
    off at 0.5608 and ran the rest to 0.5360 reports 0.5485, a price nothing
    traded at and the one Dror flagged as simply wrong. With a single exit
    the average IS the fill, so the breakdown is only worth printing when
    there was more than one.
    """
    pnl = f"{trade.רווח_הפסד:.4f}" if trade.רווח_הפסד is not None else "n/a"
    r = f"{trade.מכפיל_R:.2f}R" if trade.מכפיל_R is not None else "n/a (no stop was set)"
    lines = [
        f"Trade #{trade.מספר_עסקה} closed: {trade.סימבול} {trade.כיוון}",
        f"Entry: {_px(trade.מחיר_כניסה)}  Exit: {_px(trade.מחיר_יציאה)}"
        + ("  (avg of the closes below)" if exits and len(exits) > 1 else ""),
    ]
    if exits and len(exits) > 1:
        for exit_ in exits:
            lines.append(f"  {exit_['size']:g} @ {_px(exit_['price'])}")
    lines.append(f"P&L (after fees): {pnl}   R: {r}")
    if trade.changed_from_plan:
        lines.append("(stop/target differed from the original plan)")
    return "\n".join(lines)


def format_scale_in_message(trade: Trade, covered: float | None) -> str:
    """A resting entry leg filled, so the real position just arrived.

    Deliberately states the new position only, with no before/after deltas -
    Dror's standing preference, the same call he made for the weekly report's
    real-trades section.
    """
    size = trade.גודל_פוזיציה or 0.0
    risk = f"${trade.סכום_סיכון:.2f}" if trade.סכום_סיכון is not None else "n/a"
    lines = [
        f"Trade #{trade.מספר_עסקה} ({trade.סימבול} {trade.כיוון}): limit leg filled.",
        f"Now {size:g} @ {_px(trade.מחיר_כניסה)} — risk {risk}, stop {_px(trade.סטופ_לוס_בפועל)}.",
        f"Breakeven is now {_px(trade.מחיר_כניסה)}.",
    ]
    if covered is not None:
        lines.append(f"Take-profit covers {covered:g} of {size:g}.")
    return "\n".join(lines)


def format_partial_message(
    trade: Trade,
    closed_size: float,
    realized_pnl: float | None,
    steps: list[str] | None = None,
) -> str:
    """Reports the scale-out, and what is actually going to happen next.

    The line this used to end on - "stop should already be at entry (0.61)" -
    was an unconditional f-string. It printed on every partial including the
    two paths that never had a breakeven handler attached at all (a tracker
    re-attached by resume_open_trades after a restart, and a trade added with
    /add), so it read as confirmation of a move that nothing had made. The
    APTUSDT short of 2026-08-13 took its partial and rode the rest with its
    original stop for exactly that reason.

    Now the wording follows the trade's recorded exit plan: if the bot owns
    this trade's exits it says what it is about to do and the handler doing it
    confirms separately, and if it doesn't, it says so. The price is printed
    at _px precision rather than .2f, which on a 0.61 short was rounding the
    breakeven by up to 0.005 - 0.8%, on a stop.
    """
    total = trade.גודל_פוזיציה or 0
    pct = (closed_size / total * 100) if total else 0
    pnl = f"{realized_pnl:+.2f}" if realized_pnl is not None else "n/a"
    tag = f" ({trade.תגית_אסטרטגיה})" if trade.תגית_אסטרטגיה else ""
    lines = [
        f"{trade.סימבול} {trade.כיוון} · partial filled{tag}",
        f"Closed {_qty(closed_size)} of {_qty(total)} ({pct:.0f}%) · realised {pnl}",
    ]
    breakeven = breakeven_price(trade)
    if steps:
        # The caller already did the work and is reporting outcomes, so this is
        # the ONE message for the whole event rather than an announcement
        # followed by two confirmations.
        lines.append(" · ".join(steps))
    elif breakeven is not None:
        lines.append(f"Stop → {_px(breakeven)} breakeven, runner next.")
    else:
        lines.append(
            f"Bot does NOT manage this trade — move the stop to entry "
            f"({_px(trade.מחיר_כניסה)}) by hand."
        )
    return "\n".join(lines)
