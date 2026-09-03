"""Fill in the close INSTANT for rows that hold only a date, or nothing.

  python -m tools.backfill_closed_at            # DRY RUN - shows what it would write
  python -m tools.backfill_closed_at --write    # actually writes

DRY RUN BY DEFAULT, because this writes to the live journal and the matching
below is inference rather than a lookup. Read the proposed rows first.

WHY MATCHING IS NEEDED AT ALL

Bitget's position history has no idea which of our trade rows it corresponds
to - there is no shared id. So each unfilled trade is matched to a closed
position by symbol, direction and exit price, and each history row can be
claimed only once. Anything ambiguous is REPORTED AND SKIPPED rather than
guessed: a wrong close time is worse than a missing one, because a missing one
is visibly missing and a wrong one silently corrupts every excursion
measurement built on it later.

WHAT IT CANNOT REACH

Bitget keeps position history for a limited window, and this codebase should
not assert what that window is - run it and see. Whatever it cannot match is
listed at the end and stays NULL forever. That is the cost of the column not
having existed, and it is why it exists now.
"""

import sys
from datetime import datetime, timezone

from config import settings
from core.bitget_client import client_from_settings
from core.clock import LOCAL_TZ
from core.storage import Storage

# How close an exit price must be to call it the same trade. Prices are stored
# to the symbol's own precision on our side and averaged on Bitget's, so an
# exact equality test would reject nearly every real match; anything looser
# than this starts pairing genuinely different trades on the same symbol.
PRICE_TOLERANCE = 0.002  # 0.2%


def _matches(trade, row) -> bool:
    if (row["direction"] or "").lower() != (trade.כיוון or "").lower():
        return False
    ours, theirs = trade.מחיר_יציאה, row["exit_price"]
    if not ours or not theirs:
        return False
    return abs(ours - theirs) / theirs <= PRICE_TOLERANCE


def main() -> None:
    write = "--write" in sys.argv
    storage = Storage(settings.trades_db_path)
    bitget = client_from_settings(settings)

    needing = [t for t in storage.read_all() if t.is_closed and "T" not in (t.נסגר_בתאריך or "")]
    if not needing:
        print("Every closed trade already has a close time. Nothing to do.")
        return

    print(f"{len(needing)} closed trade(s) with no close time.\n")
    if not write:
        print("DRY RUN - nothing will be written. Re-run with --write to apply.\n")

    filled, ambiguous, unmatched = 0, [], []
    by_symbol: dict[str, list] = {}
    for trade in needing:
        by_symbol.setdefault(trade.סימבול, []).append(trade)

    for symbol, trades in sorted(by_symbol.items()):
        try:
            history = bitget.get_position_history(symbol, limit=100)
        except Exception as exc:
            print(f"{symbol}: could not read position history ({exc})")
            unmatched.extend(trades)
            continue

        claimed: set[int] = set()
        for trade in sorted(trades, key=lambda t: t.מספר_עסקה):
            candidates = [
                (i, row)
                for i, row in enumerate(history)
                if i not in claimed and _matches(trade, row)
            ]
            if not candidates:
                unmatched.append(trade)
                continue
            if len(candidates) > 1:
                # Two closed positions on the same symbol and side at the same
                # price. Nothing here can tell them apart, so neither is used.
                ambiguous.append((trade, len(candidates)))
                continue

            index, row = candidates[0]
            claimed.add(index)
            closed_at = datetime.fromtimestamp(
                row["close_time_ms"] / 1000, tz=timezone.utc
            ).astimezone(LOCAL_TZ)
            print(
                f"  #{trade.מספר_עסקה} {symbol} {trade.כיוון} @ {trade.מחיר_יציאה:g}"
                f"  ->  {closed_at.isoformat(timespec='seconds')}"
            )
            if write:
                storage.set_closed_at(
                    trade.מספר_עסקה, closed_at.isoformat(timespec="seconds")
                )
            filled += 1

    print()
    print(f"matched   {filled}")
    if ambiguous:
        print(f"ambiguous {len(ambiguous)} - skipped, more than one history row fits:")
        for trade, count in ambiguous:
            print(f"  #{trade.מספר_עסקה} {trade.סימבול} {trade.כיוון} ({count} candidates)")
    if unmatched:
        print(f"unmatched {len(unmatched)} - Bitget no longer has these:")
        for trade in unmatched:
            print(f"  #{trade.מספר_עסקה} {trade.סימבול} {trade.תאריך}")
    if filled and not write:
        print("\nDRY RUN - re-run with --write to apply the matches above.")


if __name__ == "__main__":
    main()
