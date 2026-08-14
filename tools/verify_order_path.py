"""Check what the exchange ACTUALLY holds after a trade, against what was meant.

  python -m tools.verify_order_path                  # every open position
  python -m tools.verify_order_path AAPLUSDT long    # one, in detail

STRICTLY READ-ONLY. It calls nothing that places, amends or cancels an order,
and it never will - the whole point is to be safe to run at any moment,
including while a real position is live.

WHY THIS EXISTS

The partial take-profit was rejected on every attempt for five months and
nobody knew, because the only evidence anyone looked at was whether the call
raised. It did not raise in a way that reached Dror, and each individual
failure looked transient. §33: "A 100% failure rate is not a race."

The diagnosis that replaced it - that hedge mode reads the side/tradeSide
pairing used for a close as the OPPOSITE side - is recorded in the handoff as
"Not confirmed by experiment, because confirming it means firing a real order
at a real position." The fix is therefore believed rather than proven, and has
been shipped to a live account on that basis.

This turns the next small real trade into the experiment. Placing it is Dror's
hand on the button; this reads back what happened and says which invariants
held, so the answer is a checklist rather than an impression.

WHAT EACH CHECK IS FOR

  side        a wrong side/tradeSide pairing opens the OPPOSITE position
              instead of erroring - the failure mode §6 listed first
  size        base coins vs contracts is a factor-of-100 error
  stop        must exist the moment the position does, and be on the losing
              side of entry; a stop on the wrong side is not protection
  target      the capability that never once worked
  raw fields  RWA take-profit plan-order field names have been carried as
              unverified across five handoffs. Nothing here can confirm them
              by reasoning - so the raw plan-order payload is printed, and a
              real RWA fill (AAPLUSDT, GOOGLUSDT, NVDAUSDT...) is the only
              thing that settles it. Pick one deliberately and the item closes.
"""

import sys

from config import settings
from core.bitget_client import client_from_settings

PASS, FAIL, INFO = "PASS", "FAIL", "  ? "


def _line(status: str, text: str) -> None:
    print(f"  [{status}] {text}")


def _check_side(position: dict, direction: str) -> bool:
    """The one that catches an inverted tradeSide pairing.

    A position on the opposite side to the signal is not a near miss - it is
    the bot having opened a trade against its own analysis, silently.
    """
    held = (position.get("holdSide") or position.get("direction") or "").lower()
    if held == direction:
        _line(PASS, f"position side is {held}, as intended")
        return True
    _line(FAIL, f"position side is {held!r} but the signal was {direction!r} "
                f"- a side/tradeSide pairing this wrong opens the OPPOSITE trade")
    return False


# A stop this close to entry IS the breakeven stop, not a misplaced one. The
# bot moves stops to entry by design once the partial fills, and entry is a
# blended average that almost never lands on a tick boundary - so the rounded
# stop sits a hair either side of it. Calling that a failure would report a
# working breakeven as a bug on every trade that reached its target, which is
# how a check stops being read.
BREAKEVEN_TOLERANCE = 0.001  # 0.1% of entry


def _check_stop(bitget, symbol: str, direction: str, entry: float) -> bool:
    stop, target = bitget.get_stop_target(symbol, direction)

    ok = True
    if stop is None:
        _line(FAIL, "no stop on the exchange - the position is unprotected")
        ok = False
    elif entry and abs(stop - entry) / entry <= BREAKEVEN_TOLERANCE:
        _line(PASS, f"stop at {stop:g} is at breakeven (entry {entry:g}) "
                    f"- the partial has filled and the remainder is risk-free")
    else:
        # A long is stopped BELOW entry and a short ABOVE. A stop meaningfully
        # on the wrong side would trigger instantly or never, and either way is
        # not protection.
        right_side = stop < entry if direction == "long" else stop > entry
        if right_side:
            _line(PASS, f"stop at {stop:g}, on the losing side of entry {entry:g}")
        else:
            _line(FAIL, f"stop at {stop:g} is on the WRONG side of entry {entry:g} "
                        f"- that is not protection")
            ok = False

    if target is None:
        _line(FAIL, "no take-profit - this is the capability that never once "
                    "worked from 2026-08-03 to 2026-08-13")
        ok = False
    else:
        right_side = target > entry if direction == "long" else target < entry
        _line(PASS if right_side else FAIL,
              f"take-profit at {target:g}" + ("" if right_side else " - on the WRONG side of entry"))
        ok = ok and right_side
    return ok


def _show_plan_orders(bitget, symbol: str, direction: str, is_rwa: bool) -> None:
    """Print the raw plan orders, field names and all.

    This is the only part that is not a pass/fail: the RWA take-profit field
    names cannot be confirmed by reasoning, only by looking at one that a real
    fill produced. If this position is on an RWA symbol, what prints here is
    the answer to an item that has been open since 2026-08-06.
    """
    try:
        orders = bitget.get_plan_orders(symbol, direction)
    except Exception as exc:
        _line(FAIL, f"could not read plan orders: {exc}")
        return

    label = "RWA" if is_rwa else "crypto"
    print(f"\n  raw plan orders on {symbol} ({label}) - {len(orders)} found:")
    if not orders:
        print("    (none - if a target was expected, it is not on the book)")
    for order in orders:
        print(f"    {order}")
    if is_rwa and orders:
        print("    ^ THIS is the RWA take-profit shape that has been unverified"
              "\n      across five handoffs. Compare against place_tpsl_order.")


def verify(bitget, symbol: str, direction: str) -> bool:
    print(f"\n{symbol} {direction}")
    position = bitget.get_position(symbol, direction)
    if position is None:
        _line(FAIL, "no position found - nothing filled, or it filled on the other side")
        # Deliberately keep going: an opposite-side position is the single most
        # important thing this script can find, and returning here would hide it.
        for other in bitget.get_positions(symbol):
            _line(INFO, f"but the account holds: {other}")
        return False

    entry = float(position.get("entry_price") or position.get("openPriceAvg") or 0.0)
    size = float(position.get("size") or position.get("total") or 0.0)

    ok = _check_side(position, direction)
    _line(INFO if size else FAIL, f"size {size:g} base units at entry {entry:g}")
    ok = _check_stop(bitget, symbol, direction, entry) and ok

    try:
        is_rwa = bool(bitget.get_contract_specs(symbol).get("is_rwa"))
    except Exception:
        is_rwa = False
    _show_plan_orders(bitget, symbol, direction, is_rwa)
    return ok


def main() -> int:
    bitget = client_from_settings(settings)

    if len(sys.argv) >= 3:
        ok = verify(bitget, sys.argv[1].upper(), sys.argv[2].lower())
        return 0 if ok else 1

    positions = bitget.get_all_positions()
    if not positions:
        print("No open positions. Approve a small trade, then run this again with "
              "the symbol and direction to check it.")
        return 0

    print(f"{len(positions)} open position(s):")
    ok = True
    for position in positions:
        # get_all_positions normalises to "direction"; holdSide only survives
        # inside ["raw"], which is where the un-normalised exchange payload is.
        symbol = position.get("symbol")
        direction = (position.get("direction") or "").lower()
        if symbol and direction:
            ok = verify(bitget, symbol, direction) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
