"""Check that get_funding_paid is reading the right bill rows, before the
monthly report starts subtracting its number from the balance.

  python -m tools.verify_funding            # last 30 days
  python -m tools.verify_funding 90         # last 90 days

STRICTLY READ-ONLY. It only reads the account bill.

WHY THIS EXISTS

The funding total is the one figure in the monthly reconciliation that was
written against documentation rather than against a real response. The bill
endpoint tags each row with a businessType, and get_funding_paid matches that
loosely, on the tokens "settle" and "funding". Two things can go wrong and
neither raises:

  matched nothing   Bitget spells the enum differently and every month
                    reports exactly $0.00 funding. Indistinguishable from an
                    account that never held a position overnight.
  matched too much  the token also catches some unrelated settlement row, and
                    trading P&L gets reported as a funding cost.

So this prints EVERY distinct businessType in the window with its row count
and summed amount, and marks which ones the matcher took. The check is not
"does the total look plausible" - it is "are those the right rows, and is the
total the same as Bitget's own bill page for the same window".

Until that has been done once against a real month, treat the funding line in
the monthly report as unverified.
"""

import sys
import time
from collections import defaultdict

from config import settings
from core.bitget_client import BitgetClient, client_from_settings


def _all_bill_rows(client: BitgetClient, start_ms: int, end_ms: int):
    """Every bill row in the window, unfiltered - deliberately NOT reusing
    _funding_rows, since the thing under test is which rows that one keeps.
    """
    id_less_than = None
    for _ in range(50):
        params = {
            "productType": client.account_product_type,
            "limit": "100",
            "startTime": str(start_ms),
            "endTime": str(end_ms),
        }
        if id_less_than:
            params["idLessThan"] = id_less_than
        data = client._request("GET", "/api/v2/mix/account/bill", params=params, signed=True)
        rows = (data.get("bills") or data.get("billList") or []) if isinstance(data, dict) else data
        if not rows:
            return
        yield from rows
        cursor = rows[-1].get("billId") or rows[-1].get("id")
        if not cursor or len(rows) < 100:
            return
        id_less_than = str(cursor)


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000

    client = client_from_settings(settings)
    rows = list(_all_bill_rows(client, start_ms, end_ms))

    print(f"{len(rows)} bill rows over the last {days} days\n")
    if not rows:
        print("Nothing to check. Either the account was idle, or the window is wrong.")
        return

    counts: dict[str, int] = defaultdict(int)
    sums: dict[str, float] = defaultdict(float)
    for row in rows:
        business = str(row.get("businessType") or "(missing)")
        counts[business] += 1
        sums[business] += float(row.get("amount") or 0.0)

    print(f"{'businessType':<28} {'rows':>5} {'sum amount':>12}   matched?")
    print("-" * 62)
    for business in sorted(counts, key=lambda b: -counts[b]):
        matched = any(t in business.lower() for t in BitgetClient._FUNDING_TOKENS)
        mark = "TAKEN as funding" if matched else ""
        print(f"{business:<28} {counts[business]:>5} {sums[business]:>12.4f}   {mark}")

    total = client.get_funding_paid(start_ms, end_ms)
    print()
    print(f"get_funding_paid() = {total:+.4f} USDT   (positive = cost to the account)")
    print()
    print("CHECK: does that match Bitget's own bill page, filtered to funding,")
    print(f"for the same {days}-day window? If the table above marked nothing,")
    print("the businessType enum has a name _FUNDING_TOKENS does not cover.")


if __name__ == "__main__":
    main()
