"""Thin wrapper over Bitget's v2 REST API, targeting USDT-margined futures
(productType=USDT-FUTURES) — the product the user actually trades.

Public market data (candles, ticker) needs no credentials. Authenticated
endpoints use Bitget's documented signing scheme:
HMAC-SHA256(timestamp + METHOD + requestPath[+"?"+query] + body, secret),
base64-encoded, sent via the ACCESS-* headers.

FIELD NAMES VERIFIED LIVE (2026-07-30) against a real futures position and a
real closed position — not guessed:

  open position  : symbol, holdSide, openPriceAvg, total, leverage,
                   unrealizedPL, achievedProfits, posMode
  closed position: openAvgPrice, closeAvgPrice, pnl (gross), netProfit
                   (after fees — what we record), ctime (open), utime (close)

Two things that are easy to get wrong and cost us real bugs:

1. Candles come back OLDEST-FIRST already. Do not reverse them. The final
   candle is always the still-forming one, which `closed_only` drops.
2. A position's `stopLoss`/`takeProfit` fields are only populated when the
   stop/target were attached as presets at entry. If they were placed as
   separate TP/SL orders (the user's actual habit) those fields stay EMPTY
   and the real values live on the plan-orders endpoint — see
   get_stop_target(), which checks both.

The account runs in hedge_mode, so a symbol can hold a long and a short at
once; every position lookup filters on holdSide rather than taking data[0].
"""

import base64
import hashlib
import hmac
import json
import math
import time
from urllib.parse import urlencode

import requests

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"
# Bitget rejects a candle request outright past this: 40053, "limit should be
# between (0, 1000]". It has to be enforced here rather than trusted to
# callers, because get_candles adds one to the requested count when it is
# dropping the forming candle - so a caller asking for a legal 1000 produced an
# illegal 1001 and a 400. That is what silently killed the weekly report: it
# asks for 1000 closed 15m bars, and the report has never once run since it
# shipped on 2026-08-02.
MAX_CANDLE_LIMIT = 1000
# Dror's standing rule: never cross, always isolated. Cross margin backs a
# losing position with the entire account balance, so one bad trade on 10-20x
# can reach money set aside for every other position; isolated caps the loss
# at that position's own margin, which is what the 1-2% per-trade sizing
# assumes in the first place.
MARGIN_MODE = "isolated"
DEMO_PRODUCT_TYPE = "SUSDT-FUTURES"

# Demo trading (unused by default) keeps the real symbol and marginCoin=USDT
# and only swaps productType; third-party docs claiming "SBTCSUSDT"/marginCoin
# SUSDT were wrong — confirmed by testing both against the live API.


class BitgetClient:
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        api_passphrase: str = "",
        demo: bool = False,
        timeout: float = 10.0,
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.demo = demo
        self.account_product_type = DEMO_PRODUCT_TYPE if demo else PRODUCT_TYPE
        self.account_margin_coin = MARGIN_COIN
        self.timeout = timeout
        self._session = requests.Session()
        self._contract_specs: dict[str, dict] | None = None

    # ---- public market data (no auth required) ----

    def get_candles(
        self, symbol: str, granularity: str = "1H", limit: int = 100, closed_only: bool = True
    ) -> list[list[str]]:
        """Returns [ts, open, high, low, close, base_vol, quote_vol], oldest first.

        Bitget always includes the currently-forming candle as the final entry;
        closed_only drops it so strategies never evaluate a half-formed bar.
        """
        wanted = limit + 1 if closed_only else limit
        params = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            # Clamped to what the endpoint will accept. `wanted` stays at the
            # full target so the history paging below still tops the series up
            # to it - the cap bounds one request, not the answer.
            "limit": str(min(wanted, MAX_CANDLE_LIMIT)),
        }
        data = self._request("GET", "/api/v2/mix/market/candles", params=params, signed=False)

        # This endpoint caps how far back it will go regardless of `limit` —
        # around 539 bars on 4H and only 89 on 1D, which is short of the 201 a
        # 200-period MA needs. The history endpoint pages further back from a
        # given point, so top up from there rather than silently returning a
        # series too short for the caller's indicators.
        while data and len(data) < wanted:
            older = self._request(
                "GET",
                "/api/v2/mix/market/history-candles",
                params={**params, "endTime": str(int(data[0][0])), "limit": "200"},
                signed=False,
            )
            if not older:
                break
            data = older + data

        return data[:-1] if closed_only and data else data

    def get_ticker(self, symbol: str) -> dict:
        params = {"symbol": symbol, "productType": PRODUCT_TYPE}
        data = self._request("GET", "/api/v2/mix/market/ticker", params=params, signed=False)
        return data[0] if data else {}

    def get_mark_price(self, symbol: str) -> float:
        return float(self.get_ticker(symbol)["markPrice"])

    def get_all_tickers(self) -> list[dict]:
        params = {"productType": PRODUCT_TYPE}
        return self._request("GET", "/api/v2/mix/market/tickers", params=params, signed=False)

    def get_contract_specs(self, symbol: str) -> dict:
        """Minimum order size / notional and price precision, cached for the
        process lifetime.

        price_place is how many decimals Bitget quotes this symbol to. Alerts
        need it because a fixed 2dp collapses every level of a cheap symbol to
        the same string — a DOGEUSDT alert read "Entry 0.07 Stop 0.07 Target
        0.07", which is neither actionable nor auditable. It varies from 1
        (BTCUSDT) to 9 (SHIBUSDT), so no single choice works watchlist-wide.
        """
        if self._contract_specs is None:
            params = {"productType": PRODUCT_TYPE}
            rows = self._request("GET", "/api/v2/mix/market/contracts", params=params, signed=False)
            self._contract_specs = {r["symbol"]: r for r in rows}
        spec = self._contract_specs.get(symbol, {})
        return {
            "min_size": float(spec.get("minTradeNum", 0) or 0),
            "min_notional": float(spec.get("minTradeUSDT", 0) or 0),
            "price_place": int(spec.get("pricePlace", 2) or 2),
            "volume_place": int(spec.get("volumePlace", 2) or 0),
            # Real-world asset: a tokenized stock, metal or commodity rather
            # than a coin. Intraday strategies need it because those track a
            # market that closes, while the bars keep printing regardless -
            # see notifier/sessions.py.
            "is_rwa": str(spec.get("isRwa", "NO")).upper() == "YES",
        }

    # ---- authenticated account reads ----

    def get_account_equity(self) -> float:
        """Total futures account equity in USDT — the base for position sizing."""
        params = {"productType": self.account_product_type}
        data = self._request("GET", "/api/v2/mix/account/accounts", params=params, signed=True)
        for account in data:
            if account.get("marginCoin") == self.account_margin_coin:
                return float(account["accountEquity"])
        raise RuntimeError(f"No {self.account_margin_coin} futures account found")

    def get_positions(self, symbol: str) -> list[dict]:
        """All open positions for symbol (hedge mode can return long and short)."""
        params = {
            "symbol": symbol,
            "productType": self.account_product_type,
            "marginCoin": self.account_margin_coin,
        }
        data = self._request("GET", "/api/v2/mix/position/single-position", params=params, signed=True)
        return [_parse_position(row) for row in (data or []) if float(row.get("total", 0) or 0) > 0]

    def get_all_positions(self) -> list[dict]:
        """Every open position on the account, in one call.

        get_positions() asks per symbol, which is right when you already know
        the symbol but is a hundred requests when the question is "what am I
        actually holding". Used to notice a position the bot is not tracking.
        """
        params = {"productType": self.account_product_type, "marginCoin": self.account_margin_coin}
        data = self._request("GET", "/api/v2/mix/position/all-position", params=params, signed=True)
        return [_parse_position(row) for row in (data or []) if float(row.get("total", 0) or 0) > 0]

    def get_position(self, symbol: str, direction: str | None = None) -> dict | None:
        """The open position for symbol, optionally restricted to one side."""
        positions = self.get_positions(symbol)
        if direction is not None:
            positions = [p for p in positions if p["direction"] == direction]
        return positions[0] if positions else None

    def get_plan_orders(self, symbol: str, direction: str) -> list[dict]:
        """Pending TP/SL plan orders for this symbol and side.

        Bitget uses TWO planType naming schemes here, both verified live
        against this account's own order history on 2026-08-05:

          loss_plan / profit_plan - order-level presets, created when an order
            carrying presetStopLossPrice fills. Sized to THAT leg, so a split
            entry produces one per leg (0.10 and 0.41 on the AAPLUSDT trade,
            both triggering at 310.94).
          pos_loss / pos_profit   - position-level TP/SL, as set from Bitget's
            own "Position TP/SL" panel. size 0 means "all closable".

        Matching is by substring rather than a longer exact list: these are the
        two schemes seen so far, not a documented closed set. `size` is carried
        through because how MUCH of a position a target covers is a different
        question from where it sits - a split entry's per-leg presets can leave
        a grown position only partly covered.
        """
        params = {"productType": self.account_product_type, "planType": "profit_loss"}
        data = self._request("GET", "/api/v2/mix/order/orders-plan-pending", params=params, signed=True)
        orders = data.get("entrustedList") if isinstance(data, dict) else data

        parsed = []
        for order in orders or []:
            if order.get("symbol") != symbol or order.get("posSide") != direction:
                continue
            plan_type = order.get("planType") or ""
            parsed.append(
                {
                    "plan_type": plan_type,
                    "is_stop": "loss" in plan_type,
                    "is_target": "profit" in plan_type,
                    "trigger_price": _optional_float(order.get("triggerPrice")),
                    # 0 is meaningful, not missing: it is Bitget's "all
                    # closable" sentinel for a position-level order.
                    "size": float(order.get("size") or 0),
                    "order_id": order.get("orderId"),
                }
            )
        return parsed

    def get_stop_target(self, symbol: str, direction: str) -> tuple[float | None, float | None]:
        """(stop_loss, take_profit) actually protecting the position.

        Prefers standalone TP/SL plan orders, since that's how they're usually
        set; falls back to the position's preset fields when no plan order
        exists so either mechanism is picked up.
        """
        stop = target = None
        for order in self.get_plan_orders(symbol, direction):
            if order["is_stop"]:
                stop = order["trigger_price"]
            elif order["is_target"]:
                target = order["trigger_price"]

        if stop is None or target is None:
            position = self.get_position(symbol, direction)
            if position:
                stop = stop if stop is not None else position["stop_loss"]
                target = target if target is not None else position["take_profit"]
        return stop, target

    def get_position_history(self, symbol: str, limit: int = 20) -> list[dict]:
        """Most recent closed positions for symbol, newest first."""
        params = {"symbol": symbol, "productType": self.account_product_type, "limit": str(limit)}
        data = self._request("GET", "/api/v2/mix/position/history-position", params=params, signed=True)
        rows = data.get("list", []) if isinstance(data, dict) else data
        return [_parse_closed_position(row) for row in (rows or [])]

    def find_closed_position(self, symbol: str, direction: str) -> dict | None:
        """Most recent closed position on this symbol/side. Its closeAvgPrice
        and netProfit already aggregate every partial close of that position,
        so it's the authoritative final record."""
        for row in self.get_position_history(symbol, limit=10):
            if row["direction"] == direction:
                return row
        return None

    # ---- order placement ----

    def round_size(self, symbol: str, size: float) -> float:
        """Size the exchange will accept, never larger than asked.

        Rounded DOWN throughout: rounding up would place more than the plan
        sized, so a 2%-risk trade could quietly become more. Symbols whose
        minimum trade size is a whole number also trade in multiples of it -
        PEPEUSDT floors to thousands, which is why an order for 3,262,901
        became a position of 3,262,000.
        """
        specs = self.get_contract_specs(symbol)
        step = 10 ** -specs["volume_place"]
        size = math.floor(size / step) * step
        min_size = specs["min_size"]
        if min_size >= 1:
            size = math.floor(size / min_size) * min_size
        return round(size, specs["volume_place"])

    def round_price(self, symbol: str, price: float) -> float:
        """Price at the symbol's own precision. Sending more decimals than a
        symbol quotes is rejected outright - INTCUSDT allows 2, and 91.01202
        was refused with a bare 400."""
        return round(price, self.get_contract_specs(symbol)["price_place"])

    def set_leverage(self, symbol: str, direction: str, leverage: float) -> dict:
        """Leverage is per symbol and side, and persists on the account.

        It must be set before the order: position sizing solves leverage
        dynamically per trade, and whatever value a previous trade on this
        symbol left behind would otherwise decide how much margin this one
        actually consumes.
        """
        return self._request(
            "POST",
            "/api/v2/mix/account/set-leverage",
            signed=True,
            body={
                "symbol": symbol,
                "productType": self.account_product_type,
                "marginCoin": self.account_margin_coin,
                "leverage": _trim(leverage),
                "holdSide": direction,
            },
        )

    def place_order(
        self,
        symbol: str,
        direction: str,
        size: float,
        order_type: str = "market",
        price: float | None = None,
        stop_loss: float | None = None,
        client_oid: str | None = None,
        reduce_only: bool = False,
    ) -> dict:
        """Open or reduce a position.

        The account is in hedge mode, where `side` alone is ambiguous: it is
        the pair of `side` and `tradeSide` that decides both direction and
        whether this opens or closes. Opening a short is sell/open, but
        *closing* a long is also a sell - so a wrong pairing does not error,
        it silently acts on the opposite side.

        `stop_loss` is attached to the order itself rather than sent
        afterwards, so a filled position is never briefly unprotected. On a
        resting limit it activates when the order fills.

        `client_oid` makes placement idempotent: Bitget rejects a duplicate,
        so a retry after an ambiguous failure cannot double the position.
        """
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

        if reduce_only:
            # DO NOT USE THIS BRANCH. It has never once succeeded against the
            # live account, and nothing in the bot routes through it any more.
            #
            # Every automated take-profit from 2026-08-03 to 2026-08-11 was
            # placed this way and every one was rejected - PEPEUSDT, AAPLUSDT,
            # GOOGLUSDT, ZECUSDT, WLDUSDT - almost all with 22002 "No position
            # to close" against positions that plainly existed. WLDUSDT's was
            # 155 units, complete 709ms before the first attempt, refused four
            # times across 22 seconds. Opens through the same method have
            # always worked, so the account and the credentials are fine; it
            # is this pairing that is wrong.
            #
            # The most likely reading is that hedge mode takes `side` as the
            # side of the POSITION being closed, so a long needs buy+close and
            # sending sell+close asks Bitget to close a short that does not
            # exist - which is exactly the error text. That has NOT been
            # confirmed by experiment, because confirming it means sending a
            # real order against a real position.
            #
            # Exits go through place_tpsl_order instead, which names the
            # position with `holdSide` and leaves nothing to infer. Anyone
            # tempted to revive this must verify it on one live trade first.
            side = "sell" if direction == "long" else "buy"
            trade_side = "close"
        else:
            side = "buy" if direction == "long" else "sell"
            trade_side = "open"

        # Every number crossing the wire is rounded to what this symbol
        # actually quotes. Doing it here rather than at each call site means a
        # new order type cannot forget to.
        size = self.round_size(symbol, size)
        if size <= 0:
            raise ValueError(f"{symbol}: size rounds to zero at the exchange's precision")

        body = {
            "symbol": symbol,
            "productType": self.account_product_type,
            "marginMode": MARGIN_MODE,
            "marginCoin": self.account_margin_coin,
            "size": _trim(size),
            "side": side,
            "tradeSide": trade_side,
            "orderType": order_type,
        }
        if order_type == "limit":
            if price is None:
                raise ValueError("a limit order needs a price")
            body["price"] = _trim(self.round_price(symbol, price))
        if stop_loss is not None:
            body["presetStopLossPrice"] = _trim(self.round_price(symbol, stop_loss))
        if client_oid:
            body["clientOid"] = client_oid

        return self._request("POST", "/api/v2/mix/order/place-order", signed=True, body=body)

    def place_tpsl_order(
        self,
        symbol: str,
        direction: str,
        plan_type: str,
        trigger_price: float,
        size: float | None = None,
        trigger_type: str = "mark_price",
        client_oid: str | None = None,
    ) -> dict:
        """Place a standalone TP or SL PLAN order against an existing position.

        Unlike place_order()'s `price` on a resting limit, `trigger_price` here
        is a condition, not an order sitting in the book right now - it only
        becomes an order once triggered. That is why this exists: a plain
        reduce-only limit take-profit is capped at the exchange's own
        buyLimitPriceRatio/sellLimitPriceRatio from the current mark price -
        just 2% on RWA (tokenized-stock) symbols - and gets rejected outright
        for any ordinary target beyond that, which a GOOGLUSDT trade hit with
        a completely normal ~3.7%-from-entry target. Bitget's own "Position
        TP/SL" panel places exactly this kind of order for the same reason.

        `plan_type` uses the same "profit_plan"/"loss_plan" strings already
        confirmed live in get_plan_orders() - this places the same kind of
        order directly instead of waiting for one to be auto-created from an
        order's presetStopLossPrice. `size` omitted closes the whole position.
        """
        if plan_type not in ("profit_plan", "loss_plan"):
            raise ValueError(f"plan_type must be 'profit_plan' or 'loss_plan', got {plan_type!r}")
        if direction not in ("long", "short"):
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

        body = {
            "marginCoin": self.account_margin_coin,
            "productType": self.account_product_type,
            "symbol": symbol,
            "planType": plan_type,
            "triggerPrice": _trim(self.round_price(symbol, trigger_price)),
            "triggerType": trigger_type,
            "holdSide": direction,
            "executePrice": "0",  # market execution once triggered
        }
        if size is not None:
            body["size"] = _trim(self.round_size(symbol, size))
        if client_oid:
            body["clientOid"] = client_oid

        return self._request("POST", "/api/v2/mix/order/place-tpsl-order", signed=True, body=body)

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        """Live (unfilled) orders — the source of truth after an ambiguous
        failure, and what tells us whether a resting leg is still out there."""
        params = {"productType": self.account_product_type}
        if symbol:
            params["symbol"] = symbol
        data = self._request("GET", "/api/v2/mix/order/orders-pending", params=params, signed=True)
        orders = data.get("entrustedList") if isinstance(data, dict) else data
        return orders or []

    def cancel_order(self, symbol: str, order_id: str | None = None, client_oid: str | None = None) -> dict:
        if not (order_id or client_oid):
            raise ValueError("cancelling needs either an order_id or a client_oid")
        body = {"symbol": symbol, "productType": self.account_product_type}
        if order_id:
            body["orderId"] = order_id
        if client_oid:
            body["clientOid"] = client_oid
        return self._request("POST", "/api/v2/mix/order/cancel-order", signed=True, body=body)

    def cancel_plan_order(
        self, symbol: str, plan_type: str, order_id: str | None = None, client_oid: str | None = None
    ) -> dict:
        """Cancel a TP/SL plan order — a separate endpoint from cancel_order()
        because plan orders (place_tpsl_order) live on their own book
        (orders-plan-pending), not the regular pending-orders one."""
        if not (order_id or client_oid):
            raise ValueError("cancelling needs either an order_id or a client_oid")
        body = {
            "symbol": symbol,
            "productType": self.account_product_type,
            "marginCoin": self.account_margin_coin,
            "planType": plan_type,
        }
        if order_id:
            body["orderId"] = order_id
        if client_oid:
            body["clientOid"] = client_oid
        return self._request("POST", "/api/v2/mix/order/cancel-plan-order", signed=True, body=body)

    # ---- internals ----

    def _request(
        self,
        method: str,
        request_path: str,
        params: dict | None = None,
        signed: bool = False,
        body: dict | None = None,
    ):
        query = urlencode(sorted(params.items())) if params else ""
        url = f"{BASE_URL}{request_path}"
        if query:
            url += f"?{query}"

        # The signature covers the exact bytes sent, so the body must be
        # serialised once and reused - re-dumping it for the request would
        # risk a different key order and a signature that doesn't match.
        body_text = json.dumps(body, separators=(",", ":")) if body is not None else ""

        headers = {"Content-Type": "application/json"}
        if signed:
            headers.update(self._sign_headers(method, request_path, query, body_text))
            # paptrading only applies to authenticated account calls — demo
            # trading still uses real market data, and sending it on public
            # market-data requests made Bitget reject them.
            if self.demo:
                headers["paptrading"] = "1"

        response = self._session.request(
            method, url, headers=headers, data=body_text or None, timeout=self.timeout
        )
        if response.status_code >= 400:
            # Bitget puts the real reason in the body ("size precision", "price
            # exceeds", ...). raise_for_status() alone discards it, which turned
            # a self-explaining rejection into a bare 400 and cost a live
            # debugging round trip.
            raise RuntimeError(
                f"Bitget {response.status_code} on {request_path}: {response.text[:400]}"
            )
        payload = response.json()
        if payload.get("code") != "00000":
            raise RuntimeError(f"Bitget API error {payload.get('code')}: {payload.get('msg')}")
        return payload["data"]

    def _sign_headers(self, method: str, request_path: str, query: str, body_text: str = "") -> dict:
        if not (self.api_key and self.api_secret and self.api_passphrase):
            raise RuntimeError("Bitget API credentials are required for this call")

        timestamp = str(int(time.time() * 1000))
        prehash = timestamp + method.upper() + request_path
        if query:
            prehash += f"?{query}"
        prehash += body_text

        signature = base64.b64encode(
            hmac.new(self.api_secret.encode(), prehash.encode(), hashlib.sha256).digest()
        ).decode()

        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.api_passphrase,
        }


def _parse_position(row: dict) -> dict:
    return {
        "symbol": row.get("symbol"),
        "direction": row.get("holdSide"),
        "entry_price": float(row.get("openPriceAvg", 0) or 0),
        "size": float(row.get("total", 0) or 0),
        # Empty unless set as presets at entry; get_stop_target() also checks
        # plan orders, which is where they usually live.
        "stop_loss": _optional_float(row.get("stopLoss")),
        "take_profit": _optional_float(row.get("takeProfit")),
        "unrealized_pnl": float(row.get("unrealizedPL", 0) or 0),
        "realized_pnl": float(row.get("achievedProfits", 0) or 0),
        "leverage": float(row.get("leverage", 1) or 1),
        "raw": row,
    }


def _parse_closed_position(row: dict) -> dict:
    return {
        "symbol": row.get("symbol"),
        "direction": row.get("holdSide"),
        "entry_price": float(row.get("openAvgPrice", 0) or 0),
        "exit_price": float(row.get("closeAvgPrice", 0) or 0),
        "closed_size": float(row.get("closeTotalPos", 0) or 0),
        # netProfit is after fees — what actually hit the balance. Gross `pnl`
        # can show a winner on a trade that lost money once fees are counted.
        "realized_pnl": float(row.get("netProfit", 0) or 0),
        "gross_pnl": float(row.get("pnl", 0) or 0),
        "close_time_ms": int(row.get("utime", 0) or 0),
        "raw": row,
    }


def _trim(value: float) -> str:
    """Bitget wants numbers as strings, and rejects exponent notation - which
    is exactly how Python formats the small prices on this watchlist (1e-05).
    Trailing zeros go too, since some endpoints treat '1.50' and '1.5' as
    different precision."""
    text = f"{float(value):.10f}".rstrip("0").rstrip(".")
    return text or "0"


def _optional_float(value) -> float | None:
    if value in (None, "", "0", 0):
        return None
    return float(value)


def client_from_settings(settings) -> BitgetClient:
    """Picks demo or live credentials based on BITGET_DEMO_MODE."""
    if settings.bitget_demo_mode:
        return BitgetClient(
            settings.bitget_demo_api_key,
            settings.bitget_demo_api_secret,
            settings.bitget_demo_api_passphrase,
            demo=True,
        )
    return BitgetClient(settings.bitget_api_key, settings.bitget_api_secret, settings.bitget_api_passphrase)
