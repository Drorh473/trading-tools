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
            "limit": str(wanted),
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

    def get_position(self, symbol: str, direction: str | None = None) -> dict | None:
        """The open position for symbol, optionally restricted to one side."""
        positions = self.get_positions(symbol)
        if direction is not None:
            positions = [p for p in positions if p["direction"] == direction]
        return positions[0] if positions else None

    def get_stop_target(self, symbol: str, direction: str) -> tuple[float | None, float | None]:
        """(stop_loss, take_profit) actually protecting the position.

        Prefers standalone TP/SL plan orders, since that's how they're usually
        set; falls back to the position's preset fields when no plan order
        exists so either mechanism is picked up.
        """
        stop = target = None
        params = {"productType": self.account_product_type, "planType": "profit_loss"}
        data = self._request("GET", "/api/v2/mix/order/orders-plan-pending", params=params, signed=True)
        orders = data.get("entrustedList") if isinstance(data, dict) else data

        for order in orders or []:
            if order.get("symbol") != symbol or order.get("posSide") != direction:
                continue
            # Checked live 2026-08-05 against two real orders placed from
            # Bitget's own "Position TP/SL" panel - the field actually comes
            # back as "pos_loss"/"pos_profit", not "loss_plan"/"profit_plan".
            # The old exact match against the wrong strings meant this whole
            # branch silently matched nothing: get_stop_target() returned
            # (None, None) for a position that was fully protected, for
            # every plan order ever placed this way. Substring match instead
            # of another guessed exact string, since nothing here has ever
            # been verified against Bitget's full set of planType values.
            plan_type = order.get("planType") or ""
            price = _optional_float(order.get("triggerPrice"))
            if "loss" in plan_type:
                stop = price
            elif "profit" in plan_type:
                target = price

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
            # Closing: sell to reduce a long, buy to reduce a short.
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
