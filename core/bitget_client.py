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
import time
from urllib.parse import urlencode

import requests

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_COIN = "USDT"
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
        params = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": str(limit + 1 if closed_only else limit),
        }
        data = self._request("GET", "/api/v2/mix/market/candles", params=params, signed=False)
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
            price = _optional_float(order.get("triggerPrice"))
            if order.get("planType") == "loss_plan":
                stop = price
            elif order.get("planType") == "profit_plan":
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

    # ---- internals ----

    def _request(self, method: str, request_path: str, params: dict | None = None, signed: bool = False):
        query = urlencode(sorted(params.items())) if params else ""
        url = f"{BASE_URL}{request_path}"
        if query:
            url += f"?{query}"

        headers = {"Content-Type": "application/json"}
        if signed:
            headers.update(self._sign_headers(method, request_path, query))
            # paptrading only applies to authenticated account calls — demo
            # trading still uses real market data, and sending it on public
            # market-data requests made Bitget reject them.
            if self.demo:
                headers["paptrading"] = "1"

        response = self._session.request(method, url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if payload.get("code") != "00000":
            raise RuntimeError(f"Bitget API error {payload.get('code')}: {payload.get('msg')}")
        return payload["data"]

    def _sign_headers(self, method: str, request_path: str, query: str) -> dict:
        if not (self.api_key and self.api_secret and self.api_passphrase):
            raise RuntimeError("Bitget API credentials are required for this call")

        timestamp = str(int(time.time() * 1000))
        prehash = timestamp + method.upper() + request_path
        if query:
            prehash += f"?{query}"

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
