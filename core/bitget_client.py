"""Thin wrapper over Bitget's v2 REST API, targeting USDT-margined futures
(productType=USDT-FUTURES) — the product the user actually trades.

Public market data (candles, ticker) needs no credentials. Authenticated
endpoints (positions) use Bitget's documented signing scheme:
HMAC-SHA256(timestamp + METHOD + requestPath[+"?"+query] + body, secret),
base64-encoded, sent via the ACCESS-* headers.

NOTE ON VERIFICATION (last checked 2026-07-29): public candle/ticker endpoints
are confirmed working live, for both real (USDT-FUTURES) and demo
(SUSDT-FUTURES) productTypes — including the demo-specific quirk that demo
symbols keep their real name (e.g. "SBTCSUSDT", not translated) and demo
account calls use marginCoin="USDT", not "SUSDT" despite what the contract
spec's supportMarginCoins field says (confirmed by testing both).

What's still NOT verified: get_position()/get_position_history() against an
actual open position. Auth and demo routing (paptrading header) both work —
confirmed by testing against a real demo BTCUSDT/SBTCSUSDT position — but
every symbol-scoped query against that demo position returned "Parameter
<symbol> does not exist" regardless of the marginCoin tried, while the
symbol-less list endpoints (all-position, account/accounts) succeeded but
came back empty even though the position was confirmed open in the browser
UI at the time. That's either a real Bitget-side inconsistency between the
demo UI and this API key's account context, or something about demo
positions this project hasn't found yet — not something resolved by trying
more parameter combinations. _parse_position()/_parse_closed_position()'s
field names (openPriceAvg, holdSide, presetStopLossPrice, etc.) remain
untested guesses from cross-referenced third-party SDK docs. Before trusting
this with real money: retest against either a real (tiny) live position
(the fully-confirmed-working code path) or a demo position once the
account-linkage issue above is understood.
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

# Verified live against a real demo account: demo trading uses productType
# SUSDT-FUTURES but keeps the real symbol (e.g. "BTCUSDT", no prefix) and
# marginCoin=USDT. (Third-party docs claiming an "S"-prefixed symbol like
# "SBTCSUSDT" and marginCoin=SUSDT were wrong/outdated — confirmed by testing
# both against the live API.)


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

    # ---- public market data (no auth required) ----

    def get_candles(self, symbol: str, granularity: str = "1m", limit: int = 100) -> list[list[str]]:
        """Returns [ts, open, high, low, close, base_vol, quote_vol], oldest first."""
        params = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "granularity": granularity,
            "limit": str(limit),
        }
        data = self._request("GET", "/api/v2/mix/market/candles", params=params, signed=False)
        return list(reversed(data))

    def get_ticker(self, symbol: str) -> dict:
        params = {"symbol": symbol, "productType": PRODUCT_TYPE}
        data = self._request("GET", "/api/v2/mix/market/ticker", params=params, signed=False)
        return data[0] if data else {}

    def get_mark_price(self, symbol: str) -> float:
        return float(self.get_ticker(symbol)["markPrice"])

    # ---- authenticated position reads ----

    def get_position(self, symbol: str) -> dict | None:
        """Current open position for symbol, or None if flat."""
        params = {
            "symbol": symbol,
            "productType": self.account_product_type,
            "marginCoin": self.account_margin_coin,
        }
        data = self._request("GET", "/api/v2/mix/position/single-position", params=params, signed=True)
        if not data:
            return None
        return _parse_position(data[0])

    def get_position_history(self, symbol: str, limit: int = 20) -> list[dict]:
        """Most recent closed positions for symbol, newest first."""
        params = {"symbol": symbol, "productType": self.account_product_type, "limit": str(limit)}
        data = self._request("GET", "/api/v2/mix/position/history-position", params=params, signed=True)
        rows = data.get("list", []) if isinstance(data, dict) else data
        return [_parse_closed_position(row) for row in rows]

    # ---- internals ----

    def _request(self, method: str, request_path: str, params: dict | None = None, signed: bool = False):
        query = urlencode(sorted(params.items())) if params else ""
        url = f"{BASE_URL}{request_path}"
        if query:
            url += f"?{query}"

        headers = {"Content-Type": "application/json"}
        if self.demo:
            headers["paptrading"] = "1"
        if signed:
            headers.update(self._sign_headers(method, request_path, query))

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
        "direction": "long" if row.get("holdSide") == "long" else "short",
        "entry_price": float(row.get("openPriceAvg", 0) or 0),
        "size": float(row.get("total", row.get("available", 0)) or 0),
        "stop_loss": _optional_float(row.get("presetStopLossPrice")),
        "take_profit": _optional_float(row.get("presetTakeProfitPrice")),
        "unrealized_pnl": float(row.get("unrealizedPL", 0) or 0),
        "leverage": float(row.get("leverage", 1) or 1),
        "raw": row,
    }


def _parse_closed_position(row: dict) -> dict:
    return {
        "symbol": row.get("symbol"),
        "direction": "long" if row.get("holdSide") == "long" else "short",
        "entry_price": float(row.get("openAvgPrice", 0) or 0),
        "exit_price": float(row.get("closeAvgPrice", 0) or 0),
        "realized_pnl": float(row.get("pnl", row.get("netProfit", 0)) or 0),
        "close_time_ms": int(row.get("cTime", row.get("utime", 0)) or 0),
        "raw": row,
    }


def _optional_float(value) -> float | None:
    if value in (None, "", "0"):
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
