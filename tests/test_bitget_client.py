import pytest

from core.bitget_client import MAX_CANDLE_LIMIT, BitgetClient


class FakeResponse:
    def __init__(self):
        self.status_code = 200

    def raise_for_status(self):
        pass

    def json(self):
        return {"code": "00000", "msg": "success", "data": []}


def _capture_headers(client, monkeypatch):
    captured = {}

    def fake_request(method, url, headers=None, timeout=None, data=None):
        captured["headers"] = headers
        captured["data"] = data
        captured["url"] = url
        captured["method"] = method
        return FakeResponse()

    monkeypatch.setattr(client._session, "request", fake_request)
    return captured


def test_demo_mode_does_not_send_paptrading_on_public_market_data(monkeypatch):
    client = BitgetClient(demo=True)
    captured = _capture_headers(client, monkeypatch)

    client.get_candles("BTCUSDT", "1H", 10)

    assert "paptrading" not in captured["headers"]


def test_a_candle_request_never_asks_for_more_than_bitget_allows(monkeypatch):
    """Bitget 40053: "limit should be between (0, 1000]".

    closed_only fetches one EXTRA bar so the forming candle can be dropped, so
    a caller asking for a perfectly legal 1000 produced a 1001 and a 400. The
    weekly report asks for exactly 1000 closed 15m bars and had never once run
    because of it - every Sunday since 2026-08-02 died on this line.
    """
    client = BitgetClient()
    captured = _capture_headers(client, monkeypatch)

    client.get_candles("BTCUSDT", "15m", limit=MAX_CANDLE_LIMIT, closed_only=True)

    sent = int(captured["url"].split("limit=")[1].split("&")[0])
    assert sent <= MAX_CANDLE_LIMIT, f"asked Bitget for {sent} candles"


def test_demo_mode_sends_paptrading_on_authenticated_calls(monkeypatch):
    client = BitgetClient("key", "secret", "pass", demo=True)
    captured = _capture_headers(client, monkeypatch)

    client.get_position("BTCUSDT")

    assert captured["headers"]["paptrading"] == "1"


def test_non_demo_mode_never_sends_paptrading(monkeypatch):
    client = BitgetClient("key", "secret", "pass", demo=False)
    captured = _capture_headers(client, monkeypatch)

    client.get_position("BTCUSDT")

    assert "paptrading" not in captured["headers"]


def _plan_client(monkeypatch, entrusted_list):
    """A client whose plan-orders endpoint returns `entrusted_list`.

    get_position is stubbed to None so the preset fallback contributes
    nothing - these tests are about what the plan-order branch alone reads.
    """

    class FakePlanResponse(FakeResponse):
        def json(self):
            return {"code": "00000", "msg": "success", "data": {"entrustedList": entrusted_list}}

    class PlanOnlyClient(BitgetClient):
        def get_position(self, symbol, direction=None):
            return None

    client = PlanOnlyClient("key", "secret", "pass", demo=False)
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: FakePlanResponse())
    return client


def test_get_stop_target_reads_position_level_plan_orders(monkeypatch):
    # Captured live 2026-08-05 from a real AAPLUSDT short protected via
    # Bitget's "Position TP/SL" panel: planType is "pos_loss" / "pos_profit",
    # size 0 (all closable). The old exact match on "loss_plan"/"profit_plan"
    # missed these entirely, so a hand-set target read back as None.
    client = _plan_client(
        monkeypatch,
        [
            {"planType": "pos_profit", "symbol": "AAPLUSDT", "triggerPrice": "295.91", "posSide": "short"},
            {"planType": "pos_loss", "symbol": "AAPLUSDT", "triggerPrice": "332", "posSide": "short"},
        ],
    )

    assert client.get_stop_target("AAPLUSDT", "short") == (332.0, 295.91)


def test_get_stop_target_reads_order_level_preset_plan_orders(monkeypatch):
    """The OTHER naming scheme, which the bot's own orders produce.

    presetStopLossPrice on a filled order creates a "loss_plan" sized to that
    leg. Both schemes have to keep working: the bot places the first kind and
    Dror places the second by hand, often on the same position.
    """
    client = _plan_client(
        monkeypatch,
        [
            {"planType": "loss_plan", "symbol": "AAPLUSDT", "triggerPrice": "310.94", "posSide": "short"},
            {"planType": "profit_plan", "symbol": "AAPLUSDT", "triggerPrice": "305.10", "posSide": "short"},
        ],
    )

    assert client.get_stop_target("AAPLUSDT", "short") == (310.94, 305.10)


def test_a_split_entry_reports_one_stop_for_both_leg_sized_plans(monkeypatch):
    """A split entry produces one loss_plan PER LEG at the same trigger.

    Verified live on trade #6: legs of 0.10 and 0.41 each created their own
    loss_plan at 310.94, and both executed together to close the full 0.51.
    The reported stop is that shared trigger price, not a doubling of it.
    """
    client = _plan_client(
        monkeypatch,
        [
            {"planType": "loss_plan", "symbol": "AAPLUSDT", "triggerPrice": "310.94", "posSide": "short", "size": "0.1"},
            {"planType": "loss_plan", "symbol": "AAPLUSDT", "triggerPrice": "310.94", "posSide": "short", "size": "0.41"},
        ],
    )

    stop, _ = client.get_stop_target("AAPLUSDT", "short")

    assert stop == 310.94


def test_place_tpsl_order_sends_a_trigger_not_a_resting_limit(monkeypatch):
    """The whole reason this method exists: a plain reduce-only limit
    take-profit is capped at the exchange's own price band from mark (2% on
    RWA symbols), which rejected a completely ordinary GOOGLUSDT target
    outright. A plan order's triggerPrice is a condition, not a price
    resting in the book right now, so it must go to place-tpsl-order, not
    place-order, and must carry triggerPrice/holdSide/planType rather than
    a plain limit `price`."""
    client = BitgetClient("key", "secret", "pass", demo=False)
    captured = _capture_headers(client, monkeypatch)

    client.place_tpsl_order(
        symbol="GOOGLUSDT",
        direction="long",
        plan_type="profit_plan",
        trigger_price=374.7649,
        size=0.02,
    )

    import json

    assert "/api/v2/mix/order/place-tpsl-order" in captured["url"]
    body = json.loads(captured["data"])
    assert body["symbol"] == "GOOGLUSDT"
    assert body["planType"] == "profit_plan"
    assert body["holdSide"] == "long"
    assert body["triggerPrice"] == "374.76"  # rounded to the symbol's own precision
    assert "price" not in body  # not a resting limit - nothing to reject against the band


def test_place_tpsl_order_rejects_an_unknown_plan_type(monkeypatch):
    client = BitgetClient("key", "secret", "pass", demo=False)
    _capture_headers(client, monkeypatch)

    with pytest.raises(ValueError):
        client.place_tpsl_order(symbol="GOOGLUSDT", direction="long", plan_type="bogus", trigger_price=100.0)


def test_get_stop_target_ignores_orders_for_a_different_symbol_or_side(monkeypatch):
    entrusted_list = [
        {"planType": "pos_loss", "symbol": "AAPLUSDT", "triggerPrice": "332", "posSide": "long"},
        {"planType": "pos_loss", "symbol": "TSLAUSDT", "triggerPrice": "100", "posSide": "short"},
    ]

    class FakePlanResponse(FakeResponse):
        def json(self):
            return {"code": "00000", "msg": "success", "data": {"entrustedList": entrusted_list}}

    class NoPositionClient(BitgetClient):
        def get_position(self, symbol, direction=None):
            return None

    client = NoPositionClient("key", "secret", "pass", demo=False)
    monkeypatch.setattr(
        client._session, "request", lambda *a, **kw: FakePlanResponse()
    )

    stop, target = client.get_stop_target("AAPLUSDT", "short")

    assert (stop, target) == (None, None)
