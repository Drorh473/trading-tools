from core.bitget_client import BitgetClient


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


def test_get_stop_target_reads_position_level_plan_orders(monkeypatch):
    # Captured live 2026-08-05 from a real AAPLUSDT short protected via
    # Bitget's "Position TP/SL" panel. planType comes back as "pos_loss" /
    # "pos_profit" - the old code matched "loss_plan" / "profit_plan" instead
    # (never verified against a real response) and silently returned
    # (None, None) for a position that was, in fact, fully protected.
    entrusted_list = [
        {
            "planType": "pos_profit",
            "symbol": "AAPLUSDT",
            "triggerPrice": "295.91",
            "posSide": "short",
        },
        {
            "planType": "pos_loss",
            "symbol": "AAPLUSDT",
            "triggerPrice": "332",
            "posSide": "short",
        },
    ]

    class FakePlanResponse(FakeResponse):
        def json(self):
            return {"code": "00000", "msg": "success", "data": {"entrustedList": entrusted_list}}

    client = BitgetClient("key", "secret", "pass", demo=False)
    monkeypatch.setattr(
        client._session, "request", lambda *a, **kw: FakePlanResponse()
    )

    stop, target = client.get_stop_target("AAPLUSDT", "short")

    assert stop == 332.0
    assert target == 295.91


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
