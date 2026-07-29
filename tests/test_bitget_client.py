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

    def fake_request(method, url, headers=None, timeout=None):
        captured["headers"] = headers
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
