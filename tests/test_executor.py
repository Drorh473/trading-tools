import json

import pytest

from core.bitget_client import BitgetClient
from execution.executor import (
    DryRunExecutor,
    LiveExecutor,
    ManualExecutor,
    OrderLeg,
    RoutingExecutor,
    TradeOrder,
)


class FakeResponse:
    def __init__(self, payload=None):
        self._payload = payload or {"code": "00000", "data": {"orderId": "1"}}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _client(monkeypatch, fail_on=None):
    client = BitgetClient("k", "s", "p")
    sent = []

    def fake_request(method, url, headers=None, timeout=None, data=None):
        body = json.loads(data) if data else None
        sent.append({"method": method, "url": url, "body": body, "headers": headers})
        if fail_on and fail_on in url:
            raise RuntimeError("network died")
        return FakeResponse()

    monkeypatch.setattr(client._session, "request", fake_request)
    return client, sent


def _order(**kwargs):
    defaults = dict(
        symbol="BTCUSDT",
        direction="long",
        legs=[OrderLeg(size=2.0, order_type="market")],
        stop_loss=95.0,
        leverage=10.0,
        strategy_tag="Strategy 1 1H",
    )
    defaults.update(kwargs)
    return TradeOrder(**defaults)


def test_manual_executor_places_nothing():
    assert ManualExecutor().execute(_order()).ok


def test_dry_run_reports_the_payload_without_sending(monkeypatch):
    client, sent = _client(monkeypatch)
    reported = []
    executor = DryRunExecutor(report=reported.append)

    order = _order(
        legs=[
            OrderLeg(size=0.4, order_type="market"),
            OrderLeg(size=1.6, order_type="limit", price=99.0, note="61.8% Fib"),
        ]
    )
    assert executor.execute(order).ok

    assert sent == []  # nothing left the process
    assert "DRY RUN" in reported[0]
    assert "0.4 market" in reported[0]
    assert "1.6 limit 99" in reported[0]
    assert "61.8% Fib" in reported[0]


def test_hedge_mode_pairs_side_and_tradeside_for_opening(monkeypatch):
    client, sent = _client(monkeypatch)

    LiveExecutor(client).execute(_order(direction="long"))
    order_body = sent[-1]["body"]
    assert (order_body["side"], order_body["tradeSide"]) == ("buy", "open")

    LiveExecutor(client).execute(_order(direction="short"))
    order_body = sent[-1]["body"]
    assert (order_body["side"], order_body["tradeSide"]) == ("sell", "open")


def test_reduce_only_flips_the_side_not_the_direction(monkeypatch):
    # Closing a long is also a "sell", so only the tradeSide distinguishes it
    # from opening a short - a wrong pairing here opens the opposite position
    # instead of erroring.
    client, sent = _client(monkeypatch)

    client.place_order("BTCUSDT", "long", 1.0, reduce_only=True)
    assert (sent[-1]["body"]["side"], sent[-1]["body"]["tradeSide"]) == ("sell", "close")

    client.place_order("BTCUSDT", "short", 1.0, reduce_only=True)
    assert (sent[-1]["body"]["side"], sent[-1]["body"]["tradeSide"]) == ("buy", "close")


def test_stop_rides_on_the_entry_order(monkeypatch):
    # Never a window where a filled position sits unprotected.
    client, sent = _client(monkeypatch)
    LiveExecutor(client).execute(_order(stop_loss=95.0))

    assert sent[-1]["body"]["presetStopLossPrice"] == "95"


def test_leverage_is_set_before_any_order(monkeypatch):
    client, sent = _client(monkeypatch)
    LiveExecutor(client).execute(_order(leverage=12.5))

    assert "set-leverage" in sent[0]["url"]
    assert sent[0]["body"]["leverage"] == "12.5"
    assert "place-order" in sent[1]["url"]


def test_each_leg_gets_its_own_stable_client_oid(monkeypatch):
    client, sent = _client(monkeypatch)
    order = _order(
        legs=[
            OrderLeg(size=0.4, order_type="market"),
            OrderLeg(size=1.6, order_type="limit", price=99.0),
        ]
    )
    LiveExecutor(client).execute(order)

    oids = [s["body"]["clientOid"] for s in sent if "place-order" in s["url"]]
    assert len(oids) == 2
    assert len(set(oids)) == 2  # distinct legs
    # Stable across calls, so a retry is rejected as a duplicate rather than
    # doubling the position.
    assert order.client_oid(0) == oids[0]


def test_a_failed_leg_stops_the_rest_and_reports_it(monkeypatch):
    client, sent = _client(monkeypatch, fail_on="place-order")
    order = _order(
        legs=[
            OrderLeg(size=0.4, order_type="market"),
            OrderLeg(size=1.6, order_type="limit", price=99.0),
        ]
    )

    result = LiveExecutor(client).execute(order)

    assert not result.ok
    assert "leg 1 of 2" in result.error
    placed = [s for s in sent if "place-order" in s["url"]]
    assert len(placed) == 1  # never attempted the second, never retried the first


def test_leverage_failure_prevents_any_order(monkeypatch):
    # Sizing derives margin from leverage, so placing anyway would consume
    # whatever a previous trade on this symbol left behind.
    client, sent = _client(monkeypatch, fail_on="set-leverage")

    result = LiveExecutor(client).execute(_order())

    assert not result.ok
    assert "leverage" in result.error
    assert not [s for s in sent if "place-order" in s["url"]]


def test_small_prices_never_serialise_as_exponents(monkeypatch):
    # Python renders 0.00001 as "1e-05", which Bitget rejects - and this
    # watchlist runs down to sub-cent symbols.
    client, sent = _client(monkeypatch)
    client.place_order("SHIBUSDT", "long", 1e7, order_type="limit", price=0.00001)

    assert sent[-1]["body"]["price"] == "0.00001"
    assert "e-" not in sent[-1]["body"]["size"]


def test_limit_order_without_a_price_is_rejected():
    with pytest.raises(ValueError):
        BitgetClient("k", "s", "p").place_order("BTCUSDT", "long", 1.0, order_type="limit")


def test_body_is_signed_exactly_as_sent(monkeypatch):
    client, sent = _client(monkeypatch)
    client.place_order("BTCUSDT", "long", 1.0)

    body_text = json.dumps(sent[-1]["body"], separators=(",", ":"))
    assert sent[-1]["headers"]["ACCESS-SIGN"]
    # The signature covers the serialised body, so it must round-trip byte for
    # byte; a re-dump with different separators would invalidate it.
    assert json.loads(body_text) == sent[-1]["body"]


def test_routing_sends_each_strategy_to_its_own_executor(monkeypatch):
    client, sent = _client(monkeypatch)
    live, dry = LiveExecutor(client), DryRunExecutor()
    router = RoutingExecutor({"Strategy 1 1H": live, "Strategy 2 4H/1H": dry})

    router.execute(_order(strategy_tag="Strategy 1 1H"))
    router.execute(_order(strategy_tag="Strategy 2 4H/1H"))

    placed = [s for s in sent if "place-order" in s["url"]]
    assert len(placed) == 1  # only Strategy 1 reached the exchange
    assert len(dry.orders) == 1


def test_routing_reports_live_only_for_live_strategies(monkeypatch):
    # The scanner places exit orders directly, so it must be able to ask
    # whether a given strategy is genuinely live - otherwise a dry-run
    # strategy would still get real take-profit orders placed for it.
    client, _ = _client(monkeypatch)
    router = RoutingExecutor({"Strategy 1 1H": LiveExecutor(client), "Strategy 2 4H/1H": DryRunExecutor()})

    assert router.handles_live("Strategy 1 1H") is True
    assert router.handles_live("Strategy 2 4H/1H") is False
    assert router.handles_live("Strategy 3 1D/1H") is False  # unrouted -> manual


def test_an_unrouted_strategy_places_nothing(monkeypatch):
    client, sent = _client(monkeypatch)
    router = RoutingExecutor({"Strategy 1 1H": LiveExecutor(client)})

    assert router.execute(_order(strategy_tag="Strategy 3 1D/1H")).ok
    assert not [s for s in sent if "place-order" in s["url"]]
