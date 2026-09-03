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


def test_get_candles_pages_further_back_when_the_endpoint_is_short(monkeypatch):
    """Bitget's own candles endpoint caps how far back it will go regardless
    of `limit` - around 89 bars on 1D, short of the 201 a 200-period MA
    needs. The history-candles endpoint pages further back from a given
    point; this is what tops a short first page up to the full request
    instead of silently handing the caller a series too short for its own
    indicators."""
    client = BitgetClient()

    first_page = [[str(i), "1", "1", "1", "1", "1", "1"] for i in range(10, 15)]  # 5 bars, ts 10..14
    older_page = [[str(i), "1", "1", "1", "1", "1", "1"] for i in range(1, 10)]  # 9 bars, ts 1..9
    calls = []

    class PagedResponse:
        status_code = 200

        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"code": "00000", "msg": "success", "data": self._data}

    def fake_request(method, url, headers=None, timeout=None, data=None):
        calls.append(url)
        if "history-candles" in url:
            return PagedResponse(older_page)
        return PagedResponse(first_page)

    monkeypatch.setattr(client._session, "request", fake_request)

    result = client.get_candles("BTCUSDT", "1H", limit=13, closed_only=False)

    assert len(calls) == 2, "a short first page must trigger exactly one top-up request"
    assert "endTime=10" in calls[1], "pages back from the oldest ts the first page actually returned"
    # Older candles PREPENDED - the whole series stays oldest-first.
    assert [row[0] for row in result] == [str(i) for i in range(1, 15)]


def test_get_candles_stops_paging_when_history_runs_out(monkeypatch):
    """A symbol with genuinely less history than requested (a new listing)
    must not loop forever asking history-candles for bars that don't exist -
    an empty page has to end the loop, not retry the same endTime."""
    client = BitgetClient()

    first_page = [[str(i), "1", "1", "1", "1", "1", "1"] for i in range(10, 15)]  # 5 bars
    calls = []

    class PagedResponse:
        status_code = 200

        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return {"code": "00000", "msg": "success", "data": self._data}

    def fake_request(method, url, headers=None, timeout=None, data=None):
        calls.append(url)
        if "history-candles" in url:
            return PagedResponse([])  # nothing further back exists
        return PagedResponse(first_page)

    monkeypatch.setattr(client._session, "request", fake_request)

    result = client.get_candles("BTCUSDT", "1H", limit=1000, closed_only=False)

    assert len(calls) == 2, "must stop after ONE empty page, not keep retrying"
    assert len(result) == 5


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


def _fallback_client(monkeypatch, entrusted_list, position):
    """Like _plan_client, but get_position returns a REAL position instead
    of being stubbed to None - these tests are about the PRESET FALLBACK,
    which every plan-order test above deliberately excludes by stubbing
    get_position to None."""

    class FakePlanResponse(FakeResponse):
        def json(self):
            return {"code": "00000", "msg": "success", "data": {"entrustedList": entrusted_list}}

    class FallbackClient(BitgetClient):
        def get_position(self, symbol, direction=None):
            return position

    client = FallbackClient("key", "secret", "pass", demo=False)
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: FakePlanResponse())
    return client


def test_get_stop_target_falls_back_to_the_positions_own_presets(monkeypatch):
    """With no plan order at all, the only place a stop/target can live is
    the position's own preset fields - set when they were attached AT ENTRY
    rather than as a separate plan order (see the module's own docstring).
    Deliberately unexercised by every plan-order test above."""
    client = _fallback_client(monkeypatch, [], {"stop_loss": 305.0, "take_profit": 320.0})

    assert client.get_stop_target("AAPLUSDT", "short") == (305.0, 320.0)


def test_get_stop_target_only_falls_back_for_the_missing_half(monkeypatch):
    """A plan order supplying the stop must not be overwritten by the
    position's preset - only the target, which nothing else provided,
    falls back."""
    client = _fallback_client(
        monkeypatch,
        [{"planType": "pos_loss", "symbol": "AAPLUSDT", "triggerPrice": "332", "posSide": "short"}],
        {"stop_loss": 999.0, "take_profit": 295.91},  # 999 must never surface - the plan order wins
    )

    assert client.get_stop_target("AAPLUSDT", "short") == (332.0, 295.91)


def test_get_stop_target_is_none_none_with_nothing_anywhere(monkeypatch):
    client = _fallback_client(monkeypatch, [], None)

    assert client.get_stop_target("AAPLUSDT", "short") == (None, None)


def _listing_client(monkeypatch, rows):
    class FakeListResponse(FakeResponse):
        def json(self):
            return {"code": "00000", "msg": "success", "data": rows}

    client = BitgetClient("key", "secret", "pass", demo=False)
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: FakeListResponse())
    return client


def test_get_account_equity_picks_the_matching_margin_coin(monkeypatch):
    """The account listing can carry more than one coin; only USDT is what
    every trade here is margined in."""
    client = _listing_client(
        monkeypatch,
        [
            {"marginCoin": "BTC", "accountEquity": "0.002"},
            {"marginCoin": "USDT", "accountEquity": "123.45"},
        ],
    )

    assert client.get_account_equity() == 123.45


def test_get_account_equity_raises_when_the_margin_coin_is_missing(monkeypatch):
    """Sizing off a stale or guessed equity silently corrupts every
    downstream number (notifier.scanner.tick's own reasoning for skipping a
    scan on this exact failure) - raising rather than defaulting to 0 or
    None is what makes that skip possible."""
    client = _listing_client(monkeypatch, [{"marginCoin": "BTC", "accountEquity": "0.002"}])

    with pytest.raises(RuntimeError, match="USDT"):
        client.get_account_equity()


def test_get_positions_drops_zeroed_out_rows(monkeypatch):
    """Bitget can return a row for a position that was just closed, total
    0 - the account holds nothing, and treating it as an open position
    would double-count or misreport size."""
    client = _listing_client(
        monkeypatch,
        [
            {"symbol": "BTCUSDT", "holdSide": "long", "total": "0", "openPriceAvg": "100"},
            {"symbol": "BTCUSDT", "holdSide": "short", "total": "0.5", "openPriceAvg": "100"},
        ],
    )

    positions = client.get_positions("BTCUSDT")

    assert len(positions) == 1
    assert positions[0]["direction"] == "short"


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
        client.place_tpsl_order(
            symbol="GOOGLUSDT", direction="long", plan_type="bogus", trigger_price=100.0, size=1.0
        )


def test_place_tpsl_order_refuses_to_omit_size(monkeypatch):
    """Bitget answers a sizeless plan order with 40019 "Parameter size cannot
    be empty". The docstring used to promise that omitting it closed the whole
    position; it does not, and believing that cost BZUSDT #18 its breakeven
    stop on 2026-08-17 - the first partial ever to reach that handler. Caught
    here rather than at the exchange, because a stop that fails on placement
    fails exactly when a position needs it."""
    client = BitgetClient("key", "secret", "pass", demo=False)
    _capture_headers(client, monkeypatch)

    with pytest.raises(ValueError, match="needs a size"):
        client.place_tpsl_order(
            symbol="GOOGLUSDT", direction="long", plan_type="loss_plan", trigger_price=100.0
        )


def test_position_level_stop_sends_size_zero_meaning_all_closable(monkeypatch):
    """A stop must cover the WHOLE position, including anything a later
    scale-in adds, so the breakeven and trailing paths use a position-level
    pos_loss with Bitget's size-0 "all closable" sentinel. 0 must survive to
    the wire as "0" rather than being rounded like a quantity."""
    import json

    client = BitgetClient("key", "secret", "pass", demo=False)
    captured = _capture_headers(client, monkeypatch)
    monkeypatch.setattr(
        client, "get_contract_specs",
        lambda symbol: {"min_size": 1000.0, "min_notional": 5.0, "price_place": 2,
                        "volume_place": 0, "is_rwa": False},
    )

    client.place_tpsl_order(
        symbol="PEPEUSDT", direction="long", plan_type="pos_loss", trigger_price=85.2671, size=0
    )

    body = json.loads(captured["data"])
    assert body["planType"] == "pos_loss"
    assert body["size"] == "0"  # the sentinel, not a rounded quantity
    assert body["triggerPrice"] == "85.27"


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


def _fill(order_id, price, volume, ctime, trade_side="close", profit="0"):
    return {
        "orderId": order_id,
        "price": str(price),
        "baseVolume": str(volume),
        "cTime": str(ctime),
        "tradeSide": trade_side,
        "profit": str(profit),
    }


def _fills_client(monkeypatch, fills):
    client = BitgetClient("k", "s", "p")
    monkeypatch.setattr(
        client, "_request", lambda *a, **kw: {"fillList": fills}
    )
    return client


def test_closing_exits_separates_the_partial_from_the_runner(monkeypatch):
    """APTUSDT #11. closeAvgPrice reported 0.5485 for a trade that took half
    off at 0.5608 and ran the rest to 0.5360 - an average of two closes, and
    a price nothing ever traded at."""
    client = _fills_client(monkeypatch, [
        _fill("o2", 0.5360, 35.009, 1786_800_000_000, profit="2.52459794"),
        _fill("o1", 0.5608, 35.010, 1786_600_000_000, profit="1.65642205"),
    ])

    exits = client.get_closing_exits("APTUSDT", position_size=70.019)

    assert [(round(e["size"], 3), e["price"]) for e in exits] == [(35.010, 0.5608), (35.009, 0.5360)]
    assert exits[0]["profit"] == pytest.approx(1.65642205)


def test_closing_exits_treats_one_order_filled_in_pieces_as_one_exit(monkeypatch):
    """An 81.216 + 3.242 pair on this account was one decision, not two."""
    client = _fills_client(monkeypatch, [
        _fill("o1", 0.5886, 81.216, 1786_600_000_000),
        _fill("o1", 0.5886, 3.242, 1786_600_000_000),
    ])

    exits = client.get_closing_exits("APTUSDT", position_size=84.458)

    assert len(exits) == 1
    assert exits[0]["size"] == pytest.approx(84.458)
    assert exits[0]["price"] == pytest.approx(0.5886)


def test_closing_exits_stops_at_the_previous_position_on_the_symbol(monkeypatch):
    """The endpoint returns every fill for the symbol whatever position it
    belonged to. Bounded by SIZE rather than by a timestamp, so no clock or
    timezone can drag an older position's closes into this trade."""
    client = _fills_client(monkeypatch, [
        _fill("new", 0.5360, 35.009, 1786_800_000_000),
        _fill("new0", 0.5608, 35.010, 1786_600_000_000),
        _fill("old", 0.5925, 60.984, 1786_200_000_000),  # a position from days earlier
    ])

    exits = client.get_closing_exits("APTUSDT", position_size=70.019)

    assert len(exits) == 2
    assert all(e["size"] < 40 for e in exits), "the 60.984 close belongs to the previous position"


def test_closing_exits_ignores_opening_fills(monkeypatch):
    client = _fills_client(monkeypatch, [
        _fill("open1", 0.6105, 25.351, 1786_200_000_000, trade_side="open"),
        _fill("close1", 0.5360, 35.009, 1786_800_000_000),
    ])

    exits = client.get_closing_exits("APTUSDT", position_size=35.009)

    assert [e["price"] for e in exits] == [0.5360]


def test_a_reset_connection_is_retried_at_the_transport_level():
    """SPCXUSDT #37: 38 'Poll failed... will retry' errors over six days, all
    ConnectionResetError from a pooled connection Bitget's side had already
    closed. Each one cost a full POLL_INTERVAL (10s) wait for the tracker's
    OWN retry loop to try again. A transport-level retry resolves most of
    these inside the same call, in milliseconds, rather than surfacing as a
    failure at all.

    Exercising urllib3's actual retry loop end-to-end needs either a live
    socket or patching its private internals, both too fragile for a unit
    test of a configuration decision - so this checks the configuration
    itself: the mounted adapter must actually retry, on more than zero
    attempts, with the connection-reset case covered.
    """
    client = BitgetClient(demo=True)
    retry = client._session.get_adapter("https://api.bitget.com").max_retries

    assert retry.total >= 1, "a reset must be retried at least once, not just re-raised"
    assert retry.connect is None or retry.connect >= 1, "connection-establishment failures must be covered"
    assert retry.backoff_factor > 0, "retries with no backoff hammer a server that just reset us"


def test_a_post_is_never_retried_at_the_transport_level():
    """The other half of the same rule RoutingExecutor already applies to
    order placement: a POST whose response was lost is ambiguous - it may
    have already reached Bitget - and retrying blindly is how a position
    silently doubles (see its own docstring). Transport-level retries must
    stay off for writes; only urllib3's own default-safe methods (GET, HEAD,
    PUT, DELETE, OPTIONS, TRACE) may be retried underneath us."""
    client = BitgetClient(api_key="k", api_secret="s", api_passphrase="p")
    retry = client._session.get_adapter("https://api.bitget.com").max_retries

    assert "POST" not in retry.allowed_methods
    assert "GET" in retry.allowed_methods


def test_both_schemes_get_the_retrying_adapter():
    """Mounting only https:// would silently leave http:// (and therefore
    anything misconfigured to use it) on the unpatched default adapter."""
    client = BitgetClient(demo=True)

    https_retry = client._session.get_adapter("https://api.bitget.com").max_retries
    http_retry = client._session.get_adapter("http://api.bitget.com").max_retries

    assert https_retry.total >= 1
    assert http_retry.total >= 1


class _FillsResponse:
    status_code = 200

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return {"code": "00000", "msg": "success", "data": self._data}


def _fee_fill(trade_id: str, fee: float) -> dict:
    return {"tradeId": trade_id, "feeDetail": [{"totalFee": str(-abs(fee))}]}


def test_get_fees_paid_sums_a_single_page(monkeypatch):
    client = BitgetClient("key", "secret", "pass")
    fills = [_fee_fill("1", 0.02), _fee_fill("2", 0.06), _fee_fill("3", 0.01)]
    calls = []

    def fake_request(method, url, headers=None, timeout=None, data=None):
        calls.append(url)
        return _FillsResponse(fills)

    monkeypatch.setattr(client._session, "request", fake_request)

    total = client.get_fees_paid(1000, 2000)

    assert total == pytest.approx(0.09)
    assert len(calls) == 1
    assert "symbol=" not in calls[0], "account-wide, not filtered to one symbol"
    assert "startTime=1000" in calls[0]
    assert "endTime=2000" in calls[0]


def test_get_fees_paid_pages_past_a_full_first_page(monkeypatch):
    """A single call caps at 100 fills; a full page must trigger another
    request, paged with idLessThan off the last row's tradeId."""
    client = BitgetClient("key", "secret", "pass")
    first_page = [_fee_fill(str(i), 0.01) for i in range(100)]  # exactly a full page
    second_page = [_fee_fill("100", 0.05)]  # the top-up, short of 100
    calls = []

    def fake_request(method, url, headers=None, timeout=None, data=None):
        calls.append(url)
        if "idLessThan" in url:
            return _FillsResponse(second_page)
        return _FillsResponse(first_page)

    monkeypatch.setattr(client._session, "request", fake_request)

    total = client.get_fees_paid(1000, 2000)

    assert len(calls) == 2, "a full first page must trigger exactly one more request"
    assert "idLessThan=99" in calls[1], "pages from the last row's tradeId, not the first"
    assert total == pytest.approx(100 * 0.01 + 0.05)


def test_get_fees_paid_is_zero_with_no_fills(monkeypatch):
    client = BitgetClient("key", "secret", "pass")
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: _FillsResponse([]))

    assert client.get_fees_paid(1000, 2000) == 0.0


# ---- funding ----


class _BillsResponse:
    status_code = 200

    def __init__(self, bills):
        self._bills = bills

    def raise_for_status(self):
        pass

    def json(self):
        return {"code": "00000", "msg": "success", "data": {"bills": self._bills}}


def _bill(bill_id: str, business: str, amount: float) -> dict:
    return {"billId": bill_id, "businessType": business, "amount": str(amount)}


def test_funding_paid_is_positive_when_funding_left_the_account(monkeypatch):
    """Bitget signs amount from the account's side - negative means it was
    charged. get_funding_paid re-orients that as a COST so the monthly
    reconciliation can subtract fees and funding with the same sign."""
    client = BitgetClient("key", "secret", "pass")
    bills = [_bill("1", "contract_settle_fee", -0.014), _bill("2", "contract_settle_fee", -0.006)]
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: _BillsResponse(bills))

    assert client.get_funding_paid(1000, 2000) == pytest.approx(0.02)


def test_funding_received_comes_back_negative(monkeypatch):
    """Funding is a two-way payment. A month where the account was net PAID
    must not read as a cost, or the reconciliation residual absorbs it."""
    client = BitgetClient("key", "secret", "pass")
    bills = [_bill("1", "contract_settle_fee", 0.05), _bill("2", "contract_settle_fee", -0.01)]
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: _BillsResponse(bills))

    assert client.get_funding_paid(1000, 2000) == pytest.approx(-0.04)


def test_funding_ignores_bill_rows_that_are_not_settlements(monkeypatch):
    """The bill endpoint returns every balance movement - opens, closes,
    transfers. Summing all of them would report trading P&L as funding."""
    client = BitgetClient("key", "secret", "pass")
    bills = [
        _bill("1", "open_long", -50.0),
        _bill("2", "close_long", 12.0),
        _bill("3", "trans_from_exchange", 100.0),
        _bill("4", "contract_settle_fee", -0.03),
    ]
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: _BillsResponse(bills))

    assert client.get_funding_paid(1000, 2000) == pytest.approx(0.03)


def test_funding_pages_past_a_full_first_page(monkeypatch):
    client = BitgetClient("key", "secret", "pass")
    first_page = [_bill(str(i), "contract_settle_fee", -0.001) for i in range(100)]
    second_page = [_bill("100", "contract_settle_fee", -0.05)]
    calls = []

    def fake_request(method, url, headers=None, timeout=None, data=None):
        calls.append(url)
        return _BillsResponse(second_page if "idLessThan" in url else first_page)

    monkeypatch.setattr(client._session, "request", fake_request)

    total = client.get_funding_paid(1000, 2000)

    assert len(calls) == 2
    assert "idLessThan=99" in calls[1], "pages from the last row's billId"
    assert total == pytest.approx(100 * 0.001 + 0.05)


def test_funding_stops_rather_than_looping_when_a_page_has_no_cursor(monkeypatch):
    """A full page with no billId gives no safe way to ask for the next one.
    Under-reporting is visible in the report's settlement count; an infinite
    loop against a live exchange is not."""
    client = BitgetClient("key", "secret", "pass")
    page = [{"businessType": "contract_settle_fee", "amount": "-0.001"} for _ in range(100)]
    calls = []

    def fake_request(method, url, headers=None, timeout=None, data=None):
        calls.append(url)
        return _BillsResponse(page)

    monkeypatch.setattr(client._session, "request", fake_request)

    assert client.get_funding_paid(1000, 2000) == pytest.approx(0.1)
    assert len(calls) == 1, "no cursor must stop paging, not spin"


def test_funding_is_zero_with_no_bills(monkeypatch):
    client = BitgetClient("key", "secret", "pass")
    monkeypatch.setattr(client._session, "request", lambda *a, **kw: _BillsResponse([]))

    assert client.get_funding_paid(1000, 2000) == 0.0
