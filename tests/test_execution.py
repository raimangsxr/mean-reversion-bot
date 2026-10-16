from pathlib import Path

from src.execution.engine import (
    BinanceExecutionEngine,
    BotState,
    JsonStateStore,
    PositionSnapshot,
)


class FakeExchange:
    def __init__(self):
        self.created = []
        self.fetch_calls = []
        self.cancel_calls = []
        self.tp_order = {
            "id": "tp-1",
            "status": "open",
            "filled": 0.0,
            "remaining": 1.0,
            "price": 120.0,
        }

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def create_order(self, symbol, order_type, side, amount, price, params):
        order = {
            "id": "created-1",
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "params": params,
        }
        self.created.append(order)
        return order

    def fetch_order(self, order_id, symbol, params=None):
        params = params or {}
        self.fetch_calls.append((order_id, symbol, dict(params)))
        if params.get("trigger"):
            return {"id": order_id, "status": "open", "filled": 0.0}
        return dict(self.tp_order)

    def fetch_open_orders(self, symbol, *args):
        return []

    def cancel_order(self, order_id, symbol, params=None):
        self.cancel_calls.append((order_id, symbol, dict(params or {})))
        return {"id": order_id, "status": "canceled"}

    def fetch_my_trades(self, symbol, since=None, limit=None):
        return []


def make_engine(tmp_path: Path) -> BinanceExecutionEngine:
    engine = object.__new__(BinanceExecutionEngine)
    engine.exchange = FakeExchange()
    engine.take_profit_order_mode = "maker_limit"
    engine.state_store = JsonStateStore(str(tmp_path / "state.json"))
    engine.journal_path = tmp_path / "journal.jsonl"
    return engine


def test_maker_tp_is_reduce_only_post_only_limit(tmp_path):
    engine = make_engine(tmp_path)

    order = engine._create_take_profit_limit(
        symbol="BTC/USDT:USDT",
        side="sell",
        amount=0.1,
        price=105_000.123,
        client_order_id="ztp_test",
    )

    assert order["type"] == "limit"
    assert order["price"] == 105000.12
    assert order["params"]["reduceOnly"] is True
    assert order["params"]["postOnly"] is True
    assert order["params"]["clientOrderId"] == "ztp_test"


def test_limit_tp_is_verified_as_regular_order_not_trigger(tmp_path):
    engine = make_engine(tmp_path)
    state = BotState(
        status="protected",
        symbol="BTC/USDT:USDT",
        side="long",
        stop_order_id="sl-1",
        take_profit_order_id="tp-1",
        take_profit_order_kind="limit",
        protection_mode="stop_and_tp_maker",
    )
    engine.state_store.save(state)
    engine._fetch_open_position = lambda symbol: PositionSnapshot(
        side="long",
        contracts=1.0,
        entry_price=100.0,
        notional=100.0,
    )

    result = engine.get_position_state("BTC/USDT:USDT")

    assert result["known"] is True
    assert result["has_position"] is True
    calls = engine.exchange.fetch_calls
    assert ("sl-1", "BTC/USDT:USDT", {"trigger": True}) in calls
    assert ("tp-1", "BTC/USDT:USDT", {}) in calls


def test_cancel_protection_uses_different_order_families(tmp_path):
    engine = make_engine(tmp_path)
    engine.state_store.save(
        BotState(
            status="protected",
            symbol="BTC/USDT:USDT",
            side="long",
            stop_order_id="sl-1",
            take_profit_order_id="tp-1",
            take_profit_order_kind="limit",
            protection_mode="stop_and_tp_maker",
        )
    )

    engine._cancel_bot_protection_orders("BTC/USDT:USDT")

    assert ("sl-1", "BTC/USDT:USDT", {"trigger": True}) in engine.exchange.cancel_calls
    assert ("tp-1", "BTC/USDT:USDT", {}) in engine.exchange.cancel_calls


def test_partial_maker_fill_is_journaled_once_per_new_cumulative_fill(tmp_path):
    engine = make_engine(tmp_path)
    engine.exchange.tp_order.update(
        {
            "filled": 0.25,
            "remaining": 0.75,
            "average": 120.0,
        }
    )
    engine.state_store.save(
        BotState(
            status="protected",
            symbol="BTC/USDT:USDT",
            side="long",
            stop_order_id="sl-1",
            take_profit_order_id="tp-1",
            take_profit_order_kind="limit",
            take_profit_last_filled=0.0,
            protection_mode="stop_and_tp_maker",
        )
    )
    engine._fetch_open_position = lambda symbol: PositionSnapshot(
        side="long",
        contracts=0.75,
        entry_price=100.0,
        notional=75.0,
    )

    engine.get_position_state("BTC/USDT:USDT")
    engine.get_position_state("BTC/USDT:USDT")

    lines = engine.journal_path.read_text(encoding="utf-8").splitlines()
    partial_events = [line for line in lines if '"event":"take_profit_partial_fill"' in line]
    assert len(partial_events) == 1
    assert engine.state_store.load().take_profit_last_filled == 0.25
