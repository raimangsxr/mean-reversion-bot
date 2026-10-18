from datetime import datetime, timedelta, timezone
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
        self.orders = {}
        self.cancel_calls = []

    def price_to_precision(self, symbol, price):
        return f"{float(price):.2f}"

    def amount_to_precision(self, symbol, amount):
        return f"{float(amount):.6f}"

    def create_order(self, symbol, order_type, side, amount, price, params):
        client_id = params.get("clientOrderId", "")
        if client_id.startswith("zsl_"):
            order_id = "sl-1"
        elif client_id.startswith("zptp_"):
            order_id = "ptp-1"
        elif client_id.startswith("ztp_"):
            order_id = "tp-1"
        else:
            order_id = "entry-1"
        order = {
            "id": order_id,
            "status": "open",
            "filled": 0.0,
            "symbol": symbol,
            "type": order_type,
            "side": side,
            "amount": amount,
            "price": price,
            "clientOrderId": client_id,
            "params": params,
        }
        self.created.append(order)
        self.orders[order["id"]] = dict(order)
        return dict(order)

    def fetch_order(self, order_id, symbol, params=None):
        return dict(self.orders[order_id])

    def fetch_open_orders(self, symbol, *args):
        return [
            dict(order)
            for order in self.orders.values()
            if order.get("status") == "open"
        ]

    def cancel_order(self, order_id, symbol, params=None):
        self.cancel_calls.append((order_id, symbol, dict(params or {})))
        self.orders[order_id]["status"] = "canceled"
        return dict(self.orders[order_id])

    def fetch_my_trades(self, symbol, since=None, limit=None):
        return []


def make_engine(tmp_path: Path) -> BinanceExecutionEngine:
    engine = object.__new__(BinanceExecutionEngine)
    engine.exchange = FakeExchange()
    engine.take_profit_order_mode = "maker_limit"
    engine.state_store = JsonStateStore(str(tmp_path / "state.json"))
    engine.journal_path = tmp_path / "journal.jsonl"
    return engine


def pending_state(*, expires_delta_s: int = 420, max_lateness_s: int = 60) -> BotState:
    return BotState(
        status="entry_pending",
        symbol="BTC/USDT:USDT",
        side="long",
        amount=1.0,
        entry_order_id="entry-1",
        entry_client_order_id="zentry_test",
        entry_order_kind="limit",
        pending_reference_price=100.0,
        pending_limit_price=99.97,
        pending_expires_at=(
            datetime.now(timezone.utc) + timedelta(seconds=expires_delta_s)
        ).isoformat(),
        pending_stop_distance=2.0,
        pending_risk_reward_ratio=1.5,
        pending_use_take_profit=True,
        pending_fallback_cap_bps=5.0,
        pending_fallback_max_lateness_seconds=max_lateness_s,
        pending_journal_context={"signal_time": "signal-1"},
        last_signal_timestamp="signal-1",
    )


def test_entry_limit_is_post_only_and_not_reduce_only(tmp_path):
    engine = make_engine(tmp_path)

    order = engine._create_entry_limit(
        symbol="BTC/USDT:USDT",
        side="long",
        amount=1.0,
        price=99.971,
        client_order_id="zentry_test",
    )

    assert order["type"] == "limit"
    assert order["side"] == "buy"
    assert order["price"] == 99.97
    assert order["params"]["postOnly"] is True
    assert "reduceOnly" not in order["params"]


def test_submit_passive_entry_persists_signal_and_expiry(tmp_path):
    engine = make_engine(tmp_path)
    engine.get_position_state = lambda symbol: {
        "known": True,
        "has_position": False,
        "state": BotState(),
    }

    result = engine.submit_passive_entry(
        symbol="BTC/USDT:USDT",
        side="long",
        amount=1.0,
        reference_price=100.0,
        limit_offset_bps=3.0,
        wait_seconds=420,
        fallback_cap_bps=5.0,
        fallback_max_lateness_seconds=60,
        stop_distance=2.0,
        risk_reward_ratio=1.5,
        use_take_profit=True,
        signal_timestamp="signal-1",
        journal_context={"signal_z": -2.1},
    )

    state = result["state"]
    assert state.status == "entry_pending"
    assert state.last_signal_timestamp == "signal-1"
    assert state.entry_order_id == "entry-1"
    assert state.pending_reference_price == 100.0
    assert state.pending_limit_price == 99.97
    assert state.pending_fallback_cap_bps == 5.0
    assert engine.pending_entry_remaining_seconds(state) > 400


def test_get_position_state_keeps_open_pending_limit(tmp_path):
    engine = make_engine(tmp_path)
    state = pending_state()
    engine.state_store.save(state)
    engine.exchange.orders["entry-1"] = {
        "id": "entry-1",
        "status": "open",
        "filled": 0.0,
        "clientOrderId": "zentry_test",
    }
    engine._fetch_open_position = lambda symbol: None

    result = engine.get_position_state("BTC/USDT:USDT")

    assert result["known"] is True
    assert result["has_position"] is False
    assert result["entry_pending"] is True
    assert engine.state_store.load().status == "entry_pending"


def test_pending_fill_is_adopted_instead_of_treated_as_foreign_position(tmp_path):
    engine = make_engine(tmp_path)
    state = pending_state()
    engine.state_store.save(state)
    position = PositionSnapshot(
        side="long",
        contracts=1.0,
        entry_price=99.97,
        notional=99.97,
    )
    engine._fetch_open_position = lambda symbol: position
    protected_state = BotState(status="protected", side="long", entry_price=99.97)
    engine._adopt_pending_fill = lambda symbol, state, position: {
        "state": protected_state,
    }

    result = engine.get_position_state("BTC/USDT:USDT")

    assert result["known"] is True
    assert result["has_position"] is True
    assert result["state"].status == "protected"


def test_expired_pending_outside_cap_is_cancelled_without_fallback(tmp_path):
    engine = make_engine(tmp_path)
    state = pending_state(expires_delta_s=-5)
    engine.state_store.save(state)
    engine.exchange.orders["entry-1"] = {
        "id": "entry-1",
        "status": "open",
        "filled": 0.0,
        "clientOrderId": "zentry_test",
    }
    engine._fetch_open_position = lambda symbol: None
    engine.get_market_price = lambda symbol: 100.20
    called = {"market": False}

    def market_entry(**kwargs):
        called["market"] = True
        raise AssertionError("No debe ejecutar fallback fuera del cap")

    engine.execute_trade_with_protection = market_entry

    result = engine.resolve_pending_entry("BTC/USDT:USDT")

    assert result["action"] == "cancelled_no_chase"
    assert called["market"] is False
    assert engine.state_store.load().status == "flat"
    assert engine.state_store.load().last_signal_timestamp == "signal-1"


def test_expired_pending_inside_cap_uses_market_fallback(tmp_path):
    engine = make_engine(tmp_path)
    state = pending_state(expires_delta_s=-5)
    engine.state_store.save(state)
    engine.exchange.orders["entry-1"] = {
        "id": "entry-1",
        "status": "open",
        "filled": 0.0,
        "clientOrderId": "zentry_test",
    }
    engine._fetch_open_position = lambda symbol: None
    engine.get_market_price = lambda symbol: 100.04
    final = BotState(status="protected", side="long", entry_price=100.04)
    captured = {}

    def market_entry(**kwargs):
        captured.update(kwargs)
        return {"state": final, "entry": {"id": "mkt-1"}}

    engine.execute_trade_with_protection = market_entry

    result = engine.resolve_pending_entry("BTC/USDT:USDT")

    assert result["action"] == "fallback_market"
    assert result["state"].status == "protected"
    assert captured["side"] == "long"
    assert captured["stop_distance"] == 2.0
    assert captured["journal_context"]["entry_source"] == "market_fallback_conditional"


def test_late_resume_cancels_without_market_fallback(tmp_path):
    engine = make_engine(tmp_path)
    state = pending_state(expires_delta_s=-120, max_lateness_s=60)
    engine.state_store.save(state)
    engine.exchange.orders["entry-1"] = {
        "id": "entry-1",
        "status": "open",
        "filled": 0.0,
        "clientOrderId": "zentry_test",
    }
    engine._fetch_open_position = lambda symbol: None
    engine.get_market_price = lambda symbol: (_ for _ in ()).throw(
        AssertionError("No debe consultar precio si el fallback llega demasiado tarde")
    )

    result = engine.resolve_pending_entry("BTC/USDT:USDT")

    assert result["action"] == "cancelled_late"
    assert engine.state_store.load().status == "flat"


def test_partial_pending_fill_cancels_remainder_and_installs_protection(tmp_path):
    engine = make_engine(tmp_path)
    state = pending_state()
    engine.state_store.save(state)
    engine.exchange.orders["entry-1"] = {
        "id": "entry-1",
        "status": "open",
        "filled": 0.4,
        "remaining": 0.6,
        "average": 99.97,
        "clientOrderId": "zentry_test",
    }
    position = PositionSnapshot(
        side="long",
        contracts=0.4,
        entry_price=99.97,
        notional=39.988,
    )
    engine._fetch_open_position = lambda symbol: position

    result = engine.get_position_state("BTC/USDT:USDT")

    assert result["known"] is True
    assert result["has_position"] is True
    final = engine.state_store.load()
    assert final.status == "protected"
    assert final.amount == 0.4
    assert final.entry_price == 99.97
    assert final.stop_price == 97.97
    assert final.take_profit_price == 102.97
    assert final.pending_limit_price is None
    assert ("entry-1", "BTC/USDT:USDT", {}) in engine.exchange.cancel_calls
    created_types = [order["type"] for order in engine.exchange.created]
    assert "stop_market" in created_types
    assert "limit" in created_types
    tp = next(order for order in engine.exchange.created if order["id"] == "tp-1")
    assert tp["params"]["reduceOnly"] is True
    assert tp["params"]["postOnly"] is True


def test_separates_limit_and_fallback_references(tmp_path):
    engine = make_engine(tmp_path)
    engine.get_position_state = lambda symbol: {
        "known": True,
        "has_position": False,
        "state": BotState(),
    }

    result = engine.submit_passive_entry(
        symbol="BTC/USDT:USDT",
        side="long",
        amount=1.0,
        reference_price=100.0,          # signal close
        fallback_reference_price=100.10,  # t=0
        limit_offset_bps=3.0,
        wait_seconds=420,
        fallback_cap_bps=5.0,
        fallback_max_lateness_seconds=60,
        stop_distance=2.0,
        risk_reward_ratio=1.5,
        use_take_profit=True,
        signal_timestamp="signal-current",
    )

    state = result["state"]
    assert state.pending_reference_price == 100.0
    assert state.pending_fallback_reference_price == 100.10
    assert state.pending_limit_price == 99.97
    assert state.entry_order_kind == "limit"


def test_marketable_limit_enters_wait_only_instead_of_forcing_post_only(tmp_path):
    engine = make_engine(tmp_path)
    engine.get_position_state = lambda symbol: {
        "known": True,
        "has_position": False,
        "state": BotState(),
    }

    result = engine.submit_passive_entry(
        symbol="BTC/USDT:USDT",
        side="long",
        amount=1.0,
        reference_price=100.0,          # LIMIT = 99.97
        fallback_reference_price=99.90, # ya seria marketable en t=0
        limit_offset_bps=3.0,
        wait_seconds=420,
        fallback_cap_bps=5.0,
        fallback_max_lateness_seconds=60,
        stop_distance=2.0,
        risk_reward_ratio=1.5,
        use_take_profit=True,
        signal_timestamp="signal-wait",
    )

    state = result["state"]
    assert result["order"] is None
    assert state.status == "entry_pending"
    assert state.entry_order_kind == "wait"
    assert state.entry_order_id is None
    assert state.entry_client_order_id is None
    assert engine.exchange.created == []

    engine._fetch_open_position = lambda symbol: None
    reconciled = BinanceExecutionEngine.get_position_state(
        engine, "BTC/USDT:USDT"
    )
    assert reconciled["known"] is True
    assert reconciled["entry_pending"] is True
    assert reconciled["pending_order_status"] == "wait_only"


def test_post_only_marketable_rejection_falls_back_to_wait_only(tmp_path):
    engine = make_engine(tmp_path)
    engine.get_position_state = lambda symbol: {
        "known": True,
        "has_position": False,
        "state": BotState(),
    }

    def reject_post_only(*args, **kwargs):
        raise RuntimeError("-5022 Post Only order will be rejected")

    engine.exchange.create_order = reject_post_only

    result = engine.submit_passive_entry(
        symbol="BTC/USDT:USDT",
        side="long",
        amount=1.0,
        reference_price=100.0,
        fallback_reference_price=100.10,
        limit_offset_bps=3.0,
        wait_seconds=420,
        fallback_cap_bps=5.0,
        fallback_max_lateness_seconds=60,
        stop_distance=2.0,
        risk_reward_ratio=1.5,
        use_take_profit=True,
        signal_timestamp="signal-post-only",
    )

    state = result["state"]
    assert result["order"] is None
    assert state.status == "entry_pending"
    assert state.entry_order_kind == "wait"
    assert state.entry_order_id is None
    assert state.entry_client_order_id is None


def test_move_stop_to_break_even_creates_new_stop_before_cancel(tmp_path):
    engine = make_engine(tmp_path)
    old = {
        "id": "sl-old", "status": "open", "filled": 0.0,
        "symbol": "BTC/USDT:USDT", "type": "stop_market", "side": "sell",
        "amount": 1.0, "price": None, "clientOrderId": "zsl_old", "params": {},
    }
    engine.exchange.orders["sl-old"] = dict(old)
    engine.state_store.save(BotState(
        status="protected", symbol="BTC/USDT:USDT", side="long",
        entry_price=100.0, amount=1.0, initial_stop_price=98.0, stop_price=98.0,
        stop_order_id="sl-old", stop_client_order_id="zsl_old",
        protection_mode="stop_only",
    ))
    engine._fetch_open_position = lambda symbol: PositionSnapshot(
        side="long", contracts=1.0, entry_price=100.0, notional=100.0
    )

    state = engine.move_stop_to_break_even("BTC/USDT:USDT")

    assert state.break_even_armed is True
    assert state.stop_price == 100.0
    assert state.stop_order_id == "sl-1"
    assert engine.exchange.cancel_calls[0][0] == "sl-old"


def test_partial_tp_splits_exit_quantity_and_persists_order(tmp_path, monkeypatch):
    import src.execution.engine as engine_module

    monkeypatch.setattr(engine_module, "PARTIAL_TP_TRIGGER_R", 0.75)
    monkeypatch.setattr(engine_module, "PARTIAL_TP_FRACTION", 0.50)
    monkeypatch.setattr(engine_module, "PARTIAL_TP_MOVE_STOP_TO_BE", True)
    engine = make_engine(tmp_path)

    result = engine._protect_filled_entry(
        symbol="BTC/USDT:USDT", side="long", filled=1.0, fill_price=100.0,
        entry_order_id="entry-1", entry_client_order_id="zentry_test",
        entry_order_kind="limit", stop_distance=2.0, risk_reward_ratio=1.5,
        use_take_profit=True,
    )

    state = result["state"]
    assert state.partial_take_profit_order_id == "ptp-1"
    assert state.partial_take_profit_amount == 0.5
    assert state.partial_take_profit_price == 101.5
    partial = next(o for o in engine.exchange.created if o["id"] == "ptp-1")
    final = next(o for o in engine.exchange.created if o["id"] == "tp-1")
    assert partial["amount"] == 0.5
    assert final["amount"] == 0.5
    assert final["price"] == 103.0


def test_completed_partial_tp_is_detected_for_break_even(tmp_path, monkeypatch):
    import src.execution.engine as engine_module

    monkeypatch.setattr(engine_module, "PARTIAL_TP_TRIGGER_R", 0.75)
    monkeypatch.setattr(engine_module, "PARTIAL_TP_FRACTION", 0.50)
    engine = make_engine(tmp_path)
    result = engine._protect_filled_entry(
        symbol="BTC/USDT:USDT", side="long", filled=1.0, fill_price=100.0,
        entry_order_id="entry-1", entry_client_order_id="zentry_test",
        entry_order_kind="limit", stop_distance=2.0, risk_reward_ratio=1.5,
        use_take_profit=True,
    )
    engine.exchange.orders["ptp-1"]["status"] = "closed"
    engine.exchange.orders["ptp-1"]["filled"] = 0.5
    engine._fetch_open_position = lambda symbol: PositionSnapshot(
        side="long", contracts=0.5, entry_price=100.0, notional=50.0
    )

    state_result = engine.get_position_state("BTC/USDT:USDT")

    assert state_result["known"] is True
    assert engine.state_store.load().partial_take_profit_filled is True
