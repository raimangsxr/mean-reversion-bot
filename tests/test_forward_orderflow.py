from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pytest

from src.forward.gate import (
    create_armed_signal,
    journal_arm,
    journal_flow_decision,
    recover_armed_signals,
)
from src.forward.journal import ForwardJournal
from src.forward.orderflow import AggTradeCollector


def _payload(*, agg_id: int, symbol: str, price: float, qty: float, when: datetime, buyer_maker: bool):
    ms = int(when.timestamp() * 1000)
    return json.dumps(
        {
            "stream": f"{symbol.lower()}@aggTrade",
            "data": {
                "e": "aggTrade",
                "E": ms,
                "a": agg_id,
                "s": symbol,
                "p": str(price),
                "q": str(qty),
                "T": ms,
                "m": buyer_maker,
            },
        }
    )


def test_usdm_aggtrade_url_uses_current_market_route():
    collector = AggTradeCollector(["BTC/USDT:USDT", "DOT/USDT:USDT"])
    assert collector.url.startswith("wss://fstream.binance.com/market/stream?streams=")
    assert "btcusdt@aggTrade" in collector.url
    assert "dotusdt@aggTrade" in collector.url


def test_live_long_adverse_acceptance_matches_frozen_classifier():
    collector = AggTradeCollector(["BTC/USDT:USDT"])
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    collector._connected_since = start - timedelta(seconds=5)
    rows = [
        (1, 100.00, start + timedelta(seconds=1)),
        (2, 99.96, start + timedelta(seconds=30)),
        (3, 99.90, start + timedelta(seconds=60)),
        (4, 99.89, start + timedelta(seconds=61)),
    ]
    for agg_id, price, ts in rows:
        collector._on_message(
            None,
            _payload(
                agg_id=agg_id,
                symbol="BTCUSDT",
                price=price,
                qty=10.0,
                when=ts,
                buyer_maker=True,
            ),
        )
    decision = collector.decide(
        "BTC/USDT:USDT",
        side="long",
        start=start,
        window_seconds=60,
    )
    assert decision.orderflow_class == "adverse_acceptance"
    assert decision.veto is True
    assert decision.reference_price == pytest.approx(99.90)
    assert decision.features["of_60s_side_imbalance"] == pytest.approx(-1.0)
    assert decision.features["of_60s_side_return_bps"] < -5.0


def test_disconnect_inside_window_invalidates_coverage():
    collector = AggTradeCollector(["BTC/USDT:USDT"])
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=2)
    collector._connected_since = start + timedelta(seconds=10)
    collector._on_message(
        None,
        _payload(
            agg_id=1,
            symbol="BTCUSDT",
            price=100.0,
            qty=1.0,
            when=start + timedelta(seconds=61),
            buyer_maker=False,
        ),
    )
    assert collector.has_complete_window(
        "BTC/USDT:USDT",
        start,
        start + timedelta(seconds=60),
    ) is False


def test_armed_signal_recovers_until_flow_decision(tmp_path):
    journal = ForwardJournal(str(tmp_path / "forward.jsonl"))
    start = datetime(2026, 10, 21, 0, 0, tzinfo=timezone.utc)
    arm = create_armed_signal(
        symbol="BTC/USDT:USDT",
        side="long",
        risk_per_trade=0.0075,
        signal_time="2026-10-20T23:00:00+00:00",
        decision_start=start,
        signal_close=100.0,
        atr=2.0,
        z_score=-2.1,
        shock1h_score=-2.0,
    )
    journal_arm(journal, arm, config_sha256="abc")
    recovered = recover_armed_signals(journal, config_sha256="abc")
    assert list(recovered) == [arm.key]

    journal_flow_decision(
        journal,
        arm,
        config_sha256="abc",
        flow=None,
        flow_class="missing",
        veto=True,
        reason="test",
    )
    assert recover_armed_signals(journal, config_sha256="abc") == {}
