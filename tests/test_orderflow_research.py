from __future__ import annotations

from pathlib import Path
import zipfile

import pandas as pd
import pytest

from src.research.orderflow import (
    FlowWindow,
    _extract_intervals_from_zip,
    classify_primary_window,
    compute_flow_features,
    market_id,
)


def _trades(side_bias: str = "buy") -> pd.DataFrame:
    ts = pd.date_range("2025-01-01T00:00:05Z", periods=6, freq="10s")
    buyer_maker = [False, False, False, True, True, False]
    if side_bias == "sell":
        buyer_maker = [not value for value in buyer_maker]
    prices = [100.0, 100.1, 100.2, 100.15, 100.10, 100.3]
    qty = [2.0, 1.0, 3.0, 1.0, 1.0, 4.0]
    return pd.DataFrame(
        {
            "agg_trade_id": range(1, 7),
            "price": prices,
            "quantity": qty,
            "first_trade_id": range(11, 17),
            "last_trade_id": range(11, 17),
            "timestamp": ts,
            "buyer_maker": buyer_maker,
            "quote_notional": [p * q for p, q in zip(prices, qty, strict=True)],
        }
    )


def test_market_id_from_ccxt_perpetual_symbol():
    assert market_id("BTC/USDT:USDT") == "BTCUSDT"


def test_side_adjusted_flow_is_symmetric():
    trades = _trades("buy")
    long = compute_flow_features(
        trades, start=pd.Timestamp("2025-01-01T00:00:00Z"), side="long", window_seconds=60
    )
    short = compute_flow_features(
        trades, start=pd.Timestamp("2025-01-01T00:00:00Z"), side="short", window_seconds=60
    )
    assert long["of_60s_side_imbalance"] == pytest.approx(-short["of_60s_side_imbalance"])
    assert long["of_60s_side_return_bps"] == pytest.approx(-short["of_60s_side_return_bps"])


def test_primary_classification_distinguishes_acceptance_and_absorption():
    adverse_acceptance = {
        "of_60s_trades": 20,
        "of_60s_side_imbalance": -0.45,
        "of_60s_side_return_bps": -8.0,
    }
    absorption = {
        "of_60s_trades": 20,
        "of_60s_side_imbalance": -0.45,
        "of_60s_side_return_bps": -1.0,
    }
    support = {
        "of_60s_trades": 20,
        "of_60s_side_imbalance": 0.35,
        "of_60s_side_return_bps": 3.0,
    }
    assert classify_primary_window(adverse_acceptance) == "adverse_acceptance"
    assert classify_primary_window(absorption) == "adverse_absorption"
    assert classify_primary_window(support) == "aligned_support"


def test_archive_parser_filters_windows_and_skips_header(tmp_path: Path):
    archive = tmp_path / "BTCUSDT-aggTrades-2025-01-01.zip"
    csv_body = "\n".join(
        [
            "agg_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,buyer_maker",
            "1,100,1,10,10,1735689600000,false",
            "2,101,2,11,11,1735689610000,true",
            "3,102,3,12,12,1735689700000,false",
        ]
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BTCUSDT-aggTrades-2025-01-01.csv", csv_body)

    window = FlowWindow("2025-01-01T00:00:05Z", "2025-01-01T00:00:20Z")
    df = _extract_intervals_from_zip(archive, [window])
    assert list(df["agg_trade_id"]) == [2]
    assert bool(df.iloc[0]["buyer_maker"]) is True
    assert float(df.iloc[0]["quote_notional"]) == pytest.approx(202.0)


def test_generic_replay_build_signals_uses_frozen_shock_filter(monkeypatch):
    import numpy as np
    import src.data.fetcher as fetcher
    import src.research.engine as engine

    idx_1h = pd.date_range("2024-01-01T00:00:00Z", periods=300, freq="1h")
    idx_4h = pd.date_range("2023-12-01T00:00:00Z", periods=300, freq="4h")

    def frame(idx):
        close = pd.Series(100.0 + np.arange(len(idx)) * 0.01, index=idx)
        return pd.DataFrame(
            {"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0},
            index=idx,
        )

    requested = []

    def fake_history(symbol, timeframe, start, end, cache_dir):
        requested.append((symbol, timeframe, start, end, cache_dir))
        return frame(idx_1h if timeframe == "1h" else idx_4h)

    captured = {}

    class FakeStrategy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def calculate_signals(self, df, regime_df=None):
            return df.assign(atr=1.0, entry_long=False, entry_short=False)

    monkeypatch.setattr(fetcher, "fetch_binance_history_cached", fake_history)
    monkeypatch.setattr(engine, "MeanReversionStrategy", FakeStrategy)
    out = engine.build_signals(
        "BTC/USDT:USDT",
        start="2022-01-01T00:00:00Z",
        end="2026-10-19T22:57:41Z",
    )
    assert len(out) == 300
    assert captured["shock1h_threshold"] == pytest.approx(2.175)
    assert captured["return_shock_vol_window"] == 100
    assert {row[1] for row in requested} == {"1h", "4h"}
    assert all(row[2] == "2022-01-01T00:00:00Z" for row in requested)
    assert all(row[3] == "2026-10-19T22:57:41Z" for row in requested)
