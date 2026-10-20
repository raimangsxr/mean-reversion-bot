from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.orderflow_causal import (
    build_causal_entry_candidates,
    decision_reference_price,
    resolve_delayed_entry,
)


def _minute_cache(start: str = "2025-01-01T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(start, periods=8, freq="1min")
    df = pd.DataFrame(
        {
            "open": [100.0] * 8,
            "high": [100.01] * 8,
            "low": [99.99] * 8,
            "close": [100.0] * 8,
            "volume": [1.0] * 8,
        },
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def _aggtrades(start: str, prices: list[float], *, buyer_maker: bool) -> pd.DataFrame:
    ts = pd.date_range(pd.Timestamp(start) + pd.Timedelta(seconds=5), periods=len(prices), freq="25s")
    qty = [10.0] * len(prices)
    return pd.DataFrame(
        {
            "agg_trade_id": range(1, len(prices) + 1),
            "price": prices,
            "quantity": qty,
            "first_trade_id": range(101, 101 + len(prices)),
            "last_trade_id": range(101, 101 + len(prices)),
            "timestamp": ts,
            "buyer_maker": [buyer_maker] * len(prices),
            "quote_notional": [p * q for p, q in zip(prices, qty, strict=True)],
        }
    )


def _signals() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=2, freq="1h")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0],
            "high": [101.0, 101.0],
            "low": [99.0, 99.0],
            "close": [100.0, 100.0],
            "volume": [1.0, 1.0],
            "atr": [1.0, 1.0],
            "entry_long": [True, False],
            "entry_short": [False, False],
            "exit_long_mean": [False, False],
            "exit_short_mean": [False, False],
        },
        index=idx,
    )


def test_delayed_limit_ignores_predecision_minute_and_fills_after_60s():
    minute = _minute_cache()
    minute.loc[pd.Timestamp("2025-01-01T00:00:00Z"), "low"] = 99.0
    minute.loc[pd.Timestamp("2025-01-01T00:01:00Z"), "low"] = 99.96

    fill = resolve_delayed_entry(
        minute,
        entry_hour=pd.Timestamp("2025-01-01T00:00:00Z"),
        side="long",
        reference_price=100.0,
        market_slippage_bps=2.0,
    )
    assert fill is not None
    assert fill.timestamp == pd.Timestamp("2025-01-01T00:01:00Z")
    assert fill.fill_price == pytest.approx(99.97)
    assert fill.liquidity == "maker"
    assert fill.route == "delay60_limit_maker"


def test_delayed_short_limit_is_exact_mirror():
    minute = _minute_cache()
    minute.loc[pd.Timestamp("2025-01-01T00:00:00Z"), "high"] = 101.0
    minute.loc[pd.Timestamp("2025-01-01T00:01:00Z"), "high"] = 100.04

    fill = resolve_delayed_entry(
        minute,
        entry_hour=pd.Timestamp("2025-01-01T00:00:00Z"),
        side="short",
        reference_price=100.0,
        market_slippage_bps=2.0,
    )
    assert fill is not None
    assert fill.timestamp == pd.Timestamp("2025-01-01T00:01:00Z")
    assert fill.fill_price == pytest.approx(100.03)
    assert fill.liquidity == "maker"


def test_delayed_fallback_occurs_at_original_t_plus_7m_deadline():
    minute = _minute_cache()
    minute["low"] = 99.99
    minute.loc[pd.Timestamp("2025-01-01T00:07:00Z"), "open"] = 100.02

    fill = resolve_delayed_entry(
        minute,
        entry_hour=pd.Timestamp("2025-01-01T00:00:00Z"),
        side="long",
        reference_price=100.0,
        market_slippage_bps=2.0,
    )
    assert fill is not None
    assert fill.timestamp == pd.Timestamp("2025-01-01T00:07:00Z")
    assert fill.raw_price == pytest.approx(100.02)
    assert fill.fill_price == pytest.approx(100.02 * 1.0002)
    assert fill.route == "delay60_market_fallback_conditional"


def test_delayed_fallback_respects_frozen_no_chase_cap():
    minute = _minute_cache()
    minute["low"] = 99.99
    minute.loc[pd.Timestamp("2025-01-01T00:07:00Z"), "open"] = 100.06

    fill = resolve_delayed_entry(
        minute,
        entry_hour=pd.Timestamp("2025-01-01T00:00:00Z"),
        side="long",
        reference_price=100.0,
        market_slippage_bps=2.0,
    )
    assert fill is None


def test_adverse_acceptance_is_vetoed_but_delay_control_keeps_same_fill():
    signals = _signals()
    minute = _minute_cache("2025-01-01T01:00:00Z")
    minute.loc[pd.Timestamp("2025-01-01T01:01:00Z"), "low"] = 99.85
    flow = _aggtrades("2025-01-01T01:00:00Z", [100.0, 99.90, 99.89], buyer_maker=True)

    built = build_causal_entry_candidates(
        signals=signals,
        minute_cache=minute,
        aggtrades=flow,
        side="long",
        market_slippage_bps=2.0,
    )
    assert len(built.delay_only_candidates) == 1
    assert len(built.orderflow_candidates) == 0
    assert built.decisions.iloc[0]["orderflow_class"] == "adverse_acceptance"
    assert bool(built.decisions.iloc[0]["orderflow_veto"]) is True
    assert bool(built.decisions.iloc[0]["delayed_fill"]) is True


def test_adverse_absorption_is_not_vetoed():
    signals = _signals()
    minute = _minute_cache("2025-01-01T01:00:00Z")
    minute.loc[pd.Timestamp("2025-01-01T01:01:00Z"), "low"] = 99.95
    flow = _aggtrades("2025-01-01T01:00:00Z", [100.0, 99.995, 99.99], buyer_maker=True)

    built = build_causal_entry_candidates(
        signals=signals,
        minute_cache=minute,
        aggtrades=flow,
        side="long",
        market_slippage_bps=2.0,
    )
    assert built.decisions.iloc[0]["orderflow_class"] == "adverse_absorption"
    assert len(built.delay_only_candidates) == 1
    assert len(built.orderflow_candidates) == 1
    assert built.delay_only_candidates[0].fill == built.orderflow_candidates[0].fill


def test_missing_flow_uses_boundary_minute_open_only_as_reference():
    minute = _minute_cache("2025-01-01T01:00:00Z")
    minute.loc[pd.Timestamp("2025-01-01T01:01:00Z"), "open"] = 100.25
    reference, source = decision_reference_price(
        pd.DataFrame(),
        decision_start=pd.Timestamp("2025-01-01T01:00:00Z"),
        minute_cache=minute,
    )
    assert reference == pytest.approx(100.25)
    assert source == "minute_open_fallback"


def test_cache_extension_preserves_existing_windows(monkeypatch, tmp_path: Path):
    import src.research.orderflow as module

    symbol = "BTC/USDT:USDT"
    day = pd.Timestamp("2025-01-01T00:00:00Z").date()
    cache_path, meta_path = module._cache_paths(tmp_path, symbol, day)
    old = module.FlowWindow("2025-01-01T01:00:00Z", "2025-01-01T01:07:00Z")
    new = module.FlowWindow("2025-01-01T05:00:00Z", "2025-01-01T05:01:00Z")
    module.empty_aggtrades().to_csv(cache_path, index=False, compression="gzip")
    meta_path.write_text(
        json.dumps(
            {
                "windows": [{"start": old.start.isoformat(), "end": old.end.isoformat()}],
                "archive_url": "https://example.invalid/archive.zip",
                "zip_sha256": "old",
            }
        )
    )

    captured: list[module.FlowWindow] = []

    def fake_download(url: str, destination: Path, timeout: int = 60):
        destination.write_bytes(b"placeholder")

    def fake_extract(path: Path, windows: list[module.FlowWindow]):
        captured.extend(windows)
        return module.empty_aggtrades()

    monkeypatch.setattr(module, "_download", fake_download)
    monkeypatch.setattr(module, "_verify_checksum", lambda path, url: "a" * 64)
    monkeypatch.setattr(module, "_extract_intervals_from_zip", fake_extract)

    module.ensure_aggtrade_windows(symbol, [new], cache_dir=tmp_path)
    assert len(captured) == 2
    assert captured[0] == old
    assert captured[1] == new
