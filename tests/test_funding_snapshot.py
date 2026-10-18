import json

import pandas as pd
import pytest

import src.research.engine as engine


def _funding(start="2026-01-01T00:00:00Z", periods=4):
    idx = pd.date_range(start, periods=periods, freq="8h")
    return pd.DataFrame({"funding_rate": [0.0001, -0.0002, 0.0003, 0.0001][:periods]}, index=idx)


def test_funding_snapshot_is_written_and_reused_without_refetch(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "BACKTEST_FUNDING_CACHE_DIR", str(tmp_path))
    calls = []

    def fake_fetch(symbol, start, end):
        calls.append((symbol, start, end))
        return _funding()

    monkeypatch.setattr(engine, "_fetch_funding_range", fake_fetch)
    start = "2026-01-01T00:00:00Z"
    end = "2026-01-02T08:00:00Z"
    first = engine.ensure_funding_cache("BTC/USDT:USDT", start, end, rebuild=True)
    assert len(first) == 4
    assert len(calls) == 1

    info = engine.funding_snapshot_info("BTC/USDT:USDT")
    assert info["rows"] == 4
    assert len(info["values_sha256"]) == 64
    meta = tmp_path / "btc_usdt_usdt.meta.json"
    assert json.loads(meta.read_text())["values_sha256"] == info["values_sha256"]

    calls.clear()
    second = engine.ensure_funding_cache("BTC/USDT:USDT", start, end)
    assert len(second) == 4
    assert calls == []
    assert engine.funding_snapshot_info("BTC/USDT:USDT")["values_sha256"] == info["values_sha256"]


def test_legacy_funding_cache_requires_explicit_rebuild(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "BACKTEST_FUNDING_CACHE_DIR", str(tmp_path))
    df = _funding()
    df.reset_index(names="timestamp").to_csv(tmp_path / "btc_usdt_usdt.csv", index=False)
    with pytest.raises(RuntimeError, match="--rebuild-funding"):
        engine.ensure_funding_cache(
            "BTC/USDT:USDT", "2026-01-01T00:00:00Z", "2026-01-02T08:00:00Z"
        )


def test_tampered_funding_snapshot_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "BACKTEST_FUNDING_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(engine, "_fetch_funding_range", lambda *args: _funding())
    engine.ensure_funding_cache(
        "BTC/USDT:USDT", "2026-01-01T00:00:00Z", "2026-01-02T08:00:00Z", rebuild=True
    )
    path = tmp_path / "btc_usdt_usdt.csv"
    df = pd.read_csv(path)
    df.loc[0, "funding_rate"] = 0.9
    df.to_csv(path, index=False)
    with pytest.raises(RuntimeError, match="Snapshot funding alterado"):
        engine.ensure_funding_cache(
            "BTC/USDT:USDT", "2026-01-01T00:00:00Z", "2026-01-02T08:00:00Z"
        )
