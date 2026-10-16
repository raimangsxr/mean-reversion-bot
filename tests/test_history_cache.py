import pandas as pd

import src.data.fetcher as fetcher


def test_history_cache_is_incremental(monkeypatch, tmp_path):
    calls = []

    def fake_history(symbol, timeframe, start, end, limit=1000):
        calls.append((start, end))
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        idx = pd.date_range(start_ts, end_ts - pd.Timedelta(hours=1), freq="1h")
        return pd.DataFrame(
            {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
            index=idx,
        )

    monkeypatch.setattr(fetcher, "fetch_binance_history", fake_history)
    start = "2026-01-01T00:00:00Z"
    end = "2026-01-01T05:00:00Z"
    first = fetcher.fetch_binance_history_cached(
        "BTC/USDT:USDT", "1h", start, end, str(tmp_path)
    )
    assert len(first) == 5
    assert len(calls) == 1

    second = fetcher.fetch_binance_history_cached(
        "BTC/USDT:USDT", "1h", start, end, str(tmp_path)
    )
    assert len(second) == 5
    assert len(calls) == 1  # no full-history redownload

    extended = fetcher.fetch_binance_history_cached(
        "BTC/USDT:USDT", "1h", start, "2026-01-01T07:00:00Z", str(tmp_path)
    )
    assert len(extended) == 7
    assert len(calls) == 2
    assert pd.Timestamp(calls[-1][0]) == pd.Timestamp("2026-01-01T05:00:00Z")


def test_history_cache_accepts_and_remembers_prelisting_empty_prefix(monkeypatch, tmp_path):
    symbol = "SOL/USDT:USDT"
    timeframe = "1h"
    cache_path = tmp_path / "sol_usdt_usdt_1h.csv"
    idx = pd.date_range("2020-09-14T07:00:00Z", periods=3, freq="1h")
    pd.DataFrame(
        {
            "timestamp": idx,
            "open": 1.0,
            "high": 1.0,
            "low": 1.0,
            "close": 1.0,
            "volume": 1.0,
        }
    ).to_csv(cache_path, index=False)

    calls = []

    def fake_history(symbol, timeframe, start, end, limit=1000, *, allow_empty=False):
        calls.append((start, end, allow_empty))
        assert allow_empty is True
        return pd.DataFrame(
            columns=fetcher.OHLCV_COLUMNS[1:],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    monkeypatch.setattr(fetcher, "fetch_binance_history", fake_history)

    start = "2020-01-01T00:00:00Z"
    end = "2020-09-14T10:00:00Z"
    first = fetcher.fetch_binance_history_cached(symbol, timeframe, start, end, str(tmp_path))
    assert len(first) == 3
    assert len(calls) == 1
    assert calls[0][2] is True

    # La segunda ejecución no vuelve a sondear el tramo anterior al listing.
    second = fetcher.fetch_binance_history_cached(symbol, timeframe, start, end, str(tmp_path))
    assert len(second) == 3
    assert len(calls) == 1

    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")
    assert meta_path.exists()
