import pandas as pd

import src.data.fetcher as fetcher


class FakeExchange:
    def market(self, symbol):
        assert symbol == "BTC/USDT:USDT"
        return {"id": "BTCUSDT"}

    def parse8601(self, value):
        return int(pd.Timestamp(value).timestamp() * 1000)

    def parse_timeframe(self, timeframe):
        assert timeframe == "1h"
        return 3600

    def fapiPublicGetKlines(self, params):
        ts = int(params["startTime"])
        if ts >= int(pd.Timestamp("2025-01-01T02:00:00Z").timestamp() * 1000):
            return []
        rows = []
        for _ in range(2):
            rows.append([
                ts, "100", "102", "99", "101", "10",
                ts + 3_599_999, "1000", 50, "4", "400", "0",
            ])
            ts += 3_600_000
        return rows


def test_fetch_binance_taker_history_preserves_binance_specific_fields(monkeypatch):
    monkeypatch.setattr(fetcher, "_make_public_futures_exchange", lambda: FakeExchange())
    df = fetcher.fetch_binance_taker_history(
        "BTC/USDT:USDT", "1h", "2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z"
    )
    assert len(df) == 2
    assert list(df["taker_buy_base_volume"]) == [4.0, 4.0]
    assert list(df["taker_buy_quote_volume"]) == [400.0, 400.0]
    assert list(df["trades"]) == [50, 50]
