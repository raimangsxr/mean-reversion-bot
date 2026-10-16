from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import ccxt
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


_PUBLIC_FUTURES_EXCHANGE: Optional[ccxt.binance] = None


def _make_public_futures_exchange() -> ccxt.binance:
    """
    Cliente público Binance USD-M reutilizable.

    Evitamos crear un exchange y ejecutar load_markets() en cada poll.
    Las consultas públicas son siempre contra el mercado real incluso
    cuando la ejecución está en Demo Trading.
    """
    global _PUBLIC_FUTURES_EXCHANGE

    if _PUBLIC_FUTURES_EXCHANGE is None:
        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "timeout": 15_000,
                "options": {
                    "defaultType": "future",
                },
            }
        )

        # Solo afectan a fallos HTTP/red de las consultas públicas.
        exchange.options["maxRetriesOnFailure"] = 2
        exchange.options["maxRetriesOnFailureDelay"] = 1_000

        exchange.load_markets()

        _PUBLIC_FUTURES_EXCHANGE = exchange

    return _PUBLIC_FUTURES_EXCHANGE


def _to_dataframe(ohlcv: list[list[float]]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv, columns=OHLCV_COLUMNS)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp", keep="last")
    return df.sort_values("timestamp").set_index("timestamp")


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    seconds = ccxt.Exchange.parse_timeframe(timeframe)
    return pd.Timedelta(seconds=seconds)


def keep_closed_candles(
    df: pd.DataFrame,
    timeframe: str,
    now: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Descarta la vela que todavía está formándose."""
    if df.empty:
        return df.copy()

    if now is None:
        now = pd.Timestamp(datetime.now(timezone.utc))
    elif now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    duration = timeframe_to_timedelta(timeframe)
    return df[(df.index + duration) <= now].copy()


def fetch_binance_klines(
    symbol: str = "BTC/USDT:USDT",
    timeframe: str = "1h",
    limit: int = 500,
    closed_only: bool = True,
) -> pd.DataFrame:
    """Descarga velas recientes del contrato USD-M Futures."""
    exchange = _make_public_futures_exchange()
    ohlcv = exchange.fetch_ohlcv(
        symbol,
        timeframe=timeframe,
        limit=limit,
    )
    df = _to_dataframe(ohlcv)
    return keep_closed_candles(df, timeframe) if closed_only else df


def fetch_binance_history(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Descarga OHLCV Futures paginado entre start (incl.) y end (excl.)."""
    exchange = _make_public_futures_exchange()
    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000

    since = exchange.parse8601(start)
    end_ts = exchange.parse8601(end)
    if since is None or end_ts is None:
        raise ValueError("BACKTEST_START/BACKTEST_END deben ser fechas ISO-8601 válidas.")
    if since >= end_ts:
        raise ValueError("La fecha de inicio debe ser anterior a la fecha de fin.")

    rows: list[list[float]] = []
    cursor = since

    while cursor < end_ts:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=cursor,
            limit=limit,
        )
        if not batch:
            break

        rows.extend(row for row in batch if row[0] < end_ts)

        last_ts = int(batch[-1][0])
        next_cursor = last_ts + timeframe_ms
        if next_cursor <= cursor:
            raise RuntimeError("La paginación OHLCV no avanzó; se aborta.")
        cursor = next_cursor

        if last_ts >= end_ts - timeframe_ms:
            break

    df = _to_dataframe(rows)
    if df.empty:
        raise RuntimeError("Binance Futures no devolvió velas para el periodo solicitado.")

    return keep_closed_candles(df, timeframe)
