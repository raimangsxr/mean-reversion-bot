from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import re
from typing import Optional

try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover
    ccxt = None
import pandas as pd

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


_PUBLIC_FUTURES_EXCHANGE = None


def _make_public_futures_exchange():
    """Cliente público Binance USD-M persistente y reutilizable."""
    global _PUBLIC_FUTURES_EXCHANGE

    if ccxt is None:
        raise RuntimeError("CCXT no esta instalado. Ejecuta pip install -r requirements.txt")
    if _PUBLIC_FUTURES_EXCHANGE is None:
        exchange = ccxt.binance(
            {
                "enableRateLimit": True,
                "timeout": 15_000,
                "options": {"defaultType": "future"},
            }
        )
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
    if ccxt is not None:
        seconds = ccxt.Exchange.parse_timeframe(timeframe)
        return pd.Timedelta(seconds=seconds)
    # Fallback minimo para tests/utilidades sin CCXT instalado.
    try:
        delta = pd.Timedelta(timeframe)
    except ValueError as exc:
        raise ValueError(f"Timeframe no compatible: {timeframe}") from exc
    if delta <= pd.Timedelta(0):
        raise ValueError("El timeframe debe ser positivo")
    return delta


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
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = _to_dataframe(ohlcv)
    return keep_closed_candles(df, timeframe) if closed_only else df


def fetch_binance_history(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    limit: int = 1000,
    *,
    allow_empty: bool = False,
) -> pd.DataFrame:
    """Descarga OHLCV Futures paginado entre start (incl.) y end (excl.).

    ``allow_empty`` se usa únicamente por la cache incremental al sondear un
    prefijo que puede estar antes del listing del contrato. Los callers normales
    siguen fallando si Binance no devuelve ninguna vela.
    """
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
        if allow_empty:
            return pd.DataFrame(
                columns=OHLCV_COLUMNS[1:],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )
        raise RuntimeError("Binance Futures no devolvió velas para el periodo solicitado.")

    return keep_closed_candles(df, timeframe)



def _history_cache_slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()


def _history_cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _load_history_cache_meta(cache_path: Path) -> dict:
    meta_path = _history_cache_meta_path(cache_path)
    if not meta_path.exists():
        return {}
    try:
        value = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _save_history_cache_meta(cache_path: Path, meta: dict) -> None:
    meta_path = _history_cache_meta_path(cache_path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def _empty_prefix_is_known(meta: dict, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> bool:
    try:
        known_start = pd.Timestamp(meta["empty_prefix_start"])
        known_end = pd.Timestamp(meta["empty_prefix_end"])
    except (KeyError, TypeError, ValueError):
        return False
    known_start = known_start.tz_localize("UTC") if known_start.tzinfo is None else known_start.tz_convert("UTC")
    known_end = known_end.tz_localize("UTC") if known_end.tzinfo is None else known_end.tz_convert("UTC")
    return known_start <= start_ts and known_end >= end_ts


def fetch_binance_history_cached(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cache_dir: str,
    limit: int = 1000,
) -> pd.DataFrame:
    """Incremental OHLCV cache for research history.

    The first call downloads the requested history. Later calls only request
    closed candles missing at the head/tail, avoiding six full-history downloads
    on every pullback-limit run (BTC/ETH/SOL x 1h/4h).
    """
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    delta = timeframe_to_timedelta(timeframe)
    cache_path = Path(cache_dir) / f"{_history_cache_slug(symbol)}_{_history_cache_slug(timeframe)}.csv"

    cache = pd.DataFrame(columns=OHLCV_COLUMNS[1:], index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    if cache_path.exists():
        try:
            raw = pd.read_csv(cache_path)
            if not raw.empty and "timestamp" in raw.columns:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
                cache = (
                    raw.dropna(subset=["timestamp"])
                    .drop_duplicates(subset="timestamp", keep="last")
                    .sort_values("timestamp")
                    .set_index("timestamp")
                )
        except pd.errors.EmptyDataError:
            pass

    meta = _load_history_cache_meta(cache_path)
    meta_changed = False
    cache_changed = False
    pieces = [cache] if not cache.empty else []

    if cache.empty:
        # En la primera descarga Binance suele saltar directamente al primer
        # candle listado aunque BACKTEST_START sea anterior al listing.
        downloaded = fetch_binance_history(
            symbol, timeframe, start_ts.isoformat(), end_ts.isoformat(), limit=limit
        )
        pieces.append(downloaded)
        cache_changed = True
    else:
        first = cache.index.min()
        last = cache.index.max()
        if first > start_ts:
            prefix_end = min(first, end_ts)
            if start_ts < prefix_end and not _empty_prefix_is_known(meta, start_ts, prefix_end):
                # Un contrato puede no existir todavía en BACKTEST_START (p.ej.
                # SOL USD-M en 2020). Ese prefijo vacío es válido y no debe
                # abortar el backtest. Se memoriza para no pedirlo en cada run.
                prefix = fetch_binance_history(
                    symbol,
                    timeframe,
                    start_ts.isoformat(),
                    prefix_end.isoformat(),
                    limit=limit,
                    allow_empty=True,
                )
                if prefix.empty:
                    meta["empty_prefix_start"] = start_ts.isoformat()
                    meta["empty_prefix_end"] = prefix_end.isoformat()
                    meta_changed = True
                else:
                    pieces.append(prefix)
                    cache_changed = True
                    meta.pop("empty_prefix_start", None)
                    meta.pop("empty_prefix_end", None)
                    meta_changed = True

        # next_start candle must itself be closed by end_ts. This avoids calling
        # Binance every run just for the currently-forming candle.
        next_start = last + delta
        if next_start + delta <= end_ts:
            tail = fetch_binance_history(
                symbol, timeframe, next_start.isoformat(), end_ts.isoformat(), limit=limit
            )
            pieces.append(tail)
            cache_changed = True

    if cache_changed:
        cache = pd.concat(pieces).sort_index()
        cache = cache[~cache.index.duplicated(keep="last")]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        out = cache.reset_index().rename(columns={"index": "timestamp"})
        out.to_csv(cache_path, index=False)
    elif pieces:
        cache = pieces[0]

    if meta_changed:
        _save_history_cache_meta(cache_path, meta)

    sliced = cache.loc[(cache.index >= start_ts) & (cache.index < end_ts)]
    sliced = keep_closed_candles(sliced, timeframe, now=end_ts)
    if sliced.empty:
        raise RuntimeError(
            f"Cache/history vacio para {symbol} {timeframe} entre {start} y {end}."
        )
    return sliced


def fetch_binance_taker_history(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    limit: int = 1500,
) -> pd.DataFrame:
    """Fetch raw USD-M klines including taker-buy base volume.

    CCXT ``fetch_ohlcv`` intentionally normalizes away Binance-specific kline
    fields. V10 needs taker-buy volume for one pre-registered order-flow signal,
    so this research-only helper reads the public raw futures kline endpoint.
    """
    exchange = _make_public_futures_exchange()
    market = exchange.market(symbol)
    market_id = market.get("id")
    if not market_id:
        raise RuntimeError(f"No se pudo resolver market id para {symbol}")

    since = exchange.parse8601(start)
    end_ts = exchange.parse8601(end)
    if since is None or end_ts is None or since >= end_ts:
        raise ValueError("start/end taker deben ser fechas ISO-8601 validas y ordenadas")

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    cursor = int(since)
    rows: list[dict] = []
    method = getattr(exchange, "fapiPublicGetKlines", None)
    if method is None:
        method = getattr(exchange, "fapi_public_get_klines", None)
    if method is None:
        raise RuntimeError("La version de CCXT no expone Binance USD-M public klines")

    while cursor < end_ts:
        batch = method(
            {
                "symbol": market_id,
                "interval": timeframe,
                "startTime": cursor,
                "endTime": int(end_ts - 1),
                "limit": int(limit),
            }
        )
        if not batch:
            break
        progressed = False
        for item in batch:
            if len(item) < 11:
                continue
            open_ms = int(item[0])
            if open_ms >= end_ts:
                continue
            rows.append(
                {
                    "timestamp": open_ms,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "taker_buy_base_volume": float(item[9]),
                    "taker_buy_quote_volume": float(item[10]),
                    "trades": int(item[8]),
                }
            )
            if open_ms >= cursor:
                cursor = open_ms + timeframe_ms
                progressed = True
        if not progressed:
            break
        if len(batch) < limit:
            break

    if not rows:
        raise RuntimeError("Binance Futures no devolvio klines taker para el periodo solicitado.")
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = (
        df.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    return keep_closed_candles(df, timeframe)


def fetch_binance_taker_history_cached(
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    cache_dir: str,
    limit: int = 1500,
) -> pd.DataFrame:
    """Incremental cache for V10 taker-flow research fields."""
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    start_ts = start_ts.tz_localize("UTC") if start_ts.tzinfo is None else start_ts.tz_convert("UTC")
    end_ts = end_ts.tz_localize("UTC") if end_ts.tzinfo is None else end_ts.tz_convert("UTC")
    delta = timeframe_to_timedelta(timeframe)
    cache_path = Path(cache_dir) / f"{_history_cache_slug(symbol)}_{_history_cache_slug(timeframe)}_taker.csv"
    columns = [
        "open", "high", "low", "close", "volume",
        "taker_buy_base_volume", "taker_buy_quote_volume", "trades",
    ]
    cache = pd.DataFrame(columns=columns, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    if cache_path.exists():
        try:
            raw = pd.read_csv(cache_path)
            if not raw.empty and "timestamp" in raw.columns:
                raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
                cache = (
                    raw.dropna(subset=["timestamp"])
                    .drop_duplicates(subset="timestamp", keep="last")
                    .sort_values("timestamp")
                    .set_index("timestamp")
                )
        except pd.errors.EmptyDataError:
            pass

    pieces = [cache] if not cache.empty else []
    if cache.empty:
        pieces.append(
            fetch_binance_taker_history(
                symbol, timeframe, start_ts.isoformat(), end_ts.isoformat(), limit=limit
            )
        )
    else:
        first = cache.index.min()
        last = cache.index.max()
        if first > start_ts:
            # V10 principal es BTC, por lo que el contrato existe desde el
            # inicio configurado. Si se usa otro activo y el prefijo pre-listing
            # esta vacio, el caller puede ajustar BACKTEST_START.
            pieces.append(
                fetch_binance_taker_history(
                    symbol, timeframe, start_ts.isoformat(), first.isoformat(), limit=limit
                )
            )
        next_start = last + delta
        if next_start + delta <= end_ts:
            pieces.append(
                fetch_binance_taker_history(
                    symbol, timeframe, next_start.isoformat(), end_ts.isoformat(), limit=limit
                )
            )

    merged = pd.concat(pieces).sort_index() if pieces else cache
    merged = merged[~merged.index.duplicated(keep="last")]
    if not merged.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        merged.reset_index().to_csv(cache_path, index=False)
    return merged.loc[(merged.index >= start_ts) & (merged.index < end_ts)]
