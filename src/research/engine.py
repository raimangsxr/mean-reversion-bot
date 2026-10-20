from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import re
from typing import Iterable, Optional

import numpy as np
import pandas as pd

try:  # Permite ejecutar tests unitarios de la logica sin CCXT instalado.
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover - solo entornos de test/minimos.
    ccxt = None

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    BACKTEST_DD_LIMIT,
    BACKTEST_END,
    BACKTEST_FIRST_TEST_YEAR,
    BACKTEST_FUNDING_CACHE_DIR,
    BACKTEST_HISTORY_CACHE_DIR,
    BACKTEST_MAKER_FEE,
    BACKTEST_MAX_CONCURRENT_POSITIONS,
    BACKTEST_MAX_PORTFOLIO_RISK_PCT,
    BACKTEST_MIN_TRAIN_TRADES,
    BACKTEST_MINUTE_CACHE_DIR,
    BACKTEST_PARTIAL_FRACTION,
    BACKTEST_REQUIRE_FUNDING,
    BACKTEST_START,
    BACKTEST_TAKER_FEE,
    BACKTEST_TP_CROSS_TICKS,
    CASH_BUFFER_PCT,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    ENTRY_MODE,
    INITIAL_CAPITAL,
    LEVERAGE,
    LOOKBACK_WINDOW,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    RETURN_SHOCK_VOL_WINDOW,
    SHOCK1H_THRESHOLD,
    TIMEFRAME,
    TREND_WINDOW,
    Z_THRESHOLD,
    FUNDING_MAX_GAP_HOURS,
)
from src.data.fetcher import timeframe_to_timedelta
from src.risk.manager import RiskManager
from src.strategy.mean_reversion import MeanReversionStrategy


MINUTE_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class EntryPolicy:
    limit_offset_bps: float = ENTRY_LIMIT_OFFSET_BPS
    wait_minutes: int = ENTRY_LIMIT_WAIT_SECONDS // 60
    max_chase_bps: float = ENTRY_FALLBACK_CAP_BPS


@dataclass(frozen=True)
class ExitPolicy:
    name: str
    rr: float
    use_fixed_tp: bool = True
    mean_exit: bool = False
    time_stop_hours: Optional[int] = None
    break_even_trigger_r: Optional[float] = None
    partial_trigger_r: Optional[float] = None
    partial_fraction: float = BACKTEST_PARTIAL_FRACTION
    partial_move_stop_to_be: bool = False


@dataclass
class EntryFill:
    timestamp: pd.Timestamp
    side: str
    raw_price: float
    fill_price: float
    fee_rate: float
    slippage_per_unit: float
    liquidity: str
    route: str
    baseline_open: float

    @property
    def improvement_bps(self) -> float:
        if self.baseline_open <= 0:
            return float("nan")
        if self.side == "long":
            return (self.baseline_open - self.fill_price) / self.baseline_open * 10_000.0
        return (self.fill_price - self.baseline_open) / self.baseline_open * 10_000.0


@dataclass(frozen=True)
class EntryCandidate:
    signal_time: pd.Timestamp
    entry_hour: pd.Timestamp
    side: str
    signal_atr: float
    fill: EntryFill


@dataclass(frozen=True)
class PathArrays:
    index: pd.DatetimeIndex
    opens: np.ndarray
    highs: np.ndarray
    lows: np.ndarray
    closes: np.ndarray
    exit_long_mean: np.ndarray
    exit_short_mean: np.ndarray


@dataclass(frozen=True)
class FundingSeries:
    index_ns: np.ndarray
    prefix_value: np.ndarray

    def interval_sum(self, start_exclusive: pd.Timestamp, end_inclusive: pd.Timestamp) -> float:
        if len(self.index_ns) == 0 or end_inclusive <= start_exclusive:
            return 0.0
        start_ns = int(pd.Timestamp(start_exclusive).value)
        end_ns = int(pd.Timestamp(end_inclusive).value)
        left = int(np.searchsorted(self.index_ns, start_ns, side="right"))
        right = int(np.searchsorted(self.index_ns, end_ns, side="right"))
        return float(self.prefix_value[right] - self.prefix_value[left])


@dataclass
class TradeOpportunity:
    symbol: str
    side: str
    exit_policy: str
    rr: float
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    raw_entry_price: float
    stop_distance: float
    stop_fraction: float
    exit_reason: str
    entry_liquidity: str
    entry_route: str
    entry_delay_minutes: float
    entry_improvement_bps: float
    gross_pnl_per_unit: float
    fee_per_unit: float
    slippage_per_unit: float
    funding_pnl_per_unit: float
    net_pnl_per_unit: float
    mfe_r: float
    mae_r: float
    holding_hours: float
    partial_taken: bool
    partial_trigger_r: Optional[float]
    break_even_armed: bool


@dataclass(frozen=True)
class PortfolioConfig:
    exit_policy: str
    rr: float
    risk_per_trade: float
    max_position_pct: float
    side_mode: str

    @property
    def name(self) -> str:
        return (
            f"{self.exit_policy}|rr={self.rr:g}|risk={self.risk_per_trade:.4f}|"
            f"maxpos={self.max_position_pct:.2f}|{self.side_mode}"
        )


@dataclass
class SizedTrade:
    config: str
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    exit_reason: str
    notional: float
    amount: float
    initial_risk: float
    risk_utilization: float
    limiting_factor: str
    pnl: float
    return_on_equity_at_entry: float
    funding_pnl: float
    fees: float
    slippage_cost: float


@dataclass
class PortfolioResult:
    config: PortfolioConfig
    start: pd.Timestamp
    end: pd.Timestamp
    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    trades: pd.DataFrame
    skipped_portfolio_risk: int
    skipped_concurrency: int
    skipped_margin: int


class OpportunityBook:
    """Pre-index opportunities by path/side so thousands of portfolio sims do no filtering."""

    def __init__(self, opportunities: pd.DataFrame):
        df = opportunities
        if not df.empty and not isinstance(df["entry_time"].dtype, pd.DatetimeTZDtype):
            df = df.copy()
            df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
            df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
        if not df.empty:
            df = df.sort_values(["entry_time", "symbol"], kind="stable").reset_index(drop=True)
        self.frame = df
        self._groups: dict[tuple[str, float, str], pd.DataFrame] = {}
        self._symbol_groups: dict[tuple[str, float, str, tuple[str, ...]], pd.DataFrame] = {}
        if df.empty:
            return
        for (policy, rr), group in df.groupby(["exit_policy", "rr"], sort=False):
            key_rr = round(float(rr), 10)
            self._groups[(str(policy), key_rr, "long_short")] = group
            self._groups[(str(policy), key_rr, "long_only")] = group[group["side"].eq("long")]

    def for_config(
        self, config: PortfolioConfig, symbols: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        key = (config.exit_policy, round(float(config.rr), 10), config.side_mode)
        group = self._groups.get(key)
        if group is None:
            return self.frame.iloc[0:0]
        if symbols is None:
            return group
        symbol_key = tuple(sorted(str(s) for s in symbols))
        cache_key = (*key, symbol_key)
        cached = self._symbol_groups.get(cache_key)
        if cached is None:
            cached = group[group["symbol"].isin(symbol_key)]
            self._symbol_groups[cache_key] = cached
        return cached


# -----------------------------------------------------------------------------
# Exit profile parsing.
# -----------------------------------------------------------------------------
def parse_exit_profile(name: str, rr: float) -> ExitPolicy:
    value = name.strip().lower()
    if value == "fixed_rr":
        return ExitPolicy(value, rr)
    if value == "mean":
        return ExitPolicy(value, rr, use_fixed_tp=False, mean_exit=True)
    if value == "hybrid_rr_mean":
        return ExitPolicy(value, rr, mean_exit=True)

    m = re.fullmatch(r"time_(\d+)h", value)
    if m:
        return ExitPolicy(value, rr, time_stop_hours=int(m.group(1)))

    m = re.fullmatch(r"be_([0-9]+(?:\.[0-9]+)?)r", value)
    if m:
        return ExitPolicy(value, rr, break_even_trigger_r=float(m.group(1)))

    m = re.fullmatch(r"partial_([0-9]+(?:\.[0-9]+)?)r_be", value)
    if m:
        return ExitPolicy(
            value,
            rr,
            partial_trigger_r=float(m.group(1)),
            partial_move_stop_to_be=True,
        )

    if value == "hybrid_rr_mean_time24":
        return ExitPolicy(value, rr, mean_exit=True, time_stop_hours=24)

    raise ValueError(f"Perfil de salida desconocido: {name}")


# -----------------------------------------------------------------------------
# Entrada congelada y generalizada simetricamente a LONG/SHORT.
# -----------------------------------------------------------------------------
def resolve_entry(
    minute_hour: pd.DataFrame,
    *,
    side: str,
    signal_close: float,
    market_slippage_bps: float,
    policy: EntryPolicy = EntryPolicy(),
) -> Optional[EntryFill]:
    if side not in {"long", "short"}:
        raise ValueError("side debe ser long o short")
    if minute_hour.empty:
        return None
    minute_hour = minute_hour.sort_index()
    if len(minute_hour) <= policy.wait_minutes:
        return None

    baseline_open = float(minute_hour.iloc[0]["open"])
    if side == "long":
        limit_price = signal_close * (1.0 - policy.limit_offset_bps / 10_000.0)
        resting_possible = baseline_open > limit_price
    else:
        limit_price = signal_close * (1.0 + policy.limit_offset_bps / 10_000.0)
        resting_possible = baseline_open < limit_price

    if resting_possible:
        search = minute_hour.iloc[: policy.wait_minutes]
        for ts, row in search.iterrows():
            traversed = (
                float(row["low"]) < limit_price
                if side == "long"
                else float(row["high"]) > limit_price
            )
            if traversed:
                return EntryFill(
                    timestamp=ts,
                    side=side,
                    raw_price=limit_price,
                    fill_price=limit_price,
                    fee_rate=BACKTEST_MAKER_FEE,
                    slippage_per_unit=0.0,
                    liquidity="maker",
                    route="limit_maker",
                    baseline_open=baseline_open,
                )

    fallback_ts = minute_hour.index[0] + pd.Timedelta(minutes=policy.wait_minutes)
    if fallback_ts not in minute_hour.index:
        return None
    raw = float(minute_hour.loc[fallback_ts, "open"])

    if side == "long":
        allowed = raw <= baseline_open * (1.0 + policy.max_chase_bps / 10_000.0)
        fill = raw * (1.0 + market_slippage_bps / 10_000.0)
        slip_per_unit = fill - raw
    else:
        allowed = raw >= baseline_open * (1.0 - policy.max_chase_bps / 10_000.0)
        fill = raw * (1.0 - market_slippage_bps / 10_000.0)
        slip_per_unit = raw - fill
    if not allowed:
        return None

    return EntryFill(
        timestamp=fallback_ts,
        side=side,
        raw_price=raw,
        fill_price=fill,
        fee_rate=BACKTEST_TAKER_FEE,
        slippage_per_unit=slip_per_unit,
        liquidity="taker",
        route="market_fallback_conditional",
        baseline_open=baseline_open,
    )


# -----------------------------------------------------------------------------
# Caches market data/funding.
# -----------------------------------------------------------------------------
def _slug(symbol: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", symbol).strip("_").lower()


def _empty_minute_cache() -> pd.DataFrame:
    return pd.DataFrame(
        columns=MINUTE_COLUMNS,
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )


def load_minute_cache(symbol: str) -> pd.DataFrame:
    path = Path(BACKTEST_MINUTE_CACHE_DIR) / f"{_slug(symbol)}.csv"
    if not path.exists():
        # Migracion transparente desde una cache unica previa para BTC.
        legacy = Path("state/backtest_1m_cache.csv")
        if symbol == "BTC/USDT:USDT" and legacy.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(legacy.read_bytes())
            print(f"Migrada cache BTC existente: {legacy} -> {path}")
        else:
            return _empty_minute_cache()
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return _empty_minute_cache()
    if df.empty:
        return _empty_minute_cache()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return (
        df.dropna(subset=["timestamp"])
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def save_minute_cache(symbol: str, df: pd.DataFrame) -> None:
    path = Path(BACKTEST_MINUTE_CACHE_DIR) / f"{_slug(symbol)}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
    out.to_csv(path, index=False)


def minute_period(
    cache: pd.DataFrame, start: pd.Timestamp, duration: pd.Timedelta
) -> pd.DataFrame:
    """Return an arbitrary minute-resolution period using DatetimeIndex search."""
    if cache.empty:
        return cache.iloc[0:0]
    start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
    if duration <= pd.Timedelta(0):
        raise ValueError("duration debe ser positiva")
    end = start + duration
    index = cache.index
    left = int(index.searchsorted(start, side="left"))
    right = int(index.searchsorted(end, side="left"))
    return cache.iloc[left:right]


def minute_window(cache: pd.DataFrame, hour: pd.Timestamp) -> pd.DataFrame:
    """Backward-compatible one-hour minute window."""
    return minute_period(cache, hour, pd.Timedelta(hours=1))


def candidate_entry_hours(
    signals: pd.DataFrame, include_shorts: bool, timeframe: str = TIMEFRAME
) -> list[pd.Timestamp]:
    mask = signals["entry_long"].fillna(False).astype(bool)
    if include_shorts:
        mask = mask | signals["entry_short"].fillna(False).astype(bool)
    delta = timeframe_to_timedelta(timeframe)
    return sorted(set((signals.index[mask] + delta).tolist()))


def _public_exchange():
    if ccxt is None:
        raise RuntimeError("CCXT no esta instalado. Ejecuta pip install -r requirements.txt")
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
    return exchange


def ensure_minute_cache(
    symbol: str,
    signals: pd.DataFrame,
    include_shorts: bool,
    timeframe: str = TIMEFRAME,
) -> pd.DataFrame:
    cache = load_minute_cache(symbol)
    starts = candidate_entry_hours(signals, include_shorts, timeframe=timeframe)
    duration = timeframe_to_timedelta(timeframe)
    required_minutes = int(duration / pd.Timedelta(minutes=1))
    if required_minutes < 1 or duration != pd.Timedelta(minutes=required_minutes):
        raise ValueError(f"Timeframe research no compatible con cache 1m: {timeframe}")
    if required_minutes > 1000:
        raise ValueError("Timeframe research demasiado grande para una sola descarga 1m")
    missing = [
        start for start in starts
        if len(minute_period(cache, start, duration)) < required_minutes
    ]
    print(
        f"{symbol} {timeframe}: senales candidatas={len(starts)} | periodos 1m cache={len(starts)-len(missing)} "
        f"| por descargar={len(missing)}"
    )
    if not missing:
        return cache

    exchange = _public_exchange()
    rows: list[list[float]] = []
    for i, start in enumerate(missing, start=1):
        since = int(start.timestamp() * 1000)
        batch = exchange.fetch_ohlcv(symbol, "1m", since=since, limit=required_minutes)
        rows.extend(batch)
        if i % 25 == 0 or i == len(missing):
            print(f"  {symbol} {timeframe} cache 1m: {i}/{len(missing)}")

    if rows:
        new = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        new["timestamp"] = pd.to_datetime(new["timestamp"], unit="ms", utc=True)
        new = new.set_index("timestamp")
        cache = pd.concat([cache, new]).sort_index()
        cache = cache[~cache.index.duplicated(keep="last")]
        save_minute_cache(symbol, cache)
    return cache


def _funding_paths(symbol: str) -> tuple[Path, Path]:
    base = Path(BACKTEST_FUNDING_CACHE_DIR) / _slug(symbol)
    return base.with_suffix(".csv"), base.with_suffix(".meta.json")


def load_funding_cache(symbol: str) -> pd.DataFrame:
    path, _ = _funding_paths(symbol)
    if not path.exists():
        return pd.DataFrame(
            columns=["funding_rate"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=["funding_rate"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    # Funding caches can contain valid ISO-8601 timestamps with mixed renderings
    # (space/T separator and optional fractional seconds). Pandas otherwise infers
    # one format from the first row and silently coerces the other valid rows to NaT.
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], format="mixed", utc=True, errors="coerce"
    )
    return (
        df.dropna(subset=["timestamp"])
        .drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")[["funding_rate"]]
    )


def _funding_sha256(cache: pd.DataFrame) -> str:
    h = hashlib.sha256()
    if cache.empty:
        return h.hexdigest()
    clean = cache.sort_index()
    for ts, rate in zip(clean.index, clean["funding_rate"].astype(float), strict=True):
        stamp = pd.Timestamp(ts).tz_convert("UTC").isoformat()
        h.update(f"{stamp},{float(rate):.17g}\n".encode("utf-8"))
    return h.hexdigest()


def funding_snapshot_info(symbol: str, cache: pd.DataFrame | None = None) -> dict:
    cache = load_funding_cache(symbol) if cache is None else cache.sort_index()
    return {
        "schema": 1,
        "symbol": symbol,
        "rows": int(len(cache)),
        "first": cache.index.min().isoformat() if not cache.empty else None,
        "last": cache.index.max().isoformat() if not cache.empty else None,
        "values_sha256": _funding_sha256(cache),
    }


def _write_funding_snapshot(symbol: str, cache: pd.DataFrame) -> dict:
    path, meta_path = _funding_paths(symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    cache = cache.sort_index().copy()
    cache.index.name = "timestamp"
    cache.reset_index().to_csv(path, index=False)
    info = funding_snapshot_info(symbol, cache)
    meta_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


def _verify_funding_snapshot(symbol: str, cache: pd.DataFrame) -> dict:
    _, meta_path = _funding_paths(symbol)
    if not meta_path.exists():
        raise RuntimeError(
            f"Cache funding legacy para {symbol}: falta snapshot metadata. "
            "Reconstruye el snapshot de funding una vez."
        )
    try:
        expected = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Metadata funding invalida para {symbol}: {exc}") from exc
    actual = funding_snapshot_info(symbol, cache)
    for key in ("symbol", "rows", "first", "last", "values_sha256"):
        if expected.get(key) != actual.get(key):
            raise RuntimeError(
                f"Snapshot funding alterado para {symbol}: {key} no coincide. "
                "Usa --rebuild-funding solo si quieres crear deliberadamente un nuevo snapshot."
            )
    return actual


def _fetch_funding_range(symbol: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> pd.DataFrame:
    if ccxt is None:
        if BACKTEST_REQUIRE_FUNDING:
            raise RuntimeError("CCXT no esta instalado y funding es obligatorio.")
        return pd.DataFrame(
            columns=["funding_rate"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )

    exchange = _public_exchange()
    cursor = int(start_ts.timestamp() * 1000)
    end_ms = int(end_ts.timestamp() * 1000)
    records: list[dict] = []
    while cursor < end_ms:
        batch = exchange.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        if not batch:
            break
        progressed = False
        for item in batch:
            raw_ts = item.get("timestamp")
            rate = item.get("fundingRate")
            if raw_ts is None or rate is None:
                continue
            ts = int(raw_ts)
            if ts >= end_ms:
                continue
            if ts >= int(start_ts.timestamp() * 1000):
                records.append({"timestamp": ts, "funding_rate": float(rate)})
            if ts >= cursor:
                cursor = ts + 1
                progressed = True
        if not progressed:
            break
        if int(batch[-1].get("timestamp") or 0) >= end_ms:
            break

    if not records:
        return pd.DataFrame(
            columns=["funding_rate"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
        )
    fresh = pd.DataFrame(records)
    fresh["timestamp"] = pd.to_datetime(fresh["timestamp"], unit="ms", utc=True)
    return (
        fresh.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")[["funding_rate"]]
    )


def _assert_funding_continuity(cache: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> None:
    sliced = cache.loc[(cache.index >= start_ts) & (cache.index < end_ts)]
    if sliced.empty:
        if BACKTEST_REQUIRE_FUNDING:
            raise RuntimeError("Funding snapshot vacio en el rango solicitado.")
        return
    if len(sliced) < 2:
        return
    gaps = sliced.index.to_series().diff().dropna().dt.total_seconds() / 3600.0
    max_gap = float(gaps.max()) if not gaps.empty else 0.0
    if max_gap > FUNDING_MAX_GAP_HOURS:
        raise RuntimeError(
            f"Funding snapshot tiene un hueco de {max_gap:.1f}h (> "
            f"{FUNDING_MAX_GAP_HOURS:g}h). Reconstruye con --rebuild-funding."
        )


def ensure_funding_cache(
    symbol: str,
    start: str,
    end: str,
    *,
    rebuild: bool = False,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    if end_ts <= start_ts:
        raise ValueError("Rango funding invalido")

    cache = load_funding_cache(symbol)
    _, meta_path = _funding_paths(symbol)

    if rebuild:
        cache = _fetch_funding_range(symbol, start_ts, end_ts)
        if BACKTEST_REQUIRE_FUNDING and cache.empty:
            raise RuntimeError(f"No se obtuvo funding historico para {symbol}.")
        _assert_funding_continuity(cache, start_ts, end_ts)
        _write_funding_snapshot(symbol, cache)
        return cache.loc[(cache.index >= start_ts) & (cache.index < end_ts)]

    if cache.empty:
        cache = _fetch_funding_range(symbol, start_ts, end_ts)
        if BACKTEST_REQUIRE_FUNDING and cache.empty:
            raise RuntimeError(f"No se obtuvo funding historico para {symbol}.")
        _assert_funding_continuity(cache, start_ts, end_ts)
        _write_funding_snapshot(symbol, cache)
        return cache.loc[(cache.index >= start_ts) & (cache.index < end_ts)]

    if not meta_path.exists():
        _verify_funding_snapshot(symbol, cache)  # raises with migration guidance
    _verify_funding_snapshot(symbol, cache)

    changed = False
    # Prefix extension is allowed only without overwriting existing timestamps.
    if cache.index.min() > start_ts + pd.Timedelta(hours=8):
        prefix = _fetch_funding_range(symbol, start_ts, cache.index.min())
        if not prefix.empty:
            cache = pd.concat([prefix, cache]).sort_index()
            cache = cache[~cache.index.duplicated(keep="last")]
            changed = True

    # Append-only tail extension: historical rows in the snapshot remain immutable.
    if cache.index.max() < end_ts - pd.Timedelta(hours=8):
        tail_start = cache.index.max() + pd.Timedelta(milliseconds=1)
        tail = _fetch_funding_range(symbol, tail_start, end_ts)
        if not tail.empty:
            existing = cache.copy()
            cache = pd.concat([cache, tail]).sort_index()
            cache = cache[~cache.index.duplicated(keep="first")]
            # Existing rows must remain byte-equivalent in canonical funding content.
            if _funding_sha256(cache.loc[existing.index]) != _funding_sha256(existing):
                raise RuntimeError("Extension funding intento modificar filas historicas congeladas.")
            changed = True

    _assert_funding_continuity(cache, start_ts, end_ts)
    if changed:
        _write_funding_snapshot(symbol, cache)

    sliced = cache.loc[(cache.index >= start_ts) & (cache.index < end_ts)]
    if BACKTEST_REQUIRE_FUNDING and sliced.empty:
        raise RuntimeError(f"No se obtuvo funding historico para {symbol}.")
    return sliced


def fetch_market_tick_size(symbol: str) -> float:
    if ccxt is None:
        return 0.0
    exchange = _public_exchange()
    market = exchange.market(symbol)
    for f in (market.get("info", {}).get("filters") or []):
        if f.get("filterType") == "PRICE_FILTER" and f.get("tickSize") is not None:
            try:
                tick = float(f["tickSize"])
                if tick > 0:
                    return tick
            except (TypeError, ValueError):
                pass
    precision = market.get("precision", {}).get("price")
    try:
        value = float(precision)
        return value if value > 0 and value < 1 else 0.0
    except (TypeError, ValueError):
        return 0.0


# -----------------------------------------------------------------------------
# Signal construction.
# -----------------------------------------------------------------------------
def build_signals(
    symbol: str,
    *,
    start: str = BACKTEST_START,
    end: str = BACKTEST_END,
) -> pd.DataFrame:
    # Import lazy para que tests de logica no requieran CCXT.
    from src.data.fetcher import fetch_binance_history_cached

    print(f"{symbol}: historia {TIMEFRAME} cache/incremental {start} -> {end}")
    df = fetch_binance_history_cached(
        symbol, TIMEFRAME, start, end, BACKTEST_HISTORY_CACHE_DIR
    )
    print(f"{symbol}: historia {REGIME_TIMEFRAME} cache/incremental para regimen")
    regime_df = fetch_binance_history_cached(
        symbol, REGIME_TIMEFRAME, start, end, BACKTEST_HISTORY_CACHE_DIR
    )
    min_rows = max(TREND_WINDOW, LOOKBACK_WINDOW, ATR_WINDOW) + 50
    if len(df) < min_rows:
        raise RuntimeError(f"Historia insuficiente para {symbol}: {len(df)} velas")
    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW,
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=ATR_WINDOW,
        entry_mode=ENTRY_MODE,
        regime_mode=REGIME_MODE,
        signal_timeframe=TIMEFRAME,
        regime_timeframe=REGIME_TIMEFRAME,
        regime_ema_window=REGIME_EMA_WINDOW,
        regime_slope_lookback=REGIME_SLOPE_LOOKBACK,
        shock1h_threshold=SHOCK1H_THRESHOLD,
        return_shock_vol_window=RETURN_SHOCK_VOL_WINDOW,
    )
    return strategy.calculate_signals(df, regime_df=regime_df)


def build_path_arrays(signals: pd.DataFrame) -> PathArrays:
    """Materialize hot OHLC/signal columns once for fast path simulation."""
    return PathArrays(
        index=signals.index,
        opens=signals["open"].to_numpy(dtype=float, copy=False),
        highs=signals["high"].to_numpy(dtype=float, copy=False),
        lows=signals["low"].to_numpy(dtype=float, copy=False),
        closes=signals["close"].to_numpy(dtype=float, copy=False),
        exit_long_mean=signals.get(
            "exit_long_mean", pd.Series(False, index=signals.index)
        ).fillna(False).to_numpy(dtype=bool, copy=False),
        exit_short_mean=signals.get(
            "exit_short_mean", pd.Series(False, index=signals.index)
        ).fillna(False).to_numpy(dtype=bool, copy=False),
    )


def build_funding_series(
    funding: pd.DataFrame,
    signals: pd.DataFrame,
    path_arrays: Optional[PathArrays] = None,
) -> FundingSeries:
    """Precompute price*funding-rate prefix sums for O(log n) trade funding."""
    if funding.empty:
        return FundingSeries(
            index_ns=np.asarray([], dtype=np.int64),
            prefix_value=np.asarray([0.0], dtype=float),
        )
    arrays = path_arrays or build_path_arrays(signals)
    # pandas 3.x conserva la resolucion real del DatetimeIndex (normalmente us),
    # mientras Timestamp.value siempre esta expresado en ns. FundingSeries compara
    # contra Timestamp.value, por lo que ambos indices deben normalizarse a ns.
    # Sin esto searchsorted devolvia intervalos vacios y funding=0 para todos los trades.
    idx_ns = funding.index.as_unit("ns").asi8.astype(np.int64, copy=False)
    signal_ns = arrays.index.as_unit("ns").asi8.astype(np.int64, copy=False)
    opens = arrays.opens
    closes = arrays.closes
    prices = np.empty(len(funding), dtype=float)
    for i, ts_ns in enumerate(idx_ns):
        pos = int(np.searchsorted(signal_ns, ts_ns, side="left"))
        if pos < len(signal_ns) and signal_ns[pos] == ts_ns:
            prices[i] = opens[pos]
        elif pos > 0:
            prices[i] = closes[pos - 1]
        else:
            prices[i] = np.nan
    rates = funding["funding_rate"].to_numpy(dtype=float, copy=False)
    values = np.where(np.isfinite(prices) & (prices > 0), prices * rates, 0.0)
    prefix = np.empty(len(values) + 1, dtype=float)
    prefix[0] = 0.0
    np.cumsum(values, out=prefix[1:])
    return FundingSeries(index_ns=idx_ns, prefix_value=prefix)


def prepare_entry_candidates(
    *,
    signals: pd.DataFrame,
    minute_cache: pd.DataFrame,
    market_slippage_bps: float,
    include_shorts: bool,
    signal_timeframe: str = TIMEFRAME,
) -> list[EntryCandidate]:
    """Resolve the V8/V9 entry once per signal/slippage and reuse for all exits."""
    candidates: list[EntryCandidate] = []
    delta = timeframe_to_timedelta(signal_timeframe)
    index = signals.index
    closes = signals["close"].to_numpy(dtype=float, copy=False)
    atrs = signals["atr"].to_numpy(dtype=float, copy=False)
    long_flags = signals["entry_long"].fillna(False).to_numpy(dtype=bool, copy=False)
    short_flags = signals["entry_short"].fillna(False).to_numpy(dtype=bool, copy=False)
    for i in range(1, len(signals)):
        signal_ts = index[i - 1]
        entry_hour = index[i]
        if entry_hour != signal_ts + delta:
            continue
        sides: list[str] = []
        if long_flags[i - 1]:
            sides.append("long")
        if include_shorts and short_flags[i - 1]:
            sides.append("short")
        if not sides:
            continue
        window = minute_period(minute_cache, entry_hour, delta)
        if window.empty:
            continue
        for side in sides:
            fill = resolve_entry(
                window,
                side=side,
                signal_close=float(closes[i - 1]),
                market_slippage_bps=market_slippage_bps,
            )
            if fill is not None:
                candidates.append(
                    EntryCandidate(
                        signal_time=signal_ts,
                        entry_hour=entry_hour,
                        side=side,
                        signal_atr=float(atrs[i - 1]),
                        fill=fill,
                    )
                )
    return candidates


# -----------------------------------------------------------------------------
# Exit path simulation. All economics are expressed per base unit so sizing can
# be applied later with shared portfolio equity.
# -----------------------------------------------------------------------------
def _side_sign(side: str) -> float:
    return 1.0 if side == "long" else -1.0


def _adverse_market_fill(raw: float, side: str, slippage_bps: float, is_exit: bool) -> float:
    slip = slippage_bps / 10_000.0
    if not is_exit:
        return raw * (1.0 + slip) if side == "long" else raw * (1.0 - slip)
    return raw * (1.0 - slip) if side == "long" else raw * (1.0 + slip)


def _crossed_maker_target(
    high: float, low: float, target: float, side: str, tick: float,
    cross_ticks: int = BACKTEST_TP_CROSS_TICKS,
) -> bool:
    cross = tick * cross_ticks
    if side == "long":
        return high >= target + cross
    return low <= target - cross


def _hit_stop(high: float, low: float, stop: float, side: str) -> bool:
    return low <= stop if side == "long" else high >= stop


def _gap_through_stop(open_price: float, stop: float, side: str) -> bool:
    return open_price <= stop if side == "long" else open_price >= stop


def _favorable_distance(entry: float, high: float, low: float, side: str) -> float:
    return max(0.0, high - entry) if side == "long" else max(0.0, entry - low)


def _adverse_distance(entry: float, high: float, low: float, side: str) -> float:
    return max(0.0, entry - low) if side == "long" else max(0.0, high - entry)


def _funding_price(signals: pd.DataFrame, ts: pd.Timestamp) -> float:
    if ts in signals.index:
        return float(signals.loc[ts, "open"])
    earlier = signals.loc[signals.index <= ts]
    if earlier.empty:
        return float("nan")
    return float(earlier.iloc[-1]["close"])


def _funding_per_unit(
    funding: pd.DataFrame,
    signals: pd.DataFrame,
    side: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    partial_time: Optional[pd.Timestamp],
    remaining_after_partial: float,
    funding_series: Optional[FundingSeries] = None,
) -> float:
    if exit_time <= entry_time:
        return 0.0
    if funding_series is not None:
        if partial_time is None or partial_time >= exit_time:
            base = funding_series.interval_sum(entry_time, exit_time)
        else:
            before = funding_series.interval_sum(entry_time, partial_time)
            after = funding_series.interval_sum(partial_time, exit_time)
            base = before + after * remaining_after_partial
        return (-1.0 if side == "long" else 1.0) * base

    if funding.empty:
        return 0.0
    index = funding.index
    left = int(index.searchsorted(entry_time, side="right"))
    right = int(index.searchsorted(exit_time, side="right"))
    total = 0.0
    for ts, row in funding.iloc[left:right].iterrows():
        price = _funding_price(signals, ts)
        if not math.isfinite(price) or price <= 0:
            continue
        qty_fraction = (
            remaining_after_partial
            if partial_time is not None and ts > partial_time
            else 1.0
        )
        rate = float(row["funding_rate"])
        total += (-1.0 if side == "long" else 1.0) * price * rate * qty_fraction
    return total


def simulate_trade_path(
    *,
    symbol: str,
    side: str,
    signal_time: pd.Timestamp,
    entry_hour: pd.Timestamp,
    fill: EntryFill,
    signal_atr: float,
    signals: pd.DataFrame,
    minute_hour: pd.DataFrame,
    funding: pd.DataFrame,
    policy: ExitPolicy,
    market_slippage_bps: float,
    tick_size: float,
    path_arrays: Optional[PathArrays] = None,
    funding_series: Optional[FundingSeries] = None,
    tp_cross_ticks: int = BACKTEST_TP_CROSS_TICKS,
    signal_timeframe: str = TIMEFRAME,
) -> Optional[TradeOpportunity]:
    stop_distance = signal_atr * ATR_MULTIPLIER
    if not math.isfinite(stop_distance) or stop_distance <= 0:
        return None
    entry = fill.fill_price
    sign = _side_sign(side)
    initial_stop = entry - sign * stop_distance
    fixed_tp = entry + sign * stop_distance * policy.rr
    active_stop = initial_stop
    break_even_armed = False
    pending_break_even = False
    remaining = 1.0
    partial_taken = False
    partial_time: Optional[pd.Timestamp] = None

    entry_fee = entry * fill.fee_rate
    realized_gross = 0.0
    exit_fees = 0.0
    exit_slippage = 0.0
    max_fav = 0.0
    max_adv = 0.0
    arrays = path_arrays or build_path_arrays(signals)

    def close_fraction(
        raw: float, fraction: float, fee_rate: float, market: bool
    ) -> tuple[float, float, float, float]:
        actual = (
            _adverse_market_fill(raw, side, market_slippage_bps, is_exit=True)
            if market
            else raw
        )
        gross = sign * (actual - entry) * fraction
        fee = actual * fee_rate * fraction
        slip = abs(actual - raw) * fraction if market else 0.0
        return actual, gross, fee, slip

    def process_bar(
        ts: pd.Timestamp,
        o: float,
        h: float,
        l: float,
        *,
        allow_tp: bool,
        allow_activation: bool,
    ) -> Optional[tuple[pd.Timestamp, str, float, float, float]]:
        nonlocal active_stop, break_even_armed, pending_break_even, remaining
        nonlocal partial_taken, partial_time, realized_gross, exit_fees, exit_slippage
        nonlocal max_fav, max_adv

        max_fav = max(max_fav, _favorable_distance(entry, h, l, side))
        max_adv = max(max_adv, _adverse_distance(entry, h, l, side))

        # Conservador: gap/stop se resuelve antes que cualquier objetivo que
        # tambien aparezca dentro de la misma vela.
        if _gap_through_stop(o, active_stop, side):
            raw = o
            _, gross, fee, slip = close_fraction(
                raw, remaining, BACKTEST_TAKER_FEE, True
            )
            return ts, "stop_gap", gross, fee, slip
        if _hit_stop(h, l, active_stop, side):
            raw = active_stop
            _, gross, fee, slip = close_fraction(
                raw, remaining, BACKTEST_TAKER_FEE, True
            )
            reason = (
                "break_even"
                if break_even_armed
                and math.isclose(
                    active_stop,
                    entry,
                    rel_tol=0,
                    abs_tol=max(tick_size, 1e-12),
                )
                else "stop_loss"
            )
            return ts, reason, gross, fee, slip

        if (
            policy.use_fixed_tp
            and allow_tp
            and _crossed_maker_target(h, l, fixed_tp, side, tick_size, tp_cross_ticks)
        ):
            _, gross, fee, slip = close_fraction(
                fixed_tp, remaining, BACKTEST_MAKER_FEE, False
            )
            return ts, "take_profit", gross, fee, slip

        if (
            allow_activation
            and not partial_taken
            and policy.partial_trigger_r is not None
            and remaining > 0
        ):
            partial_target = entry + sign * stop_distance * policy.partial_trigger_r
            if _crossed_maker_target(h, l, partial_target, side, tick_size, tp_cross_ticks):
                fraction = min(policy.partial_fraction, remaining)
                _, gross, fee, slip = close_fraction(
                    partial_target, fraction, BACKTEST_MAKER_FEE, False
                )
                realized_gross += gross
                exit_fees += fee
                exit_slippage += slip
                remaining -= fraction
                partial_taken = True
                partial_time = ts
                if policy.partial_move_stop_to_be:
                    pending_break_even = True
                if remaining <= 1e-12:
                    return ts, "partial_full", 0.0, 0.0, 0.0

        if (
            allow_activation
            and not break_even_armed
            and policy.break_even_trigger_r is not None
        ):
            trigger = entry + sign * stop_distance * policy.break_even_trigger_r
            reached = h >= trigger if side == "long" else l <= trigger
            if reached:
                pending_break_even = True
        return None

    def finish(
        exit_time: pd.Timestamp,
        reason: str,
    ) -> TradeOpportunity:
        funding_pnl = _funding_per_unit(
            funding,
            signals,
            side,
            fill.timestamp,
            exit_time,
            partial_time,
            remaining,
            funding_series=funding_series,
        )
        total_fee = entry_fee + exit_fees
        net = realized_gross - total_fee + funding_pnl
        return TradeOpportunity(
            symbol=symbol,
            side=side,
            exit_policy=policy.name,
            rr=policy.rr,
            signal_time=signal_time,
            entry_time=fill.timestamp,
            exit_time=exit_time,
            entry_price=entry,
            raw_entry_price=fill.raw_price,
            stop_distance=stop_distance,
            stop_fraction=stop_distance / entry,
            exit_reason=reason,
            entry_liquidity=fill.liquidity,
            entry_route=fill.route,
            entry_delay_minutes=(fill.timestamp - entry_hour).total_seconds() / 60.0,
            entry_improvement_bps=fill.improvement_bps,
            gross_pnl_per_unit=realized_gross,
            fee_per_unit=total_fee,
            slippage_per_unit=fill.slippage_per_unit + exit_slippage,
            funding_pnl_per_unit=funding_pnl,
            net_pnl_per_unit=net,
            mfe_r=max_fav / stop_distance,
            mae_r=max_adv / stop_distance,
            holding_hours=max(
                0.0, (exit_time - fill.timestamp).total_seconds() / 3600.0
            ),
            partial_taken=partial_taken,
            partial_trigger_r=policy.partial_trigger_r,
            break_even_armed=break_even_armed,
        )

    # Minutos restantes de la hora de entrada. itertuples evita construir una
    # Series por minuto; en el minuto maker de fill no se permite TP/partial/BE.
    after = minute_hour.loc[minute_hour.index >= fill.timestamp]
    for row in after.itertuples():
        minute_ts = row.Index
        same_maker_minute = fill.liquidity == "maker" and minute_ts == fill.timestamp
        outcome = process_bar(
            minute_ts,
            float(row.open),
            float(row.high),
            float(row.low),
            allow_tp=not same_maker_minute,
            allow_activation=not same_maker_minute,
        )
        if outcome is not None:
            exit_time, reason, gross, fee, slip = outcome
            realized_gross += gross
            exit_fees += fee
            exit_slippage += slip
            return finish(exit_time, reason)
        if pending_break_even:
            active_stop = entry
            break_even_armed = True
            pending_break_even = False

    # Horas siguientes sobre arrays NumPy. Evita signals.loc[mask] + iterrows
    # para cada trade, que era el principal hot path del replay.
    start_pos = int(arrays.index.searchsorted(entry_hour, side="right"))
    exited = False
    exit_time: Optional[pd.Timestamp] = None
    reason = ""
    for j in range(start_pos, len(arrays.index)):
        ts = arrays.index[j]
        outcome = process_bar(
            ts,
            float(arrays.opens[j]),
            float(arrays.highs[j]),
            float(arrays.lows[j]),
            allow_tp=True,
            allow_activation=True,
        )
        if outcome is not None:
            exit_time, reason, gross, fee, slip = outcome
            realized_gross += gross
            exit_fees += fee
            exit_slippage += slip
            exited = True
            break

        mean_hit = policy.mean_exit and bool(
            arrays.exit_long_mean[j] if side == "long" else arrays.exit_short_mean[j]
        )
        time_hit = (
            policy.time_stop_hours is not None
            and (ts + pd.Timedelta(hours=1) - fill.timestamp).total_seconds() / 3600.0
            >= policy.time_stop_hours
        )
        if mean_hit or time_hit:
            next_pos = j + 1
            if next_pos >= len(arrays.index):
                continue
            next_ts = arrays.index[next_pos]
            next_open = float(arrays.opens[next_pos])
            if _gap_through_stop(next_open, active_stop, side):
                raw = next_open
                reason = "stop_gap"
            else:
                raw = next_open
                reason = "mean_exit" if mean_hit else "time_stop"
            _, gross, fee, slip = close_fraction(
                raw, remaining, BACKTEST_TAKER_FEE, True
            )
            realized_gross += gross
            exit_fees += fee
            exit_slippage += slip
            exit_time = next_ts
            exited = True
            break

        if pending_break_even:
            active_stop = entry
            break_even_armed = True
            pending_break_even = False

    if not exited:
        # Posicion superviviente: cierre MARKET adverso en el ultimo close.
        if len(arrays.index) == 0:
            return None
        if start_pos >= len(arrays.index):
            entry_pos = int(arrays.index.searchsorted(entry_hour, side="left"))
            final_pos = min(entry_pos, len(arrays.index) - 1)
        else:
            final_pos = len(arrays.index) - 1
        raw = float(arrays.closes[final_pos])
        _, gross, fee, slip = close_fraction(
            raw, remaining, BACKTEST_TAKER_FEE, True
        )
        realized_gross += gross
        exit_fees += fee
        exit_slippage += slip
        exit_time = arrays.index[final_pos] + timeframe_to_timedelta(signal_timeframe)
        reason = "end_of_data"

    assert exit_time is not None
    return finish(exit_time, reason)

def generate_opportunities(
    *,
    symbol: str,
    signals: pd.DataFrame,
    minute_cache: pd.DataFrame,
    funding: pd.DataFrame,
    exit_policy: ExitPolicy,
    market_slippage_bps: float,
    tick_size: float,
    include_shorts: bool,
    entry_candidates: Optional[list[EntryCandidate]] = None,
    path_arrays: Optional[PathArrays] = None,
    funding_series: Optional[FundingSeries] = None,
    tp_cross_ticks: int = BACKTEST_TP_CROSS_TICKS,
    signal_timeframe: str = TIMEFRAME,
) -> pd.DataFrame:
    """Generate independent unit opportunities.

    The original implementation imposed ``symbol_busy_until`` here. That was both slow and
    conceptually too early: whether a prior signal actually consumes the symbol
    depends on portfolio sizing/concurrency. We now generate every valid signal
    path independently and let ``simulate_portfolio`` enforce one live position
    per symbol. This makes opportunity generation reusable across all sizing
    configs and side modes.
    """
    arrays = path_arrays or build_path_arrays(signals)
    funding_fast = funding_series or build_funding_series(funding, signals, arrays)
    candidates = entry_candidates or prepare_entry_candidates(
        signals=signals,
        minute_cache=minute_cache,
        market_slippage_bps=market_slippage_bps,
        include_shorts=include_shorts,
        signal_timeframe=signal_timeframe,
    )
    records: list[dict] = []
    for candidate in candidates:
        minute_hour = minute_period(
            minute_cache, candidate.entry_hour, timeframe_to_timedelta(signal_timeframe)
        )
        if minute_hour.empty:
            continue
        trade = simulate_trade_path(
            symbol=symbol,
            side=candidate.side,
            signal_time=candidate.signal_time,
            entry_hour=candidate.entry_hour,
            fill=candidate.fill,
            signal_atr=candidate.signal_atr,
            signals=signals,
            minute_hour=minute_hour,
            funding=funding,
            policy=exit_policy,
            market_slippage_bps=market_slippage_bps,
            tick_size=tick_size,
            path_arrays=arrays,
            funding_series=funding_fast,
            tp_cross_ticks=tp_cross_ticks,
            signal_timeframe=signal_timeframe,
        )
        if trade is not None:
            records.append(asdict(trade))
    return pd.DataFrame.from_records(records)


# -----------------------------------------------------------------------------
# Shared-equity portfolio sizing.
# -----------------------------------------------------------------------------
def _profit_factor(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    gains = float(values[values > 0].sum())
    losses = float(-values[values < 0].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def simulate_portfolio(
    opportunities: pd.DataFrame,
    config: PortfolioConfig,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_capital: float = INITIAL_CAPITAL,
    leverage: int = LEVERAGE,
    cash_buffer_pct: float = CASH_BUFFER_PCT,
    max_portfolio_risk_pct: float = BACKTEST_MAX_PORTFOLIO_RISK_PCT,
    max_concurrent_positions: int = BACKTEST_MAX_CONCURRENT_POSITIONS,
    prefiltered_config: bool = False,
    risk_per_trade_by_component: Optional[dict[str, float]] = None,
) -> PortfolioResult:
    if opportunities.empty:
        return PortfolioResult(
            config=config, start=start, end=end, initial_capital=initial_capital,
            final_equity=initial_capital, total_return=0.0, max_drawdown=0.0,
            trades=pd.DataFrame(), skipped_portfolio_risk=0,
            skipped_concurrency=0, skipped_margin=0,
        )

    df = opportunities
    # CSV caches arrive as strings; in-memory research frames are already UTC.
    if not isinstance(df["entry_time"].dtype, pd.DatetimeTZDtype):
        df = df.copy()
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)

    mask = (
        (df["entry_time"] >= start)
        & (df["entry_time"] <= end)
        # Anti-leakage: training only uses trades already closed by period end.
        & (df["exit_time"] <= end)
    )
    if not prefiltered_config:
        mask &= df["exit_policy"].eq(config.exit_policy)
        mask &= np.isclose(df["rr"].to_numpy(dtype=float, copy=False), config.rr)
        if config.side_mode == "long_only":
            mask &= df["side"].eq("long")
        elif config.side_mode != "long_short":
            raise ValueError(f"side_mode invalido: {config.side_mode}")

    selected = df.loc[mask]
    if selected.empty:
        return PortfolioResult(
            config=config, start=start, end=end, initial_capital=initial_capital,
            final_equity=initial_capital, total_return=0.0, max_drawdown=0.0,
            trades=pd.DataFrame(), skipped_portfolio_risk=0,
            skipped_concurrency=0, skipped_margin=0,
        )
    if not selected["entry_time"].is_monotonic_increasing:
        selected = selected.sort_values(["entry_time", "symbol"], kind="stable")

    realized_equity = float(initial_capital)
    peak_equity = realized_equity
    max_dd = 0.0
    open_positions: list[dict] = []
    sized: list[SizedTrade] = []
    skipped_risk = skipped_concurrency = skipped_margin = 0

    def realize_until(ts: pd.Timestamp) -> None:
        nonlocal realized_equity, peak_equity, max_dd, open_positions
        if not open_positions:
            return
        due = [p for p in open_positions if p["exit_time"] <= ts]
        if not due:
            return
        due.sort(key=lambda p: p["exit_time"])
        due_ids = {id(p) for p in due}
        for p in due:
            realized_equity += p["pnl"]
            peak_equity = max(peak_equity, realized_equity)
            if peak_equity > 0:
                max_dd = min(max_dd, realized_equity / peak_equity - 1.0)
        open_positions = [p for p in open_positions if id(p) not in due_ids]

    # itertuples is an order of magnitude faster than iterrows in this hot loop.
    for op in selected.itertuples(index=False):
        entry_time = op.entry_time
        realize_until(entry_time)
        if realized_equity <= 0:
            break
        if any(p["symbol"] == op.symbol for p in open_positions):
            continue
        if len(open_positions) >= max_concurrent_positions:
            skipped_concurrency += 1
            continue

        used_margin = sum(p["required_margin"] for p in open_positions)
        usable_margin_total = realized_equity * (1.0 - cash_buffer_pct)
        free_margin = max(0.0, usable_margin_total - used_margin)
        if free_margin <= 0:
            skipped_margin += 1
            continue

        stop_fraction = float(op.stop_fraction)
        if not math.isfinite(stop_fraction) or stop_fraction <= 0:
            continue
        component = str(getattr(op, "component", ""))
        component_risk_per_trade = config.risk_per_trade
        if risk_per_trade_by_component is not None:
            component_risk_per_trade = float(
                risk_per_trade_by_component.get(component, config.risk_per_trade)
            )
            if not math.isfinite(component_risk_per_trade) or component_risk_per_trade <= 0:
                raise ValueError(
                    f"Invalid component risk for {component or op.symbol}: "
                    f"{component_risk_per_trade!r}"
                )
        risk_budget = realized_equity * component_risk_per_trade
        risk_sized_notional = risk_budget / stop_fraction
        max_position_notional = realized_equity * config.max_position_pct
        margin_notional = free_margin * leverage

        open_risk = sum(p["initial_risk"] for p in open_positions)
        portfolio_risk_room = max(
            0.0, realized_equity * max_portfolio_risk_pct - open_risk
        )
        if portfolio_risk_room <= 0:
            skipped_risk += 1
            continue
        portfolio_risk_notional = portfolio_risk_room / stop_fraction

        caps = (
            ("risk", risk_sized_notional),
            ("position_pct", max_position_notional),
            ("margin", margin_notional),
            ("portfolio_risk", portfolio_risk_notional),
        )
        limiting, notional = min(caps, key=lambda item: item[1])
        if notional <= 0:
            skipped_margin += 1
            continue

        entry_price = float(op.entry_price)
        amount = notional / entry_price
        initial_risk = notional * stop_fraction
        pnl = amount * float(op.net_pnl_per_unit)
        funding_pnl = amount * float(op.funding_pnl_per_unit)
        fees = amount * float(op.fee_per_unit)
        slippage_cost = amount * float(op.slippage_per_unit)
        entry_equity = realized_equity
        risk_util = initial_risk / risk_budget if risk_budget > 0 else 0.0
        mae_r = max(0.0, float(getattr(op, "mae_r", 0.0)))

        item = {
            "symbol": op.symbol,
            "exit_time": op.exit_time,
            "required_margin": notional / leverage,
            "initial_risk": initial_risk,
            "mae_loss": initial_risk * mae_r,
            "pnl": pnl,
        }
        open_positions.append(item)
        stressed_equity = realized_equity - sum(
            p["mae_loss"] for p in open_positions
        )
        peak_equity = max(peak_equity, realized_equity)
        if peak_equity > 0:
            max_dd = min(max_dd, stressed_equity / peak_equity - 1.0)
        sized.append(
            SizedTrade(
                config=config.name,
                symbol=str(op.symbol),
                side=str(op.side),
                entry_time=entry_time,
                exit_time=op.exit_time,
                exit_reason=str(op.exit_reason),
                notional=notional,
                amount=amount,
                initial_risk=initial_risk,
                risk_utilization=risk_util,
                limiting_factor=limiting,
                pnl=pnl,
                return_on_equity_at_entry=pnl / entry_equity,
                funding_pnl=funding_pnl,
                fees=fees,
                slippage_cost=slippage_cost,
            )
        )

    realize_until(end)
    trades = pd.DataFrame.from_records([asdict(t) for t in sized])
    final_equity = realized_equity
    return PortfolioResult(
        config=config,
        start=start,
        end=end,
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return=final_equity / initial_capital - 1.0,
        max_drawdown=max_dd,
        trades=trades,
        skipped_portfolio_risk=skipped_risk,
        skipped_concurrency=skipped_concurrency,
        skipped_margin=skipped_margin,
    )


def portfolio_metrics(result: PortfolioResult) -> dict:
    t = result.trades
    return {
        "config": result.config.name,
        "exit_policy": result.config.exit_policy,
        "rr": result.config.rr,
        "risk_per_trade": result.config.risk_per_trade,
        "max_position_pct": result.config.max_position_pct,
        "side_mode": result.config.side_mode,
        "start": result.start.isoformat(),
        "end": result.end.isoformat(),
        "trades": len(t),
        "win_rate": float((t["pnl"] > 0).mean()) if not t.empty else float("nan"),
        "net_pf": _profit_factor(t["pnl"]) if not t.empty else float("nan"),
        "net_pnl": float(t["pnl"].sum()) if not t.empty else 0.0,
        "return": result.total_return,
        "max_dd": result.max_drawdown,
        "median_risk_utilization": float(t["risk_utilization"].median()) if not t.empty else float("nan"),
        "funding_pnl": float(t["funding_pnl"].sum()) if not t.empty else 0.0,
        "fees": float(t["fees"].sum()) if not t.empty else 0.0,
        "slippage_cost": float(t["slippage_cost"].sum()) if not t.empty else 0.0,
        "skipped_portfolio_risk": result.skipped_portfolio_risk,
        "skipped_concurrency": result.skipped_concurrency,
        "skipped_margin": result.skipped_margin,
    }


def rank_training_results(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return rows
    ranked = rows.copy()
    eligible = ranked[
        (ranked["trades"] >= BACKTEST_MIN_TRAIN_TRADES)
        & (ranked["max_dd"] >= -BACKTEST_DD_LIMIT)
        & np.isfinite(ranked["net_pf"])
    ].copy()
    if eligible.empty:
        eligible = ranked[ranked["trades"] >= BACKTEST_MIN_TRAIN_TRADES].copy()
    if eligible.empty:
        eligible = ranked.copy()
    # Profit primero, pero sujeto a DD. PF y utilizacion desempatan.
    return eligible.sort_values(
        ["return", "net_pf", "median_risk_utilization"],
        ascending=[False, False, False],
    )


def anchored_walk_forward(
    opportunities: pd.DataFrame,
    configs: Iterable[PortfolioConfig],
    *,
    first_test_year: int = BACKTEST_FIRST_TEST_YEAR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if opportunities.empty:
        return pd.DataFrame(), pd.DataFrame()
    configs = list(configs)
    book = OpportunityBook(opportunities)
    start_all = book.frame["entry_time"].min()
    end_all = book.frame["entry_time"].max()
    years = range(max(first_test_year, start_all.year + 1), end_all.year + 1)
    wf_rows: list[dict] = []
    selected_trades: list[pd.DataFrame] = []
    by_name = {c.name: c for c in configs}

    for year in years:
        train_start = pd.Timestamp(year=start_all.year, month=1, day=1, tz="UTC")
        train_end = pd.Timestamp(year=year - 1, month=12, day=31, hour=23, minute=59, tz="UTC")
        test_start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
        test_end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, tz="UTC")

        train_metrics: list[dict] = []
        for cfg in configs:
            source = book.for_config(cfg)
            result = simulate_portfolio(
                source, cfg, start=train_start, end=train_end, prefiltered_config=True
            )
            train_metrics.append(portfolio_metrics(result))
        ranked = rank_training_results(pd.DataFrame(train_metrics))
        if ranked.empty:
            continue
        best_name = str(ranked.iloc[0]["config"])
        best_cfg = by_name[best_name]
        test_source = book.for_config(best_cfg)
        test_result = simulate_portfolio(
            test_source, best_cfg, start=test_start, end=test_end, prefiltered_config=True
        )
        test_metrics = portfolio_metrics(test_result)
        test_metrics.update(
            {
                "test_year": year,
                "selected_config": best_name,
                "train_return": float(ranked.iloc[0]["return"]),
                "train_net_pf": float(ranked.iloc[0]["net_pf"]),
                "train_max_dd": float(ranked.iloc[0]["max_dd"]),
            }
        )
        wf_rows.append(test_metrics)
        if not test_result.trades.empty:
            detail = test_result.trades.copy()
            detail.insert(0, "test_year", year)
            selected_trades.append(detail)

    return pd.DataFrame(wf_rows), (
        pd.concat(selected_trades, ignore_index=True)
        if selected_trades
        else pd.DataFrame()
    )


def aggregate_walk_forward(wf: pd.DataFrame) -> dict:
    if wf.empty:
        return {}
    initial = INITIAL_CAPITAL
    equity = initial
    yearly_returns = []
    for _, row in wf.sort_values("test_year").iterrows():
        r = float(row["return"])
        yearly_returns.append(r)
        equity *= 1.0 + r
    return {
        "years": int(len(wf)),
        "positive_years": int((wf["net_pnl"] > 0).sum()),
        "worst_year_return": float(wf["return"].min()),
        "median_year_return": float(wf["return"].median()),
        "median_year_pf": float(wf["net_pf"].median()),
        "worst_year_pf": float(wf["net_pf"].min()),
        "compounded_return": equity / initial - 1.0,
    }


def serialize_recommendation(path: str, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
