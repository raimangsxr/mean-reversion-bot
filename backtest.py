from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import math
import os
import time
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    BACKTEST_ANNUAL_FILE,
    BACKTEST_END,
    BACKTEST_ENTRY_EXECUTION_MODES,
    BACKTEST_MAKER_FEE,
    BACKTEST_MARKET_SLIPPAGE_BPS,
    BACKTEST_MINUTE_CACHE_FILE,
    BACKTEST_START,
    BACKTEST_SUMMARY_FILE,
    BACKTEST_TAKER_FEE,
    BACKTEST_TP_RATIOS,
    BACKTEST_TRADES_FILE,
    CASH_BUFFER_PCT,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    ENTRY_MODE,
    INITIAL_CAPITAL,
    LEVERAGE,
    LOOKBACK_WINDOW,
    MAX_POSITION_PCT,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    RISK_PER_TRADE,
    SYMBOL,
    TAKE_PROFIT_ORDER_MODE,
    TIMEFRAME,
    TREND_WINDOW,
    Z_THRESHOLD,
    validate_settings,
)
from src.data.fetcher import fetch_binance_history
from src.risk.manager import RiskManager
from src.strategy.mean_reversion import MeanReversionStrategy


# -----------------------------------------------------------------------------
# Backtest V8 final. Compara la baseline MARKET inmediata con la politica V8 final
# congelada y barre solo RR/slippage. No reabre la optimizacion de microentrada.
# -----------------------------------------------------------------------------
CACHE_FILE = Path(BACKTEST_MINUTE_CACHE_FILE)
SUMMARY_FILE = Path(BACKTEST_SUMMARY_FILE)
ANNUAL_FILE = Path(BACKTEST_ANNUAL_FILE)
TRADES_FILE = Path(BACKTEST_TRADES_FILE)
MARKET_SLIPPAGE_BPS = tuple(BACKTEST_MARKET_SLIPPAGE_BPS)
RR_GRID = tuple(BACKTEST_TP_RATIOS)

@dataclass(frozen=True)
class EntryVariant:
    name: str
    kind: str  # market | limit_only | conditional
    delay_minutes: int = 0
    limit_offset_bps: float = 0.0
    wait_minutes: int = 0
    max_chase_bps: float = 0.0


# Modos finales:
# - market_0m: control MARKET inmediato + TP maker.
# - pullback_limit: V8 final, LIMIT -offset desde signal close durante 7m y
#   fallback MARKET solo si el precio al expirar no supera t=0 + cap.
if ENTRY_LIMIT_WAIT_SECONDS % 60 != 0:
    raise ValueError(
        "ENTRY_LIMIT_WAIT_SECONDS debe ser multiplo de 60 para backtest 1m."
    )

ENTRY_VARIANT_MAP = {
    "market_0m": EntryVariant("market_0m", "market", delay_minutes=0),
    "pullback_limit": EntryVariant(
        "pullback_limit",
        "conditional",
        limit_offset_bps=ENTRY_LIMIT_OFFSET_BPS,
        wait_minutes=ENTRY_LIMIT_WAIT_SECONDS // 60,
        max_chase_bps=ENTRY_FALLBACK_CAP_BPS,
    ),
}
ENTRY_VARIANTS = tuple(
    ENTRY_VARIANT_MAP[name] for name in BACKTEST_ENTRY_EXECUTION_MODES
)

@dataclass
class EntryFill:
    timestamp: pd.Timestamp
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
        # LONG: positivo = compramos mas barato que el open baseline.
        return (self.baseline_open - self.fill_price) / self.baseline_open * 10_000.0


@dataclass
class OpenPosition:
    entry_time: pd.Timestamp
    signal_time: pd.Timestamp
    entry_hour: pd.Timestamp
    raw_entry_price: float
    entry_price: float
    amount: float
    entry_fee: float
    entry_slippage_cost: float
    stop_price: float
    take_profit_price: float
    stop_distance: float
    initial_risk: float
    entry_liquidity: str
    entry_route: str
    entry_delay_minutes: float
    entry_improvement_bps: float
    max_favorable_distance: float = 0.0
    max_adverse_distance: float = 0.0


@dataclass
class ClosedTrade:
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    amount: float
    stop_price: float
    take_profit_price: float
    exit_reason: str
    entry_liquidity: str
    entry_route: str
    entry_delay_minutes: float
    entry_improvement_bps: float
    gross_before_costs: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    pnl: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    holding_hours: float


@dataclass
class SimulationResult:
    variant: str
    risk_reward_ratio: float
    slippage_bps: float
    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    attempted_signals: int
    filled_entries: int
    missed_entries: int
    maker_entries: int
    taker_entries: int
    trades: pd.DataFrame


# -----------------------------------------------------------------------------
# Datos 1m: cacheamos solo las horas posteriores a senales candidatas.
# -----------------------------------------------------------------------------
def _public_exchange() -> ccxt.binance:
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


MINUTE_COLUMNS = ["open", "high", "low", "close", "volume"]


def _empty_minute_cache() -> pd.DataFrame:
    """Cache 1m vacia con el mismo DatetimeIndex UTC que una cache cargada."""
    return pd.DataFrame(
        columns=MINUTE_COLUMNS,
        index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
    )


def _load_minute_cache() -> pd.DataFrame:
    if not CACHE_FILE.exists():
        return _empty_minute_cache()

    try:
        df = pd.read_csv(CACHE_FILE)
    except pd.errors.EmptyDataError:
        return _empty_minute_cache()

    if df.empty:
        return _empty_minute_cache()
    if "timestamp" not in df.columns:
        raise RuntimeError(
            f"Cache 1m invalida: falta la columna timestamp en {CACHE_FILE}."
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        return _empty_minute_cache()

    return (
        df.drop_duplicates(subset="timestamp", keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def _save_minute_cache(df: pd.DataFrame) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    out = df.sort_index().reset_index().rename(columns={"index": "timestamp"})
    out.to_csv(CACHE_FILE, index=False)


def _minute_window(cache: pd.DataFrame, hour: pd.Timestamp) -> pd.DataFrame:
    if cache.empty:
        return cache.iloc[0:0]
    if not isinstance(cache.index, pd.DatetimeIndex):
        raise RuntimeError("La cache 1m debe usar DatetimeIndex.")

    if hour.tzinfo is None:
        hour = hour.tz_localize("UTC")
    else:
        hour = hour.tz_convert("UTC")

    end = hour + pd.Timedelta(hours=1)
    return cache.loc[(cache.index >= hour) & (cache.index < end)]


def _candidate_entry_hours(signals: pd.DataFrame) -> list[pd.Timestamp]:
    delta = pd.Timedelta(TIMEFRAME)
    hours = [ts + delta for ts in signals.index[signals["entry_long"].fillna(False)]]
    if not hours:
        return []
    # Solo horas para las que existe una vela 1h posterior en el dataset.
    last_hour = signals.index[-1]
    return sorted({h for h in hours if h <= last_hour})


def ensure_minute_cache(signals: pd.DataFrame) -> pd.DataFrame:
    cache = _load_minute_cache()
    hours = _candidate_entry_hours(signals)
    missing = [h for h in hours if len(_minute_window(cache, h)) < 60]

    print(
        f"Senales candidatas LONG: {len(hours)} | "
        f"horas 1m en cache: {len(hours) - len(missing)} | por descargar: {len(missing)}"
    )
    if not missing:
        return cache

    exchange = _public_exchange()
    new_frames: list[pd.DataFrame] = []

    for n, hour in enumerate(missing, start=1):
        since = int(hour.timestamp() * 1000)
        rows = exchange.fetch_ohlcv(SYMBOL, timeframe="1m", since=since, limit=60)
        frame = pd.DataFrame(
            rows,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
            frame = frame[(frame["timestamp"] >= hour) & (frame["timestamp"] < hour + pd.Timedelta(hours=1))]
            frame = frame.set_index("timestamp")
            new_frames.append(frame)

        if n % 25 == 0 or n == len(missing):
            if new_frames:
                cache = pd.concat([cache, *new_frames])
                cache = cache[~cache.index.duplicated(keep="last")].sort_index()
                new_frames.clear()
                _save_minute_cache(cache)
            print(f"  cache 1m: {n}/{len(missing)} horas descargadas")

        # CCXT ya limita la tasa; esta pausa pequena evita rafagas innecesarias.
        time.sleep(0.02)

    return cache


# -----------------------------------------------------------------------------
# Entrada micro.
# -----------------------------------------------------------------------------
def _market_entry_fill(
    minute_hour: pd.DataFrame,
    delay_minutes: int,
    market_slippage_bps: float,
    route: str,
) -> Optional[EntryFill]:
    if minute_hour.empty:
        return None
    minute_hour = minute_hour.sort_index()
    target_time = minute_hour.index[0] + pd.Timedelta(minutes=delay_minutes)
    if target_time not in minute_hour.index:
        return None
    row = minute_hour.loc[target_time]
    raw = float(row["open"])
    market_slippage = market_slippage_bps / 10_000.0
    fill = raw * (1.0 + market_slippage)
    return EntryFill(
        timestamp=target_time,
        raw_price=raw,
        fill_price=fill,
        fee_rate=BACKTEST_TAKER_FEE,
        slippage_per_unit=fill - raw,
        liquidity="taker",
        route=route,
        baseline_open=float(minute_hour.iloc[0]["open"]),
    )


def _maker_limit_fill(
    minute_hour: pd.DataFrame,
    signal_close: float,
    limit_offset_bps: float,
    wait_minutes: int,
) -> Optional[EntryFill]:
    if minute_hour.empty or wait_minutes <= 0:
        return None
    minute_hour = minute_hour.sort_index()
    target = signal_close * (1.0 - limit_offset_bps / 10_000.0)
    search = minute_hour.iloc[:wait_minutes]

    # La orden se envia una sola vez al inicio de la hora. Para considerarla
    # post-only/maker, el mercado debe comenzar estrictamente por encima del
    # BUY LIMIT. Si ya abre en/por debajo, modelamos rechazo/no colocacion y
    # dejamos que la variante hibrida use su fallback MARKET.
    if search.empty or float(search.iloc[0]["open"]) <= target:
        return None

    # Una vez colocada, la LIMIT permanece descansando durante toda la ventana.
    # Exigimos que el precio atraviese el limite (low < target), no un simple
    # toque, para reducir el optimismo por posicion en cola.
    for ts, row in search.iterrows():
        if float(row["low"]) < target:
            return EntryFill(
                timestamp=ts,
                raw_price=target,
                fill_price=target,
                fee_rate=BACKTEST_MAKER_FEE,
                slippage_per_unit=0.0,
                liquidity="maker",
                route="limit_maker",
                baseline_open=float(minute_hour.iloc[0]["open"]),
            )
    return None


def resolve_entry(
    variant: EntryVariant,
    minute_hour: pd.DataFrame,
    signal_close: float,
    market_slippage_bps: float,
) -> Optional[EntryFill]:
    if minute_hour.empty:
        return None

    minute_hour = minute_hour.sort_index()

    if variant.kind == "market":
        return _market_entry_fill(
            minute_hour,
            variant.delay_minutes,
            market_slippage_bps,
            route=f"market_{variant.delay_minutes}m",
        )

    if variant.kind in {"limit_only", "conditional"}:
        maker = _maker_limit_fill(
            minute_hour,
            signal_close,
            variant.limit_offset_bps,
            variant.wait_minutes,
        )
        if maker is not None:
            return maker

        # LIMIT pura: si no existe pullback suficiente, cancelamos la senal.
        if variant.kind == "limit_only":
            return None

        # Fallback condicional: solo perseguimos la senal si, al expirar la
        # LIMIT, el mercado sigue cerca del precio que estaba disponible en t=0.
        # La referencia es el open del primer minuto tras la senal.
        fallback = _market_entry_fill(
            minute_hour,
            variant.wait_minutes,
            market_slippage_bps,
            route="market_fallback_conditional",
        )
        if fallback is None:
            return None

        baseline_open = float(minute_hour.iloc[0]["open"])
        max_raw_price = baseline_open * (1.0 + variant.max_chase_bps / 10_000.0)
        if fallback.raw_price <= max_raw_price:
            return fallback

        # Se ha escapado demasiado: no perseguimos el precio.
        return None

    raise ValueError(f"Entry kind no valido: {variant.kind}")


# -----------------------------------------------------------------------------
# Simulacion.
# -----------------------------------------------------------------------------
def _profit_factor(trades: pd.DataFrame, column: str = "pnl") -> float:
    if trades.empty:
        return float("nan")
    gains = float(trades.loc[trades[column] > 0, column].sum())
    losses = float(-trades.loc[trades[column] < 0, column].sum())
    if losses == 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def _update_excursions(position: OpenPosition, high: float, low: float) -> None:
    position.max_favorable_distance = max(
        position.max_favorable_distance,
        max(0.0, high - position.entry_price),
    )
    position.max_adverse_distance = max(
        position.max_adverse_distance,
        max(0.0, position.entry_price - low),
    )


def _risk_manager(risk_reward_ratio: float) -> RiskManager:
    return RiskManager(
        risk_per_trade=RISK_PER_TRADE,
        max_position_pct=MAX_POSITION_PCT,
        atr_multiplier=ATR_MULTIPLIER,
        risk_reward_ratio=risk_reward_ratio,
        cash_buffer_pct=CASH_BUFFER_PCT,
    )


def simulate_variant(
    signals: pd.DataFrame,
    minute_cache: pd.DataFrame,
    variant: EntryVariant,
    market_slippage_bps: float,
    risk_reward_ratio: float,
    entry_start: Optional[pd.Timestamp] = None,
    entry_end: Optional[pd.Timestamp] = None,
) -> SimulationResult:
    wallet = float(INITIAL_CAPITAL)
    risk = _risk_manager(risk_reward_ratio)
    slippage = market_slippage_bps / 10_000.0
    position: Optional[OpenPosition] = None
    trades: list[ClosedTrade] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []
    attempted = 0
    filled_entries = 0
    missed_entries = 0
    maker_entries = 0
    taker_entries = 0

    def close_position(
        *,
        exit_time: pd.Timestamp,
        fill: float,
        raw_reference: float,
        exit_fee_rate: float,
        exit_slippage_per_unit: float,
        reason: str,
    ) -> None:
        nonlocal wallet, position
        assert position is not None

        _update_excursions(position, fill, fill)
        gross_before_costs = (raw_reference - position.raw_entry_price) * position.amount
        gross_pnl = (fill - position.entry_price) * position.amount
        exit_fee = fill * position.amount * exit_fee_rate
        fees = position.entry_fee + exit_fee
        slippage_cost = position.entry_slippage_cost + exit_slippage_per_unit * position.amount
        pnl = gross_pnl - fees
        wallet += gross_pnl - exit_fee
        holding_hours = max(0.0, (exit_time - position.entry_time).total_seconds() / 3600.0)

        trades.append(
            ClosedTrade(
                signal_time=position.signal_time,
                entry_time=position.entry_time,
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=fill,
                amount=position.amount,
                stop_price=position.stop_price,
                take_profit_price=position.take_profit_price,
                exit_reason=reason,
                entry_liquidity=position.entry_liquidity,
                entry_route=position.entry_route,
                entry_delay_minutes=position.entry_delay_minutes,
                entry_improvement_bps=position.entry_improvement_bps,
                gross_before_costs=gross_before_costs,
                gross_pnl=gross_pnl,
                fees=fees,
                slippage_cost=slippage_cost,
                pnl=pnl,
                r_multiple=pnl / position.initial_risk if position.initial_risk > 0 else float("nan"),
                mfe_r=position.max_favorable_distance / position.stop_distance,
                mae_r=position.max_adverse_distance / position.stop_distance,
                holding_hours=holding_hours,
            )
        )
        position = None

    def process_bar(ts: pd.Timestamp, row: pd.Series) -> bool:
        """Procesa un bar para una posicion existente. True si cierra."""
        nonlocal position
        assert position is not None

        bar_open = float(row["open"])
        high = float(row["high"])
        low = float(row["low"])

        # Gap por debajo del stop: salida al open, no al stop teorico.
        if bar_open <= position.stop_price:
            raw = bar_open
            fill = raw * (1.0 - slippage)
            close_position(
                exit_time=ts,
                fill=fill,
                raw_reference=raw,
                exit_fee_rate=BACKTEST_TAKER_FEE,
                exit_slippage_per_unit=raw - fill,
                reason="stop_loss",
            )
            return True

        stop_hit = low <= position.stop_price
        tp_hit = high >= position.take_profit_price

        # Convencion conservadora: si OHLC no permite conocer el orden y toca ambos,
        # asumimos el caso conservador (SL primero).
        if stop_hit:
            raw = position.stop_price
            fill = raw * (1.0 - slippage)
            close_position(
                exit_time=ts,
                fill=fill,
                raw_reference=raw,
                exit_fee_rate=BACKTEST_TAKER_FEE,
                exit_slippage_per_unit=raw - fill,
                reason="stop_loss",
            )
            return True

        if tp_hit:
            raw = position.take_profit_price
            fill = raw
            close_position(
                exit_time=ts,
                fill=fill,
                raw_reference=raw,
                exit_fee_rate=BACKTEST_MAKER_FEE,
                exit_slippage_per_unit=0.0,
                reason="take_profit",
            )
            return True

        _update_excursions(position, high, low)
        return False

    if entry_start is not None:
        start_scan = entry_start - pd.Timedelta(TIMEFRAME)
        work = signals.loc[signals.index >= start_scan]
    else:
        work = signals

    for i in range(1, len(work)):
        ts = work.index[i]
        row = work.iloc[i]
        previous = work.iloc[i - 1]
        previous_ts = work.index[i - 1]
        closed_this_hour = False

        if entry_end is not None and ts > entry_end and position is None:
            break

        # Posicion que ya venia abierta al comenzar esta hora: usamos OHLC 1h.
        if position is not None:
            closed_this_hour = process_bar(ts, row)

        entry_in_window = (
            (entry_start is None or ts >= entry_start)
            and (entry_end is None or ts <= entry_end)
        )

        if (
            position is None
            and not closed_this_hour
            and entry_in_window
            and bool(previous.get("entry_long", False))
        ):
            attempted += 1
            minute_hour = _minute_window(minute_cache, ts)
            fill = resolve_entry(
                variant,
                minute_hour,
                signal_close=float(previous["close"]),
                market_slippage_bps=market_slippage_bps,
            )

            if fill is None:
                missed_entries += 1
            else:
                atr = float(previous.get("atr", np.nan))
                if math.isfinite(atr) and atr > 0 and wallet > 0:
                    plan = risk.build_plan(
                        side="long",
                        account_equity=wallet,
                        free_margin=wallet,
                        expected_entry_price=fill.fill_price,
                        atr=atr,
                        leverage=LEVERAGE,
                    )
                    amount = plan.base_amount
                    entry_fee = amount * fill.fill_price * fill.fee_rate
                    wallet -= entry_fee
                    filled_entries += 1
                    if fill.liquidity == "maker":
                        maker_entries += 1
                    else:
                        taker_entries += 1
                    position = OpenPosition(
                        entry_time=fill.timestamp,
                        signal_time=previous_ts,
                        entry_hour=ts,
                        raw_entry_price=fill.raw_price,
                        entry_price=fill.fill_price,
                        amount=amount,
                        entry_fee=entry_fee,
                        entry_slippage_cost=fill.slippage_per_unit * amount,
                        stop_price=fill.fill_price - plan.stop_distance,
                        take_profit_price=fill.fill_price + plan.stop_distance * risk_reward_ratio,
                        stop_distance=plan.stop_distance,
                        initial_risk=amount * plan.stop_distance,
                        entry_liquidity=fill.liquidity,
                        entry_route=fill.route,
                        entry_delay_minutes=(fill.timestamp - ts).total_seconds() / 60.0,
                        entry_improvement_bps=fill.improvement_bps,
                    )

                    # Para la hora de entrada usamos 1m DESDE el fill.
                    # MARKET entra al open del minuto, por lo que todo su OHLC
                    # es posterior al fill. En una LIMIT maker el fill ocurre en
                    # algun punto intraminuto: permitimos SL en ese mismo minuto
                    # (conservador y temporalmente posible al cruzar hacia abajo),
                    # pero NO permitimos TP usando su high porque podria haberse
                    # producido antes del fill. El TP se evalua desde el minuto
                    # siguiente.
                    after_fill = minute_hour.loc[minute_hour.index >= fill.timestamp]
                    for minute_ts, minute_row in after_fill.iterrows():
                        if position is None:
                            break

                        if fill.liquidity == "maker" and minute_ts == fill.timestamp:
                            low = float(minute_row["low"])
                            if low <= position.stop_price:
                                raw = position.stop_price
                                exit_fill = raw * (1.0 - slippage)
                                close_position(
                                    exit_time=minute_ts,
                                    fill=exit_fill,
                                    raw_reference=raw,
                                    exit_fee_rate=BACKTEST_TAKER_FEE,
                                    exit_slippage_per_unit=raw - exit_fill,
                                    reason="stop_loss",
                                )
                                closed_this_hour = True
                                break

                            # No usamos el high del minuto del fill LIMIT para
                            # MFE/TP porque no conocemos el orden intrabar.
                            position.max_adverse_distance = max(
                                position.max_adverse_distance,
                                max(0.0, position.entry_price - low),
                            )
                            continue

                        if process_bar(minute_ts, minute_row):
                            closed_this_hour = True
                            break

        equity = wallet
        if position is not None:
            equity += (float(row["close"]) - position.entry_price) * position.amount
        equity_values.append(equity)
        equity_index.append(ts)

    if not equity_values:
        equity_values = [float(INITIAL_CAPITAL)]
        equity_index = [signals.index[0]]

    equity_curve = pd.Series(equity_values, index=equity_index)
    running_max = equity_curve.cummax()
    max_dd = float((equity_curve / running_max - 1.0).min())
    trade_df = pd.DataFrame([asdict(t) for t in trades])

    return SimulationResult(
        variant=variant.name,
        risk_reward_ratio=risk_reward_ratio,
        slippage_bps=market_slippage_bps,
        initial_capital=INITIAL_CAPITAL,
        final_equity=float(equity_curve.iloc[-1]),
        total_return=float(equity_curve.iloc[-1] / INITIAL_CAPITAL - 1.0),
        max_drawdown=max_dd,
        attempted_signals=attempted,
        filled_entries=filled_entries,
        missed_entries=missed_entries,
        maker_entries=maker_entries,
        taker_entries=taker_entries,
        trades=trade_df,
    )


def metrics(result: SimulationResult) -> dict[str, float | str | int]:
    trades = result.trades
    n = len(trades)
    return {
        "variant": result.variant,
        "rr": result.risk_reward_ratio,
        "slippage_bps": result.slippage_bps,
        "attempted": result.attempted_signals,
        "filled": result.filled_entries,
        "missed": result.missed_entries,
        "fill_rate": (
            result.filled_entries / result.attempted_signals
            if result.attempted_signals
            else float("nan")
        ),
        "trades": n,
        "maker_share": (result.maker_entries / result.filled_entries if result.filled_entries else float("nan")),
        "taker_share": (result.taker_entries / result.filled_entries if result.filled_entries else float("nan")),
        "avg_entry_delay_min": float(trades["entry_delay_minutes"].mean()) if n else float("nan"),
        "win_rate": float((trades["pnl"] > 0).mean()) if n else float("nan"),
        "net_pf": _profit_factor(trades, "pnl"),
        "gross_pf": _profit_factor(trades, "gross_before_costs"),
        "net_expectancy": float(trades["pnl"].mean()) if n else float("nan"),
        "net_pnl": float(trades["pnl"].sum()) if n else 0.0,
        "fees": float(trades["fees"].sum()) if n else 0.0,
        "slippage_cost": float(trades["slippage_cost"].sum()) if n else 0.0,
        "return": result.total_return,
        "max_dd": result.max_drawdown,
        "entry_improvement_bps": float(trades["entry_improvement_bps"].mean()) if n else float("nan"),
        "holding_median_h": float(trades["holding_hours"].median()) if n else float("nan"),
    }


def _build_signals() -> pd.DataFrame:
    print(f"Descargando 1h: {BACKTEST_START} -> {BACKTEST_END}")
    df = fetch_binance_history(SYMBOL, TIMEFRAME, BACKTEST_START, BACKTEST_END)
    print(f"Descargando {REGIME_TIMEFRAME} para regimen...")
    regime_df = fetch_binance_history(
        SYMBOL,
        REGIME_TIMEFRAME,
        BACKTEST_START,
        BACKTEST_END,
    )

    min_rows = max(TREND_WINDOW, LOOKBACK_WINDOW, ATR_WINDOW) + 50
    if len(df) < min_rows:
        raise RuntimeError(f"Historia 1h insuficiente: {len(df)} velas")
    if len(regime_df) < REGIME_EMA_WINDOW + REGIME_SLOPE_LOOKBACK + 20:
        raise RuntimeError(f"Historia {REGIME_TIMEFRAME} insuficiente")

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
    )
    return strategy.calculate_signals(df, regime_df=regime_df)


def _print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 184)
    print(
        "BACKTEST V8 FINAL | LONG confirmed + 1h SMA200 AND 4h EMA200 | "
        "TP maker | baseline vs pullback/no-chase"
    )
    print("=" * 184)
    print(
        "variant          RR  slip attempts fill% maker% delay trades win% "
        "grossPF netPF  netExp entryImp fees  slipCost return   maxDD"
    )
    for r in rows:
        print(
            f"{r['variant']:<16} {r['rr']:>3.1f} {r['slippage_bps']:>4.1f}bp "
            f"{r['attempted']:>8} {r['fill_rate']:>5.1%} {r['maker_share']:>6.1%} "
            f"{r['avg_entry_delay_min']:>5.2f}m {r['trades']:>6} "
            f"{r['win_rate']:>5.1%} {r['gross_pf']:>7.3f} {r['net_pf']:>5.3f} "
            f"{r['net_expectancy']:>7.3f} {r['entry_improvement_bps']:>8.2f}bp "
            f"{r['fees']:>6.1f} {r['slippage_cost']:>8.1f} "
            f"{r['return']:>7.2%} {r['max_dd']:>7.2%}"
        )


def _annual_rows(signals: pd.DataFrame, minute_cache: pd.DataFrame) -> list[dict]:
    years = sorted(set(int(y) for y in signals.index.year))
    # Criterio temporal V8 final: primeros 2 anos como historia previa.
    test_years = years[2:]
    records: list[dict] = []
    tz = signals.index.tz

    for variant in ENTRY_VARIANTS:
        for rr in RR_GRID:
            for slip in MARKET_SLIPPAGE_BPS:
                for year in test_years:
                    start = pd.Timestamp(year=year, month=1, day=1, tz=tz)
                    end = pd.Timestamp(
                        year=year,
                        month=12,
                        day=31,
                        hour=23,
                        minute=59,
                        second=59,
                        tz=tz,
                    )
                    result = simulate_variant(
                        signals,
                        minute_cache,
                        variant,
                        slip,
                        rr,
                        entry_start=start,
                        entry_end=end,
                    )
                    row = metrics(result)
                    row["year"] = year
                    records.append(row)
    return records


def _print_annual(records: list[dict]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        return
    print("\nNET PF POR ANO DE TEST (V8 FINAL)")
    for slip in MARKET_SLIPPAGE_BPS:
        for rr in RR_GRID:
            print(f"\n[market slippage={slip:.1f} bp | RR={rr:.2f}]")
            subset = df[
                (df["slippage_bps"] == slip)
                & (df["rr"] == rr)
            ]
            pivot = subset.pivot(
                index="year",
                columns="variant",
                values="net_pf",
            )
            print(pivot.to_string(float_format=lambda x: f"{x:6.3f}"))


def _print_robustness(summary_rows: list[dict], annual_records: list[dict]) -> None:
    summary = pd.DataFrame(summary_rows)
    annual = pd.DataFrame(annual_records)
    if summary.empty or annual.empty:
        return

    target_slip = min(MARKET_SLIPPAGE_BPS, key=lambda x: abs(x - 2.0))
    s2 = summary[summary["slippage_bps"] == target_slip].copy()
    a2 = annual[annual["slippage_bps"] == target_slip].copy()
    yearly = (
        a2.groupby(["variant", "rr"], as_index=False)
        .agg(
            worst_year_pf=("net_pf", "min"),
            median_year_pf=("net_pf", "median"),
            positive_years=("net_pnl", lambda x: int((x > 0).sum())),
            years=("year", "nunique"),
        )
    )
    table = s2.merge(yearly, on=["variant", "rr"], how="left")
    table = table.sort_values(
        ["positive_years", "worst_year_pf", "net_pf"],
        ascending=[False, False, False],
    )

    print(
        f"\nROBUSTEZ @ {target_slip:.1f} bp "
        "(orden: anos positivos, peor PF anual, PF global)"
    )
    print(
        "variant          RR  netPF return  maxDD fill% maker% +years "
        "worstPF medianPF"
    )
    for _, r in table.iterrows():
        print(
            f"{r['variant']:<16} {r['rr']:>3.1f} {r['net_pf']:>5.3f} "
            f"{r['return']:>6.2%} {r['max_dd']:>6.2%} {r['fill_rate']:>5.1%} "
            f"{r['maker_share']:>6.1%} {int(r['positive_years']):>2}/{int(r['years']):<2} "
            f"{r['worst_year_pf']:>7.3f} {r['median_year_pf']:>8.3f}"
        )


def _validate_frozen_v8() -> None:
    validate_settings()
    if ENTRY_MODE != "confirmed":
        raise RuntimeError("V8 final requiere ENTRY_MODE=confirmed.")
    if REGIME_MODE != "1h_sma200_4h_ema200":
        raise RuntimeError(
            "V8 final requiere REGIME_MODE=1h_sma200_4h_ema200."
        )
    if TAKE_PROFIT_ORDER_MODE != "maker_limit":
        raise RuntimeError("V8 final requiere TAKE_PROFIT_ORDER_MODE=maker_limit.")


def run() -> None:
    _validate_frozen_v8()
    print("Backtest V8 FINAL: baseline MARKET vs pullback LIMIT/no-chase.")
    print(
        f"Estrategia: LONG confirmed | regimen={REGIME_MODE} | "
        f"ATR x{ATR_MULTIPLIER:.2f}"
    )
    print(
        f"V8 final: LIMIT -{ENTRY_LIMIT_OFFSET_BPS:.1f}bp desde signal close | "
        f"wait={ENTRY_LIMIT_WAIT_SECONDS // 60}m | "
        f"fallback cap=+{ENTRY_FALLBACK_CAP_BPS:.1f}bp vs t=0"
    )
    print(
        f"Fees: taker={BACKTEST_TAKER_FEE:.6f} | "
        f"maker={BACKTEST_MAKER_FEE:.6f}"
    )
    print(f"RR grid: {RR_GRID}")
    print(f"Market slippage grid: {MARKET_SLIPPAGE_BPS} bp")
    print(f"Entry modes: {tuple(v.name for v in ENTRY_VARIANTS)}")
    print(
        "Modelo LIMIT conservador: debe poder descansar postOnly en t=0 y "
        "el low debe atravesar el limite. TP no se evalua en el mismo minuto "
        "del fill LIMIT."
    )

    signals = _build_signals()
    minute_cache = ensure_minute_cache(signals)

    rows: list[dict] = []
    trade_frames: list[pd.DataFrame] = []
    for variant in ENTRY_VARIANTS:
        for rr in RR_GRID:
            for slip in MARKET_SLIPPAGE_BPS:
                result = simulate_variant(
                    signals,
                    minute_cache,
                    variant,
                    slip,
                    rr,
                )
                rows.append(metrics(result))
                if not result.trades.empty:
                    detail = result.trades.copy()
                    detail.insert(0, "slippage_bps", slip)
                    detail.insert(0, "rr", rr)
                    detail.insert(0, "variant", variant.name)
                    trade_frames.append(detail)

    _print_summary(rows)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(SUMMARY_FILE, index=False)
    print(f"\nResumen guardado en {SUMMARY_FILE}")

    if trade_frames:
        TRADES_FILE.parent.mkdir(parents=True, exist_ok=True)
        pd.concat(trade_frames, ignore_index=True).to_csv(TRADES_FILE, index=False)
        print(f"Detalle de trades guardado en {TRADES_FILE}")

    annual = _annual_rows(signals, minute_cache)
    ANNUAL_FILE.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(annual).to_csv(ANNUAL_FILE, index=False)
    _print_annual(annual)
    _print_robustness(rows, annual)
    print(f"\nDetalle anual guardado en {ANNUAL_FILE}")

    print(
        "\nCriterio final: no elegir RR por PF global aislado. Para cambiar "
        "RR=1.5 se exige mantener estabilidad anual, peor ano y DD con la entrada V8 final."
    )


if __name__ == "__main__":
    run()
