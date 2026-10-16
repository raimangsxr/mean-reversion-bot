from __future__ import annotations

from datetime import datetime, timezone
import math
import time

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    CASH_BUFFER_PCT,
    ENABLE_LONGS,
    ENABLE_SHORTS,
    ENTRY_MODE,
    EXIT_MODE,
    LEVERAGE,
    LOOKBACK_WINDOW,
    MAX_HOLDING_BARS,
    MAX_POSITION_PCT,
    MEAN_EXIT_Z,
    POLL_INTERVAL_SECONDS,
    RISK_PER_TRADE,
    RISK_REWARD_RATIO,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    SIGNAL_MAX_AGE_SECONDS,
    SYMBOL,
    TIMEFRAME,
    TAKE_PROFIT_ORDER_MODE,
    TREND_WINDOW,
    Z_THRESHOLD,
    validate_settings,
)
from src.data.fetcher import fetch_binance_klines, timeframe_to_timedelta
from src.execution.engine import BinanceExecutionEngine, ExecutionError
from src.risk.manager import RiskManager
from src.strategy.mean_reversion import MeanReversionStrategy


def _signal_side(latest) -> str | None:
    if ENABLE_LONGS and bool(latest.get("entry_long", False)):
        return "long"
    if ENABLE_SHORTS and bool(latest.get("entry_short", False)):
        return "short"
    return None


def _mean_exit_for_side(latest, side: str) -> bool:
    if side == "long":
        return bool(latest.get("exit_long_mean", False))
    if side == "short":
        return bool(latest.get("exit_short_mean", False))
    return False


def _timeout_reached(entry_time_iso: str | None) -> bool:
    if not entry_time_iso:
        return False
    try:
        entry_time = datetime.fromisoformat(entry_time_iso)
        if entry_time.tzinfo is None:
            entry_time = entry_time.replace(tzinfo=timezone.utc)
    except ValueError:
        return False

    max_seconds = (
        timeframe_to_timedelta(TIMEFRAME).total_seconds() * MAX_HOLDING_BARS
    )
    elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds()
    return elapsed >= max_seconds


def main() -> None:
    validate_settings()

    print(
        f"Z-score bot | Binance USD-M Futures | {SYMBOL} {TIMEFRAME} | "
        f"One-way | {LEVERAGE}x | entry={ENTRY_MODE} | exit={EXIT_MODE} | "
        f"longs={ENABLE_LONGS} | shorts={ENABLE_SHORTS} | regime={REGIME_MODE} | "
        f"tp_mode={TAKE_PROFIT_ORDER_MODE}"
    )

    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW,
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=ATR_WINDOW,
        entry_mode=ENTRY_MODE,
        mean_exit_z=MEAN_EXIT_Z,
        regime_mode=REGIME_MODE,
        signal_timeframe=TIMEFRAME,
        regime_timeframe=REGIME_TIMEFRAME,
        regime_ema_window=REGIME_EMA_WINDOW,
        regime_slope_lookback=REGIME_SLOPE_LOOKBACK,
    )
    risk = RiskManager(
        risk_per_trade=RISK_PER_TRADE,
        max_position_pct=MAX_POSITION_PCT,
        atr_multiplier=ATR_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO,
        cash_buffer_pct=CASH_BUFFER_PCT,
    )
    engine = BinanceExecutionEngine()
    engine.configure_account(SYMBOL)

    data_limit = max(LOOKBACK_WINDOW, TREND_WINDOW, ATR_WINDOW) + 100
    regime_data_limit = (
        REGIME_EMA_WINDOW + REGIME_SLOPE_LOOKBACK + 100
    )

    try:
        while True:
            try:
                df = fetch_binance_klines(
                    symbol=SYMBOL,
                    timeframe=TIMEFRAME,
                    limit=data_limit,
                    closed_only=True,
                )
                if df.empty:
                    raise RuntimeError("No hay velas cerradas disponibles.")

                regime_df = None
                if REGIME_MODE != "1h_sma200":
                    regime_df = fetch_binance_klines(
                        symbol=SYMBOL,
                        timeframe=REGIME_TIMEFRAME,
                        limit=regime_data_limit,
                        closed_only=True,
                    )
                    if regime_df.empty:
                        raise RuntimeError(
                            f"No hay velas {REGIME_TIMEFRAME} cerradas para el regimen."
                        )

                signals = strategy.calculate_signals(df, regime_df=regime_df)
                latest = signals.iloc[-1]
                signal_timestamp = latest.name.isoformat()

                z = float(latest.get("z_score", math.nan))
                local_sma = float(latest.get("local_sma", math.nan))
                htf_close = float(latest.get("htf_close", math.nan))
                htf_ma = float(latest.get("htf_ma", math.nan))
                htf_slope_up = bool(latest.get("htf_slope_up", False))
                print(
                    f"[{latest.name}] close={latest['close']:.2f} | "
                    f"z={z:.3f} | regime={REGIME_MODE} | "
                    f"local_sma={local_sma:.2f} | "
                    f"htf_close={htf_close:.2f} | htf_ema={htf_ma:.2f} | "
                    f"htf_slope_up={htf_slope_up}"
                )

                # V7: reconciliamos SIEMPRE la posicion/protecciones en cada poll,
                # incluso si esta vela ya fue procesada. Asi SL/TP y partial fills no
                # esperan hasta la siguiente vela para ser detectados.
                position_state = engine.get_position_state(SYMBOL)
                if not position_state.get("known"):
                    print(
                        "Estado de posicion no verificable; se reintentara: "
                        f"{position_state.get('reason', 'sin detalle')}"
                    )
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                if engine.was_signal_processed(signal_timestamp):
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # -----------------------------------------------------------------
                # Gestion de posicion abierta
                # -----------------------------------------------------------------
                if position_state.get("has_position"):
                    position = position_state["position"]
                    state = position_state["state"]

                    if EXIT_MODE == "z_reversion":
                        mean_exit = _mean_exit_for_side(latest, position.side)
                        timeout_exit = _timeout_reached(state.entry_time)

                        if mean_exit or timeout_exit:
                            # Persistimos primero para impedir una segunda decision
                            # sobre la misma vela si el cierre queda ambiguo.
                            engine.mark_signal_processed(signal_timestamp)
                            reason = (
                                f"mean_reversion z={z:.3f}"
                                if mean_exit
                                else f"timeout {MAX_HOLDING_BARS} velas"
                            )
                            print(
                                f"Cerrando {position.side.upper()} en mercado: {reason}"
                            )
                            engine.close_position_market(SYMBOL, reason)
                            print("Posicion cerrada y ordenes de proteccion limpiadas.")
                            time.sleep(POLL_INTERVAL_SECONDS)
                            continue

                    engine.mark_signal_processed(signal_timestamp)
                    print(
                        f"Posicion {position.side.upper()} activa | "
                        f"contracts={position.contracts} | "
                        f"entry={position.entry_price:.2f} | "
                        f"SL={state.stop_price} | TP={state.take_profit_price}"
                    )
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # -----------------------------------------------------------------
                # Cuenta flat: solo aqui consideramos una nueva entrada.
                # -----------------------------------------------------------------
                candle_close_time = latest.name + timeframe_to_timedelta(TIMEFRAME)
                now = datetime.now(timezone.utc)
                age_seconds = (now - candle_close_time.to_pydatetime()).total_seconds()
                if age_seconds > SIGNAL_MAX_AGE_SECONDS:
                    engine.mark_signal_processed(signal_timestamp)
                    print(
                        f"Entrada descartada: la vela cerro hace {age_seconds:.0f}s "
                        f"(maximo {SIGNAL_MAX_AGE_SECONDS}s)."
                    )
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                side = _signal_side(latest)
                if side is None:
                    engine.mark_signal_processed(signal_timestamp)
                    print("Sin senal de entrada en esta vela.")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                atr = float(latest["atr"])
                if not math.isfinite(atr) or atr <= 0:
                    raise RuntimeError(f"ATR invalido: {atr}")

                market_price = engine.get_market_price(SYMBOL)
                account = engine.get_account_snapshot("USDT")
                plan = risk.build_plan(
                    side=side,
                    account_equity=account.equity,
                    free_margin=account.free_margin,
                    expected_entry_price=market_price,
                    atr=atr,
                    leverage=LEVERAGE,
                )
                amount = engine.normalize_amount(
                    SYMBOL,
                    base_amount=plan.base_amount,
                    price=market_price,
                )

                engine.mark_signal_processed(signal_timestamp)

                if EXIT_MODE == "fixed_rr":
                    exit_description = f"TP esperado~{plan.take_profit_price:.2f}"
                else:
                    exit_description = (
                        f"exit z={MEAN_EXIT_Z:.2f} / timeout={MAX_HOLDING_BARS} velas"
                    )

                print(
                    f"{side.upper()} | market~{market_price:.2f} | "
                    f"notional~{plan.notional:.2f} USDT | "
                    f"margin~{plan.required_margin:.2f} | "
                    f"risk~{plan.estimated_risk:.2f} USDT | "
                    f"SL esperado~{plan.stop_price:.2f} | {exit_description}"
                )

                result = engine.execute_trade_with_protection(
                    symbol=SYMBOL,
                    side=side,
                    amount=amount,
                    stop_distance=plan.stop_distance,
                    risk_reward_ratio=RISK_REWARD_RATIO,
                    use_take_profit=(EXIT_MODE == "fixed_rr"),
                    journal_context={
                        "signal_time": signal_timestamp,
                        "signal_close": float(latest["close"]),
                        "signal_z": z,
                        "atr": atr,
                        "regime_mode": REGIME_MODE,
                        "local_sma": local_sma,
                        "htf_close": htf_close,
                        "htf_ema": htf_ma,
                        "expected_entry_price": market_price,
                        "risk_reward_ratio": RISK_REWARD_RATIO,
                        "estimated_notional": plan.notional,
                        "estimated_risk": plan.estimated_risk,
                    },
                )
                final_state = result["state"]
                print(
                    f"{side.upper()} abierto | fill={final_state.entry_price:.2f} | "
                    f"SL={final_state.stop_price:.2f} | "
                    f"TP={final_state.take_profit_price} | "
                    f"tp_kind={final_state.take_profit_order_kind} | "
                    f"proteccion={final_state.protection_mode}"
                )

            except ExecutionError as exc:
                print(f"Ejecucion bloqueada por seguridad: {exc}")
            except Exception as exc:
                print(f"Error en ciclo principal: {type(exc).__name__}: {exc}")

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nBot detenido por el usuario.")


if __name__ == "__main__":
    main()
