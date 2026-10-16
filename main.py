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
    ENTRY_EXECUTION_MODE,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_FALLBACK_MAX_LATENESS_SECONDS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    ENTRY_MODE,
    ENTRY_PENDING_POLL_SECONDS,
    ERROR_RETRY_SECONDS,
    EXIT_MODE,
    LEVERAGE,
    LOOKBACK_WINDOW,
    MAX_POSITION_PCT,
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


def main() -> None:
    validate_settings()

    print(
        f"Z-score bot | Binance USD-M Futures | {SYMBOL} {TIMEFRAME} | "
        f"One-way | {LEVERAGE}x | entry={ENTRY_MODE} | exit={EXIT_MODE} | "
        f"longs={ENABLE_LONGS} | shorts={ENABLE_SHORTS} | regime={REGIME_MODE} | "
        f"entry_exec={ENTRY_EXECUTION_MODE} | tp_mode={TAKE_PROFIT_ORDER_MODE}"
    )

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
                # Reconciliar primero estado de ejecucion. Mientras exista una
                # entrada V8 pendiente no necesitamos descargar 1h/4h cada 10s.
                position_state = engine.get_position_state(SYMBOL)
                if not position_state.get("known"):
                    print(
                        "Estado de posicion/entrada no verificable; "
                        f"reintento en {ERROR_RETRY_SECONDS}s: "
                        f"{position_state.get('reason', 'sin detalle')}"
                    )
                    time.sleep(ERROR_RETRY_SECONDS)
                    continue

                pending_state = position_state.get("state")
                if position_state.get("entry_pending"):
                    remaining = engine.pending_entry_remaining_seconds(pending_state)
                    if remaining is None:
                        raise ExecutionError("Entrada pendiente sin expiracion valida.")
                    if remaining > 0:
                        print(
                            f"Entrada V8 final pendiente ({pending_state.entry_order_kind}) | "
                            f"limit={pending_state.pending_limit_price} | "
                            f"restan~{remaining:.0f}s"
                        )
                        time.sleep(ENTRY_PENDING_POLL_SECONDS)
                        continue

                    resolution = engine.resolve_pending_entry(SYMBOL)
                    action = resolution.get("action")
                    if action == "fallback_market":
                        final_state = resolution["state"]
                        print(
                            "LIMIT expirada; fallback MARKET dentro del cap | "
                            f"fill={final_state.entry_price:.2f} | "
                            f"SL={final_state.stop_price:.2f} | "
                            f"TP={final_state.take_profit_price}"
                        )
                    elif action == "limit_filled":
                        final_state = resolution["state"]
                        print(
                            "LIMIT ejecutada y protegida | "
                            f"fill={final_state.entry_price:.2f} | "
                            f"SL={final_state.stop_price:.2f} | "
                            f"TP={final_state.take_profit_price}"
                        )
                    elif action == "cancelled_no_chase":
                        print(
                            "Senal V8 cancelada: el precio supero el cap no-chase "
                            f"({resolution.get('current_price'):.2f} > "
                            f"{resolution.get('cap_price'):.2f})."
                        )
                    elif action == "cancelled_late":
                        print("Senal V8 cancelada: fallback demasiado tarde.")
                    time.sleep(ENTRY_PENDING_POLL_SECONDS)
                    continue

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

                if engine.was_signal_processed(signal_timestamp):
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                # -----------------------------------------------------------------
                # Gestion de posicion abierta
                # -----------------------------------------------------------------
                if position_state.get("has_position"):
                    position = position_state["position"]
                    state = position_state["state"]

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
                # Primero comprobamos si existe realmente una senal. La ventana
                # temporal es solo tolerancia operativa, no una regla de trading.
                side = _signal_side(latest)
                if side is None:
                    engine.mark_signal_processed(signal_timestamp)
                    print("Sin senal de entrada en esta vela.")
                    time.sleep(POLL_INTERVAL_SECONDS)
                    continue

                candle_close_time = latest.name + timeframe_to_timedelta(TIMEFRAME)
                now = datetime.now(timezone.utc)
                age_seconds = (now - candle_close_time.to_pydatetime()).total_seconds()
                if age_seconds > SIGNAL_MAX_AGE_SECONDS:
                    engine.mark_signal_processed(signal_timestamp)
                    print(
                        f"Senal {side.upper()} descartada: "
                        f"la vela cerro hace {age_seconds:.0f}s "
                        f"(maximo {SIGNAL_MAX_AGE_SECONDS}s)."
                    )
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

                exit_description = f"TP esperado~{plan.take_profit_price:.2f}"

                journal_context = {
                    "signal_time": signal_timestamp,
                    "signal_close": float(latest["close"]),
                    "signal_z": z,
                    "atr": atr,
                    "regime_mode": REGIME_MODE,
                    "local_sma": local_sma,
                    "htf_close": htf_close,
                    "htf_ema": htf_ma,
                    "expected_entry_price": market_price,
                    "limit_reference_price": float(latest["close"]),
                    "fallback_reference_price": market_price,
                    "risk_reward_ratio": RISK_REWARD_RATIO,
                    "estimated_notional": plan.notional,
                    "estimated_risk": plan.estimated_risk,
                }

                print(
                    f"{side.upper()} | signal_close={float(latest['close']):.2f} | "
                    f"t0_reference~{market_price:.2f} | "
                    f"notional~{plan.notional:.2f} USDT | "
                    f"margin~{plan.required_margin:.2f} | "
                    f"risk~{plan.estimated_risk:.2f} USDT | "
                    f"SL esperado~{plan.stop_price:.2f} | {exit_description}"
                )

                result = engine.submit_passive_entry(
                    symbol=SYMBOL,
                    side=side,
                    amount=amount,
                    reference_price=float(latest["close"]),
                    fallback_reference_price=market_price,
                    limit_offset_bps=ENTRY_LIMIT_OFFSET_BPS,
                    wait_seconds=ENTRY_LIMIT_WAIT_SECONDS,
                    fallback_cap_bps=ENTRY_FALLBACK_CAP_BPS,
                    fallback_max_lateness_seconds=(
                        ENTRY_FALLBACK_MAX_LATENESS_SECONDS
                    ),
                    stop_distance=plan.stop_distance,
                    risk_reward_ratio=RISK_REWARD_RATIO,
                    use_take_profit=(EXIT_MODE == "fixed_rr"),
                    signal_timestamp=signal_timestamp,
                    journal_context=journal_context,
                )
                pending = result["state"]
                print(
                    f"{side.upper()} V8 final pendiente | "
                    f"limit_ref={pending.pending_reference_price:.2f} | "
                    f"t0_ref={pending.pending_fallback_reference_price:.2f} | "
                    f"limit={pending.pending_limit_price:.2f} "
                    f"(-{ENTRY_LIMIT_OFFSET_BPS:.1f}bp LONG / simetrico SHORT) | "
                    f"espera={ENTRY_LIMIT_WAIT_SECONDS}s | "
                    f"fallback_cap={ENTRY_FALLBACK_CAP_BPS:.1f}bp"
                )
                time.sleep(ENTRY_PENDING_POLL_SECONDS)
                continue

            except ExecutionError as exc:
                print(
                    f"Ejecucion bloqueada por seguridad: {exc} | "
                    f"reintento en {ERROR_RETRY_SECONDS}s"
                )
                time.sleep(ERROR_RETRY_SECONDS)
                continue
            except Exception as exc:
                print(
                    f"Error en ciclo principal: {type(exc).__name__}: {exc} | "
                    f"reintento en {ERROR_RETRY_SECONDS}s"
                )
                time.sleep(ERROR_RETRY_SECONDS)
                continue

            time.sleep(POLL_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nBot detenido por el usuario.")


if __name__ == "__main__":
    main()
