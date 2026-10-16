from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import math
import re
import time

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    BREAK_EVEN_TRIGGER_R,
    CANDLE_CLOSE_DELAY_SECONDS,
    CASH_BUFFER_PCT,
    ENABLE_LONGS,
    ENABLE_SHORTS,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_FALLBACK_MAX_LATENESS_SECONDS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    ENTRY_MODE,
    ENTRY_PENDING_POLL_SECONDS,
    ERROR_RETRY_SECONDS,
    EXIT_MODE,
    LEVERAGE,
    LIVE_MAX_CONCURRENT_POSITIONS,
    LIVE_SYMBOLS,
    LOOKBACK_WINDOW,
    MAX_HOLDING_HOURS,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_POSITION_PCT,
    MEAN_EXIT_Z,
    PARTIAL_TP_MOVE_STOP_TO_BE,
    PARTIAL_TP_TRIGGER_R,
    POSITION_POLL_SECONDS,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    RISK_PER_TRADE,
    RISK_REWARD_RATIO,
    SIGNAL_MAX_AGE_SECONDS,
    STATE_FILE,
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


TP_EXIT_MODES = {"fixed_rr", "fixed_rr_time", "hybrid_rr_mean", "hybrid_rr_mean_time"}
MEAN_EXIT_MODES = {"mean", "hybrid_rr_mean", "hybrid_rr_mean_time"}
TIME_EXIT_MODES = {"time_stop", "fixed_rr_time", "hybrid_rr_mean_time"}
VALID_EXIT_MODES = TP_EXIT_MODES | MEAN_EXIT_MODES | TIME_EXIT_MODES


@dataclass
class LiveContext:
    symbol: str
    engine: BinanceExecutionEngine
    strategy: MeanReversionStrategy
    risk: RiskManager
    data_limit: int
    regime_data_limit: int


def _slug(symbol: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", symbol).strip("_").lower()


def _state_file_for_symbol(symbol: str) -> str:
    if len(LIVE_SYMBOLS) == 1:
        return STATE_FILE
    base = Path(STATE_FILE)
    suffix = base.suffix or ".json"
    return str(base.with_name(f"{base.stem}_{_slug(symbol)}{suffix}"))


def _signal_side(latest) -> str | None:
    if ENABLE_LONGS and bool(latest.get("entry_long", False)):
        return "long"
    if ENABLE_SHORTS and bool(latest.get("entry_short", False)):
        return "short"
    return None


def _period_seconds() -> float:
    return timeframe_to_timedelta(TIMEFRAME).total_seconds()


def _seconds_until_next_candle_check(now: datetime | None = None) -> float:
    now = now or datetime.now(timezone.utc)
    period = _period_seconds()
    ts = now.timestamp()
    boundary = math.floor(ts / period) * period
    target = boundary + CANDLE_CLOSE_DELAY_SECONDS
    if ts < target:
        return max(0.0, target - ts)
    return max(0.0, boundary + period + CANDLE_CLOSE_DELAY_SECONDS - ts)


def _inside_candle_window(now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    elapsed = now.timestamp() % _period_seconds()
    return CANDLE_CLOSE_DELAY_SECONDS <= elapsed <= (
        CANDLE_CLOSE_DELAY_SECONDS + SIGNAL_MAX_AGE_SECONDS
    )


def _fetch_latest(ctx: LiveContext):
    df = fetch_binance_klines(
        symbol=ctx.symbol,
        timeframe=TIMEFRAME,
        limit=ctx.data_limit,
        closed_only=True,
    )
    if df.empty:
        raise RuntimeError(f"{ctx.symbol}: no hay velas cerradas.")
    regime_df = None
    if REGIME_MODE != "1h_sma200":
        regime_df = fetch_binance_klines(
            symbol=ctx.symbol,
            timeframe=REGIME_TIMEFRAME,
            limit=ctx.regime_data_limit,
            closed_only=True,
        )
        if regime_df.empty:
            raise RuntimeError(f"{ctx.symbol}: no hay velas {REGIME_TIMEFRAME} cerradas.")
    signals = ctx.strategy.calculate_signals(df, regime_df=regime_df)
    return signals.iloc[-1]


def _mean_exit_for_side(latest, side: str) -> bool:
    return bool(latest.get("exit_long_mean" if side == "long" else "exit_short_mean", False))


def _time_exit_reached(entry_time: str | None, now: datetime) -> bool:
    if not entry_time:
        return False
    try:
        entry = datetime.fromisoformat(entry_time)
        if entry.tzinfo is None:
            entry = entry.replace(tzinfo=timezone.utc)
    except ValueError:
        return False
    return (now - entry).total_seconds() >= MAX_HOLDING_HOURS * 3600


def _state_planned_risk(position_state: dict) -> float:
    if not position_state.get("known"):
        return 0.0
    state = position_state.get("state")
    if state is None:
        return 0.0
    if position_state.get("entry_pending"):
        return max(0.0, float(state.amount or 0.0) * float(state.pending_stop_distance or 0.0))
    if position_state.get("has_position"):
        position = position_state["position"]
        initial_stop = state.initial_stop_price or state.stop_price
        if not initial_stop:
            return 0.0
        return max(0.0, float(position.contracts) * abs(float(position.entry_price) - float(initial_stop)))
    return 0.0


def _active_count(states: dict[str, dict]) -> int:
    return sum(
        1
        for st in states.values()
        if st.get("known") and (st.get("entry_pending") or st.get("has_position"))
    )


def _aggregate_risk(states: dict[str, dict]) -> float:
    return sum(_state_planned_risk(st) for st in states.values())


def _maybe_arm_break_even(ctx: LiveContext, position_state: dict) -> None:
    if not position_state.get("has_position"):
        return
    state = position_state["state"]
    position = position_state["position"]
    if state.break_even_armed:
        return

    if (
        PARTIAL_TP_TRIGGER_R > 0
        and PARTIAL_TP_MOVE_STOP_TO_BE
        and state.partial_take_profit_filled
    ):
        new_state = ctx.engine.move_stop_to_break_even(
            ctx.symbol, reason="partial take profit completed"
        )
        print(
            f"{ctx.symbol}: break-even tras parcial | "
            f"entry={new_state.entry_price:.2f} SL={new_state.stop_price:.2f}"
        )
        return

    if BREAK_EVEN_TRIGGER_R <= 0:
        return
    initial_stop = state.initial_stop_price or state.stop_price
    if not state.entry_price or not initial_stop:
        return
    distance = abs(float(state.entry_price) - float(initial_stop))
    if distance <= 0:
        return
    trigger = (
        float(state.entry_price) + distance * BREAK_EVEN_TRIGGER_R
        if position.side == "long"
        else float(state.entry_price) - distance * BREAK_EVEN_TRIGGER_R
    )
    price = ctx.engine.get_market_price(ctx.symbol)
    reached = price >= trigger if position.side == "long" else price <= trigger
    if reached:
        new_state = ctx.engine.move_stop_to_break_even(
            ctx.symbol, reason=f"market reached {BREAK_EVEN_TRIGGER_R:.2f}R"
        )
        print(
            f"{ctx.symbol}: break-even activado | "
            f"entry={new_state.entry_price:.2f} SL={new_state.stop_price:.2f}"
        )


def _build_context(symbol: str) -> LiveContext:
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
    engine = BinanceExecutionEngine(state_file=_state_file_for_symbol(symbol))
    engine.configure_account(symbol)
    return LiveContext(
        symbol=symbol,
        engine=engine,
        strategy=strategy,
        risk=risk,
        data_limit=max(LOOKBACK_WINDOW, TREND_WINDOW, ATR_WINDOW) + 100,
        regime_data_limit=REGIME_EMA_WINDOW + REGIME_SLOPE_LOOKBACK + 100,
    )


def main() -> None:
    validate_settings()
    if EXIT_MODE not in VALID_EXIT_MODES:
        raise ValueError(f"EXIT_MODE={EXIT_MODE!r} no soportado: {sorted(VALID_EXIT_MODES)}")

    print(
        "Mean-reversion V9 | Binance USD-M | "
        f"symbols={LIVE_SYMBOLS} {TIMEFRAME} | {LEVERAGE}x | exit={EXIT_MODE} | "
        f"RR={RISK_REWARD_RATIO:.2f} | risk/trade={RISK_PER_TRADE:.2%} | "
        f"maxpos={MAX_POSITION_PCT:.0%} | portfolio-risk={MAX_PORTFOLIO_RISK_PCT:.2%} | "
        f"BE={BREAK_EVEN_TRIGGER_R:.2f}R | partial={PARTIAL_TP_TRIGGER_R:.2f}R"
    )
    contexts = {symbol: _build_context(symbol) for symbol in LIVE_SYMBOLS}

    try:
        while True:
            try:
                states: dict[str, dict] = {}
                unknown_symbols: list[str] = []

                # Primera pasada: reconciliacion y expiracion de entradas.
                for symbol, ctx in contexts.items():
                    st = ctx.engine.get_position_state(symbol)
                    if not st.get("known"):
                        unknown_symbols.append(symbol)
                        states[symbol] = st
                        print(f"{symbol}: UNKNOWN/fail-closed | {st.get('reason', 'sin detalle')}")
                        continue
                    if st.get("entry_pending"):
                        remaining = ctx.engine.pending_entry_remaining_seconds(st.get("state"))
                        if remaining is None:
                            unknown_symbols.append(symbol)
                            print(f"{symbol}: entrada pendiente sin expiracion valida.")
                        elif remaining <= 0:
                            resolution = ctx.engine.resolve_pending_entry(symbol)
                            print(f"{symbol}: pending -> {resolution.get('action')}")
                            st = ctx.engine.get_position_state(symbol)
                    states[symbol] = st

                global_fail_closed = bool(unknown_symbols)
                in_window = _inside_candle_window()

                # Segunda pasada: posiciones abiertas (BE, mean/time).
                for symbol, ctx in contexts.items():
                    st = states.get(symbol, {})
                    if not st.get("known") or not st.get("has_position"):
                        continue
                    _maybe_arm_break_even(ctx, st)
                    # Refrescar estado por si BE/partial lo modifico.
                    st = ctx.engine.get_position_state(symbol)
                    states[symbol] = st
                    if not st.get("known") or not st.get("has_position"):
                        continue

                    dynamic_exit = EXIT_MODE in (MEAN_EXIT_MODES | TIME_EXIT_MODES)
                    if dynamic_exit and in_window:
                        latest = _fetch_latest(ctx)
                        signal_ts = latest.name.isoformat()
                        if not ctx.engine.was_signal_processed(signal_ts):
                            state = st["state"]
                            position = st["position"]
                            mean_exit = EXIT_MODE in MEAN_EXIT_MODES and _mean_exit_for_side(
                                latest, position.side
                            )
                            timeout_exit = EXIT_MODE in TIME_EXIT_MODES and _time_exit_reached(
                                state.entry_time, datetime.now(timezone.utc)
                            )
                            ctx.engine.mark_signal_processed(signal_ts)
                            if mean_exit or timeout_exit:
                                reason = (
                                    f"mean_reversion z={float(latest['z_score']):.3f}"
                                    if mean_exit
                                    else f"time_stop {MAX_HOLDING_HOURS}h"
                                )
                                ctx.engine.close_position_market(symbol, reason)
                                print(f"{symbol}: posicion cerrada por {reason}")
                                states[symbol] = ctx.engine.get_position_state(symbol)
                                continue
                    state = st["state"]
                    position = st["position"]
                    print(
                        f"{symbol}: {position.side.upper()} activa | entry={position.entry_price:.2f} | "
                        f"SL={state.stop_price} TP={state.take_profit_price} | BE={state.break_even_armed}"
                    )

                # Tercera pasada: nuevas entradas solo en ventana de cierre y si
                # TODOS los simbolos son verificables. Un UNKNOWN bloquea el portfolio.
                if in_window and not global_fail_closed:
                    for symbol, ctx in contexts.items():
                        st = states[symbol]
                        if st.get("entry_pending") or st.get("has_position"):
                            continue
                        latest = _fetch_latest(ctx)
                        signal_ts = latest.name.isoformat()
                        if ctx.engine.was_signal_processed(signal_ts):
                            continue

                        side = _signal_side(latest)
                        if side is None:
                            ctx.engine.mark_signal_processed(signal_ts)
                            continue

                        candle_close_time = latest.name + timeframe_to_timedelta(TIMEFRAME)
                        age_seconds = (
                            datetime.now(timezone.utc) - candle_close_time.to_pydatetime()
                        ).total_seconds()
                        if age_seconds > SIGNAL_MAX_AGE_SECONDS:
                            ctx.engine.mark_signal_processed(signal_ts)
                            print(f"{symbol}: senal descartada por antiguedad ({age_seconds:.0f}s)")
                            continue

                        if _active_count(states) >= LIVE_MAX_CONCURRENT_POSITIONS:
                            ctx.engine.mark_signal_processed(signal_ts)
                            print(f"{symbol}: senal consumida por limite de concurrencia.")
                            continue

                        atr = float(latest["atr"])
                        if not math.isfinite(atr) or atr <= 0:
                            raise RuntimeError(f"{symbol}: ATR invalido {atr}")
                        market_price = ctx.engine.get_market_price(symbol)
                        account = ctx.engine.get_account_snapshot("USDT")
                        plan = ctx.risk.build_plan(
                            side=side,
                            account_equity=account.equity,
                            free_margin=account.free_margin,
                            expected_entry_price=market_price,
                            atr=atr,
                            leverage=LEVERAGE,
                        )

                        existing_risk = _aggregate_risk(states)
                        risk_room = max(0.0, account.equity * MAX_PORTFOLIO_RISK_PCT - existing_risk)
                        if risk_room <= 0:
                            ctx.engine.mark_signal_processed(signal_ts)
                            print(f"{symbol}: senal consumida; portfolio risk cap sin margen.")
                            continue
                        requested_amount = plan.base_amount
                        if plan.estimated_risk > risk_room:
                            requested_amount = risk_room / plan.stop_distance
                        amount = ctx.engine.normalize_amount(
                            symbol, base_amount=requested_amount, price=market_price
                        )
                        actual_risk = amount * plan.stop_distance
                        if actual_risk <= 0 or actual_risk > risk_room * 1.001:
                            ctx.engine.mark_signal_processed(signal_ts)
                            print(f"{symbol}: sizing no cabe en portfolio risk cap.")
                            continue

                        use_tp = EXIT_MODE in TP_EXIT_MODES
                        context = {
                            "strategy_version": "v9",
                            "signal_time": signal_ts,
                            "signal_close": float(latest["close"]),
                            "signal_z": float(latest.get("z_score", math.nan)),
                            "atr": atr,
                            "regime_mode": REGIME_MODE,
                            "expected_entry_price": market_price,
                            "risk_reward_ratio": RISK_REWARD_RATIO,
                            "risk_budget": plan.risk_budget,
                            "estimated_risk": actual_risk,
                            "risk_utilization": actual_risk / plan.risk_budget if plan.risk_budget else 0.0,
                            "limiting_factor": (
                                "portfolio_risk" if actual_risk + 1e-12 < plan.estimated_risk else plan.limiting_factor
                            ),
                            "portfolio_existing_risk": existing_risk,
                            "portfolio_risk_room": risk_room,
                            "exit_mode": EXIT_MODE,
                            "break_even_trigger_r": BREAK_EVEN_TRIGGER_R,
                            "partial_tp_trigger_r": PARTIAL_TP_TRIGGER_R,
                        }
                        result = ctx.engine.submit_passive_entry(
                            symbol=symbol,
                            side=side,
                            amount=amount,
                            reference_price=float(latest["close"]),
                            fallback_reference_price=market_price,
                            limit_offset_bps=ENTRY_LIMIT_OFFSET_BPS,
                            wait_seconds=ENTRY_LIMIT_WAIT_SECONDS,
                            fallback_cap_bps=ENTRY_FALLBACK_CAP_BPS,
                            fallback_max_lateness_seconds=ENTRY_FALLBACK_MAX_LATENESS_SECONDS,
                            stop_distance=plan.stop_distance,
                            risk_reward_ratio=RISK_REWARD_RATIO,
                            use_take_profit=use_tp,
                            signal_timestamp=signal_ts,
                            journal_context=context,
                        )
                        states[symbol] = {
                            "known": True,
                            "has_position": False,
                            "entry_pending": True,
                            "state": result["state"],
                        }
                        print(
                            f"{symbol}: {side.upper()} pendiente | notional~{amount*market_price:.2f} | "
                            f"risk~{actual_risk:.2f} | portfolio risk~{_aggregate_risk(states):.2f}"
                        )

                # Cadencia adaptativa: pending > posicion > flat alineado.
                any_pending = any(st.get("entry_pending") for st in states.values() if st.get("known"))
                any_position = any(st.get("has_position") for st in states.values() if st.get("known"))
                if global_fail_closed:
                    sleep_for = ERROR_RETRY_SECONDS
                elif any_pending:
                    sleep_for = ENTRY_PENDING_POLL_SECONDS
                elif any_position:
                    sleep_for = POSITION_POLL_SECONDS
                    if EXIT_MODE in (MEAN_EXIT_MODES | TIME_EXIT_MODES):
                        sleep_for = max(0.5, min(sleep_for, _seconds_until_next_candle_check()))
                else:
                    sleep_for = max(0.5, _seconds_until_next_candle_check())
                    print(f"Portfolio flat; proxima evaluacion en ~{sleep_for:.0f}s")
                time.sleep(sleep_for)

            except ExecutionError as exc:
                print(f"Ejecucion bloqueada por seguridad: {exc}")
                time.sleep(ERROR_RETRY_SECONDS)
            except Exception as exc:
                print(f"Error: {type(exc).__name__}: {exc}")
                time.sleep(ERROR_RETRY_SECONDS)

    except KeyboardInterrupt:
        print("\nBot detenido por el usuario.")


if __name__ == "__main__":
    main()
