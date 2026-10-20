from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import math
import re
import time

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    BREAK_EVEN_TRIGGER_R,
    BUILD_SHA,
    CANDLE_CLOSE_DELAY_SECONDS,
    CASH_BUFFER_PCT,
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
    MAX_HOLDING_HOURS,
    MEAN_EXIT_Z,
    PARTIAL_TP_MOVE_STOP_TO_BE,
    PARTIAL_TP_TRIGGER_R,
    POSITION_POLL_SECONDS,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    RETURN_SHOCK_VOL_WINDOW,
    RISK_REWARD_RATIO,
    SHOCK1H_THRESHOLD,
    SIGNAL_MAX_AGE_SECONDS,
    TIMEFRAME,
    TREND_WINDOW,
    FORWARD_EXPERIMENT_ID,
    FORWARD_JOURNAL_FILE,
    MAX_CONCURRENT_POSITIONS,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_POSITION_PCT,
    ORDERFLOW_COVERAGE_GRACE_SECONDS,
    ORDERFLOW_DECISION_SECONDS,
    RISK_BY_SYMBOL,
    SIDE_BY_SYMBOL,
    FORWARD_START_UTC,
    STATE_FILE,
    STRATEGY_VERSION,
    SYMBOLS,
    Z_THRESHOLD,
    frozen_config,
    validate_forward_settings,
)
from src.data.fetcher import fetch_binance_klines, timeframe_to_timedelta
from src.execution.engine import BinanceExecutionEngine, ExecutionError
from src.forward.config import canonical_config_sha256
from src.forward.gate import (
    ArmedSignal,
    create_armed_signal,
    journal_arm,
    journal_flow_decision,
    recover_armed_signals,
)
from src.forward.journal import ForwardJournal
from src.forward.orderflow import AggTradeCollector, FlowDecision
from src.risk.manager import RiskManager
from src.strategy.mean_reversion import MeanReversionStrategy


TP_EXIT_MODES = {"fixed_rr", "fixed_rr_time", "hybrid_rr_mean", "hybrid_rr_mean_time"}
MEAN_EXIT_MODES = {"mean", "hybrid_rr_mean", "hybrid_rr_mean_time"}
TIME_EXIT_MODES = {"time_stop", "fixed_rr_time", "hybrid_rr_mean_time"}
VALID_EXIT_MODES = TP_EXIT_MODES | MEAN_EXIT_MODES | TIME_EXIT_MODES


@dataclass
class LiveContext:
    symbol: str
    role: str
    allowed_side: str
    risk_per_trade: float
    engine: BinanceExecutionEngine | None
    strategy: MeanReversionStrategy
    risk: RiskManager | None
    data_limit: int
    regime_data_limit: int


def _slug(symbol: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", symbol).strip("_").lower()


def _state_file_for_symbol(symbol: str) -> str:
    if len(SYMBOLS) == 1:
        return STATE_FILE
    base = Path(STATE_FILE)
    suffix = base.suffix or ".json"
    return str(base.with_name(f"{base.stem}_{_slug(symbol)}{suffix}"))


def _signal_side(ctx: LiveContext, latest) -> str | None:
    column = "entry_long" if ctx.allowed_side == "long" else "entry_short"
    return ctx.allowed_side if bool(latest.get(column, False)) else None


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


def _forward_start() -> datetime:
    return datetime.fromisoformat(FORWARD_START_UTC.replace("Z", "+00:00"))


def _fetch_latest(ctx: LiveContext):
    df = fetch_binance_klines(
        symbol=ctx.symbol,
        timeframe=TIMEFRAME,
        limit=ctx.data_limit,
        closed_only=True,
    )
    if df.empty:
        raise RuntimeError(f"{ctx.symbol}: no closed candles available.")
    regime_df = None
    if REGIME_MODE != "1h_sma200":
        regime_df = fetch_binance_klines(
            symbol=ctx.symbol,
            timeframe=REGIME_TIMEFRAME,
            limit=ctx.regime_data_limit,
            closed_only=True,
        )
        if regime_df.empty:
            raise RuntimeError(
                f"{ctx.symbol}: no closed {REGIME_TIMEFRAME} candles available."
            )
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
        return max(
            0.0,
            float(position.contracts)
            * abs(float(position.entry_price) - float(initial_stop)),
        )
    return 0.0


def _active_count(states: dict[str, dict]) -> int:
    return sum(
        1
        for st in states.values()
        if st.get("known") and (st.get("entry_pending") or st.get("has_position"))
    )


def _aggregate_risk(states: dict[str, dict]) -> float:
    return sum(_state_planned_risk(st) for st in states.values())


def _require_engine(ctx: LiveContext) -> BinanceExecutionEngine:
    if ctx.role != "primary" or ctx.engine is None:
        raise RuntimeError(f"{ctx.symbol}: execution requested for non-primary context.")
    return ctx.engine


def _require_risk(ctx: LiveContext) -> RiskManager:
    if ctx.role != "primary" or ctx.risk is None:
        raise RuntimeError(f"{ctx.symbol}: risk plan requested for non-primary context.")
    return ctx.risk


def _float_or_none(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _journal_signal(
    journal: ForwardJournal,
    ctx: LiveContext,
    latest,
    *,
    config_sha256: str,
) -> None:
    signal_ts = latest.name.isoformat()
    candle_close = latest.name + timeframe_to_timedelta(TIMEFRAME)
    journal.append_once(
        "signal_evaluated",
        f"signal:{ctx.symbol}:{signal_ts}",
        experiment_id=FORWARD_EXPERIMENT_ID,
        strategy_version=STRATEGY_VERSION,
        config_sha256=config_sha256,
        build_sha=BUILD_SHA or None,
        symbol=ctx.symbol,
        role=ctx.role,
        allowed_side=ctx.allowed_side,
        risk_per_trade=ctx.risk_per_trade,
        candle_open_time=signal_ts,
        candle_close_time=candle_close.isoformat(),
        close=_float_or_none(latest.get("close")),
        z_score=_float_or_none(latest.get("z_score")),
        z_previous=_float_or_none(latest.get("z_previous")),
        z_confirmed=bool(latest.get("z_confirmed", False)),
        z_long_confirmed=bool(latest.get("z_long_confirmed", False)),
        z_short_confirmed=bool(latest.get("z_short_confirmed", False)),
        shock1h_return=_float_or_none(latest.get("shock1h_return")),
        shock1h_volatility=_float_or_none(latest.get("shock1h_volatility")),
        shock1h_score=_float_or_none(latest.get("shock1h_score")),
        shock1h_previous=_float_or_none(latest.get("shock1h_previous")),
        shock1h_confirmed=bool(latest.get("shock1h_confirmed", False)),
        shock1h_long_confirmed=bool(latest.get("shock1h_long_confirmed", False)),
        shock1h_short_confirmed=bool(latest.get("shock1h_short_confirmed", False)),
        local_regime_long=bool(latest.get("local_regime_long", False)),
        local_regime_short=bool(latest.get("local_regime_short", False)),
        htf_regime_long=bool(latest.get("htf_regime_long", False)),
        htf_regime_short=bool(latest.get("htf_regime_short", False)),
        regime_long=bool(latest.get("regime_long", False)),
        regime_short=bool(latest.get("regime_short", False)),
        atr=_float_or_none(latest.get("atr")),
        entry_long=bool(latest.get("entry_long", False)),
        entry_short=bool(latest.get("entry_short", False)),
    )


def _journal_decision(
    journal: ForwardJournal,
    ctx: LiveContext,
    signal_ts: str,
    decision: str,
    *,
    config_sha256: str,
    reason: str | None = None,
    **payload,
) -> None:
    journal.append_once(
        "signal_decision",
        f"decision:{ctx.symbol}:{signal_ts}:{decision}",
        experiment_id=FORWARD_EXPERIMENT_ID,
        strategy_version=STRATEGY_VERSION,
        config_sha256=config_sha256,
        symbol=ctx.symbol,
        role=ctx.role,
        allowed_side=ctx.allowed_side,
        risk_per_trade=ctx.risk_per_trade,
        signal_time=signal_ts,
        decision=decision,
        reason=reason,
        **payload,
    )


def _block_armed_signal(
    *,
    journal: ForwardJournal,
    ctx: LiveContext,
    arm: ArmedSignal,
    config_sha256: str,
    reason: str,
    flow_class: str,
    flow: FlowDecision | None = None,
    collector_status: dict[str, object] | None = None,
) -> None:
    engine = _require_engine(ctx)
    journal_flow_decision(
        journal,
        arm,
        config_sha256=config_sha256,
        flow=flow,
        flow_class=flow_class,
        veto=True,
        reason=reason,
        collector_status=collector_status,
    )
    _journal_decision(
        journal,
        ctx,
        arm.signal_time,
        "vetoed" if flow_class == "adverse_acceptance" else "blocked",
        config_sha256=config_sha256,
        reason=reason,
        orderflow_class=flow_class,
        orderflow_veto=True,
    )
    engine.mark_signal_processed(arm.signal_time)


def _process_ready_armed_signals(
    armed: dict[str, ArmedSignal],
    *,
    contexts: dict[str, LiveContext],
    states: dict[str, dict],
    journal: ForwardJournal,
    collector: AggTradeCollector,
    config_sha256: str,
    global_fail_closed: bool,
) -> None:
    now = datetime.now(timezone.utc)
    for key, arm in sorted(list(armed.items()), key=lambda item: item[1].decision_at):
        if now < arm.decision_at:
            continue

        ctx = contexts[arm.symbol]
        engine = _require_engine(ctx)
        risk = _require_risk(ctx)

        if now >= arm.deadline_at:
            _block_armed_signal(
                journal=journal,
                ctx=ctx,
                arm=arm,
                config_sha256=config_sha256,
                reason="orderflow_deadline_elapsed",
                flow_class="expired",
                collector_status=collector.status(),
            )
            armed.pop(key, None)
            continue

        if not collector.has_complete_window(
            arm.symbol,
            arm.decision_start,
            arm.decision_at,
        ):
            grace_deadline = arm.decision_at + timedelta(
                seconds=ORDERFLOW_COVERAGE_GRACE_SECONDS
            )
            if now < grace_deadline:
                continue
            _block_armed_signal(
                journal=journal,
                ctx=ctx,
                arm=arm,
                config_sha256=config_sha256,
                reason="incomplete_orderflow_window",
                flow_class="missing",
                collector_status=collector.status(),
            )
            print(
                f"{arm.symbol}: incomplete order-flow coverage; "
                "signal vetoed fail-closed."
            )
            armed.pop(key, None)
            continue

        try:
            flow = collector.decide(
                arm.symbol,
                side=arm.side,
                start=arm.decision_start,
                window_seconds=ORDERFLOW_DECISION_SECONDS,
            )
        except Exception as exc:
            _block_armed_signal(
                journal=journal,
                ctx=ctx,
                arm=arm,
                config_sha256=config_sha256,
                reason=f"orderflow_decision_error:{type(exc).__name__}",
                flow_class="missing",
                collector_status=collector.status(),
            )
            print(f"{arm.symbol}: order-flow decision failed closed: {exc}")
            armed.pop(key, None)
            continue

        journal_flow_decision(
            journal,
            arm,
            config_sha256=config_sha256,
            flow=flow,
            flow_class=flow.orderflow_class,
            veto=flow.veto,
            collector_status=collector.status(),
        )
        if flow.veto:
            _journal_decision(
                journal,
                ctx,
                arm.signal_time,
                "vetoed",
                config_sha256=config_sha256,
                reason="adverse_acceptance",
                orderflow_class=flow.orderflow_class,
                orderflow_veto=True,
                delay_reference_price=flow.reference_price,
                **flow.features,
            )
            engine.mark_signal_processed(arm.signal_time)
            print(
                f"{arm.symbol}: {arm.side.upper()} vetoed by adverse acceptance | "
                f"imb={flow.features.get('of_60s_side_imbalance')} | "
                f"ret={flow.features.get('of_60s_side_return_bps')}bp"
            )
            armed.pop(key, None)
            continue

        st = states.get(arm.symbol, {})
        if st.get("entry_pending") or st.get("has_position"):
            _journal_decision(
                journal,
                ctx,
                arm.signal_time,
                "blocked",
                config_sha256=config_sha256,
                reason="existing_position_or_pending_entry",
                orderflow_class=flow.orderflow_class,
            )
            engine.mark_signal_processed(arm.signal_time)
            armed.pop(key, None)
            continue
        if global_fail_closed:
            _journal_decision(
                journal,
                ctx,
                arm.signal_time,
                "blocked",
                config_sha256=config_sha256,
                reason="portfolio_fail_closed",
                orderflow_class=flow.orderflow_class,
            )
            engine.mark_signal_processed(arm.signal_time)
            armed.pop(key, None)
            continue
        if _active_count(states) >= MAX_CONCURRENT_POSITIONS:
            _journal_decision(
                journal,
                ctx,
                arm.signal_time,
                "blocked",
                config_sha256=config_sha256,
                reason="max_concurrent_positions",
                orderflow_class=flow.orderflow_class,
            )
            engine.mark_signal_processed(arm.signal_time)
            armed.pop(key, None)
            continue

        if not math.isfinite(arm.atr) or arm.atr <= 0:
            raise RuntimeError(f"{arm.symbol}: invalid armed ATR {arm.atr}")
        market_price = engine.get_market_price(arm.symbol)
        account = engine.get_account_snapshot("USDT")
        plan = risk.build_plan(
            side=arm.side,
            account_equity=account.equity,
            free_margin=account.free_margin,
            expected_entry_price=market_price,
            atr=arm.atr,
            leverage=LEVERAGE,
        )

        existing_risk = _aggregate_risk(states)
        risk_room = max(
            0.0,
            account.equity * MAX_PORTFOLIO_RISK_PCT - existing_risk,
        )
        if risk_room <= 0:
            _journal_decision(
                journal,
                ctx,
                arm.signal_time,
                "blocked",
                config_sha256=config_sha256,
                reason="portfolio_risk",
                portfolio_existing_risk=existing_risk,
                portfolio_risk_room=risk_room,
                orderflow_class=flow.orderflow_class,
            )
            engine.mark_signal_processed(arm.signal_time)
            armed.pop(key, None)
            continue

        requested_amount = plan.base_amount
        if plan.estimated_risk > risk_room:
            requested_amount = risk_room / plan.stop_distance
        amount = engine.normalize_amount(
            arm.symbol,
            base_amount=requested_amount,
            price=market_price,
        )
        actual_risk = amount * plan.stop_distance
        if actual_risk <= 0 or actual_risk > risk_room * 1.001:
            _journal_decision(
                journal,
                ctx,
                arm.signal_time,
                "blocked",
                config_sha256=config_sha256,
                reason="sizing_outside_portfolio_risk",
                estimated_risk=actual_risk,
                portfolio_risk_room=risk_room,
                orderflow_class=flow.orderflow_class,
            )
            engine.mark_signal_processed(arm.signal_time)
            armed.pop(key, None)
            continue

        remaining_wait = int(
            (arm.deadline_at - datetime.now(timezone.utc)).total_seconds()
        )
        if remaining_wait <= 0:
            _block_armed_signal(
                journal=journal,
                ctx=ctx,
                arm=arm,
                config_sha256=config_sha256,
                reason="orderflow_deadline_elapsed_before_submit",
                flow_class="expired",
                flow=flow,
                collector_status=collector.status(),
            )
            armed.pop(key, None)
            continue

        use_tp = EXIT_MODE in TP_EXIT_MODES
        context = {
            "strategy_version": STRATEGY_VERSION,
            "experiment_id": FORWARD_EXPERIMENT_ID,
            "config_sha256": config_sha256,
            "build_sha": BUILD_SHA or None,
            "signal_time": arm.signal_time,
            "signal_close": arm.signal_close,
            "signal_z": arm.z_score,
            "signal_shock1h": arm.shock1h_score,
            "atr": arm.atr,
            "regime_mode": REGIME_MODE,
            "expected_entry_price": market_price,
            "risk_reward_ratio": RISK_REWARD_RATIO,
            "risk_budget": plan.risk_budget,
            "estimated_risk": actual_risk,
            "risk_utilization": (
                actual_risk / plan.risk_budget if plan.risk_budget else 0.0
            ),
            "limiting_factor": (
                "portfolio_risk"
                if actual_risk + 1e-12 < plan.estimated_risk
                else plan.limiting_factor
            ),
            "portfolio_existing_risk": existing_risk,
            "portfolio_risk_room": risk_room,
            "exit_mode": EXIT_MODE,
            "orderflow_class": flow.orderflow_class,
            "orderflow_reference_price": flow.reference_price,
            "orderflow_decision_seconds": ORDERFLOW_DECISION_SECONDS,
            **flow.features,
        }
        _journal_decision(
            journal,
            ctx,
            arm.signal_time,
            "approved",
            config_sha256=config_sha256,
            side=arm.side,
            amount=amount,
            estimated_risk=actual_risk,
            portfolio_existing_risk=existing_risk,
            portfolio_risk_room=risk_room,
            orderflow_class=flow.orderflow_class,
            orderflow_veto=False,
            delay_reference_price=flow.reference_price,
            remaining_entry_seconds=remaining_wait,
        )

        result = engine.submit_passive_entry(
            symbol=arm.symbol,
            side=arm.side,
            amount=amount,
            reference_price=flow.reference_price,
            fallback_reference_price=flow.reference_price,
            limit_offset_bps=ENTRY_LIMIT_OFFSET_BPS,
            wait_seconds=remaining_wait,
            fallback_cap_bps=ENTRY_FALLBACK_CAP_BPS,
            fallback_max_lateness_seconds=ENTRY_FALLBACK_MAX_LATENESS_SECONDS,
            stop_distance=plan.stop_distance,
            risk_reward_ratio=RISK_REWARD_RATIO,
            use_take_profit=use_tp,
            signal_timestamp=arm.signal_time,
            journal_context=context,
        )
        journal.append_once(
            "order_submitted",
            f"order:{arm.symbol}:{arm.signal_time}",
            experiment_id=FORWARD_EXPERIMENT_ID,
            strategy_version=STRATEGY_VERSION,
            config_sha256=config_sha256,
            symbol=arm.symbol,
            signal_time=arm.signal_time,
            side=arm.side,
            amount=amount,
            orderflow_class=flow.orderflow_class,
            orderflow_reference_price=flow.reference_price,
            remaining_entry_seconds=remaining_wait,
        )
        states[arm.symbol] = {
            "known": True,
            "has_position": False,
            "entry_pending": True,
            "state": result["state"],
        }
        print(
            f"{arm.symbol}: {arm.side.upper()} order-flow approved | "
            f"class={flow.orderflow_class} ref={flow.reference_price:.8g} | "
            f"risk~{actual_risk:.2f} | remaining={remaining_wait}s"
        )
        armed.pop(key, None)


def _seconds_until_armed_action(armed: dict[str, ArmedSignal]) -> float | None:
    if not armed:
        return None
    now = datetime.now(timezone.utc)
    next_at = min(arm.decision_at for arm in armed.values())
    if next_at <= now:
        return 0.25
    return max(0.25, min(5.0, (next_at - now).total_seconds()))


def _maybe_arm_break_even(ctx: LiveContext, position_state: dict) -> None:
    if not position_state.get("has_position"):
        return
    engine = _require_engine(ctx)
    state = position_state["state"]
    position = position_state["position"]
    if state.break_even_armed:
        return

    if (
        PARTIAL_TP_TRIGGER_R > 0
        and PARTIAL_TP_MOVE_STOP_TO_BE
        and state.partial_take_profit_filled
    ):
        new_state = engine.move_stop_to_break_even(
            ctx.symbol, reason="partial take profit completed"
        )
        print(
            f"{ctx.symbol}: break-even after partial | "
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
    price = engine.get_market_price(ctx.symbol)
    reached = price >= trigger if position.side == "long" else price <= trigger
    if reached:
        new_state = engine.move_stop_to_break_even(
            ctx.symbol, reason=f"market reached {BREAK_EVEN_TRIGGER_R:.2f}R"
        )
        print(
            f"{ctx.symbol}: break-even armed | "
            f"entry={new_state.entry_price:.2f} SL={new_state.stop_price:.2f}"
        )


def _build_context(symbol: str, role: str = "primary") -> LiveContext:
    if role != "primary":
        raise ValueError("The active portfolio has executable primary components only.")
    if symbol not in SIDE_BY_SYMBOL:
        raise ValueError(f"{symbol}: not part of the frozen V12 portfolio.")

    allowed_side = SIDE_BY_SYMBOL[symbol]
    risk_per_trade = RISK_BY_SYMBOL[symbol]
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
        shock1h_threshold=SHOCK1H_THRESHOLD,
        return_shock_vol_window=RETURN_SHOCK_VOL_WINDOW,
    )

    data_limit = max(
        LOOKBACK_WINDOW,
        TREND_WINDOW,
        ATR_WINDOW,
        RETURN_SHOCK_VOL_WINDOW,
    ) + 100
    regime_data_limit = REGIME_EMA_WINDOW + REGIME_SLOPE_LOOKBACK + 100

    risk = RiskManager(
        risk_per_trade=risk_per_trade,
        max_position_pct=MAX_POSITION_PCT,
        atr_multiplier=ATR_MULTIPLIER,
        risk_reward_ratio=RISK_REWARD_RATIO,
        cash_buffer_pct=CASH_BUFFER_PCT,
    )
    engine = BinanceExecutionEngine(state_file=_state_file_for_symbol(symbol))
    engine.configure_account(symbol)
    return LiveContext(
        symbol=symbol,
        role=role,
        allowed_side=allowed_side,
        risk_per_trade=risk_per_trade,
        engine=engine,
        strategy=strategy,
        risk=risk,
        data_limit=data_limit,
        regime_data_limit=regime_data_limit,
    )


def main() -> None:
    validate_forward_settings()
    if EXIT_MODE not in VALID_EXIT_MODES:
        raise ValueError(f"Unsupported EXIT_MODE={EXIT_MODE!r}: {sorted(VALID_EXIT_MODES)}")

    frozen_contract = frozen_config()
    config_sha256 = canonical_config_sha256(frozen_contract)
    journal = ForwardJournal(FORWARD_JOURNAL_FILE)
    journal.append_once(
        "forward_run_started",
        f"run:{FORWARD_EXPERIMENT_ID}:{FORWARD_START_UTC}:{config_sha256}",
        experiment_id=FORWARD_EXPERIMENT_ID,
        strategy_version=STRATEGY_VERSION,
        config_sha256=config_sha256,
        build_sha=BUILD_SHA or None,
        forward_start_utc=FORWARD_START_UTC,
        components=[
            {
                "symbol": symbol,
                "side": SIDE_BY_SYMBOL[symbol],
                "risk_per_trade": RISK_BY_SYMBOL[symbol],
            }
            for symbol in SYMBOLS
        ],
        frozen_config=frozen_contract,
    )

    collector = AggTradeCollector(SYMBOLS)
    collector.start()
    if collector.wait_until_connected(timeout=15.0):
        print(f"Order-flow stream connected: {collector.url}")
    else:
        print(
            "Warning: order-flow stream is not connected yet. "
            "Signals will fail closed unless complete 60s coverage is proven."
        )
    armed = recover_armed_signals(journal, config_sha256=config_sha256)
    if armed:
        print(f"Recovered {len(armed)} unresolved order-flow armed signal(s).")

    sizing = ", ".join(
        f"{symbol} {SIDE_BY_SYMBOL[symbol][0].upper()}={RISK_BY_SYMBOL[symbol]:.2%}"
        for symbol in SYMBOLS
    )
    print(
        f"Mean-reversion {STRATEGY_VERSION} | Binance USD-M Demo | "
        f"{TIMEFRAME} | {LEVERAGE}x | RR={RISK_REWARD_RATIO:.2f} | "
        f"components=[{sizing}] | maxpos={MAX_POSITION_PCT:.0%} | "
        f"portfolio-risk={MAX_PORTFOLIO_RISK_PCT:.2%} | "
        f"config={config_sha256[:12]}"
    )

    contexts: dict[str, LiveContext] = {
        symbol: _build_context(symbol) for symbol in SYMBOLS
    }
    primary_contexts = contexts

    try:
        while True:
            try:
                states: dict[str, dict] = {}
                unknown_symbols: list[str] = []

                # Reconcile execution state for every frozen component.
                for symbol, ctx in primary_contexts.items():
                    engine = _require_engine(ctx)
                    st = engine.get_position_state(symbol)
                    if not st.get("known"):
                        unknown_symbols.append(symbol)
                        states[symbol] = st
                        print(f"{symbol}: UNKNOWN/fail-closed | {st.get('reason', 'no detail')}")
                        continue
                    if st.get("entry_pending"):
                        remaining = engine.pending_entry_remaining_seconds(st.get("state"))
                        if remaining is None:
                            unknown_symbols.append(symbol)
                            print(f"{symbol}: pending entry without valid expiration.")
                        elif remaining <= 0:
                            resolution = engine.resolve_pending_entry(symbol)
                            print(f"{symbol}: pending -> {resolution.get('action')}")
                            st = engine.get_position_state(symbol)
                    states[symbol] = st

                global_fail_closed = bool(unknown_symbols)
                in_window = _inside_candle_window()
                latest_by_symbol: dict[str, object] = {}

                # Evaluate the same closed candle for all frozen symbols.
                # Mainnet public candles are the strategy evidence;
                # Demo fills remain operational diagnostics only.
                if in_window:
                    for symbol in SYMBOLS:
                        ctx = contexts[symbol]
                        try:
                            latest = _fetch_latest(ctx)
                        except Exception as exc:
                            print(f"{symbol}: signal data error: {type(exc).__name__}: {exc}")
                            if ctx.role == "primary":
                                global_fail_closed = True
                            continue

                        candle_open = latest.name.to_pydatetime()
                        if candle_open < _forward_start():
                            continue

                        latest_by_symbol[symbol] = latest
                        _journal_signal(
                            journal,
                            ctx,
                            latest,
                            config_sha256=config_sha256,
                        )


                # Manage open primary positions.
                for symbol, ctx in primary_contexts.items():
                    st = states.get(symbol, {})
                    if not st.get("known") or not st.get("has_position"):
                        continue
                    engine = _require_engine(ctx)
                    _maybe_arm_break_even(ctx, st)
                    st = engine.get_position_state(symbol)
                    states[symbol] = st
                    if not st.get("known") or not st.get("has_position"):
                        continue

                    dynamic_exit = EXIT_MODE in (MEAN_EXIT_MODES | TIME_EXIT_MODES)
                    latest = latest_by_symbol.get(symbol)
                    if dynamic_exit and latest is not None:
                        signal_ts = latest.name.isoformat()
                        if not engine.was_signal_processed(signal_ts):
                            state = st["state"]
                            position = st["position"]
                            mean_exit = EXIT_MODE in MEAN_EXIT_MODES and _mean_exit_for_side(
                                latest, position.side
                            )
                            timeout_exit = EXIT_MODE in TIME_EXIT_MODES and _time_exit_reached(
                                state.entry_time, datetime.now(timezone.utc)
                            )
                            engine.mark_signal_processed(signal_ts)
                            if mean_exit or timeout_exit:
                                reason = (
                                    f"mean_reversion z={float(latest['z_score']):.3f}"
                                    if mean_exit
                                    else f"time_stop {MAX_HOLDING_HOURS}h"
                                )
                                engine.close_position_market(symbol, reason)
                                print(f"{symbol}: position closed by {reason}")
                                states[symbol] = engine.get_position_state(symbol)
                                continue
                    state = st["state"]
                    position = st["position"]
                    print(
                        f"{symbol}: {position.side.upper()} active | "
                        f"entry={position.entry_price:.2f} | "
                        f"SL={state.stop_price} TP={state.take_profit_price} | "
                        f"BE={state.break_even_armed}"
                    )

                # Arm every frozen V12 signal first. Portfolio occupancy/risk is
                # evaluated only after the causal 60s observation window, matching
                # the historical order-flow replay.
                if in_window:
                    for symbol, ctx in primary_contexts.items():
                        latest = latest_by_symbol.get(symbol)
                        if latest is None:
                            continue
                        engine = _require_engine(ctx)
                        signal_ts = latest.name.isoformat()
                        if engine.was_signal_processed(signal_ts):
                            continue

                        side = _signal_side(ctx, latest)
                        if side is None:
                            engine.mark_signal_processed(signal_ts)
                            continue

                        key = f"{symbol}|{signal_ts}"
                        if key in armed:
                            continue
                        decision_start = (
                            latest.name + timeframe_to_timedelta(TIMEFRAME)
                        ).to_pydatetime()
                        arm = create_armed_signal(
                            symbol=symbol,
                            side=side,
                            risk_per_trade=ctx.risk_per_trade,
                            signal_time=signal_ts,
                            decision_start=decision_start,
                            signal_close=float(latest["close"]),
                            atr=float(latest["atr"]),
                            z_score=_float_or_none(latest.get("z_score")),
                            shock1h_score=_float_or_none(latest.get("shock1h_score")),
                        )
                        # Persist ARMED before the process can ever place an order.
                        journal_arm(journal, arm, config_sha256=config_sha256)
                        armed[arm.key] = arm
                        print(
                            f"{symbol}: {side.upper()} ARMED | "
                            f"observe until {arm.decision_at.isoformat()} | "
                            f"deadline {arm.deadline_at.isoformat()}"
                        )

                # Resolve t+60s flow decisions even after the short candle-evaluation
                # window has closed. Missing/gapped WebSocket coverage fails closed.
                _process_ready_armed_signals(
                    armed,
                    contexts=primary_contexts,
                    states=states,
                    journal=journal,
                    collector=collector,
                    config_sha256=config_sha256,
                    global_fail_closed=global_fail_closed,
                )

                any_pending = any(
                    st.get("entry_pending") for st in states.values() if st.get("known")
                )
                any_position = any(
                    st.get("has_position") for st in states.values() if st.get("known")
                )
                if global_fail_closed:
                    sleep_for = ERROR_RETRY_SECONDS
                elif any_pending:
                    sleep_for = ENTRY_PENDING_POLL_SECONDS
                elif any_position:
                    sleep_for = POSITION_POLL_SECONDS
                    if EXIT_MODE in (MEAN_EXIT_MODES | TIME_EXIT_MODES):
                        sleep_for = max(
                            0.5,
                            min(sleep_for, _seconds_until_next_candle_check()),
                        )
                else:
                    sleep_for = max(0.5, _seconds_until_next_candle_check())
                    if not armed:
                        print(f"Portfolio flat; next evaluation in ~{sleep_for:.0f}s")
                armed_wait = _seconds_until_armed_action(armed)
                if armed_wait is not None:
                    sleep_for = min(sleep_for, armed_wait)
                time.sleep(sleep_for)

            except ExecutionError as exc:
                print(f"Execution blocked for safety: {exc}")
                time.sleep(ERROR_RETRY_SECONDS)
            except Exception as exc:
                print(f"Error: {type(exc).__name__}: {exc}")
                time.sleep(ERROR_RETRY_SECONDS)

    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    finally:
        collector.stop()


if __name__ == "__main__":
    main()
