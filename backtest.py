from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    BACKTEST_END,
    BACKTEST_EXECUTION_PROFILES,
    BACKTEST_MAKER_FEE,
    BACKTEST_MARKET_SLIPPAGE_BPS,
    BACKTEST_START,
    BACKTEST_TAKER_FEE,
    BACKTEST_TP_RATIOS,
    BACKTEST_V6_REGIME_MODE,
    BACKTEST_WALK_FORWARD,
    BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS,
    BACKTEST_YEARLY_BREAKDOWN,
    CASH_BUFFER_PCT,
    INITIAL_CAPITAL,
    LEVERAGE,
    LOOKBACK_WINDOW,
    MAX_HOLDING_BARS,
    MAX_POSITION_PCT,
    MEAN_EXIT_Z,
    RISK_PER_TRADE,
    RISK_REWARD_RATIO,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    SLIPPAGE,
    SYMBOL,
    TIMEFRAME,
    TRADING_FEE,
    TREND_WINDOW,
    Z_THRESHOLD,
    validate_settings,
)
from src.data.fetcher import fetch_binance_history
from src.risk.manager import RiskManager
from src.strategy.mean_reversion import MeanReversionStrategy
from src.utils.metrics import calculate_sharpe_ratio


@dataclass
class OpenPosition:
    side: str
    entry_time: pd.Timestamp
    entry_index: int
    signal_z: float
    raw_entry_price: float
    entry_price: float
    amount: float
    entry_fee: float
    entry_slippage_cost: float
    stop_price: float
    take_profit_price: Optional[float]
    stop_distance: float
    initial_risk: float
    max_favorable_distance: float = 0.0
    max_adverse_distance: float = 0.0


@dataclass
class ClosedTrade:
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    amount: float
    stop_price: float
    take_profit_price: Optional[float]
    exit_reason: str
    gross_before_costs: float
    gross_pnl: float
    fees: float
    slippage_cost: float
    pnl: float
    r_multiple: float
    mfe_r: float
    mae_r: float
    holding_bars: int
    holding_hours: float
    entry_z: float
    exit_z: float


@dataclass
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float
    trades: pd.DataFrame
    equity_curve: pd.Series
    open_position: Optional[OpenPosition]


@dataclass(frozen=True)
class ExecutionModel:
    profile: str
    taker_fee: float
    maker_fee: float
    market_slippage: float

    def __post_init__(self) -> None:
        if self.profile not in {"all_taker", "maker_tp"}:
            raise ValueError("profile debe ser all_taker o maker_tp.")
        if min(self.taker_fee, self.maker_fee, self.market_slippage) < 0:
            raise ValueError("Fees/slippage no pueden ser negativos.")


@dataclass(frozen=True)
class ResearchCase:
    side_mode: str
    regime_mode: str
    risk_reward_ratio: float
    allow_longs: bool
    allow_shorts: bool
    execution_profile: str
    market_slippage_bps: float


def _entry_fill(
    raw_open: float,
    side: str,
    slippage: float,
) -> tuple[float, float]:
    if side == "long":
        fill = raw_open * (1.0 + slippage)
    else:
        fill = raw_open * (1.0 - slippage)
    return fill, abs(fill - raw_open)


def _market_exit_fill(
    raw_open: float,
    side: str,
    slippage: float,
) -> tuple[float, float]:
    # LONG exits with SELL; SHORT exits with BUY.
    if side == "long":
        fill = raw_open * (1.0 - slippage)
    else:
        fill = raw_open * (1.0 + slippage)
    return fill, abs(fill - raw_open)


def _stop_fill(
    bar_open: float,
    stop_price: float,
    side: str,
    slippage: float,
) -> tuple[float, float, float]:
    if side == "long":
        raw_reference = min(bar_open, stop_price)
        fill = raw_reference * (1.0 - slippage)
    else:
        raw_reference = max(bar_open, stop_price)
        fill = raw_reference * (1.0 + slippage)
    return fill, raw_reference, abs(fill - raw_reference)


def _take_profit_fill(
    take_profit_price: float,
    side: str,
    slippage: float,
) -> tuple[float, float, float]:
    raw_reference = take_profit_price
    if side == "long":
        fill = raw_reference * (1.0 - slippage)
    else:
        fill = raw_reference * (1.0 + slippage)
    return fill, raw_reference, abs(fill - raw_reference)


def _pnl(side: str, entry_price: float, exit_price: float, amount: float) -> float:
    if side == "long":
        return (exit_price - entry_price) * amount
    return (entry_price - exit_price) * amount


def _unrealized_pnl(position: OpenPosition, mark_price: float) -> float:
    return _pnl(position.side, position.entry_price, mark_price, position.amount)


def _update_excursions(position: OpenPosition, high: float, low: float) -> None:
    if position.side == "long":
        favorable = max(0.0, high - position.entry_price)
        adverse = max(0.0, position.entry_price - low)
    else:
        favorable = max(0.0, position.entry_price - low)
        adverse = max(0.0, high - position.entry_price)

    position.max_favorable_distance = max(
        position.max_favorable_distance,
        favorable,
    )
    position.max_adverse_distance = max(
        position.max_adverse_distance,
        adverse,
    )


def _update_exit_point(position: OpenPosition, exit_price: float) -> None:
    if position.side == "long":
        favorable = max(0.0, exit_price - position.entry_price)
        adverse = max(0.0, position.entry_price - exit_price)
    else:
        favorable = max(0.0, position.entry_price - exit_price)
        adverse = max(0.0, exit_price - position.entry_price)

    position.max_favorable_distance = max(
        position.max_favorable_distance,
        favorable,
    )
    position.max_adverse_distance = max(
        position.max_adverse_distance,
        adverse,
    )


def _gap_through_stop(position: OpenPosition, bar_open: float) -> bool:
    if position.side == "long":
        return bar_open <= position.stop_price
    return bar_open >= position.stop_price


def _intrabar_exit(
    position: OpenPosition,
    bar: pd.Series,
    exit_mode: str,
    execution: ExecutionModel,
) -> Optional[tuple[str, float, float, float, float]]:
    if position.side == "long":
        stop_hit = float(bar["low"]) <= position.stop_price
        tp_hit = (
            exit_mode == "fixed_rr"
            and position.take_profit_price is not None
            and float(bar["high"]) >= position.take_profit_price
        )
    else:
        stop_hit = float(bar["high"]) >= position.stop_price
        tp_hit = (
            exit_mode == "fixed_rr"
            and position.take_profit_price is not None
            and float(bar["low"]) <= position.take_profit_price
        )

    # Si SL y TP aparecen tocados en la misma vela, asumimos SL primero.
    if stop_hit:
        fill, raw_reference, slippage_per_unit = _stop_fill(
            float(bar["open"]),
            position.stop_price,
            position.side,
            execution.market_slippage,
        )
        return (
            "stop_loss",
            fill,
            raw_reference,
            slippage_per_unit,
            execution.taker_fee,
        )

    if tp_hit and position.take_profit_price is not None:
        if execution.profile == "maker_tp":
            # TP LIMIT ya descansando en el libro: modelamos fill al precio
            # objetivo y fee maker. No se modela queue position/partial fill.
            tp_slippage = 0.0
            fee_rate = execution.maker_fee
        else:
            tp_slippage = execution.market_slippage
            fee_rate = execution.taker_fee
        fill, raw_reference, slippage_per_unit = _take_profit_fill(
            position.take_profit_price,
            position.side,
            tp_slippage,
        )
        return (
            "take_profit",
            fill,
            raw_reference,
            slippage_per_unit,
            fee_rate,
        )

    return None


def simulate(
    signals: pd.DataFrame,
    risk: RiskManager,
    exit_mode: str = "fixed_rr",
    max_holding_bars: int = MAX_HOLDING_BARS,
    initial_capital: float = INITIAL_CAPITAL,
    risk_reward_ratio: float = RISK_REWARD_RATIO,
    allow_longs: bool = True,
    allow_shorts: bool = True,
    execution_model: Optional[ExecutionModel] = None,
    entry_start: Optional[pd.Timestamp] = None,
    entry_end: Optional[pd.Timestamp] = None,
) -> BacktestResult:
    """Backtest Futures con ejecucion NEXT-OPEN.

    V3 usa principalmente ``confirmed + fixed_rr`` y permite variar de forma
    aislada el RR y el lado habilitado. ``entry_start``/``entry_end`` sirven
    para evaluaciones temporales anuales sin recalcular los indicadores ni
    permitir entradas fuera de la ventana de test.
    """
    if exit_mode not in {"fixed_rr", "z_reversion"}:
        raise ValueError("exit_mode debe ser 'fixed_rr' o 'z_reversion'.")
    if risk_reward_ratio <= 0:
        raise ValueError("risk_reward_ratio debe ser positivo.")
    if not allow_longs and not allow_shorts:
        raise ValueError("Debe habilitarse al menos LONG o SHORT.")
    if len(signals) < 2:
        raise ValueError("Se necesitan al menos dos velas para simular.")

    if execution_model is None:
        # Compatibilidad con llamadas/tests anteriores. El grid V6 siempre pasa
        # un modelo explicito.
        execution_model = ExecutionModel(
            profile="all_taker",
            taker_fee=TRADING_FEE,
            maker_fee=BACKTEST_MAKER_FEE,
            market_slippage=SLIPPAGE,
        )

    wallet = float(initial_capital)
    position: Optional[OpenPosition] = None
    closed_trades: list[ClosedTrade] = []
    equity_values: list[float] = []
    equity_index: list[pd.Timestamp] = []

    def close_position(
        *,
        exit_time: pd.Timestamp,
        exit_index: int,
        exit_price: float,
        raw_exit_price: float,
        exit_slippage_per_unit: float,
        exit_fee_rate: float,
        exit_reason: str,
        exit_z: float = float("nan"),
    ) -> None:
        nonlocal wallet, position
        assert position is not None

        _update_exit_point(position, exit_price)

        exit_notional = position.amount * exit_price
        exit_fee = exit_notional * exit_fee_rate
        gross_before_costs = _pnl(
            position.side,
            position.raw_entry_price,
            raw_exit_price,
            position.amount,
        )
        gross_pnl = _pnl(
            position.side,
            position.entry_price,
            exit_price,
            position.amount,
        )
        fees = position.entry_fee + exit_fee
        slippage_cost = (
            position.entry_slippage_cost
            + exit_slippage_per_unit * position.amount
        )
        net_pnl = gross_pnl - fees

        wallet += gross_pnl - exit_fee

        r_multiple = (
            net_pnl / position.initial_risk
            if position.initial_risk > 0
            else float("nan")
        )
        mfe_r = (
            position.max_favorable_distance / position.stop_distance
            if position.stop_distance > 0
            else float("nan")
        )
        mae_r = (
            position.max_adverse_distance / position.stop_distance
            if position.stop_distance > 0
            else float("nan")
        )
        holding_bars = max(0, exit_index - position.entry_index)
        holding_hours = max(
            0.0,
            (exit_time - position.entry_time).total_seconds() / 3600.0,
        )

        closed_trades.append(
            ClosedTrade(
                side=position.side,
                entry_time=position.entry_time,
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                amount=position.amount,
                stop_price=position.stop_price,
                take_profit_price=position.take_profit_price,
                exit_reason=exit_reason,
                gross_before_costs=gross_before_costs,
                gross_pnl=gross_pnl,
                fees=fees,
                slippage_cost=slippage_cost,
                pnl=net_pnl,
                r_multiple=r_multiple,
                mfe_r=mfe_r,
                mae_r=mae_r,
                holding_bars=holding_bars,
                holding_hours=holding_hours,
                entry_z=position.signal_z,
                exit_z=exit_z,
            )
        )
        position = None

    for i in range(len(signals)):
        timestamp = signals.index[i]
        bar = signals.iloc[i]
        previous = signals.iloc[i - 1] if i > 0 else None
        closed_this_bar = False

        # Cuando la ventana de test ya termino y no queda posicion abierta, no
        # hay nada mas que simular.
        if entry_end is not None and timestamp > entry_end and position is None:
            break

        # 1) Gestion de posicion abierta.
        if position is not None:
            if _gap_through_stop(position, float(bar["open"])):
                fill, raw_reference, slip_per_unit = _stop_fill(
                    float(bar["open"]),
                    position.stop_price,
                    position.side,
                    execution_model.market_slippage,
                )
                close_position(
                    exit_time=timestamp,
                    exit_index=i,
                    exit_price=fill,
                    raw_exit_price=raw_reference,
                    exit_slippage_per_unit=slip_per_unit,
                    exit_fee_rate=execution_model.taker_fee,
                    exit_reason="stop_loss",
                )
                closed_this_bar = True

            if position is not None and exit_mode == "z_reversion" and previous is not None:
                if position.side == "long":
                    mean_exit = bool(previous.get("exit_long_mean", False))
                else:
                    mean_exit = bool(previous.get("exit_short_mean", False))

                bars_held = i - position.entry_index
                timeout_exit = bars_held >= max_holding_bars
                if mean_exit or timeout_exit:
                    raw_reference = float(bar["open"])
                    fill, slip_per_unit = _market_exit_fill(
                        raw_reference,
                        position.side,
                        execution_model.market_slippage,
                    )
                    close_position(
                        exit_time=timestamp,
                        exit_index=i,
                        exit_price=fill,
                        raw_exit_price=raw_reference,
                        exit_slippage_per_unit=slip_per_unit,
                        exit_fee_rate=execution_model.taker_fee,
                        exit_reason="mean_reversion" if mean_exit else "timeout",
                        exit_z=float(previous.get("z_score", np.nan)),
                    )
                    closed_this_bar = True

            if position is not None:
                intrabar = _intrabar_exit(position, bar, exit_mode, execution_model)
                if intrabar is not None:
                    reason, fill, raw_reference, slip_per_unit, fee_rate = intrabar
                    close_position(
                        exit_time=timestamp,
                        exit_index=i,
                        exit_price=fill,
                        raw_exit_price=raw_reference,
                        exit_slippage_per_unit=slip_per_unit,
                        exit_fee_rate=fee_rate,
                        exit_reason=reason,
                    )
                    closed_this_bar = True
                else:
                    _update_excursions(
                        position,
                        high=float(bar["high"]),
                        low=float(bar["low"]),
                    )

        # 2) Entrada NEXT-OPEN.
        entry_in_window = (
            (entry_start is None or timestamp >= entry_start)
            and (entry_end is None or timestamp <= entry_end)
        )
        if i > 0 and position is None and not closed_this_bar and entry_in_window:
            assert previous is not None
            side: Optional[str] = None
            if allow_longs and bool(previous.get("entry_long", False)):
                side = "long"
            elif allow_shorts and bool(previous.get("entry_short", False)):
                side = "short"

            atr = float(previous.get("atr", np.nan))
            if side and math.isfinite(atr) and atr > 0 and wallet > 0:
                raw_entry = float(bar["open"])
                plan = risk.build_plan(
                    side=side,
                    account_equity=wallet,
                    free_margin=wallet,
                    expected_entry_price=raw_entry,
                    atr=atr,
                    leverage=LEVERAGE,
                )

                amount = plan.base_amount
                entry_price, entry_slippage_per_unit = _entry_fill(
                    raw_entry,
                    side,
                    execution_model.market_slippage,
                )
                entry_fee = amount * entry_price * execution_model.taker_fee
                stop_distance = atr * ATR_MULTIPLIER

                if side == "long":
                    stop_price = entry_price - stop_distance
                    take_profit_price = (
                        entry_price + stop_distance * risk_reward_ratio
                        if exit_mode == "fixed_rr"
                        else None
                    )
                else:
                    stop_price = entry_price + stop_distance
                    take_profit_price = (
                        entry_price - stop_distance * risk_reward_ratio
                        if exit_mode == "fixed_rr"
                        else None
                    )

                if amount > 0 and stop_price > 0 and (
                    take_profit_price is None or take_profit_price > 0
                ):
                    wallet -= entry_fee
                    position = OpenPosition(
                        side=side,
                        entry_time=timestamp,
                        entry_index=i,
                        signal_z=float(previous.get("z_score", np.nan)),
                        raw_entry_price=raw_entry,
                        entry_price=entry_price,
                        amount=amount,
                        entry_fee=entry_fee,
                        entry_slippage_cost=entry_slippage_per_unit * amount,
                        stop_price=stop_price,
                        take_profit_price=take_profit_price,
                        stop_distance=stop_distance,
                        initial_risk=amount * stop_distance,
                    )

                    intrabar = _intrabar_exit(position, bar, exit_mode, execution_model)
                    if intrabar is not None:
                        reason, fill, raw_reference, slip_per_unit, fee_rate = intrabar
                        close_position(
                            exit_time=timestamp,
                            exit_index=i,
                            exit_price=fill,
                            raw_exit_price=raw_reference,
                            exit_slippage_per_unit=slip_per_unit,
                            exit_fee_rate=fee_rate,
                            exit_reason=reason,
                        )
                    else:
                        _update_excursions(
                            position,
                            high=float(bar["high"]),
                            low=float(bar["low"]),
                        )

        equity = wallet
        if position is not None:
            equity += _unrealized_pnl(position, float(bar["close"]))
        equity_values.append(equity)
        equity_index.append(timestamp)

    if not equity_values:
        equity_values = [float(initial_capital)]
        equity_index = [signals.index[0]]

    equity_curve = pd.Series(equity_values, index=equity_index, name="equity")
    final_equity = float(equity_curve.iloc[-1])
    total_return = final_equity / initial_capital - 1.0

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    returns = equity_curve.pct_change().dropna()
    seconds_per_bar = pd.Timedelta(TIMEFRAME).total_seconds()
    periods_per_year = max(1, int((365.0 * 24.0 * 3600.0) / seconds_per_bar))
    sharpe = calculate_sharpe_ratio(returns, 0.0, periods_per_year)

    trade_df = pd.DataFrame([asdict(trade) for trade in closed_trades])
    return BacktestResult(
        initial_capital=initial_capital,
        final_equity=final_equity,
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe,
        trades=trade_df,
        equity_curve=equity_curve,
        open_position=position,
    )


def _profit_factor(trades: pd.DataFrame, column: str = "pnl") -> float:
    if trades.empty or column not in trades.columns:
        return float("nan")
    gross_profit = trades.loc[trades[column] > 0, column].sum()
    gross_loss = -trades.loc[trades[column] < 0, column].sum()
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else float("nan")


def _metrics_from_result(case: ResearchCase, result: BacktestResult) -> dict:
    trades = result.trades
    count = len(trades)
    return {
        "side_mode": case.side_mode,
        "regime_mode": case.regime_mode,
        "rr": case.risk_reward_ratio,
        "execution_profile": case.execution_profile,
        "slippage_bps": case.market_slippage_bps,
        "trades": count,
        "win_rate": float((trades["pnl"] > 0).mean()) if count else float("nan"),
        "gross_pf": _profit_factor(trades, "gross_before_costs"),
        "net_pf": _profit_factor(trades, "pnl"),
        "gross_expectancy": _safe_mean(trades["gross_before_costs"]) if count else float("nan"),
        "net_expectancy": _safe_mean(trades["pnl"]) if count else float("nan"),
        "gross_pnl": float(trades["gross_before_costs"].sum()) if count else 0.0,
        "net_pnl": float(trades["pnl"].sum()) if count else 0.0,
        "fees": float(trades["fees"].sum()) if count else 0.0,
        "slippage_cost": float(trades["slippage_cost"].sum()) if count else 0.0,
        "gross_profit": float(trades.loc[trades["gross_before_costs"] > 0, "gross_before_costs"].sum()) if count else 0.0,
        "gross_loss": float(-trades.loc[trades["gross_before_costs"] < 0, "gross_before_costs"].sum()) if count else 0.0,
        "net_profit": float(trades.loc[trades["pnl"] > 0, "pnl"].sum()) if count else 0.0,
        "net_loss": float(-trades.loc[trades["pnl"] < 0, "pnl"].sum()) if count else 0.0,
        "return": result.total_return,
        "max_dd": result.max_drawdown,
        "mfe_median": float(trades["mfe_r"].median()) if count else float("nan"),
        "mae_median": float(trades["mae_r"].median()) if count else float("nan"),
        "holding_median_h": float(trades["holding_hours"].median()) if count else float("nan"),
    }


def _format_pf(value: float) -> str:
    if pd.isna(value):
        return " n/a"
    if math.isinf(value):
        return " inf"
    return f"{value:5.2f}"


def _print_grid_summary(rows: list[dict]) -> None:
    print("\n\n" + "=" * 138)
    print("RESEARCH GRID V6 - ESTRATEGIA V5 FIJA + MODELO DE EJECUCION")
    print("=" * 138)
    print(
        "profile     slip  RR   trades   win%  grossPF  netPF  grossExp  netExp  "
        "fees    slipCost   return   maxDD"
    )
    for row in rows:
        print(
            f"{row['execution_profile']:<11} {row['slippage_bps']:>4.1f}bp "
            f"{row['rr']:>4.2f} {row['trades']:>7} {row['win_rate']:>6.1%} "
            f"{_format_pf(row['gross_pf']):>7} {_format_pf(row['net_pf']):>6} "
            f"{row['gross_expectancy']:>8.3f} {row['net_expectancy']:>7.3f} "
            f"{row['fees']:>7.1f} {row['slippage_cost']:>10.1f} "
            f"{row['return']:>8.2%} {row['max_dd']:>7.2%}"
        )


def _estimate_break_even_bps(group: pd.DataFrame) -> str:
    work = group.sort_values("slippage_bps")
    points = list(zip(work["slippage_bps"].astype(float), work["net_pf"].astype(float)))
    if not points:
        return "n/a"
    if points[0][1] < 1.0:
        return "<min"
    if points[-1][1] >= 1.0:
        return f">={points[-1][0]:.1f}"
    for (x1, y1), (x2, y2) in zip(points, points[1:]):
        if y1 >= 1.0 and y2 < 1.0:
            if y1 == y2:
                return f"{x1:.1f}-{x2:.1f}"
            x = x1 + (1.0 - y1) * (x2 - x1) / (y2 - y1)
            return f"~{x:.2f}"
    return "n/a"


def _print_break_even_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    print("\nSLIPPAGE DE BREAK-EVEN APROX. (NET PF = 1)")
    print("profile      RR   break-even bps   PF@0bp   PF@5bp")
    for (profile, rr), group in df.groupby(["execution_profile", "rr"], sort=False):
        zero = group.loc[group["slippage_bps"] == group["slippage_bps"].min(), "net_pf"]
        maxs = group.loc[group["slippage_bps"] == group["slippage_bps"].max(), "net_pf"]
        pf0 = float(zero.iloc[0]) if len(zero) else float("nan")
        pfm = float(maxs.iloc[0]) if len(maxs) else float("nan")
        print(
            f"{profile:<12} {rr:>4.2f} {_estimate_break_even_bps(group):>16} "
            f"{_format_pf(pf0):>7} {_format_pf(pfm):>7}"
        )


def _annual_records(case: ResearchCase, result: BacktestResult) -> list[dict]:
    trades = result.trades
    if trades.empty:
        return []
    work = trades.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True)
    work["year"] = work["entry_time"].dt.year
    records: list[dict] = []
    for year, group in work.groupby("year"):
        records.append(
            {
                "side_mode": case.side_mode,
                "regime_mode": case.regime_mode,
                "rr": case.risk_reward_ratio,
                "execution_profile": case.execution_profile,
                "slippage_bps": case.market_slippage_bps,
                "year": int(year),
                "trades": len(group),
                "gross_pf": _profit_factor(group, "gross_before_costs"),
                "net_pf": _profit_factor(group, "pnl"),
                "gross_expectancy": _safe_mean(group["gross_before_costs"]),
                "net_expectancy": _safe_mean(group["pnl"]),
                "gross_pnl": float(group["gross_before_costs"].sum()),
                "net_pnl": float(group["pnl"].sum()),
                "win_rate": float((group["pnl"] > 0).mean()),
            }
        )
    return records


def _print_yearly_cost_pivots(records: list[dict]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        return
    print("\nNET PROFIT FACTOR POR ANO Y SLIPPAGE")
    for profile in BACKTEST_EXECUTION_PROFILES:
        for rr in BACKTEST_TP_RATIOS:
            subset = df[
                (df["execution_profile"] == profile)
                & (df["rr"] == float(rr))
            ]
            if subset.empty:
                continue
            pivot = subset.pivot(
                index="year",
                columns="slippage_bps",
                values="net_pf",
            ).sort_index()
            print(f"\n[{profile} | RR={rr:.2f}]")
            print(pivot.to_string(float_format=lambda x: f"{x:6.3f}"))


def _side_flags(mode: str) -> tuple[bool, bool]:
    if mode == "both":
        return True, True
    if mode == "long":
        return True, False
    if mode == "short":
        return False, True
    raise ValueError(f"Modo de lado no valido: {mode}")


def _build_signals(
    df: pd.DataFrame,
    regime_mode: str,
    regime_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW,
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=ATR_WINDOW,
        entry_mode="confirmed",
        mean_exit_z=MEAN_EXIT_Z,
        regime_mode=regime_mode,
        signal_timeframe=TIMEFRAME,
        regime_timeframe=REGIME_TIMEFRAME,
        regime_ema_window=REGIME_EMA_WINDOW,
        regime_slope_lookback=REGIME_SLOPE_LOOKBACK,
    )
    return strategy.calculate_signals(df, regime_df=regime_df)


def _risk_manager(rr: float) -> RiskManager:
    return RiskManager(
        risk_per_trade=RISK_PER_TRADE,
        max_position_pct=MAX_POSITION_PCT,
        atr_multiplier=ATR_MULTIPLIER,
        risk_reward_ratio=rr,
        cash_buffer_pct=CASH_BUFFER_PCT,
    )


def _execution_model(case: ResearchCase) -> ExecutionModel:
    return ExecutionModel(
        profile=case.execution_profile,
        taker_fee=BACKTEST_TAKER_FEE,
        maker_fee=BACKTEST_MAKER_FEE,
        market_slippage=case.market_slippage_bps / 10_000.0,
    )


def _test_year_window(signals: pd.DataFrame, year: int) -> pd.DataFrame:
    tz = signals.index.tz
    start = pd.Timestamp(year=year, month=1, day=1, tz=tz)
    # Una vela previa basta porque los indicadores ya estan calculados sobre el
    # dataset completo. Dejamos disponible todo el futuro: simulate() deja de
    # recorrer en cuanto acaba el ano y ya no queda posicion abierta. Asi una
    # posicion del 31 de diciembre puede cerrarse correctamente sin un corte
    # arbitrario de dias.
    bar_delta = pd.Timedelta(TIMEFRAME)
    return signals.loc[signals.index >= start - bar_delta]


def _walk_forward_records(
    signals_by_regime: dict[str, pd.DataFrame],
    cases: list[ResearchCase],
) -> list[dict]:
    if not signals_by_regime:
        return []
    reference_signals = next(iter(signals_by_regime.values()))
    if reference_signals.empty:
        return []
    years = sorted(set(int(year) for year in reference_signals.index.year))
    if len(years) <= BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS:
        return []

    test_years = years[BACKTEST_WALK_FORWARD_MIN_TRAIN_YEARS:]
    records: list[dict] = []
    tz = reference_signals.index.tz

    for case in cases:
        risk = _risk_manager(case.risk_reward_ratio)
        case_signals = signals_by_regime[case.regime_mode]
        for year in test_years:
            year_signals = _test_year_window(case_signals, year)
            if len(year_signals) < 2:
                continue
            entry_start = pd.Timestamp(year=year, month=1, day=1, tz=tz)
            entry_end = pd.Timestamp(
                year=year,
                month=12,
                day=31,
                hour=23,
                minute=59,
                second=59,
                tz=tz,
            )
            result = simulate(
                year_signals,
                risk,
                exit_mode="fixed_rr",
                risk_reward_ratio=case.risk_reward_ratio,
                allow_longs=case.allow_longs,
                allow_shorts=case.allow_shorts,
                execution_model=_execution_model(case),
                entry_start=entry_start,
                entry_end=entry_end,
            )
            row = _metrics_from_result(case, result)
            row["test_year"] = year
            row["train_through"] = year - 1
            records.append(row)
    return records


def _print_walk_forward(records: list[dict]) -> None:
    df = pd.DataFrame(records)
    if df.empty:
        print("\nSin anos suficientes para validacion temporal.")
        return

    print("\n\n" + "=" * 138)
    print("VALIDACION TEMPORAL ANUAL V6 - PARAMETROS DE ESTRATEGIA FIJOS")
    print("=" * 138)
    print(
        "profile     slip  RR   years  +years  aggNetPF  medianPF  "
        "aggNetPnL  avgNetExp  worstRet"
    )

    group_cols = ["execution_profile", "slippage_bps", "rr"]
    for (profile, slip_bps, rr), group in df.groupby(group_cols, sort=False):
        trades = int(group["trades"].sum())
        positive_years = int((group["net_pnl"] > 0).sum())
        finite_pf = group["net_pf"].replace([np.inf, -np.inf], np.nan).dropna()
        median_pf = float(finite_pf.median()) if len(finite_pf) else float("nan")
        agg_net_pnl = float(group["net_pnl"].sum())
        avg_net_exp = agg_net_pnl / trades if trades > 0 else float("nan")
        agg_profit = float(group["net_profit"].sum())
        agg_loss = float(group["net_loss"].sum())
        if agg_loss > 0:
            agg_pf = agg_profit / agg_loss
        elif agg_profit > 0:
            agg_pf = float("inf")
        else:
            agg_pf = float("nan")

        print(
            f"{profile:<11} {slip_bps:>4.1f}bp {rr:>4.2f} {len(group):>6} "
            f"{positive_years:>7} {_format_pf(agg_pf):>8} "
            f"{_format_pf(median_pf):>8} {agg_net_pnl:>10.2f} "
            f"{avg_net_exp:>10.3f} {group['return'].min():>8.2%}"
        )

    print("\nNET PF POR ANO DE TEST (columnas = slippage bps)")
    for profile in BACKTEST_EXECUTION_PROFILES:
        for rr in BACKTEST_TP_RATIOS:
            subset = df[
                (df["execution_profile"] == profile)
                & (df["rr"] == float(rr))
            ]
            if subset.empty:
                continue
            pivot = subset.pivot(
                index="test_year",
                columns="slippage_bps",
                values="net_pf",
            ).sort_index()
            print(f"\n[{profile} | RR={rr:.2f}]")
            print(pivot.to_string(float_format=lambda x: f"{x:6.3f}"))


def run_backtest() -> None:
    validate_settings()
    print(
        f"Descargando Binance USD-M Futures {SYMBOL} {TIMEFRAME}: "
        f"{BACKTEST_START} -> {BACKTEST_END}..."
    )
    print(
        "V6: estrategia congelada = LONG confirmed + 1h SMA200 AND 4h EMA200. "
        "Se varia solo RR, perfil de ejecucion y slippage de mercado."
    )
    print(
        f"Fees investigacion: taker={BACKTEST_TAKER_FEE:.6f} | "
        f"maker={BACKTEST_MAKER_FEE:.6f}"
    )
    print(f"RR grid: {BACKTEST_TP_RATIOS}")
    print(f"Execution profiles: {BACKTEST_EXECUTION_PROFILES}")
    print(f"Market slippage grid (bps): {BACKTEST_MARKET_SLIPPAGE_BPS}")
    print(f"Regime fijo: {BACKTEST_V6_REGIME_MODE}")
    print(
        "Nota maker_tp: el TP se modela como LIMIT maker sin slippage al tocar el "
        "precio. OHLC no permite modelar cola, partial fills ni missed fills."
    )

    df = fetch_binance_history(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start=BACKTEST_START,
        end=BACKTEST_END,
    )

    min_rows = max(TREND_WINDOW, LOOKBACK_WINDOW, ATR_WINDOW) + 50
    if len(df) < min_rows:
        raise RuntimeError(
            f"Solo hay {len(df)} velas; se necesitan al menos {min_rows}."
        )

    regime_df = None
    if BACKTEST_V6_REGIME_MODE != "1h_sma200":
        print(
            f"Descargando regimen Binance USD-M Futures {SYMBOL} "
            f"{REGIME_TIMEFRAME}: {BACKTEST_START} -> {BACKTEST_END}..."
        )
        regime_df = fetch_binance_history(
            symbol=SYMBOL,
            timeframe=REGIME_TIMEFRAME,
            start=BACKTEST_START,
            end=BACKTEST_END,
        )
        min_regime_rows = REGIME_EMA_WINDOW + REGIME_SLOPE_LOOKBACK + 20
        if len(regime_df) < min_regime_rows:
            raise RuntimeError(
                f"Solo hay {len(regime_df)} velas {REGIME_TIMEFRAME}; "
                f"se necesitan al menos {min_regime_rows}."
            )

    signals = _build_signals(
        df,
        regime_mode=BACKTEST_V6_REGIME_MODE,
        regime_df=regime_df,
    )
    signals_by_regime = {BACKTEST_V6_REGIME_MODE: signals}

    cases: list[ResearchCase] = []
    summary_rows: list[dict] = []
    annual_rows: list[dict] = []

    for profile in BACKTEST_EXECUTION_PROFILES:
        for slippage_bps in BACKTEST_MARKET_SLIPPAGE_BPS:
            for rr in BACKTEST_TP_RATIOS:
                case = ResearchCase(
                    side_mode="long",
                    regime_mode=BACKTEST_V6_REGIME_MODE,
                    risk_reward_ratio=float(rr),
                    allow_longs=True,
                    allow_shorts=False,
                    execution_profile=profile,
                    market_slippage_bps=float(slippage_bps),
                )
                cases.append(case)
                result = simulate(
                    signals,
                    _risk_manager(case.risk_reward_ratio),
                    exit_mode="fixed_rr",
                    risk_reward_ratio=case.risk_reward_ratio,
                    allow_longs=True,
                    allow_shorts=False,
                    execution_model=_execution_model(case),
                )
                summary_rows.append(_metrics_from_result(case, result))
                if BACKTEST_YEARLY_BREAKDOWN:
                    annual_rows.extend(_annual_records(case, result))

    _print_grid_summary(summary_rows)
    _print_break_even_summary(summary_rows)

    if BACKTEST_YEARLY_BREAKDOWN:
        _print_yearly_cost_pivots(annual_rows)

    if BACKTEST_WALK_FORWARD:
        wf_rows = _walk_forward_records(signals_by_regime, cases)
        _print_walk_forward(wf_rows)

    print("\n\nReferencia live actual: NO se modifica automaticamente por V6.")
    print(
        f"REGIME_MODE={REGIME_MODE if 'REGIME_MODE' in globals() else 'config'} | "
        f"RISK_REWARD_RATIO={RISK_REWARD_RATIO:.2f}"
    )


if __name__ == "__main__":
    run_backtest()
