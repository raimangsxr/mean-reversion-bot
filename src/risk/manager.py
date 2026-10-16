from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

TradeSide = Literal["long", "short"]


@dataclass(frozen=True)
class TradeRiskPlan:
    side: TradeSide
    expected_entry_price: float
    stop_distance: float
    stop_price: float
    take_profit_price: float
    stop_fraction: float
    position_fraction: float
    notional: float
    base_amount: float
    required_margin: float
    risk_budget: float
    estimated_risk: float


class RiskManager:
    """Sizing común para backtest y live de Futures.

    1) El presupuesto de pérdida se calcula sobre el equity.
    2) El notional necesario se deriva de la distancia del stop.
    3) Se limita por MAX_POSITION_PCT y por el margen libre disponible.
    4) El leverage solo afecta al margen requerido, no al riesgo objetivo.
    """

    def __init__(
        self,
        risk_per_trade: float,
        max_position_pct: float,
        atr_multiplier: float,
        risk_reward_ratio: float,
        cash_buffer_pct: float = 0.0,
    ) -> None:
        if not 0 < risk_per_trade <= 1:
            raise ValueError("risk_per_trade debe estar entre 0 y 1.")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct debe estar entre 0 y 1.")
        if atr_multiplier <= 0:
            raise ValueError("atr_multiplier debe ser positivo.")
        if risk_reward_ratio <= 0:
            raise ValueError("risk_reward_ratio debe ser positivo.")
        if not 0 <= cash_buffer_pct < 1:
            raise ValueError("cash_buffer_pct debe estar entre 0 y 1.")

        self.risk_per_trade = risk_per_trade
        self.max_position_pct = max_position_pct
        self.atr_multiplier = atr_multiplier
        self.risk_reward_ratio = risk_reward_ratio
        self.cash_buffer_pct = cash_buffer_pct

    def build_plan(
        self,
        side: TradeSide,
        account_equity: float,
        free_margin: float,
        expected_entry_price: float,
        atr: float,
        leverage: int,
    ) -> TradeRiskPlan:
        if side not in {"long", "short"}:
            raise ValueError("side debe ser 'long' o 'short'.")
        if account_equity <= 0:
            raise ValueError("El equity de la cuenta debe ser positivo.")
        if free_margin <= 0:
            raise ValueError("No hay margen libre para abrir una posición.")
        if expected_entry_price <= 0:
            raise ValueError("El precio de entrada debe ser positivo.")
        if not math.isfinite(atr) or atr <= 0:
            raise ValueError("El ATR debe ser positivo y finito.")
        if leverage < 1:
            raise ValueError("El leverage debe ser >= 1.")

        stop_distance = atr * self.atr_multiplier
        stop_fraction = stop_distance / expected_entry_price
        if not math.isfinite(stop_fraction) or stop_fraction <= 0:
            raise ValueError("La distancia relativa del stop no es válida.")

        risk_budget = account_equity * self.risk_per_trade
        risk_sized_notional = risk_budget / stop_fraction
        max_position_notional = account_equity * self.max_position_pct

        usable_margin = free_margin * (1.0 - self.cash_buffer_pct)
        max_notional_by_margin = usable_margin * leverage

        notional = min(
            risk_sized_notional,
            max_position_notional,
            max_notional_by_margin,
        )
        if notional <= 0:
            raise ValueError("El sizing produce un notional no válido.")

        base_amount = notional / expected_entry_price
        position_fraction = notional / account_equity
        required_margin = notional / leverage
        estimated_risk = notional * stop_fraction

        if side == "long":
            stop_price = expected_entry_price - stop_distance
            take_profit_price = (
                expected_entry_price + stop_distance * self.risk_reward_ratio
            )
        else:
            stop_price = expected_entry_price + stop_distance
            take_profit_price = (
                expected_entry_price - stop_distance * self.risk_reward_ratio
            )

        if stop_price <= 0 or take_profit_price <= 0:
            raise ValueError("SL/TP calculados no válidos.")

        return TradeRiskPlan(
            side=side,
            expected_entry_price=expected_entry_price,
            stop_distance=stop_distance,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            stop_fraction=stop_fraction,
            position_fraction=position_fraction,
            notional=notional,
            base_amount=base_amount,
            required_margin=required_margin,
            risk_budget=risk_budget,
            estimated_risk=estimated_risk,
        )
