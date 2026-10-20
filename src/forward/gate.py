from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from config.settings import (
    FORWARD_EXPERIMENT_ID,
    ORDERFLOW_DECISION_SECONDS,
    ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS,
    STRATEGY_VERSION,
)
from src.forward.journal import ForwardJournal
from src.forward.orderflow import FlowDecision


@dataclass(frozen=True)
class ArmedSignal:
    symbol: str
    side: str
    risk_per_trade: float
    signal_time: str
    decision_start: datetime
    decision_at: datetime
    deadline_at: datetime
    signal_close: float
    atr: float
    z_score: float | None
    shock1h_score: float | None

    @property
    def key(self) -> str:
        return f"{self.symbol}|{self.signal_time}"


def utc(value: datetime | str) -> datetime:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def create_armed_signal(
    *,
    symbol: str,
    side: str,
    risk_per_trade: float,
    signal_time: str,
    decision_start: datetime,
    signal_close: float,
    atr: float,
    z_score: float | None,
    shock1h_score: float | None,
) -> ArmedSignal:
    start = utc(decision_start)
    return ArmedSignal(
        symbol=symbol,
        side=side,
        risk_per_trade=risk_per_trade,
        signal_time=signal_time,
        decision_start=start,
        decision_at=start + timedelta(seconds=ORDERFLOW_DECISION_SECONDS),
        deadline_at=start + timedelta(seconds=ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS),
        signal_close=float(signal_close),
        atr=float(atr),
        z_score=z_score,
        shock1h_score=shock1h_score,
    )


def journal_arm(
    journal: ForwardJournal,
    arm: ArmedSignal,
    *,
    config_sha256: str,
) -> None:
    journal.append_once(
        "orderflow_armed",
        f"arm:{arm.key}",
        experiment_id=FORWARD_EXPERIMENT_ID,
        strategy_version=STRATEGY_VERSION,
        config_sha256=config_sha256,
        symbol=arm.symbol,
        allowed_side=arm.side,
        risk_per_trade=arm.risk_per_trade,
        signal_time=arm.signal_time,
        decision_start=arm.decision_start.isoformat(),
        decision_at=arm.decision_at.isoformat(),
        deadline_at=arm.deadline_at.isoformat(),
        signal_close=arm.signal_close,
        atr=arm.atr,
        z_score=arm.z_score,
        shock1h_score=arm.shock1h_score,
        baseline_shadow_reference_price=arm.signal_close,
        baseline_shadow_deadline_at=arm.deadline_at.isoformat(),
    )


def recover_armed_signals(
    journal: ForwardJournal,
    *,
    config_sha256: str,
) -> dict[str, ArmedSignal]:
    arms: dict[str, ArmedSignal] = {}
    resolved: set[str] = set()
    for record in journal.records():
        if record.get("experiment_id") != FORWARD_EXPERIMENT_ID:
            continue
        if record.get("config_sha256") != config_sha256:
            continue
        symbol = str(record.get("symbol") or "")
        signal_time = str(record.get("signal_time") or "")
        if not symbol or not signal_time:
            continue
        key = f"{symbol}|{signal_time}"
        event = record.get("event")
        if event == "orderflow_decision":
            resolved.add(key)
            arms.pop(key, None)
            continue
        if event != "orderflow_armed" or key in resolved:
            continue
        try:
            arms[key] = ArmedSignal(
                symbol=symbol,
                side=str(record["allowed_side"]),
                risk_per_trade=float(record["risk_per_trade"]),
                signal_time=signal_time,
                decision_start=utc(str(record["decision_start"])),
                decision_at=utc(str(record["decision_at"])),
                deadline_at=utc(str(record["deadline_at"])),
                signal_close=float(record["signal_close"]),
                atr=float(record["atr"]),
                z_score=_float_or_none(record.get("z_score")),
                shock1h_score=_float_or_none(record.get("shock1h_score")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return arms


def journal_flow_decision(
    journal: ForwardJournal,
    arm: ArmedSignal,
    *,
    config_sha256: str,
    flow: FlowDecision | None,
    flow_class: str,
    veto: bool,
    reason: str | None = None,
    collector_status: dict[str, object] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "experiment_id": FORWARD_EXPERIMENT_ID,
        "strategy_version": STRATEGY_VERSION,
        "config_sha256": config_sha256,
        "symbol": arm.symbol,
        "allowed_side": arm.side,
        "risk_per_trade": arm.risk_per_trade,
        "signal_time": arm.signal_time,
        "decision_start": arm.decision_start.isoformat(),
        "decision_at": arm.decision_at.isoformat(),
        "deadline_at": arm.deadline_at.isoformat(),
        "orderflow_class": flow_class,
        "orderflow_veto": bool(veto),
        "reason": reason,
        "baseline_shadow_reference_price": arm.signal_close,
        "collector_status": collector_status or {},
    }
    if flow is not None:
        payload.update(
            {
                "delay_shadow_reference_price": flow.reference_price,
                "gate_reference_price": flow.reference_price,
                **flow.features,
            }
        )
    journal.append_once(
        "orderflow_decision",
        f"flow:{arm.key}",
        **payload,
    )


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number
