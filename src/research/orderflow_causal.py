from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd

from config.settings import (
    BACKTEST_MAKER_FEE,
    BACKTEST_TAKER_FEE,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    TIMEFRAME,
)
from src.data.fetcher import timeframe_to_timedelta
from src.research.engine import EntryCandidate, EntryFill, minute_period
from src.research.orderflow import (
    PRIMARY_DECISION_WINDOW_SECONDS,
    classify_primary_window,
    compute_flow_features,
    empty_aggtrades,
)


CAUSAL_DECISION_DELAY_SECONDS = PRIMARY_DECISION_WINDOW_SECONDS
CAUSAL_TOTAL_ENTRY_DEADLINE_SECONDS = ENTRY_LIMIT_WAIT_SECONDS


@dataclass(frozen=True)
class CausalCandidateBuild:
    delay_only_candidates: list[EntryCandidate]
    orderflow_candidates: list[EntryCandidate]
    decisions: pd.DataFrame


def _utc(value: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def iter_signal_decisions(
    signals: pd.DataFrame,
    *,
    side: str,
    signal_timeframe: str = TIMEFRAME,
) -> list[dict]:
    """Return every frozen signal that reaches the execution decision hour.

    This deliberately starts from signals rather than the original fills. A causal
    delayed route can change whether an opportunity fills and can therefore change
    portfolio concurrency. Restricting the universe to previously-filled trades would
    leak the baseline execution path into the counterfactual.
    """
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    flag = "entry_long" if side == "long" else "entry_short"
    if flag not in signals.columns:
        return []

    delta = timeframe_to_timedelta(signal_timeframe)
    flags = signals[flag].fillna(False).astype(bool).to_numpy(copy=False)
    closes = pd.to_numeric(signals["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    atrs = pd.to_numeric(signals["atr"], errors="coerce").to_numpy(dtype=float, copy=False)
    index = signals.index
    rows: list[dict] = []
    for i in range(1, len(signals)):
        if not flags[i - 1]:
            continue
        signal_time = _utc(index[i - 1])
        entry_hour = _utc(index[i])
        if entry_hour != signal_time + delta:
            continue
        signal_close = float(closes[i - 1])
        signal_atr = float(atrs[i - 1])
        if not math.isfinite(signal_close) or signal_close <= 0:
            continue
        if not math.isfinite(signal_atr) or signal_atr <= 0:
            continue
        rows.append(
            {
                "signal_time": signal_time,
                "entry_hour": entry_hour,
                "side": side,
                "signal_close": signal_close,
                "signal_atr": signal_atr,
            }
        )
    return rows


def decision_reference_price(
    aggtrades: pd.DataFrame,
    *,
    decision_start: pd.Timestamp,
    minute_cache: pd.DataFrame,
    window_seconds: int = CAUSAL_DECISION_DELAY_SECONDS,
) -> tuple[float | None, str]:
    """Price observable at the end of the order-flow decision window.

    Prefer the last executed aggTrade observed through t+window. If a public archive
    contains no trade in that interval, fall back to the 1m open at the exact decision
    boundary. The fallback is used for the delay-only control too; the order-flow class
    remains ``missing`` and is never silently promoted to a flow signal.
    """
    start = _utc(decision_start)
    end = start + pd.Timedelta(seconds=int(window_seconds))
    data = aggtrades if aggtrades is not None else empty_aggtrades()
    if not data.empty:
        window = data.loc[(data["timestamp"] >= start) & (data["timestamp"] <= end)]
        if not window.empty:
            price = float(pd.to_numeric(window["price"], errors="coerce").dropna().iloc[-1])
            if math.isfinite(price) and price > 0:
                return price, "aggtrade_last"

    if end in minute_cache.index:
        price = float(minute_cache.loc[end, "open"])
        if math.isfinite(price) and price > 0:
            return price, "minute_open_fallback"
    return None, "missing"


def resolve_delayed_entry(
    minute_cache: pd.DataFrame,
    *,
    entry_hour: pd.Timestamp,
    side: str,
    reference_price: float,
    market_slippage_bps: float,
    decision_delay_seconds: int = CAUSAL_DECISION_DELAY_SECONDS,
    total_deadline_seconds: int = CAUSAL_TOTAL_ENTRY_DEADLINE_SECONDS,
    limit_offset_bps: float = ENTRY_LIMIT_OFFSET_BPS,
    max_chase_bps: float = ENTRY_FALLBACK_CAP_BPS,
) -> EntryFill | None:
    """Resolve the causal t+60s pullback LIMIT inside the original 7m deadline.

    The order is considered only after the observation window. It then has the
    remaining time until the original t+7m deadline to fill as maker. If it is still
    unfilled at that deadline, the frozen no-chase conditional market fallback is
    evaluated against the t+60s reference price.

    The current experiment is deliberately restricted to minute-aligned timings so the
    1m replay never consumes OHLC information from before the causal decision point.
    """
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if decision_delay_seconds <= 0 or total_deadline_seconds <= decision_delay_seconds:
        raise ValueError("decision delay must be positive and before the total deadline")
    if decision_delay_seconds % 60 != 0 or total_deadline_seconds % 60 != 0:
        raise ValueError("causal replay requires minute-aligned decision/deadline")
    if reference_price <= 0 or not math.isfinite(reference_price):
        return None

    entry_hour = _utc(entry_hour)
    decision_ts = entry_hour + pd.Timedelta(seconds=decision_delay_seconds)
    deadline_ts = entry_hour + pd.Timedelta(seconds=total_deadline_seconds)
    # Include the deadline candle because its open is the exact fallback observation.
    replay = minute_period(
        minute_cache,
        decision_ts,
        deadline_ts - decision_ts + pd.Timedelta(minutes=1),
    )
    if replay.empty or deadline_ts not in replay.index:
        return None

    if side == "long":
        limit_price = reference_price * (1.0 - limit_offset_bps / 10_000.0)
    else:
        limit_price = reference_price * (1.0 + limit_offset_bps / 10_000.0)

    # Maker search only in the remaining interval [t+60s, t+7m). Strict crossing
    # matches the existing frozen research engine's conservative maker assumption.
    search = replay.loc[replay.index < deadline_ts]
    for ts, row in search.iterrows():
        traversed = (
            float(row["low"]) < limit_price
            if side == "long"
            else float(row["high"]) > limit_price
        )
        if traversed:
            return EntryFill(
                timestamp=_utc(ts),
                side=side,
                raw_price=limit_price,
                fill_price=limit_price,
                fee_rate=BACKTEST_MAKER_FEE,
                slippage_per_unit=0.0,
                liquidity="maker",
                route="delay60_limit_maker",
                baseline_open=reference_price,
            )

    raw = float(replay.loc[deadline_ts, "open"])
    if side == "long":
        allowed = raw <= reference_price * (1.0 + max_chase_bps / 10_000.0)
        fill = raw * (1.0 + market_slippage_bps / 10_000.0)
        slip_per_unit = fill - raw
    else:
        allowed = raw >= reference_price * (1.0 - max_chase_bps / 10_000.0)
        fill = raw * (1.0 - market_slippage_bps / 10_000.0)
        slip_per_unit = raw - fill
    if not allowed:
        return None

    return EntryFill(
        timestamp=deadline_ts,
        side=side,
        raw_price=raw,
        fill_price=fill,
        fee_rate=BACKTEST_TAKER_FEE,
        slippage_per_unit=slip_per_unit,
        liquidity="taker",
        route="delay60_market_fallback_conditional",
        baseline_open=reference_price,
    )


def build_causal_entry_candidates(
    *,
    signals: pd.DataFrame,
    minute_cache: pd.DataFrame,
    aggtrades: pd.DataFrame,
    side: str,
    market_slippage_bps: float,
    signal_timeframe: str = TIMEFRAME,
) -> CausalCandidateBuild:
    """Build the delay-only control and the preregistered order-flow gate.

    Both paths share exactly the same t+60s reference and delayed execution. The only
    difference is that the order-flow path vetoes ``adverse_acceptance``. This isolates
    information in executed flow from the mechanical effect of waiting one minute.
    """
    delay_candidates: list[EntryCandidate] = []
    gate_candidates: list[EntryCandidate] = []
    records: list[dict] = []

    for item in iter_signal_decisions(signals, side=side, signal_timeframe=signal_timeframe):
        entry_hour = item["entry_hour"]
        features = compute_flow_features(
            aggtrades,
            start=entry_hour,
            side=side,
            window_seconds=CAUSAL_DECISION_DELAY_SECONDS,
        )
        flow_class = classify_primary_window(features)
        veto = flow_class == "adverse_acceptance"
        reference, reference_source = decision_reference_price(
            aggtrades,
            decision_start=entry_hour,
            minute_cache=minute_cache,
            window_seconds=CAUSAL_DECISION_DELAY_SECONDS,
        )
        fill = None
        if reference is not None:
            fill = resolve_delayed_entry(
                minute_cache,
                entry_hour=entry_hour,
                side=side,
                reference_price=reference,
                market_slippage_bps=market_slippage_bps,
            )

        record = {
            **item,
            **features,
            "orderflow_class": flow_class,
            "orderflow_veto": veto,
            "decision_reference_price": reference,
            "decision_reference_source": reference_source,
            "delayed_fill": fill is not None,
            "delayed_fill_time": fill.timestamp if fill is not None else pd.NaT,
            "delayed_fill_route": fill.route if fill is not None else "",
            "delayed_fill_price": fill.fill_price if fill is not None else float("nan"),
            "gate_fill": bool(fill is not None and not veto),
        }
        records.append(record)

        if fill is None:
            continue
        candidate = EntryCandidate(
            signal_time=item["signal_time"],
            entry_hour=entry_hour,
            side=side,
            signal_atr=float(item["signal_atr"]),
            fill=fill,
        )
        delay_candidates.append(candidate)
        if not veto:
            gate_candidates.append(candidate)

    decisions = pd.DataFrame.from_records(records)
    return CausalCandidateBuild(delay_candidates, gate_candidates, decisions)
