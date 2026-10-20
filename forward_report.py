from __future__ import annotations

import argparse
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import pandas as pd

from config.settings import (
    BACKTEST_PRIMARY_SLIPPAGE_BPS,
    COMPONENT_SPECS,
    EXECUTION_JOURNAL_FILE,
    FORWARD_EXPERIMENT_ID,
    FORWARD_JOURNAL_FILE,
    FORWARD_START_UTC,
    MAX_CONCURRENT_POSITIONS,
    MAX_PORTFOLIO_RISK_PCT,
    MAX_POSITION_PCT,
    RISK_REWARD_RATIO,
    STRATEGY_VERSION,
    TIMEFRAME,
    frozen_config,
    validate_forward_settings,
)
from src.forward.config import canonical_config_sha256
from src.forward.journal import ForwardJournal
from src.research.engine import (
    EntryCandidate,
    PortfolioConfig,
    build_funding_series,
    build_path_arrays,
    build_signals,
    ensure_funding_cache,
    ensure_minute_cache,
    fetch_market_tick_size,
    generate_opportunities,
    parse_exit_profile,
    portfolio_metrics,
    prepare_entry_candidates,
    simulate_portfolio,
)
from src.research.orderflow_causal import iter_signal_decisions, resolve_delayed_entry


REPORT_WARMUP_DAYS = 90


def _utc(value: str | datetime | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _profit_factor(values: list[float]) -> float:
    gains = sum(value for value in values if value > 0)
    losses = -sum(value for value in values if value < 0)
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return gains / losses


def _current_forward_records(config_sha256: str) -> list[dict]:
    records = ForwardJournal(FORWARD_JOURNAL_FILE).records()
    return [
        row
        for row in records
        if row.get("experiment_id") == FORWARD_EXPERIMENT_ID
        and row.get("config_sha256") == config_sha256
    ]


def _arm_map(records: list[dict]) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in records:
        if row.get("event") != "orderflow_armed":
            continue
        symbol = str(row.get("symbol") or "")
        signal_time = str(row.get("signal_time") or "")
        if symbol and signal_time:
            result[(symbol, signal_time)] = row
    return result


def _flow_map(records: list[dict]) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for row in records:
        if row.get("event") != "orderflow_decision":
            continue
        symbol = str(row.get("symbol") or "")
        signal_time = str(row.get("signal_time") or "")
        if symbol and signal_time:
            result[(symbol, signal_time)] = row
    return result


def _journal_summary(records: list[dict]) -> dict[str, int]:
    events = defaultdict(int)
    classes = defaultdict(int)
    for row in records:
        events[str(row.get("event") or "unknown")] += 1
        if row.get("event") == "orderflow_decision":
            classes[str(row.get("orderflow_class") or "unknown")] += 1
    return {
        "signals_evaluated": events["signal_evaluated"],
        "armed": events["orderflow_armed"],
        "flow_decisions": events["orderflow_decision"],
        "adverse_acceptance": classes["adverse_acceptance"],
        "adverse_absorption": classes["adverse_absorption"],
        "aligned_support": classes["aligned_support"],
        "mixed": classes["mixed"],
        "missing_or_expired": classes["missing"] + classes["expired"],
        "orders_submitted": events["order_submitted"],
    }


def _execution_demo_metrics(config_sha256: str) -> dict[str, object]:
    records = _read_jsonl(EXECUTION_JOURNAL_FILE)
    open_entries: dict[str, deque[dict]] = defaultdict(deque)
    pnls: list[float] = []
    entry_slippage: list[float] = []
    exit_slippage: list[float] = []
    fills = 0
    closes = 0

    for row in records:
        event = row.get("event")
        symbol = str(row.get("symbol") or "")
        if event == "entry_fill":
            context = row.get("context") or {}
            if context.get("experiment_id") != FORWARD_EXPERIMENT_ID:
                continue
            if context.get("config_sha256") != config_sha256:
                continue
            open_entries[symbol].append(row)
            fills += 1
            try:
                value = float(row.get("entry_slippage_bps"))
                if math.isfinite(value):
                    entry_slippage.append(value)
            except (TypeError, ValueError):
                pass
            continue

        if event != "position_closed_detected" or not open_entries.get(symbol):
            continue
        entry = open_entries[symbol].popleft()
        try:
            side = str(entry["side"])
            amount = float(entry["filled_amount"])
            entry_price = float(entry["fill_price"])
            exit_price = float(row["actual_exit_price"])
        except (KeyError, TypeError, ValueError):
            continue
        direction = 1.0 if side == "long" else -1.0
        pnls.append(direction * (exit_price - entry_price) * amount)
        closes += 1
        try:
            value = float(row.get("exit_slippage_bps"))
            if math.isfinite(value):
                exit_slippage.append(value)
        except (TypeError, ValueError):
            pass

    return {
        "fills": fills,
        "closed_trades": closes,
        "open_fills": sum(len(items) for items in open_entries.values()),
        "gross_pnl": float(sum(pnls)),
        "gross_pf": _profit_factor(pnls),
        "win_rate": float(sum(value > 0 for value in pnls) / len(pnls)) if pnls else None,
        "mean_entry_slippage_bps": (
            float(sum(entry_slippage) / len(entry_slippage)) if entry_slippage else None
        ),
        "mean_exit_slippage_bps": (
            float(sum(exit_slippage) / len(exit_slippage)) if exit_slippage else None
        ),
        "note": "Demo gross PnL/PF excludes funding and any commission not observable from matched close records.",
    }


def _observed_signal_frame(
    signals: pd.DataFrame,
    *,
    symbol: str,
    side: str,
    comparable_keys: set[tuple[str, str]],
) -> pd.DataFrame:
    selected = signals.copy()
    selected["entry_long"] = False
    selected["entry_short"] = False
    flag = "entry_long" if side == "long" else "entry_short"
    for key_symbol, signal_time in comparable_keys:
        if key_symbol != symbol:
            continue
        ts = _utc(signal_time)
        if ts in selected.index:
            selected.loc[ts, flag] = True
    return selected


def _exact_atr(entry: EntryCandidate, arm_lookup: dict[tuple[str, str], dict], symbol: str) -> float:
    row = arm_lookup.get((symbol, entry.signal_time.isoformat()))
    if row is None:
        return entry.signal_atr
    try:
        return float(row["atr"])
    except (KeyError, TypeError, ValueError):
        return entry.signal_atr


def _replay_shadows(
    records: list[dict],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    arms = _arm_map(records)
    flow = _flow_map(records)
    comparable = {
        key
        for key, row in flow.items()
        if row.get("delay_shadow_reference_price") is not None
        and row.get("orderflow_class") not in {"missing", "expired"}
        and key in arms
    }
    if not comparable:
        return pd.DataFrame(), {}

    warmup = start - pd.Timedelta(days=REPORT_WARMUP_DAYS)
    exit_policy = parse_exit_profile("fixed_rr", RISK_REWARD_RATIO)
    risk_map: dict[str, float] = {}
    frames: dict[str, list[pd.DataFrame]] = {
        "baseline_shadow": [],
        "delay_shadow": [],
        "orderflow_shadow": [],
    }

    for symbol, side, risk_per_trade in COMPONENT_SPECS:
        component = f"{symbol}@{TIMEFRAME}:{side}"
        risk_map[component] = float(risk_per_trade)
        signals = build_signals(
            symbol,
            start=warmup.isoformat(),
            end=end.isoformat(),
        )
        signals = _observed_signal_frame(
            signals,
            symbol=symbol,
            side=side,
            comparable_keys=comparable,
        )
        include_shorts = side == "short"
        minute = ensure_minute_cache(symbol, signals, include_shorts=include_shorts)
        funding = ensure_funding_cache(symbol, start.isoformat(), end.isoformat())
        tick = fetch_market_tick_size(symbol)
        arrays = build_path_arrays(signals)
        funding_fast = build_funding_series(funding, signals, arrays)

        base_candidates = prepare_entry_candidates(
            signals=signals,
            minute_cache=minute,
            market_slippage_bps=BACKTEST_PRIMARY_SLIPPAGE_BPS,
            include_shorts=include_shorts,
        )
        base_candidates = [
            EntryCandidate(
                signal_time=item.signal_time,
                entry_hour=item.entry_hour,
                side=item.side,
                signal_atr=_exact_atr(item, arms, symbol),
                fill=item.fill,
            )
            for item in base_candidates
            if item.side == side
        ]

        delay_candidates: list[EntryCandidate] = []
        gate_candidates: list[EntryCandidate] = []
        for item in iter_signal_decisions(signals, side=side):
            key = (symbol, item["signal_time"].isoformat())
            flow_row = flow.get(key)
            arm_row = arms.get(key)
            if flow_row is None or arm_row is None or key not in comparable:
                continue
            try:
                reference = float(flow_row["delay_shadow_reference_price"])
                signal_atr = float(arm_row["atr"])
            except (KeyError, TypeError, ValueError):
                continue
            fill = resolve_delayed_entry(
                minute,
                entry_hour=item["entry_hour"],
                side=side,
                reference_price=reference,
                market_slippage_bps=BACKTEST_PRIMARY_SLIPPAGE_BPS,
            )
            if fill is None:
                continue
            candidate = EntryCandidate(
                signal_time=item["signal_time"],
                entry_hour=item["entry_hour"],
                side=side,
                signal_atr=signal_atr,
                fill=fill,
            )
            delay_candidates.append(candidate)
            if not bool(flow_row.get("orderflow_veto")):
                gate_candidates.append(candidate)

        for label, candidates in (
            ("baseline_shadow", base_candidates),
            ("delay_shadow", delay_candidates),
            ("orderflow_shadow", gate_candidates),
        ):
            opportunities = generate_opportunities(
                symbol=symbol,
                signals=signals,
                minute_cache=minute,
                funding=funding,
                exit_policy=exit_policy,
                market_slippage_bps=BACKTEST_PRIMARY_SLIPPAGE_BPS,
                tick_size=tick,
                include_shorts=include_shorts,
                entry_candidates=candidates,
                path_arrays=arrays,
                funding_series=funding_fast,
            )
            if opportunities.empty:
                continue
            opportunities = opportunities.loc[opportunities["side"].eq(side)].copy()
            opportunities["component"] = component
            frames[label].append(opportunities)

    config = PortfolioConfig(
        exit_policy="fixed_rr",
        rr=RISK_REWARD_RATIO,
        risk_per_trade=0.005,
        max_position_pct=MAX_POSITION_PCT,
        side_mode="long_short",
    )
    metrics: list[dict] = []
    trade_outputs: dict[str, pd.DataFrame] = {}
    duration_years = max((end - start).total_seconds() / (365.25 * 86400.0), 1 / 365.25)
    for label in ("baseline_shadow", "delay_shadow", "orderflow_shadow"):
        opportunities = (
            pd.concat(frames[label], ignore_index=True) if frames[label] else pd.DataFrame()
        )
        result = simulate_portfolio(
            opportunities,
            config,
            start=start,
            end=end,
            prefiltered_config=True,
            max_portfolio_risk_pct=MAX_PORTFOLIO_RISK_PCT,
            max_concurrent_positions=MAX_CONCURRENT_POSITIONS,
            risk_per_trade_by_component=risk_map,
        )
        row = portfolio_metrics(result)
        row["variant"] = label
        row["trades_per_year"] = row["trades"] / duration_years
        metrics.append(row)
        trade_outputs[label] = result.trades
    return pd.DataFrame.from_records(metrics), trade_outputs


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "inf" if number > 0 else "n/a"
    return f"{number:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fresh V12 order-flow forward report")
    parser.add_argument(
        "--end",
        default="",
        help="UTC replay end (default: current closed-hour boundary).",
    )
    parser.add_argument(
        "--no-replay",
        action="store_true",
        help="Only report journals/Demo execution; do not fetch public replay data.",
    )
    args = parser.parse_args()

    validate_forward_settings()
    config_sha256 = canonical_config_sha256(frozen_config())
    start = _utc(FORWARD_START_UTC)
    end = _utc(args.end) if args.end else pd.Timestamp.now(tz="UTC").floor("h")
    if end <= start:
        raise SystemExit(f"No forward interval yet: {start.isoformat()} -> {end.isoformat()}")

    records = _current_forward_records(config_sha256)
    summary = _journal_summary(records)
    demo = _execution_demo_metrics(config_sha256)
    flow_rows = _flow_map(records)
    complete_flow = sum(
        row.get("delay_shadow_reference_price") is not None
        and row.get("orderflow_class") not in {"missing", "expired"}
        for row in flow_rows.values()
    )

    print("V12 ORDER-FLOW FORWARD REPORT")
    print(f"period={start.isoformat()} -> {end.isoformat()}")
    print(f"config={config_sha256[:12]} | strategy={STRATEGY_VERSION}")
    print(
        "journal | "
        f"armed={summary['armed']} decisions={summary['flow_decisions']} "
        f"comparable={complete_flow} veto={summary['adverse_acceptance']} "
        f"missing/expired={summary['missing_or_expired']} orders={summary['orders_submitted']}"
    )
    print(
        "flow classes | "
        f"acceptance={summary['adverse_acceptance']} "
        f"absorption={summary['adverse_absorption']} "
        f"support={summary['aligned_support']} mixed={summary['mixed']}"
    )
    print(
        "Demo execution | "
        f"fills={demo['fills']} closed={demo['closed_trades']} open={demo['open_fills']} "
        f"grossPF={_fmt(demo['gross_pf'])} grossPnL={_fmt(demo['gross_pnl'], 2)}"
    )

    if args.no_replay or complete_flow == 0:
        if complete_flow == 0:
            print("No complete order-flow decisions yet; shadow replay has nothing comparable.")
        return

    metrics, _ = _replay_shadows(records, start=start, end=end)
    if metrics.empty:
        print("Shadow replay produced no closed comparable trades yet.")
        return

    print("\nFRESH SHADOW COMPARISON (only live-observed complete flow decisions)")
    for row in metrics.to_dict("records"):
        print(
            f"  {row['variant']:18s} trades={int(row['trades']):3d} | "
            f"PF={_fmt(row['net_pf'])} | return={float(row['return']):+.2%} | "
            f"DD={float(row['max_dd']):.2%}"
        )
    print(
        "\nInterpretation: baseline/delay/order-flow shadows share the exact live-observed "
        "signal sample. Demo fills are operational diagnostics and are not substituted for "
        "mainnet shadow performance."
    )


if __name__ == "__main__":
    main()
