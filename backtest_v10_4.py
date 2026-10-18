from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_END,
    BACKTEST_FEE_STRESS_MULTIPLIERS,
    BACKTEST_FUNDING_CACHE_DIR,
    BACKTEST_FUNDING_STRESS_MULTIPLIERS,
    BACKTEST_HISTORY_CACHE_DIR,
    BACKTEST_MARKET_SLIPPAGE_BPS,
    BACKTEST_PRIMARY_SLIPPAGE_BPS,
    BACKTEST_START,
    BACKTEST_TP_CROSS_TICKS,
    V10_4_ANNUAL_FILE,
    V10_4_BASE_SYMBOL,
    V10_4_CLEAN_OLD_ARTIFACTS,
    V10_4_COMPONENT_DD_LIMIT,
    V10_4_DEVELOPMENT_START_YEAR,
    V10_4_EXPANSION_FILE,
    V10_4_MARKETS_FILE,
    V10_4_MARKET_SYMBOLS,
    V10_4_MIN_COMPONENT_PF,
    V10_4_MIN_COMPONENT_TRADES,
    V10_4_MIN_PORTFOLIO_POSITIVE_YEARS,
    V10_4_MIN_POSITIVE_YEARS,
    V10_4_OPPORTUNITIES_FILE,
    V10_4_PORTFOLIO_DD_LIMIT,
    V10_4_PORTFOLIO_MIN_PF,
    V10_4_PORTFOLIOS_FILE,
    V10_4_RECOMMENDATION_FILE,
    V10_4_REFERENCE_SYMBOLS,
    V10_4_SENSITIVITY_FILE,
    V10_4_SHOCK1H_THRESHOLD,
    V10_4_STRESS_FILE,
    V10_4_TARGET_MAX_TRADES_PER_YEAR,
    V10_4_TARGET_MIN_TRADES_PER_YEAR,
)
from src.data.fetcher import fetch_binance_history_cached
from src.research.engine import (
    ExitPolicy,
    PortfolioConfig,
    build_funding_series,
    build_path_arrays,
    ensure_funding_cache,
    ensure_minute_cache,
    fetch_market_tick_size,
    funding_snapshot_info,
    generate_opportunities,
    portfolio_metrics,
    serialize_recommendation,
    simulate_portfolio,
)
from src.research.signals import build_frequency_expansion_signal, prepare_frequency_expansion_base


BASE_CONFIG = PortfolioConfig(
    exit_policy="fixed_rr",
    rr=1.8,
    risk_per_trade=0.005,
    max_position_pct=0.30,
    side_mode="long_only",
)
EXIT_POLICY = ExitPolicy("fixed_rr", 1.8)


@dataclass
class ComponentContext:
    component: str
    symbol: str
    signals: pd.DataFrame
    minute: pd.DataFrame
    funding: pd.DataFrame
    tick: float
    path_arrays: object
    funding_series: object


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def _development_bounds() -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(V10_4_DEVELOPMENT_START_YEAR, 1, 1, tz="UTC")
    end = pd.Timestamp(BACKTEST_END)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    return start, end


def _cadence(trades: pd.DataFrame) -> dict[str, float]:
    if trades.empty or "entry_time" not in trades.columns or len(trades) < 2:
        return {
            "median_gap_days": float("nan"),
            "p90_gap_days": float("nan"),
            "max_gap_days": float("nan"),
        }
    ts = pd.to_datetime(trades["entry_time"], utc=True).sort_values().drop_duplicates()
    gaps = ts.diff().dropna().dt.total_seconds() / 86400.0
    if gaps.empty:
        return {
            "median_gap_days": float("nan"),
            "p90_gap_days": float("nan"),
            "max_gap_days": float("nan"),
        }
    return {
        "median_gap_days": float(gaps.median()),
        "p90_gap_days": float(gaps.quantile(0.90)),
        "max_gap_days": float(gaps.max()),
    }


def _metrics(
    source: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    config: PortfolioConfig = BASE_CONFIG,
) -> tuple[dict, pd.DataFrame]:
    result = simulate_portfolio(source, config, start=start, end=end, prefiltered_config=True)
    metrics = portfolio_metrics(result)
    years = max((end - start).total_seconds() / (365.25 * 86400.0), 1 / 365.25)
    total_return = float(metrics["return"])
    metrics["annualized_return"] = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0)
        if total_return > -1.0
        else float("nan")
    )
    metrics["annualized_eur_on_10k"] = metrics["annualized_return"] * 10000.0
    metrics["trades_per_year"] = float(metrics["trades"] / years)
    metrics["trades_per_month"] = float(metrics["trades_per_year"] / 12.0)
    metrics.update(_cadence(result.trades))
    return metrics, result.trades


def _annual_metrics(
    source: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    label: str,
    kind: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(start.year, end.year + 1):
        ys = max(start, pd.Timestamp(year, 1, 1, tz="UTC"))
        ye = min(end, pd.Timestamp(year, 12, 31, 23, 59, 59, tz="UTC"))
        if ye <= ys:
            continue
        m, _ = _metrics(source, ys, ye)
        rows.append({"kind": kind, "name": label, "year": year, **m})
    return pd.DataFrame(rows)


def _positive_years(annual: pd.DataFrame) -> int:
    if annual.empty:
        return 0
    active = annual[annual["trades"] > 0]
    return int((active["return"] > 0).sum())


def _qualify_component(metrics: dict, annual: pd.DataFrame) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if int(metrics["trades"]) < V10_4_MIN_COMPONENT_TRADES:
        reasons.append("too_few_trades")
    if not math.isfinite(float(metrics["net_pf"])) or float(metrics["net_pf"]) < V10_4_MIN_COMPONENT_PF:
        reasons.append("pf")
    if float(metrics["return"]) <= 0:
        reasons.append("return")
    if float(metrics["max_dd"]) < -V10_4_COMPONENT_DD_LIMIT:
        reasons.append("dd")
    if _positive_years(annual) < V10_4_MIN_POSITIVE_YEARS:
        reasons.append("positive_years")
    return not reasons, reasons


def _qualify_portfolio(metrics: dict, positive_years: int) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not math.isfinite(float(metrics["net_pf"])) or float(metrics["net_pf"]) < V10_4_PORTFOLIO_MIN_PF:
        reasons.append("pf")
    if float(metrics["max_dd"]) < -V10_4_PORTFOLIO_DD_LIMIT:
        reasons.append("dd")
    tpy = float(metrics["trades_per_year"])
    if tpy < V10_4_TARGET_MIN_TRADES_PER_YEAR:
        reasons.append("frequency_low")
    if tpy > V10_4_TARGET_MAX_TRADES_PER_YEAR:
        reasons.append("frequency_high")
    if positive_years < V10_4_MIN_PORTFOLIO_POSITIVE_YEARS:
        reasons.append("positive_years")
    return not reasons, reasons


def _legacy_funding_needs_snapshot(symbol: str) -> bool:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in symbol).strip("_")
    base = Path(BACKTEST_FUNDING_CACHE_DIR) / slug
    return base.with_suffix(".csv").exists() and not base.with_suffix(".meta.json").exists()


def _cleanup_old_artifacts() -> list[str]:
    if not V10_4_CLEAN_OLD_ARTIFACTS:
        return []
    state = Path("state")
    if not state.exists():
        return []
    removed: list[str] = []
    for pattern in (
        "backtest_v9*",
        "backtest_v10*",
        "backtest_v10_1*",
        "backtest_v10_2*",
        "backtest_v10_3*",
    ):
        for path in state.glob(pattern):
            if "v10_4" in path.name:
                continue
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return sorted(set(removed))


def _build_component_context(*, symbol: str, rebuild_funding: bool) -> ComponentContext:
    component = f"{symbol}@1h"
    print(f"\n{component}: historia signal=1h, regimen=1h/4h")
    h1 = fetch_binance_history_cached(
        symbol, "1h", BACKTEST_START, BACKTEST_END, BACKTEST_HISTORY_CACHE_DIR
    )
    h4 = fetch_binance_history_cached(
        symbol, "4h", BACKTEST_START, BACKTEST_END, BACKTEST_HISTORY_CACHE_DIR
    )
    base = prepare_frequency_expansion_base(h1, h1, h4, signal_timeframe="1h", atr_hours=14.0)
    frame = build_frequency_expansion_signal(
        base,
        signal_timeframe="1h",
        shock_threshold=V10_4_SHOCK1H_THRESHOLD,
        reference_1h=h1,
        z_window_hours=20.0,
        z_threshold=2.2,
    )
    total = int(frame["entry_long"].fillna(False).sum())
    dev_start, _ = _development_bounds()
    recent = int(frame.loc[frame.index >= dev_start, "entry_long"].fillna(False).sum())
    print(f"  signals={total} | {V10_4_DEVELOPMENT_START_YEAR}+={recent}")

    minute = ensure_minute_cache(symbol, frame, include_shorts=False, timeframe="1h")
    full_start = frame.index.min()
    full_end = frame.index.max() + pd.Timedelta(hours=1)
    effective_rebuild = rebuild_funding or _legacy_funding_needs_snapshot(symbol)
    if effective_rebuild and not rebuild_funding:
        print(f"  {symbol}: funding legacy -> rebuild determinista una vez")
    funding = ensure_funding_cache(
        symbol, full_start.isoformat(), full_end.isoformat(), rebuild=effective_rebuild
    )
    snapshot = funding_snapshot_info(symbol)
    print(f"  funding rows={snapshot['rows']} sha256={snapshot['values_sha256'][:12]}...")
    tick = fetch_market_tick_size(symbol)
    if tick <= 0:
        tick = max(float(base["close"].median()) * 1e-9, 1e-12)
    arrays = build_path_arrays(frame)
    funding_series = build_funding_series(funding, frame, arrays)
    return ComponentContext(
        component=component,
        symbol=symbol,
        signals=frame,
        minute=minute,
        funding=funding,
        tick=tick,
        path_arrays=arrays,
        funding_series=funding_series,
    )


def _generate_context(ctx: ComponentContext, slip: float) -> pd.DataFrame:
    out = generate_opportunities(
        symbol=ctx.symbol,
        signals=ctx.signals,
        minute_cache=ctx.minute,
        funding=ctx.funding,
        exit_policy=EXIT_POLICY,
        market_slippage_bps=float(slip),
        tick_size=ctx.tick,
        include_shorts=False,
        path_arrays=ctx.path_arrays,
        funding_series=ctx.funding_series,
        tp_cross_ticks=BACKTEST_TP_CROSS_TICKS,
        signal_timeframe="1h",
    )
    if out.empty:
        return out
    out.insert(0, "component", ctx.component)
    out.insert(1, "signal_timeframe", "1h")
    return out


def _component_table(
    opportunities: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    annual_frames: list[pd.DataFrame] = []
    for symbol in V10_4_MARKET_SYMBOLS:
        component = f"{symbol}@1h"
        source = opportunities[opportunities["component"].eq(component)]
        m, _ = _metrics(source, start, end)
        annual = _annual_metrics(source, start, end, label=component, kind="component_market")
        eligible, reasons = _qualify_component(m, annual)
        annual_frames.append(annual)
        rows.append(
            {
                "component": component,
                "symbol": symbol,
                "is_reference": symbol in V10_4_REFERENCE_SYMBOLS,
                "trades": int(m["trades"]),
                "trades_per_year": float(m["trades_per_year"]),
                "trades_per_month": float(m["trades_per_month"]),
                "median_gap_days": float(m["median_gap_days"]),
                "p90_gap_days": float(m["p90_gap_days"]),
                "max_gap_days": float(m["max_gap_days"]),
                "net_pf": float(m["net_pf"]),
                "return": float(m["return"]),
                "annualized_return": float(m["annualized_return"]),
                "annualized_eur_on_10k": float(m["annualized_eur_on_10k"]),
                "max_dd": float(m["max_dd"]),
                "positive_years": _positive_years(annual),
                "eligible": bool(eligible),
                "rejection_reasons": ",".join(reasons),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["eligible", "net_pf", "annualized_return"], ascending=[False, False, False]
    ).reset_index(drop=True)
    annual_all = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    return table, annual_all


def _portfolio_row(
    name: str,
    source: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict, pd.DataFrame]:
    m, _ = _metrics(source, start, end)
    annual = _annual_metrics(source, start, end, label=name, kind="portfolio")
    pos_years = _positive_years(annual)
    qualifies, reasons = _qualify_portfolio(m, pos_years)
    row = {
        "portfolio": name,
        "components": ",".join(sorted(source["component"].unique())) if not source.empty else "",
        "trades": int(m["trades"]),
        "trades_per_year": float(m["trades_per_year"]),
        "trades_per_month": float(m["trades_per_month"]),
        "median_gap_days": float(m["median_gap_days"]),
        "p90_gap_days": float(m["p90_gap_days"]),
        "max_gap_days": float(m["max_gap_days"]),
        "net_pf": float(m["net_pf"]),
        "return": float(m["return"]),
        "annualized_return": float(m["annualized_return"]),
        "annualized_eur_on_10k": float(m["annualized_eur_on_10k"]),
        "max_dd": float(m["max_dd"]),
        "positive_years": pos_years,
        "frequency_in_target_range": bool(
            V10_4_TARGET_MIN_TRADES_PER_YEAR
            <= float(m["trades_per_year"])
            <= V10_4_TARGET_MAX_TRADES_PER_YEAR
        ),
        "meets_frequency_objective": bool(qualifies),
        "rejection_reasons": ",".join(reasons),
    }
    return row, annual


def _build_expansion_path(
    opportunities: pd.DataFrame,
    market_table: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    eligible = market_table[market_table["eligible"]].copy()
    eligible_components = set(eligible["component"])

    reference_components = [
        f"{symbol}@1h"
        for symbol in V10_4_REFERENCE_SYMBOLS
        if f"{symbol}@1h" in eligible_components
    ]
    expansion = eligible[~eligible["symbol"].isin(V10_4_REFERENCE_SYMBOLS)].copy()
    expansion = expansion.sort_values(["net_pf", "annualized_return"], ascending=[False, False])

    sources: dict[str, pd.DataFrame] = {}
    rows: list[dict] = []
    annual_frames: list[pd.DataFrame] = []

    current = list(reference_components)
    steps: list[tuple[str, list[str], str]] = [("reference", current.copy(), "")]
    for _, row in expansion.iterrows():
        added = str(row["component"])
        current = current + [added]
        steps.append((f"add_{len(current) - len(reference_components)}", current.copy(), added))

    # Fixed pre-registered construction rule: reference first, then individually
    # eligible expansion markets in descending individual PF. No subset search.
    for step_no, (name, components, added) in enumerate(steps):
        source = opportunities[opportunities["component"].isin(components)].copy()
        sources[name] = source
        row, annual = _portfolio_row(name, source, start, end)
        row = {
            "step": step_no,
            "added_component": added,
            "construction_rule": "reference_then_eligible_by_individual_pf",
            **row,
        }
        rows.append(row)
        annual_frames.append(annual)

    expansion_table = pd.DataFrame(rows)
    annual_all = pd.concat(annual_frames, ignore_index=True) if annual_frames else pd.DataFrame()
    return expansion_table, sources, annual_all


def _apply_economic_stress(source: pd.DataFrame, fee_mult: float, funding_mult: float) -> pd.DataFrame:
    out = source.copy()
    out["fee_per_unit"] = out["fee_per_unit"].astype(float) * float(fee_mult)
    out["funding_pnl_per_unit"] = out["funding_pnl_per_unit"].astype(float) * float(funding_mult)
    out["net_pnl_per_unit"] = (
        out["gross_pnl_per_unit"].astype(float)
        - out["fee_per_unit"].astype(float)
        + out["funding_pnl_per_unit"].astype(float)
    )
    return out


def _select_forward_candidate(expansion: pd.DataFrame) -> dict | None:
    eligible = expansion[expansion["meets_frequency_objective"]].copy()
    if eligible.empty:
        return None
    # The construction rule is sequential. Select the FIRST step reaching all
    # gates, not the historically best later subset/step.
    return eligible.sort_values("step").iloc[0].to_dict()


def _print_components(table: pd.DataFrame) -> None:
    print("\nV10.4 CROSS-ASSET 1h | edge individual obligatorio")
    cols = [
        "component",
        "trades",
        "trades_per_year",
        "trades_per_month",
        "net_pf",
        "annualized_return",
        "annualized_eur_on_10k",
        "max_dd",
        "positive_years",
        "eligible",
        "rejection_reasons",
    ]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_portfolios(table: pd.DataFrame) -> None:
    print("\nV10.4 PORTFOLIO DIAGNOSTICS")
    cols = [
        "portfolio",
        "trades",
        "trades_per_year",
        "trades_per_month",
        "median_gap_days",
        "net_pf",
        "annualized_return",
        "annualized_eur_on_10k",
        "max_dd",
        "positive_years",
        "frequency_in_target_range",
        "meets_frequency_objective",
        "rejection_reasons",
    ]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_expansion(table: pd.DataFrame) -> None:
    print("\nV10.4 PORTFOLIO EXPANSION PATH | regla fija, sin subset search")
    cols = [
        "step",
        "added_component",
        "trades",
        "trades_per_year",
        "trades_per_month",
        "median_gap_days",
        "net_pf",
        "annualized_return",
        "annualized_eur_on_10k",
        "max_dd",
        "positive_years",
        "frequency_in_target_range",
        "meets_frequency_objective",
        "rejection_reasons",
    ]
    print(table[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run(*, rebuild_funding: bool = False) -> None:
    total = time.perf_counter()
    removed = _cleanup_old_artifacts()
    if removed:
        print(f"Limpieza: eliminados {len(removed)} artifacts research antiguos (caches/live intactos).")

    print("V10.4 RESEARCH | cross-asset frequency expansion; z-score + shock1h congelado")
    print(f"Mercados 1h={V10_4_MARKET_SYMBOLS}")
    print(f"Referencia={V10_4_REFERENCE_SYMBOLS}")
    print("Regimen congelado=1h_sma200_4h_ema200 | z=20/2.2 | shock1h=2.175 | ATR=14")
    print("Execution: LIMIT -3bp | wait=7m | cap=5bp | RR=1.8 | risk=0.5% | maxpos=30%")
    print(
        f"Gate componente: trades>={V10_4_MIN_COMPONENT_TRADES}, PF>={V10_4_MIN_COMPONENT_PF:g}, "
        f"DD<={V10_4_COMPONENT_DD_LIMIT:.0%}, anos+>={V10_4_MIN_POSITIVE_YEARS}"
    )
    print(
        f"Objetivo portfolio: {V10_4_TARGET_MIN_TRADES_PER_YEAR:g}-{V10_4_TARGET_MAX_TRADES_PER_YEAR:g} "
        f"trades/ano, PF>={V10_4_PORTFOLIO_MIN_PF:g}, DD<={V10_4_PORTFOLIO_DD_LIMIT:.0%}, "
        f"anos+>={V10_4_MIN_PORTFOLIO_POSITIVE_YEARS}"
    )
    print("Construccion: BTC+BNB referencia -> anadir mercados elegibles por PF individual, sin buscar subsets.")
    print("Metodologia: thresholds NO se retunean; nuevos activos son validacion cross-sectional, no holdout temporal.")

    dev_start, dev_end = _development_bounds()
    contexts: dict[str, ComponentContext] = {}
    opp_frames: list[pd.DataFrame] = []

    phase = time.perf_counter()
    for symbol in V10_4_MARKET_SYMBOLS:
        ctx = _build_component_context(symbol=symbol, rebuild_funding=rebuild_funding)
        contexts[ctx.component] = ctx
        opps = _generate_context(ctx, BACKTEST_PRIMARY_SLIPPAGE_BPS)
        if not opps.empty:
            opp_frames.append(opps)
        print(f"  {ctx.component}: {len(opps)} trades-path @ {BACKTEST_PRIMARY_SLIPPAGE_BPS:g}bp")
    print(f"FASE data + opportunities: {_elapsed(phase)}")

    opportunities = pd.concat(opp_frames, ignore_index=True) if opp_frames else pd.DataFrame()
    if opportunities.empty:
        raise RuntimeError("V10.4 no genero oportunidades")
    Path(V10_4_OPPORTUNITIES_FILE).parent.mkdir(parents=True, exist_ok=True)
    opportunities.to_csv(V10_4_OPPORTUNITIES_FILE, index=False)

    phase = time.perf_counter()
    market_table, annual_market = _component_table(opportunities, dev_start, dev_end)
    _print_components(market_table)
    expansion, expansion_sources, annual_expansion = _build_expansion_path(
        opportunities, market_table, dev_start, dev_end
    )
    _print_expansion(expansion)

    # Additional fixed diagnostics: BTC only, BTC+BNB reference, all qualified,
    # all markets. These do not participate in selecting the forward candidate.
    diagnostics: list[dict] = []
    annual_diag: list[pd.DataFrame] = []
    diag_sources: dict[str, pd.DataFrame] = {}
    component_by_symbol = {s: f"{s}@1h" for s in V10_4_MARKET_SYMBOLS}
    eligible_components = set(market_table.loc[market_table["eligible"], "component"])
    diag_defs = {
        "btc_control": [component_by_symbol[V10_4_BASE_SYMBOL]],
        "btc_bnb_reference": [component_by_symbol[s] for s in V10_4_REFERENCE_SYMBOLS if s in component_by_symbol],
        "all_qualified": sorted(eligible_components),
        "all_markets_diagnostic": [component_by_symbol[s] for s in V10_4_MARKET_SYMBOLS],
    }
    for name, components in diag_defs.items():
        source = opportunities[opportunities["component"].isin(components)].copy()
        diag_sources[name] = source
        row, annual = _portfolio_row(name, source, dev_start, dev_end)
        diagnostics.append(row)
        annual_diag.append(annual)
    portfolios = pd.DataFrame(diagnostics).sort_values(
        ["meets_frequency_objective", "net_pf", "annualized_return"],
        ascending=[False, False, False],
    )
    _print_portfolios(portfolios)

    market_table.to_csv(V10_4_MARKETS_FILE, index=False)
    expansion.to_csv(V10_4_EXPANSION_FILE, index=False)
    portfolios.to_csv(V10_4_PORTFOLIOS_FILE, index=False)
    annual = pd.concat(
        [annual_market, annual_expansion] + annual_diag,
        ignore_index=True,
    )
    annual.to_csv(V10_4_ANNUAL_FILE, index=False)
    print(f"FASE qualification/portfolio: {_elapsed(phase)}")

    chosen = _select_forward_candidate(expansion)
    chosen_step = int(chosen["step"]) if chosen else None
    target_name = f"step_{chosen_step}" if chosen else None
    if chosen:
        target_source = expansion_sources[f"add_{chosen_step}"] if chosen_step > 0 else expansion_sources["reference"]
        target_label = "selected_expansion_step"
    else:
        # If objective is not reached, stress the final deterministic expansion step.
        final_row = expansion.sort_values("step").iloc[-1]
        final_step = int(final_row["step"])
        key = "reference" if final_step == 0 else f"add_{final_step}"
        target_source = expansion_sources[key]
        target_label = "final_expansion_step_diagnostic"

    phase = time.perf_counter()
    target_components = sorted(target_source["component"].unique())
    sensitivity_rows: list[dict] = []
    for slip in BACKTEST_MARKET_SLIPPAGE_BPS:
        if np.isclose(float(slip), BACKTEST_PRIMARY_SLIPPAGE_BPS):
            source = target_source.copy()
        else:
            regen: list[pd.DataFrame] = []
            for component in target_components:
                ctx = contexts.get(component)
                if ctx is None:
                    continue
                r = _generate_context(ctx, float(slip))
                if not r.empty:
                    regen.append(r)
            source = pd.concat(regen, ignore_index=True) if regen else pd.DataFrame()
        m, _ = _metrics(source, dev_start, dev_end)
        sensitivity_rows.append({"portfolio": target_label, "slippage_bps": float(slip), **m})
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(V10_4_SENSITIVITY_FILE, index=False)
    print(f"FASE sensibilidad final: {_elapsed(phase)}")

    phase = time.perf_counter()
    stress_rows: list[dict] = []
    for fee_mult, funding_mult in itertools.product(
        BACKTEST_FEE_STRESS_MULTIPLIERS, BACKTEST_FUNDING_STRESS_MULTIPLIERS
    ):
        stressed = _apply_economic_stress(target_source, float(fee_mult), float(funding_mult))
        m, _ = _metrics(stressed, dev_start, dev_end)
        stress_rows.append(
            {
                "portfolio": target_label,
                "fee_multiplier": float(fee_mult),
                "funding_multiplier": float(funding_mult),
                **m,
            }
        )
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(V10_4_STRESS_FILE, index=False)
    print(f"FASE stress final: {_elapsed(phase)}")

    snapshots = {symbol: funding_snapshot_info(symbol) for symbol in V10_4_MARKET_SYMBOLS}
    recommendation = {
        "method": "v10_4_cross_asset_frequency_expansion",
        "data_status": {
            "development_start_year": V10_4_DEVELOPMENT_START_YEAR,
            "fresh_temporal_holdout": False,
            "cross_asset_external_validation": True,
            "note": (
                "thresholds were discovered on BTC; new assets are cross-sectional tests. "
                "Portfolio selection performed here requires fresh forward validation."
            ),
        },
        "frozen_setup": {
            "timeframe": "1h",
            "z_window": 20,
            "z_threshold": 2.2,
            "shock1h_threshold": V10_4_SHOCK1H_THRESHOLD,
            "atr": 14,
            "regime": "1h_sma200_4h_ema200",
            "rr": 1.8,
            "risk_per_trade": 0.005,
            "max_position_pct": 0.30,
            "side_mode": "long_only",
        },
        "reference_symbols": list(V10_4_REFERENCE_SYMBOLS),
        "market_symbols": list(V10_4_MARKET_SYMBOLS),
        "component_gate": {
            "min_trades": V10_4_MIN_COMPONENT_TRADES,
            "min_pf": V10_4_MIN_COMPONENT_PF,
            "max_dd": V10_4_COMPONENT_DD_LIMIT,
            "min_positive_years": V10_4_MIN_POSITIVE_YEARS,
        },
        "frequency_objective": {
            "target_trades_per_year": [V10_4_TARGET_MIN_TRADES_PER_YEAR, V10_4_TARGET_MAX_TRADES_PER_YEAR],
            "min_portfolio_pf": V10_4_PORTFOLIO_MIN_PF,
            "max_portfolio_dd": V10_4_PORTFOLIO_DD_LIMIT,
            "min_positive_years": V10_4_MIN_PORTFOLIO_POSITIVE_YEARS,
        },
        "construction_rule": "reference_then_eligible_by_individual_pf; first passing step wins; no subset search",
        "markets": market_table.to_dict(orient="records"),
        "expansion_path": expansion.to_dict(orient="records"),
        "diagnostic_portfolios": portfolios.to_dict(orient="records"),
        "selected_historical_forward_candidate": chosen,
        "sensitivity_portfolio": target_label,
        "funding_snapshots": snapshots,
        "promotion_to_live": False,
        "promotion_reason": "cross-asset portfolio selection requires fresh forward validation",
        "live_signal_unchanged": True,
    }
    serialize_recommendation(V10_4_RECOMMENDATION_FILE, recommendation)

    print("\nDECISION V10.4")
    print(
        json.dumps(
            {
                "frequency_candidate_found": chosen is not None,
                "selected_historical_forward_candidate": chosen,
                "promotion_to_live": False,
                "reason": recommendation["promotion_reason"],
            },
            indent=2,
            default=str,
        )
    )
    print("\nFicheros V10.4:")
    for path in (
        V10_4_MARKETS_FILE,
        V10_4_EXPANSION_FILE,
        V10_4_PORTFOLIOS_FILE,
        V10_4_ANNUAL_FILE,
        V10_4_OPPORTUNITIES_FILE,
        V10_4_SENSITIVITY_FILE,
        V10_4_STRESS_FILE,
        V10_4_RECOMMENDATION_FILE,
    ):
        print(f"  {path}")
    print(f"\nTOTAL: {_elapsed(total)}")
    print("IMPORTANTE: V10.4 expande mercados en research; no modifica main.py ni live.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10.4 cross-asset frequency expansion research")
    parser.add_argument(
        "--rebuild-funding",
        action="store_true",
        help="Reconstruye deliberadamente todos los snapshots funding usados por V10.4.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(rebuild_funding=args.rebuild_funding)
