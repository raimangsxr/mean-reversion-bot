from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config.settings import (
    BACKTEST_DD_LIMIT,
    BACKTEST_END,
    BACKTEST_FEE_STRESS_MULTIPLIERS,
    BACKTEST_FUNDING_CACHE_DIR,
    BACKTEST_FUNDING_STRESS_MULTIPLIERS,
    BACKTEST_HISTORY_CACHE_DIR,
    BACKTEST_MARKET_SLIPPAGE_BPS,
    BACKTEST_PRIMARY_SLIPPAGE_BPS,
    BACKTEST_START,
    BACKTEST_SYMBOLS,
    BACKTEST_TP_CROSS_TICKS,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    INITIAL_CAPITAL,
    REGIME_MODE,
    REGIME_TIMEFRAME,
    TIMEFRAME,
    V10_2_ANNUAL_FILE,
    V10_2_CLEAN_OLD_ARTIFACTS,
    V10_2_DEVELOPMENT_START_YEAR,
    V10_2_MAX_POSITION_GRID,
    V10_2_OPPORTUNITIES_FILE,
    V10_2_RECOMMENDATION_FILE,
    V10_2_RISK_GRID,
    V10_2_SENSITIVITY_FILE,
    V10_2_SHOCK1H_THRESHOLD,
    V10_2_SHOCK3H_THRESHOLD,
    V10_2_SIZING_DD_LIMIT,
    V10_2_SIZING_FILE,
    V10_2_SIZING_MIN_PF,
    V10_2_STRESS_FILE,
    V10_2_SUMMARY_FILE,
    V10_2_TAKER_CACHE_DIR,
    V10_2_TAKER_MIN_SHOCK_ATR,
    V10_2_VWAP48H_THRESHOLD,
)
from src.data.fetcher import fetch_binance_history_cached, fetch_binance_taker_history_cached
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
    prepare_entry_candidates,
    serialize_recommendation,
    simulate_portfolio,
)
from src.research.signals import SignalSpec, apply_signal, parse_signal_spec, prepare_base_features


CONTROL = "zscore_only"
PRIMARY = "zscore_and_shock1h"
BASE_CONFIG = PortfolioConfig(
    exit_policy="fixed_rr",
    rr=1.8,
    risk_per_trade=0.005,
    max_position_pct=0.30,
    side_mode="long_only",
)
EXIT_POLICY = ExitPolicy("fixed_rr", 1.8)


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    return f"{seconds:.1f}s" if seconds < 60 else f"{seconds / 60:.1f}m"


def _common_bounds(base: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return base.index.min(), base.index.max() + pd.Timedelta(TIMEFRAME)


def _candidate_specs() -> dict[str, SignalSpec]:
    return {
        "zscore": parse_signal_spec("zscore_20_2.2"),
        "shock1h": parse_signal_spec(f"return_shock_1h_{V10_2_SHOCK1H_THRESHOLD:g}"),
        "shock3h": parse_signal_spec(f"return_shock_3h_{V10_2_SHOCK3H_THRESHOLD:g}"),
        "vwap48h": parse_signal_spec(f"vwap_atr_48h_{V10_2_VWAP48H_THRESHOLD:g}"),
        "taker": parse_signal_spec(f"taker_exhaustion_{V10_2_TAKER_MIN_SHOCK_ATR:g}"),
    }


def _build_base(symbol: str, specs: dict[str, SignalSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    print(f"{symbol}: historia {TIMEFRAME} cache/incremental {BACKTEST_START} -> {BACKTEST_END}")
    df = fetch_binance_history_cached(
        symbol, TIMEFRAME, BACKTEST_START, BACKTEST_END, BACKTEST_HISTORY_CACHE_DIR
    )
    print(f"{symbol}: historia {REGIME_TIMEFRAME} cache/incremental para regimen")
    regime = fetch_binance_history_cached(
        symbol, REGIME_TIMEFRAME, BACKTEST_START, BACKTEST_END, BACKTEST_HISTORY_CACHE_DIR
    )
    base = prepare_base_features(df, regime)
    print(f"{symbol}: historia taker-flow {TIMEFRAME} cache/incremental")
    taker = fetch_binance_taker_history_cached(
        symbol, TIMEFRAME, BACKTEST_START, BACKTEST_END, V10_2_TAKER_CACHE_DIR
    )
    return base, taker


def _and_frame(base: pd.DataFrame, trigger: pd.DataFrame, confirm: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    out["entry_long"] = (
        trigger["entry_long"].fillna(False).astype(bool)
        & confirm["entry_long"].fillna(False).astype(bool)
    )
    out["entry_short"] = False
    out["signal_value"] = trigger.get("signal_value")
    return out


def _build_frames(base: pd.DataFrame, taker: pd.DataFrame, specs: dict[str, SignalSpec]) -> dict[str, pd.DataFrame]:
    raw = {name: apply_signal(base, spec, taker=taker) for name, spec in specs.items()}
    frames = {
        CONTROL: raw["zscore"],
        PRIMARY: _and_frame(base, raw["zscore"], raw["shock1h"]),
        "zscore_and_shock3h": _and_frame(base, raw["zscore"], raw["shock3h"]),
        "zscore_and_vwap48h": _and_frame(base, raw["zscore"], raw["vwap48h"]),
        "zscore_and_taker": _and_frame(base, raw["zscore"], raw["taker"]),
    }
    for name, frame in frames.items():
        total = int(frame["entry_long"].fillna(False).sum())
        development = int(
            frame.loc[
                frame.index >= pd.Timestamp(V10_2_DEVELOPMENT_START_YEAR, 1, 1, tz="UTC"),
                "entry_long",
            ].fillna(False).sum()
        )
        print(f"  {name}: senales total={total} | {V10_2_DEVELOPMENT_START_YEAR}+={development}")
    return frames


def _union_frame(base: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    union = base.copy()
    mask = pd.Series(False, index=base.index)
    for frame in frames.values():
        mask |= frame["entry_long"].fillna(False).astype(bool)
    union["entry_long"] = mask
    union["entry_short"] = False
    return union


def _generate(
    *,
    symbol: str,
    name: str,
    signals: pd.DataFrame,
    minute: pd.DataFrame,
    funding: pd.DataFrame,
    tick: float,
    slip: float,
    path_arrays,
    funding_series,
) -> pd.DataFrame:
    candidates = prepare_entry_candidates(
        signals=signals,
        minute_cache=minute,
        market_slippage_bps=float(slip),
        include_shorts=False,
    )
    out = generate_opportunities(
        symbol=symbol,
        signals=signals,
        minute_cache=minute,
        funding=funding,
        exit_policy=EXIT_POLICY,
        market_slippage_bps=float(slip),
        tick_size=tick,
        include_shorts=False,
        entry_candidates=candidates,
        path_arrays=path_arrays,
        funding_series=funding_series,
        tp_cross_ticks=BACKTEST_TP_CROSS_TICKS,
    )
    if not out.empty:
        out.insert(0, "signal_name", name)
    return out


def _metrics(
    source: pd.DataFrame,
    config: PortfolioConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict, pd.DataFrame]:
    result = simulate_portfolio(
        source,
        config,
        start=start,
        end=end,
        prefiltered_config=True,
    )
    metrics = portfolio_metrics(result)
    years = max((end - start).total_seconds() / (365.25 * 86400), 1 / 365.25)
    total_return = float(metrics["return"])
    metrics["annualized_return"] = (
        float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1 else float("nan")
    )
    metrics["annualized_eur_on_10k"] = metrics["annualized_return"] * 10000.0
    return metrics, result.trades


def _summary(
    opportunities: pd.DataFrame,
    names: list[str],
    full_start: pd.Timestamp,
    development_start: pd.Timestamp,
    full_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name in names:
        source = opportunities[opportunities["signal_name"].eq(name)]
        full, _ = _metrics(source, BASE_CONFIG, full_start, full_end)
        dev, _ = _metrics(source, BASE_CONFIG, development_start, full_end)
        rows.append(
            {
                "signal_name": name,
                "full_trades": int(full["trades"]),
                "full_pf": float(full["net_pf"]),
                "full_return": float(full["return"]),
                "full_max_dd": float(full["max_dd"]),
                "development_trades": int(dev["trades"]),
                "development_pf": float(dev["net_pf"]),
                "development_return": float(dev["return"]),
                "development_annualized_return": float(dev["annualized_return"]),
                "development_eur_per_year_on_10k": float(dev["annualized_eur_on_10k"]),
                "development_max_dd": float(dev["max_dd"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["development_pf", "development_return"], ascending=[False, False]
    ).reset_index(drop=True)


def _annual(
    opportunities: pd.DataFrame,
    names: list[str],
    development_start: pd.Timestamp,
    full_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    for name in names:
        source = opportunities[opportunities["signal_name"].eq(name)]
        for year in range(development_start.year, full_end.year + 1):
            start = pd.Timestamp(year, 1, 1, tz="UTC")
            end = min(pd.Timestamp(year, 12, 31, 23, 59, tz="UTC"), full_end)
            m, _ = _metrics(source, BASE_CONFIG, start, end)
            m.update({"signal_name": name, "year": year})
            rows.append(m)
    return pd.DataFrame(rows)


def _sizing_grid(
    source: pd.DataFrame,
    development_start: pd.Timestamp,
    full_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    for risk, maxpos in itertools.product(V10_2_RISK_GRID, V10_2_MAX_POSITION_GRID):
        config = PortfolioConfig(
            exit_policy="fixed_rr",
            rr=1.8,
            risk_per_trade=float(risk),
            max_position_pct=float(maxpos),
            side_mode="long_only",
        )
        m, _ = _metrics(source, config, development_start, full_end)
        row = {
            "risk_per_trade": float(risk),
            "max_position_pct": float(maxpos),
            **{k: m[k] for k in (
                "trades", "net_pf", "return", "annualized_return", "annualized_eur_on_10k",
                "max_dd", "median_risk_utilization", "fees", "funding_pnl", "slippage_cost"
            )},
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    result["within_dd_limit"] = result["max_dd"] >= -V10_2_SIZING_DD_LIMIT
    result["within_robustness"] = (
        result["within_dd_limit"] & (result["net_pf"] >= V10_2_SIZING_MIN_PF)
    )
    return result.sort_values(
        ["within_robustness", "annualized_return", "net_pf"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


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


def _cleanup_old_artifacts() -> list[str]:
    if not V10_2_CLEAN_OLD_ARTIFACTS:
        return []
    state = Path("state")
    if not state.exists():
        return []
    removed: list[str] = []
    patterns = ("backtest_v9*", "backtest_v10_*", "backtest_v10.*")
    for pattern in patterns:
        for path in state.glob(pattern):
            if "v10_2" in path.name:
                continue
            if path.is_file():
                path.unlink()
                removed.append(str(path))
    return sorted(set(removed))


def _legacy_funding_needs_snapshot(symbol: str) -> bool:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in symbol).strip("_")
    base = Path(BACKTEST_FUNDING_CACHE_DIR) / slug
    return base.with_suffix(".csv").exists() and not base.with_suffix(".meta.json").exists()


def _print_summary(summary: pd.DataFrame) -> None:
    print("\nV10.2 SIGNAL/FILTER COMPARISON | 2022+ es DEVELOPMENT, no holdout nuevo")
    cols = [
        "signal_name", "development_trades", "development_pf", "development_return",
        "development_annualized_return", "development_eur_per_year_on_10k", "development_max_dd",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _print_sizing(sizing: pd.DataFrame) -> None:
    print("\nV10.2 SIZING | solo candidato principal z-score + shock1h")
    cols = [
        "risk_per_trade", "max_position_pct", "trades", "net_pf", "return",
        "annualized_return", "annualized_eur_on_10k", "max_dd", "median_risk_utilization",
        "within_robustness",
    ]
    print(sizing[cols].head(16).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def run(*, rebuild_funding: bool = False) -> None:
    total = time.perf_counter()
    if len(BACKTEST_SYMBOLS) != 1:
        raise ValueError("V10.2 requiere BACKTEST_SYMBOLS con un unico simbolo")
    symbol = BACKTEST_SYMBOLS[0]

    removed = _cleanup_old_artifacts()
    if removed:
        print(f"Limpieza: eliminados {len(removed)} artifacts research antiguos (caches/live intactos).")

    print("V10.2 RESEARCH | z-score trigger + confirmations; repo consolidado")
    print(f"Simbolo={symbol} | regimen={REGIME_MODE} ({TIMEFRAME}/{REGIME_TIMEFRAME})")
    print(f"Execution: LIMIT -{ENTRY_LIMIT_OFFSET_BPS:g}bp | wait={ENTRY_LIMIT_WAIT_SECONDS//60}m | cap={ENTRY_FALLBACK_CAP_BPS:g}bp")
    print("Frozen: LONG only | fixed_rr=1.8 | base risk=0.5% | base maxpos=30% | SL=1.5 ATR")
    print(
        "Confirmaciones pre-registradas V10.2: "
        f"shock1h={V10_2_SHOCK1H_THRESHOLD:g}, shock3h={V10_2_SHOCK3H_THRESHOLD:g}, "
        f"vwap48={V10_2_VWAP48H_THRESHOLD:g}, takerShockATR={V10_2_TAKER_MIN_SHOCK_ATR:g}"
    )
    print("Metodologia: 2022+ ya fue inspeccionado; se usa como DEVELOPMENT/diagnostico, no como holdout fresco.")

    phase = time.perf_counter()
    specs = _candidate_specs()
    base, taker = _build_base(symbol, specs)
    frames = _build_frames(base, taker, specs)
    union = _union_frame(base, frames)
    minute = ensure_minute_cache(symbol, union, include_shorts=False)

    funding_start = base.index.min().isoformat()
    funding_end = (base.index.max() + pd.Timedelta(TIMEFRAME)).isoformat()
    effective_rebuild = rebuild_funding or _legacy_funding_needs_snapshot(symbol)
    if effective_rebuild and not rebuild_funding:
        print("Funding legacy detectado: V10.2 reconstruira UNA VEZ el snapshot determinista.")
    funding = ensure_funding_cache(symbol, funding_start, funding_end, rebuild=effective_rebuild)
    snapshot = funding_snapshot_info(symbol)
    print(
        "Funding snapshot: "
        f"rows={snapshot['rows']} first={snapshot['first']} last={snapshot['last']} "
        f"sha256={snapshot['values_sha256'][:16]}..."
    )

    tick = fetch_market_tick_size(symbol)
    if tick <= 0:
        tick = max(float(base["close"].median()) * 1e-9, 1e-12)
    arrays = build_path_arrays(base)
    funding_series = build_funding_series(funding, base, arrays)
    full_start, full_end = _common_bounds(base)
    development_start = pd.Timestamp(V10_2_DEVELOPMENT_START_YEAR, 1, 1, tz="UTC")
    print(f"FASE market data/senales/funding: {_elapsed(phase)}")

    phase = time.perf_counter()
    primary_frames: list[pd.DataFrame] = []
    for name, frame in frames.items():
        opps = _generate(
            symbol=symbol,
            name=name,
            signals=frame,
            minute=minute,
            funding=funding,
            tick=tick,
            slip=BACKTEST_PRIMARY_SLIPPAGE_BPS,
            path_arrays=arrays,
            funding_series=funding_series,
        )
        primary_frames.append(opps)
        print(f"  {name}: {len(opps)} trades-path")
    opportunities = pd.concat(primary_frames, ignore_index=True) if primary_frames else pd.DataFrame()
    if opportunities.empty:
        raise RuntimeError("V10.2 no genero oportunidades")
    Path(V10_2_OPPORTUNITIES_FILE).parent.mkdir(parents=True, exist_ok=True)
    opportunities.to_csv(V10_2_OPPORTUNITIES_FILE, index=False)
    print(f"FASE oportunidades @ {BACKTEST_PRIMARY_SLIPPAGE_BPS:g}bp: {_elapsed(phase)} ({len(opportunities)})")

    phase = time.perf_counter()
    names = list(frames)
    summary = _summary(opportunities, names, full_start, development_start, full_end)
    annual = _annual(opportunities, names, development_start, full_end)
    summary.to_csv(V10_2_SUMMARY_FILE, index=False)
    annual.to_csv(V10_2_ANNUAL_FILE, index=False)
    _print_summary(summary)
    print(f"FASE comparacion: {_elapsed(phase)}")

    phase = time.perf_counter()
    primary_source = opportunities[opportunities["signal_name"].eq(PRIMARY)]
    sizing = _sizing_grid(primary_source, development_start, full_end)
    sizing.to_csv(V10_2_SIZING_FILE, index=False)
    _print_sizing(sizing)
    robust_sizing = sizing[sizing["within_robustness"]].head(1)
    print(f"FASE sizing: {_elapsed(phase)}")

    phase = time.perf_counter()
    sensitivity_rows: list[dict] = []
    for slip in BACKTEST_MARKET_SLIPPAGE_BPS:
        for name in (CONTROL, PRIMARY):
            if np.isclose(float(slip), BACKTEST_PRIMARY_SLIPPAGE_BPS):
                source = opportunities[opportunities["signal_name"].eq(name)]
            else:
                source = _generate(
                    symbol=symbol,
                    name=name,
                    signals=frames[name],
                    minute=minute,
                    funding=funding,
                    tick=tick,
                    slip=float(slip),
                    path_arrays=arrays,
                    funding_series=funding_series,
                )
            m, _ = _metrics(source, BASE_CONFIG, development_start, full_end)
            m.update({"signal_name": name, "slippage_bps": float(slip)})
            sensitivity_rows.append(m)
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(V10_2_SENSITIVITY_FILE, index=False)
    print(f"FASE sensibilidad: {_elapsed(phase)}")

    phase = time.perf_counter()
    stress_rows: list[dict] = []
    for name in (CONTROL, PRIMARY):
        source = opportunities[opportunities["signal_name"].eq(name)]
        for fee_mult, funding_mult in itertools.product(
            BACKTEST_FEE_STRESS_MULTIPLIERS, BACKTEST_FUNDING_STRESS_MULTIPLIERS
        ):
            stressed = _apply_economic_stress(source, float(fee_mult), float(funding_mult))
            m, _ = _metrics(stressed, BASE_CONFIG, development_start, full_end)
            m.update(
                {
                    "signal_name": name,
                    "fee_multiplier": float(fee_mult),
                    "funding_multiplier": float(funding_mult),
                }
            )
            stress_rows.append(m)
    stress = pd.DataFrame(stress_rows)
    stress.to_csv(V10_2_STRESS_FILE, index=False)
    print(f"FASE stress: {_elapsed(phase)}")

    control_row = summary[summary["signal_name"].eq(CONTROL)].iloc[0].to_dict()
    primary_row = summary[summary["signal_name"].eq(PRIMARY)].iloc[0].to_dict()
    historical_checks = {
        "pf_better_than_control": bool(primary_row["development_pf"] > control_row["development_pf"]),
        "return_better_than_control": bool(primary_row["development_return"] > control_row["development_return"]),
        "dd_within_limit": bool(primary_row["development_max_dd"] >= -BACKTEST_DD_LIMIT),
        "trade_retention": (
            float(primary_row["development_trades"] / control_row["development_trades"])
            if control_row["development_trades"] else float("nan")
        ),
    }
    historical_candidate = all(
        historical_checks[k]
        for k in ("pf_better_than_control", "return_better_than_control", "dd_within_limit")
    )

    recommendation = {
        "method": "v10_2_zscore_confirmation_research",
        "symbol": symbol,
        "data_status": {
            "development_start_year": V10_2_DEVELOPMENT_START_YEAR,
            "fresh_holdout": False,
            "reason": "shock1h hypothesis was discovered after inspecting 2022+ results",
        },
        "frozen_execution": {
            "side_mode": "long_only",
            "exit_policy": "fixed_rr",
            "rr": 1.8,
            "atr_stop": 1.5,
            "base_risk_per_trade": 0.005,
            "base_max_position_pct": 0.30,
        },
        "confirmations": {
            "shock1h_threshold": V10_2_SHOCK1H_THRESHOLD,
            "shock3h_threshold": V10_2_SHOCK3H_THRESHOLD,
            "vwap48h_threshold": V10_2_VWAP48H_THRESHOLD,
            "taker_min_shock_atr": V10_2_TAKER_MIN_SHOCK_ATR,
            "logic": "same-candle AND with confirmed z-score re-entry",
        },
        "funding_snapshot": snapshot,
        "control": control_row,
        "primary_candidate": primary_row,
        "historical_checks": historical_checks,
        "historical_candidate_for_forward": bool(historical_candidate),
        "promotion_to_live": False,
        "promotion_reason": "requires fresh forward validation; 2022+ is no longer independent holdout",
        "best_historical_sizing_under_dd_limit": (
            robust_sizing.iloc[0].to_dict() if not robust_sizing.empty else None
        ),
        "live_signal_unchanged": True,
    }
    serialize_recommendation(V10_2_RECOMMENDATION_FILE, recommendation)

    print("\nDECISION V10.2")
    print(json.dumps({
        "historical_candidate_for_forward": bool(historical_candidate),
        "promotion_to_live": False,
        "reason": recommendation["promotion_reason"],
        "historical_checks": historical_checks,
        "best_historical_sizing_under_dd_limit": recommendation["best_historical_sizing_under_dd_limit"],
    }, indent=2, default=str))

    print("\nFicheros V10.2:")
    for path in (
        V10_2_SUMMARY_FILE,
        V10_2_ANNUAL_FILE,
        V10_2_OPPORTUNITIES_FILE,
        V10_2_SIZING_FILE,
    V10_2_SIZING_MIN_PF,
        V10_2_SENSITIVITY_FILE,
        V10_2_STRESS_FILE,
        V10_2_RECOMMENDATION_FILE,
    ):
        print(f"  {path}")
    print(f"\nTOTAL: {_elapsed(total)}")
    print("IMPORTANTE: V10.2 no modifica el trigger live; solo propone candidato para forward test.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10.2 confirmation research")
    parser.add_argument(
        "--rebuild-funding",
        action="store_true",
        help="Reconstruye deliberadamente el snapshot funding completo antes del backtest.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(rebuild_funding=args.rebuild_funding)
