from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
import hashlib
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

from config.settings import (
    ATR_MULTIPLIER,
    ATR_WINDOW,
    BACKTEST_ANNUAL_FILE,
    BACKTEST_DD_LIMIT,
    BACKTEST_EXIT_PROFILES,
    BACKTEST_FEE_STRESS_MULTIPLIERS,
    BACKTEST_FUNDING_STRESS_MULTIPLIERS,
    BACKTEST_FIRST_TEST_YEAR,
    BACKTEST_MARKET_SLIPPAGE_BPS,
    BACKTEST_MAKER_FEE,
    BACKTEST_MAX_POSITION_PCT_GRID,
    BACKTEST_OPPORTUNITIES_FILE,
    BACKTEST_MAX_CONCURRENT_POSITIONS,
    BACKTEST_MAX_PORTFOLIO_RISK_PCT,
    BACKTEST_MIN_GLOBAL_PF,
    BACKTEST_MIN_POSITIVE_YEARS,
    BACKTEST_MIN_WORST_YEAR_PF,
    BACKTEST_MIN_WORST_YEAR_RETURN,
    BACKTEST_PARTIAL_FRACTION,
    BACKTEST_PORTFOLIO_FILE,
    BACKTEST_PRIMARY_SLIPPAGE_BPS,
    BACKTEST_REBUILD_OPPORTUNITIES,
    BACKTEST_RECOMMENDATION_FILE,
    BACKTEST_REUSE_OPPORTUNITIES,
    BACKTEST_RISK_PER_TRADE_GRID,
    BACKTEST_SENSITIVITY_FILE,
    BACKTEST_SENSITIVITY_TOP_N,
    BACKTEST_SIDE_MODES,
    BACKTEST_STRESS_FILE,
    BACKTEST_SUMMARY_FILE,
    BACKTEST_SYMBOLS,
    BACKTEST_SELECTION_SYMBOLS,
    BACKTEST_TAKER_FEE,
    BACKTEST_TP_CROSS_TICKS,
    BACKTEST_TP_CROSS_TICKS_GRID,
    BACKTEST_TP_RATIOS,
    BACKTEST_TRADES_FILE,
    BACKTEST_WALK_FORWARD_FILE,
    BACKTEST_WORKERS,
    ENTRY_FALLBACK_CAP_BPS,
    ENTRY_LIMIT_OFFSET_BPS,
    ENTRY_LIMIT_WAIT_SECONDS,
    INITIAL_CAPITAL,
    ENTRY_MODE,
    LOOKBACK_WINDOW,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    TIMEFRAME,
    TREND_WINDOW,
    Z_THRESHOLD,
    BACKTEST_REQUIRE_FUNDING,
    validate_settings,
)
from src.research.v9 import (
    OpportunityBook,
    PortfolioConfig,
    aggregate_walk_forward,
    anchored_walk_forward,
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
    rank_training_results,
    serialize_recommendation,
    simulate_portfolio,
)


OPPORTUNITY_CACHE_SCHEMA = 3


def _elapsed(start: float) -> str:
    seconds = time.perf_counter() - start
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60.0:.1f}m"


def _config_grid() -> list[PortfolioConfig]:
    configs: list[PortfolioConfig] = []
    for exit_profile, rr, risk, max_pos, side_mode in itertools.product(
        BACKTEST_EXIT_PROFILES,
        BACKTEST_TP_RATIOS,
        BACKTEST_RISK_PER_TRADE_GRID,
        BACKTEST_MAX_POSITION_PCT_GRID,
        BACKTEST_SIDE_MODES,
    ):
        # Mean puro no usa TP/RR: evitamos duplicados identicos en el grid.
        if exit_profile == "mean" and rr != BACKTEST_TP_RATIOS[0]:
            continue
        parse_exit_profile(exit_profile, rr)
        configs.append(
            PortfolioConfig(
                exit_policy=exit_profile,
                rr=rr,
                risk_per_trade=risk,
                max_position_pct=max_pos,
                side_mode=side_mode,
            )
        )
    return configs


def _path_specs(
    exit_profiles: tuple[str, ...] | list[str] | None = None,
    rrs: tuple[float, ...] | list[float] | None = None,
) -> list[tuple[str, float]]:
    profiles = exit_profiles or list(BACKTEST_EXIT_PROFILES)
    ratios = rrs or list(BACKTEST_TP_RATIOS)
    specs: list[tuple[str, float]] = []
    for profile, rr in itertools.product(profiles, ratios):
        if profile == "mean" and rr != BACKTEST_TP_RATIOS[0]:
            continue
        parse_exit_profile(profile, rr)
        specs.append((str(profile), float(rr)))
    return specs


def _build_market_data() -> dict[str, dict]:
    include_shorts = "long_short" in BACKTEST_SIDE_MODES
    data: dict[str, dict] = {}
    for symbol in BACKTEST_SYMBOLS:
        started = time.perf_counter()
        signals = build_signals(symbol)
        minute = ensure_minute_cache(symbol, signals, include_shorts)
        funding = ensure_funding_cache(
            symbol,
            signals.index.min().isoformat(),
            (signals.index.max() + pd.Timedelta(hours=1)).isoformat(),
        )
        tick = fetch_market_tick_size(symbol)
        if tick <= 0:
            print(f"AVISO {symbol}: tick size no disponible; usando epsilon relativo.")
            tick = max(float(signals["close"].median()) * 1e-9, 1e-12)
        data[symbol] = {
            "signals": signals,
            "minute": minute,
            "funding": funding,
            "tick": tick,
        }
        print(f"  {symbol}: market data listo en {_elapsed(started)}")
    return data


def _frame_signature(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"rows": 0, "first": None, "last": None}
    sig = {
        "rows": int(len(df)),
        "first": df.index.min().isoformat(),
        "last": df.index.max().isoformat(),
    }
    # El funding puede ser corregido/re-descargado manteniendo exactamente los
    # mismos timestamps. Incluimos los valores para no reutilizar oportunidades
    # calculadas con rates antiguos o con el bug de unidades temporales.
    if "funding_rate" in df.columns:
        values = pd.to_numeric(df["funding_rate"], errors="coerce").fillna(0.0)
        sig["values_sha256"] = hashlib.sha256(
            values.to_numpy(dtype=np.float64, copy=False).tobytes()
        ).hexdigest()
    return sig


def _opportunity_fingerprint(data: dict[str, dict]) -> tuple[str, dict]:
    payload = {
        "schema": OPPORTUNITY_CACHE_SCHEMA,
        "symbols": list(BACKTEST_SYMBOLS),
        "exit_profiles": list(BACKTEST_EXIT_PROFILES),
        "rr": list(BACKTEST_TP_RATIOS),
        "side_modes": list(BACKTEST_SIDE_MODES),
        "primary_slippage_bps": BACKTEST_PRIMARY_SLIPPAGE_BPS,
        "entry_limit_offset_bps": ENTRY_LIMIT_OFFSET_BPS,
        "entry_limit_wait_seconds": ENTRY_LIMIT_WAIT_SECONDS,
        "entry_fallback_cap_bps": ENTRY_FALLBACK_CAP_BPS,
        "atr_multiplier": ATR_MULTIPLIER,
        "atr_window": ATR_WINDOW,
        "lookback_window": LOOKBACK_WINDOW,
        "z_threshold": Z_THRESHOLD,
        "trend_window": TREND_WINDOW,
        "entry_mode": ENTRY_MODE,
        "timeframe": TIMEFRAME,
        "regime_mode": REGIME_MODE,
        "regime_timeframe": REGIME_TIMEFRAME,
        "regime_ema_window": REGIME_EMA_WINDOW,
        "regime_slope_lookback": REGIME_SLOPE_LOOKBACK,
        "maker_fee": BACKTEST_MAKER_FEE,
        "taker_fee": BACKTEST_TAKER_FEE,
        "tp_cross_ticks": BACKTEST_TP_CROSS_TICKS,
        "partial_fraction": BACKTEST_PARTIAL_FRACTION,
        "funding_required": BACKTEST_REQUIRE_FUNDING,
        "market": {},
    }
    for symbol, d in data.items():
        payload["market"][symbol] = {
            "signals": _frame_signature(d["signals"]),
            "minute": _frame_signature(d["minute"]),
            "funding": _frame_signature(d["funding"]),
            "tick": float(d["tick"]),
        }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest(), payload


def _opportunity_meta_path() -> Path:
    path = Path(BACKTEST_OPPORTUNITIES_FILE)
    return path.with_suffix(path.suffix + ".meta.json")


def _normalize_opportunities(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    for column in ("signal_time", "entry_time", "exit_time"):
        out[column] = pd.to_datetime(out[column], utc=True)
    out["rr"] = out["rr"].astype(float)
    out["slippage_bps"] = out["slippage_bps"].astype(float)
    return out.sort_values(["entry_time", "symbol", "exit_policy", "rr"], kind="stable").reset_index(drop=True)


def _load_primary_opportunity_cache(data: dict[str, dict]) -> pd.DataFrame | None:
    if not BACKTEST_REUSE_OPPORTUNITIES or BACKTEST_REBUILD_OPPORTUNITIES:
        return None
    path = Path(BACKTEST_OPPORTUNITIES_FILE)
    meta_path = _opportunity_meta_path()
    if not path.exists() or not meta_path.exists():
        return None
    expected, _ = _opportunity_fingerprint(data)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if meta.get("fingerprint") != expected:
        return None
    started = time.perf_counter()
    df = pd.read_csv(path)
    if df.empty:
        return None
    df = _normalize_opportunities(df)
    if not np.allclose(df["slippage_bps"].to_numpy(), BACKTEST_PRIMARY_SLIPPAGE_BPS):
        return None
    print(f"Reutilizando oportunidades primarias cacheadas: {len(df)} ({_elapsed(started)})")
    return df


def _save_primary_opportunity_cache(data: dict[str, dict], opportunities: pd.DataFrame) -> None:
    path = Path(BACKTEST_OPPORTUNITIES_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    opportunities.to_csv(path, index=False)
    fingerprint, payload = _opportunity_fingerprint(data)
    _opportunity_meta_path().write_text(
        json.dumps({"fingerprint": fingerprint, "inputs": payload}, indent=2),
        encoding="utf-8",
    )


def _effective_workers(symbol_count: int) -> int:
    if symbol_count <= 1:
        return 1
    if BACKTEST_WORKERS == 1:
        return 1
    if BACKTEST_WORKERS > 1:
        return max(1, min(BACKTEST_WORKERS, symbol_count))
    cpu = os.cpu_count() or 2
    return max(1, min(symbol_count, max(1, cpu - 1)))


def _generate_symbol_opportunities_worker(args: tuple) -> tuple[str, pd.DataFrame, int]:
    symbol, d, slip, specs, include_shorts, tp_cross_ticks = args
    signals = d["signals"]
    minute = d["minute"]
    funding = d["funding"]
    arrays = build_path_arrays(signals)
    funding_fast = build_funding_series(funding, signals, arrays)
    candidates = prepare_entry_candidates(
        signals=signals,
        minute_cache=minute,
        market_slippage_bps=float(slip),
        include_shorts=include_shorts,
    )
    frames: list[pd.DataFrame] = []
    for profile_name, rr in specs:
        policy = parse_exit_profile(profile_name, rr)
        trades = generate_opportunities(
            symbol=symbol,
            signals=signals,
            minute_cache=minute,
            funding=funding,
            exit_policy=policy,
            market_slippage_bps=float(slip),
            tick_size=float(d["tick"]),
            include_shorts=include_shorts,
            entry_candidates=candidates,
            path_arrays=arrays,
            funding_series=funding_fast,
            tp_cross_ticks=int(tp_cross_ticks),
        )
        if not trades.empty:
            trades.insert(0, "slippage_bps", float(slip))
            frames.append(trades)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return symbol, result, len(candidates)


def _generate_opportunities(
    data: dict[str, dict],
    *,
    slip: float,
    specs: list[tuple[str, float]],
    tp_cross_ticks: int = BACKTEST_TP_CROSS_TICKS,
) -> pd.DataFrame:
    include_shorts = "long_short" in BACKTEST_SIDE_MODES
    workers = _effective_workers(len(data))
    tasks = [
        (symbol, d, float(slip), specs, include_shorts, int(tp_cross_ticks))
        for symbol, d in data.items()
    ]
    started = time.perf_counter()
    frames: list[pd.DataFrame] = []
    if workers == 1:
        for task in tasks:
            symbol, frame, candidates = _generate_symbol_opportunities_worker(task)
            print(f"  {symbol}: {candidates} entradas resueltas -> {len(frame)} paths")
            if not frame.empty:
                frames.append(frame)
    else:
        print(f"  paths en paralelo: {workers} workers ({len(tasks)} simbolos)")
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_generate_symbol_opportunities_worker, task) for task in tasks]
            for future in as_completed(futures):
                symbol, frame, candidates = future.result()
                print(f"  {symbol}: {candidates} entradas resueltas -> {len(frame)} paths")
                if not frame.empty:
                    frames.append(frame)
    if not frames:
        return pd.DataFrame()
    result = _normalize_opportunities(pd.concat(frames, ignore_index=True))
    print(f"  paths completos slip={slip:g}bp: {len(result)} en {_elapsed(started)}")
    return result


def _get_primary_opportunities(data: dict[str, dict]) -> pd.DataFrame:
    cached = _load_primary_opportunity_cache(data)
    if cached is not None:
        return cached
    print("\nGenerando oportunidades SOLO al slippage primario (seleccion V9)...")
    result = _generate_opportunities(
        data,
        slip=BACKTEST_PRIMARY_SLIPPAGE_BPS,
        specs=_path_specs(),
    )
    if result.empty:
        return result
    _save_primary_opportunity_cache(data, result)
    print(f"Oportunidades primarias guardadas: {BACKTEST_OPPORTUNITIES_FILE}")
    return result


def _full_summary(opps: pd.DataFrame, configs: list[PortfolioConfig]) -> pd.DataFrame:
    book = OpportunityBook(opps)
    start = book.frame["entry_time"].min()
    end = book.frame["exit_time"].max()
    rows: list[dict] = []
    for cfg in configs:
        source = book.for_config(cfg)
        result = simulate_portfolio(
            source, cfg, start=start, end=end, prefiltered_config=True
        )
        row = portfolio_metrics(result)
        row["slippage_bps"] = BACKTEST_PRIMARY_SLIPPAGE_BPS
        rows.append(row)
    return pd.DataFrame(rows)


def _annual_table(opps: pd.DataFrame, configs: list[PortfolioConfig]) -> pd.DataFrame:
    book = OpportunityBook(opps)
    years = sorted(set(book.frame["entry_time"].dt.year))
    years = [y for y in years if y >= BACKTEST_FIRST_TEST_YEAR]
    rows: list[dict] = []
    for cfg in configs:
        source = book.for_config(cfg)
        for year in years:
            start = pd.Timestamp(year=year, month=1, day=1, tz="UTC")
            end = pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, tz="UTC")
            result = simulate_portfolio(
                source, cfg, start=start, end=end, prefiltered_config=True
            )
            row = portfolio_metrics(result)
            row["year"] = year
            row["slippage_bps"] = BACKTEST_PRIMARY_SLIPPAGE_BPS
            rows.append(row)
    return pd.DataFrame(rows)


def _robust_candidates(summary: pd.DataFrame, annual: pd.DataFrame) -> pd.DataFrame:
    """Select static V9.2 candidates using continuous robustness constraints.

    We deliberately do NOT require every calendar year to be positive. A year
    that is effectively flat must not be treated like a severe failure. The
    candidate still needs bounded DD, enough positive years, a tolerable worst
    year and a minimum global PF.
    """
    if summary.empty or annual.empty:
        return pd.DataFrame()
    grouped = annual.groupby("config", as_index=False).agg(
        positive_years=("net_pnl", lambda x: int((x > 0).sum())),
        years=("year", "nunique"),
        worst_year_pf=("net_pf", "min"),
        median_year_pf=("net_pf", "median"),
        worst_year_return=("return", "min"),
        median_year_return=("return", "median"),
    )
    table = summary.merge(grouped, on="config", how="left")
    table["required_positive_years"] = np.minimum(
        table["years"].astype(int), BACKTEST_MIN_POSITIVE_YEARS
    )
    eligible = table[
        (table["max_dd"] >= -BACKTEST_DD_LIMIT)
        & (table["positive_years"] >= table["required_positive_years"])
        & (table["worst_year_pf"] >= BACKTEST_MIN_WORST_YEAR_PF)
        & (table["worst_year_return"] >= BACKTEST_MIN_WORST_YEAR_RETURN)
        & (table["net_pf"] >= BACKTEST_MIN_GLOBAL_PF)
        & np.isfinite(table["net_pf"])
    ].copy()
    if eligible.empty:
        return eligible
    return eligible.sort_values(
        ["return", "worst_year_pf", "net_pf"],
        ascending=[False, False, False],
    )


def _btc_cross_sectional(opps: pd.DataFrame, configs: list[PortfolioConfig]) -> pd.DataFrame:
    if not BACKTEST_SYMBOLS:
        return pd.DataFrame()
    btc = BACKTEST_SYMBOLS[0]
    book = OpportunityBook(opps)
    btc_all = book.frame[book.frame["symbol"].eq(btc)]
    if btc_all.empty:
        return pd.DataFrame()
    first = int(btc_all["entry_time"].min().year)
    last = int(btc_all["entry_time"].max().year)
    by_name = {c.name: c for c in configs}
    rows: list[dict] = []
    for year in range(max(BACKTEST_FIRST_TEST_YEAR, first + 1), last + 1):
        train_start = pd.Timestamp(first, 1, 1, tz="UTC")
        train_end = pd.Timestamp(year - 1, 12, 31, 23, 59, tz="UTC")
        train_rows = []
        for cfg in configs:
            source = book.for_config(cfg, [btc])
            m = portfolio_metrics(
                simulate_portfolio(
                    source, cfg, start=train_start, end=train_end,
                    prefiltered_config=True,
                )
            )
            train_rows.append(m)
        ranked = rank_training_results(pd.DataFrame(train_rows))
        if ranked.empty:
            continue
        selected_name = str(ranked.iloc[0]["config"])
        selected = by_name[selected_name]
        test_start = pd.Timestamp(year, 1, 1, tz="UTC")
        test_end = pd.Timestamp(year, 12, 31, 23, 59, tz="UTC")
        for symbol in BACKTEST_SYMBOLS:
            source = book.for_config(selected, [symbol])
            m = portfolio_metrics(
                simulate_portfolio(
                    source, selected, start=test_start, end=test_end,
                    prefiltered_config=True,
                )
            )
            m.update({"year": year, "symbol": symbol, "selected_on": btc})
            rows.append(m)
    return pd.DataFrame(rows)


def _symbol_robustness(opps: pd.DataFrame, cfg: PortfolioConfig) -> pd.DataFrame:
    book = OpportunityBook(opps)
    years = sorted(set(book.frame["entry_time"].dt.year))
    years = [y for y in years if y >= BACKTEST_FIRST_TEST_YEAR]
    rows = []
    for symbol in BACKTEST_SYMBOLS:
        symbol_source = book.for_config(cfg, [symbol])
        yearly_returns: list[float] = []
        yearly_pfs: list[float] = []
        positive = 0
        for year in years:
            start = pd.Timestamp(year, 1, 1, tz="UTC")
            end = pd.Timestamp(year, 12, 31, 23, 59, tz="UTC")
            m = portfolio_metrics(
                simulate_portfolio(
                    symbol_source, cfg, start=start, end=end,
                    prefiltered_config=True,
                )
            )
            yearly_returns.append(float(m["return"]))
            yearly_pfs.append(float(m["net_pf"]))
            positive += int(float(m["net_pnl"]) > 0)
        if not symbol_source.empty:
            full_start = symbol_source["entry_time"].min()
            full_end = symbol_source["exit_time"].max()
            full = portfolio_metrics(
                simulate_portfolio(
                    symbol_source, cfg, start=full_start, end=full_end,
                    prefiltered_config=True,
                )
            )
        else:
            full = {"return": 0.0, "net_pf": float("nan"), "trades": 0}
        rows.append({
            "symbol": symbol,
            "positive_years": positive,
            "years": len(years),
            "worst_year_return": min(yearly_returns) if yearly_returns else float("nan"),
            "worst_year_pf": min(yearly_pfs) if yearly_pfs else float("nan"),
            "full_return": float(full["return"]),
            "full_pf": float(full["net_pf"]),
            "trades": int(full["trades"]),
        })
    return pd.DataFrame(rows)


def _sensitivity_configs(
    robust: pd.DataFrame, configs: list[PortfolioConfig]
) -> list[PortfolioConfig]:
    by_name = {c.name: c for c in configs}
    names: list[str] = []
    if not robust.empty:
        names.extend(robust.head(BACKTEST_SENSITIVITY_TOP_N)["config"].astype(str).tolist())
    control = PortfolioConfig(
        exit_policy="fixed_rr", rr=1.5, risk_per_trade=0.005,
        max_position_pct=0.10, side_mode="long_only",
    )
    if control.name in by_name:
        names.append(control.name)
    unique: list[PortfolioConfig] = []
    seen: set[str] = set()
    for name in names:
        if name not in seen and name in by_name:
            seen.add(name)
            unique.append(by_name[name])
    return unique


def _slippage_sensitivity(
    data: dict[str, dict],
    primary: pd.DataFrame,
    robust: pd.DataFrame,
    configs: list[PortfolioConfig],
) -> pd.DataFrame:
    finalists = _sensitivity_configs(robust, configs)
    if not finalists:
        return pd.DataFrame()
    path_specs = sorted({(c.exit_policy, float(c.rr)) for c in finalists})
    frames: dict[float, pd.DataFrame] = {float(BACKTEST_PRIMARY_SLIPPAGE_BPS): primary}
    for slip in BACKTEST_MARKET_SLIPPAGE_BPS:
        slip = float(slip)
        if np.isclose(slip, BACKTEST_PRIMARY_SLIPPAGE_BPS):
            continue
        print(f"\nSensibilidad {slip:g}bp: solo {len(path_specs)} paths finalistas")
        frames[slip] = _generate_opportunities(data, slip=slip, specs=path_specs)

    rows: list[dict] = []
    for slip in sorted(frames):
        frame = frames[slip]
        if frame.empty:
            continue
        book = OpportunityBook(frame)
        start = book.frame["entry_time"].min()
        end = book.frame["exit_time"].max()
        for cfg in finalists:
            source = book.for_config(cfg)
            result = simulate_portfolio(
                source, cfg, start=start, end=end, prefiltered_config=True
            )
            row = portfolio_metrics(result)
            row["slippage_bps"] = slip
            rows.append(row)
    return pd.DataFrame(rows)



def _apply_economic_stress(
    source: pd.DataFrame, *, fee_multiplier: float, funding_multiplier: float
) -> pd.DataFrame:
    stressed = source.copy()
    stressed["fee_per_unit"] = (
        stressed["fee_per_unit"].astype(float) * float(fee_multiplier)
    )
    stressed["funding_pnl_per_unit"] = (
        stressed["funding_pnl_per_unit"].astype(float) * float(funding_multiplier)
    )
    stressed["net_pnl_per_unit"] = (
        stressed["gross_pnl_per_unit"].astype(float)
        - stressed["fee_per_unit"].astype(float)
        + stressed["funding_pnl_per_unit"].astype(float)
    )
    return stressed


def _stress_candidate(
    data: dict[str, dict],
    primary: pd.DataFrame,
    candidate: PortfolioConfig,
) -> pd.DataFrame:
    """Stress only the frozen candidate; never reopen the full optimization grid."""
    path_spec = [(candidate.exit_policy, float(candidate.rr))]
    frames_by_ticks: dict[int, pd.DataFrame] = {}
    for ticks in BACKTEST_TP_CROSS_TICKS_GRID:
        ticks = int(ticks)
        if ticks == BACKTEST_TP_CROSS_TICKS:
            frames_by_ticks[ticks] = primary
        else:
            print(f"\nStress TP maker: regenerando candidato con cross={ticks} ticks")
            frames_by_ticks[ticks] = _generate_opportunities(
                data,
                slip=BACKTEST_PRIMARY_SLIPPAGE_BPS,
                specs=path_spec,
                tp_cross_ticks=ticks,
            )

    rows: list[dict] = []
    for ticks, frame in sorted(frames_by_ticks.items()):
        if frame.empty:
            continue
        book = OpportunityBook(frame)
        base_source = book.for_config(candidate)
        if base_source.empty:
            continue
        start = base_source["entry_time"].min()
        end = base_source["exit_time"].max()
        for fee_mult, funding_mult in itertools.product(
            BACKTEST_FEE_STRESS_MULTIPLIERS,
            BACKTEST_FUNDING_STRESS_MULTIPLIERS,
        ):
            source = _apply_economic_stress(
                base_source,
                fee_multiplier=float(fee_mult),
                funding_multiplier=float(funding_mult),
            )
            result = simulate_portfolio(
                source, candidate, start=start, end=end, prefiltered_config=True
            )
            row = portfolio_metrics(result)
            row.update({
                "slippage_bps": float(BACKTEST_PRIMARY_SLIPPAGE_BPS),
                "tp_cross_ticks": ticks,
                "fee_multiplier": float(fee_mult),
                "funding_multiplier": float(funding_mult),
            })
            rows.append(row)
    return pd.DataFrame(rows)

def _print_top(title: str, df: pd.DataFrame, n: int = 12) -> None:
    print("\n" + title)
    if df.empty:
        print("(sin resultados)")
        return
    columns = [
        "exit_policy", "rr", "risk_per_trade", "max_position_pct", "side_mode",
        "trades", "net_pf", "return", "max_dd", "median_risk_utilization",
    ]
    extra = [c for c in ["positive_years", "years", "worst_year_pf", "worst_year_return"] if c in df.columns]
    print(df[columns + extra].head(n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def _candidate_live_env(best: dict) -> dict[str, str]:
    profile = str(best["exit_policy"])
    env = {
        "RISK_REWARD_RATIO": str(float(best["rr"])),
        "RISK_PER_TRADE": str(float(best["risk_per_trade"])),
        "MAX_POSITION_PCT": str(float(best["max_position_pct"])),
        "ENABLE_LONGS": "True",
        "ENABLE_SHORTS": "True" if best["side_mode"] == "long_short" else "False",
        "MAX_PORTFOLIO_RISK_PCT": str(BACKTEST_MAX_PORTFOLIO_RISK_PCT),
        "LIVE_MAX_CONCURRENT_POSITIONS": str(BACKTEST_MAX_CONCURRENT_POSITIONS),
        "EXIT_MODE": "fixed_rr",
        "BREAK_EVEN_TRIGGER_R": "0.0",
        "PARTIAL_TP_TRIGGER_R": "0.0",
        "PARTIAL_TP_FRACTION": str(0.50),
        "PARTIAL_TP_MOVE_STOP_TO_BE": "True",
    }
    if profile == "mean":
        env["EXIT_MODE"] = "mean"
    elif profile == "hybrid_rr_mean":
        env["EXIT_MODE"] = "hybrid_rr_mean"
    elif profile.startswith("time_"):
        hours = profile.removeprefix("time_").removesuffix("h")
        env["EXIT_MODE"] = "fixed_rr_time"
        env["MAX_HOLDING_HOURS"] = hours
    elif profile.startswith("be_"):
        env["BREAK_EVEN_TRIGGER_R"] = profile.removeprefix("be_").removesuffix("r")
    elif profile.startswith("partial_"):
        trigger = profile.removeprefix("partial_").removesuffix("r_be")
        env["PARTIAL_TP_TRIGGER_R"] = trigger
    elif profile == "hybrid_rr_mean_time24":
        env["EXIT_MODE"] = "hybrid_rr_mean_time"
        env["MAX_HOLDING_HOURS"] = "24"
    return env


def run() -> None:
    total_started = time.perf_counter()
    validate_settings()
    configs = _config_grid()
    print("V9.2 RESEARCH | BTC static candidate + robustness/stress")
    print(f"Entrada: LIMIT -{ENTRY_LIMIT_OFFSET_BPS:.1f}bp | wait={ENTRY_LIMIT_WAIT_SECONDS//60}m | cap=+/-{ENTRY_FALLBACK_CAP_BPS:.1f}bp")
    print(f"Regimen: {REGIME_MODE} | universo={BACKTEST_SYMBOLS}")
    print(f"Seleccion candidato: simbolos={BACKTEST_SELECTION_SYMBOLS}")
    print(f"Exit profiles={BACKTEST_EXIT_PROFILES}")
    print(f"RR={BACKTEST_TP_RATIOS} | max-position={BACKTEST_MAX_POSITION_PCT_GRID}")
    print(f"side modes={BACKTEST_SIDE_MODES} | DD limit={BACKTEST_DD_LIMIT:.1%}")
    print(
        "Robustez: "
        f">={BACKTEST_MIN_POSITIVE_YEARS} anos + | "
        f"worstPF>={BACKTEST_MIN_WORST_YEAR_PF:.2f} | "
        f"worstRet>={BACKTEST_MIN_WORST_YEAR_RETURN:.1%} | "
        f"globalPF>={BACKTEST_MIN_GLOBAL_PF:.2f}"
    )
    print(f"slippage primary={BACKTEST_PRIMARY_SLIPPAGE_BPS:.1f}bp | sensitivity={BACKTEST_MARKET_SLIPPAGE_BPS}")
    print(f"workers={_effective_workers(len(BACKTEST_SYMBOLS))} | reuse opportunities={BACKTEST_REUSE_OPPORTUNITIES and not BACKTEST_REBUILD_OPPORTUNITIES}")
    print("Optimizacion: entries se resuelven una vez, paths usan NumPy, portfolio preindexado.")

    phase = time.perf_counter()
    data = _build_market_data()
    print(f"FASE market data: {_elapsed(phase)}")

    phase = time.perf_counter()
    opportunities = _get_primary_opportunities(data)
    if opportunities.empty:
        raise RuntimeError("V9 no genero oportunidades; revisa datos/cache/configuracion.")
    print(f"FASE opportunities primarias: {_elapsed(phase)} ({len(opportunities)})")

    # La optimizacion del candidato se hace solo sobre los simbolos de seleccion.
    # El resto del universo se conserva para validacion cross-sectional sin retune.
    selection_opportunities = opportunities[
        opportunities["symbol"].isin(BACKTEST_SELECTION_SYMBOLS)
    ].copy()
    if selection_opportunities.empty:
        raise RuntimeError(
            "No hay oportunidades para BACKTEST_SELECTION_SYMBOLS="
            f"{BACKTEST_SELECTION_SYMBOLS}."
        )

    phase = time.perf_counter()
    summary = _full_summary(selection_opportunities, configs)
    Path(BACKTEST_SUMMARY_FILE).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(BACKTEST_SUMMARY_FILE, index=False)
    print(f"FASE grid portfolio: {_elapsed(phase)} ({len(configs)} configs)")

    phase = time.perf_counter()
    annual = _annual_table(selection_opportunities, configs)
    Path(BACKTEST_ANNUAL_FILE).parent.mkdir(parents=True, exist_ok=True)
    annual.to_csv(BACKTEST_ANNUAL_FILE, index=False)
    robust = _robust_candidates(summary, annual)
    _print_top("TOP ROBUSTOS V9.2 @ slippage primario", robust)
    print(f"FASE robustez anual: {_elapsed(phase)}")

    phase = time.perf_counter()
    wf, wf_trades = anchored_walk_forward(selection_opportunities, configs)
    Path(BACKTEST_WALK_FORWARD_FILE).parent.mkdir(parents=True, exist_ok=True)
    wf.to_csv(BACKTEST_WALK_FORWARD_FILE, index=False)
    if not wf_trades.empty:
        wf_trades.to_csv(
            Path(BACKTEST_WALK_FORWARD_FILE).with_name("backtest_v9_walk_forward_trades.csv"),
            index=False,
        )
    wf_agg = aggregate_walk_forward(wf)
    print("\nANCHORED WALK-FORWARD DINAMICO (DIAGNOSTICO; NO SE USA PARA RECOMENDAR)")
    if wf.empty:
        print("(sin folds)")
    else:
        print(wf[["test_year", "selected_config", "trades", "net_pf", "return", "max_dd", "train_return", "train_max_dd"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("Aggregate:", json.dumps(wf_agg, indent=2))
    print(f"FASE walk-forward: {_elapsed(phase)}")

    phase = time.perf_counter()
    if len(BACKTEST_SYMBOLS) > len(BACKTEST_SELECTION_SYMBOLS):
        cross = _btc_cross_sectional(opportunities, configs)
        if not cross.empty:
            print("\nCROSS-SECTIONAL experimental: BTC train -> otros simbolos sin retune")
            pivot = cross.pivot_table(index="year", columns="symbol", values="return", aggfunc="first")
            print(pivot.to_string(float_format=lambda x: f"{x:7.2%}"))
    else:
        cross = pd.DataFrame()
        print("\nCROSS-SECTIONAL desactivado: V9.2 principal es BTC-only.")
    Path(BACKTEST_PORTFOLIO_FILE).parent.mkdir(parents=True, exist_ok=True)
    cross.to_csv(BACKTEST_PORTFOLIO_FILE, index=False)
    print(f"FASE cross-sectional: {_elapsed(phase)}")

    phase = time.perf_counter()
    selection_data = {s: data[s] for s in BACKTEST_SELECTION_SYMBOLS}
    sensitivity = _slippage_sensitivity(
        selection_data, selection_opportunities, robust, configs
    )
    Path(BACKTEST_SENSITIVITY_FILE).parent.mkdir(parents=True, exist_ok=True)
    sensitivity.to_csv(BACKTEST_SENSITIVITY_FILE, index=False)
    print(f"FASE sensibilidad finalistas: {_elapsed(phase)}")

    recommendation: dict = {
        "method": "v9_2_static_robust_candidate_with_stress",
        "dd_limit": BACKTEST_DD_LIMIT,
        "robustness_constraints": {
            "min_positive_years": BACKTEST_MIN_POSITIVE_YEARS,
            "min_worst_year_pf": BACKTEST_MIN_WORST_YEAR_PF,
            "min_worst_year_return": BACKTEST_MIN_WORST_YEAR_RETURN,
            "min_global_pf": BACKTEST_MIN_GLOBAL_PF,
        },
        "primary_slippage_bps": BACKTEST_PRIMARY_SLIPPAGE_BPS,
        "selection_symbols": list(BACKTEST_SELECTION_SYMBOLS),
        "dynamic_retuning_walk_forward_diagnostic": wf_agg,
        "performance_pipeline": {
            "selection_slippage": BACKTEST_PRIMARY_SLIPPAGE_BPS,
            "sensitivity_top_n": BACKTEST_SENSITIVITY_TOP_N,
            "workers": _effective_workers(len(BACKTEST_SYMBOLS)),
            "opportunity_cache_schema": OPPORTUNITY_CACHE_SCHEMA,
        },
    }
    if not robust.empty:
        best = robust.iloc[0].to_dict()
        recommendation["best_robust_config"] = best
        best_cfg = next(c for c in configs if c.name == best["config"])
        symbol_robust = _symbol_robustness(opportunities, best_cfg)
        # No se promueven simbolos por el mismo grid con el que se optimiza.
        # ETH/SOL son validacion cross-sectional; solo BACKTEST_SELECTION_SYMBOLS
        # participan en la seleccion del candidato para evitar dilucion/overfit.
        final_symbols = list(BACKTEST_SELECTION_SYMBOLS)
        live_env = _candidate_live_env(best)
        live_env["LIVE_SYMBOLS"] = ",".join(final_symbols)
        recommendation["symbol_robustness"] = symbol_robust.to_dict(orient="records")

        book = OpportunityBook(opportunities)
        best_source = book.for_config(best_cfg, final_symbols)
        best_start = best_source["entry_time"].min()
        best_end = best_source["exit_time"].max()
        best_result = simulate_portfolio(
            best_source, best_cfg, start=best_start, end=best_end,
            prefiltered_config=True,
        )
        best_metrics = portfolio_metrics(best_result)
        Path(BACKTEST_TRADES_FILE).parent.mkdir(parents=True, exist_ok=True)
        best_result.trades.to_csv(BACKTEST_TRADES_FILE, index=False)

        stress = _stress_candidate(selection_data, selection_opportunities, best_cfg)
        Path(BACKTEST_STRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
        stress.to_csv(BACKTEST_STRESS_FILE, index=False)
        if not stress.empty:
            print("\nSTRESS DEL CANDIDATO V9.2")
            stress_cols = [
                "tp_cross_ticks", "fee_multiplier", "funding_multiplier",
                "net_pf", "return", "max_dd",
            ]
            print(stress[stress_cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        annual_candidate = annual[annual["config"].eq(best_cfg.name)].copy()
        recommendation["candidate_annual"] = annual_candidate.to_dict(orient="records")
        recommendation["candidate_stress"] = stress.to_dict(orient="records")
        recommendation["candidate_for_demo"] = {
            "exit_policy": best["exit_policy"],
            "rr": float(best["rr"]),
            "risk_per_trade": float(best["risk_per_trade"]),
            "max_position_pct": float(best["max_position_pct"]),
            "side_mode": best["side_mode"],
            "suggested_symbols": final_symbols,
            "recomputed_metrics": best_metrics,
            "live_env": live_env,
        }
    else:
        recommendation["candidate_for_demo"] = None
        Path(BACKTEST_STRESS_FILE).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(BACKTEST_STRESS_FILE, index=False)

    control_cfg = PortfolioConfig(
        exit_policy="fixed_rr", rr=1.5, risk_per_trade=0.005,
        max_position_pct=0.10, side_mode="long_only",
    )
    book = OpportunityBook(opportunities)
    control_source = book.for_config(control_cfg, BACKTEST_SELECTION_SYMBOLS)
    if not control_source.empty:
        control_start = control_source["entry_time"].min()
        control_end = control_source["exit_time"].max()
        recommendation["current_live_control"] = portfolio_metrics(
            simulate_portfolio(
                control_source, control_cfg, start=control_start, end=control_end,
                prefiltered_config=True,
            )
        )

    serialize_recommendation(BACKTEST_RECOMMENDATION_FILE, recommendation)

    print("\nFicheros V9:")
    for p in [
        BACKTEST_SUMMARY_FILE, BACKTEST_ANNUAL_FILE, BACKTEST_WALK_FORWARD_FILE,
        BACKTEST_OPPORTUNITIES_FILE, BACKTEST_TRADES_FILE, BACKTEST_PORTFOLIO_FILE,
        BACKTEST_SENSITIVITY_FILE, BACKTEST_STRESS_FILE, BACKTEST_RECOMMENDATION_FILE,
    ]:
        print(f"  {p}")
    print(f"\nTOTAL: {_elapsed(total_started)}")
    print("IMPORTANTE: V9.2 no retunea el live por ano; recommendation.json propone una configuracion estatica.")


if __name__ == "__main__":
    run()
