import math

import pandas as pd

from backtest import _apply_economic_stress, _robust_candidates
from src.research.v9 import EntryFill, ExitPolicy, simulate_trade_path


def test_v9_2_robust_selector_allows_near_flat_year_and_rejects_fragile_pf():
    summary = pd.DataFrame([
        {
            "config": "safe_low", "return": 0.06, "max_dd": -0.02,
            "net_pf": 1.30,
        },
        {
            "config": "candidate", "return": 0.187, "max_dd": -0.062,
            "net_pf": 1.29,
        },
        {
            "config": "too_fragile", "return": 0.203, "max_dd": -0.063,
            "net_pf": 1.24,
        },
    ])
    rows = []
    for year, pf, ret in [
        (2022, 1.05, 0.003), (2023, 1.19, 0.019), (2024, 1.12, 0.015),
        (2025, 0.987, -0.0005), (2026, 1.64, 0.017),
    ]:
        rows.append({"config": "candidate", "year": year, "net_pf": pf,
                     "return": ret, "net_pnl": ret * 10_000})
    for year, pf, ret in [
        (2022, 1.05, 0.001), (2023, 1.18, 0.006), (2024, 1.23, 0.009),
        (2025, 1.29, 0.003), (2026, 2.22, 0.008),
    ]:
        rows.append({"config": "safe_low", "year": year, "net_pf": pf,
                     "return": ret, "net_pnl": ret * 10_000})
    for year, pf, ret in [
        (2022, 1.04, 0.003), (2023, 1.03, 0.005), (2024, 1.09, 0.016),
        (2025, 0.85, -0.008), (2026, 1.71, 0.029),
    ]:
        rows.append({"config": "too_fragile", "year": year, "net_pf": pf,
                     "return": ret, "net_pnl": ret * 10_000})

    ranked = _robust_candidates(summary, pd.DataFrame(rows))

    assert list(ranked["config"]) == ["candidate", "safe_low"]
    candidate = ranked.iloc[0]
    assert candidate["positive_years"] == 4
    assert math.isclose(candidate["worst_year_return"], -0.0005)


def test_fee_and_funding_stress_recomputes_net_economics():
    source = pd.DataFrame([
        {
            "gross_pnl_per_unit": 10.0,
            "fee_per_unit": 2.0,
            "funding_pnl_per_unit": -1.0,
            "net_pnl_per_unit": 7.0,
        }
    ])
    stressed = _apply_economic_stress(
        source, fee_multiplier=1.2, funding_multiplier=2.0
    )
    row = stressed.iloc[0]
    assert math.isclose(row["fee_per_unit"], 2.4)
    assert math.isclose(row["funding_pnl_per_unit"], -2.0)
    assert math.isclose(row["net_pnl_per_unit"], 5.6)


def test_tp_cross_ticks_can_be_stressed_to_two_ticks():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=4, freq="1h")
    signals = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 101.0],
            "high": [100.2, 101.61, 101.69, 101.71],
            "low": [99.8, 99.8, 100.5, 100.5],
            "close": [100.0, 101.0, 101.6, 101.7],
            "exit_long_mean": False,
            "exit_short_mean": False,
        },
        index=idx,
    )
    minute_idx = pd.date_range(idx[1], periods=2, freq="1min")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1.0},
        index=minute_idx,
    )
    fill = EntryFill(
        timestamp=minute_idx[0], side="long", raw_price=100.0, fill_price=100.0,
        fee_rate=0.0002, slippage_per_unit=0.0, liquidity="maker",
        route="limit_maker", baseline_open=100.0,
    )
    trade = simulate_trade_path(
        symbol="BTC/USDT:USDT", side="long", signal_time=idx[0], entry_hour=idx[1],
        fill=fill, signal_atr=2 / 3, signals=signals, minute_hour=minute,
        funding=pd.DataFrame(), policy=ExitPolicy("fixed_rr", 1.5),
        market_slippage_bps=0.0, tick_size=0.1, tp_cross_ticks=2,
    )
    assert trade is not None
    # TP=101.5 and 2 ticks means high must reach >=101.7. It exits only at idx[3].
    assert trade.exit_time == idx[3]
    assert trade.exit_reason == "take_profit"
