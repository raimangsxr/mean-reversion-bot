import math

import pandas as pd

from src.research.engine import (
    EntryPolicy,
    ExitPolicy,
    PortfolioConfig,
    parse_exit_profile,
    resolve_entry,
    simulate_portfolio,
    simulate_trade_path,
)


def minute_frame(rows, start="2026-01-01T01:00:00Z"):
    idx = pd.date_range(start, periods=len(rows), freq="1min")
    return pd.DataFrame(rows, index=idx)


def signal_frame(rows, start="2026-01-01T00:00:00Z"):
    idx = pd.date_range(start, periods=len(rows), freq="1h")
    return pd.DataFrame(rows, index=idx)


def test_long_entry_uses_signal_close_and_t0_no_chase():
    rows = []
    for i in range(8):
        open_price = 100.0 if i == 0 else 100.04
        rows.append({"open": open_price, "high": 100.06, "low": 99.98, "close": open_price, "volume": 1})
    fill = resolve_entry(
        minute_frame(rows), side="long", signal_close=100.0,
        market_slippage_bps=2.0, policy=EntryPolicy(3.0, 7, 5.0),
    )
    assert fill is not None
    assert fill.route == "market_fallback_conditional"
    assert math.isclose(fill.raw_price, 100.04)


def test_short_entry_is_symmetric():
    rows = [
        {"open": 100.0, "high": 100.04, "low": 99.9, "close": 100.0, "volume": 1},
        {"open": 100.0, "high": 100.05, "low": 99.9, "close": 100.0, "volume": 1},
    ]
    fill = resolve_entry(
        minute_frame(rows), side="short", signal_close=100.0,
        market_slippage_bps=2.0, policy=EntryPolicy(3.0, 1, 5.0),
    )
    assert fill is not None
    assert fill.liquidity == "maker"
    assert math.isclose(fill.fill_price, 100.03)


def test_mean_profile_has_no_fixed_tp_but_hybrid_does():
    assert parse_exit_profile("mean", 1.5).use_fixed_tp is False
    assert parse_exit_profile("hybrid_rr_mean", 1.5).use_fixed_tp is True


def test_tp_maker_requires_cross_beyond_tick():
    # Entry long 100, stop distance=1 (ATR=2/3 because ATR multiplier=1.5), TP=101.5.
    minutes = minute_frame([
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1},
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1},
    ])
    fill = resolve_entry(
        minutes, side="long", signal_close=100.03,
        market_slippage_bps=0.0, policy=EntryPolicy(3.0, 1, 5.0),
    )
    assert fill is not None
    # Force known entry price for this path test.
    fill.fill_price = 100.0
    fill.raw_price = 100.0
    fill.timestamp = minutes.index[0]
    fill.liquidity = "maker"

    signals = signal_frame([
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0, "exit_long_mean": False, "exit_short_mean": False},
        {"open": 100.0, "high": 101.5, "low": 99.8, "close": 101.0, "exit_long_mean": False, "exit_short_mean": False},
        {"open": 101.0, "high": 101.61, "low": 100.5, "close": 101.6, "exit_long_mean": False, "exit_short_mean": False},
    ])
    trade = simulate_trade_path(
        symbol="BTC/USDT:USDT", side="long", signal_time=signals.index[0],
        entry_hour=signals.index[1], fill=fill, signal_atr=2/3,
        signals=signals, minute_hour=minutes, funding=pd.DataFrame(),
        policy=ExitPolicy("fixed_rr", 1.5), market_slippage_bps=0.0,
        tick_size=0.1,
    )
    assert trade is not None
    # high==TP in first hourly bar is NOT enough; next bar crosses TP+tick.
    assert trade.exit_time == signals.index[2]
    assert trade.exit_reason == "take_profit"


def test_portfolio_position_cap_reports_low_risk_utilization():
    opp = pd.DataFrame([
        {
            "symbol": "BTC/USDT:USDT", "side": "long", "exit_policy": "fixed_rr", "rr": 1.5,
            "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T01:00:00Z"),
            "entry_price": 100.0, "stop_fraction": 0.012,
            "net_pnl_per_unit": 1.0, "funding_pnl_per_unit": 0.0,
            "fee_per_unit": 0.0, "slippage_per_unit": 0.0,
            "exit_reason": "take_profit",
        }
    ])
    cfg = PortfolioConfig("fixed_rr", 1.5, 0.005, 0.10, "long_only")
    result = simulate_portfolio(
        opp, cfg,
        start=pd.Timestamp("2026-01-01T00:00:00Z"),
        end=pd.Timestamp("2026-12-31T23:59:00Z"),
    )
    trade = result.trades.iloc[0]
    assert trade["limiting_factor"] == "position_pct"
    assert 0.23 < trade["risk_utilization"] < 0.25


def test_walk_forward_period_excludes_trade_that_closes_after_end():
    opp = pd.DataFrame([
        {
            "symbol": "BTC/USDT:USDT", "side": "long", "exit_policy": "fixed_rr", "rr": 1.5,
            "entry_time": pd.Timestamp("2022-12-31T23:00:00Z"),
            "exit_time": pd.Timestamp("2023-01-01T02:00:00Z"),
            "entry_price": 100.0, "stop_fraction": 0.01,
            "net_pnl_per_unit": 2.0, "funding_pnl_per_unit": 0.0,
            "fee_per_unit": 0.0, "slippage_per_unit": 0.0,
            "exit_reason": "take_profit", "mae_r": 0.1,
        }
    ])
    cfg = PortfolioConfig("fixed_rr", 1.5, 0.005, 0.50, "long_only")
    result = simulate_portfolio(
        opp, cfg,
        start=pd.Timestamp("2022-01-01T00:00:00Z"),
        end=pd.Timestamp("2022-12-31T23:59:59Z"),
    )
    assert result.trades.empty
    assert result.final_equity == result.initial_capital


def test_portfolio_config_name_is_deterministic_for_research_grid():
    cfg = PortfolioConfig("fixed_rr", 1.8, 0.005, 0.30, "long_only")
    assert cfg.name == "fixed_rr|rr=1.8|risk=0.0050|maxpos=0.30|long_only"
