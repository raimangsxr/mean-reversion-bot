import math

import pandas as pd

from src.research.v9 import (
    EntryCandidate,
    EntryFill,
    OpportunityBook,
    PortfolioConfig,
    build_funding_series,
    generate_opportunities,
    parse_exit_profile,
    simulate_portfolio,
)


def _opp_rows():
    return pd.DataFrame([
        {
            "symbol": "BTC/USDT:USDT", "side": "long", "exit_policy": "fixed_rr", "rr": 1.5,
            "entry_time": pd.Timestamp("2026-01-01T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-01T02:00:00Z"),
            "entry_price": 100.0, "stop_fraction": 0.01,
            "net_pnl_per_unit": 1.0, "funding_pnl_per_unit": 0.0,
            "fee_per_unit": 0.0, "slippage_per_unit": 0.0,
            "exit_reason": "take_profit", "mae_r": 0.2,
        },
        {
            "symbol": "BTC/USDT:USDT", "side": "short", "exit_policy": "fixed_rr", "rr": 1.5,
            "entry_time": pd.Timestamp("2026-01-02T00:00:00Z"),
            "exit_time": pd.Timestamp("2026-01-02T02:00:00Z"),
            "entry_price": 100.0, "stop_fraction": 0.01,
            "net_pnl_per_unit": -0.5, "funding_pnl_per_unit": 0.0,
            "fee_per_unit": 0.0, "slippage_per_unit": 0.0,
            "exit_reason": "stop_loss", "mae_r": 1.0,
        },
    ])


def test_opportunity_book_prefilter_matches_direct_portfolio():
    opp = _opp_rows()
    cfg = PortfolioConfig("fixed_rr", 1.5, 0.005, 0.50, "long_only")
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = pd.Timestamp("2026-12-31T23:59:00Z")
    direct = simulate_portfolio(opp, cfg, start=start, end=end)
    source = OpportunityBook(opp).for_config(cfg)
    prepared = simulate_portfolio(
        source, cfg, start=start, end=end, prefiltered_config=True
    )
    assert len(direct.trades) == len(prepared.trades) == 1
    assert math.isclose(direct.final_equity, prepared.final_equity, rel_tol=1e-12)
    assert math.isclose(direct.max_drawdown, prepared.max_drawdown, rel_tol=1e-12)


def test_generate_opportunities_keeps_overlapping_signals_for_portfolio_stage():
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=32, freq="1h")
    signals = pd.DataFrame(
        {
            "open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0,
            "atr": 1.0, "entry_long": False, "entry_short": False,
            "exit_long_mean": False, "exit_short_mean": False,
        },
        index=idx,
    )
    minute_idx = pd.date_range(idx[1], periods=120, freq="1min")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0, "volume": 1.0},
        index=minute_idx,
    )
    candidates = []
    for entry_hour in (idx[1], idx[2]):
        fill = EntryFill(
            timestamp=entry_hour, side="long", raw_price=100.0, fill_price=100.0,
            fee_rate=0.0002, slippage_per_unit=0.0, liquidity="maker",
            route="limit_maker", baseline_open=100.0,
        )
        candidates.append(
            EntryCandidate(
                signal_time=entry_hour - pd.Timedelta(hours=1), entry_hour=entry_hour,
                side="long", signal_atr=1.0, fill=fill,
            )
        )
    trades = generate_opportunities(
        symbol="BTC/USDT:USDT", signals=signals, minute_cache=minute,
        funding=pd.DataFrame(), exit_policy=parse_exit_profile("time_24h", 1.5),
        market_slippage_bps=2.0, tick_size=0.01, include_shorts=False,
        entry_candidates=candidates,
    )
    # Opportunity generation is sizing-independent; portfolio simulation later
    # decides which overlapping signal can actually be taken.
    assert len(trades) == 2


def test_funding_series_normalizes_microsecond_index_to_nanoseconds():
    # pandas 3.x parses ordinary datetime strings as datetime64[us]. FundingSeries
    # must still compare correctly with Timestamp.value, which is nanoseconds.
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="1h").as_unit("us")
    signals = pd.DataFrame(
        {
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "exit_long_mean": False, "exit_short_mean": False,
        },
        index=idx,
    )
    funding_idx = pd.DatetimeIndex(["2026-01-01T08:00:00Z"]).as_unit("us")
    funding = pd.DataFrame({"funding_rate": [0.001]}, index=funding_idx)
    series = build_funding_series(funding, signals)
    value = series.interval_sum(
        pd.Timestamp("2026-01-01T01:00:00Z"),
        pd.Timestamp("2026-01-01T09:00:00Z"),
    )
    assert math.isclose(value, 0.1, rel_tol=1e-12)
