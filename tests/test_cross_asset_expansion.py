import numpy as np
import pandas as pd

from backtest_v10_4 import _cadence, _qualify_portfolio, _select_forward_candidate
from src.research.signals import (
    apply_signal,
    build_frequency_expansion_signal,
    parse_signal_specs,
    prepare_base_features,
    prepare_frequency_expansion_base,
)


def _ohlcv(index, close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
        },
        index=index,
    )


def test_cross_asset_1h_matches_frozen_v10_2_candidate_logic():
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=500, freq="1h")
    values = 100 + np.sin(np.arange(len(idx)) / 8.0) * 2
    values[330] -= 12
    values[331] -= 4
    df = _ohlcv(idx, values)
    r4 = df.iloc[::4].copy()

    old_base = prepare_base_features(df, r4)
    z = apply_signal(old_base, parse_signal_specs(["zscore_20_2.2"])[0])
    shock = apply_signal(old_base, parse_signal_specs(["return_shock_1h_2.175"])[0])
    expected = z["entry_long"].fillna(False) & shock["entry_long"].fillna(False)

    base = prepare_frequency_expansion_base(df, df, r4, signal_timeframe="1h", atr_hours=14)
    actual = build_frequency_expansion_signal(
        base,
        signal_timeframe="1h",
        shock_threshold=2.175,
        reference_1h=df,
    )
    pd.testing.assert_series_equal(
        actual["entry_long"].astype(bool), expected.astype(bool), check_names=False
    )


def test_portfolio_gate_requires_pf_dd_frequency_and_positive_years():
    metrics = {"net_pf": 1.7, "max_dd": -0.04, "trades_per_year": 30.0}
    ok, reasons = _qualify_portfolio(metrics, positive_years=4)
    assert ok is True
    assert reasons == []

    metrics = {"net_pf": 1.4, "max_dd": -0.08, "trades_per_year": 45.0}
    ok, reasons = _qualify_portfolio(metrics, positive_years=2)
    assert ok is False
    assert set(reasons) == {"pf", "dd", "frequency_high", "positive_years"}


def test_first_passing_expansion_step_wins_not_best_later_pf():
    expansion = pd.DataFrame(
        [
            {"step": 0, "meets_frequency_objective": False, "net_pf": 1.9},
            {"step": 1, "meets_frequency_objective": True, "net_pf": 1.61},
            {"step": 2, "meets_frequency_objective": True, "net_pf": 1.95},
        ]
    )
    selected = _select_forward_candidate(expansion)
    assert selected is not None
    assert int(selected["step"]) == 1


def test_cadence_reports_spacing_between_entries():
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-03T00:00:00Z", "2026-01-13T00:00:00Z"],
                utc=True,
            )
        }
    )
    c = _cadence(trades)
    assert c["median_gap_days"] == 6.0
    assert c["max_gap_days"] == 10.0
