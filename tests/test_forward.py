from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import config.settings as settings
import main
from src.research.signals import (
    build_bidirectional_frequency_expansion_signal,
    prepare_frequency_expansion_base,
)
from src.strategy.mean_reversion import MeanReversionStrategy


def _ohlcv(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.ones(len(close)),
        },
        index=index,
    )


def _short_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    n = 1400
    index = pd.date_range("2023-01-01", periods=n, freq="1h", tz="UTC")
    x = np.arange(n, dtype=float)
    close = 200.0 - x * 0.03 + np.sin(x / 9.0) * 0.7
    for pos in (850, 1000, 1150, 1300):
        close[pos] += 8.0
        close[pos + 1] += 1.0
    frame = _ohlcv(index, close)
    regime = (
        frame.resample("4h", label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    return frame, regime


def test_live_short_matches_replay_exact_timestamps():
    frame, regime = _short_frames()
    base = prepare_frequency_expansion_base(
        frame,
        frame,
        regime,
        signal_timeframe="1h",
        atr_hours=14,
    )
    research = build_bidirectional_frequency_expansion_signal(
        base,
        signal_timeframe="1h",
        shock_threshold=2.175,
        shock_vol_window_hours=100,
        z_window_hours=20.0,
        z_threshold=2.2,
    )
    live = MeanReversionStrategy(
        window=20,
        z_threshold=2.2,
        trend_window=200,
        atr_window=14,
        entry_mode="confirmed",
        regime_mode="1h_sma200_4h_ema200",
        signal_timeframe="1h",
        regime_timeframe="4h",
        regime_ema_window=200,
        regime_slope_lookback=6,
        shock1h_threshold=2.175,
        return_shock_vol_window=100,
    ).calculate_signals(frame, regime_df=regime)

    expected = research["entry_short"].astype(bool)
    actual = live["entry_short"].astype(bool)
    assert expected.equals(actual)
    assert int(actual.sum()) >= 1


def test_signal_side_is_frozen_per_component():
    btc = type("Ctx", (), {"allowed_side": "long"})()
    dot = type("Ctx", (), {"allowed_side": "short"})()
    both = pd.Series({"entry_long": True, "entry_short": True})
    assert main._signal_side(btc, both) == "long"
    assert main._signal_side(dot, both) == "short"

    wrong_only = pd.Series({"entry_long": True, "entry_short": False})
    assert main._signal_side(dot, wrong_only) is None


def test_forward_sizing_is_frozen():
    assert settings.SIDE_BY_SYMBOL == {
        "BTC/USDT:USDT": "long",
        "BNB/USDT:USDT": "long",
        "DOT/USDT:USDT": "short",
        "ADA/USDT:USDT": "short",
    }
    assert settings.RISK_BY_SYMBOL == {
        "BTC/USDT:USDT": 0.0075,
        "BNB/USDT:USDT": 0.0050,
        "DOT/USDT:USDT": 0.0075,
        "ADA/USDT:USDT": 0.0050,
    }
    assert settings.MAX_POSITION_PCT == 0.50
    assert settings.MAX_PORTFOLIO_RISK_PCT == 0.015
    assert settings.MAX_CONCURRENT_POSITIONS == 3
    assert settings.ORDERFLOW_DECISION_SECONDS == 60
    assert settings.ORDERFLOW_IMBALANCE_THRESHOLD == 0.20
    assert settings.ORDERFLOW_ADVERSE_ACCEPTANCE_BPS == 5.0
    assert settings.ORDERFLOW_TOTAL_ENTRY_DEADLINE_SECONDS == 420


def test_forward_validator_requires_own_start(monkeypatch):
    monkeypatch.setattr(settings, "FORWARD_START_UTC", "")
    with pytest.raises(ValueError, match="V12_START_UTC"):
        settings.validate_forward_settings()


def test_forward_validator_requires_demo(monkeypatch):
    monkeypatch.setattr(settings, "FORWARD_START_UTC", "2026-10-20T00:00:00Z")
    monkeypatch.setattr(settings, "FORWARD_ONLY_DEMO", False)
    with pytest.raises(ValueError, match="FORWARD_ONLY_DEMO=True"):
        settings.validate_forward_settings()
