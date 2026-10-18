import numpy as np
import pandas as pd

from src.research.signals import apply_signal, parse_signal_specs


def _base(n=260):
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="1h")
    close = pd.Series(100.0 + np.sin(np.arange(n) / 10.0), index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000.0,
            "atr": 2.0,
            "regime_ma": 90.0,
            "regime_long": True,
            "regime_short": False,
            "exit_long_mean": False,
            "exit_short_mean": False,
        },
        index=idx,
    )


def test_v10_2_signal_specs_parse_without_duplicates():
    names = [
        "zscore_20_2.2",
        "return_shock_1h_2.175",
        "return_shock_3h_1.725",
        "vwap_atr_48h_1.55",
        "taker_exhaustion_0.925",
    ]
    specs = parse_signal_specs(names)
    assert [s.name for s in specs] == names
    assert specs[-1].requires_taker is True
    assert specs[-1].params["min_shock_atr"] == 0.925


def test_zscore_control_uses_confirmed_reentry_definition():
    base = _base(80)
    base.loc[base.index[40], "close"] = 80.0
    base.loc[base.index[41], "close"] = 100.0
    spec = parse_signal_specs(["zscore_20_2.2"])[0]
    out = apply_signal(base, spec)

    mean = base["close"].rolling(20, min_periods=20).mean()
    std = base["close"].rolling(20, min_periods=20).std()
    z = (base["close"] - mean) / std
    expected = (
        (z.shift(1) < -2.2)
        & (z >= -2.2)
        & base["regime_long"].astype(bool)
        & z.notna()
        & base["atr"].notna()
    )
    pd.testing.assert_series_equal(
        out["entry_long"].astype(bool), expected.fillna(False).astype(bool), check_names=False
    )


def test_taker_exhaustion_accepts_frozen_v10_2_threshold():
    base = _base(130)
    i = 110
    base.iloc[i - 1, base.columns.get_loc("close")] = 95.0
    base.iloc[i, base.columns.get_loc("close")] = 97.0
    taker = pd.DataFrame(
        {"volume": 1000.0, "taker_buy_base_volume": 500.0},
        index=base.index,
    )
    taker.iloc[i - 1, taker.columns.get_loc("taker_buy_base_volume")] = 100.0
    taker.iloc[i, taker.columns.get_loc("taker_buy_base_volume")] = 550.0

    spec = parse_signal_specs(["taker_exhaustion_0.925"])[0]
    out = apply_signal(base, spec, taker=taker)
    assert bool(out.iloc[i]["entry_long"])
