from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_zscore(close: pd.Series, window: int) -> pd.Series:
    """Classic rolling z-score used by the frozen mean-reversion trigger."""
    if window <= 1:
        raise ValueError("window debe ser > 1.")
    values = close.astype(float)
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std().replace(0.0, np.nan)
    return (values - mean) / std


def confirmed_downside_reentry(score: pd.Series, threshold: float) -> pd.Series:
    """True when a downside score crosses back inside its negative threshold."""
    if threshold <= 0:
        raise ValueError("threshold debe ser positivo.")
    return (
        (score.shift(1) < -float(threshold))
        & (score >= -float(threshold))
    ).fillna(False)


def confirmed_upside_reentry(score: pd.Series, threshold: float) -> pd.Series:
    """True when an upside score crosses back inside its positive threshold."""
    if threshold <= 0:
        raise ValueError("threshold debe ser positivo.")
    return (
        (score.shift(1) > float(threshold))
        & (score <= float(threshold))
    ).fillna(False)


def standardized_log_return_shock(
    close: pd.Series,
    *,
    horizon_bars: int,
    vol_window_bars: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (log-return, rolling volatility, standardized shock score)."""
    if horizon_bars <= 0:
        raise ValueError("horizon_bars debe ser positivo.")
    if vol_window_bars <= horizon_bars:
        raise ValueError("vol_window_bars debe ser mayor que horizon_bars.")

    values = close.astype(float)
    log_close = np.log(values.where(values > 0))
    horizon_return = log_close.diff(horizon_bars)
    realized = (
        horizon_return.rolling(
            vol_window_bars,
            min_periods=vol_window_bars,
        )
        .std()
        .replace(0.0, np.nan)
    )
    return horizon_return, realized, horizon_return / realized
