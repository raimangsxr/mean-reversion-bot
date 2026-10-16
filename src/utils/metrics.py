from __future__ import annotations

import math

import numpy as np
import pandas as pd


def periods_per_year_from_timeframe(timeframe: str) -> float:
    unit = timeframe[-1].lower()
    value = float(timeframe[:-1])
    seconds_per_unit = {
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
        "w": 7.0 * 86400.0,
    }
    if unit not in seconds_per_unit or value <= 0:
        raise ValueError(f"Timeframe no soportado para anualización: {timeframe}")
    seconds = value * seconds_per_unit[unit]
    return (365.0 * 24.0 * 3600.0) / seconds


import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 8760,
) -> float:
    """Calculate an annualized Sharpe ratio from per-period returns."""

    clean_returns = pd.Series(
        returns,
        dtype="float64",
    ).replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if clean_returns.empty or periods_per_year <= 0:
        return 0.0

    std_return = float(clean_returns.std())

    if not np.isfinite(std_return) or std_return == 0.0:
        return 0.0

    mean_return = float(clean_returns.mean())

    sharpe = (
        np.sqrt(float(periods_per_year))
        * (mean_return - float(risk_free_rate))
        / std_return
    )

    return float(sharpe) if np.isfinite(sharpe) else 0.0
