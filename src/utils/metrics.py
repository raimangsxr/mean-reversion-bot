import numpy as np
import pandas as pd

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 8760) -> float:
    mean_return = returns.mean()
    std_return = returns.std()
    if std_return == 0:
        return 0.0
    return np.sqrt(periods_per_year) * (mean_return - risk_free_rate) / std_return