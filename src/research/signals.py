from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Iterable

import numpy as np
import pandas as pd

from config.settings import (
    ATR_WINDOW,
    ENTRY_MODE,
    LOOKBACK_WINDOW,
    REGIME_EMA_WINDOW,
    REGIME_MODE,
    REGIME_SLOPE_LOOKBACK,
    REGIME_TIMEFRAME,
    TIMEFRAME,
    TREND_WINDOW,
    V10_4_RETURN_SHOCK_VOL_WINDOW,
    V10_4_TAKER_LOOKBACK,
    V10_4_TAKER_MIN_SHOCK_ATR,
    V10_4_TAKER_SELL_QUANTILE,
    Z_THRESHOLD,
)
from src.strategy.mean_reversion import MeanReversionStrategy


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    params: dict[str, float | int]
    requires_taker: bool = False


def parse_signal_spec(name: str) -> SignalSpec:
    value = name.strip().lower()
    if value == "zscore_20_2.2":
        return SignalSpec(value, "zscore", {"window": 20, "threshold": 2.2})

    m = re.fullmatch(r"robust_z_(\d+)_([0-9]+(?:\.[0-9]+)?)", value)
    if m:
        return SignalSpec(value, "robust_z", {"window": int(m.group(1)), "threshold": float(m.group(2))})

    m = re.fullmatch(r"return_shock_(\d+)h_([0-9]+(?:\.[0-9]+)?)", value)
    if m:
        return SignalSpec(value, "return_shock", {"hours": int(m.group(1)), "threshold": float(m.group(2))})

    m = re.fullmatch(r"vwap_atr_(\d+)h_([0-9]+(?:\.[0-9]+)?)", value)
    if m:
        return SignalSpec(value, "vwap_atr", {"window": int(m.group(1)), "threshold": float(m.group(2))})

    m = re.fullmatch(r"rsi(\d+)_([0-9]+(?:\.[0-9]+)?)", value)
    if m:
        return SignalSpec(value, "rsi", {"period": int(m.group(1)), "threshold": float(m.group(2))})

    m = re.fullmatch(r"connors_(\d+)_(\d+)_(\d+)_([0-9]+(?:\.[0-9]+)?)", value)
    if m:
        return SignalSpec(
            value,
            "connors_rsi",
            {
                "price_rsi": int(m.group(1)),
                "streak_rsi": int(m.group(2)),
                "percent_rank": int(m.group(3)),
                "threshold": float(m.group(4)),
            },
        )

    m = re.fullmatch(r"taker_exhaustion(?:_([0-9]+(?:\.[0-9]+)?))?", value)
    if m:
        min_shock = float(m.group(1)) if m.group(1) else V10_4_TAKER_MIN_SHOCK_ATR
        return SignalSpec(
            value,
            "taker_exhaustion",
            {
                "lookback": V10_4_TAKER_LOOKBACK,
                "sell_quantile": V10_4_TAKER_SELL_QUANTILE,
                "min_shock_atr": min_shock,
            },
            requires_taker=True,
        )

    raise ValueError(f"Senal research desconocida: {name}")


def parse_signal_specs(names: Iterable[str]) -> list[SignalSpec]:
    specs = [parse_signal_spec(name) for name in names]
    if len({s.name for s in specs}) != len(specs):
        raise ValueError("signal set contiene duplicados")
    return specs




@dataclass(frozen=True)
class FrequencyCalibration:
    base_name: str
    signal_name: str
    family: str
    parameter: str
    value: float
    target_count: int
    actual_count: int
    mismatch_ratio: float
    within_tolerance: bool


def _calibrated_label(spec: SignalSpec) -> str:
    if spec.family == "zscore":
        return spec.name
    if spec.family == "robust_z":
        return f"robust_z_{int(spec.params['window'])}_cal"
    if spec.family == "return_shock":
        return f"return_shock_{int(spec.params['hours'])}h_cal"
    if spec.family == "vwap_atr":
        return f"vwap_atr_{int(spec.params['window'])}h_cal"
    if spec.family == "rsi":
        return f"rsi{int(spec.params['period'])}_cal"
    if spec.family == "connors_rsi":
        return (
            f"connors_{int(spec.params['price_rsi'])}_"
            f"{int(spec.params['streak_rsi'])}_"
            f"{int(spec.params['percent_rank'])}_cal"
        )
    if spec.family == "taker_exhaustion":
        return "taker_exhaustion_cal"
    raise ValueError(f"Familia research no calibrable: {spec.family}")


def _calibration_parameter(spec: SignalSpec) -> tuple[str, np.ndarray]:
    if spec.family == "zscore":
        return "threshold", np.asarray([float(spec.params["threshold"])], dtype=float)
    if spec.family == "robust_z":
        return "threshold", np.arange(0.50, 4.5001, 0.025, dtype=float)
    if spec.family == "return_shock":
        return "threshold", np.arange(0.50, 4.0001, 0.025, dtype=float)
    if spec.family == "vwap_atr":
        return "threshold", np.arange(0.25, 4.0001, 0.025, dtype=float)
    if spec.family == "rsi":
        return "threshold", np.arange(5.0, 55.0001, 0.25, dtype=float)
    if spec.family == "connors_rsi":
        return "threshold", np.arange(2.0, 40.0001, 0.25, dtype=float)
    if spec.family == "taker_exhaustion":
        return "min_shock_atr", np.arange(0.10, 3.0001, 0.025, dtype=float)
    raise ValueError(f"Familia research no calibrable: {spec.family}")


def _taker_exhaustion_components(
    base: pd.DataFrame,
    spec: SignalSpec,
    taker: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    aligned = taker.reindex(base.index)
    buy = pd.to_numeric(aligned["taker_buy_base_volume"], errors="coerce")
    volume = pd.to_numeric(aligned["volume"], errors="coerce")
    sell_ratio = ((volume - buy) / volume.replace(0.0, np.nan)).clip(0.0, 1.0)
    lookback = int(spec.params["lookback"])
    q = float(spec.params["sell_quantile"])
    threshold = sell_ratio.rolling(lookback, min_periods=lookback).quantile(q).shift(1)
    shock_atr = base["close"].astype(float).diff() / base["atr"].replace(0.0, np.nan)
    sell_extreme_prev = sell_ratio.shift(1) >= threshold.shift(1)
    recovery = (base["close"] > base["close"].shift(1)) & (sell_ratio < sell_ratio.shift(1))
    return sell_ratio, shock_atr, sell_extreme_prev, recovery


def _count_entries_for_parameter(
    base: pd.DataFrame,
    spec: SignalSpec,
    value: float,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    taker: pd.DataFrame | None = None,
    score: pd.Series | None = None,
) -> int:
    period = (base.index >= start) & (base.index <= end)
    ready = base[["regime_ma", "atr"]].notna().all(axis=1) & base["regime_long"].astype(bool)

    if spec.family == "taker_exhaustion":
        if taker is None or taker.empty:
            raise ValueError("taker_exhaustion requiere historico taker")
        sell_ratio, shock_atr, sell_extreme_prev, recovery = _taker_exhaustion_components(base, spec, taker)
        ready = ready & sell_ratio.notna()
        entries = (
            (shock_atr.shift(1) <= -float(value))
            & sell_extreme_prev
            & recovery
            & ready
            & period
        )
        return int(entries.fillna(False).sum())

    if score is None:
        score = apply_signal(base, spec, taker=taker)["signal_value"]
    ready = ready & score.notna()
    previous = score.shift(1)
    if spec.family in {"zscore", "robust_z", "return_shock", "vwap_atr"}:
        entries = (previous < -float(value)) & (score >= -float(value))
    elif spec.family in {"rsi", "connors_rsi"}:
        entries = (previous < float(value)) & (score >= float(value))
    else:
        raise ValueError(f"Familia research no calibrable: {spec.family}")
    return int((entries & ready & period).fillna(False).sum())


def calibrate_signal_frequency(
    base: pd.DataFrame,
    spec: SignalSpec,
    *,
    target_count: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    tolerance: float,
    taker: pd.DataFrame | None = None,
    fixed: bool = False,
) -> tuple[SignalSpec, FrequencyCalibration]:
    if target_count <= 0:
        raise ValueError("target_count debe ser > 0")
    if tolerance < 0:
        raise ValueError("tolerance no puede ser negativa")

    parameter, values = _calibration_parameter(spec)
    score = None
    if spec.family != "taker_exhaustion":
        score = apply_signal(base, spec, taker=taker)["signal_value"]

    if fixed:
        values = np.asarray([float(spec.params[parameter])], dtype=float)

    default_value = float(spec.params[parameter])
    best_value = default_value
    best_count = -1
    best_key: tuple[float, float, float] | None = None
    for candidate in values:
        count = _count_entries_for_parameter(
            base, spec, float(candidate), start=start, end=end, taker=taker, score=score
        )
        key = (abs(count - target_count), abs(float(candidate) - default_value), float(candidate))
        if best_key is None or key < best_key:
            best_key = key
            best_value = float(candidate)
            best_count = count

    params = dict(spec.params)
    params[parameter] = best_value
    calibrated = replace(spec, name=_calibrated_label(spec), params=params)
    mismatch = abs(best_count - target_count) / target_count
    result = FrequencyCalibration(
        base_name=spec.name,
        signal_name=calibrated.name,
        family=spec.family,
        parameter=parameter,
        value=best_value,
        target_count=int(target_count),
        actual_count=int(best_count),
        mismatch_ratio=float(mismatch),
        within_tolerance=bool(mismatch <= tolerance),
    )
    return calibrated, result


def _wilder_rsi(values: pd.Series, period: int) -> pd.Series:
    delta = values.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # Si no hubo perdidas durante la ventana, RSI=100. Si tampoco hubo ganancias,
    # el mercado esta plano y usamos 50.
    rsi = rsi.where(avg_loss > 0.0, 100.0)
    rsi = rsi.where((avg_gain > 0.0) | (avg_loss > 0.0), 50.0)
    return rsi


def _streak(close: pd.Series) -> pd.Series:
    diff = close.diff().to_numpy(dtype=float)
    out = np.zeros(len(close), dtype=float)
    for i in range(1, len(out)):
        if not np.isfinite(diff[i]) or diff[i] == 0:
            out[i] = 0.0
        elif diff[i] > 0:
            out[i] = out[i - 1] + 1.0 if out[i - 1] > 0 else 1.0
        else:
            out[i] = out[i - 1] - 1.0 if out[i - 1] < 0 else -1.0
    return pd.Series(out, index=close.index)


def _percent_rank_last(values: np.ndarray) -> float:
    if len(values) == 0 or not np.isfinite(values[-1]):
        return float("nan")
    prior = values[:-1]
    prior = prior[np.isfinite(prior)]
    if len(prior) == 0:
        return float("nan")
    return float(np.mean(prior < values[-1]) * 100.0)


def prepare_base_features(df: pd.DataFrame, regime_df: pd.DataFrame) -> pd.DataFrame:
    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW,
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=ATR_WINDOW,
        entry_mode=ENTRY_MODE,
        regime_mode=REGIME_MODE,
        signal_timeframe=TIMEFRAME,
        regime_timeframe=REGIME_TIMEFRAME,
        regime_ema_window=REGIME_EMA_WINDOW,
        regime_slope_lookback=REGIME_SLOPE_LOOKBACK,
    )
    result = df.copy().sort_index()
    result["atr"] = strategy.calculate_atr(result)
    result = strategy._apply_regime(result, regime_df)  # research-only reuse of frozen regime logic
    result["exit_long_mean"] = False
    result["exit_short_mean"] = False
    return result


def _confirmed_from_score(base: pd.DataFrame, score: pd.Series, threshold: float) -> pd.DataFrame:
    result = base.copy()
    result["signal_value"] = score
    ready = result[["signal_value", "regime_ma", "atr"]].notna().all(axis=1)
    previous = result["signal_value"].shift(1)
    result["entry_long"] = (
        ready
        & (previous < -threshold)
        & (result["signal_value"] >= -threshold)
        & result["regime_long"].astype(bool)
    )
    result["entry_short"] = False
    return result


def apply_signal(
    base: pd.DataFrame,
    spec: SignalSpec,
    *,
    taker: pd.DataFrame | None = None,
) -> pd.DataFrame:
    close = base["close"].astype(float)

    if spec.family == "zscore":
        window = int(spec.params["window"])
        threshold = float(spec.params["threshold"])
        mean = close.rolling(window, min_periods=window).mean()
        std = close.rolling(window, min_periods=window).std().replace(0.0, np.nan)
        return _confirmed_from_score(base, (close - mean) / std, threshold)

    if spec.family == "robust_z":
        window = int(spec.params["window"])
        threshold = float(spec.params["threshold"])
        median = close.rolling(window, min_periods=window).median()
        mad = close.rolling(window, min_periods=window).apply(
            lambda x: float(np.median(np.abs(x - np.median(x)))), raw=True
        )
        scale = (1.4826 * mad).replace(0.0, np.nan)
        return _confirmed_from_score(base, (close - median) / scale, threshold)

    if spec.family == "return_shock":
        hours = int(spec.params["hours"])
        threshold = float(spec.params["threshold"])
        log_close = np.log(close.where(close > 0))
        ret_1h = log_close.diff()
        horizon_return = log_close.diff(hours)
        realized = ret_1h.rolling(
            V10_4_RETURN_SHOCK_VOL_WINDOW,
            min_periods=V10_4_RETURN_SHOCK_VOL_WINDOW,
        ).std()
        denom = (realized * math.sqrt(hours)).replace(0.0, np.nan)
        return _confirmed_from_score(base, horizon_return / denom, threshold)

    if spec.family == "vwap_atr":
        window = int(spec.params["window"])
        threshold = float(spec.params["threshold"])
        volume = base["volume"].astype(float)
        typical = (base["high"] + base["low"] + base["close"]) / 3.0
        numerator = (typical * volume).rolling(window, min_periods=window).sum()
        denominator = volume.rolling(window, min_periods=window).sum().replace(0.0, np.nan)
        vwap = numerator / denominator
        score = (close - vwap) / base["atr"].replace(0.0, np.nan)
        return _confirmed_from_score(base, score, threshold)

    if spec.family == "rsi":
        period = int(spec.params["period"])
        threshold = float(spec.params["threshold"])
        rsi = _wilder_rsi(close, period)
        result = base.copy()
        result["signal_value"] = rsi
        ready = result[["signal_value", "regime_ma", "atr"]].notna().all(axis=1)
        result["entry_long"] = (
            ready
            & (rsi.shift(1) < threshold)
            & (rsi >= threshold)
            & result["regime_long"].astype(bool)
        )
        result["entry_short"] = False
        return result

    if spec.family == "connors_rsi":
        price_period = int(spec.params["price_rsi"])
        streak_period = int(spec.params["streak_rsi"])
        rank_window = int(spec.params["percent_rank"])
        threshold = float(spec.params["threshold"])
        price_rsi = _wilder_rsi(close, price_period)
        streak_rsi = _wilder_rsi(_streak(close), streak_period)
        one_bar_return = close.pct_change()
        percent_rank = one_bar_return.rolling(
            rank_window + 1, min_periods=rank_window + 1
        ).apply(_percent_rank_last, raw=True)
        connors = (price_rsi + streak_rsi + percent_rank) / 3.0
        result = base.copy()
        result["signal_value"] = connors
        ready = result[["signal_value", "regime_ma", "atr"]].notna().all(axis=1)
        result["entry_long"] = (
            ready
            & (connors.shift(1) < threshold)
            & (connors >= threshold)
            & result["regime_long"].astype(bool)
        )
        result["entry_short"] = False
        return result

    if spec.family == "taker_exhaustion":
        if taker is None or taker.empty:
            raise ValueError("taker_exhaustion requiere historico taker")
        min_shock_atr = float(spec.params["min_shock_atr"])
        sell_ratio, shock_atr, sell_extreme_prev, recovery = _taker_exhaustion_components(
            base, spec, taker
        )
        extreme_prev = (shock_atr.shift(1) <= -min_shock_atr) & sell_extreme_prev
        result = base.copy()
        result["signal_value"] = sell_ratio
        ready = result[["signal_value", "regime_ma", "atr"]].notna().all(axis=1)
        result["entry_long"] = (
            ready & extreme_prev & recovery & result["regime_long"].astype(bool)
        )
        result["entry_short"] = False
        return result

    raise ValueError(f"Familia research no implementada: {spec.family}")

# -----------------------------------------------------------------------------
# V10.4 frequency-expansion helpers.
# These functions keep the 1h/4h regime fixed while allowing the signal
# timeframe to change. They do not tune any threshold.
# -----------------------------------------------------------------------------
def _hours_per_bar(timeframe: str) -> float:
    from src.data.fetcher import timeframe_to_timedelta

    delta = timeframe_to_timedelta(timeframe)
    return float(delta / pd.Timedelta(hours=1))


def scaled_window_for_hours(hours: float, timeframe: str, *, minimum: int = 2) -> int:
    """Map a fixed wall-clock lookback to whole bars for a research timeframe."""
    per_bar = _hours_per_bar(timeframe)
    if per_bar <= 0:
        raise ValueError("timeframe invalido")
    return max(minimum, int(round(float(hours) / per_bar)))


def _align_by_availability(
    target_index: pd.DatetimeIndex,
    target_timeframe: str,
    source: pd.DataFrame,
    source_timeframe: str,
    columns: list[str],
) -> pd.DataFrame:
    """Align only information whose source candle is closed at target close."""
    from src.data.fetcher import timeframe_to_timedelta

    left = pd.DataFrame(
        {
            "_row_id": np.arange(len(target_index), dtype=int),
            "available_at": target_index + timeframe_to_timedelta(target_timeframe),
        }
    ).sort_values("available_at")
    right = source[columns].copy().sort_index().reset_index()
    idx_name = source.index.name or "index"
    right = right.rename(columns={idx_name: "source_open_time"})
    right["source_available_at"] = (
        right["source_open_time"] + timeframe_to_timedelta(source_timeframe)
    )
    right = right[["source_available_at", *columns]].sort_values("source_available_at")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="available_at",
        right_on="source_available_at",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("_row_id")
    return aligned[columns].set_axis(target_index)


def prepare_frequency_expansion_base(
    signal_df: pd.DataFrame,
    regime_1h: pd.DataFrame,
    regime_4h: pd.DataFrame,
    *,
    signal_timeframe: str,
    atr_hours: float = 14.0,
) -> pd.DataFrame:
    """Build signal-timeframe OHLCV with the frozen 1h SMA200 + 4h EMA200 regime.

    The local trend is deliberately sourced from closed 1h candles rather than
    recomputed on 30m/2h. This keeps the V9.2/V10.4 regime semantics unchanged.
    ATR is scaled to approximately the same 14-hour wall-clock horizon.
    """
    result = signal_df.copy().sort_index()
    atr_bars = scaled_window_for_hours(atr_hours, signal_timeframe, minimum=2)
    strategy = MeanReversionStrategy(
        window=LOOKBACK_WINDOW,
        z_threshold=Z_THRESHOLD,
        trend_window=TREND_WINDOW,
        atr_window=atr_bars,
        entry_mode=ENTRY_MODE,
        regime_mode=REGIME_MODE,
        signal_timeframe=signal_timeframe,
        regime_timeframe=REGIME_TIMEFRAME,
        regime_ema_window=REGIME_EMA_WINDOW,
        regime_slope_lookback=REGIME_SLOPE_LOOKBACK,
    )
    result["atr"] = strategy.calculate_atr(result)

    local = regime_1h[["close"]].copy().sort_index()
    local["local_sma"] = local["close"].rolling(
        TREND_WINDOW, min_periods=TREND_WINDOW
    ).mean()
    local["local_regime_long"] = local["close"] > local["local_sma"]
    local["local_regime_short"] = local["close"] < local["local_sma"]
    local = local.rename(columns={"close": "local_close"})

    htf = regime_4h[["close"]].copy().sort_index()
    htf["htf_ma"] = htf["close"].ewm(
        span=REGIME_EMA_WINDOW,
        adjust=False,
        min_periods=REGIME_EMA_WINDOW,
    ).mean()
    htf["htf_regime_long"] = htf["close"] > htf["htf_ma"]
    htf["htf_regime_short"] = htf["close"] < htf["htf_ma"]
    htf["htf_slope_up"] = htf["htf_ma"] > htf["htf_ma"].shift(REGIME_SLOPE_LOOKBACK)
    htf["htf_slope_down"] = htf["htf_ma"] < htf["htf_ma"].shift(REGIME_SLOPE_LOOKBACK)
    htf = htf.rename(columns={"close": "htf_close"})

    local_aligned = _align_by_availability(
        result.index,
        signal_timeframe,
        local,
        "1h",
        ["local_close", "local_sma", "local_regime_long", "local_regime_short"],
    )
    htf_aligned = _align_by_availability(
        result.index,
        signal_timeframe,
        htf,
        "4h",
        [
            "htf_close",
            "htf_ma",
            "htf_regime_long",
            "htf_regime_short",
            "htf_slope_up",
            "htf_slope_down",
        ],
    )
    for col in local_aligned.columns:
        result[col] = local_aligned[col].to_numpy()
    for col in htf_aligned.columns:
        result[col] = htf_aligned[col].to_numpy()

    for col in [
        "local_regime_long",
        "local_regime_short",
        "htf_regime_long",
        "htf_regime_short",
        "htf_slope_up",
        "htf_slope_down",
    ]:
        result[col] = pd.array(result[col], dtype="boolean").fillna(False)

    result["regime_long"] = result["local_regime_long"] & result["htf_regime_long"]
    result["regime_short"] = result["local_regime_short"] & result["htf_regime_short"]
    result["regime_close"] = result["local_close"]
    result["regime_ma"] = result["local_sma"]
    result["regime_slope_up"] = result["htf_slope_up"]
    result["regime_slope_down"] = result["htf_slope_down"]
    result["sma_trend"] = result["local_sma"]
    result["exit_long_mean"] = False
    result["exit_short_mean"] = False
    return result


def _confirmed_event_from_score(score: pd.Series, threshold: float) -> pd.Series:
    return ((score.shift(1) < -float(threshold)) & (score >= -float(threshold))).fillna(False)


def _shock1h_event_on_frame(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    threshold: float,
    vol_window_hours: float = 100.0,
) -> pd.Series:
    """Confirmed standardized one-hour downside shock on <=1h bars."""
    per_bar = _hours_per_bar(timeframe)
    if per_bar > 1.0 + 1e-12:
        raise ValueError("_shock1h_event_on_frame solo soporta timeframes <=1h")
    horizon = max(1, int(round(1.0 / per_bar)))
    if not math.isclose(horizon * per_bar, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"{timeframe} no divide exactamente una hora")
    vol_window = scaled_window_for_hours(vol_window_hours, timeframe, minimum=horizon + 2)
    log_close = np.log(frame["close"].astype(float).where(frame["close"].astype(float) > 0))
    one_hour_return = log_close.diff(horizon)
    realized = one_hour_return.rolling(vol_window, min_periods=vol_window).std().replace(0.0, np.nan)
    score = one_hour_return / realized
    return _confirmed_event_from_score(score, threshold)


def build_frequency_expansion_signal(
    base: pd.DataFrame,
    *,
    signal_timeframe: str,
    shock_threshold: float,
    reference_1h: pd.DataFrame | None = None,
    z_window_hours: float = 20.0,
    z_threshold: float = 2.2,
) -> pd.DataFrame:
    """Frozen z-score + shock1h candidate adapted to a research timeframe.

    Z lookback is kept near 20 wall-clock hours. For <=1h bars the shock is a
    true rolling 1-hour return. For >1h bars the shock confirmation is sourced
    from the last 1h candle closing at the same signal-close timestamp.
    """
    result = base.copy()
    z_bars = scaled_window_for_hours(z_window_hours, signal_timeframe, minimum=3)
    close = result["close"].astype(float)
    mean = close.rolling(z_bars, min_periods=z_bars).mean()
    std = close.rolling(z_bars, min_periods=z_bars).std().replace(0.0, np.nan)
    z = (close - mean) / std
    z_event = _confirmed_event_from_score(z, z_threshold)

    if _hours_per_bar(signal_timeframe) <= 1.0 + 1e-12:
        shock_event = _shock1h_event_on_frame(
            result, timeframe=signal_timeframe, threshold=shock_threshold
        )
    else:
        if reference_1h is None or reference_1h.empty:
            raise ValueError("reference_1h es obligatorio para timeframes >1h")
        ref_event = _shock1h_event_on_frame(
            reference_1h, timeframe="1h", threshold=shock_threshold
        )
        from src.data.fetcher import timeframe_to_timedelta

        available = pd.Series(
            ref_event.to_numpy(dtype=bool),
            index=reference_1h.index + timeframe_to_timedelta("1h"),
        )
        target_available = result.index + timeframe_to_timedelta(signal_timeframe)
        shock_event = pd.Series(
            available.reindex(target_available, fill_value=False).to_numpy(dtype=bool),
            index=result.index,
        )

    ready = result[["regime_ma", "atr"]].notna().all(axis=1)
    result["signal_value"] = z
    result["entry_long"] = (
        ready
        & z_event
        & shock_event
        & result["regime_long"].astype(bool)
    )
    result["entry_short"] = False
    return result
