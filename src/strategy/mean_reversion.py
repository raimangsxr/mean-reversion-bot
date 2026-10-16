from __future__ import annotations

import numpy as np
import pandas as pd


VALID_REGIME_MODES = {
    "1h_sma200",
    "4h_ema200",
    "4h_ema200_slope",
    "1h_sma200_4h_ema200",
    "1h_sma200_4h_ema200_slope",
}

HTF_REGIME_MODES = {
    "4h_ema200",
    "4h_ema200_slope",
    "1h_sma200_4h_ema200",
    "1h_sma200_4h_ema200_slope",
}


class MeanReversionStrategy:
    """Z-score mean reversion dentro de un regimen de tendencia.

    Modos de entrada:
    - ``extreme``: entra mientras z permanece fuera del umbral.
    - ``confirmed``: espera a que z vuelva a cruzar hacia dentro del umbral.

    Regimenes:
    - ``1h_sma200``: close 1h sobre/bajo SMA local.
    - ``4h_ema200``: ultimo close 4h CERRADO sobre/bajo EMA 4h.
    - ``4h_ema200_slope``: lo anterior + pendiente EMA 4h.
    - ``1h_sma200_4h_ema200``: exige simultaneamente tendencia local y macro.
    - ``1h_sma200_4h_ema200_slope``: lo anterior + pendiente macro.

    Para cualquier modo que usa 4h, la alineacion temporal utiliza solo velas
    4h ya cerradas cuando termina cada vela de senal 1h. No hay look-ahead.
    """

    def __init__(
        self,
        window: int = 20,
        z_threshold: float = 2.2,
        trend_window: int = 200,
        atr_window: int = 14,
        entry_mode: str = "confirmed",
        mean_exit_z: float = 0.0,
        regime_mode: str = "1h_sma200",
        signal_timeframe: str = "1h",
        regime_timeframe: str = "4h",
        regime_ema_window: int = 200,
        regime_slope_lookback: int = 6,
    ) -> None:
        if entry_mode not in {"extreme", "confirmed"}:
            raise ValueError("entry_mode debe ser 'extreme' o 'confirmed'.")
        if mean_exit_z < 0:
            raise ValueError("mean_exit_z no puede ser negativo.")
        if regime_mode not in VALID_REGIME_MODES:
            raise ValueError(
                "regime_mode debe ser uno de: " + ", ".join(sorted(VALID_REGIME_MODES))
            )
        if regime_ema_window <= 0:
            raise ValueError("regime_ema_window debe ser positivo.")
        if regime_slope_lookback <= 0:
            raise ValueError("regime_slope_lookback debe ser positivo.")

        self.window = window
        self.z_threshold = z_threshold
        self.trend_window = trend_window
        self.atr_window = atr_window
        self.entry_mode = entry_mode
        self.mean_exit_z = mean_exit_z
        self.regime_mode = regime_mode
        self.signal_timeframe = signal_timeframe
        self.regime_timeframe = regime_timeframe
        self.regime_ema_window = regime_ema_window
        self.regime_slope_lookback = regime_slope_lookback

    @staticmethod
    def _timeframe_delta(timeframe: str) -> pd.Timedelta:
        try:
            delta = pd.Timedelta(timeframe)
        except ValueError as exc:
            raise ValueError(f"Timeframe no compatible con pandas: {timeframe}") from exc
        if delta <= pd.Timedelta(0):
            raise ValueError("El timeframe debe ser positivo.")
        return delta

    def calculate_atr(self, df: pd.DataFrame) -> pd.Series:
        previous_close = df["close"].shift(1)
        true_range = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - previous_close).abs(),
                (df["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return true_range.rolling(
            self.atr_window,
            min_periods=self.atr_window,
        ).mean()

    def _add_local_regime(self, result: pd.DataFrame) -> pd.DataFrame:
        result["local_close"] = result["close"]
        result["local_sma"] = result["close"].rolling(
            self.trend_window,
            min_periods=self.trend_window,
        ).mean()
        result["local_regime_long"] = result["local_close"] > result["local_sma"]
        result["local_regime_short"] = result["local_close"] < result["local_sma"]
        return result

    def _add_aligned_htf_regime(
        self,
        result: pd.DataFrame,
        regime_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if regime_df is None or regime_df.empty:
            raise ValueError(
                f"regime_df es obligatorio para regime_mode={self.regime_mode}."
            )
        if "close" not in regime_df.columns:
            raise ValueError("regime_df debe contener la columna 'close'.")
        if not isinstance(regime_df.index, pd.DatetimeIndex):
            raise ValueError("regime_df debe usar DatetimeIndex.")

        htf = regime_df[["close"]].copy().sort_index()
        htf["htf_ma"] = htf["close"].ewm(
            span=self.regime_ema_window,
            adjust=False,
            min_periods=self.regime_ema_window,
        ).mean()
        htf["htf_slope_up"] = (
            htf["htf_ma"] > htf["htf_ma"].shift(self.regime_slope_lookback)
        )
        htf["htf_slope_down"] = (
            htf["htf_ma"] < htf["htf_ma"].shift(self.regime_slope_lookback)
        )
        htf["htf_regime_long"] = htf["close"] > htf["htf_ma"]
        htf["htf_regime_short"] = htf["close"] < htf["htf_ma"]

        signal_delta = self._timeframe_delta(self.signal_timeframe)
        regime_delta = self._timeframe_delta(self.regime_timeframe)

        left = pd.DataFrame(
            {
                "_row_id": np.arange(len(result), dtype=int),
                "signal_available_at": result.index + signal_delta,
            }
        ).sort_values("signal_available_at")

        right = htf.reset_index().rename(
            columns={
                htf.index.name or "index": "htf_open_time",
                "close": "htf_close",
            }
        )
        right["htf_available_at"] = right["htf_open_time"] + regime_delta
        right = right[
            [
                "htf_available_at",
                "htf_close",
                "htf_ma",
                "htf_slope_up",
                "htf_slope_down",
                "htf_regime_long",
                "htf_regime_short",
            ]
        ].sort_values("htf_available_at")

        aligned = pd.merge_asof(
            left,
            right,
            left_on="signal_available_at",
            right_on="htf_available_at",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("_row_id")

        for column in [
            "htf_close",
            "htf_ma",
            "htf_slope_up",
            "htf_slope_down",
            "htf_regime_long",
            "htf_regime_short",
        ]:
            result[column] = aligned[column].to_numpy()

        for column in [
            "htf_slope_up",
            "htf_slope_down",
            "htf_regime_long",
            "htf_regime_short",
        ]:
            result[column] = pd.array(result[column], dtype="boolean").fillna(False)

        return result

    def _apply_regime(
        self,
        result: pd.DataFrame,
        regime_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        result = self._add_local_regime(result)

        # Columnas HTF siempre presentes para facilitar logs/diagnostico.
        if self.regime_mode in HTF_REGIME_MODES:
            result = self._add_aligned_htf_regime(result, regime_df)
        else:
            result["htf_close"] = np.nan
            result["htf_ma"] = np.nan
            result["htf_slope_up"] = False
            result["htf_slope_down"] = False
            result["htf_regime_long"] = False
            result["htf_regime_short"] = False

        if self.regime_mode == "1h_sma200":
            result["regime_long"] = result["local_regime_long"]
            result["regime_short"] = result["local_regime_short"]
            result["regime_close"] = result["local_close"]
            result["regime_ma"] = result["local_sma"]
            result["regime_slope_up"] = True
            result["regime_slope_down"] = True

        elif self.regime_mode == "4h_ema200":
            result["regime_long"] = result["htf_regime_long"]
            result["regime_short"] = result["htf_regime_short"]
            result["regime_close"] = result["htf_close"]
            result["regime_ma"] = result["htf_ma"]
            result["regime_slope_up"] = result["htf_slope_up"]
            result["regime_slope_down"] = result["htf_slope_down"]

        elif self.regime_mode == "4h_ema200_slope":
            result["regime_long"] = (
                result["htf_regime_long"] & result["htf_slope_up"]
            )
            result["regime_short"] = (
                result["htf_regime_short"] & result["htf_slope_down"]
            )
            result["regime_close"] = result["htf_close"]
            result["regime_ma"] = result["htf_ma"]
            result["regime_slope_up"] = result["htf_slope_up"]
            result["regime_slope_down"] = result["htf_slope_down"]

        elif self.regime_mode == "1h_sma200_4h_ema200":
            result["regime_long"] = (
                result["local_regime_long"] & result["htf_regime_long"]
            )
            result["regime_short"] = (
                result["local_regime_short"] & result["htf_regime_short"]
            )
            # El regimen generico conserva la referencia local; las columnas
            # htf_* muestran por separado la condicion macro.
            result["regime_close"] = result["local_close"]
            result["regime_ma"] = result["local_sma"]
            result["regime_slope_up"] = result["htf_slope_up"]
            result["regime_slope_down"] = result["htf_slope_down"]

        elif self.regime_mode == "1h_sma200_4h_ema200_slope":
            result["regime_long"] = (
                result["local_regime_long"]
                & result["htf_regime_long"]
                & result["htf_slope_up"]
            )
            result["regime_short"] = (
                result["local_regime_short"]
                & result["htf_regime_short"]
                & result["htf_slope_down"]
            )
            result["regime_close"] = result["local_close"]
            result["regime_ma"] = result["local_sma"]
            result["regime_slope_up"] = result["htf_slope_up"]
            result["regime_slope_down"] = result["htf_slope_down"]

        for column in [
            "local_regime_long",
            "local_regime_short",
            "regime_long",
            "regime_short",
            "regime_slope_up",
            "regime_slope_down",
        ]:
            result[column] = pd.array(result[column], dtype="boolean").fillna(False)

        # Alias usado por logs y diagnostico.
        result["sma_trend"] = result["local_sma"]
        return result

    def calculate_signals(
        self,
        df: pd.DataFrame,
        regime_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        required = {"open", "high", "low", "close", "volume"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"Faltan columnas OHLCV: {sorted(missing)}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("df debe usar DatetimeIndex.")

        result = df.copy().sort_index()

        result["sma"] = result["close"].rolling(
            self.window,
            min_periods=self.window,
        ).mean()
        result["std"] = result["close"].rolling(
            self.window,
            min_periods=self.window,
        ).std()
        result["std"] = result["std"].replace(0.0, np.nan)
        result["z_score"] = (result["close"] - result["sma"]) / result["std"]
        result["atr"] = self.calculate_atr(result)

        result = self._apply_regime(result, regime_df)

        ready = result[["z_score", "regime_ma", "atr"]].notna().all(axis=1)
        long_regime = result["regime_long"]
        short_regime = result["regime_short"]

        result["entry_long_extreme"] = (
            ready & (result["z_score"] < -self.z_threshold) & long_regime
        )
        result["entry_short_extreme"] = (
            ready & (result["z_score"] > self.z_threshold) & short_regime
        )

        previous_z = result["z_score"].shift(1)
        result["entry_long_confirmed"] = (
            ready
            & (previous_z < -self.z_threshold)
            & (result["z_score"] >= -self.z_threshold)
            & long_regime
        )
        result["entry_short_confirmed"] = (
            ready
            & (previous_z > self.z_threshold)
            & (result["z_score"] <= self.z_threshold)
            & short_regime
        )

        if self.entry_mode == "extreme":
            result["entry_long"] = result["entry_long_extreme"]
            result["entry_short"] = result["entry_short_extreme"]
        else:
            result["entry_long"] = result["entry_long_confirmed"]
            result["entry_short"] = result["entry_short_confirmed"]

        result["exit_long_mean"] = ready & (result["z_score"] >= self.mean_exit_z)
        result["exit_short_mean"] = ready & (
            result["z_score"] <= -self.mean_exit_z
        )

        return result
