import unittest

import numpy as np
import pandas as pd

from src.data.fetcher import keep_closed_candles
from src.strategy.mean_reversion import MeanReversionStrategy


class StrategyTests(unittest.TestCase):
    def test_incomplete_candle_is_removed(self):
        index = pd.to_datetime(
            ["2026-01-01T10:00:00Z", "2026-01-01T11:00:00Z"], utc=True
        )
        df = pd.DataFrame(
            {
                "open": [1, 1],
                "high": [1, 1],
                "low": [1, 1],
                "close": [1, 1],
                "volume": [1, 1],
            },
            index=index,
        )
        closed = keep_closed_candles(
            df,
            "1h",
            now=pd.Timestamp("2026-01-01T11:30:00Z"),
        )
        self.assertEqual(list(closed.index), [index[0]])

    def test_no_signal_before_indicators_are_ready(self):
        n = 50
        index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        close = np.linspace(100, 120, n)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n),
            },
            index=index,
        )
        strategy = MeanReversionStrategy(
            window=20,
            z_threshold=2.2,
            trend_window=200,
            atr_window=14,
            entry_mode="confirmed",
        )
        result = strategy.calculate_signals(df)
        self.assertFalse(result["entry_long"].any())
        self.assertFalse(result["entry_short"].any())

    def test_confirmed_long_waits_for_cross_back_inside_threshold(self):
        close = list(np.linspace(100, 200, 120)) + [180, 190, 195, 200, 205]
        close = np.array(close, dtype=float)
        index = pd.date_range("2026-01-01", periods=len(close), freq="1h", tz="UTC")
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(len(close)),
            },
            index=index,
        )

        result = MeanReversionStrategy(
            window=20,
            z_threshold=1.5,
            trend_window=50,
            atr_window=14,
            entry_mode="confirmed",
        ).calculate_signals(df)

        signal_rows = result.index[result["entry_long"]]
        self.assertEqual(len(signal_rows), 1)
        idx = signal_rows[0]
        pos = result.index.get_loc(idx)
        self.assertLess(result.iloc[pos - 1]["z_score"], -1.5)
        self.assertGreaterEqual(result.iloc[pos]["z_score"], -1.5)
        self.assertGreater(result.iloc[pos]["close"], result.iloc[pos]["sma_trend"])

    def test_strategy_exposes_mean_exit_columns(self):
        n = 250
        index = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
        close = 100 + np.sin(np.linspace(0, 20, n)) * 5
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n),
            },
            index=index,
        )
        result = MeanReversionStrategy().calculate_signals(df)
        self.assertIn("entry_long", result.columns)
        self.assertIn("entry_short", result.columns)
        self.assertIn("exit_long_mean", result.columns)
        self.assertIn("exit_short_mean", result.columns)

    def test_4h_regime_uses_only_closed_htf_candles(self):
        index = pd.date_range("2026-01-01T00:00:00Z", periods=12, freq="1h")
        close = np.linspace(100.0, 111.0, len(index))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(len(index)),
            },
            index=index,
        )

        regime_index = pd.date_range(
            "2026-01-01T00:00:00Z", periods=3, freq="4h"
        )
        regime_df = pd.DataFrame(
            {
                "open": [90.0, 190.0, 290.0],
                "high": [101.0, 201.0, 301.0],
                "low": [89.0, 189.0, 289.0],
                "close": [100.0, 200.0, 300.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=regime_index,
        )

        result = MeanReversionStrategy(
            window=2,
            z_threshold=1.0,
            trend_window=2,
            atr_window=2,
            entry_mode="confirmed",
            regime_mode="4h_ema200",
            signal_timeframe="1h",
            regime_timeframe="4h",
            regime_ema_window=1,
            regime_slope_lookback=1,
        ).calculate_signals(df, regime_df=regime_df)

        # 02:00 cierra a las 03:00: la vela 4h 00:00-04:00 aun NO existe.
        self.assertTrue(pd.isna(result.loc[index[2], "regime_close"]))
        # 03:00 cierra a las 04:00: ya puede usar 00:00-04:00.
        self.assertEqual(result.loc[index[3], "regime_close"], 100.0)
        # 06:00 cierra a las 07:00: sigue usando la misma 4h cerrada.
        self.assertEqual(result.loc[index[6], "regime_close"], 100.0)
        # 07:00 cierra a las 08:00: ya puede usar 04:00-08:00.
        self.assertEqual(result.loc[index[7], "regime_close"], 200.0)

    def test_4h_slope_mode_exposes_closed_ema_slope(self):
        index = pd.date_range("2026-01-01T00:00:00Z", periods=13, freq="1h")
        close = np.linspace(100.0, 112.0, len(index))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(len(index)),
            },
            index=index,
        )
        regime_index = pd.date_range(
            "2026-01-01T00:00:00Z", periods=3, freq="4h"
        )
        regime_df = pd.DataFrame(
            {
                "open": [99.0, 109.0, 119.0],
                "high": [101.0, 111.0, 121.0],
                "low": [98.0, 108.0, 118.0],
                "close": [100.0, 110.0, 120.0],
                "volume": [1.0, 1.0, 1.0],
            },
            index=regime_index,
        )

        result = MeanReversionStrategy(
            window=2,
            z_threshold=1.0,
            trend_window=2,
            atr_window=2,
            regime_mode="4h_ema200_slope",
            signal_timeframe="1h",
            regime_timeframe="4h",
            regime_ema_window=1,
            regime_slope_lookback=1,
        ).calculate_signals(df, regime_df=regime_df)

        # A las 08:00 ya conocemos dos EMA 4h cerradas: 100 -> 110.
        self.assertTrue(bool(result.loc[index[7], "regime_slope_up"]))
        self.assertFalse(bool(result.loc[index[7], "regime_slope_down"]))

    def test_combined_regime_requires_local_and_htf(self):
        index = pd.date_range("2026-01-01T00:00:00Z", periods=40, freq="1h")
        close = np.linspace(100.0, 140.0, len(index))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(len(index)),
            },
            index=index,
        )

        regime_index = pd.date_range(
            "2026-01-01T00:00:00Z", periods=8, freq="4h"
        )
        # El ultimo close 4h queda por encima de su EMA corta tras un rebote,
        # pero la EMA sigue por debajo de donde estaba 6 velas atras.
        regime_close = np.array([120, 115, 110, 105, 100, 95, 90, 100], dtype=float)
        regime_df = pd.DataFrame(
            {
                "open": regime_close,
                "high": regime_close + 1.0,
                "low": regime_close - 1.0,
                "close": regime_close,
                "volume": np.ones(len(regime_close)),
            },
            index=regime_index,
        )

        no_slope = MeanReversionStrategy(
            window=2,
            z_threshold=1.0,
            trend_window=3,
            atr_window=2,
            regime_mode="1h_sma200_4h_ema200",
            signal_timeframe="1h",
            regime_timeframe="4h",
            regime_ema_window=2,
            regime_slope_lookback=6,
        ).calculate_signals(df, regime_df=regime_df)

        with_slope = MeanReversionStrategy(
            window=2,
            z_threshold=1.0,
            trend_window=3,
            atr_window=2,
            regime_mode="1h_sma200_4h_ema200_slope",
            signal_timeframe="1h",
            regime_timeframe="4h",
            regime_ema_window=2,
            regime_slope_lookback=6,
        ).calculate_signals(df, regime_df=regime_df)

        # La vela 1h 31:00 cierra a las 32:00 y ya conoce la 4h 28:00-32:00.
        row = index[31]
        self.assertTrue(bool(no_slope.loc[row, "local_regime_long"]))
        self.assertTrue(bool(no_slope.loc[row, "htf_regime_long"]))
        self.assertTrue(bool(no_slope.loc[row, "regime_long"]))

        self.assertFalse(bool(with_slope.loc[row, "htf_slope_up"]))
        self.assertFalse(bool(with_slope.loc[row, "regime_long"]))

    def test_combined_regime_rejects_when_local_trend_is_not_long(self):
        index = pd.date_range("2026-01-01T00:00:00Z", periods=20, freq="1h")
        close = np.linspace(120.0, 100.0, len(index))
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(len(index)),
            },
            index=index,
        )
        regime_index = pd.date_range(
            "2026-01-01T00:00:00Z", periods=5, freq="4h"
        )
        regime_close = np.array([100, 105, 110, 115, 120], dtype=float)
        regime_df = pd.DataFrame(
            {
                "open": regime_close,
                "high": regime_close + 1.0,
                "low": regime_close - 1.0,
                "close": regime_close,
                "volume": np.ones(len(regime_close)),
            },
            index=regime_index,
        )

        result = MeanReversionStrategy(
            window=2,
            z_threshold=1.0,
            trend_window=3,
            atr_window=2,
            regime_mode="1h_sma200_4h_ema200",
            signal_timeframe="1h",
            regime_timeframe="4h",
            regime_ema_window=2,
            regime_slope_lookback=1,
        ).calculate_signals(df, regime_df=regime_df)

        row = index[15]
        self.assertFalse(bool(result.loc[row, "local_regime_long"]))
        self.assertTrue(bool(result.loc[row, "htf_regime_long"]))
        self.assertFalse(bool(result.loc[row, "regime_long"]))


if __name__ == "__main__":
    unittest.main()
