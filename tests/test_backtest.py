import unittest

import pandas as pd

from backtest import ExecutionModel, simulate
from src.risk.manager import RiskManager


class BacktestExecutionCostTests(unittest.TestCase):
    def setUp(self):
        self.risk = RiskManager(
            risk_per_trade=0.005,
            max_position_pct=0.10,
            atr_multiplier=1.5,
            risk_reward_ratio=1.8,
            cash_buffer_pct=0.05,
        )

    @staticmethod
    def _tp_signals() -> pd.DataFrame:
        index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
        return pd.DataFrame(
            {
                "open": [100.0, 100.0, 103.0],
                "high": [101.0, 104.0, 104.0],
                "low": [99.5, 99.0, 102.0],
                "close": [100.0, 103.0, 103.0],
                "atr": [1.0, 1.0, 1.0],
                "entry_long": [True, False, False],
                "entry_short": [False, False, False],
            },
            index=index,
        )

    def test_maker_tp_reduces_costs_vs_all_taker(self):
        signals = self._tp_signals()
        all_taker = ExecutionModel(
            profile="all_taker",
            taker_fee=0.0005,
            maker_fee=0.0002,
            market_slippage=0.0005,
        )
        maker_tp = ExecutionModel(
            profile="maker_tp",
            taker_fee=0.0005,
            maker_fee=0.0002,
            market_slippage=0.0005,
        )

        result_taker = simulate(
            signals,
            self.risk,
            allow_longs=True,
            allow_shorts=False,
            execution_model=all_taker,
        )
        result_maker = simulate(
            signals,
            self.risk,
            allow_longs=True,
            allow_shorts=False,
            execution_model=maker_tp,
        )

        self.assertEqual(len(result_taker.trades), 1)
        self.assertEqual(len(result_maker.trades), 1)
        self.assertEqual(result_maker.trades.iloc[0]["exit_reason"], "take_profit")
        self.assertLess(
            result_maker.trades.iloc[0]["fees"],
            result_taker.trades.iloc[0]["fees"],
        )
        self.assertLess(
            result_maker.trades.iloc[0]["slippage_cost"],
            result_taker.trades.iloc[0]["slippage_cost"],
        )
        self.assertGreater(
            result_maker.trades.iloc[0]["pnl"],
            result_taker.trades.iloc[0]["pnl"],
        )

    def test_stop_remains_taker_in_maker_tp_profile(self):
        index = pd.date_range("2026-01-01", periods=3, freq="1h", tz="UTC")
        signals = pd.DataFrame(
            {
                "open": [100.0, 100.0, 98.0],
                "high": [101.0, 100.5, 99.0],
                "low": [99.5, 98.0, 97.0],
                "close": [100.0, 98.5, 98.0],
                "atr": [1.0, 1.0, 1.0],
                "entry_long": [True, False, False],
                "entry_short": [False, False, False],
            },
            index=index,
        )
        model = ExecutionModel(
            profile="maker_tp",
            taker_fee=0.0005,
            maker_fee=0.0002,
            market_slippage=0.0005,
        )
        result = simulate(
            signals,
            self.risk,
            allow_longs=True,
            allow_shorts=False,
            execution_model=model,
        )
        self.assertEqual(len(result.trades), 1)
        trade = result.trades.iloc[0]
        self.assertEqual(trade["exit_reason"], "stop_loss")
        self.assertGreater(trade["slippage_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()
