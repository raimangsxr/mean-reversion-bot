import unittest

from src.risk.manager import RiskManager


class RiskManagerTests(unittest.TestCase):
    def setUp(self):
        self.risk = RiskManager(
            risk_per_trade=0.005,
            max_position_pct=0.10,
            atr_multiplier=1.5,
            risk_reward_ratio=1.8,
            cash_buffer_pct=0.05,
        )

    def test_long_plan(self):
        plan = self.risk.build_plan(
            side="long",
            account_equity=10_000,
            free_margin=10_000,
            expected_entry_price=100_000,
            atr=1_000,
            leverage=2,
        )
        self.assertAlmostEqual(plan.stop_distance, 1_500)
        self.assertAlmostEqual(plan.stop_price, 98_500)
        self.assertAlmostEqual(plan.take_profit_price, 102_700)
        self.assertLessEqual(plan.notional, 1_000)
        self.assertAlmostEqual(plan.required_margin, plan.notional / 2)

    def test_short_plan_is_symmetric(self):
        plan = self.risk.build_plan(
            side="short",
            account_equity=10_000,
            free_margin=10_000,
            expected_entry_price=100_000,
            atr=1_000,
            leverage=2,
        )
        self.assertAlmostEqual(plan.stop_price, 101_500)
        self.assertAlmostEqual(plan.take_profit_price, 97_300)
        self.assertLessEqual(plan.estimated_risk, plan.risk_budget + 1e-9)

    def test_leverage_does_not_raise_notional_when_other_caps_bind(self):
        low = self.risk.build_plan(
            side="long",
            account_equity=10_000,
            free_margin=10_000,
            expected_entry_price=100_000,
            atr=1_000,
            leverage=2,
        )
        high = self.risk.build_plan(
            side="long",
            account_equity=10_000,
            free_margin=10_000,
            expected_entry_price=100_000,
            atr=1_000,
            leverage=20,
        )
        self.assertAlmostEqual(low.notional, high.notional)


if __name__ == "__main__":
    unittest.main()
