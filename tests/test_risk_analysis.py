"""Hand-checkable risk metrics with zero-trade sessions included."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from risk_analysis import StrategyRiskAnalyzer


class RiskAnalysisTests(unittest.TestCase):
    def test_zero_trade_days_and_drawdown_are_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trades = pd.DataFrame({
                "timestamp": pd.to_datetime(["2025-01-06", "2025-01-08"]),
                "exit_timestamp": pd.to_datetime(["2025-01-06 10:00", "2025-01-08 10:00"]),
                "gross_pnl": [12.0, -3.0],
                "net_pnl": [10.0, -5.0],
                "total_costs": [2.0, 2.0],
            })
            trades.to_parquet(root / "trades.parquet", index=False)
            pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-06", "2025-01-08"])}).to_csv(
                root / "equity.csv", index=False
            )
            pd.DataFrame({"trade_date": ["2025-01-06", "2025-01-07", "2025-01-08"]}).to_parquet(
                root / "calendar.parquet", index=False
            )
            analyzer = StrategyRiskAnalyzer(
                root / "trades.parquet", root / "equity.csv", root / "risk.csv",
                initial_capital=100.0, risk_free_rate=0.0,
                calendar_file=root / "calendar.parquet",
            )
            metrics = analyzer.run_analysis().set_index("Metric").Value
            equity = pd.read_csv(root / "equity.csv")
            self.assertEqual(len(equity), 3)
            self.assertEqual(equity.net_pnl.tolist(), [10.0, 0.0, -5.0])
            self.assertAlmostEqual(float(metrics["Gross P&L (INR)"]), 9.0)
            self.assertAlmostEqual(float(metrics["Trading Costs (INR)"]), 4.0)
            self.assertAlmostEqual(float(metrics["Net P&L (INR)"]), 5.0)
            self.assertAlmostEqual(float(metrics["Maximum Drawdown (INR)"]), 5.0)
            self.assertEqual(int(metrics["Trading Days"]), 3)


if __name__ == "__main__":
    unittest.main()
