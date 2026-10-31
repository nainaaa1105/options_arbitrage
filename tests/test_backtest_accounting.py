"""Small, deterministic checks for next-bar fills and portfolio accounting."""

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backtest import OptionsArbitrageBacktester


BASE = pd.Timestamp("2025-01-06 09:15", tz="Asia/Kolkata")


def _market(key="OPT1", prices=None, lot_size=25):
    prices = prices or [10, 11, 12, 13, 14, 15, 16, 17, 18]
    return pd.DataFrame({
        "instrument_key": [key] * len(prices),
        "timestamp": [BASE + pd.Timedelta(minutes=5 * i) for i in range(len(prices))],
        "open": prices,
        "lot_size": [lot_size] * len(prices),
    })


def _signal(key="OPT1", timestamp=BASE, direction="BUY"):
    return {"instrument_key": key, "timestamp": timestamp, "trade_signal": direction,
            "market_price": 10.5, "trading_symbol": key, "option_type": "CE"}


def _engine(**kwargs):
    return OptionsArbitrageBacktester(
        "signals.parquet", "master.parquet", "out",
        initial_capital=kwargs.pop("initial_capital", 100_000),
        half_spread_points=kwargs.pop("half_spread_points", 0.0),
        slippage_points=kwargs.pop("slippage_points", 0.0),
        brokerage_per_order=kwargs.pop("brokerage_per_order", 0.0),
        exchange_txn_rate=kwargs.pop("exchange_txn_rate", 0.0),
        sebi_rate=kwargs.pop("sebi_rate", 0.0),
        stamp_duty_rate=kwargs.pop("stamp_duty_rate", 0.0),
        gst_rate=kwargs.pop("gst_rate", 0.0),
        **kwargs,
    )


class BacktestAccountingTests(unittest.TestCase):
    def test_entry_is_next_bar_open_and_exit_uses_lot_size(self):
        engine = _engine()
        trades, _, _ = engine.simulate(pd.DataFrame([_signal()]), _market())
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0].entry_price, 11)
        self.assertEqual(trades.iloc[0].exit_price, 17)
        self.assertEqual(trades.iloc[0].quantity, 25)
        self.assertEqual(trades.iloc[0].gross_pnl, 150)

    def test_missing_exit_is_skipped_instead_of_zero_move_trade(self):
        engine = _engine()
        trades, _, _ = engine.simulate(pd.DataFrame([_signal()]), _market(prices=[10, 11]))
        self.assertTrue(trades.empty)
        self.assertEqual(engine.skipped.get("missing_exit_bar"), 1)

    def test_same_contract_positions_cannot_overlap(self):
        engine = _engine()
        signals = pd.DataFrame([_signal(), _signal(timestamp=BASE + pd.Timedelta(minutes=5))])
        trades, _, _ = engine.simulate(signals, _market())
        self.assertEqual(len(trades), 1)
        self.assertEqual(engine.skipped.get("overlapping_same_contract"), 1)

    def test_capital_limit_blocks_unaffordable_second_position(self):
        engine = _engine(initial_capital=400, max_concurrent_positions=5)
        market = pd.concat([_market("OPT1"), _market("OPT2")], ignore_index=True)
        signals = pd.DataFrame([_signal("OPT1"), _signal("OPT2")])
        trades, _, _ = engine.simulate(signals, market)
        self.assertEqual(len(trades), 1)
        self.assertEqual(engine.skipped.get("capital_limit"), 1)

    def test_short_signal_is_excluded_without_margin_proxy(self):
        engine = _engine()
        trades, _, _ = engine.simulate(
            pd.DataFrame([_signal(direction="SELL")]), _market()
        )
        self.assertTrue(trades.empty)
        self.assertEqual(engine.skipped.get("short_margin_data_unavailable"), 1)

    def test_spread_slippage_and_statutory_costs_are_separate(self):
        engine = _engine(
            half_spread_points=0.15,
            slippage_points=0.10,
            stamp_duty_rate=0.00003,
        )
        trades, _, _ = engine.simulate(pd.DataFrame([_signal()]), _market())
        trade = trades.iloc[0]
        self.assertAlmostEqual(trade.entry_price, 11.25)
        self.assertAlmostEqual(trade.exit_price, 16.75)
        self.assertAlmostEqual(trade.gross_pnl, 150.0)
        self.assertAlmostEqual(trade.spread_cost, 7.5)
        self.assertAlmostEqual(trade.slippage, 5.0)
        self.assertAlmostEqual(trade.stt, 16.75 * 25 * 0.001)
        self.assertAlmostEqual(trade.stamp_duty, 11.25 * 25 * 0.00003)
        self.assertAlmostEqual(
            trade.net_pnl,
            trade.gross_pnl - trade.spread_cost - trade.slippage - trade.transaction_costs,
        )


if __name__ == "__main__":
    unittest.main()
