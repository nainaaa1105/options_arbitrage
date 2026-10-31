"""Regression checks for the core pricing primitives."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from black_scholes import black_scholes_price
from greeks import calculate_greeks
from implied_volatility import implied_volatility
from parity import calculate_parity_residual


class FinancialCalculationTests(unittest.TestCase):
    def test_black_scholes_put_call_parity(self):
        spot, strike, time, rate, vol = 25_000.0, 25_100.0, 30 / 365, 0.06, 0.2
        call = black_scholes_price(spot, strike, time, vol, rate, "CE")
        put = black_scholes_price(spot, strike, time, vol, rate, "PE")
        residual = calculate_parity_residual(call, put, spot, strike, time, rate)
        self.assertAlmostEqual(float(residual), 0.0, places=8)

    def test_iv_solver_recovers_generating_volatility(self):
        price = black_scholes_price(25_000, 25_000, 30 / 365, 0.2, 0.06, "CE")
        iv, status = implied_volatility(price, 25_000, 25_000, 30 / 365, 0.06, "CE")
        self.assertEqual(status, "valid")
        self.assertAlmostEqual(iv, 0.2, places=6)

    def test_greek_signs_and_call_put_common_greeks(self):
        call = calculate_greeks(25_000, 25_000, 30 / 365, 0.2, 0.06, "CE")
        put = calculate_greeks(25_000, 25_000, 30 / 365, 0.2, 0.06, "PE")
        self.assertGreater(call[0], 0)
        self.assertLess(put[0], 0)
        self.assertAlmostEqual(call[1], put[1])
        self.assertAlmostEqual(call[2], put[2])
        self.assertGreater(call[1], 0)
        self.assertGreater(call[2], 0)

    def test_expired_option_returns_intrinsic_value(self):
        self.assertEqual(black_scholes_price(25_100, 25_000, 0, 0.2, 0.06, "CE"), 100)
        self.assertEqual(black_scholes_price(24_900, 25_000, 0, 0.2, 0.06, "PE"), 100)


if __name__ == "__main__":
    unittest.main()
