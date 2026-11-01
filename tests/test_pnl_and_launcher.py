from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import run_paper_day
from accuracy_v2 import pnl_experiment as pnl
from deep_hedging.config import HedgeConfig

FREE = HedgeConfig(option_half_spread=0., option_slippage=0., brokerage_per_order=0., option_exchange_rate=0., sebi_rate=0., option_ipft_rate=0.,
                   option_stamp_rate=0., gst_rate=0., option_stt_override=0.)
T0 = pd.Timestamp("2026-09-30 10:00", tz="Asia/Kolkata")


class RoundTripTests(unittest.TestCase):
    def test_without_costs_pnl_is_the_price_move_times_quantity(self):
        value = pnl.round_trip_pnl(100.0, 103.0, T0, T0 + pd.Timedelta(minutes=30), 75, FREE)
        self.assertAlmostEqual(value, 4 * 75 * 3.0, places=6)

    def test_costs_always_reduce_pnl_and_a_flat_trade_loses(self):
        free = pnl.round_trip_pnl(100.0, 100.0, T0, T0 + pd.Timedelta(minutes=30), 75, FREE)
        costly = pnl.round_trip_pnl(100.0, 100.0, T0, T0 + pd.Timedelta(minutes=30), 75, HedgeConfig())
        self.assertAlmostEqual(free, 0.0, places=6)
        self.assertLess(costly, -100.0)

    def test_unexecutable_prices_give_nan_not_an_error(self):
        self.assertTrue(np.isnan(pnl.round_trip_pnl(0.0, 5.0, T0, T0 + pd.Timedelta(minutes=30), 75, FREE)))

    def test_attach_pnl_pairs_entry_and_exit_bars_exactly_and_leaves_gaps_nan(self):
        step = pd.Timedelta(minutes=5)
        decision = T0
        bars = pd.DataFrame({"instrument_key": "A", "timestamp": [decision + step, decision + step + pd.Timedelta(minutes=30)], "open": [100., 104.], "lot_size": 75})
        frame = pd.DataFrame({"instrument_key": ["A", "B"], "timestamp": [decision, decision]})
        out = pnl.attach_pnl(frame, bars, FREE)
        self.assertAlmostEqual(out.net_pnl.iloc[0], 4 * 75 * 4.0, places=6)
        self.assertTrue(np.isnan(out.net_pnl.iloc[1]))                      # contract B has no bars: nothing is filled in


class BootstrapTests(unittest.TestCase):
    def test_interval_brackets_the_mean_and_is_reproducible(self):
        rng = np.random.default_rng(0)
        values, days = rng.normal(5, 20, 400), np.repeat(np.arange(40), 10)
        mean, lo, hi = pnl.day_bootstrap(values, days, repeats=300)
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertEqual((mean, lo, hi), pnl.day_bootstrap(values, days, repeats=300))

    def test_paired_difference_of_a_set_with_itself_is_zero(self):
        values, days = np.arange(60, dtype=float), np.repeat(np.arange(12), 5)
        point, lo, hi = pnl.paired_difference(values, days, values, days, repeats=200)
        self.assertEqual((point, lo, hi), (0.0, 0.0, 0.0))


class LauncherTests(unittest.TestCase):
    def test_latest_master_picks_the_newest_dated_snapshot(self):
        with TemporaryDirectory() as folder:
            for name in ("NSE_20261031.json.gz", "NSE_20261101.json.gz"):
                (Path(folder) / name).write_bytes(b"x")
            with mock.patch.object(run_paper_day, "MASTER_DIR", Path(folder)):
                self.assertEqual(run_paper_day.latest_master().name, "NSE_20261101.json.gz")
            with mock.patch.object(run_paper_day, "MASTER_DIR", Path(folder) / "empty"):
                self.assertIn("missing", run_paper_day.latest_master().name)

    def test_preflight_reports_missing_and_stale_masters_offline(self):
        with TemporaryDirectory() as folder:
            missing = Path(folder) / "NSE_x.json.gz"
            self.assertTrue(any("missing" in p for p in run_paper_day.preflight(missing, live=False)))
            old = Path(folder) / "NSE_old.json.gz"
            old.write_bytes(b"x")
            import os
            long_ago = pd.Timestamp.now().timestamp() - 3 * 86400
            os.utime(old, (long_ago, long_ago))
            self.assertTrue(any("days old" in p for p in run_paper_day.preflight(old, live=False)))
            fresh = Path(folder) / "NSE_new.json.gz"
            fresh.write_bytes(b"x")
            self.assertEqual(run_paper_day.preflight(fresh, live=False), [])


if __name__ == "__main__":
    unittest.main()
