from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from synthetic_demo import generate

REQUIRED = {"timestamp", "trade_date", "expiry", "instrument_key", "strike", "option_type", "lot_size", "open", "high", "low", "close",
            "volume", "open_interest", "nifty_spot", "feature_available_at", "spot_timestamp", "spot_available_at"}


class SyntheticDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        cls.master_path, cls.daily_path = generate(Path(cls.tmp.name) / "a", expiries=1, sessions_per_expiry=2, seed=3, strikes_each_side=2)
        cls.master = pd.read_parquet(cls.master_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_schema_matches_what_the_pipeline_reads_and_has_no_gaps(self):
        m = self.master
        self.assertTrue(REQUIRED <= set(m.columns))
        self.assertFalse(m.duplicated(["instrument_key", "timestamp"]).any())
        self.assertFalse(m[list(REQUIRED)].isna().any().any())
        self.assertEqual(m.groupby(["instrument_key", "trade_date"]).size().unique().tolist(), [75])
        self.assertEqual(m.instrument_key.nunique(), 2 * 5)                      # CE and PE at five strikes

    def test_prices_are_valid_option_prices_on_the_tick_grid(self):
        m = self.master
        intrinsic = np.where(m.option_type == "CE", np.maximum(m.nifty_spot - m.strike, 0), np.maximum(m.strike - m.nifty_spot, 0))
        self.assertTrue((m.close >= intrinsic - 1e-9).all())
        self.assertTrue(np.allclose(m.close / .05, np.round(m.close / .05), atol=1e-6))
        self.assertTrue((m.low <= m[["open", "close"]].min(axis=1) + 1e-9).all() and (m.high >= m[["open", "close"]].max(axis=1) - 1e-9).all())

    def test_features_are_available_only_when_the_bar_has_closed(self):
        m = self.master
        self.assertTrue((m.feature_available_at - m.timestamp == pd.Timedelta(minutes=5)).all())
        self.assertTrue((m.spot_available_at == m.feature_available_at).all())
        self.assertTrue((m.spot_timestamp < m.spot_available_at).all())

    def test_generation_is_deterministic_for_a_seed_and_differs_across_seeds(self):
        with TemporaryDirectory() as folder:
            again, _ = generate(Path(folder) / "b", expiries=1, sessions_per_expiry=2, seed=3, strikes_each_side=2)
            other, _ = generate(Path(folder) / "c", expiries=1, sessions_per_expiry=2, seed=4, strikes_each_side=2)
            pd.testing.assert_frame_equal(pd.read_parquet(again), self.master)
            self.assertFalse(np.allclose(pd.read_parquet(other).close.to_numpy(), self.master.close.to_numpy()))

    def test_daily_history_provides_a_volatility_warm_up_window(self):
        daily = pd.read_parquet(self.daily_path)
        first_session = pd.Timestamp(self.master.trade_date.min(), tz="Asia/Kolkata")
        self.assertGreaterEqual((pd.to_datetime(daily.timestamp) < first_session).sum(), 25)


if __name__ == "__main__":
    unittest.main()
