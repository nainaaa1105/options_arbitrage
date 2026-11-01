from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import chain_features
from synthetic_demo import generate

STAGE = ROOT / "data/research/expanded_20261030"
DAILY = ROOT / "data/raw/upstox/nifty_daily_to_20261029.parquet"
COMPARE = ["implied_volatility", "reference_iv", "reference_theoretical_price", "reference_price_deviation", "parity_residual"]
KEY = ["instrument_key", "timestamp"]


class CausalityTests(unittest.TestCase):
    """A bar's features must not change when later bars are added to the table (no look-ahead)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = TemporaryDirectory()
        master_path, daily_path = generate(Path(cls.tmp.name), expiries=1, sessions_per_expiry=1, seed=5, strikes_each_side=3)
        cls.master = pd.read_parquet(master_path)
        cls.master["timestamp"] = pd.to_datetime(cls.master.timestamp)
        cls.daily = pd.read_parquet(daily_path)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_features_at_a_bar_are_identical_with_and_without_the_future(self):
        stamps = sorted(self.master.timestamp.unique())
        cut = pd.Timestamp(stamps[40])
        full = chain_features.run_stages(self.master, self.daily)
        part = chain_features.run_stages(self.master[self.master.timestamp <= cut], self.daily)
        a = full[full.timestamp == cut].set_index("instrument_key").sort_index()
        b = part[part.timestamp == cut].set_index("instrument_key").sort_index()
        self.assertGreater(len(a), 0)
        for column in COMPARE:
            np.testing.assert_allclose(a[column].to_numpy(float), b[column].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True, err_msg=column)
        self.assertTrue((a.signal_eligible.astype(bool) == b.signal_eligible.astype(bool)).all())

    def test_service_rejects_out_of_order_and_multi_timestamp_pushes(self):
        service = chain_features.ChainFeatureService(self.daily, None, None)
        first = self.master[self.master.timestamp == self.master.timestamp.min()]
        with self.assertRaises(ValueError):
            service.push(self.master[self.master.timestamp.isin(self.master.timestamp.unique()[:2])])
        service.table, service.session = first, first.timestamp.iloc[0].date()
        with self.assertRaises(ValueError):
            service.push(first)


@unittest.skipUnless((STAGE / "data/processed/options_parity_features.parquet").exists() and DAILY.exists(), "recorded research data not present")
class StoredDataParityTests(unittest.TestCase):
    def test_recorded_session_matches_the_batch_pipeline_exactly(self):
        day = pd.Timestamp("2025-12-15").date()
        master = pd.read_parquet(STAGE / "data/processed/master_options.parquet", filters=[("trade_date", "==", str(day))])
        master["timestamp"] = pd.to_datetime(master.timestamp)
        if master.trade_date.dtype == object and master.empty:
            self.skipTest("session not in this dataset")
        live = chain_features.run_stages(master, pd.read_parquet(DAILY))
        stored = pd.read_parquet(STAGE / "data/processed/options_parity_features.parquet", columns=KEY + COMPARE, filters=[("trade_date", "==", day)])
        for frame in (live, stored):
            frame["timestamp"] = pd.to_datetime(frame.timestamp)
        both = live.merge(stored, on=KEY, suffixes=("", "_s"), validate="one_to_one")
        self.assertEqual(len(both), len(stored))
        for column in COMPARE:
            np.testing.assert_allclose(both[column].to_numpy(float), both[f"{column}_s"].to_numpy(float), rtol=0, atol=1e-9, equal_nan=True, err_msg=column)


if __name__ == "__main__":
    unittest.main()
