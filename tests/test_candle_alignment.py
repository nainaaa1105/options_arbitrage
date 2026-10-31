import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from candle_alignment import align_option_spot_candles


class CandleAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-07-09 09:15", tz="Asia/Kolkata")
        self.options = pd.DataFrame({"timestamp": [self.start], "close": [20.0]})

    def test_five_minute_option_matches_last_completed_minute_at_its_close(self):
        spot = pd.DataFrame({
            "timestamp": [self.start, self.start + pd.Timedelta(minutes=4),
                          self.start + pd.Timedelta(minutes=5)],
            "close": [100.0, 104.0, 105.0],
        })
        result = align_option_spot_candles(self.options, spot).iloc[0]
        self.assertEqual(result.nifty_spot, 104.0)
        self.assertEqual(result.feature_available_at, self.start + pd.Timedelta(minutes=5))
        self.assertEqual(result.feature_available_at, result.spot_available_at)
        self.assertEqual(result.timestamp, self.start)

    def test_missing_exact_minute_does_not_substitute_stale_or_future_close(self):
        spot = pd.DataFrame({
            "timestamp": [self.start + pd.Timedelta(minutes=3),
                          self.start + pd.Timedelta(minutes=5)],
            "close": [103.0, 105.0],
        })
        result = align_option_spot_candles(self.options, spot)
        self.assertTrue(result.nifty_spot.isna().all())

    def test_duplicate_spot_timestamps_raise(self):
        spot = pd.DataFrame({"timestamp": [self.start, self.start], "close": [100.0, 101.0]})
        with self.assertRaisesRegex(ValueError, "Duplicate spot"):
            align_option_spot_candles(self.options, spot)

    def test_matching_five_minute_spot_bars_can_use_same_start_timestamp(self):
        spot = pd.DataFrame({"timestamp": [self.start], "close": [104.], "bar_minutes": [5]})
        result = align_option_spot_candles(self.options, spot, spot_minutes=5).iloc[0]
        self.assertEqual(result.nifty_spot, 104.)
        self.assertEqual(result.feature_available_at, result.spot_available_at)
        with self.assertRaisesRegex(ValueError, "provenance"):
            align_option_spot_candles(self.options, spot, spot_minutes=1)


if __name__ == "__main__":
    unittest.main()
