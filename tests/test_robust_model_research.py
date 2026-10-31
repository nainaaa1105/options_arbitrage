import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_model_research import purged_folds, qualifies_selective, selective_metrics


class ResearchProtocolTests(unittest.TestCase):
    def test_purge_uses_future_candle_availability_not_only_its_start(self):
        ts = pd.bdate_range("2026-01-01 10:00", periods=12, tz="Asia/Kolkata")
        d = pd.DataFrame({"timestamp": ts, "future_timestamp_30m": ts + pd.Timedelta(minutes=30)})
        d.loc[3, "future_timestamp_30m"] = ts[4] - pd.Timedelta(minutes=1)
        _, a, _ = next(purged_folds(d, n_splits=2, gap_sessions=0))
        self.assertNotIn(3, a)

    def test_date_folds_have_session_gap_and_purge_crossing_labels(self):
        ts = pd.bdate_range("2026-01-01 10:00", periods=12, tz="Asia/Kolkata")
        d = pd.DataFrame({"timestamp": ts, "future_timestamp_30m": ts + pd.Timedelta(minutes=30)})
        d.loc[0, "future_timestamp_30m"] = ts[-1]
        for _, a, b in purged_folds(d, n_splits=2, gap_sessions=1):
            self.assertNotIn(0, a)
            self.assertLess(a[-1] + 1, b[0])
            self.assertTrue((d.iloc[a].future_timestamp_30m < d.iloc[b].timestamp.min()).all())

    def test_negative_dominated_95_percent_accuracy_is_not_two_class_success(self):
        y = np.zeros(1000, dtype=int)
        p = np.full(1000, .01)
        p[:50] = .99
        dates = np.tile(np.arange(10), 100)
        metrics = selective_metrics(y, p, dates, .4)
        self.assertEqual(metrics["selective_accuracy"], .95)
        self.assertFalse(qualifies_selective(metrics, .93))

    def test_both_classes_with_sample_and_session_support_can_qualify(self):
        y = np.tile([0, 1], 100)
        p = np.where(y == 1, .99, .01)
        dates = np.repeat(np.arange(10), 20)
        metrics = selective_metrics(y, p, dates, .4)
        self.assertTrue(qualifies_selective(metrics, .94))


if __name__ == "__main__":
    unittest.main()
