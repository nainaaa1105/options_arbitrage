import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from convergence import build_future_convergence_target


class ConvergenceLabelTests(unittest.TestCase):
    def frame(self, minutes, deviations):
        base = pd.Timestamp("2026-07-09 10:00", tz="Asia/Kolkata")
        return pd.DataFrame({
            "timestamp": [base + pd.Timedelta(minutes=m) for m in minutes],
            "instrument_key": ["A"] * len(minutes),
            "expiry": [pd.Timestamp("2026-07-14")] * len(minutes),
            "signal_eligible": [True] * len(minutes),
            "reference_price_deviation": deviations,
        })

    def first(self, data):
        output = build_future_convergence_target(data, 30)
        return output.loc[output.timestamp == data.timestamp.iloc[0]].iloc[0]

    def test_endpoint_is_not_temporary_intrawindow_convergence(self):
        row = self.first(self.frame([0, 10, 30], [10., 2., 9.]))
        self.assertTrue(row.label_valid_30m)
        self.assertFalse(row.converged_30m)
        self.assertAlmostEqual(row.deviation_reduction_30m, .1)

    def test_tolerance_boundary_and_missing_future(self):
        self.assertTrue(self.first(self.frame([0, 35], [10., 4.])).converged_30m)
        self.assertFalse(self.first(self.frame([0, 36], [10., 4.])).label_valid_30m)

    def test_expiry_and_session_boundaries_are_invalid(self):
        data = self.frame([0, 30], [10., 2.])
        data.loc[1, "expiry"] = pd.Timestamp("2026-07-21")
        self.assertFalse(self.first(data).label_valid_30m)
        data = self.frame([0, 30], [10., 2.])
        data["timestamp"] += pd.Timedelta(hours=13, minutes=45)
        self.assertFalse(self.first(data).label_valid_30m)

    def test_nonfinite_deviations_and_duplicate_keys_are_rejected(self):
        self.assertFalse(self.first(self.frame([0, 30], [10., np.inf])).label_valid_30m)
        data = self.frame([0, 0], [10., 2.])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_future_convergence_target(data, 30)


if __name__ == "__main__":
    unittest.main()
