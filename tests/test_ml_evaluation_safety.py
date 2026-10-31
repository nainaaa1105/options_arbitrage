"""Tests that future, label-derived, or sample-dependent fields cannot train."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from train_models import (
    _feature_columns,
    _classification_metrics,
    _operating_point_table,
    _validation_thresholds,
    build_features,
    build_point_in_time_context,
    _make_estimator,
)


class MLEvaluationSafetyTests(unittest.TestCase):
    def test_lightgbm_subsample_has_effective_sampling_frequency(self):
        model = _make_estimator("lightgbm", params={"subsample": .75})
        self.assertEqual(model.get_params()["model__subsample_freq"], 1)

    def test_expiry_features_survive_metadata_filter(self):
        frame = pd.DataFrame({
            "expiry": [20260106], "time_to_expiry": [0.01],
            "days_to_expiry": [3.65], "is_near_expiry": [False],
            "future_expiry_price": [42.0],
        })
        self.assertEqual(
            _feature_columns(frame),
            ["time_to_expiry", "days_to_expiry", "is_near_expiry"],
        )

    def test_feature_values_are_invariant_to_input_order_and_batch_size(self):
        times = pd.to_datetime([
            "2026-01-06 10:00", "2026-01-06 09:20", "2026-01-06 09:15",
        ]).tz_localize("Asia/Kolkata")
        d = pd.DataFrame({"timestamp": times, "instrument_key": ["B", "A", "A"]})
        d["trade_date"] = d.timestamp.dt.date
        d["expiry"] = pd.Timestamp("2026-01-06")
        d["option_type"] = "CE"
        for name in [
            "market_price", "reference_price_deviation", "volume", "implied_volatility",
            "absolute_price_deviation", "strike", "nifty_spot", "log_moneyness",
            "parity_residual", "parity_residual_pct", "iv_surface_residual",
            "iv_surface_residual_pct", "reference_iv", "delta", "gamma", "vega",
            "theta", "open_interest", "iv_minus_hv_5d", "iv_minus_hv_10d",
            "iv_minus_hv_20d", "iv_hv_ratio_5d", "iv_hv_ratio_10d",
            "iv_hv_ratio_20d", "historical_volatility_5d", "historical_volatility_20d",
        ]:
            d[name] = 1.0
        d["time_to_expiry"] = 0.25 / 365
        context = build_point_in_time_context(d)
        with patch("train_models._load_point_in_time_context", return_value=context):
            first = build_features({"sample": d})["sample"]
            shuffled = build_features({"sample": d.sample(frac=1, random_state=7)})["sample"]
            singleton = build_features({"sample": d.iloc[[0]]})["sample"]
        expected = first.timestamp.dt.hour * 60 + first.timestamp.dt.minute - 555
        pd.testing.assert_series_equal(first.intraday_minute, expected, check_names=False)
        pd.testing.assert_frame_equal(first, shuffled)
        self.assertEqual(singleton.intraday_minute.iloc[0], 45)
        self.assertTrue((first.days_to_expiry == 0.25).all())
        self.assertTrue((first.is_0dte == 1).all())

    def test_future_and_sample_dependent_fields_are_excluded(self):
        frame = pd.DataFrame({
            "spot": [1.0],
            "target_converged_30m": [1],
            "future_timestamp_30m": [pd.Timestamp("2025-01-01")],
            "future_deviation_30m": [0.1],
            "label_valid_30m": [True],
            "lag1_price": [0.9],
            "mkt_total_vol": [100],
            "first_passage": [1], "endpoint": [0], "net_pnl_rupees": [50.],
            "observed_path_points": [6], "exit_price": [100.],
            "signal_eligible": [True],
        })
        columns = _feature_columns(frame)
        self.assertEqual(columns, ["spot", "signal_eligible"])

    def test_metric_summary_uses_the_requested_threshold(self):
        result = _classification_metrics([0, 0, 1, 1], [0.1, 0.8, 0.8, 0.9], 0.75)
        self.assertEqual(result["threshold"], 0.75)
        self.assertEqual((result["tn"], result["fp"], result["fn"], result["tp"]), (1, 1, 0, 2))
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertIn("brier_score", result)
        self.assertIn("log_loss", result)
        self.assertIn("expected_calibration_error_10bin", result)

    def test_operating_point_report_includes_fixed_cutoffs_and_validation_selection(self):
        y_validation = [0, 0, 1, 1, 1, 0]
        validation_probabilities = [0.1, 0.4, 0.45, 0.7, 0.8, 0.6]
        thresholds = _validation_thresholds(y_validation, validation_probabilities)
        report = _operating_point_table(
            [0, 0, 1, 1], [0.1, 0.8, 0.8, 0.9], thresholds
        ).set_index("decision_policy")

        self.assertIn("fixed_0.50", report.index)
        self.assertIn("fixed_0.60", report.index)
        self.assertIn("validation_max_f1", report.index)
        self.assertEqual(report.loc["fixed_0.50", "threshold_selected_on"], "predeclared")
        self.assertEqual(report.loc["validation_max_f1", "threshold_selected_on"], "validation")
        self.assertEqual(report.loc["always_negative_baseline", "accuracy"], 0.5)
        self.assertEqual(report.loc["always_negative_baseline", "roc_auc"], 0.5)
        self.assertEqual(report.loc["always_positive_baseline", "roc_auc"], 0.5)
        self.assertAlmostEqual(report.loc["always_negative_baseline", "brier_score"], 0.5)
        self.assertIn("matthews_corrcoef", report.columns)


if __name__ == "__main__":
    unittest.main()
