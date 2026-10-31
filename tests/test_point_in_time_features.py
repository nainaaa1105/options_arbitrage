"""Regression tests for point-in-time volatility and reference IV features."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_reference_iv import add_reference_iv_surface, calculate_reference_iv
from build_volatility_features import (
    add_historical_volatility_features,
    calculate_daily_hv,
    make_daily_spot,
)
from train_models import build_point_in_time_context


def _spot_rows(prices):
    dates = pd.bdate_range("2025-01-01", periods=len(prices))
    rows = []
    for date, price in zip(dates, prices):
        for minute in (0, 5):
            rows.append(
                {
                    "timestamp": date + pd.Timedelta(hours=10, minutes=minute),
                    "trade_date": date.date(),
                    "nifty_spot": price + minute,
                    "implied_volatility": 0.2,
                }
            )
    return pd.DataFrame(rows)


def _surface(expiry, scale=1.0, target_iv=0.3):
    xs = [-0.02, -0.01, 0.0, 0.01, 0.02]
    ivs = [0.1, 0.2, target_iv, 0.4, 0.5]
    return pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2025-01-02 10:00")] * len(xs),
            "expiry": [pd.Timestamp(expiry)] * len(xs),
            "option_type": ["CE"] * len(xs),
            "strike": [25000 * np.exp(x) for x in xs],
            "log_moneyness": xs,
            "implied_volatility": [iv * scale for iv in ivs],
        }
    )


class PointInTimeFeatureTests(unittest.TestCase):
    def test_context_features_use_only_prior_candles_and_current_surface(self):
        times = pd.to_datetime([
            "2025-01-06 09:15", "2025-01-06 09:20", "2025-01-06 09:15",
            "2025-01-06 09:20", "2025-01-06 09:25",
        ]).tz_localize("Asia/Kolkata")
        context = pd.DataFrame({
            "timestamp": times,
            "trade_date": [pd.Timestamp("2025-01-06").date()] * 5,
            "instrument_key": ["A", "A", "B", "B", "A"],
            "market_price": [10.0, 11.0, 20.0, 21.0, 12.0],
            "reference_price_deviation": [1.0, 2.0, 3.0, 4.0, 5.0],
            "volume": [10, 20, 30, 40, 50],
            "implied_volatility": [0.2, 0.21, 0.3, 0.31, 0.22],
            "strike": [100, 100, 110, 110, 100],
            "nifty_spot": [105] * 5,
        })
        features = build_point_in_time_context(context)
        first_a = features.loc[
            (features.instrument_key == "A")
            & (features.timestamp == times[0])
        ].iloc[0]
        self.assertEqual(first_a.mkt_mean_abs_dev, 2.0)
        self.assertTrue(np.isnan(first_a.lag1_price))

        changed = context.copy()
        changed.loc[changed.timestamp == times[-1], "market_price"] = 9999
        changed_features = build_point_in_time_context(changed)
        first_a_changed = changed_features.loc[
            (changed_features.instrument_key == "A")
            & (changed_features.timestamp == times[0])
        ].iloc[0]
        self.assertEqual(first_a.mkt_mean_abs_dev, first_a_changed.mkt_mean_abs_dev)
        second_a = changed_features.loc[
            (changed_features.instrument_key == "A")
            & (changed_features.timestamp == times[1])
        ].iloc[0]
        self.assertEqual(second_a.lag1_price, 10.0)

    def test_hv_uses_completed_prior_days_only(self):
        data = _spot_rows([100, 101, 99, 103, 102, 105, 100, 110])
        result = add_historical_volatility_features(data, windows=(2,))
        target_day = pd.Timestamp("2025-01-06").date()
        target = result.loc[
            result.trade_date == target_day, "historical_volatility_2d"
        ].iloc[0]

        changed = data.copy()
        changed["nifty_spot"] = changed["nifty_spot"].astype(float)
        changed.loc[changed.trade_date >= target_day, "nifty_spot"] *= 1.7
        changed_result = add_historical_volatility_features(changed, windows=(2,))
        changed_target = changed_result.loc[
            changed_result.trade_date == target_day, "historical_volatility_2d"
        ].iloc[0]
        self.assertEqual(target, changed_target)
        self.assertTrue(np.isfinite(target))

    def test_hv_is_missing_until_full_prior_return_window_exists(self):
        data = _spot_rows([100, 101, 102, 103, 104, 105, 106])
        result = add_historical_volatility_features(data, windows=(3,))
        values = result.groupby("trade_date")["historical_volatility_3d"].first()
        self.assertTrue(values.iloc[:4].isna().all())
        self.assertTrue(np.isfinite(values.iloc[4]))

    def test_missing_daily_spot_does_not_bridge_return_window(self):
        data = _spot_rows([100, 101, 102, 103, 104, 105, 106])
        missing_day = pd.Timestamp("2025-01-03").date()
        data.loc[data.trade_date == missing_day, "nifty_spot"] = np.nan
        daily = make_daily_spot(data)
        self.assertTrue(daily.loc[daily.trade_date == missing_day, "nifty_spot"].isna().all())
        hv = calculate_daily_hv(daily, windows=(2,))
        by_date = hv.set_index("trade_date")["historical_volatility_2d"]
        self.assertTrue(np.isnan(by_date.loc[pd.Timestamp("2025-01-06").date()]))
        self.assertTrue(np.isnan(by_date.loc[pd.Timestamp("2025-01-08").date()]))
        self.assertTrue(np.isfinite(by_date.loc[pd.Timestamp("2025-01-09").date()]))

    def test_reference_iv_interpolates_and_excludes_own_iv(self):
        group = _surface("2025-02-01")
        result = calculate_reference_iv(group)
        center = result.loc[result.log_moneyness == 0.0, "reference_iv"].iloc[0]
        self.assertAlmostEqual(center, 0.3)

        changed = group.copy()
        changed.loc[changed.log_moneyness == 0.0, "implied_volatility"] = 4.5
        changed_center = calculate_reference_iv(changed).loc[
            changed.log_moneyness == 0.0, "reference_iv"
        ].iloc[0]
        self.assertAlmostEqual(changed_center, center)

    def test_reference_surface_isolates_expiries(self):
        first = _surface("2025-02-01")
        second = _surface("2025-03-01", scale=2.0)
        combined = pd.concat([first, second], ignore_index=True)
        baseline = add_reference_iv_surface(combined)

        changed = combined.copy()
        changed.loc[
            changed.expiry == pd.Timestamp("2025-03-01"), "implied_volatility"
        ] *= 1.5
        altered = add_reference_iv_surface(changed)
        mask = baseline.expiry == pd.Timestamp("2025-02-01")
        np.testing.assert_allclose(
            baseline.loc[mask, "reference_iv"],
            altered.loc[mask, "reference_iv"],
            equal_nan=True,
        )

    def test_reference_surface_handles_insufficient_strikes_and_far_extrapolation(self):
        sparse = _surface("2025-02-01").iloc[:2].copy()
        result = calculate_reference_iv(sparse)
        self.assertTrue(result.reference_iv.isna().all())

        edge = _surface("2025-02-01").iloc[:4].copy()
        edge.loc[edge.index[-1], "log_moneyness"] = 0.2
        result = calculate_reference_iv(edge)
        self.assertEqual(
            result.loc[edge.index[-1], "reference_iv_method"],
            "extrapolation_too_far",
        )
        self.assertTrue(np.isnan(result.loc[edge.index[-1], "reference_iv"]))


if __name__ == "__main__":
    unittest.main()
