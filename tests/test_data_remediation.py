import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from unittest.mock import patch
from datetime import date
import io

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_volatility_features import calculate_daily_hv, add_historical_volatility_features
from collect_spot_history import request_windows
from data_remediation_study import policy_rows_weights, fit_policy
from pricing_research import black76_price, parity_forward_excluding_strike, robust_loo_total_variance
from black_scholes import black_scholes_price
from quote_capture import normalize_snapshot
from research_targets import session_horizon_eligible, build_path_targets, executable_cashflow_targets
from generate_trade_signals import generate_actionable_trade_signals


class RemediationTests(unittest.TestCase):
    def test_complete_daily_history_supplies_hv_across_option_coverage_gap(self):
        dates = pd.bdate_range("2026-01-01", "2026-04-01")
        daily = pd.DataFrame({"trade_date": dates.date, "nifty_spot": 100 + np.arange(len(dates)) ** 1.1})
        options = pd.DataFrame({"timestamp": [pd.Timestamp("2026-04-01 10:00")],
                                "trade_date": [date(2026, 4, 1)], "nifty_spot": [200.], "implied_volatility": [.2]})
        result = add_historical_volatility_features(options, daily_spot=daily)
        self.assertTrue(np.isfinite(result.historical_volatility_20d.iloc[0]))

    def test_minute_requests_are_split_at_calendar_month_boundaries(self):
        windows = request_windows(date(2026, 1, 30), date(2026, 3, 2), "minutes")
        self.assertEqual(windows, [(date(2026, 1, 30), date(2026, 1, 31)),
                                   (date(2026, 2, 1), date(2026, 2, 28)),
                                   (date(2026, 3, 1), date(2026, 3, 2))])

    def test_existing_raw_collection_is_preserved_before_any_api_request(self):
        from collect_expiry import collect_expiry
        with TemporaryDirectory() as temp:
            existing = Path(temp) / "expiry_2026-07-14.parquet"
            existing.write_bytes(b"preserve")
            with patch("collect_expiry.UPSTOX_RAW_DIR", Path(temp)), patch("collect_expiry.prepare_expiry") as request:
                with self.assertRaises(FileExistsError):
                    collect_expiry("2026-07-14")
                request.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"preserve")

    def test_relative_input_policy_does_not_depend_on_absolute_index_level(self):
        x = np.tile(np.linspace(-1, 1, 20), 8)
        d = pd.DataFrame({"timestamp": np.repeat(pd.bdate_range("2026-01-01", periods=8), 20),
                          "nifty_spot": np.arange(len(x)) + 25000., "relative": x})
        model, _ = fit_policy(d, ["nifty_spot", "relative"], (x > 0).astype(int),
                              {"balanced": False, "drop_absolute_index_levels": True})
        original = model.predict_proba(d[["nifty_spot", "relative"]])
        changed = d[["nifty_spot", "relative"]].copy()
        changed.nifty_spot += 10000
        np.testing.assert_allclose(original, model.predict_proba(changed))

    def test_late_high_probability_observation_cannot_become_trade(self):
        times = pd.to_datetime(["2026-07-09 14:55", "2026-07-09 15:00"]).tz_localize("Asia/Kolkata")
        d = pd.DataFrame({"timestamp": times, "option_type": "CE", "market_price": 10.,
                          "theoretical_price": 12., "price_deviation": -2.,
                          "predicted_probability": .99, "probability_threshold": .5})
        with TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            source, destination = Path(temp) / "input.parquet", Path(temp) / "output.parquet"
            d.to_parquet(source)
            result = generate_actionable_trade_signals(source, destination)
        self.assertEqual(result.trade_signal.tolist(), ["BUY", "HOLD"])

    def test_long_calendar_gap_resets_hv_until_complete_new_history(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05", "2026-04-01",
                                "2026-04-02", "2026-04-03", "2026-04-06"])
        daily = pd.DataFrame({"trade_date": dates.date, "nifty_spot": [100, 101, 103, 150, 151, 153, 154]})
        hv = calculate_daily_hv(daily, windows=(2,))
        self.assertTrue(hv.historical_volatility_2d.iloc[3:6].isna().all())
        self.assertTrue(np.isfinite(hv.historical_volatility_2d.iloc[6]))

    def test_session_horizon_uses_bar_close_and_does_not_allow_late_or_preopen_signals(self):
        ts = pd.Series(pd.to_datetime(["2026-07-09 14:55", "2026-07-09 15:00", "2026-07-09 08:00"]).tz_localize("Asia/Kolkata"))
        self.assertEqual(session_horizon_eligible(ts).tolist(), [True, False, False])

    def path_frame(self):
        ts = pd.date_range("2026-07-09 10:00", periods=7, freq="5min", tz="Asia/Kolkata")
        return pd.DataFrame({"timestamp": ts, "instrument_key": "A", "expiry": pd.Timestamp("2026-07-14"),
                             "signal_eligible": True, "reference_price_deviation": [10., 4., 3., 8., 9., 9., 9.]})

    def test_targets_retain_endpoint_and_first_passage_distinction(self):
        first = build_path_targets(self.path_frame()).iloc[0]
        self.assertEqual(first.endpoint, 0)
        self.assertEqual(first.first_passage, 1)
        self.assertEqual(first.sustained_two_bars, 1)
        self.assertTrue(first.complete_path)

    def test_missing_path_is_censored_not_a_negative_outcome(self):
        first = build_path_targets(self.path_frame().drop(index=2)).iloc[0]
        self.assertFalse(first.complete_path)
        self.assertTrue(pd.isna(first.endpoint))
        self.assertTrue(pd.isna(first.first_passage))

    def test_session_balancing_and_recency_depend_only_on_fit_rows(self):
        d = pd.DataFrame({"timestamp": pd.to_datetime(["2026-01-01"] * 4 + ["2026-01-02"] * 2)})
        pos, weights = policy_rows_weights(d, {"balanced": True})
        self.assertAlmostEqual(weights[:4].sum(), weights[4:].sum())
        pos, weights = policy_rows_weights(d, {"balanced": True, "half_life": 1})
        self.assertAlmostEqual(weights[4:].sum() / weights[:4].sum(), 2.)
        self.assertEqual(len(pos), len(d))

    def test_black76_matches_black_scholes_when_forward_uses_same_carry(self):
        s, k, r, t, v = 25000., 25100., .06, .1, .2
        for kind in ("CE", "PE"):
            self.assertAlmostEqual(black76_price(s * np.exp(r * t), k, t, v, r, kind),
                                   black_scholes_price(s, k, t, v, r, kind), places=8)

    def test_forward_estimate_does_not_use_the_target_pair(self):
        d = pd.DataFrame({"strike": [99., 100., 101., 102.], "call_mid": [3., 2., 1., .5],
                          "put_mid": [1., 1., 1., 1.5], "time_to_expiry": .1, "risk_free_rate": 0.})
        baseline = parity_forward_excluding_strike(d, 100.)
        d.loc[d.strike.eq(100), "call_mid"] = 999.
        self.assertEqual(baseline, parity_forward_excluding_strike(d, 100.))
        self.assertAlmostEqual(baseline, 101.)

    def test_robust_surface_excludes_own_iv_and_rejects_expiry_mixing(self):
        d = pd.DataFrame({"log_moneyness": np.linspace(-.03, .03, 7), "implied_volatility": .2,
                          "time_to_expiry": .1, "expiry": pd.Timestamp("2026-07-14")})
        baseline = robust_loo_total_variance(d).research_reference_iv.iloc[3]
        d.loc[3, "implied_volatility"] = 3.
        self.assertAlmostEqual(baseline, robust_loo_total_variance(d).research_reference_iv.iloc[3])
        d.loc[0, "expiry"] = pd.Timestamp("2026-07-21")
        with self.assertRaises(ValueError):
            robust_loo_total_variance(d)

    def test_cashflow_target_uses_actual_quantities_hedge_and_costs(self):
        d = pd.DataFrame({"signal_available_at": ["2026-07-09T09:20+05:30"],
                          "entry_time": ["2026-07-09T09:20+05:30"], "exit_time": ["2026-07-09T09:50+05:30"],
                          "option_side": [1], "option_quantity": [25], "option_entry_price": [100.],
                          "option_exit_price": [105.], "hedge_quantity": [-10],
                          "hedge_entry_price": [25000.], "hedge_exit_price": [25002.],
                          "hedge_instrument_key": ["NSE_FO|FUTURE"],
                          "total_costs_rupees": [50.], "cost_basis": ["measured"]})
        result = executable_cashflow_targets(d).iloc[0]
        self.assertEqual(result.net_pnl_rupees, 55.)
        self.assertEqual(result.profitable, 1)
        d["hedge_instrument_key"] = "NSE_INDEX|Nifty 50"
        self.assertFalse(executable_cashflow_targets(d).label_valid.iloc[0])
        d["hedge_instrument_key"] = "NSE_FO|FUTURE"
        d["entry_time"] = "2026-07-09T09:15+05:30"
        self.assertFalse(executable_cashflow_targets(d).label_valid.iloc[0])

    def test_quote_mid_is_not_replaced_with_ltp_when_depth_is_missing(self):
        payload = {"data": {"A": {"timestamp": "2026-07-09T10:00:00+05:30", "last_price": 999.,
                                   "depth": {"buy": [{"price": 10., "quantity": 100}],
                                             "sell": [{"price": 12., "quantity": 100}]}}}}
        row = normalize_snapshot(payload, "2026-07-09T10:00:01+05:30").iloc[0]
        self.assertEqual(row.mid, 11.)
        self.assertTrue(row.executable_quote_valid)
        payload["data"]["A"]["depth"]["sell"] = []
        row = normalize_snapshot(payload, "2026-07-09T10:00:01+05:30").iloc[0]
        self.assertTrue(pd.isna(row.mid))
        self.assertEqual(row.quality_reason, "missing_two_sided_depth")

    def test_stale_and_crossed_quotes_are_flagged(self):
        payload = {"data": {"A": {"timestamp": "2026-07-09T10:00:00+05:30",
                                   "depth": {"buy": [{"price": 12., "quantity": 100}],
                                             "sell": [{"price": 10., "quantity": 100}]}}}}
        row = normalize_snapshot(payload, "2026-07-09T10:00:01+05:30").iloc[0]
        self.assertEqual(row.quality_reason, "crossed_book")
        payload["data"]["A"]["depth"]["buy"][0]["price"] = 9.
        row = normalize_snapshot(payload, "2026-07-09T10:01:00+05:30").iloc[0]
        self.assertEqual(row.quality_reason, "stale_snapshot")


if __name__ == "__main__":
    unittest.main()
