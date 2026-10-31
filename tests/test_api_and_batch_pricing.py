import ssl
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from build_features import process_row
from diagnose_upstox import candle_quality, classify_response
from http_transport import verified_session
from vector_pricing import batch_pricing_features, batch_reference_prices
from black_scholes import black_scholes_price
from market_regime_features import market_table
from time_to_expiry import add_time_to_expiry
import upstox_api


MASTER = Path(__file__).resolve().parents[1] / "data/processed/master_options.parquet"


class APIAndBatchTests(unittest.TestCase):
    def test_market_features_are_prefix_invariant_and_missing_vix_stays_missing(self):
        ts = pd.date_range("2026-09-15 09:15", periods=20, freq="5min", tz="Asia/Kolkata")
        spot = pd.DataFrame({"timestamp": ts, "open": np.arange(20.) + 100,
                             "high": np.arange(20.) + 102, "low": np.arange(20.) + 99,
                             "close": np.arange(20.) + 101})
        vix = spot.copy()
        vix["close"] = np.linspace(10, 20, len(vix))
        full = market_table(spot, vix)
        prefix = market_table(spot.iloc[:14], vix.iloc[:14])
        pd.testing.assert_frame_equal(prefix, full.iloc[:14].reset_index(drop=True))
        missing = market_table(spot, vix.iloc[1:])
        self.assertTrue(pd.isna(missing.vix_level.iloc[0]))
        gap = market_table(spot.drop(index=4), vix)
        self.assertTrue(pd.isna(gap.loc[gap.timestamp.eq(ts[5]), "spot_return_5m"].iloc[0]))

    def test_reference_price_batch_matches_scalar_with_invalid_inputs(self):
        d = pd.DataFrame({"nifty_spot": [22000., 22000., 22000., -1.],
                          "strike": [22100., 22100., 22100., 22100.], "time_to_expiry": [.02, .02, 0., .02],
                          "reference_iv": [.2, .2, .2, .2], "risk_free_rate": [.06]*4,
                          "option_type": ["CE", "PE", "CE", "CE"]})
        result = batch_reference_prices(d)
        for i in (0, 1):
            self.assertAlmostEqual(result[i], black_scholes_price(22000, 22100, .02, .2, .06, d.option_type.iloc[i]), places=9)
        self.assertTrue(np.isnan(result[2:]).all())

    def test_system_trust_still_verifies_certificates_and_hostnames(self):
        session = verified_session()
        context = session.adapters["https://"].poolmanager.connection_pool_kw["ssl_context"]
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertTrue(session.verify)
        session.close()

    def test_200_with_error_body_is_not_success_and_empty_is_not_data(self):
        self.assertEqual(classify_response(200, {"status": "error"}), "invalid_success_payload")
        self.assertFalse(candle_quality([])["usable"])
        candles = [["2026-09-15T09:15:00+05:30", 100, 90, 80, 95, 1, 0]]
        self.assertEqual(candle_quality(candles, 5)["invalid_ohlc"], 1)

    def test_access_denial_is_not_retried(self):
        session = Mock()
        session.get.return_value = Mock(status_code=403, text="denied")
        with patch.object(upstox_api, "_session", return_value=session), patch.object(upstox_api, "UPSTOX_ACCESS_TOKEN", "test"):
            with self.assertRaises(RuntimeError):
                upstox_api.make_request("https://api.upstox.com/v3/historical-candle/example")
        self.assertEqual(session.get.call_count, 1)

    def test_client_rejects_success_http_with_api_error_payload(self):
        session = Mock()
        session.get.return_value = Mock(status_code=200)
        session.get.return_value.json.return_value = {"status": "error", "data": {}}
        with patch.object(upstox_api, "_session", return_value=session), patch.object(upstox_api, "UPSTOX_ACCESS_TOKEN", "test"):
            with self.assertRaises(upstox_api.UpstoxAPIError):
                upstox_api.make_request("https://api.upstox.com/v3/historical-candle/example")
        self.assertEqual(session.get.call_count, 1)

    @unittest.skipUnless(MASTER.exists(), "local market data not present (fresh clone); the synthetic variant below still runs")
    def test_batch_matches_scalar_status_prices_and_greeks(self):
        d = pd.read_parquet(MASTER).sample(150, random_state=87)
        self.check_batch_matches_scalar(add_time_to_expiry(d))

    def test_batch_matches_scalar_on_synthetic_contracts(self):
        rng = np.random.default_rng(87)
        n = 160
        stamps = pd.Timestamp("2026-03-02 09:20", tz="Asia/Kolkata") + pd.to_timedelta(rng.integers(0, 70, n) * 5, unit="m")
        spot = rng.uniform(22000, 25500, n)
        strike = np.round(spot * np.exp(rng.normal(0, .03, n)) / 50) * 50
        kind = np.where(rng.random(n) < .5, "CE", "PE")
        d = pd.DataFrame({"timestamp": stamps, "feature_available_at": stamps + pd.Timedelta(minutes=5),
                          "expiry": pd.Timestamp("2026-03-10") + pd.to_timedelta(rng.integers(0, 14, n), unit="D"),
                          "nifty_spot": spot, "strike": strike, "option_type": kind})
        d = add_time_to_expiry(d)
        vol = rng.uniform(.08, .6, n)
        price = [black_scholes_price(s, k, t, v, 0.06, o) for s, k, t, v, o in zip(spot, strike, d.time_to_expiry, vol, kind)]
        d["close"] = np.round(np.array(price) * rng.choice([1., 1., 1., .5, 1.8], n), 2)       # some prices break no-arbitrage bounds
        self.check_batch_matches_scalar(d)

    def check_batch_matches_scalar(self, d):
        extra = d.iloc[:8].copy()
        extra.loc[extra.index[:2], "close"] = [np.nan, -1]
        extra.loc[extra.index[2:4], "nifty_spot"] = [np.nan, -1]
        extra.loc[extra.index[4:6], "time_to_expiry"] = [0, np.nan]
        extra.loc[extra.index[6], "close"] = 1e9
        extra.loc[extra.index[7], "option_type"] = "invalid"
        d = pd.concat([d, extra], ignore_index=True)
        scalar = d.apply(process_row, axis=1)
        batch = batch_pricing_features(d)
        np.testing.assert_array_equal(scalar.iv_status, batch.iv_status)
        for name in ["implied_volatility", "theoretical_price", "delta", "gamma", "vega", "theta"]:
            np.testing.assert_allclose(scalar[name], batch[name], rtol=2e-5, atol=2e-5, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
