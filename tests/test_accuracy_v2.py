from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
warnings.filterwarnings('ignore')
from accuracy_v2.features import GROUPS, REQUIRED, V2_COLUMNS, build_v2_features
from accuracy_v2.experiment import FORBIDDEN, assert_no_forbidden, best_threshold, make_estimator
from robust_model_research import purged_folds

SURFACE = ROOT / 'data/research/expanded_20261030/data/processed/options_parity_features.parquet'


def synthetic_surface(prices=None, volumes=None, deviations=None, slope=.5):
    """One expiry, one session, CE and PE at five strikes, several bars; IV is an exact linear smile."""
    ts = pd.date_range('2026-01-06 09:15', periods=8, freq='5min', tz='Asia/Kolkata')
    strikes = [23800., 23900., 24000., 24100., 24200.]
    rows = []
    for t_index, t in enumerate(ts):
        for strike in strikes:
            for kind in ('CE', 'PE'):
                lm = np.log(strike / 24000.)
                key = f'{kind}{int(strike)}'
                rows.append({'instrument_key': key, 'timestamp': t, 'trade_date': t.date(), 'expiry': pd.Timestamp('2026-01-13'),
                             'option_type': kind, 'strike': strike, 'nifty_spot': 24000., 'log_moneyness': lm,
                             'market_price': (prices[t_index] if prices is not None else 100. + t_index) + (0 if strike == 24000. else 1.),
                             'volume': volumes[t_index] if volumes is not None else 5.,
                             'open_interest': 1000., 'implied_volatility': .2 + slope * lm + (.01 if kind == 'CE' else 0.),
                             'reference_iv': .2, 'reference_price_deviation': deviations[t_index] if deviations is not None else -1.,
                             'iv_surface_residual': 0., 'parity_residual': 10., 'time_to_expiry': .02})
    return pd.DataFrame(rows)[REQUIRED]


class FeatureCorrectnessTests(unittest.TestCase):
    def pick(self, features, key):
        return features[features.instrument_key == key].reset_index(drop=True)

    def test_staleness_and_zero_volume_runs_match_hand_calculation(self):
        f = build_v2_features(synthetic_surface(prices=[100, 100, 100, 101, 101, 102, 102, 102], volumes=[5, 0, 0, 3, 0, 0, 0, 1]))
        row = self.pick(f, 'CE24000')
        self.assertEqual(row.stale_run.tolist(), [0, 1, 2, 0, 1, 0, 1, 2])
        self.assertEqual(row.zero_volume_run.tolist(), [0, 1, 2, 0, 1, 2, 3, 0])

    def test_deviation_dynamics_use_only_past_and_present(self):
        deviations = [-1., -2., -3., 2., 3., 4., -5., -6.]
        f = self.pick(build_v2_features(synthetic_surface(deviations=deviations)), 'CE24000')
        self.assertTrue(np.isnan(f.dev_lag3[2]))
        self.assertEqual(f.dev_lag3[3], -1.)                       # value at t-3
        self.assertEqual(f.dev_lag3[7], 3.)
        self.assertEqual(f.dev_same_sign_run.tolist()[:6], [1, 2, 3, 1, 2, 3])
        self.assertEqual(f.dev_sign_flips6[4], 1.)                 # one sign change among the available bars
        self.assertAlmostEqual(f.absdev_frac_of_day_max[6], 5 / 5, places=5)  # |-5| is the running maximum
        self.assertAlmostEqual(f.absdev_frac_of_day_max[5], 4 / 4, places=5)

    def test_exact_linear_smile_gives_its_slope_curvature_and_zero_residual(self):
        f = build_v2_features(synthetic_surface(slope=.5))
        puts = f[f.instrument_key.str.startswith('PE')]
        self.assertTrue(np.allclose(puts.smile_slope, .5, atol=2e-3))
        self.assertTrue(np.allclose(puts.smile_curv, 0., atol=.5))        # log-moneyness is nonlinear in strike; curvature ~ 0
        self.assertTrue((puts.iv_minus_smile.abs() < 5e-3).all())
        calls = self.pick(f, 'CE24000')
        self.assertTrue(np.allclose(calls.put_call_iv_spread, .01, atol=1e-9))   # call IV is 0.01 above put IV by construction

    def test_input_row_order_does_not_change_output(self):
        s = synthetic_surface(deviations=[-1, -2, 3, 2, -4, 1, 0.5, -3])
        shuffled = s.sample(frac=1., random_state=3).reset_index(drop=True)
        a, b = build_v2_features(s), build_v2_features(shuffled)
        pd.testing.assert_frame_equal(a, b)

    def test_features_never_reference_labels_or_future_fields(self):
        assert_no_forbidden(V2_COLUMNS)
        self.assertFalse([c for c in REQUIRED if any(token in c.lower() for token in FORBIDDEN)])
        self.assertEqual(len(V2_COLUMNS), len(set(V2_COLUMNS)))
        self.assertEqual(sum(len(v) for v in GROUPS.values()), len(V2_COLUMNS))
        with self.assertRaises(AssertionError):
            assert_no_forbidden(['lag1', 'converged_30m'])

    def test_duplicate_rows_and_missing_inputs_are_rejected(self):
        s = synthetic_surface()
        with self.assertRaises(ValueError):
            build_v2_features(pd.concat([s, s.iloc[:1]]))
        with self.assertRaises(ValueError):
            build_v2_features(s.drop(columns=['volume']))


@unittest.skipUnless(SURFACE.exists(), 'research surface table not present on this machine')
class RealDataCausalityTests(unittest.TestCase):
    def test_truncating_the_future_does_not_change_earlier_features(self):
        surface = pd.read_parquet(SURFACE, columns=REQUIRED)
        surface['timestamp'] = pd.to_datetime(surface.timestamp)
        days = sorted(surface.timestamp.dt.date.unique())[:8]
        full = surface[surface.timestamp.dt.date.isin(days)]
        cut = surface[surface.timestamp.dt.date.isin(days[:5])]
        a = build_v2_features(full)
        b = build_v2_features(cut)
        a = a[a.timestamp.dt.date.isin(days[:5])].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)

    def test_mid_session_truncation_is_also_invariant(self):
        surface = pd.read_parquet(SURFACE, columns=REQUIRED)
        surface['timestamp'] = pd.to_datetime(surface.timestamp)
        day = sorted(surface.timestamp.dt.date.unique())[3]
        one = surface[surface.timestamp.dt.date == day]
        noon = pd.Timestamp(f'{day} 12:00', tz='Asia/Kolkata')
        a = build_v2_features(one)
        b = build_v2_features(one[one.timestamp <= noon])
        a = a[a.timestamp <= noon].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_exact=False, rtol=1e-9, atol=1e-12)


class PurgeAndModelTests(unittest.TestCase):
    def frame(self):
        days = pd.bdate_range('2026-01-01', periods=40, tz='Asia/Kolkata')
        rows = [{'timestamp': d + pd.Timedelta(hours=9, minutes=15 + 5 * k), 'k': k} for d in days for k in range(60)]
        f = pd.DataFrame(rows)
        f['future_timestamp_30m'] = f.timestamp + pd.Timedelta(minutes=30)
        return f

    def test_training_labels_end_before_each_block_and_an_embargo_session_separates_them(self):
        f = self.frame()
        for fold, train, valid in purged_folds(f, n_splits=5, gap_sessions=1):
            boundary = f.timestamp.iloc[valid].min()
            self.assertLess((f.future_timestamp_30m.iloc[train] + pd.Timedelta(minutes=5)).max(), boundary)
            self.assertLess(f.timestamp.iloc[train].dt.date.max(), boundary.date())
            gap_days = len(pd.bdate_range(f.timestamp.iloc[train].max().date(), boundary.date())) - 2
            self.assertGreaterEqual(gap_days, 1)
            self.assertEqual(set(train) & set(valid), set())
            self.assertGreater(f.timestamp.iloc[valid].min(), f.timestamp.iloc[train].max())

    def test_fit_is_deterministic_and_reload_gives_identical_predictions(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(600, 8)), columns=[f'x{i}' for i in range(8)])
        y = (X.x0 + .5 * rng.normal(size=600) > 0).astype(int)
        for name in ('lgb', 'xgb', 'et'):
            a = make_estimator(name).fit(X, y).predict_proba(X)[:, 1]
            b = make_estimator(name).fit(X, y).predict_proba(X)[:, 1]
            if name == 'et':        # threaded per-tree accumulation order can differ at ~1e-16
                self.assertTrue(np.allclose(a, b, rtol=0, atol=1e-12), name)
            else:
                self.assertTrue(np.array_equal(a, b), name)
        with TemporaryDirectory() as folder:
            model = make_estimator('lgb').fit(X, y)
            joblib.dump(model, Path(folder) / 'm.joblib')
            self.assertTrue(np.array_equal(model.predict_proba(X), joblib.load(Path(folder) / 'm.joblib').predict_proba(X)))

    def test_shuffled_labels_carry_no_signal_out_of_time(self):
        rng = np.random.default_rng(1)
        X = pd.DataFrame(rng.normal(size=(3000, 10)), columns=[f'x{i}' for i in range(10)])
        y = rng.integers(0, 2, 3000)
        m = make_estimator('lgb', {'n_estimators': 60}).fit(X.iloc[:2000], y[:2000])
        self.assertLess(abs(roc_auc_score(y[2000:], m.predict_proba(X.iloc[2000:])[:, 1]) - .5), .06)

    def test_threshold_search_uses_only_supplied_scores(self):
        y = np.array([0, 0, 0, 1, 1, 1, 0, 1])
        p = np.array([.1, .2, .3, .7, .8, .9, .35, .6])
        t = best_threshold(y, p)
        self.assertEqual(((p >= t) == y).mean(), 1.0)


if __name__ == '__main__':
    unittest.main()
