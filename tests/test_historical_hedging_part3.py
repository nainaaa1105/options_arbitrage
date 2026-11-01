import copy
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from deep_hedging.config import HedgeConfig
from deep_hedging.engine import run_episode
from deep_hedging.historical_data import attach_states, pack_paths, split_episodes, prediction_blocks, fit_predict_blocks
from deep_hedging.historical_evaluate import evaluate_historical, paired_session_comparisons
from deep_hedging.historical_policy import (HistoricalEnvironment, HistoricalPolicy, HistoricalScalarPolicy, batch_features,
                                           rollout_historical, save_historical_policy, load_historical_policy, observed_features)
from deep_hedging.historical_train import HistoricalTraining, train_ablation
from deep_hedging.policy import HedgePolicy
from deep_hedging.simulation import MarketSpec, simulate
from deep_hedging.simulation_bridge import episode_frame


def fixture(n=12):
    c = HedgeConfig()
    batch = simulate(MarketSpec(), n, seed=398)
    episodes, paths = [], []
    for i in range(n):
        ep, path = episode_frame(batch, i, c)
        start = pd.Timestamp("2026-03-20 09:35", tz="Asia/Kolkata")+pd.Timedelta(days=i)
        path["timestamp"] = pd.date_range(start, periods=7, freq="5min")
        path["observed_at"] = path.timestamp
        path["episode_id"] = ep["episode_id"]
        path["step"] = np.arange(7)
        path["option_close_observed"] = path.option_mark
        path["future_close_observed"] = path.future_mark
        path["observed_iv"] = path.volatility
        path["observed_delta_used"] = path.delta
        path["pricing_state_valid"] = True
        path["probability"] = .3+.5*i/max(n, 1)
        path["ml_available"] = True
        ep.update(entry_time=start, exit_time=start+pd.Timedelta(minutes=30),
                  strike=float(batch["strike"][i]), option_type="CE" if batch["call"][i] else "PE")
        episodes.append(ep)
        paths.append(path)
    return pd.DataFrame(episodes), pd.concat(paths, ignore_index=True), c


class HistoricalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_split_purges_cross_boundary_episode(self):
        e, _, _ = fixture(4)
        e["entry_time"] = pd.to_datetime(["2026-06-30 23:50", "2026-07-01 09:35", "2026-09-15 23:50", "2026-09-16 09:35"]).tz_localize("Asia/Kolkata")
        e["exit_time"] = e.entry_time+pd.Timedelta(minutes=30)
        self.assertEqual(split_episodes(e).split.tolist(), ["purged", "validation", "purged", "test"])

    def test_historical_moneyness_preserves_iv_calendar_variance_and_explicit_day_count(self):
        c = HedgeConfig()
        s, k, t, v = np.array([23100.]), np.array([23000.]), np.array([3/365]), np.array([.2**2])
        x = observed_features(s, k, t, v, .5, 1., True, 1., 0, c.initial_capital,
                              0., 100., 23100., 260, np.array([65]), 0., 100., c.initial_capital, c)
        self.assertAlmostEqual(float(x[0, 0]), float(np.log(s[0]/k[0])/np.sqrt(v[0]*t[0])), places=6)
        self.assertAlmostEqual(float(x[0, 1]), .3, places=6)

    def observation_fixture(self):
        e, p, c = fixture(2)
        raw = p.drop(columns=["strike", "is_call", "volatility", "remaining_time", "recent_return",
                              "option_close_observed", "observed_iv", "observed_delta_used", "pricing_state_valid",
                              "probability", "ml_available"])
        o = p.merge(e[["episode_id", "instrument_key"]], on="episode_id")
        o = pd.DataFrame({"instrument_key": o.instrument_key, "timestamp": o.timestamp-pd.Timedelta(minutes=5),
                          "close": o.option_mark, "implied_volatility": o.volatility,
                          "time_to_expiry": o.remaining_time*252/365, "feature_available_at": o.timestamp})
        return e, raw, o, c

    def test_observation_join_carries_only_past_valid_state_and_preserves_rows(self):
        e, p, o, c = self.observation_fixture()
        o.loc[2, "implied_volatility"] = np.nan
        p.loc[2, "delta"] = np.nan
        joined = attach_states(e, p, o)
        first = joined.loc[joined.episode_id.eq(e.episode_id.iloc[0])].sort_values("timestamp")
        self.assertEqual(len(joined), len(p))
        self.assertFalse(first.pricing_state_valid.iloc[2])
        self.assertEqual(first.observed_iv.iloc[2], first.observed_iv.iloc[1])
        self.assertEqual(first.observed_delta_used.iloc[2], first.observed_delta_used.iloc[1])
        self.assertTrue(joined.probability.eq(.5).all())
        self.assertFalse(joined.ml_available.any())

    def test_observation_join_rejects_initial_missing_state_and_future_availability(self):
        e, p, o, c = self.observation_fixture()
        o.loc[0, "implied_volatility"] = np.nan
        with self.assertRaisesRegex(ValueError, "initial historical state"):
            attach_states(e, p, o)
        e, p, o, c = self.observation_fixture()
        o["feature_available_at"] += pd.Timedelta(minutes=5)
        with self.assertRaisesRegex(ValueError, "unverified historical observation"):
            attach_states(e, p, o)

    def test_execution_open_does_not_change_later_observed_account_features(self):
        e, p, c = fixture()
        b = pack_paths(e, p, c)
        changed = copy.deepcopy(b)
        changed["option"][:, 1] *= 1.1
        changed["future"][:, 1] += 10
        arrays = []
        for data in (b, changed):
            env = HistoricalEnvironment(data, c)
            env.prepare(0)
            env.act(np.full(len(e), -1))
            env.finish()
            env.prepare(1)
            arrays.append(batch_features(env))
        np.testing.assert_allclose(arrays[0], arrays[1], rtol=0, atol=1e-6)

    def test_expanding_blocks_have_burn_in_and_fixed_test_classifier(self):
        days = pd.date_range("2026-01-01", "2026-10-01").date
        blocks = prediction_blocks(days)
        self.assertEqual(blocks[0][0], days[60])
        self.assertEqual(blocks[-1], (pd.Timestamp("2026-09-16").date(), days[-1]))
        self.assertTrue(all(a[1] < b[0] for a, b in zip(blocks, blocks[1:])))

    def test_classifier_training_excludes_unmatured_labels_and_future_label_changes(self):
        ts = pd.date_range("2026-01-01 09:15", periods=300, freq="5min", tz="Asia/Kolkata")
        d = pd.DataFrame({"timestamp": ts, "instrument_key": "x", "x": np.sin(np.arange(300)),
                          "future_timestamp_30m": ts+pd.Timedelta(minutes=30),
                          "target_converged_30m": np.arange(300)%2})
        query = d.iloc[-12:][["instrument_key", "timestamp", "x"]]
        blocks = [(ts[-12].date(), ts[-1].date())]
        estimator = make_pipeline(SimpleImputer(), LogisticRegression())
        with TemporaryDirectory() as tmp:
            first, audit = fit_predict_blocks(d, query, ["x"], estimator, blocks, Path(tmp)/"a")
            cutoff = pd.Timestamp(blocks[0][0], tz="Asia/Kolkata")
            changed = d.copy()
            changed.loc[changed.timestamp.ge(cutoff), "target_converged_30m"] = 1
            second, _ = fit_predict_blocks(changed, query, ["x"], estimator, blocks, Path(tmp)/"b")
            np.testing.assert_array_equal(first.probability, second.probability)
            self.assertLess(pd.Timestamp(audit[0]["maximum_label_available_at"]), cutoff)

    def test_historical_per_episode_tax_rates_change(self):
        e, p, c = fixture(16)
        b = pack_paths(e, p, c)
        self.assertEqual(set(b["future_stt"]), {.0002, .0005})
        self.assertEqual(set(b["option_stt"]), {.001, .0015})

    def test_all_baselines_match_scalar_on_both_tax_regimes(self):
        e, p, c = fixture(16)
        b = pack_paths(e, p, c)
        for name in ("unhedged", "static_delta", "dynamic_delta", "band_delta"):
            batch = rollout_historical(b, c, baseline=name)[0]
            scalar = [run_episode(ep, p.loc[p.episode_id.eq(ep.episode_id)], name, c.initial_capital, c)[0]["net_pnl"] for _, ep in e.iterrows()]
            np.testing.assert_allclose(batch["net_pnl"], scalar, atol=1e-7, rtol=0)

    def test_neural_scalar_batch_and_reloaded_checkpoint_match(self):
        e, p, c = fixture(16)
        b = pack_paths(e, p, c)
        torch.manual_seed(17)
        model = HistoricalPolicy(HedgePolicy(), True)
        torch.nn.init.normal_(model.residual.weight, std=.1)
        with TemporaryDirectory() as tmp:
            file = Path(tmp)/"policy.pt"
            save_historical_policy(file, model, c, {})
            reloaded, _ = load_historical_policy(file)
            batch = rollout_historical(b, c, reloaded)[0]
        scalar = []
        for _, ep in e.iterrows():
            result, _, _ = run_episode(ep, p.loc[p.episode_id.eq(ep.episode_id)], "ml", c.initial_capital, c,
                                       policy=HistoricalScalarPolicy(model, c.initial_capital))
            scalar.append(result["net_pnl"])
        np.testing.assert_allclose(batch["net_pnl"], scalar, atol=1e-7, rtol=0)

    def test_first_features_do_not_use_unknown_option_or_future_open(self):
        e, p, c = fixture()
        b = pack_paths(e, p, c)
        original = HistoricalEnvironment(b, c)
        original.prepare(0)
        first = batch_features(original)
        changed = copy.deepcopy(b)
        changed["option"][:, 0] *= 1.1
        changed["future"][:, 0] += 20
        alternate = HistoricalEnvironment(changed, c)
        alternate.prepare(0)
        np.testing.assert_array_equal(first, batch_features(alternate))

    def test_future_path_does_not_change_current_features_or_logits(self):
        e, p, c = fixture()
        b = pack_paths(e, p, c)
        changed = copy.deepcopy(b)
        for key in ("option", "future", "spot", "probability", "variance"):
            changed[key][:, 1:] *= 1.2
        a, z = HistoricalEnvironment(b, c), HistoricalEnvironment(changed, c)
        a.prepare(0)
        z.prepare(0)
        np.testing.assert_array_equal(batch_features(a), batch_features(z))

    def test_masked_ablation_cannot_use_probabilities_or_availability(self):
        model = HistoricalPolicy(HedgePolicy(), False)
        torch.nn.init.normal_(model.residual.weight)
        x = torch.randn(20, 16)
        changed = x.clone()
        changed[:, -2:] += 2
        torch.testing.assert_close(model(x), model(changed), atol=0, rtol=0)

    def test_ml_residual_gradients_and_frozen_base(self):
        model = HistoricalPolicy(HedgePolicy(), True)
        x = torch.ones(12, 16)
        model(x).square().mean().backward()
        self.assertGreater(model.residual.weight.grad[:, -2:].abs().sum().item(), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.base.parameters()))

    def test_invalid_pricing_state_holds_hedge_without_dropping_episode(self):
        e, p, c = fixture()
        p.loc[p.step.eq(2), "pricing_state_valid"] = False
        p.loc[p.step.eq(2), "delta"] = np.nan
        b = pack_paths(e, p, c)
        result = rollout_historical(b, c, HedgePolicy())[0]
        np.testing.assert_array_equal(result["actions"][:, 2], result["actions"][:, 1])
        self.assertEqual(len(result["net_pnl"]), len(e))

    def test_tiny_training_saves_validation_selected_policy(self):
        e, p, c = fixture(16)
        train = pack_paths(e.iloc[:8], p, c)
        validation = pack_paths(e.iloc[8:], p, c)
        spec = HistoricalTraining(seeds=(8,), epochs=1, updates_per_epoch=2, batch_size=8)
        with TemporaryDirectory() as tmp:
            output = Path(tmp)/"training"
            result = train_ablation(HedgePolicy(), train, validation, c, spec, output, True)
            loaded, metadata = load_historical_policy(output/result["selected"]["checkpoint"])
            self.assertIn(metadata["best_epoch"], (0, 1))
            self.assertTrue(np.isfinite(rollout_historical(validation, c, loaded)[0]["net_pnl"]).all())

    def test_sequential_report_reconciles_and_retains_zero_trade_day(self):
        e, p, c = fixture(4)
        dates = list(pd.date_range("2026-03-20", periods=5).date)
        with TemporaryDirectory() as tmp:
            results, report = evaluate_historical(e, p, dates, c, {"adapted_no_ml": HedgePolicy()}, Path(tmp)/"evaluation")
            self.assertTrue(report.episodes.eq(4).all())
            self.assertTrue(report.calendar_sessions.eq(5).all())
            self.assertLess(results.reconciliation_error.abs().max(), 1e-6)
            daily = pd.read_csv(Path(tmp)/"evaluation/daily_equity.csv")
            self.assertTrue(daily.loc[daily.trade_date.eq(str(dates[-1])), "net_pnl"].eq(0).all())
            paired = paired_session_comparisons(results, dates, repeats=20)
            self.assertTrue(paired.paired_episodes.eq(4).all())

    def test_common_capital_rejection_applies_to_every_strategy(self):
        e, p, c = fixture(4)
        # First episode fits; the second has a known huge option premium and
        # is rejected identically, rather than dropping it for one hedger only.
        p.loc[p.episode_id.eq(e.episode_id.iloc[1]), "option_mark"] = 100000.
        with TemporaryDirectory() as tmp:
            results, _ = evaluate_historical(e, p, list(e.entry_time.dt.date), c, {"adapted_no_ml": HedgePolicy()}, Path(tmp)/"evaluation")
            self.assertNotIn(e.episode_id.iloc[1], set(results.episode_id))
            self.assertTrue(results.groupby("strategy").size().eq(3).all())


if __name__ == "__main__":
    unittest.main()
