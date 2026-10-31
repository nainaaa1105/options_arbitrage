import sys
import unittest
from pathlib import Path
from dataclasses import replace
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from deep_hedging.simulation import MarketSpec, HestonOracle, simulate, bs_marks, heston_cf
from deep_hedging.config import HedgeConfig
from deep_hedging.batch_environment import BatchEnvironment, rollout_baseline
from deep_hedging.engine import run_episode
from deep_hedging.policy import HedgePolicy, rollout_policy, current_features, FEATURE_NAMES, load_policy
from deep_hedging.objectives import cvar_objective, score_function_loss
from deep_hedging.simulation_bridge import scalar_replay, episode_frame
from deep_hedging.train import TrainingSpec, fit_seed
from deep_hedging.metrics import empirical_loss_risk


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = MarketSpec(model="heston")
        cls.oracle = HestonOracle(cls.spec)

    def test_gbm_and_heston_reproduce_identical_paths_from_seed(self):
        for spec, oracle in ((MarketSpec(), None), (self.spec, self.oracle)):
            first = simulate(spec, 32, seed=713, oracle=oracle)
            second = simulate(spec, 32, seed=713, oracle=oracle)
            for key in first:
                np.testing.assert_array_equal(first[key], second[key])
            self.assertTrue((first["variance"] >= 0).all())
            self.assertTrue((first["maturity"] > 0).all())

    def test_independent_seeds_produce_distinct_paths(self):
        a = simulate(MarketSpec(), 32, seed=9)
        b = simulate(MarketSpec(), 32, seed=10)
        self.assertFalse(np.array_equal(a["spot"], b["spot"]))

    def test_gbm_is_zero_drift_martingale_within_sampling_tolerance(self):
        b = simulate(MarketSpec(), 20000, seed=702)
        ratio = b["spot"][:, -1]/b["spot"][:, 0]
        self.assertLess(abs(ratio.mean()-1), 5*ratio.std(ddof=1)/np.sqrt(len(ratio)))

    def test_characteristic_function_normalization_and_martingale(self):
        args = (.02, .04, 2., .04, .5, -.7)
        self.assertAlmostEqual(abs(heston_cf(0, *args)-1), 0, places=12)
        self.assertAlmostEqual(abs(heston_cf(-1j, *args)-1), 0, places=12)

    def test_heston_constant_vol_limit_recovers_black_scholes(self):
        proto = object.__new__(HestonOracle)
        proto.spec = replace(self.spec, vol_of_vol=1e-8)
        s = np.array([22500., 23000., 23500.])
        k, t, v, call = np.full(3, 23000.), np.full(3, .02), np.full(3, .04), np.array([True, True, False])
        heston_price, delta = proto.direct_marks(s, k, t, v, call)
        bs_price, bs_delta = bs_marks(s, k, t, v, call)
        np.testing.assert_allclose(heston_price, bs_price, atol=1e-5, rtol=0)
        np.testing.assert_allclose(delta, bs_delta, atol=1e-6, rtol=0)

    def test_heston_parity_and_direct_delta_finite_difference(self):
        s, k, t, v = np.array([23000.]), np.array([23100.]), np.array([.02]), np.array([.04])
        call, delta = self.oracle.direct_marks(s, k, t, v, True)
        put, put_delta = self.oracle.direct_marks(s, k, t, v, False)
        np.testing.assert_allclose(call-put, s-k, atol=1e-8)
        np.testing.assert_allclose(delta-put_delta, 1, atol=1e-12)
        up = self.oracle.direct_marks(s+.1, k, t, v, True)[0]
        down = self.oracle.direct_marks(s-.1, k, t, v, True)[0]
        np.testing.assert_allclose((up-down)/.2, delta, atol=1e-6)

    def test_grid_interpolation_is_audited_against_direct_pricing(self):
        b = simulate(self.spec, 128, seed=791, oracle=self.oracle)
        p, d = self.oracle.direct_marks(b["spot"][:,0], b["strike"], b["maturity"][:,0], b["variance"][:,0], b["call"])
        self.assertLess(np.max(abs(p-b["option"][:,0])), 1.)
        self.assertLess(np.max(abs(d-b["delta"][:,0])), .005)
        with self.assertRaises(ValueError):
            self.oracle.marks(23000., 23000., .01, .0001, True)

    def test_longer_horizon_does_not_invent_expiry_payoffs(self):
        b = simulate(MarketSpec(horizon_minutes=60, steps=12), 16, seed=83)
        self.assertEqual(b["option"].shape, (16, 13))
        self.assertTrue((b["maturity"][:,-1] > 0).all())
        self.assertTrue(np.any(b["option"][:,-1] > np.maximum(np.where(b["call"], b["spot"][:,-1]-b["strike"], b["strike"]-b["spot"][:,-1]), 0)))


class NeuralAccountingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.batch = simulate(MarketSpec(), 32, seed=17)
        cls.config = HedgeConfig()

    def test_all_baselines_match_original_scalar_accounting(self):
        for strategy in ("unhedged", "static_delta", "dynamic_delta", "band_delta"):
            vector = rollout_baseline(self.batch, self.config, strategy)
            scalar = scalar_replay(self.batch, self.config, strategy, count=16)
            np.testing.assert_allclose(vector["net_pnl"][:16], scalar, atol=1e-6, rtol=0)
            self.assertLess(abs(vector["reconciliation_error"]).max(), 1e-6)

    def test_unknown_baseline_is_not_silently_treated_as_delta_hedging(self):
        with self.assertRaises(ValueError):
            rollout_baseline(self.batch, self.config, "misspelled_policy")

    def test_neural_policy_matches_scalar_replay(self):
        torch.manual_seed(23)
        model = HedgePolicy()
        with torch.no_grad():
            result = rollout_policy(self.batch, self.config, model)[0]
        scalar = scalar_replay(self.batch, self.config, "neural", model=model, count=16)
        np.testing.assert_allclose(result["net_pnl"][:16], scalar, atol=1e-6, rtol=0)
        self.assertTrue(np.all(result["actions"] == np.round(result["actions"])))
        self.assertLessEqual(abs(result["actions"]).max(), self.config.max_future_lots)

    def test_future_prices_cannot_change_earlier_neural_actions(self):
        model = HedgePolicy()
        changed = {k: v.copy() for k, v in self.batch.items()}
        changed["future"][:,4:] *= 1.01
        changed["spot"][:,4:] *= 1.01
        changed["delta"][:,4:] *= .5
        with torch.no_grad():
            a = rollout_policy(self.batch, self.config, model)[0]
            b = rollout_policy(changed, self.config, model)[0]
        np.testing.assert_array_equal(a["actions"][:,:4], b["actions"][:,:4])

    def test_cvar_matches_empirical_tail_and_has_correct_loss_sign(self):
        pnl = np.array([10., -20., -100., 0.])
        actual = cvar_objective(torch.tensor(-pnl), .95)
        self.assertAlmostEqual(float(actual), empirical_loss_risk(pnl)[1])
        self.assertGreater(float(actual), 0)

    def test_cvar_policy_gradient_is_finite_and_changes_parameters(self):
        torch.manual_seed(8)
        model = HedgePolicy()
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        before = [p.detach().clone() for p in model.parameters()]
        result, logp, entropy = rollout_policy(self.batch, self.config, model, stochastic=True)
        loss, risk = score_function_loss(result["net_pnl"], logp, entropy, 20000.)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        optimizer.step()
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,model.parameters())))

    def test_margin_and_forced_exit_match_scalar_engine(self):
        b = {k: v[:8].copy() for k, v in self.batch.items()}
        b["future"][:,2:] *= 4
        c = replace(self.config, initial_capital=1_000_000)
        result = rollout_baseline(b, c, "static_delta")
        scalar = scalar_replay(b, c, "static_delta", count=8)
        np.testing.assert_allclose(result["net_pnl"], scalar, atol=1e-6, rtol=0)
        self.assertGreater(result["forced_exits"].sum(), 0)

    def test_checkpoint_load_reproduces_actions(self):
        model = HedgePolicy()
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.pt"
            torch.save({"state_dict": model.state_dict(), "max_lots": 4, "hidden": 32,
                        "feature_names": list(FEATURE_NAMES)}, path)
            loaded, _ = load_policy(path)
            with torch.no_grad():
                a = rollout_policy(self.batch, self.config, model)[0]
                b = rollout_policy(self.batch, self.config, loaded)[0]
            np.testing.assert_array_equal(a["actions"], b["actions"])
            np.testing.assert_array_equal(a["net_pnl"], b["net_pnl"])

    def test_control_variate_does_not_feed_future_information_to_actor(self):
        model = HedgePolicy()
        result, logp, entropy = rollout_policy(self.batch, self.config, model, stochastic=True)
        reference = rollout_baseline(self.batch, self.config, "band_delta")["net_pnl"]
        eta = torch.nn.Parameter(torch.tensor(.3))
        loss, _ = score_function_loss(result["net_pnl"], logp, entropy, 20000., eta=eta, reference_pnl=reference)
        loss.backward()
        self.assertIsNone(eta.grad)
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        threshold = cvar_objective(-torch.tensor(result["net_pnl"], dtype=torch.float32)/20000., eta=eta)
        threshold.backward()
        self.assertTrue(torch.isfinite(eta.grad))

    def test_actor_has_no_inputs_from_future_times(self):
        e = BatchEnvironment(self.batch, self.config)
        e.prepare(0)
        first = current_features(e)
        modified = {k: v.copy() for k, v in self.batch.items()}
        for name in ("spot", "future", "option", "delta", "variance", "maturity"):
            modified[name][:, 1:] *= 1.1
        other = BatchEnvironment(modified, self.config)
        other.prepare(0)
        np.testing.assert_array_equal(first, current_features(other))

    def test_actual_training_saves_post_cvar_checkpoint_and_curves(self):
        spec = TrainingSpec(seeds=(77,), epochs=1, updates_per_epoch=2, batch_size=32,
                            warm_updates=2, validation_paths=128, test_paths=128, stress_paths=128)
        validation = simulate(MarketSpec(), 128, seed=8761)
        with TemporaryDirectory() as tmp:
            out = fit_seed(MarketSpec(), None, self.config, spec, validation, 77, Path(tmp))
            self.assertEqual(out["best_epoch"], 1)
            self.assertGreater(out["parameter_change_after_warm_start"], 0)
            model, saved = load_policy(Path(tmp)/out["checkpoint"])
            self.assertEqual(saved["best_epoch"], 1)
            self.assertTrue((Path(tmp)/"learning_curve_seed_77.csv").exists())


if __name__ == "__main__":
    unittest.main()
