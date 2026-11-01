from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from continuous_hedging.config import ResearchConfig, FitConfig
from continuous_hedging.market import simulate
from continuous_hedging.accounting import pnl
from continuous_hedging.policies import ContinuousPolicy, FEATURES, features, positions
from continuous_hedging.recurrent import (Ensemble, GRUStep, LegacyStep, MLPStep, PolicySpec, build, feature_mask,
                                          fit_recurrent, infer, load_step_policy, parameter_count, rollout)


def randomize(model, seed=5, scale=.5):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=generator) * scale)
    return model


class RecurrentPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.c = ResearchConfig(steps=6, cost_rate=.001)
        cls.batch = simulate(cls.c, 24, seed=3)

    def test_gru_and_mlp_start_exactly_at_the_declared_prior(self):
        for model in (GRUStep('delta', 1.2, 8), MLPStep('delta', 1.2, 8)):
            h = rollout(model, self.batch, self.c)
            self.assertTrue(torch.allclose(h, self.batch['delta'].clamp(-1.2, 1.2), atol=1e-6))
        flat = rollout(GRUStep('flat', 1.2, 8), self.batch, self.c)
        expected = torch.where(self.batch['call'][:, None], .5, -.5).expand_as(flat)
        self.assertTrue(torch.allclose(flat, expected, atol=1e-6))

    def test_mlp_step_matches_the_phase4_policy_with_full_mask(self):
        legacy = randomize(ContinuousPolicy('delta', 1.2, 32), seed=9)
        step = MLPStep('delta', 1.2, 32)
        step.core.load_state_dict(legacy.state_dict())
        self.assertTrue(torch.allclose(rollout(step, self.batch, self.c), positions(legacy, self.batch, self.c), atol=1e-6))
        self.assertTrue(torch.allclose(rollout(LegacyStep(legacy), self.batch, self.c), positions(legacy, self.batch, self.c), atol=1e-6))

    def test_positions_are_causal_in_future_prices(self):
        model = randomize(GRUStep('flat', 1.2, 8), seed=2)
        base = rollout(model, self.batch, self.c)
        for cut in range(self.c.steps - 1):
            changed = {k: v.clone() for k, v in self.batch.items()}
            changed['spot'][:, cut + 1:] *= 1.05
            again = rollout(model, changed, self.c)
            self.assertTrue(torch.equal(again[:, :cut + 1], base[:, :cut + 1]), f'future spot changed position at step <= {cut}')

    def test_rows_are_independent_and_chunking_is_invariant(self):
        model = randomize(GRUStep('delta', 1.2, 8), seed=4)
        full = rollout(model, self.batch, self.c)
        order = torch.randperm(24, generator=torch.Generator().manual_seed(1))
        shuffled = rollout(model, {k: v[order] if v.ndim and len(v) == 24 else v for k, v in self.batch.items()}, self.c)
        self.assertTrue(torch.allclose(shuffled, full[order], atol=1e-6))
        self.assertTrue(torch.allclose(infer(model, self.batch, self.c, chunk=5), full, atol=1e-6))
        other = {k: v.clone() for k, v in self.batch.items()}
        other['spot'][1:] *= 1.1
        self.assertTrue(torch.equal(rollout(model, other, self.c)[0], full[0]))

    def test_hidden_state_is_fresh_for_each_rollout_and_reset_ablation_forgets(self):
        model = randomize(GRUStep('delta', 1.2, 8), seed=6)
        self.assertTrue(torch.equal(rollout(model, self.batch, self.c), rollout(model, self.batch, self.c)))
        x = features(self.batch, self.c, torch.zeros(24), 2)
        zero, noisy = model.initial_state(24), torch.randn(24, 8)
        self.assertFalse(torch.allclose(model.step(x, zero)[0], model.step(x, noisy)[0]))
        model.reset_memory = True
        self.assertTrue(torch.equal(model.step(x, zero)[0], model.step(x, noisy)[0]))

    def test_gradients_reach_recurrent_weights(self):
        model = GRUStep('delta', 1.2, 8)
        with torch.no_grad():
            model.head.weight.fill_(.3)
        result = pnl(self.batch, rollout(model, self.batch, self.c), self.c)
        (-result['net_pnl'].mean()).backward()
        self.assertGreater(float(model.cell.weight_ih.grad.abs().sum()), 0)
        self.assertGreater(float(model.cell.weight_hh.grad.abs().sum()), 0)

    def test_ablation_mask_removes_only_the_network_input(self):
        model = randomize(MLPStep('flat', 1.2, 8, drop=('absolute_delta',)), seed=7)
        x = features(self.batch, self.c, torch.zeros(24), 1)
        changed = x.clone()
        changed[..., FEATURES.index('absolute_delta')] += .3
        self.assertTrue(torch.equal(model.step(x, None)[0], model.step(changed, None)[0]))
        expected = torch.where(self.batch['call'], 1., -1.)
        self.assertTrue(bool((model.step(x, None)[0].detach() * expected >= 0).all()))
        self.assertEqual(float(feature_mask(('absolute_delta',)).sum()), len(FEATURES) - 1)
        with self.assertRaises(ValueError):
            feature_mask(('future_return',))

    def test_ensemble_is_closed_loop_on_its_own_inventory(self):
        a, b = randomize(GRUStep('delta', 1.2, 8), 1), randomize(MLPStep('delta', 1.2, 8), 2)
        ensemble = rollout(Ensemble([a, b]), self.batch, self.c)
        previous, states, manual = torch.zeros(24), [a.initial_state(24), None], []
        for i in range(self.c.steps):
            x = features(self.batch, self.c, previous, i)
            p1, states[0] = a.step(x, states[0])
            p2, _ = b.step(x, None)
            previous = (p1 + p2) / 2
            manual.append(previous)
        self.assertTrue(torch.allclose(ensemble, torch.stack(manual, 1), atol=1e-6))
        open_loop = (rollout(a, self.batch, self.c) + rollout(b, self.batch, self.c)) / 2
        self.assertFalse(torch.allclose(ensemble, open_loop, atol=1e-4))
        same = rollout(Ensemble([a, a]), self.batch, self.c)
        self.assertTrue(torch.allclose(same, rollout(a, self.batch, self.c), atol=1e-6))

    def test_accounting_bounds_and_parameter_disclosure(self):
        model = randomize(GRUStep('flat', 1.2, 8), seed=8, scale=2.)
        h = rollout(model, self.batch, self.c)
        self.assertLessEqual(float(h.abs().max()), 1.2 + 1e-6)
        result = pnl(self.batch, h, self.c)
        self.assertLess(float(result['reconciliation_error'].abs().max()), 1e-5)
        self.assertEqual(parameter_count(GRUStep(hidden=16)), 3 * 16 * (7 + 16) + 6 * 16 + 17)
        self.assertEqual(parameter_count(MLPStep(hidden=32)), 7 * 32 + 32 + 32 * 32 + 32 + 32 + 1)

    def test_spec_validation(self):
        with self.assertRaises(ValueError):
            PolicySpec('x', 'lstm')
        with self.assertRaises(ValueError):
            PolicySpec('x', 'mlp', reset_memory=True)
        with self.assertRaises(ValueError):
            PolicySpec('x', 'gru', drop=('nonsense',))

    def test_tiny_fit_checkpoint_reload_is_exact_and_rejects_bad_schema(self):
        c = ResearchConfig(steps=4, cost_rate=.001)
        fit_config = FitConfig(seeds=(1,), epochs=2, flat_epochs=2, updates=2, batch=32, validation_paths=64, test_paths=64)
        validation = simulate(c, 64, seed=11)
        test = simulate(c, 48, seed=12)
        with TemporaryDirectory() as folder:
            for spec in (PolicySpec('g', 'gru', 'delta', 8), PolicySpec('m', 'mlp', 'flat', 8, drop=('previous_magnitude',))):
                for objective in ('cvar', 'entropic'):
                    model, meta = fit_recurrent(c, fit_config, validation, objective, spec, 1, Path(folder))
                    reloaded, saved = load_step_policy(Path(folder) / meta['checkpoint'])
                    self.assertTrue(torch.equal(infer(model, test, c), infer(reloaded, test, c)))
                    self.assertEqual(saved['parameter_count'], parameter_count(model))
                    self.assertLessEqual(meta['validation_objective'], float(
                        __import__('continuous_hedging.objectives', fromlist=['risk']).risk(
                            pnl(validation, rollout(build(spec, c.max_position).eval(), validation, c), c)['net_pnl'], c, objective)) + 1e-9)
            bad = torch.load(Path(folder) / 'g_cvar_1.pt', weights_only=True)
            bad['features'] = bad['features'][:-1]
            torch.save(bad, Path(folder) / 'bad.pt')
            with self.assertRaises(ValueError):
                load_step_policy(Path(folder) / 'bad.pt')
            bad = torch.load(Path(folder) / 'g_cvar_1.pt', weights_only=True)
            bad['schema_version'] = 99
            torch.save(bad, Path(folder) / 'bad2.pt')
            with self.assertRaises(ValueError):
                load_step_policy(Path(folder) / 'bad2.pt')

    def test_matched_training_stream_is_independent_of_family(self):
        a = torch.Generator().manual_seed(11 + 7000)
        b = torch.Generator().manual_seed(11 + 7000)
        x, y = simulate(self.c, 8, generator=a), simulate(self.c, 8, generator=b)
        self.assertTrue(torch.equal(x['spot'], y['spot']))


if __name__ == '__main__':
    unittest.main()
