from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from deep_hedging.data import build_episodes, validate_sources
from signal_handoff.episodes import build_chosen_episodes
from signal_handoff.run import choose_all, date_cluster_bootstrap, name_of, period_of
from signal_handoff.selection import atm_choice, max_deviation_choice, ml_choice, random_choice, select_slot
from signal_handoff.universe import slot_frame, slot_valid
from deep_hedging.engine import compare_baselines
from deep_hedging.baselines import STRATEGIES
from test_hedging_part1 import source_fixture, zero_config


def candidates(probabilities, deviations, kinds=None, keys=None):
    n = len(probabilities)
    return pd.DataFrame({'instrument_key': keys or [f'K{i}' for i in range(n)],
                         'option_type': kinds or ['CE'] * n, 'expiry': ['2026-01-13'] * n,
                         'strike': [1000. + 50 * i for i in range(n)], 'nifty_spot': [1000.] * n,
                         'reference_price_deviation': deviations, 'probability': probabilities})


class SelectionTests(unittest.TestCase):
    def test_ml_picks_highest_probability_at_or_above_cutoff_and_abstains_below(self):
        c = candidates([.4, .7, .55], [-2., -1., -3.])
        self.assertEqual(ml_choice(c, .5).instrument_key, 'K1')
        self.assertIsNone(ml_choice(c, .8))
        tie = candidates([.7, .7], [-1., -4.])
        self.assertEqual(ml_choice(tie, .5).instrument_key, 'K1')     # larger |deviation| breaks the tie
        self.assertIsNone(ml_choice(candidates([np.nan, np.nan], [-1., -1.]), .5))  # missing probability never trades

    def test_long_only_guard_rejects_overvalued_candidates(self):
        with self.assertRaises(ValueError):
            ml_choice(candidates([.9], [+2.]), .5)
        with self.assertRaises(ValueError):
            max_deviation_choice(candidates([.9], [0.]))

    def test_controls_do_not_depend_on_probability_and_are_deterministic(self):
        c = candidates([.9, .1, .2], [-1., -5., -2.])
        self.assertEqual(max_deviation_choice(c).instrument_key, 'K1')
        scrambled = c.assign(probability=[.1, .9, .5])
        self.assertEqual(max_deviation_choice(scrambled).instrument_key, 'K1')
        t = pd.Timestamp('2026-01-06 10:30', tz='Asia/Kolkata')
        self.assertEqual(random_choice(c, t).instrument_key, random_choice(scrambled, t).instrument_key)
        self.assertIn(random_choice(c, t).instrument_key, set(c.instrument_key))

    def test_atm_control_matches_the_ml_option_type_and_uses_no_signal_fields(self):
        admitted = candidates([np.nan] * 4, [np.nan] * 4, kinds=['CE', 'PE', 'CE', 'PE'])
        admitted['strike'] = [1000., 1005., 1010., 990.]
        self.assertEqual(atm_choice(admitted, 'PE').instrument_key, 'K1')
        self.assertEqual(atm_choice(admitted, 'CE').instrument_key, 'K0')
        self.assertIsNone(atm_choice(admitted.assign(strike=1500.), 'CE'))

    def test_controls_only_trade_where_ml_fires(self):
        c = candidates([.4, .3], [-1., -2.])
        t = pd.Timestamp('2026-01-06 10:30', tz='Asia/Kolkata')
        self.assertIsNone(select_slot('random', c, c, t, .5, ml_pick=None))
        with self.assertRaises(ValueError):
            select_slot('unknown', c, c, t, .5, ml_pick=c.iloc[0])


class UniverseAndBuilderTests(unittest.TestCase):
    def setUp(self):
        self.options, self.futures = source_fixture()
        self.config = zero_config()

    def test_slot_window_matches_part1_rule(self):
        ok = pd.Timestamp('2026-01-06 14:30', tz='Asia/Kolkata')
        late = pd.Timestamp('2026-01-06 15:00', tz='Asia/Kolkata')
        self.assertTrue(slot_valid(ok, self.config))
        self.assertFalse(slot_valid(late, self.config))

    def test_slot_rows_are_the_decision_bar_only(self):
        o, _ = validate_sources(self.options, self.futures)
        slots = slot_frame(o, self.config, decision_times=('09:30',))
        self.assertTrue((slots.decision_time == slots.timestamp).all())
        self.assertTrue((slots.entry_time == slots.decision_time + pd.Timedelta(minutes=5)).all())

    def test_injected_choice_builder_reproduces_part1_exactly(self):
        expected = build_episodes(self.options, self.futures, self.config, decision_times=('09:15',))
        self.assertEqual(len(expected[0]), 1)
        chosen = expected[0][['decision_time', 'instrument_key']]
        episodes, paths, skipped = build_chosen_episodes(self.options, self.futures, self.config, chosen)
        pd.testing.assert_frame_equal(episodes.reset_index(drop=True), expected[0].reset_index(drop=True))
        pd.testing.assert_frame_equal(paths.reset_index(drop=True), expected[1].reset_index(drop=True))
        self.assertTrue(skipped.empty)

    def test_builder_rejects_overlap_and_missing_bars_without_inventing_paths(self):
        decision = pd.Timestamp('2026-01-06 09:15', tz='Asia/Kolkata')
        chosen = pd.DataFrame({'decision_time': [decision, decision + pd.Timedelta(minutes=10)], 'instrument_key': ['OPT', 'OPT']})
        episodes, paths, skipped = build_chosen_episodes(self.options, self.futures, self.config, chosen)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(skipped.reason.iloc[0], 'overlapping_episode_schedule')
        late = pd.DataFrame({'decision_time': [decision + pd.Timedelta(minutes=30)], 'instrument_key': ['OPT']})
        episodes, _, skipped = build_chosen_episodes(self.options, self.futures, self.config, late)
        self.assertTrue(episodes.empty)
        self.assertEqual(skipped.reason.iloc[0], 'missing_exact_option_or_future_bar')

    def test_selection_is_independent_of_future_bars(self):
        o, _ = validate_sources(self.options, self.futures)
        base = slot_frame(o, self.config, decision_times=('09:30',))
        later = self.options.copy()
        later.loc[later.timestamp > pd.Timestamp('2026-01-06 09:30', tz='Asia/Kolkata'), ['open', 'close']] *= 7
        o2, _ = validate_sources(later, self.futures)
        again = slot_frame(o2, self.config, decision_times=('09:30',))
        pd.testing.assert_frame_equal(base.reset_index(drop=True), again.reset_index(drop=True))


class ProtocolTests(unittest.TestCase):
    def test_matched_slots_and_names(self):
        t1, t2 = (pd.Timestamp(f'2026-01-06 {c}', tz='Asia/Kolkata') for c in ('09:30', '10:30'))
        rows = []
        for t, probabilities in ((t1, [.8, .2]), (t2, [.3, .2])):
            for i, p in enumerate(probabilities):
                rows.append({'decision_time': t, 'instrument_key': f'{t:%H%M}K{i}', 'option_type': 'CE', 'expiry': '2026-01-13',
                             'strike': 1000. + i, 'nifty_spot': 1000., 'reference_price_deviation': -2. - i, 'probability': p,
                             'admitted': True, 'in_domain': True, 'undervalued': True})
        picks = choose_all(pd.DataFrame(rows), (.5,))
        self.assertEqual(set(picks[picks.rule == 'ml'].decision_time), {t1})
        for rule in ('random', 'max_deviation', 'atm_nearest'):
            self.assertLessEqual(set(picks[picks.rule == rule].decision_time), {t1})
        self.assertEqual(name_of('ml', .5), 'ml_tau50')
        self.assertEqual(period_of([pd.Timestamp('2026-09-16', tz='Asia/Kolkata')])[0], 'test')

    def test_cluster_bootstrap_of_constant_difference_is_degenerate(self):
        dates = np.repeat(pd.date_range('2026-01-01', periods=10).date, 3)
        mean, lo, hi = date_cluster_bootstrap(np.full(30, 2.5), dates, repeats=200)
        self.assertEqual((mean, lo, hi), (2.5, 2.5, 2.5))


class EndToEndTests(unittest.TestCase):
    """Selection -> frozen choice -> episodes -> shared engine, on the synthetic fixture (software test, not evidence)."""

    def setUp(self):
        self.options, self.futures = source_fixture()
        self.config = zero_config()

    def candidates(self, probability, deviation):
        o, _ = validate_sources(self.options, self.futures)
        domain = o[['instrument_key', 'timestamp']].copy()
        domain['signal_eligible'] = True
        domain['reference_price_deviation'] = deviation
        domain['session_ok'] = True
        slots = slot_frame(o, self.config, decision_times=('09:15',), domain=domain)
        slots['probability'] = probability
        return slots

    def test_ml_entry_flows_through_the_shared_engine_with_reconciled_accounting(self):
        slots = self.candidates(.8, -2.)
        picks = choose_all(slots, (.5,))
        self.assertEqual(set(picks.rule), {'ml', 'random', 'max_deviation', 'atm_nearest'})
        ml = picks[picks.rule == 'ml']
        self.assertEqual(len(ml), 1)
        episodes, paths, skipped = build_chosen_episodes(self.options, self.futures, self.config, ml[['decision_time', 'instrument_key']])
        self.assertEqual(len(episodes), 1)
        results, orders, equity, rejected = compare_baselines(episodes, paths, self.config)
        self.assertEqual(set(results.strategy), set(STRATEGIES))
        self.assertLess(results.reconciliation_error.abs().max(), 1e-6)
        self.assertTrue((results.option_gross_pnl == results.option_gross_pnl.iloc[0]).all())   # identical exposure across hedgers
        self.assertTrue(results.net_pnl.notna().all())

    def test_probability_below_the_cutoff_or_overvalued_options_produce_no_entry(self):
        self.assertTrue(choose_all(self.candidates(.4, -2.), (.5,)).empty)
        overvalued = self.candidates(.9, +2.)
        self.assertTrue(choose_all(overvalued, (.5,)).empty)          # long-only: positive deviation never qualifies


if __name__ == '__main__':
    unittest.main()
