"""Software tests for the Phase 9 infrastructure.

Accepted-path cases build observation rows IN MEMORY to exercise the decision logic. They are not market evidence and
must never be mistaken for forward results: the repository has no genuine forward sessions yet.
"""
import copy
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from forward_paper import PaperSession, freeze, digest, runtime_manifest
import forward_paper
from run_paper_demo import demo_contracts, demo_events
from forward_validation.acceptance import evaluate, loss_cvar
from forward_validation.observations import ingest_session, open_store
from forward_validation.protocol import (default_spec, freeze_protocol, load_protocol, session_conformance, verify_chain)
from forward_validation.reconcile import reconcile_session
from forward_validation_cli import RISK_AND_QUOTE_RULES, STRATEGIES

PART3 = ROOT / 'data/research/historical_hedging_part3_final_20261031'


def allowed_for(session_dir, **override):
    p = json.loads((Path(session_dir) / 'protocol.json').read_text())
    cfg = dict(p['execution_config'])
    cfg.pop('option_tick_size')
    allowed = {'code_sha256': [p['code_sha256']], 'runtime_sha256': p['runtime_sha256'], 'strategies': STRATEGIES,
               'execution_config': cfg, 'risk_and_quote_rules': RISK_AND_QUOTE_RULES, 'policy_sha256': [], 'classifier_sha256': []}
    allowed.update(override)
    return allowed


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        run = self.root / 'run'
        run.mkdir()
        (run / 'experiment_plan.json').write_text(json.dumps({'config': {}, 'test_end': '2026-10-27T15:00:00+05:30'}))
        contracts = self.root / 'c.json'
        contracts.write_text(json.dumps(demo_contracts()))
        self.run_dir, self.contracts = run, contracts

    def tearDown(self):
        self.tmp.cleanup()

    def session(self, name='s', strategy='dynamic_delta', process=True):
        folder = self.root / name
        freeze(self.run_dir, self.contracts, folder, strategy, 'synthetic_demo')
        if process:
            runner = PaperSession(folder)
            try:
                for event in demo_events():
                    self.assertEqual(runner.process(event)['status'], 'accepted')
                runner.export()
            finally:
                runner.close()
        return folder

    def protocol(self, folder):
        registry, path = self.root / 'registry.jsonl', self.root / 'validation.json'
        return freeze_protocol(default_spec(allowed_for(folder)), path, registry), path, registry


class ProtocolTests(Base):
    def test_freeze_load_and_tamper_detection(self):
        folder = self.session(process=False)
        document, path, registry = self.protocol(folder)
        self.assertEqual(load_protocol(path, registry)['version'], 1)
        with self.assertRaises(FileExistsError):
            freeze_protocol(default_spec(allowed_for(folder)), path, registry)
        path.write_text(path.read_text().replace('"minimum_forward_sessions": 60', '"minimum_forward_sessions": 3'))
        with self.assertRaisesRegex(ValueError, 'modified after freeze'):
            load_protocol(path, registry)

    def test_versions_are_hash_chained_and_chain_breaks_are_detected(self):
        folder = self.session(process=False)
        _, _, registry = self.protocol(folder)
        second = freeze_protocol(default_spec(allowed_for(folder)), self.root / 'v2.json', registry)
        self.assertEqual(second['version'], 2)
        self.assertTrue(verify_chain(registry))
        lines = registry.read_text().splitlines()
        registry.write_text('\n'.join([lines[1], lines[0]]) + '\n')
        with self.assertRaisesRegex(ValueError, 'chain broken'):
            verify_chain(registry)

    def test_frozen_rules_match_what_the_runner_actually_writes(self):
        folder = self.session(process=False)
        written = json.loads((folder / 'protocol.json').read_text())
        for key, value in RISK_AND_QUOTE_RULES.items():
            self.assertEqual(written[key], value, key)
        self.assertEqual(written['code_sha256'], digest(forward_paper.__file__))
        self.assertEqual(written['runtime_sha256'], runtime_manifest())

    def test_conformance_flags_mode_registration_time_hashes_and_tampering(self):
        folder = self.session(process=False)
        document, _, _ = self.protocol(folder)
        problems = session_conformance(document, folder)
        self.assertTrue(any('not forward evidence' in p for p in problems))
        self.assertTrue(any('before the validation protocol was frozen' in p for p in problems))   # session predates the freeze
        wrong = copy.deepcopy(document)
        wrong['allowed']['code_sha256'] = ['0' * 64]
        wrong['allowed']['execution_config']['option_lots'] = 9
        problems = session_conformance(wrong, folder)
        self.assertTrue(any('code hash' in p for p in problems))
        self.assertTrue(any('option_lots' in p for p in problems))
        (folder / 'protocol.json').write_text((folder / 'protocol.json').read_text().replace('"forward"', '"x"'))
        (folder / 'protocol.json').write_text((folder / 'protocol.json').read_text() + ' ')
        self.assertTrue(any('modified after its own freeze' in p for p in session_conformance(document, folder)))


class IngestAndReconcileTests(Base):
    def test_ingest_is_idempotent_and_synthetic_data_is_never_forward_evidence(self):
        folder = self.session()
        document, _, _ = self.protocol(folder)
        store = self.root / 'obs.sqlite'
        first = ingest_session(document, folder, store)
        again = ingest_session(document, folder, store)
        self.assertEqual(first, again)
        db = open_store(store)
        daily = pd.read_sql_query('SELECT * FROM daily', db)
        episodes = pd.read_sql_query('SELECT * FROM episodes', db)
        sessions = pd.read_sql_query('SELECT * FROM sessions', db)
        db.close()
        self.assertEqual(len(daily), 1)
        row = daily.iloc[0]
        self.assertEqual((row.accepted, row.rejected, row.fills, row.exposure_at_close), (7, 0, 4, 0))
        self.assertEqual(int(row.forward_eligible), 0)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(int(episodes.forward_eligible.iloc[0]), 0)
        self.assertEqual(int(sessions.conforms.iloc[0]), 0)

    def test_reconciliation_passes_clean_and_detects_a_tampered_fill(self):
        folder = self.session()
        clean = reconcile_session(folder)
        self.assertTrue(clean['ok'], clean)
        self.assertLess(clean['max_error'], 1e-6)
        db = sqlite3.connect(folder / 'paper.sqlite')
        row = db.execute('SELECT rowid, payload FROM fills ORDER BY rowid LIMIT 1').fetchone()
        payload = json.loads(row[1])
        payload['fill'] += 0.5
        with db:
            db.execute('UPDATE fills SET payload=? WHERE rowid=?', (json.dumps(payload), row[0]))
        db.close()
        broken = reconcile_session(folder)
        self.assertFalse(broken['ok'])
        self.assertGreater(broken['max_error'], 1.)

    def test_exports_that_disagree_with_the_database_fail_reconciliation(self):
        folder = self.session()
        events = pd.read_csv(folder / 'events.csv').iloc[:-1]
        events.to_csv(folder / 'events.csv', index=False)
        self.assertFalse(reconcile_session(folder)['ok'])


def synthetic_store(path, days, candidate_pnl, reference_pnl, ml=False, coverage=1.0, drawdown=.02, exposure=0, forward=True,
                    contracts=('C1',), bad_first_days=0):
    """In-memory construction of observation rows for decision-logic tests only."""
    db = open_store(path)
    plan_events = 6 * 7
    strategies = [('cand', 'dynamic_delta', candidate_pnl), ('ref', 'unhedged', reference_pnl)]
    if ml:
        strategies += [('ml', 'adapted_ml', candidate_pnl), ('noml', 'adapted_no_ml', reference_pnl)]
    with db:
        for contract in contracts:
            for name, strategy, pnl in strategies:
                session = f'{name}_{contract}'
                db.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)',
                           (session, strategy, 'forward', '2027-01-01T00:00:00+00:00', 1, '[]', days * 42, 0, 'x', contract))
                for day in range(days):
                    label = (pd.Timestamp('2027-01-04') + pd.Timedelta(days=day)).date().isoformat()
                    rate = 0.1 if day < bad_first_days else coverage
                    db.execute('INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                               (session, label, plan_events, int(plan_events * rate), 0, 4, 1., 1., 5e6, 0., drawdown, 0, exposure, int(forward), '{}'))
                    for k in range(3):
                        db.execute('INSERT INTO episodes VALUES (?,?,?,?,?)',
                                   (session, f'{label}T{10 + k}:00', label, float(pnl[(day * 3 + k) % len(pnl)]), int(forward)))
    db.close()


class AcceptanceTests(Base):
    def spec(self):
        folder = self.session(process=False)
        document, _, _ = self.protocol(folder)
        return document

    def good(self, tmp):
        rng = np.random.default_rng(3)
        return rng.normal(-5., 4., 600), np.where(rng.random(600) < .1, -400., 0.) + rng.normal(0., 30., 600)

    def test_no_forward_data_is_pending_not_accepted(self):
        folder = self.session()
        document, _, _ = self.protocol(folder)
        store = self.root / 'o.sqlite'
        ingest_session(document, folder, store)
        report = evaluate(document, store, [reconcile_session(folder)])
        self.assertEqual(report['overall'], 'pending')
        self.assertEqual(report['observed_forward_days'], 0)
        self.assertEqual({r['criterion']: r['status'] for r in report['criteria']}['forward_sessions'], 'fail')

    def test_enough_sessions_with_a_real_risk_reduction_is_accepted_by_the_rule(self):
        document = self.spec()
        store = self.root / 'a.sqlite'
        candidate, reference = self.good(None)
        synthetic_store(store, 63, candidate, reference)
        # candidate and reference share per-episode keys, so the pairing is on identical positions
        report = evaluate(document, store, [{'ok': True, 'max_error': 1e-9}])
        statuses = {r['criterion']: r['status'] for r in report['criteria']}
        self.assertEqual(statuses['cvar95_reduction'], 'pass', report['criteria'])
        self.assertEqual(statuses['ml_incremental_pnl'], 'not_applicable')
        self.assertEqual(report['overall'], 'accepted')

    def test_fifty_nine_counted_days_is_pending_even_if_everything_else_passes(self):
        document = self.spec()
        store = self.root / 'p.sqlite'
        candidate, reference = self.good(None)
        synthetic_store(store, 62, candidate, reference)
        self.assertEqual(evaluate(document, store, [{'ok': True, 'max_error': 0.}])['overall'], 'pending')

    def test_hard_failures_reject_immediately_regardless_of_sample_size(self):
        document = self.spec()
        candidate, reference = self.good(None)
        for name, kwargs, recon in (('drawdown', {'drawdown': .25}, {'ok': True, 'max_error': 0.}),
                                    ('exposure', {'exposure': 1}, {'ok': True, 'max_error': 0.}),
                                    ('reconciliation', {}, {'ok': False, 'max_error': 5.})):
            store = self.root / f'{name}.sqlite'
            synthetic_store(store, 10, candidate, reference, **kwargs)
            self.assertEqual(evaluate(document, store, [recon])['overall'], 'rejected', name)

    def test_low_coverage_with_enough_sessions_is_rejected_and_no_benefit_is_not_claimed(self):
        document = self.spec()
        candidate, reference = self.good(None)
        store = self.root / 'c.sqlite'
        synthetic_store(store, 63, candidate, reference, coverage=.5)
        self.assertEqual(evaluate(document, store, [{'ok': True, 'max_error': 0.}])['overall'], 'rejected')
        flat = self.root / 'f.sqlite'
        same = np.random.default_rng(1).normal(-5., 4., 600)
        synthetic_store(flat, 63, same, same)
        report = evaluate(document, flat, [{'ok': True, 'max_error': 0.}])
        self.assertEqual({r['criterion']: r['status'] for r in report['criteria']}['cvar95_reduction'], 'fail')
        self.assertEqual(report['overall'], 'rejected')

    def test_ml_claim_is_evaluated_only_when_an_ml_strategy_is_registered(self):
        document = self.spec()
        candidate, reference = self.good(None)
        store = self.root / 'm.sqlite'
        synthetic_store(store, 63, candidate, reference, ml=True)
        statuses = {r['criterion']: r['status'] for r in evaluate(document, store, [{'ok': True, 'max_error': 0.}])['criteria']}
        self.assertIn(statuses['ml_incremental_pnl'], ('pass', 'fail'))

    def test_unreconciled_session_is_pending_not_silently_passed(self):
        document = self.spec()
        candidate, reference = self.good(None)
        store = self.root / 'u.sqlite'
        synthetic_store(store, 63, candidate, reference)
        statuses = {r['criterion']: r['status'] for r in evaluate(document, store, [])['criteria']}
        self.assertEqual(statuses['ledger_reconciliation'], 'pending')


    def test_shakedown_days_are_excluded_from_rates_but_not_from_hard_gates(self):
        document = self.spec()
        candidate, reference = self.good(None)
        store = self.root / 's.sqlite'
        synthetic_store(store, 63, candidate, reference, bad_first_days=3)       # awful coverage on the 3 declared shakedown days
        report = evaluate(document, store, [{'ok': True, 'max_error': 0.}])
        self.assertEqual(len(report['shakedown_days_excluded']), 3)
        self.assertEqual(report['observed_forward_days'], 60)
        self.assertEqual(report['overall'], 'accepted')
        worse = self.root / 's2.sqlite'
        synthetic_store(worse, 63, candidate, reference, bad_first_days=4)       # the 4th bad day is a counted day
        status = {r['criterion']: r['status'] for r in evaluate(document, worse, [{'ok': True, 'max_error': 0.}])['criteria']}
        self.assertEqual(status['event_coverage'], 'pass')                       # one bad day of 60 does not breach 90%

    def test_open_position_days_are_tolerated_up_to_the_declared_incident_count(self):
        document = self.spec()
        candidate, reference = self.good(None)
        for exposure_days, expected in ((2, 'pass'), (3, 'fail')):
            store = self.root / f'e{exposure_days}.sqlite'
            synthetic_store(store, 63, candidate, reference)
            db = open_store(store)
            with db:
                db.execute("UPDATE daily SET exposure_at_close=1 WHERE rowid IN (SELECT rowid FROM daily WHERE session='cand_C1' LIMIT ?)", (exposure_days,))
            db.close()
            status = {r['criterion']: r['status'] for r in evaluate(document, store, [{'ok': True, 'max_error': 0.}])['criteria']}
            self.assertEqual(status['unresolved_exposure_days'], expected, exposure_days)

    def test_pairing_is_per_contract_and_never_crosses_contracts(self):
        document = self.spec()
        rng = np.random.default_rng(5)
        store = self.root / 'pair.sqlite'
        synthetic_store(store, 63, rng.normal(-5., 4., 600), np.where(rng.random(600) < .1, -400., 0.) + rng.normal(0., 30., 600), contracts=('A', 'B'))
        report = evaluate(document, store, [{'ok': True, 'max_error': 0.}])
        economic = {r['criterion']: r for r in report['criteria']}['cvar95_reduction']
        self.assertEqual(economic['observed']['paired_episodes'], 2 * 60 * 3)    # 2 contracts x 60 counted days x 3 episodes

    def test_edited_evaluator_code_after_freeze_is_an_integrity_failure(self):
        from forward_validation.acceptance import evaluator_manifest
        document = copy.deepcopy(self.spec())
        document['allowed']['evaluator_sha256'] = evaluator_manifest()
        candidate, reference = self.good(None)
        store = self.root / 'ev.sqlite'
        synthetic_store(store, 63, candidate, reference)
        self.assertEqual(evaluate(document, store, [{'ok': True, 'max_error': 0.}])['overall'], 'accepted')
        document['allowed']['evaluator_sha256']['acceptance.py'] = '0' * 64
        self.assertEqual(evaluate(document, store, [{'ok': True, 'max_error': 0.}])['overall'], 'rejected')

    def test_cost_efficiency_fails_when_hedging_gives_up_more_than_it_buys(self):
        document = self.spec()
        rng = np.random.default_rng(2)
        reference = np.where(rng.random(600) < .1, -300., 0.) + rng.normal(0., 20., 600)
        expensive = np.maximum(reference, -100.) - 150.   # caps the worst losses (tail reduction about 50) but gives up about 150 per episode
        store = self.root / 'cost.sqlite'
        synthetic_store(store, 63, expensive, reference)
        status = {r['criterion']: r for r in evaluate(document, store, [{'ok': True, 'max_error': 0.}])['criteria']}
        self.assertEqual(status['cvar95_reduction']['status'], 'pass')
        self.assertEqual(status['cost_efficiency']['status'], 'fail')
        self.assertGreater(status['cost_efficiency']['observed']['ratio'], 1.0)

    def test_loss_cvar_matches_a_hand_computation(self):
        pnl = np.array([-10., -20., 5., 8., 1., 2., 3., 4., 6., 7., 9., 11., 12., 13., 14., 15., 16., 17., 18., -30.])
        loss = np.sort(-pnl)
        var = loss[int(np.ceil(len(loss) * .95)) - 1]
        self.assertAlmostEqual(loss_cvar(pnl), var + np.maximum(loss - var, 0).mean() / .05)


@unittest.skipUnless(PART3.exists(), 'Part 3 artifacts not present on this machine')
class CommandLineTests(Base):
    def test_freeze_builds_the_spec_from_live_hashes_and_evaluate_is_pending_on_synthetic_data(self):
        from forward_validation_cli import build_spec, main
        spec = build_spec(PART3)
        self.assertEqual(spec['allowed']['code_sha256'], [digest(forward_paper.__file__)])
        self.assertGreater(len(spec['allowed']['policy_sha256']), 0)
        protocol, registry, store = self.root / 'p.json', self.root / 'r.jsonl', self.root / 'o.sqlite'
        self.assertEqual(main(['freeze', '--run-dir', str(PART3), '--output', str(protocol), '--registry', str(registry)]), 0)
        folder = self.session()
        self.assertEqual(main(['ingest', '--protocol', str(protocol), '--registry', str(registry), '--store', str(store), '--session', str(folder)]), 0)
        code = main(['evaluate', '--protocol', str(protocol), '--registry', str(registry), '--store', str(store), '--session', str(folder)])
        self.assertEqual(code, 3)                                                  # pending


if __name__ == '__main__':
    unittest.main()
