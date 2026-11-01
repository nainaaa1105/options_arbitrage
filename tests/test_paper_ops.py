import copy
import gzip
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from forward_paper import PaperSession, freeze
from run_paper_demo import demo_contracts, demo_events
from paper_ops.catalog import CatalogError, InstrumentCatalog, freeze_validated
from paper_ops.health import append_alerts, assess
from paper_ops.liquidation import liquidation_report
from paper_ops.quarantine import consume_file, parse_line
from paper_ops.service import LockHeld, SingleWriterLock, run_cycle, run_loop
from paper_ops.state_builder import CompletedBarStateBuilder, Unavailable, decision_record

IST = 'Asia/Kolkata'


def write_events(path, events, extra_lines=()):
    lines = [json.dumps(e) for e in events] + list(extra_lines)
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')


class SessionCase(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        run = self.root / 'run'
        run.mkdir()
        (run / 'experiment_plan.json').write_text(json.dumps({'config': {}, 'test_end': '2026-10-27T15:00:00+05:30'}))
        contracts = self.root / 'contracts.json'
        contracts.write_text(json.dumps(demo_contracts()))
        self.session = self.root / 'paper'
        freeze(run, contracts, self.session, mode='synthetic_demo')
        self.inbox = self.root / 'inbox'
        self.inbox.mkdir()
        self.events = demo_events()

    def tearDown(self):
        self.tmp.cleanup()

    def open_session(self):
        return PaperSession(self.session)


class QuarantineTests(SessionCase):
    def test_malformed_records_are_quarantined_and_later_events_still_run(self):
        bad = ['not json at all', '{"event_id": "x", "type": "hedge", "decision_at": NaN, "received_at": "t"}',
               json.dumps({'type': 'open'}), json.dumps([1, 2, 3]), json.dumps({'event_id': ' ', 'type': 'open', 'decision_at': 'a', 'received_at': 'b'})]
        file = self.root / 'events.jsonl'
        write_events(file, self.events[:3], bad[:2] + [json.dumps(e) for e in self.events[3:]] + bad[2:])
        runner = self.open_session()
        try:
            summary = consume_file(runner, file, self.session / 'quarantine.jsonl')
            self.assertEqual(summary['accepted'], 7)
            self.assertEqual(summary['quarantined'], 5)
            self.assertEqual((runner.state()['q'], runner.state()['h']), (0, 0))
            again = consume_file(runner, file, self.session / 'quarantine.jsonl')
            self.assertEqual(again['duplicate'], 7)
            self.assertEqual(again['new_quarantine_records'], 0)
        finally:
            runner.close()
        rows = [json.loads(l) for l in (self.session / 'quarantine.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r['reason'] and r['line_sha256'] for r in rows))

    def test_conflicting_duplicate_is_quarantined_not_fatal(self):
        changed = copy.deepcopy(self.events[0])
        changed['state']['delta'] = .6
        file = self.root / 'events.jsonl'
        write_events(file, [self.events[0], changed, self.events[1]])
        runner = self.open_session()
        try:
            summary = consume_file(runner, file, self.session / 'quarantine.jsonl')
        finally:
            runner.close()
        self.assertEqual(summary['accepted'], 2)
        self.assertEqual(summary['quarantined'], 1)

    def test_parse_line_contract(self):
        self.assertIsNone(parse_line(json.dumps(self.events[0]))[1])
        self.assertEqual(parse_line('{"a": Infinity}')[1][:12], 'invalid_json')


class LiquidationAndHealthTests(SessionCase):
    def open_position(self):
        runner = self.open_session()
        try:
            self.assertEqual(runner.process(self.events[0])['status'], 'accepted')
            runner.export()
        finally:
            runner.close()

    def test_liquidation_equity_is_below_mid_and_accounts_for_costs(self):
        self.open_position()
        report = liquidation_report(self.session, now=pd.Timestamp('2026-11-02 10:00:30', tz=IST))
        self.assertTrue(report['exposure'] and report['executable'])
        self.assertLess(report['liquidation_equity'], report['mid_equity'])
        self.assertAlmostEqual(report['liquidation_cost_vs_mid'], report['liquidation_equity'] - report['mid_equity'])
        self.assertGreater(report['liquidation_fees'], 0)

    def test_flat_account_liquidation_equals_cash(self):
        runner = self.open_session()
        try:
            for event in self.events:
                runner.process(event)
        finally:
            runner.close()
        report = liquidation_report(self.session)
        self.assertFalse(report['exposure'])
        self.assertEqual(report['liquidation_equity'], report['cash'])

    def test_thin_book_after_open_makes_liquidation_unexecutable_and_alerts(self):
        self.open_position()
        thin = copy.deepcopy(self.events[1])
        thin['quotes']['option']['bid_size'] = 1
        runner = self.open_session()
        try:
            # A futures-only hedge is not depth-checked on the untraded option leg, so it may be accepted;
            # what matters is that the latest recorded book can no longer absorb a liquidation.
            runner.process(thin)
        finally:
            runner.close()
        report = liquidation_report(self.session)
        self.assertFalse(report['executable'])
        health = assess(self.session, now=pd.Timestamp('2026-11-02 10:06:00', tz=IST))
        self.assertIn('cannot_liquidate_on_displayed_book', {a['code'] for a in health['alerts']})
        self.assertFalse(health['healthy'])

    def test_stale_feed_and_overdue_position_are_critical_only_when_exposed(self):
        self.open_position()
        fresh = assess(self.session, now=pd.Timestamp('2026-11-02 10:02:00', tz=IST))
        self.assertNotIn('stale_feed_while_exposed', {a['code'] for a in fresh['alerts']})
        stale = assess(self.session, now=pd.Timestamp('2026-11-02 10:20:00', tz=IST))
        self.assertIn('stale_feed_while_exposed', {a['code'] for a in stale['alerts']})
        overdue = assess(self.session, now=pd.Timestamp('2026-11-02 10:31:00', tz=IST))
        self.assertIn('overdue_open_position', {a['code'] for a in overdue['alerts']})
        self.session.parent / 'flat'
        # A flat account with the same age raises no exposure alerts.
        runner = self.open_session()
        try:
            for event in self.events[1:]:
                runner.process(event)
        finally:
            runner.close()
        flat = assess(self.session, now=pd.Timestamp('2026-11-02 12:00:00', tz=IST))
        self.assertFalse({'stale_feed_while_exposed', 'overdue_open_position'} & {a['code'] for a in flat['alerts']})

    def test_alert_log_is_deduplicated(self):
        self.open_position()
        report = assess(self.session, now=pd.Timestamp('2026-11-02 10:20:00', tz=IST))
        first = append_alerts(self.session / 'alerts.jsonl', report)
        self.assertGreater(len(first), 0)
        self.assertEqual(append_alerts(self.session / 'alerts.jsonl', report), [])

    def test_unreadable_session_is_a_critical_alert_not_an_exception(self):
        health = assess(self.session)                                   # frozen but never opened: no database yet
        self.assertEqual([a['code'] for a in health['alerts']], ['session_unreadable'])
        self.assertFalse(health['healthy'])

    def test_tampered_protocol_is_a_critical_integrity_alert(self):
        self.open_session().close()
        protocol = json.loads((self.session / 'protocol.json').read_text())
        protocol['max_daily_loss_fraction'] = .9
        (self.session / 'protocol.json').write_text(json.dumps(protocol))
        health = assess(self.session)
        self.assertIn('integrity_failure', {a['code'] for a in health['alerts']})


class ServiceTests(SessionCase):
    def test_cycle_is_idempotent_and_writes_health_and_heartbeat(self):
        write_events(self.inbox / 'a.jsonl', self.events)
        first = run_cycle(self.session, self.inbox)
        self.assertTrue(first['ok'])
        self.assertEqual(first['files']['a.jsonl']['accepted'], 7)
        second = run_cycle(self.session, self.inbox)
        self.assertEqual(second['files']['a.jsonl'], 'unchanged_skipped')
        self.assertEqual(json.loads((self.session / 'heartbeat.json').read_text())['cycles'], 2)
        self.assertTrue((self.session / 'health.json').exists())
        self.assertFalse((self.session / 'service.lock').exists())
        runner = self.open_session()
        try:
            self.assertEqual(runner.db.execute('SELECT COUNT(*) FROM fills').fetchone()[0], 4)
        finally:
            runner.close()

    def test_restart_after_partial_processing_matches_a_single_pass(self):
        write_events(self.inbox / 'a.jsonl', self.events[:3])
        run_cycle(self.session, self.inbox)
        write_events(self.inbox / 'a.jsonl', self.events)              # file grew; first 3 are duplicates
        result = run_cycle(self.session, self.inbox)
        self.assertEqual(result['files']['a.jsonl']['duplicate'], 3)
        self.assertEqual(result['files']['a.jsonl']['accepted'], 4)
        reference = TemporaryDirectory()
        try:
            run = Path(reference.name) / 'run'
            run.mkdir()
            (run / 'experiment_plan.json').write_text(json.dumps({'config': {}, 'test_end': '2026-10-27T15:00:00+05:30'}))
            contracts = Path(reference.name) / 'c.json'
            contracts.write_text(json.dumps(demo_contracts()))
            single = Path(reference.name) / 's'
            freeze(run, contracts, single, mode='synthetic_demo')
            r = PaperSession(single)
            for e in self.events:
                r.process(e)
            expected = r.state()
            r.close()
        finally:
            reference.cleanup()
        runner = self.open_session()
        try:
            actual = runner.state()
        finally:
            runner.close()
        for key in ('cash', 'q', 'h', 'fees'):
            self.assertAlmostEqual(actual[key], expected[key], places=6)

    def test_lock_contention_and_stale_takeover(self):
        write_events(self.inbox / 'a.jsonl', self.events)
        lock = self.session / 'service.lock'
        lock.write_text(json.dumps({'pid': 1, 'token': 'other'}))
        with self.assertRaises(LockHeld):
            run_cycle(self.session, self.inbox)
        self.assertTrue(lock.exists())                                  # a held lock is never removed by a loser
        old = time.time() - 5000
        os.utime(lock, (old, old))
        result = run_cycle(self.session, self.inbox)
        self.assertTrue(result['ok'] and result['lock_takeover'])
        self.assertFalse(lock.exists())

    def test_second_writer_cannot_enter_while_first_holds_the_lock(self):
        with SingleWriterLock(self.session / 'x.lock'):
            with self.assertRaises(LockHeld):
                with SingleWriterLock(self.session / 'x.lock'):
                    pass
        self.assertFalse((self.session / 'x.lock').exists())

    def test_integrity_failure_stops_processing_and_alerts(self):
        write_events(self.inbox / 'a.jsonl', self.events)
        protocol = json.loads((self.session / 'protocol.json').read_text())
        protocol['max_total_drawdown_fraction'] = .99
        (self.session / 'protocol.json').write_text(json.dumps(protocol))
        result = run_cycle(self.session, self.inbox)
        self.assertFalse(result['ok'])
        self.assertEqual(result['fatal'], 'integrity_or_open_failure')
        self.assertIn('integrity_failure', (self.session / 'alerts.jsonl').read_text())
        self.assertFalse(any(self.session.glob('fills.csv')))

    def test_loop_is_bounded_and_stops_after_fatal(self):
        write_events(self.inbox / 'a.jsonl', self.events)
        sleeps = []
        results = run_loop(self.session, self.inbox, 3, interval_seconds=7., sleep=sleeps.append)
        self.assertEqual(len(results), 3)
        self.assertEqual(sleeps, [7., 7.])
        with self.assertRaises(ValueError):
            run_loop(self.session, self.inbox, 0)
        protocol = json.loads((self.session / 'protocol.json').read_text())
        protocol['minimum_forward_sessions'] = 1
        (self.session / 'protocol.json').write_text(json.dumps(protocol))
        self.assertEqual(len(run_loop(self.session, self.inbox, 5, sleep=lambda s: None)), 1)

    def test_bad_inbox_record_does_not_block_the_closing_packet(self):
        write_events(self.inbox / 'a.jsonl', self.events[:6], ['{broken'] + [json.dumps(self.events[6])])
        result = run_cycle(self.session, self.inbox)
        self.assertEqual(result['files']['a.jsonl']['accepted'], 7)
        self.assertEqual(result['files']['a.jsonl']['quarantined'], 1)
        self.assertEqual(liquidation_report(self.session)['exposure'], False)


def catalog_records():
    expiry = lambda s: int(pd.Timestamp(s).timestamp() * 1000)
    base = {'segment': 'NSE_FO', 'underlying_key': 'NSE_INDEX|Nifty 50', 'underlying_symbol': 'NIFTY', 'lot_size': 65}
    return [{**base, 'instrument_key': 'NSE_FO|SYNTHETIC_OPTION', 'instrument_type': 'CE', 'strike_price': 24000., 'tick_size': 5.,
             'expiry': expiry('2026-11-05T15:30:00+05:30'), 'trading_symbol': 'NIFTY SYNTH CE'},
            {**base, 'instrument_key': 'NSE_FO|SYNTHETIC_FUTURE', 'instrument_type': 'FUT', 'tick_size': 10.,
             'expiry': expiry('2026-11-26T15:30:00+05:30'), 'trading_symbol': 'NIFTY SYNTH FUT'}]


class CatalogTests(unittest.TestCase):
    NOW = pd.Timestamp('2026-11-02 09:00', tz=IST)

    def make(self, mutate=None, as_of='2026-11-02 08:00'):
        records = catalog_records()
        if mutate:
            mutate(records)
        return InstrumentCatalog(records, 'abc', pd.Timestamp(as_of, tz=IST))

    def test_matching_contracts_validate_with_provenance(self):
        report = self.make().validate(demo_contracts(), now=self.NOW)
        self.assertEqual(report['catalog_sha256'], 'abc')
        self.assertEqual(set(report['contracts']), {'option', 'future'})

    def test_each_kind_of_mismatch_is_rejected(self):
        cases = {
            'lot': lambda r: r[0].update(lot_size=75),
            'strike': lambda r: r[0].update(strike_price=24050.),
            'tick': lambda r: r[1].update(tick_size=5.),
            'underlying': lambda r: r[0].update(underlying_key='NSE_INDEX|Nifty Bank', underlying_symbol='BANKNIFTY'),
            'type': lambda r: r[0].update(instrument_type='PE'),
            'segment': lambda r: r[1].update(segment='NSE_EQ'),
            'missing': lambda r: r.pop(0),
            'expiry': lambda r: r[0].update(expiry=int(pd.Timestamp('2026-11-12T15:30:00+05:30').timestamp() * 1000)),
        }
        for name, mutate in cases.items():
            with self.assertRaises(CatalogError, msg=name):
                self.make(mutate).validate(demo_contracts(), now=self.NOW)

    def test_stale_snapshot_and_expired_contract_are_rejected(self):
        with self.assertRaises(CatalogError):
            self.make(as_of='2026-10-20 08:00').validate(demo_contracts(), now=self.NOW)
        with self.assertRaises(CatalogError):
            self.make().validate(demo_contracts(), now=pd.Timestamp('2026-11-06 09:00', tz=IST))

    def test_loader_reads_plain_and_gzipped_snapshots_and_freezes_with_provenance(self):
        with TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / 'master.json').write_text(json.dumps(catalog_records()))
            (folder / 'master.json.gz').write_bytes(gzip.compress(json.dumps(catalog_records()).encode()))
            plain = InstrumentCatalog.from_file(folder / 'master.json', self.NOW - pd.Timedelta(hours=1))
            zipped = InstrumentCatalog.from_file(folder / 'master.json.gz', self.NOW - pd.Timedelta(hours=1))
            self.assertEqual(len(plain.records), len(zipped.records))
            run = folder / 'run'
            run.mkdir()
            (run / 'experiment_plan.json').write_text(json.dumps({'config': {}, 'test_end': '2026-10-27T15:00:00+05:30'}))
            contracts = folder / 'contracts.json'
            contracts.write_text(json.dumps(demo_contracts()))
            protocol, report = freeze_validated(run, contracts, plain, folder / 'session', mode='synthetic_demo', now=self.NOW)
            saved = json.loads((folder / 'session/catalog_validation.json').read_text())
            self.assertEqual(saved['catalog_sha256'], plain.sha256)
            self.assertEqual(saved['protocol_sha256'], __import__('hashlib').sha256((folder / 'session/protocol.json').read_bytes()).hexdigest())
            bad = catalog_records()
            bad[0]['lot_size'] = 75
            (folder / 'bad.json').write_text(json.dumps(bad))
            with self.assertRaises(CatalogError):
                freeze_validated(run, contracts, InstrumentCatalog.from_file(folder / 'bad.json', self.NOW), folder / 'session2',
                                 mode='synthetic_demo', now=self.NOW)
            self.assertFalse((folder / 'session2').exists())            # nothing registered on failure


class StateBuilderTests(SessionCase):
    def builder(self):
        return CompletedBarStateBuilder(demo_contracts())

    def test_live_state_matches_the_batch_research_pipeline(self):
        from build_features import process_row
        from time_to_expiry import add_time_to_expiry
        starts = pd.date_range('2026-11-02 09:55', periods=6, freq='5min', tz=IST)
        spots, options = 24000. + np.arange(6) * 7, 100. + np.arange(6) * 1.5
        builder, live = self.builder(), []
        for s, sp, op in zip(starts, spots, options):
            live.append(builder.push(s, sp, op, sp + 10))
        frame = pd.DataFrame({'timestamp': starts, 'feature_available_at': starts + pd.Timedelta(minutes=5),
                              'expiry': pd.Timestamp('2026-11-05'), 'close': options, 'nifty_spot': spots,
                              'strike': 24000., 'option_type': 'CE'})
        frame = add_time_to_expiry(frame)
        for state, (_, row) in zip(live, frame.iterrows()):
            batch = process_row(row)
            self.assertAlmostEqual(state['remaining_time'], row.time_to_expiry, places=12)
            self.assertAlmostEqual(state['volatility'], batch.implied_volatility, places=9)
            self.assertAlmostEqual(state['delta'], batch.delta, places=9)

    def test_gaps_invalid_prices_and_order_never_carry_forward(self):
        b = self.builder()
        t0 = pd.Timestamp('2026-11-02 09:55', tz=IST)
        first = b.push(t0, 24000., 100., 24010.)
        self.assertEqual(first['recent_return'], 0.)
        second = b.push(t0 + pd.Timedelta(minutes=5), 24024., 101., 24034.)
        self.assertAlmostEqual(second['recent_return'], np.log(24024. / 24000.))
        bad = b.push(t0 + pd.Timedelta(minutes=10), float('nan'), 101., 24034.)
        self.assertIsInstance(bad, Unavailable)
        self.assertEqual(bad.reason, 'invalid_spot_close')
        after = b.push(t0 + pd.Timedelta(minutes=15), 24030., 102., 24040.)
        self.assertEqual(after['recent_return'], 0.)                    # previous bar was unusable: no stale return
        self.assertIsInstance(b.push(t0 + pd.Timedelta(minutes=15), 24030., 102., 24040.), Unavailable)
        self.assertIsInstance(b.push(t0, 24000., 100., 24010.), Unavailable)
        self.assertIsInstance(b.push(t0 + pd.Timedelta(minutes=20), 24000., 0.01, 24010.), Unavailable)   # below intrinsic -> no IV
        with self.assertRaises(ValueError):
            b.push(pd.Timestamp('2026-11-02 10:30'), 1., 1., 1.)

    def test_prefix_invariance(self):
        starts = pd.date_range('2026-11-02 09:55', periods=8, freq='5min', tz=IST)
        full, short = self.builder(), self.builder()
        series = [(s, 24000. + 3 * i, 100. + i, 24010. + 3 * i) for i, s in enumerate(starts)]
        a = [full.push(*row) for row in series]
        b = [short.push(*row) for row in series[:5]]
        self.assertEqual(a[:5], b)

    def test_builder_output_is_accepted_by_the_frozen_runner(self):
        b = self.builder()
        state = b.push(pd.Timestamp('2026-11-02 09:55', tz=IST), 24000., 100., 24010.)
        event = copy.deepcopy(self.events[0])
        record = decision_record('built_0', 'open', state)
        event.update(event_id='built_0', decision_at=record['decision_at'], state=state)
        runner = self.open_session()
        try:
            result = runner.process(event)
        finally:
            runner.close()
        self.assertEqual(result['status'], 'accepted', result)


if __name__ == '__main__':
    unittest.main()
