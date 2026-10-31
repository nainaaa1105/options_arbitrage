import copy
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from forward_paper import freeze, PaperSession, inspect_session
from run_paper_demo import demo_contracts, demo_events
from assemble_paper_events import assemble
from build_project_dashboard import build


class PaperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root/"run"
        self.run.mkdir()
        (self.run/"experiment_plan.json").write_text(json.dumps({"config": {}, "test_end": "2026-10-27T15:00:00+05:30"}))
        self.contracts = self.root/"contracts.json"
        self.contracts.write_text(json.dumps(demo_contracts()))
        self.session = self.root/"paper"
        freeze(self.run, self.contracts, self.session, mode="synthetic_demo")
        self.runner = PaperSession(self.session)

    def tearDown(self):
        self.runner.close()
        self.tmp.cleanup()

    def test_whole_episode_cash_reconciles_and_inventories_close(self):
        for event in demo_events():
            result = self.runner.process(event)
            self.assertEqual(result["status"], "accepted", result)
            self.assertLess(abs(result["equity"]["reconciliation_error"]), 1e-6)
        s = self.runner.state()
        self.assertEqual((s["q"], s["h"]), (0, 0))
        self.assertGreater(s["fees"], 0)
        self.assertEqual(self.runner.export()["mode"], "synthetic_demo")

    def test_restart_and_duplicate_do_not_repeat_orders(self):
        e = demo_events()[0]
        self.runner.process(e)
        before = self.runner.state()
        self.runner.close()
        self.runner = PaperSession(self.session)
        self.assertTrue(self.runner.process(e)["duplicate"])
        self.assertEqual(before, self.runner.state())
        self.assertEqual(self.runner.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0], 2)
        changed = copy.deepcopy(e)
        changed["state"]["delta"] = .6
        with self.assertRaisesRegex(ValueError, "Conflicting duplicate"):
            self.runner.process(changed)

    def test_bad_quotes_reject_atomically(self):
        for change in ("stale", "crossed", "depth", "predecision", "wrong_instrument", "skew"):
            event = copy.deepcopy(demo_events()[0])
            event["event_id"] = change
            q = event["quotes"]["future"]
            if change == "stale": q["quote_time"] = "2026-11-02T09:59:00+05:30"
            if change == "crossed": q["bid"] = q["ask"]+1
            if change == "depth": q["bid_size"] = 1
            if change == "predecision": q["quote_time"] = event["decision_at"]
            if change == "wrong_instrument": q["instrument_key"] = "NSE_INDEX|Nifty 50"
            if change == "skew": event["quotes"]["option"]["quote_time"] = "2026-11-02T09:59:59+05:30"
            self.assertEqual(self.runner.process(event)["status"], "rejected", change)
            self.assertEqual(self.runner.state()["q"], 0)
            self.assertEqual(self.runner.db.execute("SELECT COUNT(*) FROM fills").fetchone()[0], 0)

    def test_future_or_stale_feature_state_is_rejected(self):
        for when in ("2026-11-02T10:05:00+05:30", "2026-11-02T09:55:00+05:30"):
            event = copy.deepcopy(demo_events()[0])
            event["event_id"] = when
            event["state"]["available_at"] = when
            self.assertEqual(self.runner.process(event)["status"], "rejected")

    def test_clock_maturity_mismatch_is_rejected(self):
        event = demo_events()[0]
        event["state"]["remaining_time"] *= 365/252
        self.assertEqual(self.runner.process(event)["status"], "rejected")

    def test_missed_boundary_recovers_with_audited_gap(self):
        self.runner.process(demo_events()[0])
        event = demo_events()[2]
        self.assertEqual(self.runner.process(event)["status"], "accepted")
        self.assertEqual(self.runner.state()["missed_boundaries"], 1)
        self.assertEqual(self.runner.process(demo_events()[3])["status"], "accepted")

    def test_failed_risk_liquidation_latches_across_restart_and_recovers_only_by_closing(self):
        self.runner.process(demo_events()[0])
        event = demo_events()[1]
        event["quotes"]["future"].update(bid=25000., ask=25000.2, ask_size=1, bid_size=1)
        self.assertEqual(self.runner.process(event)["status"], "rejected")
        self.assertTrue(self.runner.state()["halted"])
        self.assertTrue(self.runner.state()["pending_liquidation"])
        self.assertEqual(self.runner.state()["q"], 260)
        self.runner.close()
        self.runner = PaperSession(self.session)
        result = self.runner.process(demo_events()[2])
        self.assertEqual(result["status"], "accepted", result)
        self.assertTrue(result["equity"]["forced_close"])
        self.assertEqual(self.runner.state()["q"], 0)
        self.assertTrue(self.runner.state()["halted"])

    def test_missing_liquidation_depth_keeps_exposure_and_alerts(self):
        self.runner.process(demo_events()[0])
        event = demo_events()[-1]
        event["quotes"]["option"]["bid_size"] = 1
        result = self.runner.process(event)
        self.assertEqual(result["status"], "rejected")
        self.assertGreater(self.runner.state()["q"], 0)
        self.assertIn("leaves inventory open", self.runner.export()["risk_note"])

    def test_margin_failure_does_not_partially_open_option(self):
        event = demo_events()[0]
        event["quotes"]["option"]["ask"] = 100000.
        self.assertEqual(self.runner.process(event)["status"], "rejected")
        self.assertEqual(self.runner.state()["cash"], 5000000.)
        self.assertEqual(self.runner.state()["q"], 0)

    def test_horizon_automatically_closes_hedge_request(self):
        self.runner.process(demo_events()[0])
        event = demo_events()[-1]
        event["type"] = "hedge"
        result = self.runner.process(event)
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["equity"]["forced_close"])
        self.assertEqual(self.runner.state()["q"], 0)

    def test_protocol_tampering_detected(self):
        (self.session/"protocol.json").write_text("{}")
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            PaperSession(self.session)

    def test_old_session_read_only_inspection_does_not_change_account(self):
        self.runner.process(demo_events()[0])
        before=self.runner.state()
        report=inspect_session(self.session)
        self.assertTrue(report['read_only'])
        self.assertFalse(report['execution_authorized_by_inspection'])
        self.assertEqual(report['state'],before)
        self.assertEqual(self.runner.state(),before)

    def test_forward_mode_rejects_historical_or_future_receipt(self):
        directory = self.root/"forward"
        freeze(self.run, self.contracts, directory, mode="forward")
        runner = PaperSession(directory)
        try:
            event = demo_events()[0]
            event["decision_at"] = "2026-10-20T10:00:00+05:30"
            self.assertEqual(runner.process(event)["status"], "rejected")
        finally:
            runner.close()

    def test_quote_adapter_keeps_unavailable_decisions_visible(self):
        event = demo_events()[0]
        decisions = self.root/"decisions.jsonl"
        decisions.write_text(json.dumps({k:v for k,v in event.items() if k not in ("received_at", "quotes")}))
        rows = [{**q, "received_at": event["received_at"], "executable_quote_valid": True} for q in event["quotes"].values()]
        quotes = self.root/"quotes.parquet"
        pd.DataFrame(rows).to_parquet(quotes)
        events, unavailable = assemble(self.session, decisions, quotes, self.root/"packets")
        self.assertEqual((len(events),len(unavailable)), (1,0))
        pd.DataFrame(rows[:1]).to_parquet(quotes)
        events, unavailable = assemble(self.session, decisions, quotes, self.root/"missing")
        self.assertEqual((len(events),len(unavailable)), (0,1))

    def test_daily_loss_stop_closes_and_blocks_further_entries(self):
        self.runner.process(demo_events()[0])
        event = demo_events()[1]
        event["quotes"]["future"].update(bid=25000., ask=25000.2)
        result = self.runner.process(event)
        self.assertEqual(result["status"], "accepted", result)
        self.assertTrue(result["equity"]["halted"])
        self.assertEqual(self.runner.state()["q"], 0)
        following = demo_events()[2]
        following["type"] = "open"
        self.assertEqual(self.runner.process(following)["status"], "rejected")

    def test_dashboard_uses_artifacts_and_escapes_session_names(self):
        run = Path(__file__).resolve().parents[1]/"data/research/historical_hedging_part3_final_20261031"
        if not run.exists():
            self.skipTest("Historical artifacts not installed")
        malicious = self.root/"malicious_label"
        malicious.mkdir()
        (malicious/"status.json").write_text(json.dumps({"mode": "synthetic_demo", "strategy": "<script>", "accepted_events": 0,
             "rejected_events": 0, "state": {"q": 0,"h": 0,"halted": False}}))
        output = self.root/"dashboard.html"
        build(run, output, [malicious])
        text = output.read_text(encoding="utf-8")
        self.assertIn("Retrospective research", text)
        self.assertNotIn('"strategy": "<script>"', text)
        self.assertIn('\\u003cscript\\u003e', text)
        with self.assertRaises(FileExistsError):
            build(run, output)


if __name__ == "__main__":
    unittest.main()
