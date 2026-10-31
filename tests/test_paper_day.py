"""Daily runner tests on a SYNTHETIC replay session (software verification only; not market evidence)."""
import json
from pathlib import Path
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from black_scholes import black_scholes_price
from forward_validation.reconcile import reconcile_session
from paper_ops.catalog import InstrumentCatalog
from paper_ops.contracts import select_contracts
from paper_ops.daily import register_sessions, run_day, schedule, session_name
from paper_ops.market import INDEX_KEY, ReplayMarket
from test_paper_ops_contracts import records

IST = "Asia/Kolkata"
DAY = "2026-11-02"


def frames_for(selection, drop=None):
    times = pd.date_range(f"{DAY} 09:15", f"{DAY} 15:25", freq="5min", tz=IST)
    rng = np.random.default_rng(4)
    spot = 23050. * np.exp(np.cumsum(rng.normal(0, .0004, len(times))))
    future_key = next(iter(selection.values()))["future"]["instrument_key"]

    def frame(price):
        return pd.DataFrame({"timestamp": times, "open": price, "high": price + .1, "low": price - .1, "close": price, "volume": 100, "open_interest": 1000})
    out = {INDEX_KEY: frame(spot), future_key: frame(spot + 20.)}
    expiry = pd.Timestamp("2026-11-09 15:30", tz=IST)
    for c in selection.values():
        o = c["option"]
        t = np.maximum((expiry - times).total_seconds() / (365 * 86400), 1e-6)
        price = np.array([black_scholes_price(s, o["strike"], tt, .2, .06, o["option_type"]) for s, tt in zip(spot, t)])
        out[o["instrument_key"]] = frame(price + .05)
    if drop:
        key, when = drop
        out[key] = out[key][out[key].timestamp != pd.Timestamp(f"{DAY} {when}", tz=IST)].reset_index(drop=True)
    return out


class NoQuoteAt(ReplayMarket):
    def __init__(self, frames, skip):
        super().__init__(frames)
        self.skip = {pd.Timestamp(f"{DAY} {s}", tz=IST) for s in skip}

    def quotes(self, keys, at):
        if pd.Timestamp(at) in self.skip:
            return pd.Timestamp(at).tz_convert("UTC"), {"status": "success", "data": {}}
        return super().quotes(keys, at)


class DailyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.base = Path(self.tmp.name) / "paper"
        run = Path(self.tmp.name) / "run"
        run.mkdir()
        (run / "experiment_plan.json").write_text(json.dumps({"config": {}, "test_end": "2026-10-27T15:00:00+05:30"}))
        self.run_dir = run
        self.selection = select_contracts(records(), pd.Timestamp(DAY), 23050.)
        self.catalog = InstrumentCatalog(records(), "synthetic", pd.Timestamp(f"{DAY} 07:00", tz=IST))
        self.now = pd.Timestamp(f"{DAY} 08:30", tz=IST)

    def tearDown(self):
        self.tmp.cleanup()

    def register(self, strategies=("unhedged", "dynamic_delta")):
        return register_sessions(self.base, self.selection, self.run_dir, self.catalog, "replay", strategies=strategies, now=self.now)

    def counts(self, folder):
        db = sqlite3.connect(folder / "paper.sqlite")
        try:
            return dict(db.execute("SELECT status, COUNT(*) FROM events GROUP BY status").fetchall()), db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        finally:
            db.close()

    def test_schedule_has_six_episodes_of_seven_events_plus_close_retries(self):
        events = schedule(DAY)
        self.assertEqual([k for _, k, _ in events].count("open"), 6)
        self.assertEqual([k for _, k, _ in events].count("hedge"), 30)
        self.assertEqual([k for _, k, _ in events].count("close"), 6)
        self.assertEqual(events[0][0], pd.Timestamp(f"{DAY} 09:35", tz=IST))
        self.assertLessEqual(max(t for t, _, _ in events), pd.Timestamp(f"{DAY} 15:25", tz=IST))
        self.assertEqual(len({(t) for t, _, _ in events}), len(events))

    def test_full_clean_day_opens_hedges_and_closes_every_episode_and_reconciles(self):
        sessions = self.register()
        counters = run_day(self.base, DAY, ReplayMarket(frames_for(self.selection)), self.selection, sessions, quiet=True)
        self.assertEqual(counters["unavailable"], 0)
        for (label, strategy), folder in sessions.items():
            status, state = self.counts(folder)
            self.assertEqual(status.get("accepted"), 42, (label, strategy, status))
            self.assertNotIn("rejected", status)
            s = json.loads(state[0])
            self.assertEqual((s["q"], s["h"]), (0, 0))
            self.assertTrue(reconcile_session(folder)["ok"])
        names = {f.name for f in (self.base / "sessions").iterdir()}
        self.assertEqual(len(names), 8)
        self.assertIn(session_name("ATM_CE", self.selection["ATM_CE"], "unhedged"), names)

    def test_a_missing_bar_skips_that_event_without_fabrication_and_the_episode_still_closes(self):
        sessions = self.register(("dynamic_delta",))
        option = self.selection["ATM_CE"]["option"]["instrument_key"]
        market = ReplayMarket(frames_for(self.selection, drop=(option, "10:00")))      # 10:05 completed bar of one option is absent
        counters = run_day(self.base, DAY, market, self.selection, sessions, quiet=True)
        self.assertGreaterEqual(counters["unavailable"], 1)
        log = (self.base / "unavailable" / f"{DAY}.jsonl").read_text()
        self.assertTrue("missing_completed_bar" in log or "option_quote_missing" in log)         # which one depends on the event type
        folder = sessions[("ATM_CE", "dynamic_delta")]
        status, state = self.counts(folder)
        s = json.loads(state[0])
        self.assertEqual((s["q"], s["h"]), (0, 0))                                      # still flat at day end
        other = self.counts(sessions[("ATM_PE", "dynamic_delta")])[0]
        self.assertEqual(other.get("accepted"), 42)                                     # other contracts unaffected

    def test_a_failed_close_is_retried_at_the_next_boundary(self):
        sessions = self.register(("unhedged",))
        market = NoQuoteAt(frames_for(self.selection), skip=["10:05"])                  # first close of slot 1 has no quote
        counters = run_day(self.base, DAY, market, self.selection, sessions, quiet=True)
        self.assertGreaterEqual(counters["unavailable"], 4)
        for folder in sessions.values():
            s = json.loads(self.counts(folder)[1][0])
            self.assertEqual((s["q"], s["h"]), (0, 0))
            self.assertTrue(reconcile_session(folder)["ok"])
        retried = [json.loads(l) for l in (self.base / "inbox" / sessions[("ATM_CE", "unhedged")].name / "events.jsonl").read_text().splitlines()]
        self.assertTrue(any(e["event_id"].endswith("retry_close") for e in retried))

    def test_packets_carry_no_future_information(self):
        sessions = self.register(("unhedged",))
        run_day(self.base, DAY, ReplayMarket(frames_for(self.selection)), self.selection, sessions, quiet=True)
        folder = sessions[("ATM_CE", "unhedged")]
        for line in (self.base / "inbox" / folder.name / "events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if "state" in e:
                self.assertEqual(pd.Timestamp(e["state"]["available_at"]), pd.Timestamp(e["decision_at"]))
                self.assertLess(pd.Timestamp(e["state"]["bar_start"]), pd.Timestamp(e["decision_at"]))
            quote_times = [pd.Timestamp(q["quote_time"]) for q in e["quotes"].values()]
            self.assertTrue(all(q > pd.Timestamp(e["decision_at"]) for q in quote_times))


if __name__ == "__main__":
    unittest.main()
