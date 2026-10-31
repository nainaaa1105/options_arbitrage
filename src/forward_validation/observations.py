"""Persistent, idempotent observation records derived from paper sessions (read-only on the session store)."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .protocol import session_conformance

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(session TEXT PRIMARY KEY, strategy TEXT, mode TEXT, created_at_utc TEXT, conforms INTEGER,
    violations TEXT, total_events INTEGER, quarantined INTEGER, ingested_at_utc TEXT, contract TEXT);
CREATE TABLE IF NOT EXISTS daily(session TEXT, day TEXT, events INTEGER, accepted INTEGER, rejected INTEGER, fills INTEGER, fees REAL,
    execution_impact REAL, equity_close REAL, net_pnl_close REAL, drawdown_fraction REAL, signal_eligible_events INTEGER,
    exposure_at_close INTEGER, forward_eligible INTEGER, reject_reasons TEXT, PRIMARY KEY(session, day));
CREATE TABLE IF NOT EXISTS episodes(session TEXT, open_decision_at TEXT, day TEXT, net_pnl REAL, forward_eligible INTEGER,
    PRIMARY KEY(session, open_decision_at));
"""


def open_store(path):
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    return db


def _read_session(directory):
    uri = (Path(directory) / "paper.sqlite").resolve().as_uri() + "?mode=ro"
    db = sqlite3.connect(uri, uri=True)
    try:
        events = db.execute("SELECT event_id,payload,status,detail FROM events ORDER BY rowid").fetchall()
        equity = {k: json.loads(v) for k, v in db.execute("SELECT event_id,payload FROM equity")}
        fills = {}
        for k, v in db.execute("SELECT event_id,payload FROM fills ORDER BY rowid"):
            fills.setdefault(k, []).append(json.loads(v))
    finally:
        db.close()
    return events, equity, fills


def ingest_session(protocol, session_dir, store_path, now=None):
    """Aggregate one session into the store. Re-ingesting the same session replaces its rows (idempotent)."""
    d = Path(session_dir)
    session = json.loads((d / "protocol.json").read_text())
    initial = session["execution_config"]["initial_capital"]
    frozen_at = pd.Timestamp(protocol["frozen_at_utc"]).tz_convert("UTC")
    violations = session_conformance(protocol, d)
    conforms = not violations
    events, equity, fills = _read_session(d)
    quarantine = d / "quarantine.jsonl"
    quarantined = sum(1 for l in quarantine.read_text(encoding="utf-8").splitlines() if l.strip()) if quarantine.exists() else 0
    days, episodes = {}, []
    peak, previous_equity, open_event, equity_before_open = initial, initial, None, initial
    for event_id, payload, status, detail in events:
        event = json.loads(payload)
        received = pd.Timestamp(event["received_at"]).tz_convert("UTC")
        decision = pd.Timestamp(event["decision_at"]).tz_convert("UTC")
        day = received.tz_convert("Asia/Kolkata").date().isoformat()
        eligible = bool(conforms and session["mode"] == "forward" and decision > frozen_at)
        row = days.setdefault(day, {"events": 0, "accepted": 0, "rejected": 0, "fills": 0, "fees": 0., "impact": 0., "equity": None,
                                    "dd": 0., "signal": 0, "exposure": 0, "eligible": eligible, "reasons": {}})
        row["events"] += 1
        row["eligible"] = row["eligible"] and eligible
        if event.get("state", {}).get("signal_eligible"):
            row["signal"] += 1
        if status != "accepted":
            row["rejected"] += 1
            row["reasons"][detail] = row["reasons"].get(detail, 0) + 1
            continue
        mark = equity[event_id]
        row["accepted"] += 1
        row["fills"] += len(fills.get(event_id, []))
        row["equity"] = mark["equity"]
        row["fees"], row["impact"] = mark["fees"], mark["execution_impact"]
        peak = max(peak, mark["equity"])
        row["dd"] = max(row["dd"], (peak - mark["equity"]) / peak if peak > 0 else 0.)
        row["exposure"] = int(bool(mark["option_quantity"] or mark["future_quantity"]))
        if event["type"] == "open":
            open_event, equity_before_open = event["decision_at"], previous_equity
        if open_event and not (mark["option_quantity"] or mark["future_quantity"]):
            episodes.append((d.name, open_event, day, mark["equity"] - equity_before_open, int(eligible)))
            open_event = None
        previous_equity = mark["equity"]
    db = open_store(store_path)
    with db:
        db.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?,?,?,?,?)",
                   (d.name, session["strategy"], session["mode"], session["created_at_utc"], int(conforms), json.dumps(violations),
                    len(events), quarantined, (now or datetime.now(timezone.utc)).isoformat(),
                    session["contracts"]["option"]["instrument_key"]))
        db.execute("DELETE FROM daily WHERE session=?", (d.name,))
        db.execute("DELETE FROM episodes WHERE session=?", (d.name,))
        for day, r in sorted(days.items()):
            db.execute("INSERT INTO daily VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (d.name, day, r["events"], r["accepted"], r["rejected"], r["fills"], r["fees"], r["impact"], r["equity"],
                        None if r["equity"] is None else r["equity"] - initial, r["dd"], r["signal"], r["exposure"],
                        int(r["eligible"]), json.dumps(r["reasons"], sort_keys=True)))
        db.executemany("INSERT INTO episodes VALUES (?,?,?,?,?)", episodes)
    db.close()
    return {"session": d.name, "conforms": conforms, "violations": violations, "days": len(days), "episodes": len(episodes)}
