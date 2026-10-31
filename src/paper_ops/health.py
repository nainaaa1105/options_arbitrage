"""Health assessment and deduplicated local alert log. Writes files only; sends nothing anywhere."""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forward_paper import inspect_session
from .liquidation import liquidation_report

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


def market_open(ts):
    local = pd.Timestamp(ts).tz_convert("Asia/Kolkata")
    return local.weekday() < 5 and "09:15" <= local.strftime("%H:%M") < "15:30"


def _count_lines(path):
    p = Path(path)
    return sum(1 for line in p.read_text(encoding="utf-8").splitlines() if line.strip()) if p.exists() else 0


def assess(session_dir, now=None, stale_seconds=310., quarantine_file=None):
    d = Path(session_dir)
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    alerts = []

    def alert(code, severity, detail):
        alerts.append({"code": code, "severity": severity, "detail": detail})

    try:
        info = inspect_session(d)
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        alert("session_unreadable", "critical", f"{type(error).__name__}: {error}")
        return {"session": d.name, "checked_at_utc": now.isoformat(), "alerts": alerts, "healthy": False}
    protocol = json.loads((d / "protocol.json").read_text())
    horizon = protocol["execution_config"]["horizon_minutes"]
    s = info["state"] or {}
    counts = info["event_counts"]
    if not info["protocol_hash_matches"] or not info["runner_matches"]:
        alert("integrity_failure", "critical", f"protocol_hash_matches={info['protocol_hash_matches']} runner_matches={info['runner_matches']}")
    exposed = bool(s.get("q") or s.get("h"))
    if s.get("halted"):
        alert("risk_stop_active", "critical", s.get("liquidation_reason", "risk stop latched"))
    if s.get("pending_liquidation"):
        alert("pending_liquidation", "critical", "closing packet failed; valid close quotes required")
    last = pd.Timestamp(s["last_received_at"]) if s.get("last_received_at") else None
    age = (now - last).total_seconds() if last is not None else None
    if exposed and s.get("entry_at") and now > pd.Timestamp(s["entry_at"]) + pd.Timedelta(minutes=horizon):
        alert("overdue_open_position", "critical", "holding horizon elapsed with inventory still open")
    if exposed and market_open(now) and (age is None or age > stale_seconds):
        alert("stale_feed_while_exposed", "critical", f"seconds_since_last_event={age}")
    elif protocol["mode"] == "forward" and market_open(now) and (age is None or age > 6 * stale_seconds):
        alert("no_recent_data", "warning", f"seconds_since_last_event={age}")
    total = sum(counts.values())
    if total >= 10 and counts.get("rejected", 0) / total >= .5:
        alert("high_rejection_rate", "warning", f"{counts.get('rejected', 0)}/{total} events rejected")
    if s.get("missed_boundaries", 0):
        alert("missed_decision_boundaries", "warning", f"count={s['missed_boundaries']}")
    quarantined = _count_lines(quarantine_file or d / "quarantine.jsonl")
    if quarantined:
        alert("quarantined_input", "warning", f"{quarantined} malformed or conflicting records quarantined")
    liquidation = liquidation_report(d, now)
    if exposed and not liquidation["executable"]:
        alert("cannot_liquidate_on_displayed_book", "critical", liquidation.get("reason", "unknown"))
    alerts.sort(key=lambda a: (SEVERITY_ORDER[a["severity"]], a["code"]))
    return {"session": d.name, "mode": protocol["mode"], "checked_at_utc": now.isoformat(), "seconds_since_last_event": age,
            "event_counts": counts, "state_flags": {k: s.get(k) for k in ("halted", "pending_liquidation", "q", "h")},
            "liquidation": liquidation, "quarantined_records": quarantined, "alerts": alerts,
            "healthy": not any(a["severity"] in ("critical", "warning") for a in alerts)}


def append_alerts(alert_file, report):
    """Append only alerts not already logged for the same session/day/code/detail."""
    path = Path(alert_file)
    seen = set()
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                seen.add((r["session"], r["day"], r["code"], r["detail"]))
            except (ValueError, KeyError):
                continue
    day = pd.Timestamp(report["checked_at_utc"]).tz_convert("Asia/Kolkata").date().isoformat()
    fresh = []
    for a in report["alerts"]:
        key = (report["session"], day, a["code"], a["detail"])
        if key not in seen:
            seen.add(key)
            fresh.append({"session": report["session"], "day": day, "logged_at_utc": datetime.now(timezone.utc).isoformat(), **a})
    if fresh:
        with open(path, "a", encoding="utf-8") as handle:
            for r in fresh:
                handle.write(json.dumps(r) + "\n")
    return fresh
