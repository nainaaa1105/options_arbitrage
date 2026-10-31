"""Per-record envelope validation so one malformed packet can never block later close/risk events."""
import hashlib
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

REQUIRED = ("event_id", "type", "decision_at", "received_at")


def _reject_constant(name):
    raise ValueError(f"Non-finite JSON constant: {name}")


def _finite(value):
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, list):
        return all(_finite(v) for v in value)
    return True


def parse_line(line):
    """Return (event, None) or (None, reason). Never raises for bad content."""
    try:
        event = json.loads(line, parse_constant=_reject_constant)
    except (ValueError, RecursionError) as error:
        return None, f"invalid_json: {str(error)[:120]}"
    if not isinstance(event, dict):
        return None, "envelope_not_an_object"
    missing = [k for k in REQUIRED if k not in event]
    if missing:
        return None, f"missing_fields: {','.join(missing)}"
    if not isinstance(event["event_id"], str) or not event["event_id"].strip():
        return None, "invalid_event_id"
    if not _finite(event):
        return None, "non_finite_value"
    return event, None


def _seen(path):
    if not Path(path).exists():
        return set()
    out = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.add(json.loads(line)["line_sha256"])
        except (ValueError, KeyError):
            continue
    return out


def consume_file(session, events_file, quarantine_file):
    """Feed a JSONL file to an open PaperSession. Idempotent: duplicates are no-ops, quarantine is deduplicated."""
    seen = _seen(quarantine_file)
    summary = {"accepted": 0, "rejected": 0, "duplicate": 0, "quarantined": 0, "new_quarantine_records": 0}
    records = []

    def quarantine(number, line, reason):
        summary["quarantined"] += 1
        digest = hashlib.sha256(line.encode("utf-8", "replace")).hexdigest()
        if digest not in seen:
            seen.add(digest)
            summary["new_quarantine_records"] += 1
            records.append({"quarantined_at_utc": datetime.now(timezone.utc).isoformat(), "file": str(events_file),
                            "line_number": number, "line_sha256": digest, "reason": reason, "raw": line[:2000]})

    for number, line in enumerate(Path(events_file).read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        event, reason = parse_line(line)
        if reason:
            quarantine(number, line, reason)
            continue
        try:
            result = session.process(event)
        except sqlite3.OperationalError:
            raise                      # locked/corrupt store is an operational fault, not a bad record
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            quarantine(number, line, f"processing_error: {type(error).__name__}: {str(error)[:120]}")
            continue
        if result.get("duplicate"):
            summary["duplicate"] += 1
        else:
            summary["accepted" if result["status"] == "accepted" else "rejected"] += 1
    if records:
        with open(quarantine_file, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
    return summary
