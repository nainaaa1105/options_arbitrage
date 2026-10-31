"""Bounded, restart-safe, single-writer orchestration of the paper pipeline.

Cycle: inbox JSONL packets -> quarantine-safe consume -> export -> health/liquidation assessment -> local alerts -> heartbeat.
Everything is file based and idempotent; nothing is sent over a network and no broker order exists.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forward_paper import PaperSession
from .health import append_alerts, assess
from .quarantine import consume_file


class LockHeld(RuntimeError):
    pass


class SingleWriterLock:
    """Exclusive-create lock file. Staleness is age based (no pid probing: os.kill(pid, 0) terminates on Windows)."""

    def __init__(self, path, stale_after_seconds=900.):
        self.path, self.stale = Path(path), stale_after_seconds
        self.token = None
        self.took_over = False

    def _create(self):
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        self.token = hashlib.sha256(f"{os.getpid()}|{time.time_ns()}".encode()).hexdigest()[:16]
        with os.fdopen(fd, "w") as handle:
            json.dump({"pid": os.getpid(), "token": self.token, "created_utc": datetime.now(timezone.utc).isoformat()}, handle)

    def __enter__(self):
        try:
            self._create()
        except FileExistsError:
            age = time.time() - self.path.stat().st_mtime
            if age <= self.stale:
                raise LockHeld(f"Another writer holds {self.path.name} (age {age:.0f}s)") from None
            self.path.unlink()
            self.took_over = True
            try:
                self._create()
            except FileExistsError:
                raise LockHeld("Lost a race while replacing a stale lock") from None
        return self

    def touch(self):
        os.utime(self.path, None)

    def __exit__(self, *exc):
        try:
            if json.loads(self.path.read_text()).get("token") == self.token:
                self.path.unlink()
        except (OSError, ValueError):
            pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cycle(session, inbox, now=None, retries=2, backoff_seconds=.2, sleep=time.sleep, lock_stale_seconds=900.):
    """One bounded pass. Returns a summary dict; raises LockHeld if another writer is active."""
    session, inbox = Path(session), Path(inbox)
    started = time.monotonic()
    summary = {"session": session.name, "started_utc": datetime.now(timezone.utc).isoformat(), "files": {}, "errors": [], "ok": True}
    ledger_path = session / "processed_inbox.json"
    with SingleWriterLock(session / "service.lock", lock_stale_seconds) as lock:
        summary["lock_takeover"] = lock.took_over
        try:
            runner = PaperSession(session)          # integrity gates live here; never bypassed
        except (ValueError, OSError, KeyError, sqlite3.Error) as error:
            summary.update(ok=False, fatal="integrity_or_open_failure")
            summary["errors"].append(f"{type(error).__name__}: {error}")
            report = {"session": session.name, "checked_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                      "alerts": [{"code": "integrity_failure", "severity": "critical", "detail": str(error)[:300]}], "healthy": False}
            append_alerts(session / "alerts.jsonl", report)
            (session / "health.json").write_text(json.dumps(report, indent=2))
            _heartbeat(session, summary, started)
            return summary
        try:
            ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
            for file in sorted(inbox.glob("*.jsonl")):
                digest = _sha(file)
                if ledger.get(file.name) == digest:
                    summary["files"][file.name] = "unchanged_skipped"
                    continue
                for attempt in range(retries + 1):
                    try:
                        summary["files"][file.name] = consume_file(runner, file, session / "quarantine.jsonl")
                        ledger[file.name] = digest
                        break
                    except sqlite3.OperationalError as error:
                        if attempt == retries:
                            summary.update(ok=False)
                            summary["errors"].append(f"{file.name}: {error}")
                        else:
                            sleep(backoff_seconds * (attempt + 1))
                lock.touch()
            ledger_path.write_text(json.dumps(ledger, indent=2, sort_keys=True))
            runner.export()
        finally:
            runner.close()
        report = assess(session, now=now, quarantine_file=session / "quarantine.jsonl")
        (session / "health.json").write_text(json.dumps(report, indent=2, default=str))
        summary["new_alerts"] = len(append_alerts(session / "alerts.jsonl", report))
        summary["healthy"] = report["healthy"]
        _heartbeat(session, summary, started)
    return summary


def _heartbeat(session, summary, started):
    path = Path(session) / "heartbeat.json"
    previous = json.loads(path.read_text()) if path.exists() else {"cycles": 0}
    beat = {"cycles": previous.get("cycles", 0) + 1, "last_cycle_utc": datetime.now(timezone.utc).isoformat(),
            "last_cycle_seconds": round(time.monotonic() - started, 3), "last_ok": summary["ok"], "last_errors": summary["errors"][-3:]}
    path.write_text(json.dumps(beat, indent=2))


def run_loop(session, inbox, max_cycles, interval_seconds=60., sleep=time.sleep, now_fn=None):
    """Bounded supervisor: never runs forever. Stops early on a fatal integrity failure."""
    if max_cycles < 1:
        raise ValueError("max_cycles must be a positive bound")
    results = []
    for index in range(max_cycles):
        try:
            result = run_cycle(session, inbox, now=now_fn() if now_fn else None)
        except LockHeld as error:
            result = {"ok": False, "errors": [str(error)], "skipped": "lock_held"}
        results.append(result)
        if result.get("fatal"):
            break
        if index + 1 < max_cycles:
            sleep(interval_seconds)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--interval-seconds", type=float, default=60.)
    args = parser.parse_args()
    results = run_loop(args.session, args.inbox, args.cycles, args.interval_seconds)
    print(json.dumps(results, indent=2, default=str))
    raise SystemExit(0 if all(r.get("ok") for r in results) else 2)


if __name__ == "__main__":
    main()
