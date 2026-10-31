r"""30-second demonstration that the evidence chain detects tampering. Works on temporary copies; nothing real is modified.

Run (PowerShell):  $env:PYTHONPATH="src"; .\.venv\Scripts\python.exe tools\demo_tamper_check.py
Needs the Part 3 run folder for the spec and, for step 3, a finished paper session (e.g. data\\paper\\demo_20261030).
"""
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from forward_validation.protocol import freeze_protocol, load_protocol, verify_chain
from forward_validation.reconcile import reconcile_session
from forward_validation_cli import build_spec

RUN = ROOT / "data/research/historical_hedging_part3_final_20261031"


def say(text):
    print(text, flush=True)


def main():
    with TemporaryDirectory() as folder:
        folder = Path(folder)
        protocol, registry = folder / "protocol.json", folder / "registry.jsonl"
        say("1. Freeze a validation protocol (demo copy): criteria, code hashes and allowed models are fixed and hash-chained.")
        freeze_protocol(build_spec(RUN), protocol, registry)
        say(f"   loaded OK, version {load_protocol(protocol, registry)['version']}, registry chain valid: {verify_chain(registry)}")

        say("\n2. Someone quietly lowers the evidence bar from 60 to 3 days...")
        original = protocol.read_text()
        protocol.write_text(original.replace('"minimum_forward_sessions": 60', '"minimum_forward_sessions": 3'))
        try:
            load_protocol(protocol, registry)
            say("   NOT DETECTED (this would be a bug)")
        except ValueError as error:
            say(f"   REJECTED: {error}")
        protocol.write_text(original)

        say("\n3. Someone edits one simulated fill in a finished paper account's ledger...")
        demo = next(iter(sorted((ROOT / "data/paper").glob("demo_*/sessions/*unhedged"))), None)
        if demo is None:
            say("   (skipped: run the short replay first, see docs/DEMO_SCRIPT.md)")
            return
        copy = folder / "session"
        shutil.copytree(demo, copy)
        say(f"   before: reconciles = {reconcile_session(copy)['ok']}")
        db = sqlite3.connect(copy / "paper.sqlite")
        rowid, payload = db.execute("SELECT rowid, payload FROM fills ORDER BY rowid LIMIT 1").fetchone()
        fill = json.loads(payload)
        fill["fill"] += 0.5
        with db:
            db.execute("UPDATE fills SET payload=? WHERE rowid=?", (json.dumps(fill), rowid))
        db.close()
        result = reconcile_session(copy)
        say(f"   after:  reconciles = {result['ok']}  (max error INR {result['max_error']:,.2f} found by independent replay of the ledger)")
        say("\nThe evidence chain refuses edits to criteria, code and ledgers.")


if __name__ == "__main__":
    main()
