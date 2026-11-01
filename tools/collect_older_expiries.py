"""Collect expired NIFTY weekly expiries older than the main dataset into a SEPARATE raw folder (read-only API use).

The main collector writes into data/raw/upstox, which default scripts glob. Collecting older history there would silently change
every downstream dataset, so this wrapper redirects the output folder before the collector modules read it.

Run (PowerShell):  $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\collect_older_expiries.py --to-expiry 2025-09-08
Existing files in the target folder are never overwritten; an interrupted run can simply be repeated.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/upstox_older")
    parser.add_argument("--from-expiry", default=None)
    parser.add_argument("--to-expiry", default="2025-09-08", help="Last expiry to collect (the main dataset starts 2025-09-09)")
    parser.add_argument("--sessions-per-expiry", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    import config
    config.UPSTOX_RAW_DIR = args.output           # must happen before collect_all / collect_expiry are imported
    import collect_all
    assert collect_all.UPSTOX_RAW_DIR == args.output, "collector did not pick up the redirected folder"
    forwarded = ["--run-mode", "ALL", "--to-expiry", args.to_expiry, "--sessions-per-expiry", str(args.sessions_per_expiry)]
    if args.from_expiry:
        forwarded += ["--from-expiry", args.from_expiry]
    if args.plan_only:
        forwarded.append("--plan-only")
    sys.argv = ["collect_all.py", *forwarded]
    collect_all.main()


if __name__ == "__main__":
    main()
