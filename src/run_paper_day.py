"""Daily paper-trading driver (read-only market data, simulated fills, no orders).

  replay : run one past session from stored real candles with synthetic quotes (never counts as evidence)
  live   : run today's session against live Upstox candles and quotes
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forward_paper import inspect_session
from forward_validation.reconcile import reconcile_session
from paper_ops.catalog import InstrumentCatalog
from paper_ops.contracts import load_master, needs_reroll, select_contracts
from paper_ops.daily import STRATEGIES, register_sessions, run_day
from paper_ops.health import assess
from paper_ops.market import INDEX_KEY, ReplayMarket, UpstoxMarket, fetch_replay_frames

IST = "Asia/Kolkata"
DEFAULT_RUN = Path("data/research/historical_hedging_part3_final_20261031")
MASTER_DIR = Path("data/raw/upstox/instruments")


def latest_master():
    """Newest dated snapshot in the instruments folder, so a refreshed file is picked up without editing code."""
    found = sorted(MASTER_DIR.glob("NSE_*.json.gz"))
    return found[-1] if found else MASTER_DIR / "NSE_missing.json.gz"


DEFAULT_MASTER = latest_master()


def preflight(master, live):
    problems = []
    if not Path(master).exists():
        problems.append(f"instrument master missing: {master}")
    else:
        age = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(Path(master).stat().st_mtime, unit="s", tz="UTC")).total_seconds() / 86400
        if age > 1.5:
            problems.append(f"instrument master is {age:.1f} days old; refresh it before registering contracts")
    if live:
        try:
            UpstoxMarket().quotes([INDEX_KEY])
        except Exception as error:                              # authentication, network or entitlement failure
            problems.append(f"Upstox quote check failed: {str(error)[:150]}")
        now = pd.Timestamp.now(tz=IST)
        if now.weekday() >= 5:
            problems.append("today is a weekend; there is no session")
    return problems


def finish(base, day, sessions):
    rows = []
    for (label, strategy), folder in sorted(sessions.items()):
        info = inspect_session(folder)
        state, reconciliation = info["state"] or {}, reconcile_session(folder)
        health = assess(folder, now=pd.Timestamp(f"{day} 15:35", tz=IST) if pd.Timestamp(day).date() < pd.Timestamp.now(tz=IST).date() else None)
        rows.append({"contract": label, "strategy": strategy, "events": info["event_counts"], "open_option_units": state.get("q"),
                     "open_future_units": state.get("h"), "cash": state.get("cash"), "fees": state.get("fees"), "reconciles": reconciliation["ok"],
                     "max_error": reconciliation["max_error"], "alerts": [a["code"] for a in health["alerts"]]})
    report = {"day": str(day), "generated_utc": datetime.now(timezone.utc).isoformat(), "sessions": rows,
              "all_reconcile": all(r["reconciles"] for r in rows), "any_open_exposure": any((r["open_option_units"] or r["open_future_units"]) for r in rows)}
    (Path(base) / f"daily_report_{day}.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("replay", "live"):
        s = sub.add_parser(name)
        s.add_argument("--base", type=Path, required=True)
        s.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
        s.add_argument("--master", type=Path, default=DEFAULT_MASTER)
        if name == "replay":
            s.add_argument("--date", required=True)
            s.add_argument("--only", nargs="+", help="contract labels to run, e.g. ATM_CE (short demo)")
            s.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=list(STRATEGIES))
        else:
            s.add_argument("--mode", choices=["forward", "replay"], default="forward",
                           help="forward counts as evidence only if the session is registered after the validation protocol freeze")
    args = parser.parse_args(argv)
    live = args.command == "live"
    problems = preflight(args.master, live)
    if problems:
        print("PREFLIGHT FAILED:\n  " + "\n  ".join(problems))
        return 2
    records = load_master(args.master)
    catalog = InstrumentCatalog.from_file(args.master, pd.Timestamp(args.master.stat().st_mtime, unit="s", tz="UTC").tz_convert(IST))
    args.base.mkdir(parents=True, exist_ok=True)
    if live:
        day, market, mode = pd.Timestamp.now(tz=IST).date(), UpstoxMarket(), args.mode
        _, payload = market.quotes([INDEX_KEY])
        spot = float(next(iter(payload["data"].values()))["last_price"])
    else:
        day, mode = pd.Timestamp(args.date).date(), "replay"
        index_frames = fetch_replay_frames([INDEX_KEY], day)
        first = index_frames[INDEX_KEY].iloc[0]
        spot = float(first.close)                               # 09:15 bar closes at 09:20: the same spot a live 09:20 run would use
    selection = None
    previous = args.base / "active.json"
    if live and previous.exists():                              # keep existing accounts unless the roll rule fires
        held = json.loads(previous.read_text())["selection"]
        reasons = {label: needs_reroll(c, pd.Timestamp(day), spot) for label, c in held.items()}
        if not any(reasons.values()):
            selection = held
            print("Reusing active contracts (no roll trigger)", flush=True)
        else:
            print("Roll triggered:", {k: v for k, v in reasons.items() if v}, flush=True)
    if selection is None:
        selection = select_contracts(records, pd.Timestamp(day), spot)
    if not live and args.only:
        unknown = set(args.only) - set(selection)
        if unknown:
            raise SystemExit(f"Unknown contract labels: {sorted(unknown)}; available {sorted(selection)}")
        selection = {k: v for k, v in selection.items() if k in args.only}
    selection_file = args.base / f"selection_{day}.json"
    selection_file.write_text(json.dumps(selection, indent=2, default=str))
    print(f"{day} spot {spot:.1f}; contracts: " + ", ".join(f"{k}={v['option']['strike']:.0f}{v['option']['option_type']}" for k, v in selection.items()), flush=True)
    strategies = tuple(args.strategies) if not live else STRATEGIES
    sessions = register_sessions(args.base, selection, args.run_dir, catalog, mode, strategies=strategies)
    if not live:
        keys = {INDEX_KEY, *(c["option"]["instrument_key"] for c in selection.values()), next(iter(selection.values()))["future"]["instrument_key"]}
        market = ReplayMarket(fetch_replay_frames(sorted(keys), day))
    counters = run_day(args.base, day, market, selection, sessions)
    report = finish(args.base, day, sessions)
    print("RESULT (replay, synthetic quotes: software verification, not market evidence)" if not live else "RESULT")
    print(f"{'contract':<8} {'strategy':<14} {'accepted':>8} {'rejected':>8} {'fees':>9} {'net cash change':>16}  reconciles")
    for r in report["sessions"]:
        print(f"{r['contract']:<8} {r['strategy']:<14} {r['events'].get('accepted', 0):>8} {r['events'].get('rejected', 0):>8} "
              f"{(r['fees'] or 0):>9,.0f} {((r['cash'] or 0) - 5_000_000):>16,.0f}  {r['reconciles']}")
    print(json.dumps({"counters": counters, "all_reconcile": report["all_reconcile"], "any_open_exposure": report["any_open_exposure"]}))
    return 0 if report["all_reconcile"] and not report["any_open_exposure"] else 3


if __name__ == "__main__":
    sys.exit(main())
