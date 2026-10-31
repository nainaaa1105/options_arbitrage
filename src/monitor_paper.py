"""Refresh session exports and report stale feeds/unresolved forward exposure."""

import argparse
import json
from pathlib import Path
import pandas as pd
from forward_paper import PaperSession, stamp


def monitor(directory):
    runner = PaperSession(directory)
    try:
        report = runner.export()
        state = report["state"]
        now = pd.Timestamp.now(tz="UTC")
        report["checked_at_utc"] = now.isoformat()
        report["seconds_since_last_accepted_event"] = (now-stamp(state["last_received_at"])).total_seconds() if state["last_received_at"] else None
        alerts = []
        if state["halted"]:
            alerts.append("risk_stop_active")
        if report["mode"] == "forward":
            if state["q"] and state["entry_at"] and now > stamp(state["entry_at"])+pd.Timedelta(minutes=runner.c.horizon_minutes):
                alerts.append("overdue_open_position_requires_valid_close_quotes")
            if state["q"] and report["seconds_since_last_accepted_event"] > 310:
                alerts.append("stale_feed_while_position_open")
        report["alerts"] = alerts
        report["healthy"] = not alerts
        (Path(directory)/"monitor.json").write_text(json.dumps(report, indent=2))
        return report
    finally:
        runner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    args = parser.parse_args()
    result = monitor(args.session)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["healthy"] else 2)
