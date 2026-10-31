"""Explicitly synthetic integration demonstration; never forward evidence."""

import argparse
import json
from pathlib import Path
import pandas as pd
from forward_paper import freeze, PaperSession


def demo_contracts():
    return {"option": {"instrument_key": "NSE_FO|SYNTHETIC_OPTION", "asset_type": "OPT", "lot_size": 65,
                       "tick_size": .05, "option_type": "CE", "strike": 24000., "expiry_at": "2026-11-05T15:30:00+05:30"},
            "future": {"instrument_key": "NSE_FO|SYNTHETIC_FUTURE", "asset_type": "FUT", "lot_size": 65,
                       "tick_size": .1, "expiry_at": "2026-11-26T15:30:00+05:30"}}


def demo_events():
    events = []
    contracts = demo_contracts()
    for i in range(7):
        dt = pd.Timestamp("2026-11-02 10:00", tz="Asia/Kolkata")+pd.Timedelta(minutes=5*i)
        qt, received = dt+pd.Timedelta(seconds=.5), dt+pd.Timedelta(seconds=1)
        state = {"available_at": dt.isoformat(), "bar_start": (dt-pd.Timedelta(minutes=5)).isoformat(),
                 "spot_observed": 24000.+i*3, "option_close_observed": 100.+i, "future_close_observed": 24010.+i*3,
                 "volatility": .2, "remaining_time": (pd.Timestamp(contracts["option"]["expiry_at"])-dt).total_seconds()/(365*86400),
                 "delta": .5, "recent_return": 0., "signal_eligible": False}
        quotes = {}
        for asset, mid in (("option", 100.+i), ("future", 24010.+i*3)):
            quotes[asset] = {"instrument_key": contracts[asset]["instrument_key"], "quote_time": qt.isoformat(),
                             "bid": mid-.1, "ask": mid+.1, "bid_size": 10000, "ask_size": 10000}
        events.append({"event_id": f"synthetic_{i}", "type": "open" if i == 0 else "close" if i == 6 else "hedge",
                       "decision_at": dt.isoformat(), "received_at": received.isoformat(), "state": state, "quotes": quotes})
    return events


def run(run_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    contracts = output/"synthetic_contracts.json"
    contracts.write_text(json.dumps(demo_contracts(), indent=2))
    events = demo_events()
    (output/"synthetic_events.jsonl").write_text("\n".join(json.dumps(e) for e in events)+"\n")
    for strategy in ("unhedged", "static_delta", "dynamic_delta", "band_delta", "adapted_no_ml", "adapted_ml"):
        session = output/strategy
        freeze(run_dir, contracts, session, strategy, "synthetic_demo")
        runner = PaperSession(session)
        try:
            for event in events:
                result = runner.process(event)
                if result["status"] != "accepted":
                    raise AssertionError(result)
            runner.export()
        finally:
            runner.close()
    (output/"EVIDENCE_MODE.txt").write_text("SYNTHETIC DEMONSTRATION ONLY. No actual market quotes, broker orders, or forward performance.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.run_dir, args.output)
