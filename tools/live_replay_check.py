"""Replay recorded sessions bar by bar through ChainFeatureService with the frozen accuracy bundle and compare with stored predictions.

This is the end-to-end check for live ML entries: the service sees only bars up to the current one, exactly as it would live, and its
probabilities must equal the ones the frozen model produced in batch. Also reports the per-bar compute time.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\live_replay_check.py --sessions 1 --every 6
"""
import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import chain_features

STAGE = ROOT / "data/research/expanded_20261030"
DAILY = ROOT / "data/raw/upstox/nifty_daily_to_20261029.parquet"
BUNDLE = ROOT / "data/research/accuracy_v2_20261031/final_model.joblib"
PREDICTIONS = ROOT / "data/research/accuracy_v2_20261031/fresh_predictions.parquet"
DESIGN = STAGE / "regime_study/frozen_design.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--every", type=int, default=6, help="Compare every n-th bar (all bars are still pushed)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    bundle = joblib.load(BUNDLE)
    stored = pd.read_parquet(PREDICTIONS)
    stored["timestamp"] = pd.to_datetime(stored.timestamp)
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    spot, vix = pd.read_parquet(design["spot_input"]), pd.read_parquet(design["vix_input"])
    daily = pd.read_parquet(DAILY)
    master = pd.read_parquet(STAGE / "data/processed/master_options.parquet")
    master["timestamp"] = pd.to_datetime(master.timestamp)
    sessions = sorted(stored.timestamp.dt.date.unique())
    picks = [sessions[i] for i in np.linspace(5, len(sessions) - 1, args.sessions).astype(int)]
    report = []
    for day in picks:
        service = chain_features.ChainFeatureService(daily, spot, vix, bundle)
        rows = master[master.trade_date.astype(str) == str(day)]
        diffs, seconds, compared = [], [], 0
        for number, (stamp, bar) in enumerate(rows.groupby("timestamp", sort=True)):
            start = time.perf_counter()
            live = service.push(bar)
            seconds.append(time.perf_counter() - start)
            if number % args.every or live.empty:
                continue
            expected = stored[stored.timestamp == stamp][["instrument_key", "probability"]]
            both = live[["instrument_key", "probability"]].merge(expected, on="instrument_key", how="outer", suffixes=("", "_stored"), indicator=True)
            missing = int((both._merge != "both").sum())
            matched = both[both._merge == "both"]
            diffs.append((missing, float((matched.probability - matched.probability_stored).abs().max()) if len(matched) else 0.0, len(matched)))
            compared += len(matched)
        report.append({"session": str(day), "bars": len(seconds), "compared_rows": compared,
                       "unmatched_rows": int(sum(d[0] for d in diffs)), "max_abs_probability_diff": max((d[1] for d in diffs), default=None),
                       "seconds_per_bar_median": float(np.median(seconds)), "seconds_per_bar_max": float(np.max(seconds))})
        print(report[-1], flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
