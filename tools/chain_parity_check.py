"""Batch/live parity: rerun the chain-feature stages on recorded sessions and compare with what the batch pipeline stored.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\chain_parity_check.py --sessions 3
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import chain_features

STAGE = ROOT / "data/research/expanded_20261030"
DAILY = ROOT / "data/raw/upstox/nifty_daily_to_20261029.parquet"
COMPARE = ["implied_volatility", "reference_iv", "reference_theoretical_price", "reference_price_deviation", "iv_surface_residual", "parity_residual"]
FLAGS = ["signal_eligible", "parity_usable"]
KEY = ["instrument_key", "timestamp"]


def compare(session_date, master_all, stored_columns):
    master = master_all[master_all.trade_date == session_date].reset_index(drop=True)
    daily = pd.read_parquet(DAILY)
    live = chain_features.run_stages(master, daily)
    stored = pd.read_parquet(STAGE / "data/processed/options_parity_features.parquet", columns=KEY + stored_columns, filters=[("trade_date", "==", pd.Timestamp(session_date).date())])
    for frame in (live, stored):
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
    both = live.merge(stored, on=KEY, suffixes=("", "_stored"), validate="one_to_one")
    result = {"session": str(session_date), "rows": len(both), "rows_live": len(live), "rows_stored": len(stored)}
    for column in COMPARE:
        a, b = both[column].to_numpy(float), both[f"{column}_stored"].to_numpy(float)
        same_nan = np.isnan(a) == np.isnan(b)
        finite = ~np.isnan(a) & ~np.isnan(b)
        result[column] = {"nan_pattern_equal": float(same_nan.mean()), "max_abs_diff": float(np.abs(a[finite] - b[finite]).max()) if finite.any() else None}
    for column in FLAGS:
        result[column] = float((both[column].astype(bool) == both[f"{column}_stored"].astype(bool)).mean())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    master = pd.read_parquet(STAGE / "data/processed/master_options.parquet")
    master["timestamp"] = pd.to_datetime(master.timestamp)
    days = sorted(master.trade_date.unique())
    picks = [days[i] for i in np.linspace(len(days) // 4, len(days) - 1, args.sessions).astype(int)]
    results = [compare(day, master, COMPARE + FLAGS) for day in picks]
    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
