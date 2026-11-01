"""Descriptive split of the long-entry round trip into gross price move and costs, by size of the initial mispricing.

For every executable undervalued candidate (the same population as pnl_experiment): buy at the next bar's open, sell 30 minutes later.
Gross = quantity * (raw exit open - raw entry open). Net = the Part 3 cost model result. Cost = gross - net.
No model, no selection, no tuning: this only shows how much of the loss is cost and whether larger gaps change the picture.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\cost_decomposition.py --output docs\\results\\cost_decomposition
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from accuracy_v2.pnl_experiment import PART3_PLAN, attach_pnl, day_bootstrap
from accuracy_v2.experiment import load_tables
from deep_hedging.config import HedgeConfig

STAGE = ROOT / "data/research/expanded_20261030"
DAILY = ROOT / "data/raw/upstox/nifty_daily_to_20261029.parquet"
BINS = [1, 2, 4, 8, 16, np.inf]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    config = HedgeConfig(**json.loads(PART3_PLAN.read_text())["config"])
    history, fresh, _, _ = load_tables(STAGE, DAILY, STAGE / "regime_study")
    opens = pd.read_parquet(STAGE / "data/processed/options_features.parquet", columns=["instrument_key", "timestamp", "open", "lot_size"])
    opens["timestamp"] = pd.to_datetime(opens.timestamp)
    frames = []
    for name, frame in (("history", history), ("later_period", fresh)):
        c = attach_pnl(frame[frame.reference_price_deviation < 0], opens, config)
        c = c[c.net_pnl.notna()].copy()
        c["period"], c["day"] = name, c.timestamp.dt.date
        c["quantity"] = config.option_lots * c.lot_size.astype(int)
        c["gross_pnl"] = c.quantity * (c.exit_open - c.entry_open)
        c["cost"] = c.gross_pnl - c.net_pnl
        c["abs_deviation"] = c.reference_price_deviation.abs()
        c["bucket"] = pd.cut(c.abs_deviation, BINS, right=False).astype(str)
        frames.append(c)
    d = pd.concat(frames)
    rows = []
    for (period, bucket), g in list(d.groupby(["period", "bucket"], sort=False)) + [((p, "all"), x) for p, x in d.groupby("period")]:
        mean, lo, hi = day_bootstrap(g.gross_pnl, g.day)
        rows.append({"period": period, "abs_deviation_points": bucket, "trades": len(g), "mean_gross_inr": mean, "gross_lower95": lo, "gross_upper95": hi,
                     "mean_cost_inr": g.cost.mean(), "mean_net_inr": g.net_pnl.mean(), "gross_win_rate": float((g.gross_pnl > 0).mean()),
                     "net_win_rate": float((g.net_pnl > 0).mean())})
    table = pd.DataFrame(rows)
    table.to_csv(args.output / "cost_decomposition.csv", index=False)
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
