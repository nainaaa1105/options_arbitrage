"""Is the constant 6% carry used in pricing consistent with the market? Compare it with parity-implied and futures-implied carry.

Option side: at-the-money call/put pairs give a forward F = K + (C - P) * exp(r*T) under the assumed rate r, so the implied
annualised carry is ln(F/S)/T. Futures side: ln(F_fut/S)/T_fut for the front-month contract. If the pricing rate were right, both
would be close to r. The size of the resulting mispricing in index points is S * (exp((c - r) T) - 1).

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\carry_diagnostic.py --output docs\\results\\carry_diagnostic
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PARITY = ROOT / "data/research/expanded_20261030/data/processed/options_parity_features.parquet"
FUTURES = ROOT / "data/raw/upstox/hedging_futures_20261030/front_month_futures.parquet"
RATE = 0.06


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(PARITY, columns=["timestamp", "expiry", "strike", "nifty_spot", "time_to_expiry", "call_price", "put_price", "volume"])
    d = d.dropna(subset=["call_price", "put_price"])
    d = d[(np.abs(np.log(d.strike / d.nifty_spot)) < .006) & (d.time_to_expiry > 1 / 365)].copy()
    d = d.sort_values("volume").drop_duplicates(["timestamp", "expiry"], keep="last")       # most active near-ATM strike per snapshot
    T = d.time_to_expiry
    d["forward"] = d.strike + (d.call_price - d.put_price) * np.exp(RATE * T)
    d["implied_carry"] = np.log(d.forward / d.nifty_spot) / T
    d["assumed_forward"] = d.nifty_spot * np.exp(RATE * T)
    d["forward_error_points"] = d.forward - d.assumed_forward
    d["date"] = pd.to_datetime(d.timestamp).dt.date
    d["days_to_expiry"] = np.ceil(T * 365)
    clean = d[np.isfinite(d.implied_carry) & (np.abs(d.implied_carry) < 1.0)]
    daily = clean.groupby("date").agg(implied_carry=("implied_carry", "median"), error_points=("forward_error_points", "median"), snapshots=("forward", "size"))
    by_dte = clean.groupby(clean.days_to_expiry.clip(upper=7)).agg(implied_carry=("implied_carry", "median"), error_points=("forward_error_points", "median"),
                                                                   abs_error_points=("forward_error_points", lambda s: float(s.abs().median())), snapshots=("forward", "size"))
    out = {"assumed_annual_rate": RATE, "snapshots_used": int(len(clean)), "sessions": int(daily.shape[0]),
           "parity_implied_carry": {"median": float(clean.implied_carry.median()), "p10": float(clean.implied_carry.quantile(.1)), "p90": float(clean.implied_carry.quantile(.9))},
           "forward_error_points": {"median": float(clean.forward_error_points.median()), "median_abs": float(clean.forward_error_points.abs().median()),
                                    "p90_abs": float(clean.forward_error_points.abs().quantile(.9))}}
    if FUTURES.exists():
        f = pd.read_parquet(FUTURES, columns=["timestamp", "close", "expiry"])
        f["timestamp"] = pd.to_datetime(f.timestamp)
        s = pd.read_parquet(PARITY, columns=["timestamp", "nifty_spot"]).drop_duplicates("timestamp")
        s["timestamp"] = pd.to_datetime(s.timestamp)
        m = f.merge(s, on="timestamp")
        m["T"] = (pd.to_datetime(m.expiry).dt.tz_localize(None).dt.normalize() + pd.Timedelta(hours=15, minutes=30) - m.timestamp.dt.tz_localize(None)).dt.total_seconds() / (365 * 86400)
        m = m[m["T"] > 5 / 365]
        m["futures_carry"] = np.log(m.close / m.nifty_spot) / m["T"]
        out["futures_implied_carry"] = {"median": float(m.futures_carry.median()), "p10": float(m.futures_carry.quantile(.1)), "p90": float(m.futures_carry.quantile(.9)), "snapshots": int(len(m))}
    daily.to_csv(args.output / "daily_implied_carry.csv")
    by_dte.to_csv(args.output / "by_days_to_expiry.csv")
    (args.output / "summary.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(by_dte.round(3).to_string())


if __name__ == "__main__":
    main()
