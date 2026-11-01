"""End-to-end demonstration on SYNTHETIC option data, so the pipeline can be run from a clean clone without any vendor data.

Generates weekly NIFTY-like option candles (GBM spot, a volatility smile, Black-Scholes prices plus tick-level bid/ask-bounce noise) in the
exact schema of the real master dataset, then runs the real pipeline stages (features, volatility, reference surface, deviations,
eligibility, parity, labels, ML dataset) in an isolated folder and fits a small model with a chronological split.

The data is invented. A good score here only shows that the code runs and that the labels are learnable when the noise structure
is made up; it says nothing about real markets.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\synthetic_demo.py --output data\\synthetic_demo
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
IST = "Asia/Kolkata"
LOT = 75
RATE = 0.06


def bs_call_put(spot, strike, years, vol):
    d1 = (np.log(spot / strike) + (RATE + .5 * vol ** 2) * years) / (vol * np.sqrt(years))
    d2 = d1 - vol * np.sqrt(years)
    call = spot * norm.cdf(d1) - strike * np.exp(-RATE * years) * norm.cdf(d2)
    put = strike * np.exp(-RATE * years) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return call, put


def generate(output, expiries=6, sessions_per_expiry=5, seed=7, strikes_each_side=10):
    """Write master_options.parquet and a daily spot file; return their paths."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    first_expiry = pd.Timestamp("2025-03-06")                       # a Thursday; weekly cadence
    spot, rows, daily = 22000., [], []
    # Daily history before the first session so historical volatility has a warm-up window.
    for i in range(40):
        spot *= np.exp(rng.normal(0, .008))
        daily.append({"timestamp": pd.Timestamp("2025-01-01", tz=IST) + pd.Timedelta(days=i), "close": spot})
    for e in range(expiries):
        expiry = first_expiry + pd.Timedelta(days=7 * e)
        days = pd.bdate_range(end=expiry, periods=sessions_per_expiry)
        base = float(np.round(spot / 50) * 50)
        strikes = base + 50 * np.arange(-strikes_each_side, strikes_each_side + 1)
        sigma0 = rng.uniform(.12, .22)
        for day in days:
            spot *= np.exp(rng.normal(0, .004))                         # overnight gap
            stamps = pd.date_range(f"{day.date()} 09:15", periods=75, freq="5min", tz=IST)
            closes = spot * np.exp(np.cumsum(rng.normal(0, .0009, 75)))
            for kind in ("CE", "PE"):
                for strike in strikes:
                    previous = None
                    for t, s in zip(stamps, closes):
                        available = t + pd.Timedelta(minutes=5)
                        years = max((expiry + pd.Timedelta(hours=15, minutes=30) - available.tz_localize(None)).total_seconds(), 60.) / (365 * 86400)
                        m = np.log(strike / s)
                        vol = sigma0 * (1 + .5 * m + 6. * m ** 2)
                        call, put = bs_call_put(s, strike, years, vol)
                        fair = call if kind == "CE" else put
                        intrinsic = max(s - strike, 0.) if kind == "CE" else max(strike - s, 0.)
                        price = max(fair + .05 * rng.integers(-2, 3), intrinsic + .05, .05)      # bid/ask-bounce style noise
                        price = round(round(price / .05) * .05, 2)
                        open_ = price if previous is None else previous
                        rows.append({"timestamp": t, "trade_date": str(day.date()), "expiry": expiry, "instrument_key": f"SYNTH|{expiry.date()}|{int(strike)}|{kind}",
                                     "trading_symbol": f"SYN {int(strike)} {kind}", "strike": float(strike), "option_type": kind, "lot_size": LOT,
                                     "open": open_, "high": max(open_, price) + .05, "low": max(min(open_, price) - .05, .05), "close": price,
                                     "volume": int(rng.poisson(40 + 400 * np.exp(-abs(m) * 60))), "open_interest": int(5000 + 200000 * np.exp(-abs(m) * 40)),
                                     "nifty_spot": float(s), "feature_available_at": available, "spot_timestamp": t + pd.Timedelta(minutes=4),
                                     "spot_available_at": available, "source_file": f"synthetic_{expiry.date()}.parquet"})
                        previous = price
            spot = float(closes[-1])
            daily.append({"timestamp": pd.Timestamp(day.date(), tz=IST), "close": spot})
    master = pd.DataFrame(rows).sort_values(["timestamp", "instrument_key"]).reset_index(drop=True)
    master_path, daily_path = output / "master_options.parquet", output / "daily_spot.parquet"
    master.to_parquet(master_path, index=False)
    pd.DataFrame(daily).to_parquet(daily_path, index=False)
    return master_path, daily_path


def fit_and_report(run_folder):
    """Chronological split on the generated ML dataset; report AUC against the base rate."""
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    data = pd.read_parquet(Path(run_folder) / "data/processed/ml_dataset.parquet")
    data["timestamp"] = pd.to_datetime(data.timestamp)
    target = "target_converged_30m"
    features = [c for c in ("reference_price_deviation", "absolute_price_deviation", "iv_surface_residual", "implied_volatility", "reference_iv",
                            "time_to_expiry", "log_moneyness", "delta", "gamma", "vega", "theta", "volume", "open_interest", "parity_residual")
                if c in data]
    days = sorted(data.timestamp.dt.date.unique())
    cut = days[int(len(days) * .7)]
    train, test = data[data.timestamp.dt.date < cut], data[data.timestamp.dt.date > cut]     # one-session gap, like the real study
    model = lgb.LGBMClassifier(n_estimators=150, learning_rate=.05, num_leaves=15, min_child_samples=40, random_state=1, verbosity=-1, n_jobs=2)
    model.fit(train[features], train[target].astype(int))
    p = model.predict_proba(test[features])[:, 1]
    y = test[target].astype(int)
    return {"ml_rows": len(data), "train_rows": len(train), "test_rows": len(test), "test_positive_rate": float(y.mean()),
            "majority_accuracy": float(max(y.mean(), 1 - y.mean())), "test_accuracy": float(((p >= .5) == y).mean()),
            "test_auc": float(roc_auc_score(y, p)), "features_used": features}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New folder for the synthetic data and the isolated pipeline run")
    parser.add_argument("--expiries", type=int, default=6)
    parser.add_argument("--sessions-per-expiry", type=int, default=5)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise SystemExit(f"{args.output} already exists; choose a new folder")
    master, daily = generate(args.output / "input", args.expiries, args.sessions_per_expiry)
    print(f"Generated {len(pd.read_parquet(master)):,} synthetic option candles", flush=True)
    from run_aligned_research import run
    run(args.output / "pipeline", master, daily, trials=1, pipeline_only=True)
    report = fit_and_report(args.output / "pipeline")
    report["note"] = "SYNTHETIC data: demonstrates that the pipeline runs end to end; says nothing about real markets"
    (args.output / "demo_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
