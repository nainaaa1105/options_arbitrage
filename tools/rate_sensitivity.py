"""How much do the deviation signal and the convergence labels depend on the assumed carry rate?

For each candidate rate, rerun the production steps (batch implied volatility, leave-one-out reference IV surface, reference price,
deviation) on a sample of sessions and compare with the 6% baseline: correlation and sign agreement of the deviation, overlap of the
population with |deviation| >= 1, and agreement of the 30-minute endpoint labels. Uses the same code as the main pipeline.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\rate_sensitivity.py --output docs\\results\\rate_sensitivity
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import vector_pricing
from build_reference_iv import add_reference_iv_surface
from vector_pricing import batch_pricing_features, batch_reference_prices

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data/research/expanded_20261030/data/processed/options_features.parquet"
COLUMNS = ["timestamp", "trade_date", "expiry", "instrument_key", "strike", "option_type", "nifty_spot", "close", "time_to_expiry", "log_moneyness"]
BASE_RATE = 0.06


def deviations(frame, rate):
    f = frame.copy()
    vector_pricing.RISK_FREE_RATE = rate
    f["risk_free_rate"] = rate
    f["implied_volatility"] = batch_pricing_features(f).implied_volatility.to_numpy()
    f = add_reference_iv_surface(f)
    f["reference_price"] = batch_reference_prices(f)
    f["deviation"] = f.close - f.reference_price
    return f[["instrument_key", "timestamp", "deviation"]]


def labels(frame, deviation, horizon_minutes=30, min_deviation=1.0):
    d = frame[["instrument_key", "timestamp"]].copy()
    d["deviation"] = deviation
    d = d.sort_values(["instrument_key", "timestamp"])
    group = d.groupby("instrument_key", sort=False)
    d["future"] = group.deviation.shift(-6)
    gap = group.timestamp.shift(-6) - d.timestamp
    valid = gap.eq(pd.Timedelta(minutes=horizon_minutes)) & d.deviation.abs().ge(min_deviation) & d.future.notna()
    reduction = (d.deviation.abs() - d.future.abs()) / d.deviation.abs()
    d["label"] = np.where(valid, (reduction >= .5).astype(float), np.nan)
    return d.set_index(["instrument_key", "timestamp"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=30)
    parser.add_argument("--rates", type=float, nargs="+", default=[.03, .045, .075, .09])
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    d = pd.read_parquet(FEATURES, columns=COLUMNS)
    d["timestamp"] = pd.to_datetime(d.timestamp)
    days = sorted(d.trade_date.unique())
    chosen = [days[i] for i in np.linspace(0, len(days) - 1, args.sessions).astype(int)]
    sample = d[d.trade_date.isin(chosen)].reset_index(drop=True)
    print(f"{len(sample):,} rows over {len(chosen)} sessions", flush=True)
    base = deviations(sample, BASE_RATE).set_index(["instrument_key", "timestamp"])
    base_labels = labels(sample, base.deviation.reindex(pd.MultiIndex.from_frame(sample[["instrument_key", "timestamp"]])).to_numpy())
    rows = []
    for rate in args.rates:
        alt = deviations(sample, rate).set_index(["instrument_key", "timestamp"])
        both = base.join(alt, lsuffix="_base", rsuffix="_alt").dropna()
        shifted = both.deviation_alt - both.deviation_base
        alt_labels = labels(sample, alt.deviation.reindex(pd.MultiIndex.from_frame(sample[["instrument_key", "timestamp"]])).to_numpy())
        joined = base_labels.label.to_frame("base").join(alt_labels.label.rename("alt"))
        in_base, in_alt = joined.base.notna(), joined.alt.notna()
        common = joined[in_base & in_alt]
        rows.append({"assumed_rate": rate, "rows_compared": len(both),
                     "deviation_correlation": float(both.deviation_base.corr(both.deviation_alt)),
                     "sign_agreement": float((np.sign(both.deviation_base) == np.sign(both.deviation_alt)).mean()),
                     "median_abs_deviation_shift_points": float(shifted.abs().median()),
                     "population_overlap_jaccard": float((in_base & in_alt).sum() / (in_base | in_alt).sum()),
                     "labels_in_both": len(common), "label_agreement": float((common.base == common.alt).mean()),
                     "positive_rate_base": float(common.base.mean()), "positive_rate_alt": float(common.alt.mean())})
        print(rows[-1], flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.output / "rate_sensitivity.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps({"baseline_rate": BASE_RATE, "sessions": len(chosen), "rows": len(sample),
                                                           "rates_tested": args.rates}, indent=2))
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
