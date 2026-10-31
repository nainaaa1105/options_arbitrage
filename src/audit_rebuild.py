"""Measure raw validity and the effects of synchronized spot pricing."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def describe_market(d):
    prices = d[["open", "high", "low", "close"]].to_numpy(float)
    bad = (~np.isfinite(prices).all(axis=1) | (prices <= 0).any(axis=1)
           | (prices[:, 1] < prices.max(axis=1)) | (prices[:, 2] > prices.min(axis=1)))
    ts = pd.to_datetime(d.timestamp)
    offset = ts.dt.hour * 60 + ts.dt.minute - 555
    keys = ["instrument_key", "timestamp"]
    sorted_d = d.sort_values(keys)
    group = sorted_d.groupby(["instrument_key", "trade_date"])
    prior = group.close.shift(1)
    same_close = sorted_d.close.eq(prior)
    count = d.groupby(["instrument_key", "trade_date"]).size()
    return {"rows": len(d), "contracts": int(d.instrument_key.nunique()),
            "sessions": int(ts.dt.date.nunique()), "expiries": int(d.expiry.nunique()),
            "duplicates": int(d.duplicated(keys).sum()), "invalid_ohlc": int(bad.sum()),
            "nonpositive_or_missing_spot": int((~np.isfinite(d.nifty_spot) | d.nifty_spot.le(0)).sum()),
            "negative_volume_or_oi": int((d.volume.lt(0) | d.open_interest.lt(0)).sum()),
            "zero_volume": int(d.volume.eq(0).sum()), "zero_oi": int(d.open_interest.eq(0).sum()),
            "nonpositive_lot_size": int(d.lot_size.le(0).sum()),
            "off_five_minute_grid": int(offset.mod(5).ne(0).sum()),
            "outside_regular_session": int((offset.lt(0) | offset.gt(370)).sum()),
            "contract_sessions_with_fewer_than_75_rows": int(count.lt(75).sum()),
            "contract_sessions": len(count),
            "unchanged_adjacent_close_fraction": float(same_close.mean()),
            "within_timestamp_spot_disagreement_groups": int(d.groupby("timestamp").nifty_spot.nunique().gt(1).sum())}


def run(stage, output):
    output.mkdir(parents=True, exist_ok=False)
    old_master = pd.read_parquet(ROOT / "data/processed/master_options.parquet")
    new_master = pd.read_parquet(stage / "data/processed/master_options.parquet")
    result = {"old_market": describe_market(old_master), "new_market": describe_market(new_master)}
    merged = old_master.merge(new_master, on=["instrument_key", "timestamp"], suffixes=("_old", "_new"), validate="one_to_one")
    delta = merged.nifty_spot_new - merged.nifty_spot_old
    result["spot_repair"] = {"matched_rows": len(merged), "changed_rows": int(delta.abs().gt(1e-8).sum()),
                             "median_absolute_change": float(delta.abs().median()),
                             "p95_absolute_change": float(delta.abs().quantile(.95)),
                             "maximum_absolute_change": float(delta.abs().max())}
    provenance = new_master.feature_available_at.eq(new_master.spot_available_at)
    result["aligned_close_availability_rows"] = int(provenance.sum())
    for version, folder in (("old", ROOT / "data/processed"), ("new", stage / "data/processed")):
        f = folder / "options_features.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["iv_status"])
            result[version + "_iv_status"] = {str(k): int(v) for k, v in d.iv_status.value_counts().items()}
        f = folder / "options_convergence_labels.parquet"
        if f.exists():
            d = pd.read_parquet(f, columns=["instrument_key", "timestamp", "converged_30m", "label_valid_30m", "signal_eligible", "absolute_price_deviation"])
            d.groupby(d.timestamp.dt.strftime("%Y-%m")).agg(rows=("timestamp", "size"),
                eligible=("signal_eligible", "sum"), mean_absolute_residual=("absolute_price_deviation", "mean")).to_csv(output / f"{version}_monthly.csv")
            result[version + "_eligible"] = int(d.signal_eligible.sum())
            result[version + "_valid_targets"] = int(d.label_valid_30m.sum())
            if version == "old":
                old_labels = d
            else:
                compare = old_labels.merge(d, on=["instrument_key", "timestamp"], suffixes=("_old", "_new"), validate="one_to_one")
                common = (compare.label_valid_30m_old.fillna(False).astype(bool)
                          & compare.label_valid_30m_new.fillna(False).astype(bool))
                result["label_changes"] = {"common_observed_targets": int(common.sum()),
                    "changed_targets": int(compare.loc[common, "converged_30m_old"].ne(compare.loc[common, "converged_30m_new"]).sum())}
    (output / "rebuild_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.output)
