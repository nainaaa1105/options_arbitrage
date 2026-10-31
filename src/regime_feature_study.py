"""Choose causal market input profiles using historical training folds only."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

import train_models
from build_volatility_features import make_daily_spot, calculate_daily_hv
from data_remediation_study import corrected_features, fit_policy, POLICIES
from market_regime_features import market_table, augment, SPOT_FEATURES, VIX_FEATURES
from robust_model_research import purged_folds


def run(stage, daily_file, spot_file, vix_file, output):
    output.mkdir(parents=True, exist_ok=False)
    processed = stage / "data/processed"
    train_models.CONTEXT_FILE = processed / "options_signal_eligible.parquet"
    daily = pd.read_parquet(daily_file).rename(columns={"close": "nifty_spot"})
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    daily = calculate_daily_hv(make_daily_spot(daily))
    train = corrected_features(pd.read_parquet(processed / "ml_splits/train.parquet"), daily)
    market = market_table(pd.read_parquet(spot_file), pd.read_parquet(vix_file))
    train = augment(train, market)
    source = stage / "model_study/full_training_candidate.joblib"
    base = joblib.load(source)
    target_config = json.loads((stage / "target_study/frozen_training_policy.json").read_text(encoding="utf-8"))
    policy = POLICIES[target_config["selected_policy"]]
    original = [c for c in train_models._feature_columns(train, allow_point_in_time_context=True)
                if c not in SPOT_FEATURES + VIX_FEATURES]
    profiles = {"frozen_ensemble": base["feature_columns"], "lgb_spot": original + SPOT_FEATURES,
                "lgb_vix": original + VIX_FEATURES, "lgb_spot_vix": original + SPOT_FEATURES + VIX_FEATURES}
    rows = []
    for profile, names in profiles.items():
        for fold, a, b in purged_folds(train):
            if profile == "frozen_ensemble":
                model = clone(base["model"]).fit(train.iloc[a][names], train.iloc[a][train_models.TARGET].astype(int))
            else:
                model, _ = fit_policy(train.iloc[a], names, train.iloc[a][train_models.TARGET].astype(int), policy)
            probability = model.predict_proba(train.iloc[b][names])[:, 1]
            rows.append({"profile": profile, "fold": fold,
                         **train_models._classification_metrics(train.iloc[b][train_models.TARGET], probability, .5)})
        print(profile, np.mean([r["roc_auc"] for r in rows if r["profile"] == profile]), flush=True)
    pd.DataFrame(rows).to_csv(output / "cv_folds.csv", index=False)
    summary = pd.DataFrame(rows).groupby("profile").agg(mean_auc=("roc_auc", "mean"), std_auc=("roc_auc", "std"),
                                                                      mean_accuracy=("accuracy", "mean"))
    summary["selection_score"] = summary.mean_auc - .25 * summary.std_auc
    summary = summary.sort_values("selection_score", ascending=False)
    summary.to_csv(output / "cv_summary.csv")
    winner = str(summary.index[0])
    if winner == "frozen_ensemble":
        selected = {**base, "profile": winner}
    else:
        model, _ = fit_policy(train, profiles[winner], train[train_models.TARGET].astype(int), policy)
        selected = {"model": model, "feature_columns": profiles[winner], "feature_schema_version": 2,
                    "threshold": .5, "training_policy": policy, "profile": winner}
    joblib.dump(selected, output / "selected_candidate.joblib")
    (output / "frozen_design.json").write_text(json.dumps({
        "winner": winner, "selection": "training purged CV mean AUC minus .25 sample SD",
        "fresh_outcomes_opened": False, "validation_used_for_selection": False,
        "spot_input": str(spot_file.resolve()), "vix_input": str(vix_file.resolve()),
        "source_candidate_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "threshold": .5}, indent=2), encoding="utf-8")
    print(summary.to_string(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("stage", "daily-spot", "spot", "vix", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.daily_spot, args.spot, args.vix, args.output)
