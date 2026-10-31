"""Refit frozen designs on completed history, then evaluate later dates once."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

import train_models
from build_volatility_features import calculate_daily_hv, make_daily_spot
from data_remediation_study import corrected_features, fit_policy, POLICIES
from research_targets import build_path_targets
from robust_model_research import grouped_intervals, selective_metrics
from market_regime_features import market_table, augment


def run(stage, daily_file, output, regime_study=None):
    output.mkdir(parents=True, exist_ok=False)
    processed = stage / "data/processed"
    train_models.CONTEXT_FILE = processed / "options_signal_eligible.parquet"
    daily = pd.read_parquet(daily_file).rename(columns={"close": "nifty_spot"})
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    daily = calculate_daily_hv(make_daily_spot(daily))
    history_raw = pd.read_parquet(processed / "ml_dataset.parquet")
    history = corrected_features(history_raw, daily)
    market = None
    regime_config = None
    if regime_study is not None:
        regime_config = json.loads((regime_study / "frozen_design.json").read_text(encoding="utf-8"))
        market = market_table(pd.read_parquet(regime_config["spot_input"]), pd.read_parquet(regime_config["vix_input"]))
        history = augment(history, market)
    history["timestamp"] = pd.to_datetime(history.timestamp)
    assert history.timestamp.dt.date.max() <= pd.Timestamp("2026-09-15").date()
    candidate_file = (regime_study / "selected_candidate.joblib") if regime_study is not None else stage / "model_study/full_training_candidate.joblib"
    candidate = joblib.load(candidate_file)
    columns = candidate["feature_columns"]
    if "training_policy" in candidate:
        model, _ = fit_policy(history, columns, history[train_models.TARGET].astype(int), candidate["training_policy"])
    else:
        model = clone(candidate["model"])
        model.fit(history[columns], history[train_models.TARGET].astype(int))
    bundle = {**candidate, "model": model, "training_end": str(history.timestamp.max()), "threshold": .5}
    joblib.dump(bundle, output / "endpoint_frozen_model.joblib")
    target_config = json.loads((stage / "target_study/frozen_training_policy.json").read_text(encoding="utf-8"))
    policy = POLICIES[target_config["selected_policy"]]
    # Restrict construction to completed history before fitting path targets.
    context = pd.read_parquet(processed / "options_parity_features.parquet", columns=[
        "instrument_key", "timestamp", "expiry", "reference_price_deviation", "signal_eligible"])
    context["timestamp"] = pd.to_datetime(context.timestamp)
    historical_mask = context.timestamp.dt.date <= pd.Timestamp("2026-09-15").date()
    paths = build_path_targets(context.loc[historical_mask])
    path_history = history.merge(paths.drop(columns="expiry"), on=["instrument_key", "timestamp"], validate="one_to_one")
    path_history = path_history.loc[path_history.complete_path].copy()
    path_columns = joblib.load(stage / "target_study/target_candidate_first_passage.joblib")["feature_columns"]
    if regime_config is not None and regime_config["winner"] != "frozen_ensemble":
        path_columns = columns
    path_models = {}
    for target in ("first_passage", "sustained_two_bars"):
        fitted, _ = fit_policy(path_history, path_columns, path_history[target].astype(int), policy)
        path_models[target] = fitted
        joblib.dump({"model": fitted, "feature_columns": path_columns, "target": target,
                     "threshold": .5, "feature_schema_version": 2}, output / f"{target}_frozen_model.joblib")
    model_files = list(output.glob("*_frozen_model.joblib"))
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
              "history_rows": len(history), "history_end": str(history.timestamp.max()),
              "fresh_start_date": "2026-09-16", "primary_target": "endpoint",
              "alternative_targets": ["first_passage", "sustained_two_bars"],
              "threshold": .5, "predeclared_selective_margins": [0., .1, .2, .3, .4],
              "models_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in model_files},
              "selection": "historical training CV only; no fresh-period threshold or model selection",
              "base_model_config": json.loads((stage / "model_study/frozen_config.json").read_text(encoding="utf-8")),
              "target_policy": target_config}
    frozen["market_feature_design"] = regime_config
    (output / "frozen_before_fresh_evaluation.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("Models and thresholds frozen; beginning fresh-period evaluation", flush=True)
    fresh = corrected_features(pd.read_parquet(processed / "fresh_ml_dataset.parquet"), daily)
    if market is not None:
        fresh = augment(fresh, market)
    assert fresh.timestamp.min() > history.timestamp.max()
    assert (pd.to_datetime(history.future_timestamp_30m) + pd.Timedelta(minutes=5)).max() < fresh.timestamp.min()
    fresh_paths = build_path_targets(context.loc[~historical_mask])
    fresh_path_features = fresh.merge(fresh_paths.drop(columns="expiry"), on=["instrument_key", "timestamp"], validate="one_to_one")
    fresh_path_features = fresh_path_features.loc[fresh_path_features.complete_path].copy()
    rows, selective, intervals = [], [], {}
    fitting_metrics = {}
    experiments = [("endpoint", model, fresh, columns, train_models.TARGET, history[train_models.TARGET])]
    experiments += [(target, fitted, fresh_path_features, path_columns, target, path_history[target])
                    for target, fitted in path_models.items()]
    for name, fitted, frame, names, target, fit_y in experiments:
        probability = fitted.predict_proba(frame[names])[:, 1]
        y = frame[target].astype(int).to_numpy()
        baseline = int(np.mean(fit_y) >= .5)
        metrics = train_models._classification_metrics(y, probability, .5)
        fitting_frame = history if name == "endpoint" else path_history
        fitting_target = train_models.TARGET if name == "endpoint" else target
        fitting_metrics[name] = train_models._classification_metrics(
            fitting_frame[fitting_target].astype(int), fitted.predict_proba(fitting_frame[names])[:, 1], .5)
        prior_metrics = train_models._classification_metrics(y, np.full(len(y), float(np.mean(fit_y))), .5)
        rows.append({"target": name, "rows": len(frame), "sessions": int(frame.timestamp.dt.date.nunique()),
                     "positive_rate": float(y.mean()), "training_majority_baseline_accuracy": float((y == baseline).mean()),
                     **metrics})
        rows[-1].update({"prior_baseline_brier": prior_metrics["brier_score"],
                         "prior_baseline_log_loss": prior_metrics["log_loss"]})
        intervals[name] = grouped_intervals(y, probability, frame.timestamp.dt.date, .5)
        for margin in (0., .1, .2, .3, .4):
            selective.append({"target": name, **selective_metrics(y, probability, frame.timestamp.dt.date, margin)})
        prediction = frame[["timestamp", "instrument_key"]].copy()
        prediction["actual"] = y
        prediction["probability"] = probability
        prediction.to_parquet(output / f"{name}_fresh_predictions.parquet", index=False)
    pd.DataFrame(rows).to_csv(output / "fresh_metrics.csv", index=False)
    pd.DataFrame(selective).to_csv(output / "fresh_selective_metrics.csv", index=False)
    (output / "session_confidence_intervals.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    (output / "fitting_metrics_not_holdout.json").write_text(json.dumps(fitting_metrics, indent=2), encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--daily-spot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--regime-study", type=Path)
    args = parser.parse_args()
    run(args.stage, args.daily_spot, args.output, args.regime_study)
