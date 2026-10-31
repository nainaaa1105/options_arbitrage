"""Audited model development using training data and historical validation only.

Hyperparameters use purged expanding folds in an early training block. A second
block fits sigmoid calibration; a third chooses calibration and thresholds. The
existing validation block is scored only after those choices are frozen. A raw
model refitted on the complete training split is also a predeclared comparison.
The previously exposed test split is never opened by this experiment.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from train_models import (
    TRAIN_FILE, VALID_FILE, CONTEXT_FILE, TARGET, _classification_metrics, _feature_columns,
    _make_estimator, _validation_thresholds, build_features,
)


def purged_folds(frame, n_splits=3, gap_sessions=1):
    dates = frame.timestamp.dt.normalize()
    unique = pd.DatetimeIndex(sorted(dates.unique()))
    for fold, (a, b) in enumerate(
        TimeSeriesSplit(n_splits=n_splits, gap=gap_sessions).split(unique), 1
    ):
        train_mask = dates.isin(unique[a])
        valid_mask = dates.isin(unique[b])
        boundary = frame.loc[valid_mask, "timestamp"].min()
        available = pd.to_datetime(frame.future_timestamp_30m) + pd.Timedelta(minutes=5)
        train_mask &= available < boundary
        yield fold, np.flatnonzero(train_mask), np.flatnonzero(valid_mask)


def logit(probability):
    p = np.clip(probability, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


def selective_metrics(y, p, dates, margin):
    y, p, dates = np.asarray(y), np.asarray(p), np.asarray(dates)
    low, high = p <= 0.5 - margin, p >= 0.5 + margin
    accepted = low | high
    count = int(accepted.sum())
    return {
        "margin": float(margin), "low_cutoff": float(0.5 - margin),
        "high_cutoff": float(0.5 + margin), "accepted_rows": count,
        "coverage": float(accepted.mean()),
        "selective_accuracy": float((high[accepted] == y[accepted]).mean()) if count else None,
        "positive_rows": int(high.sum()), "negative_rows": int(low.sum()),
        "positive_precision": float(y[high].mean()) if high.any() else None,
        "negative_precision": float(1 - y[low].mean()) if low.any() else None,
        "positive_sessions": int(len(np.unique(dates[high]))),
        "negative_sessions": int(len(np.unique(dates[low]))),
    }


def qualifies_selective(row, target):
    # Require evidence for both decisions; a near-perfect negative tail alone
    # does not establish a 93-94% predictor of positive convergence signals.
    return (
        row["accepted_rows"] >= 200 and row["positive_rows"] >= 30
        and row["negative_rows"] >= 30 and row["positive_sessions"] >= 5
        and row["negative_sessions"] >= 5
        and row["selective_accuracy"] >= target
        and row["positive_precision"] >= target
        and row["negative_precision"] >= target
    )


def estimator(family, params, threads):
    result = _make_estimator(family, params=params)
    if family != "logistic_regression":
        result.set_params(model__n_jobs=threads)
    return result


def suggest(trial, family):
    common = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 350, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.12, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 100.0, log=True),
    }
    if family == "xgboost":
        common.update({
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "min_child_weight": trial.suggest_float("min_child_weight", 5.0, 60.0),
        })
    else:
        common.update({
            "max_depth": trial.suggest_int("max_depth", 3, 6),
            "num_leaves": trial.suggest_int("num_leaves", 7, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 120),
            "subsample_freq": 1,
        })
    return common


def grouped_intervals(y, probability, dates, threshold, repeats=300):
    rng = np.random.default_rng(42)
    dates = np.asarray(dates)
    sessions = np.unique(dates)
    members = [np.flatnonzero(dates == date) for date in sessions]
    values = []
    for _ in range(repeats):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        if len(np.unique(y[idx])) < 2:
            continue
        values.append({
            "accuracy": float(np.mean((probability[idx] >= threshold) == y[idx])),
            "roc_auc": float(roc_auc_score(y[idx], probability[idx])),
        })
    return {
        metric: {"lower_95": float(np.quantile([r[metric] for r in values], 0.025)),
                 "upper_95": float(np.quantile([r[metric] for r in values], 0.975))}
        for metric in ("accuracy", "roc_auc")
    }


def selective_grouped_intervals(y, probability, dates, margin, repeats=300):
    rng = np.random.default_rng(42)
    dates = np.asarray(dates)
    members = [np.flatnonzero(dates == date) for date in np.unique(dates)]
    statistics = {name: [] for name in ("coverage", "selective_accuracy", "positive_precision", "negative_precision")}
    for _ in range(repeats):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        values = selective_metrics(y[idx], probability[idx], dates[idx], margin)
        for name in statistics:
            if values[name] is not None:
                statistics[name].append(values[name])
    return {
        name: {"lower_95": float(np.quantile(values, .025)),
               "upper_95": float(np.quantile(values, .975))} if values else None
        for name, values in statistics.items()
    }


def run(output, trials, threads):
    output.mkdir(parents=True, exist_ok=False)
    raw = pd.read_parquet(TRAIN_FILE)
    validation_raw = pd.read_parquet(VALID_FILE)
    frames = build_features({"train": raw, "validation": validation_raw}, feature_schema_version=2)
    train, validation = frames["train"], frames["validation"]
    legacy = build_features({"train": raw}, feature_schema_version=1)["train"]
    features = _feature_columns(train, allow_point_in_time_context=True)
    dates = train.timestamp.dt.normalize()
    sessions = pd.DatetimeIndex(sorted(dates.unique()))
    fit_end, calibration_end = int(len(sessions) * .70), int(len(sessions) * .85)
    fit = train.loc[dates.isin(sessions[:fit_end])].copy().reset_index(drop=True)
    calibration = train.loc[dates.isin(sessions[fit_end:calibration_end])].copy()
    decision = train.loc[dates.isin(sessions[calibration_end:])].copy()
    fit = fit.loc[pd.to_datetime(fit.future_timestamp_30m) + pd.Timedelta(minutes=5) < calibration.timestamp.min()].reset_index(drop=True)
    calibration = calibration.loc[
        pd.to_datetime(calibration.future_timestamp_30m) + pd.Timedelta(minutes=5) < decision.timestamp.min()
    ].copy()
    decision = decision.loc[
        pd.to_datetime(decision.future_timestamp_30m) + pd.Timedelta(minutes=5) < validation.timestamp.min()
    ].copy()
    audit = {"feature_schema_version": 2, "features": features,
             "legacy_session_time_mismatches": int((legacy.intraday_minute != train.intraday_minute).sum()),
             "legacy_excluded_expiry_features": [c for c in features if "expiry" in c],
             "target_definition": "50% absolute deviation reduction at 30-35 minutes; endpoint, not first passage",
             "test_split_opened": False, "blocks": {},
             "spot_alignment_provenance_present": {
                 "feature_available_at", "spot_available_at", "spot_timestamp"
             }.issubset(set(pq.read_schema(CONTEXT_FILE).names))}
    audit["financial_data_status"] = (
        "alignment provenance available; validate close timestamps before deployment"
        if audit["spot_alignment_provenance_present"]
        else "legacy spot/option close alignment is unrepaired; research metrics remain provisional"
    )
    for name, data in {"fit_tune": fit, "calibration": calibration,
                       "decision": decision, "validation": validation}.items():
        endpoint = pd.to_datetime(data.future_timestamp_30m)
        elapsed = (endpoint - data.timestamp).dt.total_seconds() / 60
        expected_target = (data.deviation_reduction_30m >= .5).astype(int)
        audit["blocks"][name] = {
            "rows": len(data), "sessions": data.timestamp.dt.date.nunique(),
            "start": str(data.timestamp.min()), "end": str(data.timestamp.max()),
            "positive_rate": float(data[TARGET].mean()),
            "duplicate_keys": int(data.duplicated(["timestamp", "instrument_key"]).sum()),
            "target_inconsistencies": int((expected_target != data[TARGET]).sum()),
            "horizon_inconsistencies": int((~elapsed.between(30, 35)).sum()),
            "cross_session_labels": int((endpoint.dt.date != data.timestamp.dt.date).sum()),
        }
        if any(audit["blocks"][name][key] for key in (
            "duplicate_keys", "target_inconsistencies", "horizon_inconsistencies", "cross_session_labels"
        )):
            raise ValueError(f"Label integrity check failed for {name}")
    (output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in audit.items() if k != "features"}, indent=2), flush=True)

    x, y = fit[features], fit[TARGET].astype(int).to_numpy()
    folds = list(purged_folds(fit))
    fold_rows = []

    def score_model(name, factory, record=False):
        scores = []
        for fold, a, b in folds:
            model = factory()
            model.fit(x.iloc[a], y[a])
            p = model.predict_proba(x.iloc[b])[:, 1]
            score = roc_auc_score(y[b], p)
            scores.append(score)
            if record:
                fold_rows.append({"candidate": name, "fold": fold,
                                  **_classification_metrics(y[b], p, .5)})
        return float(np.mean(scores) - .25 * np.std(scores)), float(np.mean(scores)), float(np.std(scores))

    specs, trials_rows = {}, []
    for family in ("xgboost", "lightgbm"):
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
        )

        def objective(trial):
            params = suggest(trial, family)
            value, mean, std = score_model(family, lambda: estimator(family, params, threads))
            trial.set_user_attr("params", params)
            trials_rows.append({"family": family, "trial": trial.number,
                               "objective": value, "mean_auc": mean, "std_auc": std,
                               "params": json.dumps(params, sort_keys=True)})
            print(f"{family} trial {trial.number + 1}/{trials}: CV AUC {mean:.4f}, SD {std:.4f}", flush=True)
            return value

        study.optimize(objective, n_trials=trials)
        specs[family] = (family, study.best_trial.user_attrs["params"])
    specs["random_forest"] = ("random_forest", {
        "n_estimators": 150, "max_depth": 10, "min_samples_leaf": 30, "max_features": "sqrt"
    })
    specs["logistic_regression"] = ("logistic_regression", {"C": .01})

    def make(name):
        if name == "equal_ensemble":
            return VotingClassifier(
                [(family, estimator(family, specs[family][1], threads))
                 for family in ("xgboost", "lightgbm")], voting="soft", n_jobs=1
            )
        family, params = specs[name]
        return estimator(family, params, threads)

    cv_rows = []
    for name in [*specs, "equal_ensemble"]:
        score, mean, std = score_model(name, lambda: make(name), record=True)
        cv_rows.append({"candidate": name, "selection_score": score, "mean_auc": mean, "std_auc": std})
    cv = pd.DataFrame(cv_rows).sort_values("selection_score", ascending=False)
    winner = str(cv.iloc[0].candidate)
    print("Selected only by internal CV:", winner, flush=True)
    cv.to_csv(output / "cv_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "cv_fold_metrics.csv", index=False)
    pd.DataFrame(trials_rows).to_csv(output / "tuning_trials.csv", index=False)

    partial = make(winner).fit(fit[features], fit[TARGET].astype(int))
    cal_probability = partial.predict_proba(calibration[features])[:, 1]
    sigmoid = LogisticRegression(C=100.0, solver="lbfgs", max_iter=1000).fit(
        logit(cal_probability), calibration[TARGET].astype(int)
    )
    decision_raw = partial.predict_proba(decision[features])[:, 1]
    decision_cal = sigmoid.predict_proba(logit(decision_raw))[:, 1]
    decision_y = decision[TARGET].astype(int).to_numpy()
    use_calibration = (
        sigmoid.coef_[0, 0] > 0
        and log_loss(decision_y, decision_cal) < log_loss(decision_y, decision_raw)
    )
    decision_p = decision_cal if use_calibration else decision_raw
    thresholds = _validation_thresholds(decision_y, decision_p)
    threshold = thresholds["validation_max_accuracy"]
    selective_decision = [selective_metrics(
        decision_y, decision_p, decision.timestamp.dt.date, margin
    ) for margin in np.arange(.05, .451, .025)]
    policies = {}
    for target in (.93, .94):
        feasible = [r for r in selective_decision if qualifies_selective(r, target)]
        policies[str(target)] = max(feasible, key=lambda r: r["coverage"]) if feasible else None
    # Freeze all model/threshold/calibration decisions before reading validation outcomes.
    config = {"winner": winner, "specs": specs, "feature_schema_version": 2,
              "selection": "internal purged CV mean AUC minus 0.25 * SD",
              "calibration_selected_on": "decision_block_log_loss",
              "use_sigmoid": bool(use_calibration), "decision_threshold": float(threshold),
              "selective_93_94_policies": policies, "test_split_opened": False,
              "financial_data_status": audit["financial_data_status"]}
    (output / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    pd.DataFrame(selective_decision).to_csv(output / "selective_decision_grid.csv", index=False)

    full = make(winner).fit(train[features], train[TARGET].astype(int))
    full_p = full.predict_proba(validation[features])[:, 1]
    partial_p = partial.predict_proba(validation[features])[:, 1]
    if use_calibration:
        partial_p = sigmoid.predict_proba(logit(partial_p))[:, 1]
    valid_y = validation[TARGET].astype(int).to_numpy()
    validation_rows = []
    for name, p, cut in (
        ("full_train_raw_fixed_0.50", full_p, .5),
        ("separate_calibration_and_decision_blocks", partial_p, threshold),
    ):
        validation_rows.append({"model": name, **_classification_metrics(valid_y, p, cut)})
    metrics = pd.DataFrame(validation_rows)
    metrics.to_csv(output / "validation_metrics.csv", index=False)
    fixed_selective = [
        {"model": name, **selective_metrics(valid_y, p, validation.timestamp.dt.date, margin)}
        for name, p in (("full_train_raw", full_p), ("strict_calibration", partial_p))
        for margin in (.10, .20, .30, .40)
    ]
    pd.DataFrame(fixed_selective).to_csv(output / "validation_predeclared_selective_bands.csv", index=False)
    intervals = grouped_intervals(valid_y, full_p, validation.timestamp.dt.date, .5)
    (output / "validation_session_bootstrap.json").write_text(json.dumps(intervals, indent=2), encoding="utf-8")
    selective_intervals = {
        str(margin): selective_grouped_intervals(
            valid_y, full_p, validation.timestamp.dt.date, margin
        ) for margin in (.30, .40)
    }
    (output / "validation_selective_session_intervals.json").write_text(
        json.dumps(selective_intervals, indent=2), encoding="utf-8"
    )
    predictions = validation[["timestamp", "instrument_key", "option_type", "time_to_expiry"]].copy()
    predictions["actual"] = valid_y
    predictions["full_probability"] = full_p
    predictions["strict_probability"] = partial_p
    predictions.to_parquet(output / "validation_predictions.parquet", index=False)
    regime_rows = []
    for column, labels in (
        ("month", validation.timestamp.dt.strftime("%Y-%m")),
        ("option_type", validation.option_type),
        ("expiry_days", pd.cut(validation.time_to_expiry * 365, [0, 1, 3, 7, np.inf], include_lowest=True).astype(str)),
    ):
        for value in sorted(labels.unique()):
            mask = (labels == value).to_numpy()
            if len(np.unique(valid_y[mask])) > 1:
                regime_rows.append({"dimension": column, "value": value, "rows": int(mask.sum()),
                                    **_classification_metrics(valid_y[mask], full_p[mask], .5)})
    pd.DataFrame(regime_rows).to_csv(output / "validation_regimes.csv", index=False)
    joblib.dump({"model": full, "feature_columns": features, "feature_schema_version": 2,
                 "threshold": .5, "financial_data_status": audit["financial_data_status"]}, output / "full_training_candidate.joblib")
    joblib.dump({"model": partial, "calibrator": sigmoid if use_calibration else None,
                 "feature_columns": features, "feature_schema_version": 2,
                 "threshold": threshold}, output / "strict_calibrated_candidate.joblib")
    print(metrics.to_string(index=False), flush=True)
    print("93-94% two-class selective policies:", json.dumps(policies), flush=True)
    print("Session bootstrap intervals:", json.dumps(intervals), flush=True)
    print("Research outputs:", str(output.resolve()), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "processed" / "research" / ("audit_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    args = parser.parse_args()
    if args.trials < 1 or args.threads < 1:
        parser.error("trials and threads must be positive")
    run(args.output, args.trials, args.threads)
