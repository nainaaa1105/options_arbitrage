"""Chronological, label-purged model benchmark on the training period.

This experiment uses expanding date folds inside the existing training split.
It never reads the validation or test split and does not overwrite the selected
production model. Metrics use a fixed 0.50 probability cutoff for comparability.
"""

from pathlib import Path
import json

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit

from train_models import (
    OUTPUT_DIR,
    TARGET,
    TRAIN_FILE,
    VALID_FILE,
    _make_estimator,
    _feature_columns,
    _classification_metrics,
    build_features,
)


FOLDS = 3
RANDOM_STATE = 42


def _candidate_models():
    """Small predeclared regularization grid, not a test-set search."""
    return {
        "logistic_c_0.01": ("logistic_regression", {"C": 0.01}),
        "logistic_c_0.1": ("logistic_regression", {"C": 0.1}),
        "rf_depth_6_leaf_30": (
            "random_forest",
            {"n_estimators": 120, "max_depth": 6, "min_samples_leaf": 30,
             "max_features": "sqrt"},
        ),
        "rf_depth_10_leaf_30": (
            "random_forest",
            {"n_estimators": 120, "max_depth": 10, "min_samples_leaf": 30,
             "max_features": "sqrt"},
        ),
        "xgb_depth_2_150": (
            "xgboost",
            {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 2,
             "min_child_weight": 20.0, "subsample": 0.8,
             "colsample_bytree": 0.8, "reg_lambda": 20.0},
        ),
        "xgb_depth_3_150": (
            "xgboost",
            {"n_estimators": 150, "learning_rate": 0.03, "max_depth": 3,
             "min_child_weight": 20.0, "subsample": 0.8,
             "colsample_bytree": 0.8, "reg_lambda": 20.0},
        ),
        "lgb_leaves_7_150": (
            "lightgbm",
            {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 7,
             "max_depth": 3, "min_child_samples": 60, "subsample": 0.8,
             "colsample_bytree": 0.8, "reg_lambda": 20.0},
        ),
        "lgb_leaves_15_150": (
            "lightgbm",
            {"n_estimators": 150, "learning_rate": 0.03, "num_leaves": 15,
             "max_depth": 4, "min_child_samples": 60, "subsample": 0.8,
             "subsample_freq": 1,
             "colsample_bytree": 0.8, "reg_lambda": 20.0},
        ),
    }


def _metrics(y_true, probabilities):
    metrics = _classification_metrics(y_true, probabilities, 0.50)
    metrics.update({"rows": len(y_true), "positive_rate": float(np.mean(y_true))})
    return metrics


def run_benchmark():
    raw = pd.read_parquet(TRAIN_FILE)
    if "future_timestamp_30m" not in raw:
        raise ValueError("Training rows must include future_timestamp_30m for purge checks")
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw["future_timestamp_30m"] = pd.to_datetime(raw["future_timestamp_30m"])
    engineered = build_features({"train": raw})["train"]
    engineered["_date"] = engineered.timestamp.dt.normalize()
    dates = pd.DatetimeIndex(sorted(engineered._date.unique()))
    splitter = TimeSeriesSplit(n_splits=FOLDS)
    candidates = _candidate_models()
    rows = []

    for fold_no, (train_date_idx, valid_date_idx) in enumerate(splitter.split(dates), 1):
        train_dates = dates[train_date_idx]
        valid_dates = dates[valid_date_idx]
        first_valid_timestamp = engineered.loc[
            engineered._date.isin(valid_dates), "timestamp"
        ].min()
        fold_train = engineered.loc[engineered._date.isin(train_dates)].copy()
        # Purge training labels whose 30-minute outcome window reaches validation.
        fold_train = fold_train.loc[
            fold_train.future_timestamp_30m + pd.Timedelta(minutes=5) < first_valid_timestamp
        ].copy()
        fold_valid = engineered.loc[engineered._date.isin(valid_dates)].copy()
        features = _feature_columns(fold_train, allow_point_in_time_context=True)
        x_train, y_train = fold_train[features], fold_train[TARGET].astype(int).to_numpy()
        x_valid, y_valid = fold_valid[features], fold_valid[TARGET].astype(int).to_numpy()

        print(
            f"Fold {fold_no}/{FOLDS}: train {train_dates[0].date()}..{train_dates[-1].date()} "
            f"({len(fold_train):,} rows), validation "
            f"{valid_dates[0].date()}..{valid_dates[-1].date()} ({len(fold_valid):,} rows)"
        )
        for candidate_name, (family, params) in candidates.items():
            estimator = _make_estimator(family, params=params)
            estimator.fit(x_train, y_train)
            probabilities = estimator.predict_proba(x_valid)[:, 1]
            result = _metrics(y_valid, probabilities)
            result.update({
                "fold": fold_no,
                "candidate": candidate_name,
                "family": family,
                "train_start": str(train_dates[0].date()),
                "train_end": str(train_dates[-1].date()),
                "valid_start": str(valid_dates[0].date()),
                "valid_end": str(valid_dates[-1].date()),
                "train_rows": len(fold_train),
                "features": len(features),
            })
            rows.append(result)
            print(
                f"  {candidate_name}: ROC-AUC={result['roc_auc']:.4f}, "
                f"PR-AUC={result['pr_auc']:.4f}, accuracy@0.50={result['accuracy']:.4f}"
            )

    fold_results = pd.DataFrame(rows)
    metric_columns = [
        "accuracy", "precision", "recall", "f1", "specificity",
        "false_positive_rate", "predicted_positive_rate", "matthews_corrcoef",
        "roc_auc", "pr_auc", "balanced_accuracy", "brier_score", "log_loss",
        "expected_calibration_error_10bin", "positive_rate",
    ]
    summary = fold_results.groupby(["candidate", "family"])[metric_columns].agg(
        ["mean", "std"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(
        ["roc_auc_mean", "pr_auc_mean"], ascending=False
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_results.to_csv(OUTPUT_DIR / "walk_forward_model_folds.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "walk_forward_model_summary.csv", index=False)

    # Select only by internal expanding-window ROC-AUC, refit on the full
    # training period, and score the existing validation block once at 0.50.
    selected_name = summary.iloc[0]["candidate"]
    family, params = candidates[selected_name]
    train_raw = pd.read_parquet(TRAIN_FILE)
    valid_raw = pd.read_parquet(VALID_FILE)
    terminal = build_features({"train": train_raw, "validation": valid_raw})
    train_df, valid_df = terminal["train"], terminal["validation"]
    features = _feature_columns(train_df, allow_point_in_time_context=True)
    selected_model = _make_estimator(family, params=params)
    selected_model.fit(train_df[features], train_df[TARGET].astype(int).to_numpy())
    valid_probabilities = selected_model.predict_proba(valid_df[features])[:, 1]
    terminal_metrics = _metrics(
        valid_df[TARGET].astype(int).to_numpy(), valid_probabilities
    )
    terminal_metrics.update({
        "candidate": selected_name,
        "family": family,
        "threshold": 0.50,
        "threshold_selected_on": "predeclared",
        "validation_rows": len(valid_df),
        "validation_start": str(pd.to_datetime(valid_df.timestamp).min().date()),
        "validation_end": str(pd.to_datetime(valid_df.timestamp).max().date()),
        "walk_forward_mean_roc_auc": float(summary.iloc[0]["roc_auc_mean"]),
        "walk_forward_mean_pr_auc": float(summary.iloc[0]["pr_auc_mean"]),
    })
    pd.DataFrame([terminal_metrics]).to_csv(
        OUTPUT_DIR / "walk_forward_selected_validation_metrics.csv", index=False
    )
    joblib.dump(selected_model, OUTPUT_DIR / "walk_forward_selected_model.joblib")
    joblib.dump(features, OUTPUT_DIR / "walk_forward_selected_feature_columns.joblib")
    with open(OUTPUT_DIR / "walk_forward_selected_model_config.json", "w", encoding="utf-8") as stream:
        json.dump({
            "candidate": selected_name,
            "family": family,
            "params": params,
            "selection": "highest mean ROC-AUC across three expanding, label-purged folds",
            "threshold": 0.50,
            "threshold_selected_on": "predeclared",
            "training_rows": len(train_df),
            "feature_count": len(features),
            "feature_schema_version": 2,
        }, stream, indent=2)
    print("\nWalk-forward mean and standard deviation by candidate:")
    print(summary.to_string(index=False))
    print("\nSelected candidate validation metrics at fixed 0.50 threshold:")
    print(pd.Series(terminal_metrics).to_string())
    return fold_results, summary


if __name__ == "__main__":
    run_benchmark()
