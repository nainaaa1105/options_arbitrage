"""Coverage/censoring audit, drift remedies, target comparison and pricing diagnostics.

Creates an isolated report directory. Reads training and historical validation
for model development, never the test split. Existing data remain provisional
until option/spot close alignment has been repaired from authentic prices.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_pinball_loss
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from build_volatility_features import make_daily_spot, calculate_daily_hv
from pricing_research import black76_price, robust_loo_total_variance, static_price_diagnostics
from research_targets import build_path_targets
from robust_model_research import purged_folds, grouped_intervals
from train_models import TRAIN_FILE, VALID_FILE, TARGET, _classification_metrics, _feature_columns, build_features


POLICIES = {
    "expanding_uniform": {"balanced": False},
    "expanding_equal_sessions": {"balanced": True},
    "expanding_recency_40": {"balanced": True, "half_life": 40},
    "expanding_recency_20": {"balanced": True, "half_life": 20},
    "rolling_60_sessions": {"balanced": True, "window": 60},
    "rolling_40_sessions": {"balanced": True, "window": 40},
    "expanding_relative_inputs": {"balanced": False, "drop_absolute_index_levels": True},
    "recency_40_relative_inputs": {"balanced": True, "half_life": 40, "drop_absolute_index_levels": True},
}


def policy_rows_weights(data, policy):
    """Weights depend only on the available fitting period, never validation."""
    dates = pd.to_datetime(data.timestamp).dt.normalize()
    sessions = pd.DatetimeIndex(sorted(dates.unique()))
    keep = dates.isin(sessions[-policy["window"]:]) if policy.get("window") else np.ones(len(data), bool)
    positions = np.flatnonzero(keep)
    selected_dates = dates.iloc[positions]
    weight = np.ones(len(positions), dtype=float)
    if policy.get("balanced"):
        counts = selected_dates.value_counts()
        weight /= selected_dates.map(counts).to_numpy()
    if policy.get("half_life"):
        age = {date: len(sessions) - i - 1 for i, date in enumerate(sessions)}
        weight *= np.exp2(-selected_dates.map(age).to_numpy() / policy["half_life"])
    return positions, weight / weight.mean()


def make_model(regression=False, quantile=None):
    params = dict(n_estimators=200, learning_rate=.05, num_leaves=15,
                  max_depth=4, min_child_samples=60, reg_lambda=20.,
                  subsample=.8, subsample_freq=1, colsample_bytree=.8,
                  n_jobs=4, random_state=42, verbosity=-1)
    if quantile is not None:
        model = lgb.LGBMRegressor(objective="quantile", alpha=quantile, **params)
    else:
        model = lgb.LGBMRegressor(objective="regression_l1", **params) if regression else lgb.LGBMClassifier(**params)
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("model", model)])


def fit_policy(data, features, target, policy, regression=False, quantile=None):
    positions, weights = policy_rows_weights(data, policy)
    model = make_model(regression=regression, quantile=quantile)
    if policy.get("drop_absolute_index_levels"):
        kept = [name for name in features if name not in {"nifty_spot", "strike", "mkt_spot_mean"}]
        model = Pipeline([("select", ColumnTransformer([("relative_inputs", "passthrough", kept)],
                                                       remainder="drop")), *model.steps])
    y = np.asarray(target)[positions]
    model.fit(data.iloc[positions][features], y, model__sample_weight=weights)
    dates = pd.to_datetime(data.iloc[positions].timestamp).dt.date
    daily_weight = pd.Series(weights, index=dates).groupby(level=0).sum().to_numpy()
    information = {"rows": len(positions), "sessions": int(dates.nunique()),
                   "weight_effective_sessions": float(daily_weight.sum() ** 2 / np.sum(daily_weight ** 2))}
    return model, information


def censoring_audit(labels):
    current = labels.reference_price_deviation.abs()
    eligible = labels.signal_eligible.fillna(False).astype(bool)
    large = np.isfinite(current) & current.ge(1.)
    result = {"total_rows": len(labels), "signal_eligible": int(eligible.sum()),
              "eligible_below_one_point": int((eligible & ~large).sum()),
              "candidate_deviation_at_least_one_point": int((eligible & large).sum())}
    tables = []
    for horizon in (30, 60):
        future = pd.to_datetime(labels[f"future_timestamp_{horizon}m"])
        deadline = labels.timestamp + pd.Timedelta(minutes=horizon + 5)
        close = labels.timestamp.dt.normalize() + pd.Timedelta(hours=15, minutes=30)
        reason = np.select(
            [~eligible, eligible & ~large, eligible & large & future.isna() & deadline.gt(close),
             eligible & large & future.isna(),
             eligible & large & future.notna() & labels[f"future_deviation_{horizon}m"].isna()],
            ["not_signal_eligible", "below_target_deviation_floor", "session_end_censored",
             "unexpected_missing_future", "missing_reference_at_future"], default="observed_outcome")
        reason = pd.Series(reason, index=labels.index)
        result[str(horizon)] = {str(key): int(value) for key, value in reason.value_counts().items()}
        selected = labels.loc[eligible & large, ["timestamp", "instrument_key", "expiry"]].copy()
        selected["horizon_minutes"] = horizon
        selected["outcome_status"] = reason.loc[selected.index]
        tables.append(selected)
    return result, pd.concat(tables, ignore_index=True)


def corrected_features(raw, daily):
    d = raw.copy()
    hv_columns = [f"historical_volatility_{window}d" for window in (5, 10, 20)]
    d["trade_date"] = pd.to_datetime(d.trade_date).dt.date
    d = d.drop(columns=hv_columns).merge(daily, on="trade_date", how="left", validate="many_to_one")
    for window in (5, 10, 20):
        hv = d[f"historical_volatility_{window}d"]
        d[f"iv_minus_hv_{window}d"] = d.implied_volatility - hv
        d[f"iv_hv_ratio_{window}d"] = np.where(hv > 0, d.implied_volatility / hv, np.nan)
        d[f"hv_history_available_{window}d"] = hv.notna().astype(int)
    return build_features({"sample": d}, feature_schema_version=2)["sample"]


def covariate_drift(train, validation, features):
    rows = []
    for feature in features:
        a = train[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        b = validation[feature].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if len(a) < 10 or len(b) < 10:
            continue
        edges = np.unique(np.quantile(a, np.linspace(0, 1, 11)))
        if len(edges) < 3:
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        pa = np.histogram(a, edges)[0] / len(a)
        pb = np.histogram(b, edges)[0] / len(b)
        pa, pb = np.maximum(pa, 1e-5), np.maximum(pb, 1e-5)
        rows.append({"feature": feature, "psi": float(np.sum((pb - pa) * np.log(pb / pa))),
                     "ks_statistic": float(ks_2samp(a, b).statistic),
                     "train_missing_rate": float(train[feature].isna().mean()),
                     "validation_missing_rate": float(validation[feature].isna().mean())})
    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def pricing_diagnostics(context, training_end):
    selected = context.loc[(context.timestamp <= training_end) & context.timestamp.dt.minute.eq(0)
                           & context.timestamp.dt.hour.isin([10, 12, 14])].copy()
    dates = sorted(selected.trade_date.unique())[:15]
    selected = selected.loc[selected.trade_date.isin(dates)]
    rows = []
    for (timestamp, expiry, kind), group in selected.groupby(["timestamp", "expiry", "option_type"]):
        candidate = robust_loo_total_variance(group)
        candidate_price = black76_price(
            candidate.nifty_spot * np.exp(candidate.risk_free_rate * candidate.time_to_expiry),
            candidate.strike, candidate.time_to_expiry, candidate.research_reference_iv,
            candidate.risk_free_rate, kind,
        )
        common = np.isfinite(candidate_price) & np.isfinite(group.reference_theoretical_price)
        if common.sum() < 3:
            continue
        for method, prices in (("existing_reference", group.reference_theoretical_price.to_numpy()),
                               ("robust_local_variance", candidate_price)):
            row = static_price_diagnostics(group.strike.to_numpy()[common], prices[common], kind,
                                           float(np.exp(-group.risk_free_rate.iloc[0] * group.time_to_expiry.iloc[0])))
            rows.append({"timestamp": timestamp, "expiry": expiry, "option_type": kind,
                         "method": method, "common_strikes": int(common.sum()), **row})
    return pd.DataFrame(rows)


def run(output, daily_spot_file=None):
    output.mkdir(parents=True, exist_ok=False)
    processed = TRAIN_FILE.parents[1]
    columns = ["timestamp", "trade_date", "instrument_key", "expiry", "signal_eligible",
               "reference_price_deviation", "nifty_spot", "implied_volatility", "reference_iv",
               "time_to_expiry", "risk_free_rate", "option_type", "strike", "log_moneyness",
               "reference_theoretical_price"]
    context = pd.read_parquet(processed / "options_parity_features.parquet", columns=columns)
    context["timestamp"] = pd.to_datetime(context.timestamp)
    label_columns = ["timestamp", "instrument_key", "expiry", "signal_eligible", "reference_price_deviation"]
    label_columns += [f"{prefix}_{horizon}m" for horizon in (30, 60)
                      for prefix in ("future_timestamp", "future_deviation", "label_valid")]
    labels = pd.read_parquet(processed / "options_convergence_labels.parquet", columns=label_columns)
    labels["timestamp"] = pd.to_datetime(labels.timestamp)
    missingness, outcome_rows = censoring_audit(labels)
    (output / "outcome_audit.json").write_text(json.dumps(missingness, indent=2), encoding="utf-8")
    outcome_rows.to_parquet(output / "outcome_status.parquet", index=False)
    coverage = context.groupby(context.timestamp.dt.strftime("%Y-%m")).agg(
        rows=("timestamp", "size"), sessions=("trade_date", "nunique"), expiries=("expiry", "nunique")
    )
    coverage.to_csv(output / "coverage_by_month.csv")
    observed_daily = make_daily_spot(context)
    observed_daily["calendar_gap_days"] = pd.to_datetime(observed_daily.trade_date).diff().dt.days
    observed_daily.to_csv(output / "observed_daily_history.csv", index=False)
    hv_source = observed_daily
    if daily_spot_file is not None:
        hv_source = pd.read_parquet(daily_spot_file).rename(columns={"close": "nifty_spot"})
        hv_source["timestamp"] = pd.to_datetime(hv_source.timestamp)
        hv_source["trade_date"] = hv_source.timestamp.dt.date
        hv_source = make_daily_spot(hv_source)
    daily = calculate_daily_hv(hv_source)
    daily.to_csv(output / "gap_aware_daily_hv.csv", index=False)
    raw_train, raw_valid = pd.read_parquet(TRAIN_FILE), pd.read_parquet(VALID_FILE)
    train, validation = corrected_features(raw_train, daily), corrected_features(raw_valid, daily)
    features = _feature_columns(train, allow_point_in_time_context=True)
    covariate_drift(train, validation, features).to_csv(output / "feature_drift.csv", index=False)
    folds = list(purged_folds(train))
    cv_rows = []
    for name, policy in POLICIES.items():
        for fold, a, b in folds:
            fitting, evaluating = train.iloc[a], train.iloc[b]
            model, info = fit_policy(fitting, features, fitting[TARGET].to_numpy(), policy)
            p = model.predict_proba(evaluating[features])[:, 1]
            metrics = _classification_metrics(evaluating[TARGET], p, .5)
            cv_rows.append({"policy": name, "fold": fold, **info, **metrics})
        scores = [r["roc_auc"] for r in cv_rows if r["policy"] == name]
        print(f"{name}: CV AUC {np.mean(scores):.4f} (+/- {np.std(scores):.4f})", flush=True)
    cv = pd.DataFrame(cv_rows)
    summary = cv.groupby("policy").agg(mean_auc=("roc_auc", "mean"), std_auc=("roc_auc", "std"),
                                      mean_accuracy=("accuracy", "mean"), mean_pr_auc=("pr_auc", "mean"))
    summary["selection_score"] = summary.mean_auc - .25 * summary.std_auc
    summary = summary.sort_values("selection_score", ascending=False)
    winner = str(summary.index[0])
    cv.to_csv(output / "drift_cv_folds.csv", index=False)
    summary.to_csv(output / "drift_cv_summary.csv")
    frozen = {"selected_policy": winner, "policies": POLICIES, "feature_schema_version": 2,
              "gap_aware_hv": True, "threshold": .5, "test_split_opened": False,
              "data_status": "provisional: historical option/spot alignment not yet repaired"}
    (output / "frozen_training_policy.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    validation_rows = []
    for name in dict.fromkeys(["expanding_uniform", winner]):
        model, info = fit_policy(train, features, train[TARGET].to_numpy(), POLICIES[name])
        p = model.predict_proba(validation[features])[:, 1]
        validation_rows.append({"policy": name, **info, **_classification_metrics(validation[TARGET], p, .5)})
        if name == winner:
            joblib.dump({"model": model, "feature_columns": features, **frozen}, output / "drift_candidate.joblib")
            (output / "validation_session_intervals.json").write_text(json.dumps(grouped_intervals(
                validation[TARGET].to_numpy(), p, validation.timestamp.dt.date, .5
            ), indent=2), encoding="utf-8")
    pd.DataFrame(validation_rows).to_csv(output / "drift_validation_metrics.csv", index=False)

    paths = build_path_targets(context)
    paths.to_parquet(output / "path_target_comparison.parquet", index=False)
    complete = paths.loc[paths.complete_path].drop(columns=["expiry"])
    path_train = train.merge(complete, on=["timestamp", "instrument_key"], validate="one_to_one")
    path_valid = validation.merge(complete, on=["timestamp", "instrument_key"], validate="one_to_one")
    target_rows = []
    for target in ("endpoint", "first_passage", "sustained_two_bars"):
        y = path_train[target].astype(int).to_numpy()
        actual = path_valid[target].astype(int).to_numpy()
        model, info = fit_policy(path_train, features, y, POLICIES[winner])
        p = model.predict_proba(path_valid[features])[:, 1]
        baseline_class = int(np.mean(y) >= .5)
        target_rows.append({"target": target, "train_rows": len(y), "validation_rows": len(actual),
                            "train_positive_rate": float(y.mean()), "validation_positive_rate": float(actual.mean()),
                            "training_majority_baseline_accuracy": float(np.mean(actual == baseline_class)),
                            **_classification_metrics(actual, p, .5)})
        joblib.dump({"model": model, "feature_columns": features, "target": target,
                     "threshold": .5, **frozen}, output / f"target_candidate_{target}.joblib")
    target_table = pd.DataFrame(target_rows)
    target_table.to_csv(output / "target_validation_comparison.csv", index=False)
    regressor, _ = fit_policy(path_train, features, path_train.log_future_current_deviation,
                              POLICIES[winner], regression=True)
    prediction = regressor.predict(path_valid[features])
    actual = path_valid.log_future_current_deviation.to_numpy()
    baseline = np.full(len(actual), path_train.log_future_current_deviation.median())
    regression = {"target": "log((future_absolute_deviation + .05)/(current_absolute_deviation + .05))",
                  "mae": mean_absolute_error(actual, prediction),
                  "training_median_baseline_mae": mean_absolute_error(actual, baseline)}
    (output / "continuous_target_metrics.json").write_text(json.dumps(regression, indent=2), encoding="utf-8")
    joblib.dump({"model": regressor, "feature_columns": features,
                 "target": "log_future_current_deviation", **frozen, "threshold": None}, output / "continuous_target_candidate.joblib")
    quantiles, quantile_metrics = {}, []
    for q in (.1, .9):
        model, _ = fit_policy(path_train, features, path_train.log_future_current_deviation,
                              POLICIES[winner], regression=True, quantile=q)
        forecast = model.predict(path_valid[features])
        quantiles[q] = forecast
        quantile_metrics.append({"quantile": q,
                                 "pinball_loss": mean_pinball_loss(actual, forecast, alpha=q),
                                 "training_quantile_baseline_pinball": mean_pinball_loss(
                                     actual, np.full(len(actual), path_train.log_future_current_deviation.quantile(q)), alpha=q)})
        joblib.dump({"model": model, "feature_columns": features, "quantile": q,
                     "target": "log_future_current_deviation", **frozen, "threshold": None}, output / f"quantile_{q}_candidate.joblib")
    low, high = np.minimum(quantiles[.1], quantiles[.9]), np.maximum(quantiles[.1], quantiles[.9])
    uncertainty = {"nominal_coverage": .8, "observed_coverage": float(((actual >= low) & (actual <= high)).mean()),
                   "mean_interval_width": float((high - low).mean()),
                   "raw_crossed_interval_rate": float((quantiles[.1] > quantiles[.9]).mean()),
                   "quantile_metrics": quantile_metrics}
    (output / "uncertainty_forecast_metrics.json").write_text(json.dumps(uncertainty, indent=2), encoding="utf-8")
    prices = pricing_diagnostics(context, train.timestamp.max())
    prices.to_csv(output / "pricing_surface_diagnostics.csv", index=False)
    if len(prices):
        prices.groupby("method")[["observations", "vertical_spread_violations", "convexity_violations"]].sum().to_csv(
            output / "pricing_diagnostic_summary.csv")
    print("Outcome audit:", json.dumps(missingness), flush=True)
    print("Selected policy:", winner, flush=True)
    print(pd.DataFrame(validation_rows).to_string(index=False), flush=True)
    print(target_table.to_string(index=False), flush=True)
    print("Continuous target:", json.dumps(regression), flush=True)
    print("Outputs:", output.resolve(), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "data/processed/research" / ("remediation_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")))
    parser.add_argument("--daily-spot", type=Path, help="Complete independent daily spot history for corrected HV")
    args = parser.parse_args()
    run(args.output, args.daily_spot)
