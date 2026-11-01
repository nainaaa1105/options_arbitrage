"""Accuracy programme v2: purged chronological model search, then ONE frozen evaluation on the post-cutoff rows.

Protocol (also written to protocol.json before fitting):
  * model/feature/threshold selection uses history <= 2026-09-15 only; the post-cutoff rows are never read until freeze;
  * five expanding validation blocks V1..V5, one-session embargo, training labels must have ended before the block;
  * Optuna tunes on V1..V3 only; all candidates are compared on V4..V5 (never used for tuning);
  * thresholds and calibrators are fitted on EARLIER blocks' out-of-fold predictions and applied to later blocks.
The post-cutoff period was reviewed during baseline development, so it is not a pristine holdout; only dates
collected after this run can confirm the result.
"""
import argparse
import hashlib
import json
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline

import train_models
from build_volatility_features import calculate_daily_hv, make_daily_spot
from data_remediation_study import corrected_features
from market_regime_features import augment, market_table
from robust_model_research import purged_folds

from .features import GROUPS, KEY, REQUIRED, V2_COLUMNS, build_v2_features

warnings.filterwarnings("ignore")
TARGET = train_models.TARGET
CUTOFF = pd.Timestamp("2026-09-15").date()
DROP_ABSOLUTE = {"nifty_spot", "strike", "mkt_spot_mean"}
THREADS = 4
BASELINE_LGB = dict(n_estimators=200, learning_rate=.05, num_leaves=15, max_depth=4, min_child_samples=60, reg_lambda=20.,
                    subsample=.8, subsample_freq=1, colsample_bytree=.8, n_jobs=THREADS, random_state=42, verbosity=-1)
FORBIDDEN = ("target", "future", "label", "converged", "reduction", "actual_", "exit_", "entry_", "net_pnl", "profitable")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_tables(stage, daily_file, regime_dir):
    processed = stage / "data/processed"
    train_models.CONTEXT_FILE = processed / "options_signal_eligible.parquet"
    daily = pd.read_parquet(daily_file).rename(columns={"close": "nifty_spot"})
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    daily = calculate_daily_hv(make_daily_spot(daily))
    design = json.loads((regime_dir / "frozen_design.json").read_text(encoding="utf-8"))
    market = market_table(pd.read_parquet(design["spot_input"]), pd.read_parquet(design["vix_input"]))
    candidate = joblib.load(regime_dir / "selected_candidate.joblib")
    base_columns = [c for c in candidate["feature_columns"] if c not in DROP_ABSOLUTE]
    history = augment(corrected_features(pd.read_parquet(processed / "ml_dataset.parquet"), daily), market)
    fresh = augment(corrected_features(pd.read_parquet(processed / "fresh_ml_dataset.parquet"), daily), market)
    surface = pd.read_parquet(processed / "options_parity_features.parquet", columns=REQUIRED)
    v2 = build_v2_features(surface)                      # full surface, all dates, causal by construction
    history["timestamp"] = pd.to_datetime(history.timestamp)
    fresh["timestamp"] = pd.to_datetime(fresh.timestamp)
    history = history.merge(v2, on=KEY, how="left", validate="one_to_one")
    fresh = fresh.merge(v2, on=KEY, how="left", validate="one_to_one")
    assert history.timestamp.dt.date.max() <= CUTOFF < fresh.timestamp.dt.date.min()
    return history.reset_index(drop=True), fresh.reset_index(drop=True), base_columns, candidate


def assert_no_forbidden(columns):
    bad = [c for c in columns if any(token in c.lower() for token in FORBIDDEN)]
    if bad:
        raise AssertionError(f"Forbidden predictor names: {bad}")


def make_estimator(name, params=None):
    p = dict(params or {})
    if name == "lgb":
        return lgb.LGBMClassifier(**{**BASELINE_LGB, **p})
    if name == "lgb_balanced":
        return lgb.LGBMClassifier(**{**BASELINE_LGB, "class_weight": "balanced", **p})
    if name == "xgb":
        return xgb.XGBClassifier(n_estimators=300, learning_rate=.05, max_depth=4, min_child_weight=20, subsample=.8,
                                 colsample_bytree=.8, reg_lambda=20., tree_method="hist", n_jobs=THREADS, random_state=42,
                                 eval_metric="logloss", **p)
    if name == "cat":
        from catboost import CatBoostClassifier        # optional dependency (requirements/accuracy-v2.txt); only this model needs it
        return CatBoostClassifier(iterations=400, depth=5, learning_rate=.06, l2_leaf_reg=10., verbose=0, thread_count=THREADS,
                                  random_seed=42, **p)
    if name == "et":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             ExtraTreesClassifier(n_estimators=300, min_samples_leaf=40, max_features=.4, n_jobs=THREADS, random_state=42))
    if name == "rf":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             RandomForestClassifier(n_estimators=200, min_samples_leaf=40, max_features=.3, n_jobs=THREADS, random_state=42))
    raise ValueError(name)


def fold_metrics(y, p):
    out = {"auc": roc_auc_score(y, p), "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
           "accuracy_050": float(((p >= .5) == y).mean()), "majority": float(max(y.mean(), 1 - y.mean()))}
    return out


def best_threshold(y, p):
    grid = np.linspace(.2, .8, 61)
    scores = [((p >= t) == y).mean() for t in grid]
    return float(grid[int(np.argmax(scores))])


def run(stage, daily_file, regime_dir, output, trials=20, smoke=False):
    output.mkdir(parents=True, exist_ok=False)
    import shutil
    shutil.copytree(Path(__file__).parent, output / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    history, fresh, base_columns, baseline = load_tables(stage, daily_file, regime_dir)
    if smoke:                                     # software check only: thin rows AFTER causal features are built
        history, fresh = history.iloc[::8].reset_index(drop=True), fresh.iloc[::8].reset_index(drop=True)
    y = history[TARGET].astype(int).to_numpy()
    sets = {"base": base_columns, "v2": base_columns + V2_COLUMNS}
    for name, columns in sets.items():
        assert_no_forbidden(columns)
    protocol = {"cutoff": str(CUTOFF), "folds": 5, "embargo_sessions": 1, "tune_on": "V1-V3", "compare_on": "V4-V5",
                "history_rows": len(history), "fresh_rows_not_read_until_freeze": len(fresh),
                "base_features": len(base_columns), "new_features": len(V2_COLUMNS), "feature_groups": GROUPS,
                "baseline_hyperparameters": BASELINE_LGB, "optuna_trials": trials, "selection_rule": "highest mean AUC on V4-V5 among candidates; ties to simpler",
                "fresh_status": "reviewed during baseline development; single evaluation after freeze; NOT pristine"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, default=str))
    folds = list(purged_folds(history, n_splits=5, gap_sessions=1))
    rows, oof = [], {}

    def evaluate(name, columns, estimator_name, params=None, fold_ids=(1, 2, 3, 4, 5), model=None):
        pred = np.full(len(history), np.nan)
        for fold, train, valid in folds:
            if fold not in fold_ids:
                continue
            m = clone(model) if model is not None else make_estimator(estimator_name, params)
            m.fit(history.iloc[train][columns], y[train])
            pred[valid] = m.predict_proba(history.iloc[valid][columns])[:, 1]
            rows.append({"candidate": name, "fold": fold, "train_rows": len(train), "valid_rows": len(valid), **fold_metrics(y[valid], pred[valid])})
        oof[name] = pred
        got = [r for r in rows if r["candidate"] == name]
        print(f"{name}: " + ", ".join(f"V{r['fold']} AUC {r['auc']:.4f} acc {r['accuracy_050']:.4f}" for r in got), flush=True)

    # Baseline exactly as frozen (median imputer + LightGBM) and the same learner on NaN-native inputs.
    baseline_model = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                               ("model", lgb.LGBMClassifier(**BASELINE_LGB))])           # the frozen design, on the same blocks
    evaluate("baseline_frozen", base_columns, None, model=baseline_model)
    evaluate("lgb_base_features", base_columns, "lgb")
    evaluate("lgb_v2_all", sets["v2"], "lgb")
    for group, columns in GROUPS.items():
        evaluate(f"lgb_base+{group}", base_columns + columns, "lgb")
    evaluate("lgb_v2_balanced", sets["v2"], "lgb_balanced")
    for name in ("xgb", "cat", "et", "rf"):
        evaluate(f"{name}_v2", sets["v2"], name)

    def objective(trial):
        params = dict(n_estimators=trial.suggest_int("n_estimators", 150, 600), learning_rate=trial.suggest_float("learning_rate", .02, .1, log=True),
                      num_leaves=trial.suggest_int("num_leaves", 7, 63), max_depth=trial.suggest_int("max_depth", 3, 8),
                      min_child_samples=trial.suggest_int("min_child_samples", 20, 300), reg_lambda=trial.suggest_float("reg_lambda", 1, 100, log=True),
                      colsample_bytree=trial.suggest_float("colsample_bytree", .3, .9), subsample=trial.suggest_float("subsample", .5, .9),
                      min_split_gain=trial.suggest_float("min_split_gain", 0, 1))
        scores = []
        for fold, train, valid in folds[:3]:
            m = make_estimator("lgb", params)
            m.fit(history.iloc[train][sets["v2"]], y[train])
            scores.append(log_loss(y[valid], np.clip(m.predict_proba(history.iloc[valid][sets["v2"]])[:, 1], 1e-6, 1 - 1e-6)))
        return float(np.mean(scores))

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    tuned = study.best_params
    (output / "optuna_best_params.json").write_text(json.dumps({"params": tuned, "tuning_logloss_V1_V3": study.best_value}, indent=2))
    evaluate("lgb_v2_tuned", sets["v2"], "lgb", tuned)

    members = ["lgb_v2_tuned", "xgb_v2", "cat_v2", "et_v2"]
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    fold_of = np.full(len(history), 0)
    for fold, _, valid in folds:
        fold_of[valid] = fold
    blend = np.nanmean(np.column_stack([oof[m] for m in members]), axis=1)
    blend[fold_of == 0] = np.nan
    oof["blend_mean"] = blend
    stack = np.full(len(history), np.nan)
    for k in (3, 4, 5):                                         # meta-learner fitted only on earlier blocks' OOF
        earlier = (fold_of >= 2) & (fold_of < k)
        meta = LogisticRegression(C=1.0).fit(np.column_stack([logit(oof[m][earlier]) for m in members]), y[earlier])
        now = fold_of == k
        stack[now] = meta.predict_proba(np.column_stack([logit(oof[m][now]) for m in members]))[:, 1]
    oof["stack_logistic"] = stack
    for name in ("blend_mean", "stack_logistic"):
        for k in (3, 4, 5):
            now = fold_of == k
            if np.isfinite(oof[name][now]).all():
                rows.append({"candidate": name, "fold": k, "train_rows": None, "valid_rows": int(now.sum()), **fold_metrics(y[now], oof[name][now])})
    table = pd.DataFrame(rows)
    table.to_csv(output / "cv_folds.csv", index=False)
    window = table[table.fold.isin([4, 5])].groupby("candidate").agg(auc=("auc", "mean"), logloss=("logloss", "mean"),
                                                                    accuracy_050=("accuracy_050", "mean"), majority=("majority", "mean"))
    window = window.sort_values("auc", ascending=False)
    window.to_csv(output / "selection_window_V4_V5.csv")
    print(window.round(4).to_string(), flush=True)
    pd.DataFrame({"row": np.arange(len(history)), "fold": fold_of, "y": y, **{k: v for k, v in oof.items()}}).to_parquet(output / "oof_predictions.parquet")
    # ---- select, threshold/calibrate on EARLIER OOF only (V2..V3 for V4-V5 estimate), then refit and freeze
    chosen = window.index[0]
    print("selected:", chosen, flush=True)
    used = np.isfinite(oof[chosen]) & (fold_of >= 2)
    threshold = best_threshold(y[used & (fold_of <= 3)], oof[chosen][used & (fold_of <= 3)])
    realised = {f"V{k}": {"accuracy_at_threshold": float(((oof[chosen][fold_of == k] >= threshold) == y[fold_of == k]).mean()),
                           "accuracy_050": float(((oof[chosen][fold_of == k] >= .5) == y[fold_of == k]).mean()),
                           "majority": float(max(y[fold_of == k].mean(), 1 - y[fold_of == k].mean()))} for k in (4, 5) if np.isfinite(oof[chosen][fold_of == k]).all()}
    final_threshold = best_threshold(y[used], oof[chosen][used])
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(oof[chosen][used], y[used])
    final_members = {}
    spec = {"lgb_v2_tuned": ("lgb", tuned, "v2"), "xgb_v2": ("xgb", None, "v2"), "cat_v2": ("cat", None, "v2"), "et_v2": ("et", None, "v2"),
            "rf_v2": ("rf", None, "v2"), "lgb_v2_all": ("lgb", None, "v2"), "lgb_base_features": ("lgb", None, "base"), "lgb_v2_balanced": ("lgb_balanced", None, "v2")}
    names = members if chosen in ("blend_mean", "stack_logistic") else [chosen]
    for name in names:
        if name == "baseline_frozen":
            raise RuntimeError("Baseline won; no improved model to freeze")
        est, params, which = spec.get(name, ("lgb", None, "base" if name == "baseline_frozen" else "v2")) if not name.startswith("lgb_base+") else ("lgb", None, None)
        columns = sets[which] if which else base_columns + GROUPS[name.split("+")[1]]
        m = make_estimator(est, params).fit(history[columns], y)
        final_members[name] = {"model": m, "columns": columns}
    meta = None
    if chosen == "stack_logistic":
        earlier = (fold_of >= 2)
        meta = LogisticRegression(C=1.0).fit(np.column_stack([logit(oof[m][earlier]) for m in members]), y[earlier])
    bundle = {"version": "accuracy_v2", "chosen": chosen, "members": final_members, "meta": meta, "calibrator": calibrator,
              "threshold": final_threshold, "feature_schema": "v2.1", "base_columns": base_columns, "v2_columns": V2_COLUMNS,
              "training_rows": len(history), "training_end": str(history.timestamp.max())}
    joblib.dump(bundle, output / "final_model.joblib")
    frozen = {"frozen_before_fresh_read": True, "chosen": chosen, "threshold": final_threshold, "threshold_selected_on": "OOF V2-V5",
              "dev_estimates_at_early_threshold": realised, "final_model_sha256": sha(output / "final_model.joblib"),
              "cv_selection_table_sha256": sha(output / "selection_window_V4_V5.csv"), "optuna": tuned}
    (output / "frozen_before_fresh.json").write_text(json.dumps(frozen, indent=2, default=str))
    print("FROZEN; opening post-cutoff rows once", flush=True)
    evaluate_fresh(output, bundle, history, fresh, baseline_predictions=stage / "fresh_evaluation/endpoint_fresh_predictions.parquet")


def predict_bundle(bundle, frame):
    members = bundle["members"]
    probs = {n: m["model"].predict_proba(frame[m["columns"]])[:, 1] for n, m in members.items()}
    if bundle["chosen"] == "stack_logistic":
        clip = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
        return bundle["meta"].predict_proba(np.column_stack([clip(probs[n]) for n in members]))[:, 1]
    return np.mean(np.column_stack(list(probs.values())), axis=1)


def evaluate_fresh(output, bundle, history, fresh, baseline_predictions):
    y = fresh[TARGET].astype(int).to_numpy()
    p = predict_bundle(bundle, fresh)
    threshold = bundle["threshold"]
    result = {"rows": len(fresh), "sessions": int(fresh.timestamp.dt.date.nunique()), "positive_rate": float(y.mean()),
              "training_majority_accuracy": float((y == int(history[TARGET].mean() >= .5)).mean())}
    result["at_050"] = train_models._classification_metrics(y, p, .5)
    result["at_frozen_threshold"] = train_models._classification_metrics(y, p, threshold)
    base = pd.read_parquet(baseline_predictions)
    aligned = fresh[KEY].merge(base.rename(columns={"probability": "baseline_probability"}), on=["timestamp", "instrument_key"], how="left", validate="one_to_one")
    assert aligned.baseline_probability.notna().all() and (aligned.actual.to_numpy() == y).all(), "Baseline rows do not align"
    bp = aligned.baseline_probability.to_numpy()
    result["baseline_at_050"] = train_models._classification_metrics(y, bp, .5)
    dates = fresh.timestamp.dt.date.to_numpy()
    rng = np.random.default_rng(7)
    unique = np.unique(dates)
    members = [np.flatnonzero(dates == d) for d in unique]
    _diffs, aucs, accs = [], [], []
    for _ in range(2000):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        accs.append(float(((p[idx] >= .5) == y[idx]).mean() - ((bp[idx] >= .5) == y[idx]).mean()))
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], bp[idx]))
    result["paired_vs_baseline"] = {"accuracy_diff": float(((p >= .5) == y).mean() - ((bp >= .5) == y).mean()),
                                    "accuracy_ci95": [float(np.quantile(accs, .025)), float(np.quantile(accs, .975))],
                                    "auc_diff": float(roc_auc_score(y, p) - roc_auc_score(y, bp)),
                                    "auc_ci95": [float(np.quantile(aucs, .025)), float(np.quantile(aucs, .975))], "method": "session-cluster bootstrap"}
    selective = []
    for margin in np.arange(0, .5, .05):
        lo, hi = p <= .5 - margin, p >= .5 + margin
        acc = lo | hi
        selective.append({"margin": round(float(margin), 2), "coverage": float(acc.mean()), "accepted": int(acc.sum()),
                          "accuracy": float((hi[acc] == y[acc]).mean()) if acc.any() else None, "positives": int(hi.sum()),
                          "positive_precision": float(y[hi].mean()) if hi.any() else None, "negatives": int(lo.sum()),
                          "negative_precision": float(1 - y[lo].mean()) if lo.any() else None})
    pd.DataFrame(selective).to_csv(output / "fresh_selective.csv", index=False)
    f = fresh.assign(p=p, y=y, hit=((p >= .5) == y))
    f["confidence"] = pd.cut(np.abs(p - .5), [-.01, .1, .2, .3, .4, .51])
    def slice_table(label, key):
        return f.groupby(key, observed=True).agg(rows=("y", "size"), positive_rate=("y", "mean"), accuracy=("hit", "mean")).reset_index().rename(columns={key.name if hasattr(key, "name") else key: "slice"}).assign(by=label)
    regimes = [("confidence", f["confidence"]), ("option_type", f.option_type), ("session", pd.cut(f.intraday_minute, [-1, 45, 240, 400], labels=["open", "mid", "close"])),
               ("dte", pd.cut(f.days_to_expiry, [-.1, 1, 3, 100], labels=["<=1d", "1-3d", ">3d"])), ("day_of_week", f.day_of_week),
               ("spot_vol", pd.qcut(f.spot_realized_vol_30m, 3, labels=["low", "mid", "high"], duplicates="drop")),
               ("expiry", f.expiry.astype(str)), ("moneyness", pd.cut(f.abs_log_moneyness, [-.01, .01, .03, .2], labels=["atm", "near", "far"]))]
    out = []
    for label, key in regimes:
        t = f.groupby(key, observed=True).agg(rows=("y", "size"), positive_rate=("y", "mean"), accuracy=("hit", "mean")).reset_index()
        t.columns = ["slice", "rows", "positive_rate", "accuracy"]
        t["by"] = label
        t["majority_accuracy"] = np.maximum(t.positive_rate, 1 - t.positive_rate)
        out.append(t)
    pd.concat(out).to_csv(output / "fresh_regimes.csv", index=False)
    importance = []
    for name, m in bundle["members"].items():
        model = m["model"]
        raw = model.feature_importances_ if hasattr(model, "feature_importances_") else model[-1].feature_importances_
        importance.append(pd.DataFrame({"member": name, "feature": m["columns"], "importance": raw / (np.sum(raw) + 1e-12)}))
    pd.concat(importance).to_csv(output / "feature_importance.csv", index=False)
    pd.DataFrame({"timestamp": fresh.timestamp, "instrument_key": fresh.instrument_key, "actual": y, "probability": p, "baseline_probability": bp}).to_parquet(output / "fresh_predictions.parquet", index=False)
    (output / "fresh_result.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: result[k] for k in ("at_050", "baseline_at_050", "paired_vs_baseline")}, indent=2, default=str)[:3000], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=Path("data/research/expanded_20261030"))
    parser.add_argument("--daily-spot", type=Path, default=Path("data/raw/upstox/nifty_daily_to_20261029.parquet"))
    parser.add_argument("--regime-dir", type=Path, default=Path("data/research/expanded_20261030/regime_study"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run(args.stage, args.daily_spot, args.regime_dir, args.output, args.trials, args.smoke)
