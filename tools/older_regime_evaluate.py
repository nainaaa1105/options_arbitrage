"""Evaluate the FROZEN accuracy model on a never-inspected earlier period (Sep 2024 - Aug 2025).

Nothing is refit or tuned: the bundle trained on Sep 2025 - Sep 2026 is applied once to sessions that precede all of its training data.
The test period is earlier in time than the training period, so this checks regime transfer (different year, different volatility),
not forward prediction. Features go through the same corrected-feature, market-regime and v2 paths as the frozen study.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\older_regime_evaluate.py --stage data\\research\\older_regime_20261101\\pipeline --output docs\\results\\older_regime
"""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_models
from accuracy_v2.experiment import TARGET, predict_bundle
from accuracy_v2.features import KEY, REQUIRED, build_v2_features
from build_volatility_features import calculate_daily_hv, make_daily_spot
from data_remediation_study import corrected_features
from market_regime_features import augment, market_table

OLDER = ROOT / "data/raw/upstox_older"
BUNDLE = ROOT / "data/research/accuracy_v2_20261031/final_model.joblib"
TRAINING_MAJORITY_RATE = None                  # computed from the frozen history below


def session_bootstrap(frame, column, repeats=2000, seed=7):
    days = frame.timestamp.dt.date.to_numpy()
    members = [np.flatnonzero(days == d) for d in np.unique(days)]
    rng = np.random.default_rng(seed)
    value = frame[column].to_numpy(float)
    draws = [value[np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])].mean() for _ in range(repeats)]
    return float(value.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    processed = args.stage / "data/processed"
    train_models.CONTEXT_FILE = processed / "options_signal_eligible.parquet"
    bundle = joblib.load(BUNDLE)
    daily = pd.read_parquet(OLDER / "nifty_daily_2024-08-01_to_2025-09-08.parquet").rename(columns={"close": "nifty_spot"})
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    daily = calculate_daily_hv(make_daily_spot(daily))
    market = market_table(pd.read_parquet(OLDER / "nifty_5minute_2024-09-20_to_2025-09-08.parquet"),
                          pd.read_parquet(OLDER / "india_vix_5minute_2024-09-20_to_2025-09-08.parquet"))
    data = augment(corrected_features(pd.read_parquet(processed / "ml_dataset.parquet"), daily), market)
    data["timestamp"] = pd.to_datetime(data.timestamp)
    v2 = build_v2_features(pd.read_parquet(processed / "options_parity_features.parquet", columns=REQUIRED))
    data = data.merge(v2, on=KEY, how="left", validate="one_to_one").reset_index(drop=True)
    y = data[TARGET].astype(int).to_numpy()
    p = predict_bundle(bundle, data)
    history_rate = float(json.loads((ROOT / "data/research/accuracy_v2_20261031/fresh_result.json").read_text())["training_majority_accuracy"])
    majority_class = 0 if history_rate >= .5 else 1                        # the class that was the majority in the frozen history
    data["correct"] = ((p >= .5).astype(int) == y).astype(float)
    data["majority_correct"] = (y == majority_class).astype(float)
    data["probability"] = p
    metrics = {"rows": len(data), "sessions": int(data.timestamp.dt.date.nunique()), "period": [str(data.timestamp.min()), str(data.timestamp.max())],
               "positive_rate": float(y.mean()), "model_accuracy_at_0.5": float(data.correct.mean()),
               "majority_class_accuracy": float(data.majority_correct.mean()), "roc_auc": float(roc_auc_score(y, p)),
               "frozen_threshold": float(bundle["threshold"]), "accuracy_at_frozen_threshold": float(((p >= bundle["threshold"]).astype(int) == y).mean())}
    data["gain"] = data.correct - data.majority_correct
    metrics["accuracy_gain_over_majority"] = dict(zip(("mean", "lower95", "upper95"), session_bootstrap(data, "gain")))
    metrics["model_accuracy_session_ci95"] = dict(zip(("mean", "lower95", "upper95"), session_bootstrap(data, "correct")))
    month = data.timestamp.dt.strftime("%Y-%m")
    by_month = pd.DataFrame({"month": month, "correct": data.correct, "majority": data.majority_correct, "y": y, "p": p}).groupby("month").apply(
        lambda g: pd.Series({"rows": len(g), "positive_rate": g.y.mean(), "model_accuracy": g.correct.mean(), "majority_accuracy": g.majority.mean(),
                             "auc": roc_auc_score(g.y, g.p) if g.y.nunique() == 2 else np.nan}), include_groups=False)
    by_month.to_csv(args.output / "by_month.csv")
    (args.output / "summary.json").write_text(json.dumps(metrics, indent=2, default=float))
    print(json.dumps(metrics, indent=2, default=float))
    print(by_month.round(4).to_string())


if __name__ == "__main__":
    main()
