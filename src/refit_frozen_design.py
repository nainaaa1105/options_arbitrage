"""Refit previously selected hyperparameters after adding historical data."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone

from train_models import TRAIN_FILE, VALID_FILE, TARGET, build_features, _classification_metrics


def run(source, output):
    output.mkdir(parents=True, exist_ok=False)
    candidate = joblib.load(source / "full_training_candidate.joblib")
    frames = build_features({"train": pd.read_parquet(TRAIN_FILE), "validation": pd.read_parquet(VALID_FILE)}, feature_schema_version=2)
    model = clone(candidate["model"])
    features = candidate["feature_columns"]
    model.fit(frames["train"][features], frames["train"][TARGET].astype(int))
    probability = model.predict_proba(frames["validation"][features])[:, 1]
    metrics = _classification_metrics(frames["validation"][TARGET], probability, .5)
    pd.DataFrame([metrics]).to_csv(output / "frozen_design_validation_metrics.csv", index=False)
    joblib.dump({**candidate, "model": model}, output / "full_training_candidate.joblib")
    config = json.loads((source / "frozen_config.json").read_text(encoding="utf-8"))
    config["frozen_design_source"] = str(source.resolve())
    config["expanded_history_hyperparameter_retuning"] = False
    config["refit_training_rows"] = len(frames["train"])
    (output / "frozen_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.output)
