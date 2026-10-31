"""Run a frozen research bundle with its matching causal feature context."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import train_models
from build_volatility_features import make_daily_spot, calculate_daily_hv
from data_remediation_study import corrected_features
from market_regime_features import market_table, augment
from research_targets import session_horizon_eligible


def predict(run_dir, input_file, target, output):
    if output.exists():
        raise FileExistsError("Preserving existing prediction output")
    evaluation = run_dir / "fresh_evaluation"
    manifest = json.loads((evaluation / "inference_manifest.json").read_text(encoding="utf-8"))
    frozen = json.loads((evaluation / "frozen_before_fresh_evaluation.json").read_text(encoding="utf-8"))
    model_file = evaluation / f"{target}_frozen_model.joblib"
    if hashlib.sha256(model_file.read_bytes()).hexdigest() != frozen['models_sha256'][model_file.name]:
        raise ValueError('Frozen research model hash mismatch')
    train_models.CONTEXT_FILE = run_dir / "data/processed/options_signal_eligible.parquet"
    bundle = joblib.load(evaluation / f"{target}_frozen_model.joblib")
    daily = pd.read_parquet(manifest["daily_spot_input"]).rename(columns={"close": "nifty_spot"})
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    features = corrected_features(pd.read_parquet(input_file), calculate_daily_hv(make_daily_spot(daily)))
    profile = frozen.get("market_feature_design")
    if profile is not None:
        market = market_table(pd.read_parquet(profile["spot_input"]), pd.read_parquet(profile["vix_input"]))
        features = augment(features, market)
    probability = bundle["model"].predict_proba(features[bundle["feature_columns"]])[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError("Non-finite predicted probabilities")
    result = features[["timestamp", "instrument_key"]].copy()
    result["target_definition"] = target
    result["probability"] = probability
    result["predicted_class_at_0_50"] = (probability >= .5).astype(int)
    result["session_horizon_eligible"] = session_horizon_eligible(features.timestamp).to_numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--registry", type=Path, help="Verify the matching canonical artifact registry before inference")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--target", choices=["endpoint", "first_passage", "sustained_two_bars"], default="endpoint")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.registry:
        from research_registry import verify
        args.run_dir = verify(args.registry)
    result = predict(args.run_dir, args.input, args.target, args.output)
    print(f"Saved {len(result):,} predictions to {args.output}")
