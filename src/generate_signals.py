"""
generate_signals.py
===================
Generates out-of-sample options arbitrage trading signals using the trained production XGBoost model.

Model:  data/processed/models/best_model_xgboost.joblib
Input:  data/processed/ml_splits/test.parquet
Output: data/processed/trading_signals.parquet

Fields Produced:
- timestamp
- strike
- option_type
- expiry
- market_price
- theoretical_price
- price_deviation
- predicted_probability
- predicted_class
- confidence (abs(probability - 0.5))
- confidence_margin
- signal_action (CONVERGENCE_CANDIDATE, NO_TRADE)
- is_high_confidence
- trading_symbol
- instrument_key
"""

import argparse
import json
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Import feature builder from train_models
from train_models import build_features
from research_targets import session_horizon_eligible


# ============================================================
# PATHS & CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_FILE   = PROJECT_ROOT / "data" / "processed" / "ml_splits" / "test.parquet"
DEFAULT_MODEL_FILE  = PROJECT_ROOT / "data" / "processed" / "models" / "best_model.joblib"
DEFAULT_FEAT_FILE   = PROJECT_ROOT / "data" / "processed" / "models" / "best_model_feature_columns.joblib"
DEFAULT_MED_FILE    = PROJECT_ROOT / "data" / "processed" / "models" / "best_model_imputer_medians.joblib"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "trading_signals.parquet"
DEFAULT_CONF_MARGIN = None


def generate_trading_signals(
    test_path: Path = DEFAULT_TEST_FILE,
    model_path: Path = DEFAULT_MODEL_FILE,
    feat_cols_path: Path = DEFAULT_FEAT_FILE,
    medians_path: Path = DEFAULT_MED_FILE,
    output_path: Path = DEFAULT_OUTPUT_FILE,
    confidence_margin: float | None = DEFAULT_CONF_MARGIN
) -> pd.DataFrame:
    """
    Generates predictions and trading signals on the test dataset using the pre-trained model.
    """
    print("=" * 80)
    print("GENERATING OPTIONS ARBITRAGE TRADING SIGNALS")
    print("=" * 80)
    print(f"Test dataset:         {test_path}")
    print(f"Production model:     {model_path}")
    config_path = model_path.parent / "best_model_config.json"
    config = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as stream:
            config = json.load(stream)
    if confidence_margin is None:
        if config_path.exists():
            with open(config_path, encoding="utf-8") as stream:
                probability_threshold = float(json.load(stream)["threshold"])
        else:
            probability_threshold = 0.5
        confidence_margin = probability_threshold - 0.5
    else:
        probability_threshold = 0.5 + confidence_margin
    print(f"Validation-selected probability threshold: {probability_threshold:.3f}")

    # 1. Load test dataset
    if not test_path.exists():
        raise FileNotFoundError(f"Test dataset not found at {test_path}")
    test_raw = pd.read_parquet(test_path)
    print(f"Loaded test samples:  {len(test_raw):,} rows")

    # Preserve original ordering
    test_raw = test_raw.copy()
    test_raw["_original_order"] = np.arange(len(test_raw))

    # 2. Load model artifacts
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at {model_path}")
    model = joblib.load(model_path)
    feature_columns = joblib.load(feat_cols_path)
    print(f"Model loaded:         {type(model).__name__}")
    print(f"Feature columns:      {len(feature_columns)} features")

    # 3. Build features
    print("\nEngineering features on test split...")
    processed_dict = build_features(
        {"test": test_raw}, feature_schema_version=config.get("feature_schema_version", 1)
    )
    test_processed = processed_dict["test"]

    # Restore exact original row order
    test_processed = test_processed.sort_values("_original_order").reset_index(drop=True)

    # 4. Prepare feature matrix with exact production imputation
    X_test_df = test_processed[feature_columns]
    X_test_mat = X_test_df.values

    # 5. Model Inference
    print("Predicting convergence probabilities...")
    pred_probs = model.predict_proba(X_test_mat)[:, 1]

    # 6. Compute Confidence and Signal Decisions
    confidence = np.abs(pred_probs - 0.50)
    pred_classes = np.where(pred_probs >= probability_threshold, 1, 0)

    hi_thr = probability_threshold
    horizon_eligible = session_horizon_eligible(test_processed["timestamp"], horizon_minutes=30)
    active = (pred_probs >= hi_thr) & horizon_eligible.to_numpy()
    signal_actions = np.where(active, "CONVERGENCE_CANDIDATE", "NO_TRADE")

    # 7. Construct Trading Signals DataFrame
    signals_df = pd.DataFrame({
        "timestamp": test_processed["timestamp"],
        "strike": test_processed["strike"].astype(float),
        "option_type": test_processed["option_type"].astype(str),
        "expiry": test_processed["expiry"],
        "market_price": test_processed["market_price"].astype(float),
        "theoretical_price": test_processed["reference_theoretical_price"].astype(float),
        "price_deviation": test_processed["reference_price_deviation"].astype(float),
        "predicted_probability": pred_probs.astype(float),
        "predicted_class": pred_classes.astype(int),
        "predicted_class_at_0_50": (pred_probs >= .50).astype(int),
        "horizon_eligible": horizon_eligible.to_numpy(),
        "confidence": confidence.astype(float),
        "confidence_margin": float(confidence_margin),
        "probability_threshold": probability_threshold,
        "signal_action": signal_actions,
        "is_high_confidence": active.astype(bool),
        "instrument_key": test_processed["instrument_key"],
        "trading_symbol": test_processed["trading_symbol"],
    })

    # Include actual target if present for evaluation
    if "target_converged_30m" in test_processed.columns:
        signals_df["actual_target"] = test_processed["target_converged_30m"].astype(int)

    # 8. Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signals_df.to_parquet(output_path, index=False)
    print(f"\nTrading signals saved to: {output_path}")

    # 9. Summary & Metrics
    n_total = len(signals_df)
    n_high_conf = int(signals_df["is_high_confidence"].sum())
    n_buy = int((signals_df["signal_action"] == "CONVERGENCE_CANDIDATE").sum())
    n_no_trade = int((signals_df["signal_action"] == "NO_TRADE").sum())
    n_abstain = 0

    print("\n" + "=" * 80)
    print("TRADING SIGNALS SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Options:      {n_total:,}")
    print(f"High-Confidence Signals:      {n_high_conf:,} ({(n_high_conf / n_total) * 100:.1f}% coverage)")
    print(f"  - CONVERGENCE_CANDIDATE (P>={hi_thr:.3f}): {n_buy:,}")
    print(f"  - NO_TRADE:                                  {n_no_trade:,}")
    print(f"Mean Predicted Probability:   {pred_probs.mean():.4f}")
    print(f"Mean Confidence Score:        {confidence.mean():.4f}")

    if "actual_target" in signals_df.columns:
        from sklearn.metrics import accuracy_score
        active_mask = signals_df["is_high_confidence"]
        if active_mask.sum() > 0:
            active_acc = accuracy_score(
                signals_df.loc[active_mask, "actual_target"],
                signals_df.loc[active_mask, "predicted_class"]
            )
            print(f"System Accuracy on Active Signals: {active_acc * 100:.2f}%")
        if n_buy > 0:
            candidate_precision = accuracy_score(
                signals_df.loc[signals_df["signal_action"] == "CONVERGENCE_CANDIDATE", "actual_target"],
                np.ones(n_buy)
            )
            print(f"Convergence precision among candidates: {candidate_precision * 100:.2f}%")

    print("=" * 80)
    return signals_df


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate trading signals using the production options arbitrage model."
    )
    parser.add_argument('--allow-legacy-artifacts', action='store_true', help='Explicit opt-in for preserved data/processed model/context.')
    parser.add_argument('--context-file', type=Path, help='Matching causal context for a custom legacy-format model.')
    parser.add_argument('--feature-columns-file', type=Path)
    parser.add_argument('--medians-file', type=Path)
    parser.add_argument("--confidence-margin", type=float, default=DEFAULT_CONF_MARGIN,
                        help="Optional compatibility override; default uses threshold selected on validation.")
    parser.add_argument(
        "--test-file",
        type=str,
        default=str(DEFAULT_TEST_FILE),
        help="Path to input test parquet file."
    )
    parser.add_argument(
        "--model-file",
        type=str,
        default=str(DEFAULT_MODEL_FILE),
        help="Path to production trained model .joblib file."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DEFAULT_OUTPUT_FILE),
        help="Path to output trading signals parquet file."
    )
    args = parser.parse_args()
    if Path(args.model_file).resolve() == DEFAULT_MODEL_FILE.resolve() and not args.allow_legacy_artifacts:
        parser.error('Legacy default model is not the corrected research model. Use predict_research_model.py --registry REGISTRY --input INPUT --output OUTPUT, or explicitly opt into --allow-legacy-artifacts.')
    if Path(args.model_file).resolve() != DEFAULT_MODEL_FILE.resolve() and args.context_file is None:
        parser.error('A custom model requires its matching --context-file; frozen research bundles use predict_research_model.py.')
    if args.context_file:
        import train_models
        train_models.CONTEXT_FILE = args.context_file

    generate_trading_signals(
        test_path=Path(args.test_file),
        model_path=Path(args.model_file),
        feat_cols_path=args.feature_columns_file or Path(args.model_file).parent/DEFAULT_FEAT_FILE.name,
        medians_path=args.medians_file or Path(args.model_file).parent/DEFAULT_MED_FILE.name,
        output_path=Path(args.output_file),
        confidence_margin=args.confidence_margin
    )


if __name__ == "__main__":
    main()
