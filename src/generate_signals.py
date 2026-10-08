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
- signal_action (CONVERGENCE_BUY, NO_TRADE, ABSTAIN_BORDERLINE)
- is_high_confidence
- trading_symbol
- instrument_key
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

# Import feature builder from train_models
from train_models import build_features


# ============================================================
# PATHS & CONSTANTS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TEST_FILE   = PROJECT_ROOT / "data" / "processed" / "ml_splits" / "test.parquet"
DEFAULT_MODEL_FILE  = PROJECT_ROOT / "data" / "processed" / "models" / "best_model_xgboost.joblib"
DEFAULT_FEAT_FILE   = PROJECT_ROOT / "data" / "processed" / "models" / "best_model_feature_columns.joblib"
DEFAULT_MED_FILE    = PROJECT_ROOT / "data" / "processed" / "models" / "best_model_imputer_medians.joblib"
DEFAULT_OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "trading_signals.parquet"
DEFAULT_CONF_MARGIN = 0.20


def generate_trading_signals(
    test_path: Path = DEFAULT_TEST_FILE,
    model_path: Path = DEFAULT_MODEL_FILE,
    feat_cols_path: Path = DEFAULT_FEAT_FILE,
    medians_path: Path = DEFAULT_MED_FILE,
    output_path: Path = DEFAULT_OUTPUT_FILE,
    confidence_margin: float = DEFAULT_CONF_MARGIN
) -> pd.DataFrame:
    """
    Generates predictions and trading signals on the test dataset using the pre-trained model.
    """
    print("=" * 80)
    print("GENERATING OPTIONS ARBITRAGE TRADING SIGNALS")
    print("=" * 80)
    print(f"Test dataset:         {test_path}")
    print(f"Production model:     {model_path}")
    print(f"Confidence threshold: +/- {confidence_margin:.2f}")

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
    imputer_medians = joblib.load(medians_path)
    print(f"Model loaded:         {type(model).__name__}")
    print(f"Feature columns:      {len(feature_columns)} features")

    # 3. Build features
    print("\nEngineering features on test split...")
    processed_dict = build_features({"test": test_raw})
    test_processed = processed_dict["test"]

    # Restore exact original row order
    test_processed = test_processed.sort_values("_original_order").reset_index(drop=True)

    # 4. Prepare feature matrix with exact production imputation
    X_test_df = test_processed[feature_columns].fillna(imputer_medians)
    X_test_mat = X_test_df.values

    # 5. Model Inference
    print("Predicting convergence probabilities...")
    pred_probs = model.predict_proba(X_test_mat)[:, 1]

    # 6. Compute Confidence and Signal Decisions
    confidence = np.abs(pred_probs - 0.50)
    pred_classes = np.where(pred_probs >= 0.50, 1, 0)

    hi_thr = 0.50 + confidence_margin
    lo_thr = 0.50 - confidence_margin

    signal_actions = np.where(
        pred_probs >= hi_thr,
        "CONVERGENCE_BUY",
        np.where(pred_probs <= lo_thr, "NO_TRADE", "ABSTAIN_BORDERLINE")
    )

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
        "confidence": confidence.astype(float),
        "confidence_margin": float(confidence_margin),
        "signal_action": signal_actions,
        "is_high_confidence": (confidence >= confidence_margin).astype(bool),
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
    n_buy = int((signals_df["signal_action"] == "CONVERGENCE_BUY").sum())
    n_no_trade = int((signals_df["signal_action"] == "NO_TRADE").sum())
    n_abstain = int((signals_df["signal_action"] == "ABSTAIN_BORDERLINE").sum())

    print("\n" + "=" * 80)
    print("TRADING SIGNALS SUMMARY")
    print("=" * 80)
    print(f"Total Evaluated Options:      {n_total:,}")
    print(f"High-Confidence Signals:      {n_high_conf:,} ({(n_high_conf / n_total) * 100:.1f}% coverage)")
    print(f"  - CONVERGENCE_BUY (P>={hi_thr:.2f}): {n_buy:,}")
    print(f"  - NO_TRADE (P<={lo_thr:.2f}):        {n_no_trade:,}")
    print(f"  - ABSTAIN_BORDERLINE:       {n_abstain:,}")
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
            buy_acc = accuracy_score(
                signals_df.loc[signals_df["signal_action"] == "CONVERGENCE_BUY", "actual_target"],
                np.ones(n_buy)
            )
            print(f"Long Entry Win Rate (CONVERGENCE_BUY):   {buy_acc * 100:.2f}%")

    print("=" * 80)
    return signals_df


# ============================================================
# CLI INTERFACE
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate trading signals using the production options arbitrage model."
    )
    parser.add_argument(
        "--confidence-margin",
        type=float,
        default=DEFAULT_CONF_MARGIN,
        help=f"Confidence margin around 0.50 (default: {DEFAULT_CONF_MARGIN:.2f} => P<={0.50-DEFAULT_CONF_MARGIN:.2f} or P>={0.50+DEFAULT_CONF_MARGIN:.2f})."
    )
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

    generate_trading_signals(
        test_path=Path(args.test_file),
        model_path=Path(args.model_file),
        feat_cols_path=DEFAULT_FEAT_FILE,
        medians_path=DEFAULT_MED_FILE,
        output_path=Path(args.output_file),
        confidence_margin=args.confidence_margin
    )


if __name__ == "__main__":
    main()
