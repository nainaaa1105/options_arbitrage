"""Inference wrapper for the validation-selected 30-minute model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from research_targets import session_horizon_eligible


class HighAccuracyArbitrageClassifier:
    """Load the selected model; no fixed accuracy or profitability is promised."""

    def __init__(self, models_dir=None, confidence_margin=None):
        self.models_dir = Path(models_dir) if models_dir else (
            Path(__file__).resolve().parent.parent / "data" / "processed" / "models"
        )
        self.feature_columns = joblib.load(
            self.models_dir / "best_model_feature_columns.joblib"
        )
        selected = self.models_dir / "best_model.joblib"
        self.legacy = not selected.exists()
        self.feature_schema_version = 1
        if self.legacy:
            self.imputer_medians = joblib.load(
                self.models_dir / "best_model_imputer_medians.joblib"
            )
            self.xgb_model = joblib.load(self.models_dir / "best_model_xgboost.joblib")
            self.lgb_model = joblib.load(self.models_dir / "best_model_lightgbm.joblib")
            threshold = 0.73
        else:
            self.model = joblib.load(selected)
            with open(self.models_dir / "best_model_config.json", encoding="utf-8") as stream:
                config = json.load(stream)
                threshold = float(config["threshold"])
                self.feature_schema_version = config.get("feature_schema_version", 1)
        if confidence_margin is not None:
            threshold = 0.5 + float(confidence_margin)
        self.threshold = float(threshold)
        self.confidence_margin = self.threshold - 0.5

    def predict_proba(self, X_df):
        supplied_version = X_df.attrs.get("feature_schema_version")
        if supplied_version is not None and supplied_version != self.feature_schema_version:
            raise ValueError("Feature schema differs from the fitted model; rebuild features using the model's schema version")
        X = X_df[self.feature_columns]
        if not self.legacy:
            return self.model.predict_proba(X)[:, 1]
        matrix = X.fillna(self.imputer_medians).values
        return (
            0.55 * self.xgb_model.predict_proba(matrix)[:, 1]
            + 0.45 * self.lgb_model.predict_proba(matrix)[:, 1]
        )

    def predict(self, X_df, return_abstain=False):
        probabilities = self.predict_proba(X_df)
        eligible = np.ones(len(X_df), dtype=bool)
        if "timestamp" in X_df:
            eligible = session_horizon_eligible(X_df["timestamp"], horizon_minutes=30).to_numpy()
        active_mask = (probabilities >= self.threshold) & eligible
        if return_abstain:
            # Below cutoff means no convergence trade; it is not a second signal.
            return np.where(active_mask, 1, -1)
        return np.ones(int(active_mask.sum()), dtype=int), active_mask

    def evaluate(self, X_df, y_true):
        probabilities = self.predict_proba(X_df)
        predictions = (probabilities >= self.threshold).astype(int)
        accuracy = accuracy_score(y_true, predictions)
        matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
        print(f"Threshold: {self.threshold:.4f}")
        print(f"Overall classification accuracy: {accuracy:.4f}")
        print(classification_report(y_true, predictions, zero_division=0))
        print(f"Confusion matrix (TN,FP/FN,TP):\n{matrix}")
        return {"accuracy": accuracy, "threshold": self.threshold, "confusion_matrix": matrix}


if __name__ == "__main__":
    from train_models import SPLIT_DIR, TARGET, build_features

    test_raw = pd.read_parquet(SPLIT_DIR / "test.parquet")
    classifier = HighAccuracyArbitrageClassifier()
    test_df = build_features(
        {"test": test_raw}, feature_schema_version=classifier.feature_schema_version
    )["test"]
    classifier.evaluate(test_df, test_df[TARGET])
