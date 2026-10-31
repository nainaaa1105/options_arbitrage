"""Display persisted model, threshold, backtest, and risk reports."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = PROCESSED / "models"


def _show(title, path):
    print(f"\n{title}")
    print("=" * len(title))
    if not path.exists():
        print(f"Not available: {path}")
        return
    if path.suffix.lower() == ".csv":
        print(pd.read_csv(path).to_string(index=False))
    else:
        print(f"Available: {path}")


def print_metrics():
    studies = sorted((PROCESSED / "research").glob("audit_*"))
    if studies and (studies[-1] / "audit.json").exists():
        audit = json.loads((studies[-1] / "audit.json").read_text(encoding="utf-8"))
        print("Historical data status:", audit.get("financial_data_status", "unverified"))
        _show("Latest audited development validation metrics", studies[-1] / "validation_metrics.csv")
        _show("Predeclared selective validation bands and coverage", studies[-1] / "validation_predeclared_selective_bands.csv")
    _show("Validation model comparison", MODELS / "model_validation_comparison.csv")
    _show("Held-out test metrics for selected model (diagnostic)", MODELS / "best_model_test_metrics.csv")
    _show("Test metrics by fixed and validation-selected cutoff", MODELS / "best_model_test_operating_points.csv")
    _show("Validation-only probability cutoff diagnostic", MODELS / "best_model_confidence_accuracy.csv")
    _show("Test probability calibration diagnostic", MODELS / "best_model_test_calibration.csv")
    _show("Walk-forward model comparison", MODELS / "walk_forward_model_summary.csv")
    _show("Walk-forward selected candidate validation metrics", MODELS / "walk_forward_selected_validation_metrics.csv")
    _show("Backtest summary", PROCESSED / "backtest_summary.csv")
    _show("Daily risk metrics", PROCESSED / "risk_metrics.csv")


if __name__ == "__main__":
    print_metrics()
