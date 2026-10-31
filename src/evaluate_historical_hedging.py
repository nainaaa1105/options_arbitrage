"""Secondary frozen-policy friction, coverage and classifier diagnostics."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from deep_hedging.historical_evaluate import evaluate_historical, paired_session_comparisons
from predict_historical_hedge import load_frozen_inputs


def evaluate(run_dir, stage, output):
    output.mkdir(parents=True, exist_ok=False)
    e, p, dates, c, models = load_frozen_inputs(run_dir)
    stress = replace(c, option_half_spread=2*c.option_half_spread, option_slippage=2*c.option_slippage,
                     future_half_spread=2*c.future_half_spread, future_slippage=2*c.future_slippage)
    results, report = evaluate_historical(e, p, dates, stress, models, output/"double_execution_friction")
    paired_session_comparisons(results, dates).to_csv(output/"double_execution_friction/paired_comparisons.csv", index=False)
    # Classifier diagnostic rows require observed labels. Prediction generation
    # did not require those labels; their absence is reported, not called failure.
    import train_models
    keys = ["instrument_key", "timestamp"]
    targets = pd.concat([pd.read_parquet(stage/"data/processed/ml_dataset.parquet", columns=keys+[train_models.TARGET]),
                         pd.read_parquet(stage/"data/processed/fresh_ml_dataset.parquet", columns=keys+[train_models.TARGET])], ignore_index=True)
    predictions = pd.read_parquet(run_dir/"walk_forward_predictions.parquet")
    known = predictions.merge(targets, on=keys, how="left", validate="one_to_one")
    v, t = pd.Timestamp("2026-07-01", tz="Asia/Kolkata"), pd.Timestamp("2026-09-16", tz="Asia/Kolkata")
    known["split"] = np.select([known.timestamp.lt(v), known.timestamp.lt(t)], ["train", "validation"], default="test")
    diagnostics = []
    for split, frame in known.groupby("split"):
        scored = frame.loc[frame[train_models.TARGET].notna()]
        metrics = train_models._classification_metrics(scored[train_models.TARGET].astype(int), scored.probability.to_numpy(), .5) if len(scored) else {}
        diagnostics.append({"split": split, "available_predictions": len(frame), "observed_labels_for_diagnostics": len(scored),
                            "scope": "Distinct eligible hedge observation queries; not the full original classification test set", **metrics})
    pd.DataFrame(diagnostics).to_csv(output/"classifier_query_diagnostics.csv", index=False)
    original = pd.read_parquet(run_dir/"test/episode_results.parquet")
    original.assign(month=pd.to_datetime(original.entry_time).dt.strftime("%Y-%m")).groupby(["strategy", "month"]).agg(
        episodes=("episode_id", "size"), net_pnl=("net_pnl", "sum"), fees=("fees", "sum"), impact=("execution_impact", "sum")
    ).to_csv(output/"monthly_pnl.csv")
    all_paths = pd.read_parquet(run_dir/"episode_paths.parquet")
    universe = pd.read_parquet(run_dir/"episode_universe.parquet")
    all_paths = all_paths.merge(universe[["episode_id", "split"]], on="episode_id")
    rows = []
    for split, frame in all_paths.loc[all_paths.step.lt(c.horizon_minutes//c.bar_minutes)].groupby("split"):
        rows.append({"split": split, "decision_rows": len(frame), "ml_available_rows": int(frame.ml_available.sum()),
                     "ml_coverage": float(frame.ml_available.mean()), "invalid_pricing_hold_rows": int((~frame.pricing_state_valid).sum()),
                     "iv_outside_nominal_simulation_range": float((frame.observed_iv.lt(.12) | frame.observed_iv.gt(.35)).mean()),
                     "maturity_outside_nominal_simulation_range": float(((frame.remaining_time*365).lt(1) | (frame.remaining_time*365).gt(7)).mean())})
    pd.DataFrame(rows).to_csv(output/"coverage_and_transfer_audit.csv", index=False)
    (output/"scope.json").write_text(json.dumps({"policies_refit": False, "scope": "Secondary diagnostics of frozen choices",
                                                "fees": "Original per-date fee schedule; execution half-spread and slippage doubled only",
                                                "classifier_diagnostics_are_not_hedge_accuracy": True}, indent=2))
    print(report[["strategy", "net_pnl", "loss_cvar_95"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parents[1]/"data/research/expanded_20261030")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.run_dir, args.stage, args.output)
