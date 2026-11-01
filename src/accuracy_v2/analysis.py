"""Bottleneck analysis from saved out-of-fold predictions (development history only; the post-cutoff rows are not read)."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .experiment import TARGET, load_tables


def run(stage, daily_file, regime_dir, run_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    history, _, _, _ = load_tables(stage, daily_file, regime_dir)
    oof = pd.read_parquet(run_dir / "oof_predictions.parquet")
    frozen = json.loads((run_dir / "frozen_before_fresh.json").read_text())
    chosen = frozen["chosen"]
    h = history.assign(fold=oof.fold.to_numpy(), baseline=oof["baseline_frozen"].to_numpy(), improved=oof[chosen].to_numpy())
    h = h[(h.fold >= 3) & h.improved.notna() & h.baseline.notna()].copy()      # blocks where the chosen model has out-of-fold scores
    h[TARGET].astype(int)
    out = {"rows_evaluated": len(h), "chosen": chosen}
    # 1. Label proximity to the decision boundary: how much of the target is decided by sub-tick or sub-spread noise?
    cur = h.absolute_price_deviation.abs()
    fut = cur * (1 - h.deviation_reduction_30m)
    margin = (fut - .5 * cur).abs()
    out["label_boundary"] = {f"within_{w}_points": float((margin < w).mean()) for w in (.05, .10, .25, .50)}
    out["label_boundary"]["note"] = "Share of labels whose future deviation lies within w index-option points of the 50%-reduction cutoff (tick = 0.05)"
    # 2. Accuracy and AUC by deviation size, staleness, liquidity
    def slice_stats(name, key):
        rows = []
        for label, g in h.groupby(key, observed=True):
            yy = g[TARGET].astype(int)
            rows.append({"slice": name, "bucket": str(label), "rows": len(g), "positive_rate": float(yy.mean()),
                         "majority_accuracy": float(max(yy.mean(), 1 - yy.mean())),
                         "baseline_accuracy": float(((g.baseline >= .5) == yy).mean()), "improved_accuracy": float(((g.improved >= .5) == yy).mean()),
                         "baseline_auc": float(roc_auc_score(yy, g.baseline)) if yy.nunique() == 2 else np.nan,
                         "improved_auc": float(roc_auc_score(yy, g.improved)) if yy.nunique() == 2 else np.nan})
        return rows
    table = []
    table += slice_stats("abs_deviation_points", pd.cut(cur, [0, 1.5, 2, 3, 5, 10, 1e9]))
    table += slice_stats("stale_run", pd.cut(h.stale_run, [-1, 0, 1, 3, 100]))
    table += slice_stats("volume_last3_log", pd.qcut(h.vol_sum3, 4, duplicates="drop"))
    table += slice_stats("days_to_expiry", pd.cut(h.days_to_expiry, [-.1, 1, 3, 100]))
    table += slice_stats("time_of_day", pd.cut(h.intraday_minute, [-1, 45, 120, 240, 400]))
    pd.DataFrame(table).to_csv(output / "accuracy_by_slice.csv", index=False)
    # 3. Learning-curve evidence from the fold structure: AUC per validation block vs rows available for training.
    folds = pd.read_csv(run_dir / "cv_folds.csv")
    out["fold_auc_vs_training_rows"] = folds[folds.candidate.isin(["baseline_frozen", chosen])][["candidate", "fold", "train_rows", "auc"]].dropna().to_dict("records")
    (output / "bottlenecks.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out["label_boundary"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=Path("data/research/expanded_20261030"))
    parser.add_argument("--daily-spot", type=Path, default=Path("data/raw/upstox/nifty_daily_to_20261029.parquet"))
    parser.add_argument("--regime-dir", type=Path, default=Path("data/research/expanded_20261030/regime_study"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.daily_spot, args.regime_dir, args.run_dir, args.output)
