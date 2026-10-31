"""Combine expiry-isolated pricing components and reserve later dates."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = ["master_options", "options_features", "options_volatility_features", "options_reference_iv",
         "options_deviation_features", "options_signal_eligible", "options_parity_features",
         "options_convergence_labels", "ml_dataset"]


def run(components, output, daily, cutoff="2026-09-15", trials=20, frozen_model_study=None):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source, processed = output / "src", output / "data/processed"
    source.mkdir()
    processed.mkdir(parents=True)
    for path in (ROOT / "src").glob("*.py"):
        shutil.copy2(path, source / path.name)
    provenance = {"components": [str(c.resolve()) for c in components],
                  "development_end_date": cutoff, "fresh_start_date": "2026-09-16",
                  "active_artifacts_replaced": False, "inputs_sha256": {}, "rows": {}}
    for name in FILES:
        parts = []
        for component in components:
            file = component / "data/processed" / (name + ".parquet")
            provenance["inputs_sha256"][str(file)] = hashlib.sha256(file.read_bytes()).hexdigest()
            parts.append(pd.read_parquet(file))
        d = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "instrument_key"]).reset_index(drop=True)
        if d.duplicated(["instrument_key", "timestamp"]).any():
            raise ValueError("Components overlap; refusing to combine")
        provenance["rows"][name] = len(d)
        if name == "ml_dataset":
            historical = pd.to_datetime(d.timestamp).dt.date <= pd.Timestamp(cutoff).date()
            d.loc[~historical].to_parquet(processed / "fresh_ml_dataset.parquet", index=False)
            d = d.loc[historical].copy()
            provenance["development_ml_rows"] = len(d)
        d.to_parquet(processed / (name + ".parquet"), index=False)
        print(f"Merged {name}: {len(d):,} rows", flush=True)
        del d, parts
    (output / "merge_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    commands = [("prepare_ml_split.py", []),
                ("data_remediation_study.py", ["--daily-spot", str(daily.resolve()), "--output", str(output / "target_study")]),
                ("robust_model_research.py", ["--trials", str(trials), "--threads", "4", "--output", str(output / "model_study")])]
    if frozen_model_study is not None:
        commands[-1] = ("refit_frozen_design.py", ["--source", str(frozen_model_study.resolve()), "--output", str(output / "model_study")])
    for script, arguments in commands:
        if script == "refit_frozen_design.py":
            deadline = time.monotonic() + 1800
            while not (frozen_model_study / "full_training_candidate.joblib").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("Frozen base design has not finished; refit was not started")
                print("Waiting for the historical CV model design to finish", flush=True)
                time.sleep(20)
        print(f"START {script}", flush=True)
        with (output / (script + ".log")).open("w", encoding="utf-8") as log:
            subprocess.run([sys.executable, str(source / script), *arguments], cwd=output, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"END {script}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--components", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--daily-spot", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--frozen-model-study", type=Path, help="Reuse a historical CV-selected design rather than repeating hyperparameter search")
    args = parser.parse_args()
    run(args.components, args.output, args.daily_spot, trials=args.trials, frozen_model_study=args.frozen_model_study)
