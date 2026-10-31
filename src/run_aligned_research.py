"""Snapshot source and rebuild/retrain separately from active production files."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(output, master, daily, trials=20, pipeline_only=False, resume=False):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=resume)
    source = output / "src"
    previous = None
    if resume:
        previous = json.loads((output / "run_provenance.json").read_text(encoding="utf-8"))
        if previous["master_sha256"] != hashlib.sha256(master.read_bytes()).hexdigest():
            raise ValueError("Resume input master differs from the original run")
        if previous["daily_sha256"] != hashlib.sha256(daily.read_bytes()).hexdigest():
            raise ValueError("Resume daily history differs from the original run")
        successful = {s["script"] for s in previous["steps"] if s["returncode"] == 0}
        for name in successful:
            if hashlib.sha256((ROOT / "src" / name).read_bytes()).hexdigest() != previous["source_sha256"][name]:
                raise ValueError(f"Completed script changed: {name}; use a new run")
        shutil.copytree(source, output / ("source_before_resume_" + datetime.now(timezone.utc).strftime("%H%M%S")))
    source.mkdir(exist_ok=resume)
    hashes = {}
    for path in (ROOT / "src").glob("*.py"):
        shutil.copy2(path, source / path.name)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    processed = output / "data/processed"
    processed.mkdir(parents=True, exist_ok=resume)
    if not resume:
        shutil.copy2(master, processed / "master_options.parquet")
    provenance = {"created_at": datetime.now(timezone.utc).isoformat(),
                  "source_sha256": hashes, "master_input": str(master.resolve()),
                  "master_sha256": hashlib.sha256(master.read_bytes()).hexdigest(),
                  "daily_input": str(daily.resolve()),
                  "daily_sha256": hashlib.sha256(daily.read_bytes()).hexdigest(),
                  "active_artifacts_replaced": False, "steps": []}
    if previous:
        provenance["steps"] = previous["steps"]
        provenance["previous_source_sha256"] = previous["source_sha256"]
        provenance["resumed_at"] = datetime.now(timezone.utc).isoformat()
    steps = [
        ("build_features.py", ["--vectorized"]),
        ("build_volatility_features.py", ["--daily-spot", str(daily.resolve())]),
        ("build_reference_iv.py", []), ("build_deviation_features.py", ["--vectorized"]),
        ("build_signal_eligibility.py", []), ("build_parity_features.py", []),
        ("build_convergence_labels.py", []), ("build_ml_dataset.py", []),
        ("prepare_ml_split.py", []),
        ("data_remediation_study.py", ["--daily-spot", str(daily.resolve()), "--output", str(output / "target_study")]),
        ("robust_model_research.py", ["--trials", str(trials), "--threads", "4", "--output", str(output / "model_study")]),
    ]
    if pipeline_only:
        steps = steps[:8]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    for name, arguments in steps:
        if resume and name in successful:
            continue
        started = time.monotonic()
        print(f"START {name}", flush=True)
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(source / name), *arguments], cwd=output,
                                    env=env, stdout=log, stderr=subprocess.STDOUT)
        provenance["steps"].append({"script": name, "arguments": arguments,
                                    "seconds": time.monotonic() - started, "returncode": result.returncode})
        (output / "run_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        print(f"END {name}: exit={result.returncode}, seconds={time.monotonic()-started:.1f}", flush=True)
        if result.returncode:
            raise RuntimeError(f"Step failed: inspect {output / (name + '.log')}")
    print(f"Completed isolated rebuild and model research: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--daily-spot", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--pipeline-only", action="store_true", help="Build labels and ML rows without splitting or fitting any models")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete steps with unchanged inputs; archive the prior source snapshot")
    args = parser.parse_args()
    run(args.output, args.master, args.daily_spot, args.trials, args.pipeline_only, args.resume)
