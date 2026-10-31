"""Replay a frozen Part 2 policy on saved simulated paths, without fitting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from deep_hedging.config import HedgeConfig
from deep_hedging.policy import load_policy, rollout_policy
from deep_hedging.evaluate_simulation import metrics


def run(checkpoint, paths, output):
    output.mkdir(parents=True, exist_ok=False)
    model, saved = load_policy(checkpoint)
    config = HedgeConfig(**saved["execution_config"])
    torch.set_num_threads(2)
    with np.load(paths, allow_pickle=False) as archive:
        batch = {name: archive[name] for name in archive.files}
    with torch.no_grad():
        result = rollout_policy(batch, config, model)[0]
    np.savez_compressed(output / "execution.npz", **result)
    pd.DataFrame({name: value for name, value in result.items() if value.ndim == 1}).to_parquet(output / "path_results.parquet", index=False)
    report = {"checkpoint": str(checkpoint.resolve()), "paths": str(paths.resolve()),
              "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "execution_config": config.to_dict(), "metrics": metrics(result), "training_performed": False}
    (output / "replay.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["metrics"], indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--paths", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.paths, args.output)
