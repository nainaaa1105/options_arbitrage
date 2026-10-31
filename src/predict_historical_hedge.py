"""Replay all frozen Part 3 hedgers without fitting or choosing any parameters."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from deep_hedging.config import HedgeConfig
from deep_hedging.historical_data import sha256
from deep_hedging.historical_evaluate import evaluate_historical
from deep_hedging.historical_policy import load_historical_policy
from deep_hedging.policy import load_policy


def load_frozen_inputs(run_dir):
    torch.set_num_threads(2)
    frozen = json.loads((run_dir/"frozen_before_historical_test.json").read_text())
    plan = json.loads((run_dir/"experiment_plan.json").read_text())
    policies = {}
    for name, expected in frozen["historical_state_sha256"].items():
        if sha256(run_dir/name) != expected:
            raise ValueError(f"Historical input changed: {name}")
    for name, expected in frozen["checkpoint_sha256"].items():
        path = run_dir/name
        if sha256(path) != expected:
            raise ValueError("Historical checkpoint changed")
        policies[Path(name).parts[0]], _ = load_historical_policy(path)
    for name, expected in frozen["parent_checkpoint_sha256"].items():
        path = Path(name)
        if sha256(path) != expected:
            raise ValueError("Simulation checkpoint changed")
        policies["frozen_"+path.parent.name], _ = load_policy(path)
    episodes = pd.read_parquet(run_dir/"episode_universe.parquet")
    episodes = episodes.loc[episodes.split.eq("test")]
    paths = pd.read_parquet(run_dir/"episode_paths.parquet")
    paths = paths.loc[paths.episode_id.isin(episodes.episode_id)]
    dates = pd.to_datetime(pd.read_csv(run_dir/"test/daily_equity.csv").trade_date).dt.date.unique()
    return episodes, paths, dates, HedgeConfig(**plan["config"]), policies


def replay(run_dir, output):
    episodes, paths, dates, config, policies = load_frozen_inputs(run_dir)
    results, report = evaluate_historical(episodes, paths, dates, config, policies, output)
    original = pd.read_parquet(run_dir/"test/episode_results.parquet")
    keys = ["episode_id", "strategy"]
    a = results.set_index(keys).sort_index().net_pnl
    b = original.set_index(keys).sort_index().net_pnl
    pd.testing.assert_series_equal(a, b, check_exact=True)
    verification = {"source_run": str(run_dir.resolve()), "all_frozen_test_pnl_reproduced_exactly": True,
                    "strategies": len(policies)+4, "episodes_per_strategy": int(episodes.shape[0]), "training_performed": False}
    (output/"replay_verification.json").write_text(json.dumps(verification, indent=2))
    print(json.dumps(verification, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay(args.run_dir, args.output)
