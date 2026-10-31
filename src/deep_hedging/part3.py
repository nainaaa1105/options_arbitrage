"""Walk-forward classifier inputs and historical neural/delta hedge comparison."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from .config import HedgeConfig
from .historical_data import attach_states, pack_paths, prepare_ml, sha256, split_episodes
from .historical_evaluate import evaluate_historical, paired_session_comparisons
from .historical_policy import load_historical_policy, rollout_historical
from .historical_train import HistoricalTraining, train_ablation
from .policy import load_policy


def run(output, stage, part1, part2, spec):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output/"source_snapshot"
    shutil.copytree(Path(__file__).parent, snapshot, ignore=shutil.ignore_patterns("__pycache__"))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads((part1/"manifest.json").read_text())
    config = HedgeConfig(**manifest["config"])
    if config.horizon_minutes != 30 or config.option_side != 1:
        raise ValueError("This protocol matches 30-minute long-option Part 1 and Part 2 artifacts")
    options_file = Path(manifest["inputs"]["options"])
    futures_file = Path(manifest["inputs"]["futures"])
    # Verify all upstream data rather than silently mixing an old episode
    # universe with subsequently modified option or futures prices.
    for path in (options_file, futures_file):
        expected = next(value for key, value in manifest["input_sha256"].items() if Path(key).resolve() == path.resolve())
        if sha256(path) != expected:
            raise ValueError(f"Part 1 input changed: {path.name}")
    episodes = split_episodes(pd.read_parquet(part1/"episode_universe.parquet"))
    raw_paths = pd.read_parquet(part1/"episode_paths.parquet")
    options = pd.read_parquet(options_file, columns=["instrument_key", "timestamp", "close", "implied_volatility", "time_to_expiry", "feature_available_at"])
    paths = attach_states(episodes, raw_paths, options)
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "training": asdict(spec), "config": config.to_dict(),
            "validation_start": "2026-07-01", "test_start": "2026-09-16", "test_end": str(episodes.entry_time.max()),
            "classifier": "Established endpoint architecture; clone and fit on completed past labels, 60-session burn-in, 20-session blocks; frozen at test start",
            "adaptation": "Frozen Heston policy plus 85-parameter residual; matched ML and masked-ML ablations",
            "selection": "Validation episode CVaR only, epoch-zero fallback allowed; no test retuning",
            "holdout_status": "Retrospective chronological evaluation: these market dates and simulation-policy results were previously inspected",
            "positions": "Same fixed alternating long CE/PE ATM episodes for all strategies; convergence is a hedge feature, not a buy/sell instruction",
            "stress": "Same frozen hedgers and test positions with twice the configured option/futures half-spread and slippage; historical fee schedule unchanged",
            "execution": "Completed observation -> same-boundary next-bar adverse open proxy; historical per-episode tax schedule; configured margin proxy",
            "inputs_sha256": {str(path): sha256(path) for path in (options_file, futures_file, part1/"episode_universe.parquet", part1/"episode_paths.parquet")}}
    (output/"experiment_plan.json").write_text(json.dumps(plan, indent=2))
    print("Preparing expanding-window ML predictions", flush=True)
    probabilities, fold_audit, ml_info = prepare_ml(stage, paths, output/"classifier_folds")
    probabilities.to_parquet(output/"walk_forward_predictions.parquet", index=False)
    paths = attach_states(episodes, raw_paths, options, probabilities)
    # Exclude only the predeclared 60-session model burn-in, identically for
    # every hedger. Later unavailable probabilities do not remove positions.
    start = pd.Timestamp(fold_audit[0]["prediction_start"], tz="Asia/Kolkata")
    episodes = episodes.loc[episodes.entry_time.ge(start) & episodes.split.ne("purged")].copy()
    paths = paths.loc[paths.episode_id.isin(episodes.episode_id)].copy()
    episodes.to_parquet(output/"episode_universe.parquet", index=False)
    paths.to_parquet(output/"episode_paths.parquet", index=False)
    coverage = paths.loc[paths.step.lt(config.horizon_minutes//config.bar_minutes)].merge(episodes[["episode_id", "split"]], on="episode_id")
    coverage.groupby("split").agg(decision_rows=("episode_id", "size"), ml_available=("ml_available", "sum"),
                                   valid_pricing_rows=("pricing_state_valid", "sum")).to_csv(output/"input_coverage.csv")
    models = {}
    parent_hashes = {}
    for name in ("gbm", "heston"):
        selection = json.loads((part2/name/"training_selection.json").read_text())["selected"]
        file = part2/name/selection["checkpoint"]
        expected = json.loads((part2/"frozen_before_test.json").read_text())["checkpoint_sha256"][str(file.relative_to(part2))]
        if sha256(file) != expected:
            raise ValueError("Frozen Part 2 checkpoint changed")
        model, _ = load_policy(file)
        models["frozen_"+name] = model
        parent_hashes[str(file)] = sha256(file)
    training = pack_paths(episodes.loc[episodes.split.eq("train")], paths, config)
    validation = pack_paths(episodes.loc[episodes.split.eq("validation")], paths, config)
    selections = {}
    for use_ml, name in ((False, "adapted_no_ml"), (True, "adapted_ml")):
        selections[name] = train_ablation(models["frozen_heston"], training, validation, config, spec, output/name, use_ml)
        model, _ = load_historical_policy(output/name/selections[name]["selected"]["checkpoint"])
        models[name] = model
    frozen_files = [output/name/selection["selected"]["checkpoint"] for name, selection in selections.items()]
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selections": selections,
              "checkpoint_sha256": {str(file.relative_to(output)): sha256(file) for file in frozen_files},
              "parent_checkpoint_sha256": parent_hashes, "ml_design": ml_info,
              "historical_state_sha256": {name: sha256(output/name) for name in ("episode_universe.parquet", "episode_paths.parquet", "walk_forward_predictions.parquet")},
              "classifier_folds": fold_audit, "test_used_for_selection": False,
              "test_was_previously_inspected_elsewhere": True}
    (output/"frozen_before_historical_test.json").write_text(json.dumps(frozen, indent=2))
    test = episodes.loc[episodes.split.eq("test")]
    test_paths = paths.loc[paths.episode_id.isin(test.episode_id)]
    dates = sorted(set(pd.to_datetime(pd.read_parquet(futures_file, columns=["timestamp"]).timestamp).dt.date)
                   | set(pd.to_datetime(options.timestamp).dt.date))
    dates = [day for day in dates if day >= pd.Timestamp(plan["test_start"]).date()]
    print(f"Choices frozen; evaluating {len(test)} common test episodes over {len(dates)} calendar sessions", flush=True)
    results, report = evaluate_historical(test, test_paths, dates, config, models, output/"test")
    paired = paired_session_comparisons(results, dates)
    paired.to_csv(output/"test/paired_comparisons.csv", index=False)
    stress = replace(config, option_half_spread=2*config.option_half_spread, option_slippage=2*config.option_slippage,
                     future_half_spread=2*config.future_half_spread, future_slippage=2*config.future_slippage)
    _, stress_report = evaluate_historical(test, test_paths, dates, stress, models, output/"double_execution_friction")
    # Daily P&L already follows separate sequential accounts. These calendar
    # aggregates do not reset accounts or invent annualized intratrade returns.
    monthly = results.assign(month=pd.to_datetime(results.entry_time).dt.strftime("%Y-%m"))
    monthly.groupby(["strategy", "month"]).agg(episodes=("episode_id", "size"), net_pnl=("net_pnl", "sum"),
                                               fees=("fees", "sum"), impact=("execution_impact", "sum")).to_csv(output/"test/monthly_pnl.csv")
    # Separately audit fresh-process-compatible selected checkpoint inference.
    replay = {}
    groups = {key: group for key, group in paths.groupby("episode_id")}
    audit_episodes = test.head(16)
    batch = pack_paths(audit_episodes, paths, config)
    from .engine import run_episode
    from .historical_policy import HistoricalScalarPolicy
    for name, model in models.items():
        if name in selections:
            model, _ = load_historical_policy(output/name/selections[name]["selected"]["checkpoint"])
        with torch.no_grad():
            vector = rollout_historical(batch, config, model)[0]["net_pnl"]
        scalar = []
        for _, ep in audit_episodes.sort_values("entry_time").iterrows():
            value, _, _ = run_episode(ep, groups[ep.episode_id], name, config.initial_capital, config,
                                      policy=HistoricalScalarPolicy(model, config.initial_capital))
            scalar.append(value["net_pnl"])
        replay[name] = float(np.max(abs(vector-scalar)))
        if replay[name] > 1e-6:
            raise AssertionError(f"Historical batched/scalar policy mismatch: {name}")
    (output/"replay_checks.json").write_text(json.dumps(replay, indent=2))
    write_results(output, report, paired, plan)
    print(report[["strategy", "episodes", "net_pnl", "loss_cvar_95", "daily_sharpe_annualized", "max_daily_drawdown_fraction"]].to_string(index=False), flush=True)
    return report


def write_results(output, report, paired, plan):
    lines = ["# Part 3 historical ML-assisted hedge evaluation", "", "Retrospective chronological research, not a pristine holdout or live performance.",
             "Identical long-option positions, sequential separate accounts, configured OHLC/margin proxies.", "",
             "| Strategy | Episodes | Net P&L | Fees | Loss CVaR 95% | Daily Sharpe | Daily max drawdown |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in report.iterrows():
        lines.append(f"| {row.strategy} | {int(row.episodes)} | {row.net_pnl:,.2f} | {row.fees:,.2f} | {row.loss_cvar_95:,.2f} | {row.daily_sharpe_annualized:.3f} | {row.max_daily_drawdown_fraction:.2%} |")
    lines += ["", "Lower loss CVaR is better; higher mean P&L is better. Convergence accuracy is not hedge performance.",
              "Read test/paired_comparisons.csv for session-bootstrap intervals and test/daily_equity.csv for zero-trade sessions.",
              "No option entries were chosen from test outcomes or ML probabilities. Missing ML probabilities preserve positions with a neutral/missing state.",
              "Selected residual checkpoints may be epoch zero if adaptation did not improve validation CVaR.",
              "Historical prices, policies and these dates have been inspected before. No superiority or deployment claim follows from this experiment alone."]
    (output/"RESULTS.md").write_text("\n".join(lines)+"\n")


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=root/"data/research/expanded_20261030")
    parser.add_argument("--part1", type=Path, default=root/"data/research/hedging_part1_complete_20261031")
    parser.add_argument("--part2", type=Path, default=root/"data/research/deep_hedging_part2_final_20261031")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--updates-per-epoch", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    spec = HistoricalTraining(epochs=args.epochs, updates_per_epoch=args.updates_per_epoch, batch_size=args.batch_size)
    run(args.output, args.stage, args.part1, args.part2, spec)
