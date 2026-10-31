"""Train GBM/Heston deep hedgers, freeze, then evaluate predefined scenarios."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch

from .config import HedgeConfig
from .simulation import MarketSpec, HestonOracle, simulate
from .train import TrainingSpec, train_market
from .evaluate_simulation import evaluate_scenario


def oracle_audit(oracle, count=256):
    batch = simulate(oracle.spec, count, seed=701, oracle=oracle)
    direct_price, direct_delta = oracle.direct_marks(batch["spot"][:, 0], batch["strike"],
        batch["maturity"][:, 0], batch["variance"][:, 0], batch["call"])
    price_error = float(np.max(abs(direct_price-batch["option"][:, 0])))
    delta_error = float(np.max(abs(direct_delta-batch["delta"][:, 0])))
    if price_error > 1. or delta_error > .005:
        raise ArithmeticError("Pricing interpolation error exceeds declared audit tolerance")
    return {"states": count, "max_price_error_rupees_per_option_unit": price_error,
            "max_delta_error": delta_error, "pricing": "Zero-rate Heston Fourier integration and audited grid interpolation"}


def doubled_costs(config):
    # A hypothetical stress, not a claim that statutory schedules change.
    overrides = {name: getattr(config, name)*2 for name in (
        "option_half_spread", "option_slippage", "future_half_spread", "future_slippage",
        "brokerage_per_order", "future_brokerage_cap_rate", "option_exchange_rate", "future_exchange_rate",
        "sebi_rate", "option_ipft_rate", "future_ipft_rate", "option_stamp_rate", "future_stamp_rate")}
    overrides.update(option_stt_override=.003, future_stt_override=.001)
    return replace(config, **overrides)


def run(output, training, config=None, test_seed_offset=1_000_000):
    config = config or HedgeConfig()
    if config.max_future_lots < 1:
        raise ValueError("Neural hedging requires positive action capacity")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, snapshot / source.name)
    torch.set_num_threads(training.threads)
    torch.use_deterministic_algorithms(True)
    specs = {name: MarketSpec(model=name, horizon_minutes=config.horizon_minutes,
                             steps=config.horizon_minutes//5) for name in ("gbm", "heston")}
    stress_spec = replace(specs["heston"], sigma_low=.25, sigma_high=.45,
                          maturity_days_low=.5, maturity_days_high=2., kappa=1., theta=.09, vol_of_vol=.8, rho=-.85)
    plan = {"created_at": datetime.now(timezone.utc).isoformat(), "training": asdict(training),
            "execution": config.to_dict(), "markets": {k: asdict(v) for k, v in specs.items()},
            "stress_market": asdict(stress_spec), "test_scenarios": ["gbm", "heston", "heston_parameter_shift", "gbm_double_cost", "heston_double_cost"],
            "seeds": {"gbm_validation": 50001, "heston_validation": 150001,
                      "gbm_test": 90001+test_seed_offset, "heston_test": 190001+test_seed_offset,
                      "heston_parameter_shift": 290001+test_seed_offset},
            "test_protocol": "Independent test bank offset from the initial exploratory implementation run; no selection on prior test metrics",
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "numpy": np.__version__, "scipy": scipy.__version__, "device": "cpu"},
            "scope": "Stylized uncalibrated zero-drift simulations, independent whole paths and four-lot fixed option positions",
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.py")}}
    (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    oracle = HestonOracle(specs["heston"])
    oracle.save(output / "heston_pricing_grid.npz")
    audits = {"heston": oracle_audit(oracle)}
    (output / "pricing_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    selections = {}
    for name in ("gbm", "heston"):
        selections[name] = train_market(specs[name], oracle if name == "heston" else None, config, training,
                                         output / name, 50001 if name == "gbm" else 150001)
    checkpoint_hashes = {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob("*/policy_seed_*.pt")}
    frozen = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "selections": selections,
              "checkpoint_sha256": checkpoint_hashes, "test_generated_before_freeze": False,
              "selection": "Seed and post-training epoch chosen only on validation CVaR; all test scenarios predeclared"}
    (output / "frozen_before_test.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("All policy choices frozen; generating independent test paths", flush=True)
    batches = {name: simulate(specs[name], training.test_paths, seed=(90001 if name == "gbm" else 190001)+test_seed_offset,
                              oracle=oracle if name == "heston" else None) for name in ("gbm", "heston")}
    stress_oracle = HestonOracle(stress_spec)
    stress_oracle.save(output / "stress_heston_pricing_grid.npz")
    audits["heston_parameter_shift"] = oracle_audit(stress_oracle)
    batches["heston_parameter_shift"] = simulate(stress_spec, training.stress_paths, seed=290001+test_seed_offset, oracle=stress_oracle)
    (output / "pricing_audit.json").write_text(json.dumps(audits, indent=2), encoding="utf-8")
    for name, batch in batches.items():
        np.savez_compressed(output / f"{name}_test_paths.npz", **batch)
    rows, paired, replays = [], [], {}
    scenarios = [("gbm", "gbm", config), ("heston", "heston", config),
                 ("heston_parameter_shift", "heston", config),
                 ("gbm_double_cost", "gbm", doubled_costs(config)),
                 ("heston_double_cost", "heston", doubled_costs(config))]
    for scenario, family, execution in scenarios:
        batch = batches["heston_parameter_shift" if scenario == "heston_parameter_shift" else family]
        result, comparisons, replay = evaluate_scenario(scenario, family, batch, execution, output / family,
            selections[family], output / "evaluation" / scenario, include_bs=family == "heston")
        rows.extend(result)
        paired.extend(comparisons)
        replays[scenario] = replay
        print(f"Evaluated {scenario}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(output / "simulation_comparison.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    (output / "scalar_replay_checks.json").write_text(json.dumps(replays, indent=2), encoding="utf-8")
    selected = table.loc[table.strategy.isin(["unhedged", "static_delta", "dynamic_delta", "band_delta", "bs_dynamic"])
                         | table.selected_on_validation.fillna(False).astype(bool)]
    columns = ["scenario", "strategy", "mean_net_pnl", "loss_cvar_95", "mean_future_orders"]
    print(selected[columns].to_string(index=False), flush=True)
    lines = ["# Part 2 simulation results", "", "Models were selected on validation CVaR before test generation.",
             "Independent simulated paths have no natural calendar ordering: no annualized Sharpe or arbitrary path-order drawdown is reported.", "",
             "| Scenario | Strategy | Mean net P&L | Loss CVaR 95% | Mean futures orders |", "|---|---|---:|---:|---:|"]
    for _, row in selected.iterrows():
        lines.append(f"| {row.scenario} | {row.strategy} | {row.mean_net_pnl:,.2f} | {row.loss_cvar_95:,.2f} | {row.mean_future_orders:.2f} |")
    lines += ["", "These are synthetic, uncalibrated results with configured transaction costs and margin proxies, not historical or live trading results.",
              "Paired path-bootstrap intervals and all random-seed metrics are reported separately. Negative loss-CVaR differences favour the neural policy."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--updates-per-epoch", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warm-updates", type=int, default=60)
    parser.add_argument("--seeds", default="11,22,33")
    parser.add_argument("--validation-paths", type=int, default=2048)
    parser.add_argument("--test-paths", type=int, default=8192)
    parser.add_argument("--stress-paths", type=int, default=4096)
    parser.add_argument("--test-seed-offset", type=int, default=1_000_000)
    args = parser.parse_args()
    spec = TrainingSpec(seeds=tuple(int(s) for s in args.seeds.split(",")), epochs=args.epochs,
        updates_per_epoch=args.updates_per_epoch, batch_size=args.batch_size, warm_updates=args.warm_updates,
        validation_paths=args.validation_paths, test_paths=args.test_paths, stress_paths=args.stress_paths)
    run(args.output, spec, test_seed_offset=args.test_seed_offset)
