"""Independent-path risk comparison. No Sharpe from arbitrary simulation order."""

import numpy as np
import pandas as pd
import torch

from .batch_environment import rollout_baseline
from .metrics import empirical_loss_risk
from .policy import rollout_policy, load_policy
from .simulation_bridge import scalar_replay


def metrics(result, alpha=.95):
    pnl = result["net_pnl"]
    var, cvar = empirical_loss_risk(pnl, alpha)
    return {"paths": len(pnl), "mean_net_pnl": float(pnl.mean()), "net_pnl_std": float(pnl.std(ddof=1)),
            "mean_gross_pnl": float(result["gross_pnl"].mean()), "loss_var_95": var, "loss_cvar_95": cvar,
            "worst_loss": float(-pnl.min()), "mean_fees": float(result["fees"].mean()),
            "mean_execution_impact": float(result["execution_impact"].mean()),
            "mean_future_orders": float(result["future_orders"].mean()),
            "forced_exits": int(result["forced_exits"].sum()), "writeoffs": int(result["writeoffs"].sum()),
            "margin_constrained_actions": int(result["margin_constrained"].sum()),
            "maximum_margin": float(result["peak_margin"].max()),
            "minimum_free_cash": float(result["minimum_free_cash"].min()),
            "max_reconciliation_error": float(abs(result["reconciliation_error"]).max())}


def paired_intervals(neural, reference, repeats=1000):
    rng = np.random.default_rng(887)
    means, risks = [], []
    n = len(neural)
    for _ in range(repeats):
        idx = rng.integers(0, n, n)
        means.append(float((neural[idx]-reference[idx]).mean()))
        risks.append(empirical_loss_risk(neural[idx])[1]-empirical_loss_risk(reference[idx])[1])
    return {"mean_pnl_difference": float((neural-reference).mean()),
            "mean_pnl_difference_lower_95": float(np.quantile(means, .025)),
            "mean_pnl_difference_upper_95": float(np.quantile(means, .975)),
            "loss_cvar_difference": empirical_loss_risk(neural)[1]-empirical_loss_risk(reference)[1],
            "loss_cvar_difference_lower_95": float(np.quantile(risks, .025)),
            "loss_cvar_difference_upper_95": float(np.quantile(risks, .975))}


def evaluate_scenario(name, market_name, batch, config, model_folder, selection, output, include_bs=False):
    output.mkdir(parents=True, exist_ok=False)
    rows, pnls = [], {}
    strategies = ["unhedged", "static_delta", "dynamic_delta", "band_delta"]
    if include_bs:
        strategies.append("bs_dynamic")
    replay_errors = {}
    for strategy in strategies:
        result = rollout_baseline(batch, config, strategy)
        rows.append({"scenario": name, "strategy": strategy, **metrics(result)})
        pnls[strategy] = result["net_pnl"]
        np.savez_compressed(output / f"{strategy}_execution.npz", **result)
        if strategy != "bs_dynamic":
            scalar = scalar_replay(batch, config, strategy, count=16)
            error = float(np.max(abs(scalar-result["net_pnl"][:len(scalar)])))
            replay_errors[strategy] = error
            if error > 1e-6:
                raise AssertionError("Scalar/batch baseline execution differs")
    selected_seed = selection["selected"]["seed"]
    for trial in selection["trials"]:
        model, checkpoint = load_policy(model_folder / trial["checkpoint"])
        with torch.no_grad():
            result = rollout_policy(batch, config, model)[0]
        strategy = f"neural_seed_{trial['seed']}"
        selected = trial["seed"] == selected_seed
        rows.append({"scenario": name, "strategy": strategy, "selected_on_validation": selected,
                     "best_epoch": checkpoint["best_epoch"], **metrics(result)})
        pnls[strategy] = result["net_pnl"]
        np.savez_compressed(output / f"{strategy}_execution.npz", **result)
        if selected:
            scalar = scalar_replay(batch, config, "neural_policy", model=model, count=32)
            error = float(np.max(abs(scalar-result["net_pnl"][:len(scalar)])))
            replay_errors[strategy] = error
            if error > 1e-6:
                raise AssertionError("Scalar/batch neural execution differs")
    selected_name = f"neural_seed_{selected_seed}"
    paired = [{"scenario": name, "neural_policy": selected_name, "reference": reference,
               **paired_intervals(pnls[selected_name], pnls[reference])} for reference in strategies]
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(pnls).to_parquet(output / "paired_path_pnl.parquet", index=False)
    return rows, paired, replay_errors
