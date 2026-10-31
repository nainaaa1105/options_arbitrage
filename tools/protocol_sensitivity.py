"""Sensitivity and power analysis for the Phase 9 acceptance criteria, using the Part 3 historical test episodes.

Question answered: if hedging behaved on live data as it did in the 28 historical test sessions, how often would the predeclared
criteria pass after D forward days, and how often would they pass when hedging has half the effect or no effect at all?
The bootstrap resamples whole days from one 28-day period, so it understates regime risk; read the numbers as optimistic upper bounds.

Run: $env:PYTHONPATH="src"; .\\.venv\\Scripts\\python.exe tools\\protocol_sensitivity.py --output docs\\results\\protocol_sensitivity
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forward_validation.acceptance import loss_cvar

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data/research/historical_hedging_part3_final_20261031/test/episode_results.parquet"
RATIOS = (.25, .5, .75, 1.0, 1.5, 2.0)


def icc(values, groups):
    """One-way random-effects intraclass correlation (ANOVA estimator) for equal group sizes."""
    frame = pd.DataFrame({"v": values, "g": groups})
    k = frame.groupby("g").size().iloc[0]
    grand = frame.v.mean()
    means = frame.groupby("g").v.mean()
    ms_between = k * ((means - grand) ** 2).sum() / (len(means) - 1)
    ms_within = ((frame.v - frame.g.map(means)) ** 2).sum() / (len(frame) - len(means))
    return float(max((ms_between - ms_within) / (ms_between + (k - 1) * ms_within), 0.)), int(k)


def one_trial(candidate, reference, day_index, rng, boot):
    """One simulated forward window: point estimates and a day-cluster bootstrap interval for the CVaR reduction."""
    reduction = loss_cvar(reference) - loss_cvar(candidate)
    cost = float(reference.mean() - candidate.mean())
    members = [np.flatnonzero(day_index == d) for d in np.unique(day_index)]
    draws = []
    for _ in range(boot):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        draws.append(loss_cvar(reference[idx]) - loss_cvar(candidate[idx]))
    return reduction, cost, float(np.quantile(draws, .025))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=300)
    parser.add_argument("--bootstrap", type=int, default=300)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    r = pd.read_parquet(RESULTS)
    r["day"] = pd.to_datetime(r.entry_time).dt.date
    wide = r.pivot(index="episode_id", columns="strategy", values="net_pnl")
    day = r.drop_duplicates("episode_id").set_index("episode_id").day.reindex(wide.index)
    unhedged, dynamic = wide.unhedged.to_numpy(), wide.dynamic_delta.to_numpy()
    days = np.array([d for d in day])
    unique_days = sorted(set(days))
    by_day = {d: np.flatnonzero(days == d) for d in unique_days}

    out = {"episodes": len(wide), "days": len(unique_days), "episodes_per_day": len(wide) // len(unique_days),
           "observed_cvar95_reduction": loss_cvar(unhedged) - loss_cvar(dynamic),
           "observed_cost_per_episode": float(unhedged.mean() - dynamic.mean())}
    out["observed_cost_over_reduction"] = out["observed_cost_per_episode"] / out["observed_cvar95_reduction"]
    out["icc"] = {}
    for name, series in (("unhedged_pnl", unhedged), ("dynamic_delta_pnl", dynamic), ("paired_difference", unhedged - dynamic)):
        rho, m = icc(series, days)
        out["icc"][name] = {"icc": rho, "design_effect": 1 + (m - 1) * rho,
                            "effective_episodes_of_%d" % len(series): len(series) / (1 + (m - 1) * rho)}

    rng = np.random.default_rng(2026)
    noise_sd = float(np.std(dynamic - unhedged))
    scenarios = {"historical_effect": dynamic,
                 "half_effect": unhedged + .5 * (dynamic - unhedged),
                 "double_cost": dynamic - float(unhedged.mean() - dynamic.mean()),      # same tail benefit, hedging costs twice as much
                 "no_effect_with_noise": unhedged + rng.normal(0, noise_sd, len(unhedged))}
    rows = []
    for scenario, candidate in scenarios.items():
        for D in (30, 45, 60, 90, 120):
            passes_cvar, ratios, reductions = 0, [], []
            for _ in range(args.simulations):
                picked = rng.integers(0, len(unique_days), D)
                index = np.concatenate([by_day[unique_days[i]] for i in picked])
                day_index = np.concatenate([np.full(len(by_day[unique_days[i]]), k) for k, i in enumerate(picked)])
                reduction, cost, lower = one_trial(candidate[index], unhedged[index], day_index, rng, args.bootstrap)
                passes_cvar += lower > 0
                reductions.append(reduction)
                ratios.append(cost / reduction if reduction > 0 else np.inf)
            ratios = np.array(ratios)
            row = {"scenario": scenario, "forward_days": D, "p_cvar_interval_excludes_zero": passes_cvar / args.simulations,
                   "median_cvar95_reduction": float(np.median(reductions)), "median_cost_over_reduction": float(np.median(ratios[np.isfinite(ratios)])) if np.isfinite(ratios).any() else None}
            for t in RATIOS:
                row[f"p_cost_ratio_le_{t}"] = float((ratios <= t).mean())
            rows.append(row)
            print(f"{scenario:<22} D={D:<4} P(CVaR interval>0)={row['p_cvar_interval_excludes_zero']:.2f}  P(ratio<=1.0)={row['p_cost_ratio_le_1.0']:.2f}", flush=True)
    pd.DataFrame(rows).to_csv(args.output / "power.csv", index=False)
    (args.output / "icc_and_observed.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: out[k] for k in ("observed_cvar95_reduction", "observed_cost_per_episode", "observed_cost_over_reduction")}, indent=2))
    print(json.dumps(out["icc"], indent=2))


if __name__ == "__main__":
    main()
