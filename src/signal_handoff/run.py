"""Phase 8 (Objective B): ML-selected long entries, hedged identically, against slot-matched non-ML entries.

Order of operations (enforced in code):
  1. verify the Part 1 inputs and reproduce Part 1's preserved episodes with the injected-choice builder;
  2. generate out-of-time probabilities at every decision-slot candidate (no outcome used);
  3. choose entries for every rule and FREEZE the selection table (hash written) before any P&L exists;
  4. only then build paths, run the shared hedging engine and evaluate.
"""
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from deep_hedging.baselines import STRATEGIES
from deep_hedging.config import HedgeConfig
from deep_hedging.data import OPTION_COLUMNS, validate_sources
from deep_hedging.engine import compare_baselines
from deep_hedging.historical_data import prepare_ml, sha256
from deep_hedging.metrics import empirical_loss_risk, make_reports
from research_targets import session_horizon_eligible

from .episodes import build_chosen_episodes
from .selection import RULES, select_slot
from .universe import DECISION_TIMES, slot_frame

VALIDATION_START = pd.Timestamp("2026-07-01", tz="Asia/Kolkata")
TEST_START = pd.Timestamp("2026-09-16", tz="Asia/Kolkata")
PRIMARY_TAU = .5
TAUS = (.5, .6, .7)
CONTROLS = ("random", "max_deviation", "atm_nearest")


def period_of(times):
    t = pd.to_datetime(times)
    return np.select([t < VALIDATION_START, t < TEST_START], ["development", "validation"], default="test")


def name_of(rule, tau):
    return f"{rule}_tau{int(round(tau * 100)):02d}"


def md(frame, columns, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for _, r in frame.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            cells.append("" if pd.isna(v) else (formats[c].format(v) if c in formats and not isinstance(v, str) else str(v)))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def verify_part1_equivalence(o, f, config, part1):
    """The injected-choice builder must reproduce Part 1's preserved episodes and paths exactly."""
    episodes = pd.read_parquet(part1 / "episode_universe.parquet")
    paths = pd.read_parquet(part1 / "episode_paths.parquet")
    chosen = episodes[["decision_time", "instrument_key"]]
    again, again_paths, skipped = build_chosen_episodes(o, f, config, chosen, validated=True)
    pd.testing.assert_frame_equal(again.reset_index(drop=True), episodes.reset_index(drop=True), check_dtype=False)
    pd.testing.assert_frame_equal(again_paths.reset_index(drop=True), paths.reset_index(drop=True), check_dtype=False)
    return {"part1_episodes": len(episodes), "part1_path_rows": len(paths), "rebuild_skipped": len(skipped)}


def choose_all(slots, taus):
    """Select entries for every rule. Uses decision-bar fields and out-of-time probabilities only."""
    rows = []
    for decision, group in slots.groupby("decision_time", sort=True):
        admitted = group[group.admitted]
        candidates = admitted[admitted.in_domain & admitted.undervalued]
        for tau in taus:
            pick = select_slot("ml", candidates, admitted, decision, tau)
            if pick is None:
                continue
            rows.append(_row("ml", tau, decision, pick, len(candidates)))
            for rule in CONTROLS:
                control = select_slot(rule, candidates, admitted, decision, tau, ml_pick=pick)
                if control is not None:
                    rows.append(_row(rule, tau, decision, control, len(candidates)))
    return pd.DataFrame(rows)


def _row(rule, tau, decision, pick, n_candidates):
    return {"strategy": name_of(rule, tau), "rule": rule, "tau": tau, "decision_time": decision,
            "instrument_key": pick.instrument_key, "option_type": pick.option_type, "probability": float(pick.probability)
            if np.isfinite(pick.probability) else np.nan,
            "reference_price_deviation": float(pick.reference_price_deviation) if pd.notna(pick.reference_price_deviation) else np.nan,
            "candidates_in_slot": n_candidates}


def date_cluster_bootstrap(values, dates, repeats=2000, seed=77):
    values, dates = np.asarray(values, float), np.asarray(dates)
    unique = np.unique(dates)
    members = [np.flatnonzero(dates == d) for d in unique]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        index = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        means.append(values[index].mean())
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def run(output, part1, stage, taus=TAUS, repeats=2000):
    output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(Path(__file__).parent, output / "source_snapshot", ignore=shutil.ignore_patterns("__pycache__"))
    manifest = json.loads((part1 / "manifest.json").read_text())
    config = HedgeConfig(**manifest["config"])
    if config.option_side != 1 or config.horizon_minutes != 30:
        raise ValueError("Objective B protocol: long options, 30-minute horizon")
    options_file, futures_file = Path(manifest["inputs"]["options"]), Path(manifest["inputs"]["futures"])
    for path in (options_file, futures_file):
        expected = next(v for k, v in manifest["input_sha256"].items() if Path(k).resolve() == path.resolve())
        if sha256(path) != expected:
            raise ValueError(f"Part 1 input changed: {path.name}")
    plan = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "objective": "B: ML-selected long entries then hedge",
            "config": config.to_dict(), "decision_times": list(DECISION_TIMES), "taus": list(taus), "primary_tau": PRIMARY_TAU,
            "universe": "Part 1 admission + classifier domain (signal_eligible, |dev|>=1, session horizon) + undervalued (negative market-minus-reference deviation); long options only",
            "entry": "At most one option per decision slot; buy at the next bar's open; 30-minute hold; same Part 1 execution/cost/margin engine",
            "ml_rule": "Highest out-of-time endpoint-convergence probability >= tau at the decision bar; ties larger |deviation| then instrument key",
            "controls": {"random": "seeded uniform among eligible undervalued candidates", "max_deviation": "largest |deviation| among the same candidates",
                         "atm_nearest": "Part 1 no-signal nearest-expiry/ATM option of the ML trade's option type"},
            "matching": "Controls trade only in slots where the ML rule fires; unmatched slots are skipped by every rule",
            "hedgers": list(STRATEGIES), "periods": {"development": "< 2026-07-01 (after 60-session classifier burn-in)",
                                                      "validation": "2026-07-01..2026-09-15", "test": ">= 2026-09-16"},
            "status_of_test": "Retrospective: these dates were inspected in earlier phases; confirmatory evidence is Phase 9 only",
            "claim_rule": "ML selection adds value only if the slot-paired date-cluster interval of (ML - control) excludes zero for the same hedger in a period",
            "inputs_sha256": manifest["input_sha256"]}
    (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2, default=str))
    options = pd.read_parquet(options_file, columns=OPTION_COLUMNS)
    futures = pd.read_parquet(futures_file)
    o, f = validate_sources(options, futures, config.bar_minutes)
    print("Verifying injected-choice builder against preserved Part 1 episodes", flush=True)
    equivalence = verify_part1_equivalence(o, f, config, part1)
    domain = pd.read_parquet(stage / "data/processed/options_parity_features.parquet",
                             columns=["instrument_key", "timestamp", "signal_eligible", "reference_price_deviation"])
    domain["timestamp"] = pd.to_datetime(domain.timestamp).dt.tz_convert("Asia/Kolkata")
    domain["session_ok"] = session_horizon_eligible(domain.timestamp)
    slots = slot_frame(o, config, domain=domain)
    query = slots.loc[slots.admitted & slots.in_domain & slots.undervalued, ["instrument_key", "timestamp"]]
    query = query.rename(columns={"timestamp": "observation_timestamp"})
    print(f"Generating out-of-time probabilities for {len(query):,} candidate rows", flush=True)
    probabilities, fold_audit, ml_info = prepare_ml(stage, query, output / "classifier_folds")
    probabilities.to_parquet(output / "walk_forward_probabilities.parquet", index=False)
    probabilities["timestamp"] = pd.to_datetime(probabilities.timestamp).dt.tz_convert("Asia/Kolkata")
    probabilities["classifier_fit_before"] = pd.to_datetime(probabilities.classifier_fit_before).dt.tz_convert("Asia/Kolkata")
    if not (probabilities.timestamp >= probabilities.classifier_fit_before).all():
        raise AssertionError("A probability was produced by a classifier fitted after its decision bar")
    slots = slots.merge(probabilities[["instrument_key", "timestamp", "probability"]], on=["instrument_key", "timestamp"],
                        how="left", validate="one_to_one")
    selections = choose_all(slots, taus)
    selections["period"] = period_of(selections.decision_time)
    selections.to_parquet(output / "selections.parquet", index=False)
    frozen = {"frozen_at": datetime.now(timezone.utc).isoformat(), "selections_sha256": sha256(output / "selections.parquet"),
              "rows": len(selections), "outcomes_computed_before_freeze": False}
    (output / "selections_frozen.json").write_text(json.dumps(frozen, indent=2))
    first_fold = pd.Timestamp(fold_audit[0]["prediction_start"], tz="Asia/Kolkata")
    for tau in taus:
        sets = {r: set(selections[selections.strategy == name_of(r, tau)].decision_time) for r in RULES}
        if not all(sets[r] <= sets["ml"] for r in CONTROLS):
            raise AssertionError("A control traded outside the ML slots")
    ml = selections[selections.rule == "ml"]
    if (ml.probability < ml.tau).any() or ml.decision_time.lt(first_fold).any():
        raise AssertionError("ML entry violates its rule or precedes the classifier burn-in")
    dates = sorted(set(o.timestamp.dt.date) | set(f.timestamp.dt.date))
    results, reports, skips, caprej = {}, [], [], []
    for strategy, chosen in selections.groupby("strategy", sort=True):
        episodes, paths, skipped = build_chosen_episodes(o, f, config, chosen[["decision_time", "instrument_key"]], validated=True)
        skipped["strategy"] = strategy
        skips.append(skipped)
        if episodes.empty:
            continue
        res, orders, equity, cap = compare_baselines(episodes, paths, config)
        cap["strategy"] = strategy
        caprej.append(cap)
        report, daily = make_reports(res, orders, equity, dates, config)
        report.insert(0, "selection", strategy)
        reports.append(report)
        folder = output / "strategies" / strategy
        folder.mkdir(parents=True)
        episodes.to_parquet(folder / "episodes.parquet", index=False)
        res.to_parquet(folder / "episode_results.parquet", index=False)
        orders.to_parquet(folder / "transaction_ledger.parquet", index=False)
        equity.to_parquet(folder / "marked_equity.parquet", index=False)
        daily.to_csv(folder / "daily_equity.csv", index=False)
        res = res.merge(chosen[["decision_time", "instrument_key", "probability", "reference_price_deviation"]],
                        on=["decision_time", "instrument_key"], how="left")
        res["selection"], res["period"] = strategy, period_of(res.decision_time)
        results[strategy] = res
        print(f"{strategy}: {len(chosen)} selected, {res.episode_id.nunique()} executed", flush=True)
    pd.concat(reports).to_csv(output / "account_level_reports.csv", index=False)
    pd.concat(skips).to_csv(output / "build_skips.csv", index=False)
    pd.concat(caprej).to_csv(output / "capital_rejections.csv", index=False)
    everything = pd.concat(results.values(), ignore_index=True)
    # Episode-level economics by selection rule, hedger and period.
    rows = []
    for (selection, hedger, period), g in list(everything.groupby(["selection", "strategy", "period"])) + \
            [((s, h, "all"), g) for (s, h), g in everything.groupby(["selection", "strategy"])]:
        var, cvar = empirical_loss_risk(g.net_pnl) if len(g) >= 20 else (np.nan, np.nan)
        rows.append({"selection": selection, "hedger": hedger, "period": period, "episodes": len(g), "net_pnl_sum": g.net_pnl.sum(),
                     "net_pnl_mean": g.net_pnl.mean(), "win_rate": float((g.net_pnl > 0).mean()), "fees_mean": g.fees.mean(),
                     "execution_impact_mean": g.execution_impact.mean(), "loss_cvar95": cvar})
    economics = pd.DataFrame(rows)
    economics.to_csv(output / "episode_economics.csv", index=False)
    # Slot-paired ML vs control comparisons with date-cluster bootstrap.
    pairs = []
    for tau in taus:
        mlr = everything[everything.selection == name_of("ml", tau)]
        for rule in CONTROLS:
            ctl = everything[everything.selection == name_of(rule, tau)]
            for hedger in STRATEGIES:
                a = mlr[mlr.strategy == hedger].set_index("decision_time").net_pnl
                b = ctl[ctl.strategy == hedger].set_index("decision_time").net_pnl
                joined = pd.concat([a.rename("ml"), b.rename("control")], axis=1, join="inner")
                for period in ("all", "development", "validation", "test"):
                    sel = joined if period == "all" else joined[period_of(joined.index) == period]
                    if len(sel) < 20:
                        continue
                    diff = sel.ml - sel.control
                    mean, lo, hi = date_cluster_bootstrap(diff.to_numpy(), pd.to_datetime(sel.index).date, repeats)
                    pairs.append({"tau": tau, "control": rule, "hedger": hedger, "period": period, "paired_slots": len(sel),
                                  "mean_diff_ml_minus_control": mean, "lower95": lo, "upper95": hi,
                                  "ml_better_share": float((diff > 0).mean()),
                                  "verdict": "ML higher" if lo > 0 else ("ML lower" if hi < 0 else "no detectable difference")})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(output / "paired_ml_vs_control.csv", index=False)
    # Preserved original-target evaluation: how often did selected entries actually converge?
    labels = pd.concat([pd.read_parquet(stage / "data/processed" / n, columns=["instrument_key", "timestamp", "target_converged_30m"])
                        for n in ("ml_dataset.parquet", "fresh_ml_dataset.parquet")], ignore_index=True)
    labels["timestamp"] = pd.to_datetime(labels.timestamp).dt.tz_convert("Asia/Kolkata")
    labelled = selections.merge(labels, left_on=["instrument_key", "decision_time"], right_on=["instrument_key", "timestamp"], how="left")
    base = slots.loc[slots.admitted & slots.in_domain & slots.undervalued & slots.probability.notna()].merge(
        labels, on=["instrument_key", "timestamp"], how="left")
    precision = [{"selection": "all_eligible_candidates", "period": "all", "selected": len(base), "labelled": int(base.target_converged_30m.notna().sum()),
                  "converged_rate": float(base.target_converged_30m.mean())}]
    for strategy, g in labelled.groupby("strategy"):
        for period, h in [("all", g)] + list(g.groupby("period")):
            precision.append({"selection": strategy, "period": period, "selected": len(h), "labelled": int(h.target_converged_30m.notna().sum()),
                              "converged_rate": float(h.target_converged_30m.mean()) if h.target_converged_30m.notna().any() else np.nan})
    precision = pd.DataFrame(precision)
    precision.to_csv(output / "convergence_precision.csv", index=False)
    # Coverage.
    valid = slots.drop_duplicates("decision_time")
    cover = []
    for period in ("development", "validation", "test"):
        in_period = valid[period_of(valid.decision_time) == period]
        in_period = in_period[in_period.decision_time >= first_fold]
        cands = slots[(slots.decision_time.isin(in_period.decision_time)) & slots.admitted & slots.in_domain & slots.undervalued]
        row = {"period": period, "slots_after_burn_in": len(in_period), "slots_with_candidates": cands.decision_time.nunique(),
               "mean_candidates_per_slot": float(cands.groupby("decision_time").size().mean()) if len(cands) else np.nan,
               "candidates_with_probability": int(cands.probability.notna().sum()), "candidates": len(cands)}
        for tau in taus:
            row[f"ml_slots_tau{int(tau * 100)}"] = int(((selections.strategy == name_of("ml", tau)) & (selections.period == period)).sum())
        cover.append(row)
    cover = pd.DataFrame(cover)
    cover.to_csv(output / "coverage.csv", index=False)
    verification = {"part1_equivalence": equivalence, "builder_reproduces_part1_exactly": True,
                    "max_reconciliation_error": float(everything.reconciliation_error.abs().max()),
                    "probabilities_from_past_fit_only": True, "controls_within_ml_slots": True,
                    "selections_frozen_before_outcomes": frozen, "shorts_used": False,
                    "primary_tau_all_period_verdicts": {f"{r.control}/{r.hedger}": r.verdict for r in pairs[
                        (pairs.tau == PRIMARY_TAU) & (pairs.period == "all")].itertuples()},
                    "profitability_established": False}
    (output / "verification.json").write_text(json.dumps(verification, indent=2, default=str))
    write_report(output, plan, cover, precision, economics, pairs, verification)
    print("Phase 8 complete", flush=True)


def write_report(output, plan, cover, precision, economics, pairs, verification):
    fmt = {c: "{:,.2f}" for c in ("net_pnl_sum", "net_pnl_mean", "fees_mean", "execution_impact_mean", "loss_cvar95",
                                  "mean_diff_ml_minus_control", "lower95", "upper95")}
    fmt.update({"win_rate": "{:.3f}", "converged_rate": "{:.3f}", "ml_better_share": "{:.3f}", "mean_candidates_per_slot": "{:.2f}"})
    primary = f"tau{int(PRIMARY_TAU * 100):02d}"
    econ = economics[economics.selection.str.endswith(primary) & economics.hedger.isin(["unhedged", "dynamic_delta", "band_delta"])]
    pair = pairs[(pairs.tau == PRIMARY_TAU) & pairs.hedger.isin(["unhedged", "dynamic_delta"])]
    text = ["# Phase 8: ML-selected entries and hedging (Objective B)", "",
            "Retrospective research; all figures use configured OHLC execution and margin proxies. "
            "Test-period dates were inspected in earlier phases, so nothing here is confirmatory evidence.", "",
            "## Contract", ""] + [f"- **{k}**: {v}" for k, v in plan.items() if k in
                                   ("objective", "universe", "entry", "ml_rule", "controls", "matching", "claim_rule", "status_of_test")] + \
           ["", "## Coverage", "", md(cover, list(cover.columns), fmt), "",
            "## Did selected entries converge? (original 30-minute endpoint label)", "",
            md(precision[precision.selection.str.contains(primary) | precision.selection.eq("all_eligible_candidates")],
               ["selection", "period", "selected", "labelled", "converged_rate"], fmt), "",
            f"## Episode economics at tau={PRIMARY_TAU}", "", md(econ.sort_values(["hedger", "period", "selection"]),
                ["selection", "hedger", "period", "episodes", "net_pnl_sum", "net_pnl_mean", "win_rate", "fees_mean", "loss_cvar95"], fmt), "",
            f"## Slot-paired ML minus control at tau={PRIMARY_TAU}", "", md(pair, ["control", "hedger", "period", "paired_slots",
                "mean_diff_ml_minus_control", "lower95", "upper95", "ml_better_share", "verdict"], fmt), "",
            "## Verification", "", "```json", json.dumps(verification, indent=2, default=str), "```", ""]
    (output / "PHASE8_REPORT.md").write_text("\n".join(text), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part1", type=Path, default=Path("data/research/hedging_part1_complete_20261031"))
    parser.add_argument("--stage", type=Path, default=Path("data/research/expanded_20261030"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    run(args.output, args.part1, args.stage, repeats=args.repeats)


if __name__ == "__main__":
    main()
