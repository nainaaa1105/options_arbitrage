"""Reproducible Part 1 historical benchmark; no orders or model training."""

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import HedgeConfig
from .data import OPTION_COLUMNS, build_episodes
from .engine import compare_baselines
from .metrics import make_reports, paired_comparison


def run(options_file, futures_file, output, config, decision_times=None, reuse_execution=None):
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source_snapshot"
    snapshot.mkdir()
    for source_file in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source_file, snapshot / source_file.name)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.glob("*.py")}
    options = pd.read_parquet(options_file, columns=OPTION_COLUMNS)
    futures = pd.read_parquet(futures_file)
    kwargs = {"decision_times": tuple(decision_times)} if decision_times is not None else {}
    episodes, paths, skips, dates = build_episodes(options, futures, config, **kwargs)
    episodes.to_parquet(output / "episode_universe.parquet", index=False)
    paths.to_parquet(output / "episode_paths.parquet", index=False)
    skips.to_csv(output / "data_rejections.csv", index=False)
    if episodes.empty:
        raise RuntimeError("No valid hedging episodes; inspect data_rejections.csv")
    print(f"Prepared {len(episodes):,} fixed episodes and {len(paths):,} path rows", flush=True)
    if reuse_execution is not None:
        previous = json.loads((reuse_execution / "manifest.json").read_text(encoding="utf-8"))
        if previous["config"] != config.to_dict():
            raise ValueError("Reuse requires identical experiment configuration")
        for name in ("engine.py", "accounting.py", "baselines.py", "costs.py"):
            digest = hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
            if digest != previous["source_sha256"][name]:
                raise ValueError("Cannot reuse execution after an accounting/policy change")
        pd.testing.assert_frame_equal(episodes, pd.read_parquet(reuse_execution / "episode_universe.parquet"))
        pd.testing.assert_frame_equal(paths, pd.read_parquet(reuse_execution / "episode_paths.parquet"))
        results = pd.read_parquet(reuse_execution / "episode_results.parquet")
        orders = pd.read_parquet(reuse_execution / "transaction_ledger.parquet")
        equity = pd.read_parquet(reuse_execution / "marked_equity.parquet")
        old_rejections = reuse_execution / "capital_rejections.csv"
        capital_rejections = pd.read_csv(old_rejections) if old_rejections.stat().st_size > 3 else pd.DataFrame()
    else:
        results, orders, equity, capital_rejections = compare_baselines(episodes, paths, config)
    capital_rejections.to_csv(output / "capital_rejections.csv", index=False)
    if results.empty:
        raise RuntimeError("No episodes fit the common capital reserve; inspect capital_rejections.csv")
    report, daily = make_reports(results, orders, equity, dates, config)
    paired = paired_comparison(results, dates)
    results.to_parquet(output / "episode_results.parquet", index=False)
    orders.to_parquet(output / "transaction_ledger.parquet", index=False)
    equity.to_parquet(output / "marked_equity.parquet", index=False)
    daily.to_csv(output / "daily_equity.csv", index=False)
    report.to_csv(output / "baseline_comparison.csv", index=False)
    paired.to_csv(output / "paired_comparison.csv", index=False)
    missing_dates = sorted(set(dates) - set(pd.to_datetime(futures.timestamp).dt.date))
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config.to_dict(),
                "inputs": {"options": str(options_file.resolve()), "futures": str(futures_file.resolve())},
                "input_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (options_file, futures_file)},
                "source_sha256": source_hashes,
                "option_rows": len(options), "future_rows": len(futures), "calendar_sessions": len(dates),
                "sessions_without_futures": [str(d) for d in missing_dates], "prepared_episodes": len(episodes),
                "calendar_basis": "Union of observed option and front-month futures dates; includes zero-trade sessions",
                "sessions_without_options": [str(d) for d in sorted(set(dates)-set(pd.to_datetime(options.timestamp).dt.date))],
                "reused_verified_execution": str(reuse_execution.resolve()) if reuse_execution else None,
                "common_executed_episodes": int(results.episode_id.nunique()), "data_rejections": len(skips),
                "capital_rejections": len(capital_rejections), "maximum_reconciliation_error": float(results.reconciliation_error.abs().max()),
                "position_sampling": "Alternating long CE/PE, nearest expiry then ATM, fixed decision times; no ML/outcome selection",
                "execution": "Immediately completed candle state, next bar open, configured adverse spread/slippage and tick rounding",
                "futures_sensitivity": "Observed F/S deterministic-carry approximation, not an estimated stochastic-basis hedge",
                "margin": "Configured notional fraction and explicit short-option proxy; not historical SPAN",
                "cash": "Premium/realized futures P&L/fees affect cash; no futures notional purchase and no positive unrealized MTM credit",
                "zero_bid_rule": "A terminal long option with no positive proxy bid is conservatively written off at zero; not a claimed executable fill",
                "scope": "Sequential intraday fixed positions; no concurrent multi-option netting, overnight settlement, deep learning or live execution"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    columns = ["strategy", "episodes", "net_pnl", "fees", "execution_impact", "loss_cvar_95", "daily_sharpe_annualized"]
    print(report[columns].to_string(index=False), flush=True)
    lines = ["# Part 1 historical hedge benchmark", "", "All results use configured OHLC execution and margin proxies.", "",
             "| Strategy | Episodes | Gross P&L | Net P&L | Fees | Loss CVaR 95% | Daily Sharpe |", "|---|---:|---:|---:|---:|---:|---:|"]
    for _, r in report.iterrows():
        lines.append(f"| {r.strategy} | {int(r.episodes)} | {r.gross_pnl:,.2f} | {r.net_pnl:,.2f} | {r.fees:,.2f} | {r.loss_cvar_95:,.2f} | {r.daily_sharpe_annualized:.3f} |")
    lines += ["", "These are conventional hedging baselines, not ML/deep-hedging performance.",
              "The same episode IDs are used for all strategies; capital rejections are common.",
              "See manifest.json for assumptions and paired_comparison.csv for session-bootstrap comparisons."]
    (output / "RESULTS.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--options", type=Path, default=root / "data/research/expanded_20261030/data/processed/options_features.parquet")
    parser.add_argument("--futures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="JSON overrides for HedgeConfig; defaults are explicit configured proxies")
    parser.add_argument("--horizon", type=int, choices=[30, 60])
    parser.add_argument("--decision-times", help="Comma-separated completed candle START labels; decisions are available five minutes later")
    parser.add_argument("--reuse-execution", type=Path, help="Reuse prior ledgers only when config, policies and every episode/path row match; recompute reports")
    args = parser.parse_args()
    values = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    if args.horizon is not None:
        values["horizon_minutes"] = args.horizon
    config = HedgeConfig(**values)
    run(args.options, args.futures, args.output, config,
        args.decision_times.split(",") if args.decision_times else None, args.reuse_execution)
