"""Profit-based target: does anything known at the decision bar predict NET P&L of a long option held 30 minutes?

The convergence label asks whether a pricing gap closes. This experiment asks the economic question directly. For every undervalued
candidate it computes the real net P&L of buying at the next bar's open and selling 30 minutes later, using the same spread,
slippage, tick rounding, brokerage, exchange charges and taxes as the hedging engine. A classifier is then trained on "net P&L > 0"
with the same purged chronological blocks as the main study, and selection rules are judged by the mean net P&L of what they pick.

Selection and thresholds use history up to 2026-09-15 only. The later period is read once, after the model and cutoff are frozen,
and it was reviewed during earlier work, so it is retrospective. The original convergence target is untouched.
"""
import argparse
import json
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from deep_hedging.config import HedgeConfig
from deep_hedging.costs import execution_price, transaction_fees
from robust_model_research import purged_folds

from .experiment import BASELINE_LGB, load_tables
from .features import V2_COLUMNS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
PART3_PLAN = ROOT / "data/research/historical_hedging_part3_final_20261031/experiment_plan.json"
STEP = pd.Timedelta(minutes=5)
SELECT_QUANTILE = .8                     # act on the top 20% by predicted probability


def round_trip_pnl(entry_open, exit_open, entry_time, exit_time, lot_size, config):
    """Net rupee P&L for a long option bought at one open and sold at another, or NaN if either leg is not executable."""
    try:
        quantity = config.option_lots * int(lot_size)
        buy = execution_price(entry_open, quantity, config.option_tick_size, config.option_half_spread, config.option_slippage)
        sell = execution_price(exit_open, -quantity, config.option_tick_size, config.option_half_spread, config.option_slippage)
        fees = transaction_fees("option", quantity, buy, entry_time, config)["total"] + transaction_fees("option", -quantity, sell, exit_time, config)["total"]
        return float(quantity * (sell - buy) - fees)
    except ValueError:
        return np.nan


def attach_pnl(frame, opens, config):
    """Add entry/exit opens and net P&L. Rows without both exact bars stay NaN; nothing is filled."""
    d = frame.copy()
    d["entry_time"], d["exit_time"] = d.timestamp + STEP, d.timestamp + STEP + pd.Timedelta(minutes=30)
    for side, column in (("entry", "entry_time"), ("exit", "exit_time")):
        o = opens.rename(columns={"timestamp": column, "open": f"{side}_open"})[["instrument_key", column, f"{side}_open"]]
        d = d.merge(o, on=["instrument_key", column], how="left", validate="many_to_one")
    lots = opens.drop_duplicates("instrument_key").set_index("instrument_key").lot_size
    d["lot_size"] = d.instrument_key.map(lots)
    d["net_pnl"] = [round_trip_pnl(a, b, t0, t1, ls, config) if np.isfinite(a) and np.isfinite(b) else np.nan
                    for a, b, t0, t1, ls in zip(d.entry_open, d.exit_open, d.entry_time, d.exit_time, d.lot_size)]
    return d


def day_bootstrap(values, days, repeats=2000, seed=11):
    values, days = np.asarray(values, float), np.asarray(days)
    unique = np.unique(days)
    members = [np.flatnonzero(days == d) for d in unique]
    rng = np.random.default_rng(seed)
    means = [values[np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])].mean() for _ in range(repeats)]
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def paired_difference(selected_pnl, selected_days, all_pnl, all_days, repeats=2000, seed=12):
    """Mean(selected) - mean(all candidates), day-cluster bootstrap over the days that have both."""
    s = pd.DataFrame({"d": selected_days, "v": selected_pnl}).groupby("d").v.agg(["sum", "count"])
    a = pd.DataFrame({"d": all_days, "v": all_pnl}).groupby("d").v.agg(["sum", "count"])
    days = sorted(set(s.index) & set(a.index))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(repeats):
        pick = rng.integers(0, len(days), len(days))
        d = [days[i] for i in pick]
        diffs.append(s.loc[d, "sum"].sum() / s.loc[d, "count"].sum() - a.loc[d, "sum"].sum() / a.loc[d, "count"].sum())
    point = s.loc[days, "sum"].sum() / s.loc[days, "count"].sum() - a.loc[days, "sum"].sum() / a.loc[days, "count"].sum()
    return float(point), float(np.quantile(diffs, .025)), float(np.quantile(diffs, .975))


def run(stage, daily_file, regime_dir, output):
    output.mkdir(parents=True, exist_ok=False)
    config = HedgeConfig(**json.loads(PART3_PLAN.read_text())["config"])
    history, fresh, base_columns, _ = load_tables(stage, daily_file, regime_dir)
    history["row"] = np.arange(len(history))
    opens = pd.read_parquet(stage / "data/processed/options_features.parquet", columns=["instrument_key", "timestamp", "open", "lot_size"])
    opens["timestamp"] = pd.to_datetime(opens.timestamp)
    columns = base_columns + V2_COLUMNS
    candidates = {}
    for name, frame in (("history", history), ("fresh", fresh)):
        c = frame[frame.reference_price_deviation < 0].copy()          # long-only: undervalued side
        c = attach_pnl(c, opens, config)
        c = c[c.net_pnl.notna()].reset_index(drop=True)
        c["day"] = c.timestamp.dt.date
        c["profit"] = (c.net_pnl > 0).astype(int)
        candidates[name] = c
        print(f"{name}: {len(c):,} executable candidates, mean net P&L {c.net_pnl.mean():,.0f}, win rate {c.profit.mean():.3f}", flush=True)
    h, f = candidates["history"], candidates["fresh"]
    summary = {"candidates": {k: {"rows": len(v), "mean_net_pnl": float(v.net_pnl.mean()), "win_rate": float(v.profit.mean())} for k, v in candidates.items()},
               "cost_model": "Part 3 HedgeConfig: half-spread and slippage 0.15 each, tick 0.05, brokerage, exchange, SEBI, stamp, GST, STT; 4 lots; fills at candle opens"}
    folds = list(purged_folds(h, n_splits=5, gap_sessions=1))
    oof = np.full(len(h), np.nan)
    fold_of = np.zeros(len(h), int)
    rows = []
    for fold, train, valid in folds:
        model = lgb.LGBMClassifier(**BASELINE_LGB).fit(h.iloc[train][columns], h.profit.iloc[train])
        oof[valid] = model.predict_proba(h.iloc[valid][columns])[:, 1]
        fold_of[valid] = fold
        y = h.profit.iloc[valid]
        rows.append({"block": fold, "rows": len(valid), "base_rate": float(y.mean()), "auc": float(roc_auc_score(y, oof[valid])) if y.nunique() == 2 else np.nan})
    pd.DataFrame(rows).to_csv(output / "profit_classifier_blocks.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False), flush=True)
    # Selection rule: cutoff = 80th percentile of earlier blocks' out-of-fold scores, applied to each later block.
    picks = []
    for k in (3, 4, 5):
        cutoff = np.nanquantile(oof[(fold_of >= 2) & (fold_of < k)], SELECT_QUANTILE)
        now = fold_of == k
        picks.append(h[now & (oof >= cutoff)].assign(rule="profit_classifier"))
    selected = pd.concat(picks)
    evaluated = h[fold_of >= 3]
    result = {"all_candidates_blocks_3_5": dict(zip(("mean", "lower95", "upper95"), day_bootstrap(evaluated.net_pnl, evaluated.day)))}
    result["profit_classifier_top20pct"] = dict(zip(("mean", "lower95", "upper95"), day_bootstrap(selected.net_pnl, selected.day)))
    result["profit_classifier_top20pct"].update(rows=len(selected), win_rate=float(selected.profit.mean()))
    diff = paired_difference(selected.net_pnl, selected.day, evaluated.net_pnl, evaluated.day)
    result["selected_minus_all"] = dict(zip(("mean", "lower95", "upper95"), diff))
    # Comparison selector: the convergence classifier's out-of-fold probability (stack, blocks 3-5) from the accuracy study.
    oof_conv = ROOT / "data/research/accuracy_v2_20261031/oof_predictions.parquet"
    if oof_conv.exists():
        conv = pd.read_parquet(oof_conv)
        h["convergence_p"] = h.row.map(dict(zip(conv.row, conv["stack_logistic"])))
        picks = []
        for k in (4, 5):
            cutoff = np.nanquantile(h.loc[(fold_of >= 3) & (fold_of < k), "convergence_p"], SELECT_QUANTILE)
            now = (fold_of == k) & (h.convergence_p >= cutoff)
            picks.append(h[now])
        conv_sel = pd.concat(picks)
        ev45 = h[fold_of >= 4]
        result["convergence_classifier_top20pct_blocks_4_5"] = dict(zip(("mean", "lower95", "upper95"), day_bootstrap(conv_sel.net_pnl, conv_sel.day)))
        result["convergence_classifier_top20pct_blocks_4_5"]["rows"] = len(conv_sel)
        result["all_candidates_blocks_4_5"] = dict(zip(("mean", "lower95", "upper95"), day_bootstrap(ev45.net_pnl, ev45.day)))
        result["convergence_selected_minus_all_blocks_4_5"] = dict(zip(("mean", "lower95", "upper95"), paired_difference(conv_sel.net_pnl, conv_sel.day, ev45.net_pnl, ev45.day)))
    # Freeze: fit on all history, cutoff from OOF blocks 2-5, then read the later period once.
    final = lgb.LGBMClassifier(**BASELINE_LGB).fit(h[columns], h.profit)
    cutoff = float(np.nanquantile(oof[fold_of >= 2], SELECT_QUANTILE))
    (output / "frozen_before_fresh.json").write_text(json.dumps({"cutoff": cutoff, "quantile": SELECT_QUANTILE, "history_rows": len(h)}, indent=2))
    f["score"] = final.predict_proba(f[columns])[:, 1]
    chosen = f[f.score >= cutoff]
    result["fresh_all_candidates"] = dict(zip(("mean", "lower95", "upper95"), day_bootstrap(f.net_pnl, f.day)))
    result["fresh_selected"] = dict(zip(("mean", "lower95", "upper95"), day_bootstrap(chosen.net_pnl, chosen.day)))
    result["fresh_selected"].update(rows=len(chosen), win_rate=float(chosen.profit.mean()), share_selected=float(len(chosen) / len(f)))
    result["fresh_selected_minus_all"] = dict(zip(("mean", "lower95", "upper95"), paired_difference(chosen.net_pnl, chosen.day, f.net_pnl, f.day)))
    result["fresh_auc"] = float(roc_auc_score(f.profit, f.score))
    summary["results_inr_per_trade"] = result
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, default=ROOT / "data/research/expanded_20261030")
    parser.add_argument("--daily-spot", type=Path, default=ROOT / "data/raw/upstox/nifty_daily_to_20261029.parquet")
    parser.add_argument("--regime-dir", type=Path, default=ROOT / "data/research/expanded_20261030/regime_study")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.stage, args.daily_spot, args.regime_dir, args.output)
