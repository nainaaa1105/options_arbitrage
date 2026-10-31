"""Sequential common-position evaluation and session-cluster uncertainty."""

import numpy as np
import pandas as pd

from .baselines import STRATEGIES
from .costs import execution_price, transaction_fees
from .engine import run_episode
from .historical_policy import HistoricalScalarPolicy
from .metrics import empirical_loss_risk, make_reports


def evaluate_historical(episodes, paths, dates, config, policies, output):
    """All strategies use identical episodes, common admission and separate cash."""
    output.mkdir(parents=True, exist_ok=False)
    strategies = (*STRATEGIES, *policies)
    capital = dict.fromkeys(strategies, config.initial_capital)
    groups = {key: group.sort_values("timestamp") for key, group in paths.groupby("episode_id")}
    results, orders, marks, rejections = [], [], [], []
    last_end = None
    for _, episode in episodes.sort_values("entry_time").iterrows():
        if last_end is not None and episode.entry_time < last_end:
            raise ValueError("Historical account cannot hold overlapping episodes")
        d = groups[episode.episode_id]
        first = d.iloc[0]
        max_units = config.max_future_lots*int(first.future_lot_size)
        try:
            fill = execution_price(first.option_mark, episode.option_quantity, config.option_tick_size,
                                   config.option_half_spread, config.option_slippage)
            reserve = (max(episode.option_quantity, 0)*fill
                       +transaction_fees("option", episode.option_quantity, fill, first.timestamp, config)["total"]
                       +max_units*first.future_mark*config.future_margin_fraction
                       +transaction_fees("future", -max_units, first.future_mark, first.timestamp, config)["total"]
                       +max_units*(config.future_half_spread+config.future_slippage+first.future_tick_size)
                       +(config.option_lots*config.short_option_margin_per_lot if episode.option_quantity < 0 else 0.))
        except ValueError:
            rejections.append({"episode_id": episode.episode_id, "reason": "unexecutable_initial_proxy"})
            continue
        if reserve > min(capital.values())*config.capital_fraction:
            rejections.append({"episode_id": episode.episode_id, "reason": "common_capital_limit"})
            continue
        for strategy in strategies:
            callback = HistoricalScalarPolicy(policies[strategy], capital[strategy]) if strategy in policies else None
            result, trades, equity = run_episode(episode, d, strategy, capital[strategy], config, policy=callback)
            capital[strategy] = result["ending_equity"]
            results.append(result)
            orders.extend(trades)
            marks.extend(equity)
        last_end = episode.exit_time
    e, t, m = pd.DataFrame(results), pd.DataFrame(orders), pd.DataFrame(marks)
    if e.empty:
        raise ValueError("No common historical episodes fit the capital reserve")
    report, daily = make_reports(e, t, m, dates, config, strategies=strategies)
    for frame, name in ((e, "episode_results"), (t, "transaction_ledger"), (m, "marked_equity")):
        frame.to_parquet(output/f"{name}.parquet", index=False)
    report.to_csv(output/"strategy_comparison.csv", index=False)
    daily.to_csv(output/"daily_equity.csv", index=False)
    pd.DataFrame(rejections, columns=["episode_id", "reason"]).to_csv(output/"capital_rejections.csv", index=False)
    return e, report


def paired_session_comparisons(results, dates, repeats=2000):
    matrix = results.pivot(index="episode_id", columns="strategy", values="net_pnl")
    if matrix.isna().any().any():
        raise ValueError("Unequal historical episode universe")
    day = pd.to_datetime(results.drop_duplicates("episode_id").set_index("episode_id").entry_time).dt.date.reindex(matrix.index)
    members = [np.flatnonzero(day.to_numpy() == date) for date in sorted(set(dates))]
    rng = np.random.default_rng(873)
    comparisons = [(name, "dynamic_delta") for name in matrix.columns if name != "dynamic_delta"]
    if {"adapted_ml", "adapted_no_ml"}.issubset(matrix.columns):
        comparisons.append(("adapted_ml", "adapted_no_ml"))
    samples = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(repeats)]
    records = []
    for strategy, reference in comparisons:
        a, b = matrix[strategy].to_numpy(), matrix[reference].to_numpy()
        means, risk = [], []
        for idx in samples:
            if len(idx):
                means.append(float((a[idx]-b[idx]).mean()))
                risk.append(empirical_loss_risk(a[idx])[1]-empirical_loss_risk(b[idx])[1])
        records.append({"strategy": strategy, "reference": reference, "paired_episodes": len(a),
                        "mean_pnl_difference": float((a-b).mean()),
                        "mean_difference_lower_95": float(np.quantile(means, .025)),
                        "mean_difference_upper_95": float(np.quantile(means, .975)),
                        "cvar_difference": empirical_loss_risk(a)[1]-empirical_loss_risk(b)[1],
                        "cvar_difference_lower_95": float(np.quantile(risk, .025)),
                        "cvar_difference_upper_95": float(np.quantile(risk, .975)),
                        "uncertainty": "Trading-session cluster bootstrap; retrospective, short final period"})
    return pd.DataFrame(records)
