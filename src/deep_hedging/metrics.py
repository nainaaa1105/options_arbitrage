"""Risk reports from the marked-to-market ledger, including zero-trade dates."""

import numpy as np
import pandas as pd

from .baselines import STRATEGIES


def empirical_loss_risk(pnl, alpha=.95):
    values = np.asarray(pnl, dtype=float)
    if not 0 < alpha < 1 or not np.isfinite(values).all():
        raise ValueError("Invalid risk inputs")
    if not len(values):
        return np.nan, np.nan
    loss = -values
    var = float(np.quantile(loss, alpha, method="higher"))
    cvar = var + float(np.maximum(loss - var, 0).mean()) / (1-alpha)
    return var, cvar


def daily_equity(marks, dates, initial_capital, strategy):
    calendar = pd.Index(sorted(set(dates)), name="trade_date")
    m = marks.loc[marks.strategy.eq(strategy)].sort_values("timestamp", kind="stable").copy()
    m["trade_date"] = pd.to_datetime(m.timestamp).dt.date
    observed = m.groupby("trade_date").equity.last()
    equity = observed.reindex(calendar).ffill().fillna(initial_capital)
    d = equity.rename("equity").reset_index()
    d["strategy"] = strategy
    previous = d.equity.shift(1).fillna(initial_capital)
    d["net_pnl"] = d.equity - previous
    d["daily_return"] = np.where(previous > 0, d.net_pnl / previous, np.nan)
    peak = np.maximum.accumulate(np.r_[initial_capital, d.equity.to_numpy()])[1:]
    d["peak_equity"] = peak
    d["drawdown_rupees"] = peak - d.equity
    d["drawdown_fraction"] = np.where(peak > 0, (peak-d.equity) / peak, np.nan)
    return d


def make_reports(episodes, trades, marks, dates, config, strategies=None):
    rows, daily = [], []
    for strategy in (STRATEGIES if strategies is None else strategies):
        e = episodes.loc[episodes.strategy.eq(strategy)]
        t = trades.loc[trades.strategy.eq(strategy)]
        m = marks.loc[marks.strategy.eq(strategy)].sort_values("timestamp", kind="stable")
        d = daily_equity(marks, dates, config.initial_capital, strategy)
        daily.append(d)
        returns = d.daily_return.to_numpy(float)
        excess = returns - config.risk_free_rate / 252
        std = float(np.std(returns, ddof=1)) if len(returns) > 1 else np.nan
        downside = float(np.sqrt(np.mean(np.minimum(excess, 0)**2)))
        var, cvar = empirical_loss_risk(e.net_pnl)
        peak = np.maximum.accumulate(np.r_[config.initial_capital, m.equity.to_numpy()])[1:]
        future = t.loc[t.asset.eq("future") & t.executed_order]
        rows.append({"strategy": strategy, "episodes": len(e), "calendar_sessions": len(d),
                     "initial_capital": config.initial_capital, "ending_equity": float(d.equity.iloc[-1]),
                     "gross_pnl": float(e.gross_pnl.sum()), "net_pnl": float(e.net_pnl.sum()),
                     "net_return": float(e.net_pnl.sum()/config.initial_capital),
                     "fees": float(e.fees.sum()), "execution_impact": float(e.execution_impact.sum()),
                     "funding": float(e.funding.sum()), "win_rate": float(e.net_pnl.gt(0).mean()),
                     "mean_episode_pnl": float(e.net_pnl.mean()),
                     "episode_pnl_std": float(e.net_pnl.std(ddof=1)),
                     "loss_var_95": var, "loss_cvar_95": cvar, "worst_episode_pnl": float(e.net_pnl.min()),
                     "daily_sharpe_annualized": float(excess.mean()/std*np.sqrt(252)) if std > 0 else np.nan,
                     "daily_sortino_annualized": float(excess.mean()/downside*np.sqrt(252)) if downside > 0 else np.nan,
                     "max_daily_drawdown_rupees": float(d.drawdown_rupees.max()),
                     "max_daily_drawdown_fraction": float(d.drawdown_fraction.max()),
                     "max_intraday_drawdown_rupees": float(np.max(peak-m.equity.to_numpy())),
                     "future_orders": len(future), "future_turnover": float(future.turnover.sum()),
                     "peak_margin_proxy": float((m.futures_margin+m.short_option_margin).max()),
                     "minimum_free_cash": float(m.free_cash.min()),
                     "mean_absolute_observed_net_delta": float(m.net_spot_delta.abs().mean()),
                     "margin_constrained_actions": int(e.margin_constrained_actions.sum()),
                     "forced_exits": int(e.forced_exits.sum()),
                     "zero_recovery_writeoffs": int(e.zero_recovery_writeoffs.sum()),
                     "invalid_delta_events": int(e.invalid_delta_events.sum()),
                     "max_reconciliation_error": float(e.reconciliation_error.abs().max()),
                     "cost_basis": config.cost_basis})
        if not np.isclose(d.net_pnl.sum(), e.net_pnl.sum(), atol=1e-5):
            raise AssertionError("Daily equity and episode P&L do not reconcile")
    return pd.DataFrame(rows), pd.concat(daily, ignore_index=True)


def paired_comparison(episodes, dates, repeats=1000):
    matrix = episodes.pivot(index="episode_id", columns="strategy", values="net_pnl")
    if matrix[list(STRATEGIES)].isna().any().any():
        raise AssertionError("Baseline episode universe differs")
    day = episodes.drop_duplicates("episode_id").set_index("episode_id").entry_time
    day = pd.to_datetime(day).dt.date.reindex(matrix.index)
    members = [np.flatnonzero(day.to_numpy() == date) for date in sorted(set(dates))]
    rng = np.random.default_rng(42)
    samples = [np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))]) for _ in range(repeats)]
    base = matrix.unhedged.to_numpy()
    rows = []
    for name in STRATEGIES[1:]:
        values = matrix[name].to_numpy()
        delta = values - base
        _, base_cvar = empirical_loss_risk(base)
        _, current_cvar = empirical_loss_risk(values)
        means, risk_diffs = [], []
        for idx in samples:
            if len(idx):
                means.append(float(delta[idx].mean()))
                risk_diffs.append(empirical_loss_risk(values[idx])[1] - empirical_loss_risk(base[idx])[1])
        rows.append({"strategy": name, "reference": "unhedged", "paired_episodes": len(matrix),
                     "mean_pnl_difference": float(delta.mean()),
                     "mean_difference_lower_95": float(np.quantile(means, .025)),
                     "mean_difference_upper_95": float(np.quantile(means, .975)),
                     "cvar_difference": current_cvar - base_cvar,
                     "cvar_difference_lower_95": float(np.quantile(risk_diffs, .025)),
                     "cvar_difference_upper_95": float(np.quantile(risk_diffs, .975)),
                     "interpretation": "Higher P&L difference is better; lower loss-CVaR difference is better",
                     "uncertainty": "Trading-session cluster bootstrap; not a future-performance guarantee"})
    return pd.DataFrame(rows)
