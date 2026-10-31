"""Intraday, whole-lot hedge accounting with a common episode universe."""

from dataclasses import replace
import math

import numpy as np
import pandas as pd

from .accounting import FuturesPosition
from .baselines import STRATEGIES, target_lots
from .config import HedgeConfig
from .costs import execution_price, transaction_fees


def _future_change(book, cash, change, mark, tick, timestamp, config):
    projected = replace(book)
    if change == 0:
        return projected, cash, None
    fill = execution_price(mark, change, tick, config.future_half_spread, config.future_slippage)
    fees = transaction_fees("future", change, fill, timestamp, config)
    realized = projected.trade(change, fill)
    return projected, cash + realized - fees["total"], {
        "asset": "future", "quantity_change": change, "mark": mark, "fill": fill,
        "execution_impact": change * (fill - mark), "realized_future_pnl": realized,
        "fees": fees["total"], "turnover": abs(change) * fill,
        "executed_order": True, **{f"fee_{k}": v for k, v in fees.items() if k != "total"}}


def margin_state(cash, option_quantity, option_mark, book, future_mark, option_lots, config):
    unrealized = book.unrealized(future_mark)
    equity = cash + option_quantity * option_mark + unrealized
    futures_margin = abs(book.quantity) * future_mark * config.future_margin_fraction
    short_margin = option_lots * config.short_option_margin_per_lot if option_quantity < 0 else 0.
    # Unrealized futures gains are not assumed withdrawable intraday.
    free_cash = cash + min(unrealized, 0.) - futures_margin - short_margin
    reserve = max(equity, 0.) * (1 - config.capital_fraction)
    return {"cash": cash, "option_value": option_quantity * option_mark,
            "future_unrealized": unrealized, "equity": equity,
            "futures_margin": futures_margin, "short_option_margin": short_margin,
            "free_cash": free_cash, "cash_buffer_required": reserve}


def run_episode(episode, path, strategy, starting_cash, config: HedgeConfig, policy=None):
    if strategy not in STRATEGIES and policy is None:
        raise ValueError("Unknown hedger")
    d = path.sort_values("timestamp").reset_index(drop=True)
    if len(d) != config.horizon_minutes // config.bar_minutes + 1:
        raise ValueError("Incomplete episode path")
    ts = pd.to_datetime(d.timestamp)
    observed = pd.to_datetime(d.observed_at)
    if ts.duplicated().any() or not ts.diff().iloc[1:].eq(pd.Timedelta(minutes=config.bar_minutes)).all():
        raise ValueError("Episode path must contain exact consecutive bars")
    if ts.dt.tz is None or observed.dt.tz is None or observed.isna().any() or observed.gt(ts).any():
        raise ValueError("Future/invalid observations in hedge state")
    if not observed.eq(ts).all():
        raise ValueError("Part 1 requires the immediately completed observation at each boundary")
    if ts.dt.date.nunique() != 1:
        raise ValueError("Part 1 episodes must remain intraday")
    if d.future_instrument_key.nunique() != 1 or d.future_instrument_key.str.startswith("NSE_INDEX|").any():
        raise ValueError("Invalid hedge instrument or intraday roll")
    for col in ("option_mark", "future_mark", "future_lot_size", "future_tick_size"):
        if (~np.isfinite(d[col]) | d[col].le(0)).any():
            raise ValueError(f"Invalid path prices/units: {col}")
    if d.future_lot_size.nunique() != 1 or d.future_tick_size.nunique() != 1:
        raise ValueError("Contract units changed during episode")
    if d.future_lot_size.mod(1).ne(0).any():
        raise ValueError("Fractional futures lot size")
    raw_quantity = episode["option_quantity"]
    if not np.isfinite(raw_quantity) or int(raw_quantity) != raw_quantity:
        raise ValueError("Option quantity must be an integer number of units")
    quantity = int(raw_quantity)
    if quantity != config.option_side * config.option_lots * int(episode["option_lot_size"]):
        raise ValueError("Option position disagrees with experiment configuration")
    if starting_cash <= 0 or not math.isfinite(starting_cash):
        raise ValueError("Positive starting capital required")
    cash = float(starting_cash)
    book = FuturesPosition()
    option_position = 0
    trades, marks = [], []
    fees_total = impact_total = funding_total = future_gross = 0.
    margin_constraints = invalid_delta_events = 0
    force_closed = write_offs = 0
    previous_future = float(d.future_mark.iloc[0])
    previous_time = ts.iloc[0]
    funding_base = 0.
    first_option = float(d.option_mark.iloc[0])

    def option_trade(change, row, closing=False):
        nonlocal cash, option_position, fees_total, impact_total, write_offs
        try:
            fill = execution_price(row.option_mark, change, config.option_tick_size,
                                   config.option_half_spread, config.option_slippage)
            fees = transaction_fees("option", change, fill, row.timestamp, config)
            executed = True
        except ValueError:
            if not closing or change >= 0:
                raise
            # Preserve the loss rather than discard a bad-outcome episode.
            # Explicit conservative zero recovery, not a claimed market fill.
            fill = 0.
            fees = transaction_fees("option", 0, 1., row.timestamp, config)
            write_offs += 1
            executed = False
        cash -= change * fill + fees["total"]
        option_position += change
        impact = change * (fill - row.option_mark)
        fees_total += fees["total"]
        impact_total += impact
        trades.append({"timestamp": row.timestamp, "asset": "option", "quantity_change": change,
                       "mark": row.option_mark, "fill": fill, "execution_impact": impact,
                       "realized_future_pnl": 0., "fees": fees["total"], "turnover": abs(change) * fill,
                       "executed_order": executed, "event": "trade" if executed else "zero_recovery_writeoff",
                       **{f"fee_{k}": v for k, v in fees.items() if k != "total"}})

    def future_trade(change, row):
        nonlocal book, cash, fees_total, impact_total
        book, cash, record = _future_change(book, cash, change, row.future_mark,
                                           row.future_tick_size, row.timestamp, config)
        if record:
            fees_total += record["fees"]
            impact_total += record["execution_impact"]
            trades.append({"timestamp": row.timestamp, "event": "trade", **record})

    exit_reason = "scheduled_exit"
    for i, row in d.iterrows():
        future_gross += book.quantity * (row.future_mark - previous_future)
        elapsed_years = (row.timestamp - previous_time).total_seconds() / (365 * 86400.)
        funding = funding_base * config.funding_rate * elapsed_years
        cash -= funding
        funding_total += funding
        if i == 0:
            option_trade(quantity, row)
        account = margin_state(cash, option_position, row.option_mark, book, row.future_mark, config.option_lots, config)
        breach = account["equity"] <= 0 or account["free_cash"] < account["cash_buffer_required"] - 1e-8
        terminal = i == len(d) - 1 or breach
        policy_reason = "terminal"
        if terminal:
            if breach:
                force_closed = 1
                exit_reason = "forced_cash_or_margin_exit"
            future_trade(-book.quantity, row)
            option_trade(-option_position, row, closing=True)
        else:
            state = row.to_dict()
            state["option_quantity"] = quantity
            state.update(account)
            state["remaining_steps"] = len(d)-1-i
            current_lots = book.quantity // int(row.future_lot_size)
            desired, policy_reason = (target_lots(strategy, state, current_lots, i == 0, config)
                                      if policy is None else policy(state, current_lots, i == 0, config))
            if int(desired) != desired or abs(desired) > config.max_future_lots:
                raise ValueError("Policy must return bounded integer futures lots")
            desired = int(desired)
            invalid_delta_events += int(policy_reason == "invalid_delta_hold")
            candidates = [int(np.sign(desired)) * n for n in range(abs(desired), -1, -1)]
            chosen = None
            for lots in candidates:
                change = lots * int(row.future_lot_size) - book.quantity
                preview, proposed_cash, _ = _future_change(book, cash, change, row.future_mark,
                                                          row.future_tick_size, row.timestamp, config)
                proposed = margin_state(proposed_cash, option_position, row.option_mark, preview,
                                         row.future_mark, config.option_lots, config)
                if proposed["equity"] > 0 and proposed["free_cash"] >= proposed["cash_buffer_required"] - 1e-8:
                    chosen = lots
                    break
            if chosen is None:
                chosen = current_lots
            if chosen != desired:
                margin_constraints += 1
                policy_reason = "margin_constrained_target"
            future_trade(chosen * int(row.future_lot_size) - book.quantity, row)
        account = margin_state(cash, option_position, row.option_mark, book, row.future_mark, config.option_lots, config)
        delta = row.delta if np.isfinite(row.delta) else np.nan
        marks.append({"timestamp": row.timestamp, "observed_at": row.observed_at, **account,
                      "option_quantity": option_position, "future_quantity": book.quantity,
                      "future_lots": book.quantity // int(row.future_lot_size), "option_delta_observed": delta,
                      "net_spot_delta": option_position * delta + book.quantity * row.future_spot_delta if option_position else 0.,
                      "policy_reason": policy_reason, "cumulative_fees": fees_total,
                      "cumulative_execution_impact": impact_total, "cumulative_funding": funding_total})
        previous_future, previous_time = row.future_mark, row.timestamp
        funding_base = max(option_position, 0) * first_option + account["futures_margin"] + account["short_option_margin"]
        if terminal:
            break
    option_gross = quantity * (row.option_mark - first_option)
    net = cash - starting_cash
    calculated = option_gross + future_gross - fees_total - impact_total - funding_total
    if not np.isclose(net, calculated, atol=1e-6, rtol=1e-10) or book.quantity != 0 or option_position != 0:
        raise AssertionError("Self-financing cash/P&L reconciliation failed")
    result = {**dict(episode), "strategy": strategy, "actual_exit_time": row.timestamp,
              "starting_equity": starting_cash, "ending_equity": cash,
              "option_gross_pnl": option_gross, "future_gross_pnl": future_gross,
              "gross_pnl": option_gross + future_gross, "fees": fees_total,
              "execution_impact": impact_total, "funding": funding_total, "net_pnl": net,
              "exit_reason": exit_reason, "forced_exits": force_closed, "zero_recovery_writeoffs": write_offs,
              "margin_constrained_actions": margin_constraints, "invalid_delta_events": invalid_delta_events,
              "reconciliation_error": net - calculated, "cost_basis": config.cost_basis}
    for collection in (trades, marks):
        for record in collection:
            record.update({"episode_id": episode["episode_id"], "strategy": strategy})
    return result, trades, marks


def compare_baselines(episodes, paths, config):
    capital = {name: config.initial_capital for name in STRATEGIES}
    summaries, orders, equity, skipped = [], [], [], []
    groups = {key: d for key, d in paths.groupby("episode_id", sort=False)}
    previous_end = None
    for _, episode in episodes.sort_values("entry_time").iterrows():
        d = groups[episode.episode_id].sort_values("timestamp")
        if previous_end is not None and episode.entry_time < previous_end:
            raise ValueError("Portfolio episodes overlap; shared-account mode is sequential")
        first = d.iloc[0]
        try:
            fill = execution_price(first.option_mark, episode.option_quantity, config.option_tick_size,
                                   config.option_half_spread, config.option_slippage)
            option_fee = transaction_fees("option", episode.option_quantity, fill, first.timestamp, config)["total"]
            maximum_units = config.max_future_lots * int(first.future_lot_size)
            # Conservative reserve for all strategies, including a worst-side
            # futures opening order. No future prices or outcomes enter admission.
            worst_fee = transaction_fees("future", -maximum_units, first.future_mark, first.timestamp, config)["total"]
            reserve = (max(episode.option_quantity, 0) * fill + option_fee
                       + maximum_units * first.future_mark * config.future_margin_fraction + worst_fee
                       + maximum_units * (config.future_half_spread + config.future_slippage + first.future_tick_size)
                       + (config.option_lots * config.short_option_margin_per_lot if episode.option_quantity < 0 else 0.))
        except ValueError:
            skipped.append({"episode_id": episode.episode_id, "reason": "unexecutable_initial_option_proxy"})
            continue
        if reserve > min(capital.values()) * config.capital_fraction:
            skipped.append({"episode_id": episode.episode_id, "reason": "common_initial_capital_limit"})
            continue
        for strategy in STRATEGIES:
            summary, trades, marks = run_episode(episode, d, strategy, capital[strategy], config)
            capital[strategy] = summary["ending_equity"]
            summaries.append(summary)
            orders.extend(trades)
            equity.extend(marks)
        previous_end = episode.exit_time
    return pd.DataFrame(summaries), pd.DataFrame(orders), pd.DataFrame(equity), pd.DataFrame(skipped)
