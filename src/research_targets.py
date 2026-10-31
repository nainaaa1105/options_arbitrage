"""Explicit research targets and session censoring; never predictor columns."""

import numpy as np
import pandas as pd


def session_horizon_eligible(timestamps, horizon_minutes=30, bar_minutes=5,
                             session_close_hour=15, session_close_minute=30):
    """Require a completed signal bar plus the holding horizon before close."""
    if horizon_minutes <= 0 or bar_minutes < 0:
        raise ValueError("Invalid horizon or bar duration")
    ts = pd.to_datetime(timestamps)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("Asia/Kolkata")
    else:
        ts = ts.dt.tz_convert("Asia/Kolkata")
    end = ts.dt.normalize() + pd.Timedelta(hours=session_close_hour, minutes=session_close_minute)
    opened = ts.dt.normalize() + pd.Timedelta(hours=9, minutes=15)
    return ts.notna() & ts.ge(opened) & ((ts + pd.Timedelta(minutes=bar_minutes + horizon_minutes)) <= end)


def build_path_targets(context, horizon_minutes=30, step_minutes=5,
                       min_deviation=1.0, reduction_threshold=.5):
    """Compare endpoint, first passage and two-bar sustained convergence.

    All three binary targets require the same complete path. Missing outcomes
    remain nullable, never negative. The path is sampled candle closes, so an
    unobserved intrabar touch is not inferred from OHLC or interpolation.
    """
    if horizon_minutes < 2 * step_minutes or horizon_minutes % step_minutes:
        raise ValueError("Horizon must be a multiple of step_minutes with at least two steps")
    source = context[["instrument_key", "timestamp", "expiry",
                      "reference_price_deviation", "signal_eligible"]].copy()
    source["timestamp"] = pd.to_datetime(source.timestamp)
    if source.duplicated(["instrument_key", "timestamp"]).any():
        raise ValueError("Duplicate contract/timestamp outcomes")
    eligible = source.signal_eligible.fillna(False).astype(bool)
    current = source.reference_price_deviation.abs()
    candidates = source.loc[eligible & np.isfinite(current) & current.ge(min_deviation)].copy()
    candidates = candidates.reset_index(drop=True)
    key = pd.MultiIndex.from_frame(source[["instrument_key", "timestamp"]])
    deviations = source.reference_price_deviation.abs().to_numpy(float)
    expiries = pd.to_datetime(source.expiry).to_numpy()
    paths = []
    same_expiry = np.ones(len(candidates), dtype=bool)
    candidate_expiry = pd.to_datetime(candidates.expiry).to_numpy()
    for minute in range(step_minutes, horizon_minutes + 1, step_minutes):
        future_key = pd.MultiIndex.from_arrays([
            candidates.instrument_key,
            candidates.timestamp + pd.Timedelta(minutes=minute),
        ])
        locations = key.get_indexer(future_key)
        values = np.full(len(candidates), np.nan)
        found = locations >= 0
        values[found] = deviations[locations[found]]
        same_expiry[found] &= expiries[locations[found]] == candidate_expiry[found]
        same_expiry[~found] = False
        paths.append(values)
    path = np.column_stack(paths)
    has_horizon = session_horizon_eligible(candidates.timestamp, horizon_minutes, step_minutes).to_numpy()
    complete = np.isfinite(path).all(axis=1) & same_expiry & has_horizon
    absolute = candidates.reference_price_deviation.abs().to_numpy()
    reduced = path <= absolute[:, None] * (1 - reduction_threshold)
    targets = {
        "endpoint": reduced[:, -1],
        "first_passage": reduced.any(axis=1),
        "sustained_two_bars": (reduced[:, 1:] & reduced[:, :-1]).any(axis=1),
    }
    result = candidates[["instrument_key", "timestamp", "expiry"]].copy()
    result["complete_path"] = complete
    result["observed_path_points"] = np.isfinite(path).sum(axis=1)
    result["expected_path_points"] = path.shape[1]
    result["censor_reason"] = np.select(
        [~has_horizon, has_horizon & ~complete],
        ["session_end", "missing_or_invalid_path"], default="observed",
    )
    result["label_end_available_at"] = candidates.timestamp + pd.Timedelta(minutes=horizon_minutes + step_minutes)
    for name, target in targets.items():
        result[name] = pd.array(np.where(complete, target.astype(float), np.nan), dtype="Int64")
    result["deviation_reduction"] = np.where(complete, (absolute - path[:, -1]) / absolute, np.nan)
    result["log_future_current_deviation"] = np.where(
        complete, np.log((path[:, -1] + .05) / (absolute + .05)), np.nan
    )
    return result


def executable_cashflow_targets(fills):
    """Derive useful profitability targets from supplied executable cashflows.

    Hedge quantity is the signed number of traded hedge units, not index delta
    assumed executable. Spread is already in bid/ask fills; costs here exclude
    that spread and include brokerage/taxes/slippage not already in the fills.
    This function does not invent fills, margins or transaction costs.
    """
    required = ["signal_available_at", "entry_time", "exit_time", "option_side",
                "option_quantity", "option_entry_price", "option_exit_price",
                "hedge_quantity", "hedge_entry_price", "hedge_exit_price",
                "total_costs_rupees", "cost_basis", "hedge_instrument_key"]
    missing = sorted(set(required) - set(fills.columns))
    if missing:
        raise ValueError(f"Executable target inputs are missing: {missing}")
    d = fills.copy()
    for name in ("signal_available_at", "entry_time", "exit_time"):
        d[name] = pd.to_datetime(d[name], utc=True)
    numeric = ["option_side", "option_quantity", "option_entry_price", "option_exit_price",
               "hedge_quantity", "hedge_entry_price", "hedge_exit_price", "total_costs_rupees"]
    values = d[numeric].apply(pd.to_numeric, errors="coerce")
    d[numeric] = values
    finite = np.isfinite(values).all(axis=1)
    valid = (finite & d.option_side.isin([-1, 1]) & d.option_quantity.gt(0)
             & d.option_entry_price.gt(0) & d.option_exit_price.gt(0)
             & d.total_costs_rupees.ge(0)
             & (d.hedge_quantity.eq(0) | (d.hedge_entry_price.gt(0) & d.hedge_exit_price.gt(0)))
             & (d.hedge_quantity.eq(0) | (d.hedge_instrument_key.fillna("").astype(str).ne("")
                                         & ~d.hedge_instrument_key.fillna("").astype(str).str.startswith("NSE_INDEX|")))
             & d.cost_basis.isin(["measured", "configured_proxy"])
             & d.entry_time.ge(d.signal_available_at) & d.exit_time.gt(d.entry_time))
    gross = (d.option_side * d.option_quantity * (d.option_exit_price - d.option_entry_price)
             + d.hedge_quantity * (d.hedge_exit_price - d.hedge_entry_price))
    net = gross - d.total_costs_rupees
    result = pd.DataFrame({"label_valid": valid, "net_pnl_rupees": net.where(valid),
                           "cost_basis": d.cost_basis})
    result["profitable"] = pd.array(np.where(valid, net.gt(0).astype(float), np.nan), dtype="Int64")
    return result
