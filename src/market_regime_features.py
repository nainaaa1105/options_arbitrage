"""Causal finished-bar spot and volatility regime inputs for research models."""

import numpy as np
import pandas as pd


SPOT_FEATURES = ["spot_return_5m", "spot_return_15m", "spot_return_30m",
                 "spot_realized_vol_30m", "spot_realized_vol_60m",
                 "spot_session_return", "spot_session_range", "spot_opening_gap"]
VIX_FEATURES = ["vix_level", "vix_return_5m", "iv_minus_vix", "reference_iv_minus_vix"]


def market_table(spot, vix):
    d = spot.sort_values("timestamp").copy()
    d["timestamp"] = pd.to_datetime(d.timestamp)
    if d.timestamp.duplicated().any():
        raise ValueError("Duplicate spot timestamps")
    day = d.timestamp.dt.normalize()
    group = d.groupby(day, sort=False)
    for bars, label in ((1, "5m"), (3, "15m"), (6, "30m")):
        previous = group.close.shift(bars)
        elapsed = d.timestamp - group.timestamp.shift(bars)
        d["spot_return_" + label] = np.log(d.close / previous).where(elapsed.eq(pd.Timedelta(minutes=5*bars)))
    returns = d.spot_return_5m
    for bars, label in ((6, "30m"), (12, "60m")):
        d["spot_realized_vol_" + label] = returns.groupby(day).transform(
            lambda values: values.rolling(bars, min_periods=bars).std(ddof=1)) * np.sqrt(75 * 252)
    first_open = group.open.transform("first")
    d["spot_session_return"] = d.close / first_open - 1
    d["spot_session_range"] = (group.high.cummax() - group.low.cummin()) / first_open
    daily = d.groupby(day, sort=True).agg(first_open=("open", "first"), close=("close", "last"))
    gap = daily.first_open / daily.close.shift(1) - 1
    d["spot_opening_gap"] = day.map(gap)
    v = vix.sort_values("timestamp").copy()
    v["timestamp"] = pd.to_datetime(v.timestamp)
    if v.timestamp.duplicated().any():
        raise ValueError("Duplicate VIX timestamps")
    vday = v.timestamp.dt.normalize()
    prior = v.groupby(vday).close.shift(1)
    elapsed = v.timestamp - v.groupby(vday).timestamp.shift(1)
    v["vix_level"] = v.close / 100.
    v["vix_return_5m"] = np.log(v.close / prior).where(elapsed.eq(pd.Timedelta(minutes=5)))
    return d[["timestamp", *SPOT_FEATURES]].merge(
        v[["timestamp", "vix_level", "vix_return_5m"]], on="timestamp", how="left", validate="one_to_one")


def augment(frame, market):
    attrs = frame.attrs.copy()
    out = frame.merge(market, on="timestamp", how="left", validate="many_to_one")
    out["iv_minus_vix"] = out.implied_volatility - out.vix_level
    out["reference_iv_minus_vix"] = out.reference_iv - out.vix_level
    out.attrs.update(attrs)
    return out
