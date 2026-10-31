"""Batch equivalent of the scalar BS/IV/Greeks pipeline for staged rebuilds."""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from config import RISK_FREE_RATE, IV_MIN, IV_MAX


def bs_arrays(spot, strike, time, vol, rate, call):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        root = np.sqrt(time)
        d1 = (np.log(spot / strike) + (rate + .5 * vol * vol) * time) / (vol * root)
        d2 = d1 - vol * root
        discounted = strike * np.exp(-rate * time)
        price = np.where(call, spot * ndtr(d1) - discounted * ndtr(d2),
                         discounted * ndtr(-d2) - spot * ndtr(-d1))
    return price


def batch_pricing_features(frame):
    s, k, t, p = [frame[c].to_numpy(float) for c in ("nifty_spot", "strike", "time_to_expiry", "close")]
    call = frame.option_type.eq("CE").to_numpy()
    kind_ok = frame.option_type.isin(["CE", "PE"]).to_numpy()
    r = RISK_FREE_RATE
    status = np.full(len(frame), "pending", dtype=object)

    def reject(mask, name):
        status[(status == "pending") & mask] = name

    reject(~np.isfinite(p), "invalid_market_price")
    reject(~np.isfinite(s), "invalid_spot")
    reject(~np.isfinite(k), "invalid_strike")
    reject(~np.isfinite(t), "invalid_time")
    reject(p <= 0, "zero_or_negative_price")
    reject((s <= 0) | (k <= 0), "invalid_parameters")
    reject(t <= 0, "expired")
    reject(~kind_ok, "invalid_bounds")
    with np.errstate(invalid="ignore", over="ignore"):
        discounted = k * np.exp(-r * t)
        lower = np.maximum(np.where(call, s - discounted, discounted - s), 0.)
        upper = np.where(call, s, discounted)
    reject(p < lower, "below_lower_bound")
    reject(p > upper, "above_upper_bound")
    lo = np.full(len(frame), IV_MIN)
    hi = np.full(len(frame), IV_MAX)
    low_error = bs_arrays(s, k, t, lo, r, call) - p
    high_error = bs_arrays(s, k, t, hi, r, call) - p
    reject(low_error * high_error > 0, "solver_not_bracketed")
    active = status == "pending"
    ix = np.flatnonzero(active)
    a, b = lo[ix], hi[ix]
    # 40 halvings resolve sigma below 5e-12. Exact boundary roots follow Brent.
    for _ in range(40):
        mid = (a + b) * .5
        error = bs_arrays(s[ix], k[ix], t[ix], mid, r, call[ix]) - p[ix]
        b = np.where(error >= 0, mid, b)
        a = np.where(error < 0, mid, a)
    iv = np.full(len(frame), np.nan)
    iv[ix] = (a + b) * .5
    iv[active & (low_error == 0)] = IV_MIN
    iv[active & (high_error == 0)] = IV_MAX
    status[active] = "valid"
    theoretical = bs_arrays(s, k, t, iv, r, call)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        root = np.sqrt(t)
        d1 = (np.log(s / k) + (r + .5 * iv * iv) * t) / (iv * root)
        d2 = d1 - iv * root
        density = np.exp(-.5 * d1 * d1) / np.sqrt(2 * np.pi)
        delta = ndtr(d1) - (~call).astype(float)
        gamma = density / (s * iv * root)
        vega = s * density * root / 100.
        theta = (-s * density * iv / (2 * root)
                 + np.where(call, -r * discounted * ndtr(d2), r * discounted * ndtr(-d2))) / 365.
        deviation = p - theoretical
        pct = np.where(theoretical > 1e-8, deviation / theoretical, np.nan)
    result = pd.DataFrame({"implied_volatility": iv, "iv_status": status,
                           "theoretical_price": theoretical, "price_deviation": deviation,
                           "price_deviation_pct": pct, "delta": delta, "gamma": gamma,
                           "vega": vega, "theta": theta}, index=frame.index)
    result.loc[~active, result.columns.difference(["iv_status"])] = np.nan
    return result


def batch_reference_prices(frame):
    s, k, t, v, r = [frame[c].to_numpy(float) for c in
                      ("nifty_spot", "strike", "time_to_expiry", "reference_iv", "risk_free_rate")]
    valid = (np.isfinite(np.column_stack([s, k, t, v, r])).all(axis=1)
             & (s > 0) & (k > 0) & (t > 0) & (v > 0)
             & frame.option_type.isin(["CE", "PE"]).to_numpy())
    prices = bs_arrays(s, k, t, v, r, frame.option_type.eq("CE").to_numpy())
    return np.where(valid & np.isfinite(prices), prices, np.nan)
