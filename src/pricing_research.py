"""Conservative pricing research primitives; not a certified arbitrage-free fit."""

import numpy as np
import pandas as pd
from scipy.stats import norm


def black76_price(forward, strike, time_to_expiry, volatility, risk_free_rate, option_type):
    f, k, t, v, r = np.broadcast_arrays(*[np.asarray(a, dtype=float) for a in (
        forward, strike, time_to_expiry, volatility, risk_free_rate
    )])
    call = np.asarray(option_type) == "CE"
    put = np.asarray(option_type) == "PE"
    valid = np.isfinite(f + k + t + v + r) & (f > 0) & (k > 0) & (t >= 0) & (v >= 0) & (call | put)
    discount = np.exp(-r * t)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(f / k) + .5 * v * v * t) / (v * np.sqrt(t))
        d2 = d1 - v * np.sqrt(t)
        price = discount * np.where(call, f * norm.cdf(d1) - k * norm.cdf(d2),
                                    k * norm.cdf(-d2) - f * norm.cdf(-d1))
    deterministic = discount * np.where(call, np.maximum(f - k, 0), np.maximum(k - f, 0))
    price = np.where((v == 0) | (t == 0), deterministic, price)
    result = np.where(valid, price, np.nan)
    return float(result) if result.ndim == 0 else result


def parity_forward_excluding_strike(pairs, target_strike, min_pairs=3):
    """Robust forward estimate without either target-strike option's price.

    Pairs must be contemporaneous call/put mids from one expiry. Candles can be
    used for diagnostics only; they do not establish simultaneous executable mids.
    """
    for name in ("timestamp", "expiry"):
        if name in pairs and pairs[name].nunique(dropna=False) != 1:
            raise ValueError("Forward estimates must isolate timestamp and expiry")
    peers = pairs.loc[~np.isclose(pairs.strike, target_strike)].copy()
    f = peers.strike + np.exp(peers.risk_free_rate * peers.time_to_expiry) * (peers.call_mid - peers.put_mid)
    valid = (np.isfinite(f) & f.gt(0) & peers.call_mid.ge(0) & peers.put_mid.ge(0)
             & peers.time_to_expiry.gt(0))
    f = f[valid]
    return float(np.median(f)) if len(f) >= min_pairs else np.nan


def robust_loo_total_variance(group, neighbors=7, max_log_distance=.03):
    """Robust local quadratic variance reference excluding each target row.

    No edge extrapolation. This is a smoother benchmark candidate, not an SVI
    implementation or a guarantee of butterfly/calendar arbitrage constraints.
    """
    for name in ("timestamp", "expiry", "option_type"):
        if name in group and group[name].nunique(dropna=False) != 1:
            raise ValueError("Reference surface must isolate timestamp, expiry and option type")
    coordinate = "log_forward_moneyness" if "log_forward_moneyness" in group else "log_moneyness"
    x = pd.to_numeric(group[coordinate], errors="coerce").to_numpy(float)
    iv = pd.to_numeric(group.implied_volatility, errors="coerce").to_numpy(float)
    t = pd.to_numeric(group.time_to_expiry, errors="coerce").to_numpy(float)
    valid = np.isfinite(x + iv + t) & (iv > 0) & (iv <= 5) & (t > 0)
    w = iv * iv * t
    predictions = np.full(len(group), np.nan)
    methods = np.full(len(group), "insufficient_or_unbracketed", dtype=object)
    for i, target in enumerate(x):
        peers = np.flatnonzero(valid & (np.arange(len(group)) != i) & (np.abs(x - target) <= max_log_distance))
        peers = peers[np.argsort(np.abs(x[peers] - target))[:neighbors]]
        if not np.isfinite(target) or not np.isfinite(t[i]) or t[i] <= 0 or len(peers) < 4:
            continue
        if not (x[peers].min() <= target <= x[peers].max()) or np.unique(x[peers]).size < 3:
            continue
        scale = max(float(np.std(x[peers])), .001)
        z = (x[peers] - target) / scale
        design = np.column_stack([np.ones(len(peers)), z, z * z])
        base_weights = 1 / (1 + z * z)
        weights = base_weights.copy()
        for _ in range(4):
            normal = design.T @ (weights[:, None] * design) + np.diag([1e-12, 1e-9, 1e-7])
            coefficients = np.linalg.solve(normal, design.T @ (weights * w[peers]))
            residual = w[peers] - design @ coefficients
            robust_scale = max(float(np.median(np.abs(residual - np.median(residual))) * 1.4826), 1e-9)
            weights = base_weights * np.minimum(1, 1.5 * robust_scale / np.maximum(np.abs(residual), 1e-12))
        prediction = float(coefficients[0])
        if np.isfinite(prediction) and prediction > 0:
            predictions[i] = np.sqrt(prediction / t[i])
            methods[i] = "robust_local_variance_loo"
    result = group.copy()
    result["research_reference_iv"] = predictions
    result["research_reference_method"] = methods
    return result


def static_price_diagnostics(strikes, prices, option_type, discount_factor=1., tolerance=1e-7):
    k, p = np.asarray(strikes, float), np.asarray(prices, float)
    valid = np.isfinite(k + p)
    order = np.argsort(k[valid])
    k, p = k[valid][order], p[valid][order]
    distinct = np.diff(k) > 0
    slopes = np.diff(p)[distinct] / np.diff(k)[distinct]
    if option_type == "CE":
        vertical = (slopes > tolerance) | (slopes < -discount_factor - tolerance)
    elif option_type == "PE":
        vertical = (slopes < -tolerance) | (slopes > discount_factor + tolerance)
    else:
        raise ValueError("option_type must be CE or PE")
    return {"observations": len(k), "vertical_spread_violations": int(vertical.sum()),
            "convexity_violations": int((np.diff(slopes) < -tolerance).sum())}
