"""Causal surface, dynamics and staleness features (schema v2.1) for the 30-minute convergence target.

Every value for a row at time t uses only (a) rows of the same option observed at or before t on the same session, or
(b) rows of the whole option surface observed at exactly t. Nothing reads a later bar or any label. The builder works on the
FULL surface table (including rows without a label) so lags are not distorted by label filtering.
"""
import numpy as np
import pandas as pd

KEY = ["instrument_key", "timestamp"]
REQUIRED = ["instrument_key", "timestamp", "trade_date", "expiry", "option_type", "strike", "nifty_spot", "market_price",
            "volume", "open_interest", "log_moneyness", "implied_volatility", "reference_iv", "reference_price_deviation",
            "iv_surface_residual", "parity_residual", "time_to_expiry"]
EPS = 1e-6

GROUPS = {
    "dev_dynamics": ["dev_lag3", "dev_lag4", "dev_lag6", "dev_roll_mean6", "dev_roll_std6", "dev_z6", "dev_ewm", "absdev_mom3",
                     "absdev_mom6", "absdev_frac_of_day_max", "dev_sign_flips6", "dev_same_sign_run"],
    "price_dynamics": ["ret3", "ret6", "ret_vol6", "price_range_pos6"],
    "staleness": ["stale_run", "stale_frac6", "zero_volume_run", "vol_sum3", "vol_ratio_3_12", "oi_change1"],
    "iv_dynamics": ["iv_change3", "iv_change6", "ref_iv_change3", "resid_lag1", "resid_mean3", "parity_lag1", "parity_mean3"],
    "smile": ["smile_slope", "smile_curv", "smile_n", "iv_minus_smile", "atm_iv_own", "put_call_iv_spread"],
    # Cross-expiry term structure is NOT available: collection keeps ~5 sessions per weekly expiry, so one expiry exists per timestamp.
    "smile_dynamics": ["atm_iv_change6", "smile_slope_change6", "atm_iv_vs_open"],
    "cross_section": ["absdev_rank_ts", "volume_rank_expiry", "absdev_x_logvol3"],
}
V2_COLUMNS = [c for cols in GROUPS.values() for c in cols]


def _lags(frame, column, group, n):
    g = frame.groupby(group, sort=False)[column]
    return np.column_stack([frame[column].to_numpy(float)] + [g.shift(k).to_numpy(float) for k in range(1, n)])


def _fit_smile(group):
    x, y = group["log_moneyness"].to_numpy(float), group["implied_volatility"].to_numpy(float)
    ok = np.isfinite(x) & np.isfinite(y) & (y > 0)
    out = {"smile_slope": np.nan, "smile_curv": np.nan, "smile_n": float(ok.sum()), "atm_iv": np.nan}
    if ok.sum() >= 4 and np.ptp(x[ok]) > 1e-6:
        design = np.column_stack([np.ones(ok.sum()), x[ok], x[ok] ** 2])
        coef = np.linalg.lstsq(design, y[ok], rcond=None)[0]
        out.update(smile_slope=coef[1], smile_curv=coef[2], atm_iv=coef[0])
    return out


def build_v2_features(surface):
    missing = [c for c in REQUIRED if c not in surface.columns]
    if missing:
        raise ValueError(f"Surface table is missing: {missing}")
    d = surface[REQUIRED].copy()
    d["timestamp"] = pd.to_datetime(d.timestamp)
    if d.duplicated(KEY).any():
        raise ValueError("Duplicate instrument/timestamp rows")
    d["trade_date"] = pd.to_datetime(d.trade_date).dt.date
    d = d.sort_values(["instrument_key", "trade_date", "timestamp"]).reset_index(drop=True)
    day = ["instrument_key", "trade_date"]
    g = d.groupby(day, sort=False)
    dev, absdev = d.reference_price_deviation, d.reference_price_deviation.abs()
    price = d.market_price

    L = _lags(d, "reference_price_deviation", day, 7)       # t, t-1 ... t-6
    d["dev_lag3"], d["dev_lag4"], d["dev_lag6"] = L[:, 3], L[:, 4], L[:, 6]
    window = L[:, :6]
    with np.errstate(all="ignore"):
        valid = np.isfinite(window).sum(1) >= 3
        mean, std = np.where(valid, np.nanmean(window, 1), np.nan), np.where(valid, np.nanstd(window, 1), np.nan)
        past = L[:, 1:6]
        past_ok = np.isfinite(past).sum(1) >= 3
        pmean, pstd = np.where(past_ok, np.nanmean(past, 1), np.nan), np.where(past_ok, np.nanstd(past, 1), np.nan)
    d["dev_roll_mean6"], d["dev_roll_std6"] = mean, std
    d["dev_z6"] = np.clip((dev.to_numpy() - pmean) / (pstd + 0.05), -6, 6)
    d["dev_ewm"] = g["reference_price_deviation"].transform(lambda s: s.ewm(alpha=.3, adjust=False, ignore_na=True).mean())
    A = np.abs(_lags(d, "reference_price_deviation", day, 7))
    d["absdev_mom3"], d["absdev_mom6"] = A[:, 0] - A[:, 3], A[:, 0] - A[:, 6]
    running_max = g["reference_price_deviation"].transform(lambda s: s.abs().cummax())
    d["absdev_frac_of_day_max"] = absdev / (running_max + EPS)
    signs = np.sign(L[:, :6])
    d["dev_sign_flips6"] = np.nansum(np.abs(np.diff(signs, axis=1)) > 1, axis=1).astype(float)
    flips = np.where(np.isfinite(signs[:, :-1]) & np.isfinite(signs[:, 1:]), (signs[:, :-1] * signs[:, 1:] < 0), 0)
    run = np.ones(len(d))
    for k in range(5):
        run = np.where(flips[:, : k + 1].sum(1) == 0, k + 2, run)
    d["dev_same_sign_run"] = np.minimum(run, np.isfinite(signs).sum(1))      # cannot exceed the bars actually observed

    P = _lags(d, "market_price", day, 7)
    with np.errstate(all="ignore"):
        d["ret3"], d["ret6"] = P[:, 0] / (P[:, 3] + EPS) - 1, P[:, 0] / (P[:, 6] + EPS) - 1
        steps = P[:, :6] / (P[:, 1:7][:, :6] + EPS) - 1
        ok = np.isfinite(steps).sum(1) >= 3
        d["ret_vol6"] = np.where(ok, np.nanstd(np.clip(steps, -1, 1), 1), np.nan)
        hi, lo = np.nanmax(P[:, :6], 1), np.nanmin(P[:, :6], 1)
        d["price_range_pos6"] = np.where(hi > lo, (P[:, 0] - lo) / (hi - lo + EPS), np.nan)
    unchanged = (price.to_numpy() == g["market_price"].shift(1).to_numpy())
    run_id = (~unchanged).cumsum()
    d["stale_run"] = pd.Series(unchanged.astype(int)).groupby([d.instrument_key.to_numpy(), d.trade_date.to_numpy(), run_id]).cumsum().to_numpy()
    changes = np.column_stack([(P[:, k] == P[:, k + 1]) for k in range(6)]).astype(float)
    changes[~np.isfinite(P[:, 1:7])] = np.nan
    d["stale_frac6"] = np.nanmean(changes, axis=1)
    zero = (d.volume.to_numpy() <= 0)
    zrun_id = (~zero).cumsum()
    d["zero_volume_run"] = pd.Series(zero.astype(int)).groupby([d.instrument_key.to_numpy(), d.trade_date.to_numpy(), zrun_id]).cumsum().to_numpy()
    d["_lv"] = np.log1p(d.volume.clip(lower=0))
    LV = _lags(d, "_lv", day, 12)
    with np.errstate(all="ignore"):
        d["vol_sum3"] = np.nansum(LV[:, :3], 1)
        d["vol_ratio_3_12"] = np.nanmean(LV[:, :3], 1) - np.nanmean(LV, 1)
    d["oi_change1"] = (d.open_interest - g["open_interest"].shift(1)) / (g["open_interest"].shift(1).abs() + 1.)

    for name, column, k in (("iv_change3", "implied_volatility", 3), ("iv_change6", "implied_volatility", 6), ("ref_iv_change3", "reference_iv", 3)):
        d[name] = d[column] - g[column].shift(k)
    d["resid_lag1"] = g["iv_surface_residual"].shift(1)
    R = _lags(d, "iv_surface_residual", day, 3)
    with np.errstate(all="ignore"):
        d["resid_mean3"] = np.nanmean(R, 1)
    d["parity_lag1"] = g["parity_residual"].shift(1)
    Q = _lags(d, "parity_residual", day, 3)
    with np.errstate(all="ignore"):
        d["parity_mean3"] = np.nanmean(Q, 1)

    # ---- cross-section at exactly t: smile per (timestamp, expiry, type)
    cs_keys = ["timestamp", "expiry", "option_type"]
    fits = d.groupby(cs_keys, sort=False)[["log_moneyness", "implied_volatility"]].apply(
        lambda frame: pd.Series(_fit_smile(frame))).reset_index()
    d = d.merge(fits[cs_keys + ["smile_slope", "smile_curv", "smile_n", "atm_iv"]], on=cs_keys, how="left", validate="many_to_one")
    d["atm_iv_own"] = d["atm_iv"]
    fitted = d.atm_iv + d.smile_slope * d.log_moneyness + d.smile_curv * d.log_moneyness ** 2
    d["iv_minus_smile"] = d.implied_volatility - fitted
    other = d[["timestamp", "expiry", "strike", "option_type", "implied_volatility"]].copy()
    other["option_type"] = other.option_type.map({"CE": "PE", "PE": "CE"})
    other = other.rename(columns={"implied_volatility": "_other_iv"})
    d = d.merge(other, on=["timestamp", "expiry", "strike", "option_type"], how="left", validate="many_to_one")
    d["put_call_iv_spread"] = d.implied_volatility - d._other_iv
    # ---- term structure and smile dynamics from the per-(timestamp, expiry, type) table
    cs = d.drop_duplicates(cs_keys)[cs_keys + ["trade_date", "atm_iv", "smile_slope"]].sort_values(["expiry", "option_type", "timestamp"])
    gc = cs.groupby(["expiry", "option_type", "trade_date"], sort=False)
    cs["atm_iv_change6"] = cs.atm_iv - gc.atm_iv.shift(6)
    cs["smile_slope_change6"] = cs.smile_slope - gc.smile_slope.shift(6)
    cs["atm_iv_vs_open"] = cs.atm_iv - gc.atm_iv.transform("first")
    d = d.merge(cs[cs_keys + ["atm_iv_change6", "smile_slope_change6", "atm_iv_vs_open"]], on=cs_keys, how="left", validate="many_to_one")
    d["absdev_rank_ts"] = d.assign(_a=d.reference_price_deviation.abs()).groupby("timestamp")["_a"].rank(pct=True)
    d["volume_rank_expiry"] = d.groupby(["timestamp", "expiry"])["volume"].rank(pct=True)
    d["absdev_x_logvol3"] = d.reference_price_deviation.abs() * d.vol_sum3
    out = d[KEY + V2_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return out.sort_values(KEY).reset_index(drop=True)
