"""Causal option-chain feature pipeline as pure functions on in-memory frames, for live use and for batch/live parity tests.

Every feature in the research pipeline is causal: a row at time t depends only on rows of the same contract earlier the same session
and on rows of the whole chain observed at exactly t. Re-running the batch stage functions on the session's table of completed bars
therefore reproduces, for the newest bar, exactly what the batch pipeline stores. That gives parity by construction; the
script-level stages (deviation, eligibility, parity) are ported here and checked against the stored datasets by
tools/chain_parity_check.py and the tests.

Input rows use the master-dataset schema: timestamp (bar start), trade_date, expiry, instrument_key, trading_symbol, strike,
option_type, lot_size, open, high, low, close, volume, open_interest, nifty_spot, feature_available_at (+ optional provenance columns).
The chain must contain every contract recorded for the session, because cross-sectional features (market means, ranks) depend on it.
"""
import numpy as np
import pandas as pd

import train_models
from build_reference_iv import add_reference_iv_surface
from build_volatility_features import add_historical_volatility_features, make_daily_spot
from config import RISK_FREE_RATE
from market_regime_features import augment, market_table
from research_targets import session_horizon_eligible
from time_to_expiry import add_time_to_expiry
from vector_pricing import batch_pricing_features, batch_reference_prices

# Thresholds copied from build_signal_eligibility.py and build_deviation_features.py (the batch stages).
MIN_MARKET_PRICE, MIN_REFERENCE_PRICE_ELIGIBILITY, MIN_TIME_TO_EXPIRY_DAYS = 1.0, 1.0, 0.25
MIN_LOG_MONEYNESS, MAX_LOG_MONEYNESS, MIN_IV, MAX_IV, MIN_VOLUME, MIN_OPEN_INTEREST = -0.15, 0.15, 0.01, 3.0, 1, 1
MIN_REFERENCE_PRICE_DEVIATION, MIN_PARITY_PRICE = 1e-6, 1.0
MIN_ABSOLUTE_DEVIATION = 1.0                   # the classifier's own domain: at least one index point away from the reference


def stage_pricing(master):
    """Port of build_features.main(vectorized=True): time to expiry, moneyness, implied volatility and Greeks."""
    df = add_time_to_expiry(master)
    df["market_price"] = df["close"].astype(float)
    spot, strike = df["nifty_spot"], df["strike"]
    df["intrinsic_value"] = np.where(df.option_type.eq("CE"), np.maximum(spot - strike, 0.), np.where(df.option_type.eq("PE"), np.maximum(strike - spot, 0.), np.nan))
    df["moneyness"] = strike / spot
    df["log_moneyness"] = np.log(strike / spot)
    df["zero_volume"], df["zero_open_interest"], df["zero_market_price"] = df.volume <= 0, df.open_interest <= 0, df.market_price <= 0
    features = batch_pricing_features(df)
    df = pd.concat([df, features], axis=1)
    df["iv_reasonable"] = df.implied_volatility.notna() & (df.implied_volatility <= 5.0)
    df["risk_free_rate"] = RISK_FREE_RATE
    return df


def stage_volatility(df, daily_spot):
    """Prior-completed-day historical volatility from an independent daily spot history (columns timestamp, close)."""
    daily = daily_spot.rename(columns={"close": "nifty_spot"}).copy()
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    return add_historical_volatility_features(df, daily_spot=make_daily_spot(daily))


def stage_deviation(df):
    """Port of build_deviation_features.py (vectorized path)."""
    out = df.copy()
    out["reference_theoretical_price"] = batch_reference_prices(out)
    out["reference_price_deviation"] = out.market_price - out.reference_theoretical_price
    out["reference_price_deviation_pct"] = np.where(out.reference_theoretical_price > MIN_REFERENCE_PRICE_DEVIATION,
                                                    out.reference_price_deviation / out.reference_theoretical_price, np.nan)
    out["reference_price_deviation_pct"] = out.reference_price_deviation_pct.replace([np.inf, -np.inf], np.nan)
    out["iv_surface_residual"] = out.implied_volatility - out.reference_iv
    out["iv_surface_residual_pct"] = np.where(out.reference_iv > 0, (out.implied_volatility - out.reference_iv) / out.reference_iv, np.nan)
    out["iv_surface_residual_pct"] = out.iv_surface_residual_pct.replace([np.inf, -np.inf], np.nan)
    out["absolute_price_deviation"] = out.reference_price_deviation.abs()
    out["absolute_price_deviation_pct"] = out.reference_price_deviation_pct.abs()
    for column in ("reference_theoretical_price", "reference_price_deviation", "reference_price_deviation_pct", "iv_surface_residual",
                   "iv_surface_residual_pct", "absolute_price_deviation", "absolute_price_deviation_pct"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)
    return out


def stage_eligibility(df):
    """Port of build_signal_eligibility.py."""
    out = df.copy()
    out["valid_reference"] = out.reference_iv.notna() & out.reference_theoretical_price.notna() & (out.reference_iv > 0) & (out.reference_theoretical_price >= 0)
    out["valid_market_price"] = out.market_price.notna() & (out.market_price >= MIN_MARKET_PRICE)
    out["valid_spot"] = out.nifty_spot.notna() & (out.nifty_spot > 0)
    out["valid_strike"] = out.strike.notna() & (out.strike > 0)
    out["expiry_usable"] = out.time_to_expiry.notna() & (out.time_to_expiry >= MIN_TIME_TO_EXPIRY_DAYS / 365.0)
    out["reference_price_usable"] = out.reference_theoretical_price.notna() & (out.reference_theoretical_price >= MIN_REFERENCE_PRICE_ELIGIBILITY)
    out["iv_usable"] = out.implied_volatility.notna() & (out.implied_volatility >= MIN_IV) & (out.implied_volatility <= MAX_IV)
    out["reference_iv_usable"] = out.reference_iv.notna() & (out.reference_iv >= MIN_IV) & (out.reference_iv <= MAX_IV)
    out["moneyness_usable"] = out.log_moneyness.notna() & (out.log_moneyness >= MIN_LOG_MONEYNESS) & (out.log_moneyness <= MAX_LOG_MONEYNESS)
    out["volume_usable"] = out.volume.fillna(0) >= MIN_VOLUME
    out["open_interest_usable"] = out.open_interest.fillna(0) >= MIN_OPEN_INTEREST
    out["liquid"] = out.volume_usable & out.open_interest_usable
    out["deviation_usable"] = (out.reference_price_deviation.notna() & out.reference_price_deviation_pct.notna()
                               & np.isfinite(out.reference_price_deviation) & np.isfinite(out.reference_price_deviation_pct))
    out["signal_eligible"] = (out.valid_reference & out.valid_market_price & out.valid_spot & out.valid_strike & out.expiry_usable
                              & out.reference_price_usable & out.iv_usable & out.reference_iv_usable & out.moneyness_usable & out.liquid & out.deviation_usable)
    return out


def stage_parity(df):
    """Port of build_parity_features.py: call/put pairs at the same timestamp, expiry and strike."""
    from parity import calculate_parity_residual
    keys = ["timestamp", "expiry", "strike"]
    calls = df[df.option_type == "CE"][keys + ["market_price", "nifty_spot", "time_to_expiry", "risk_free_rate", "signal_eligible"]].rename(
        columns={"market_price": "call_price", "signal_eligible": "call_eligible"})
    puts = df[df.option_type == "PE"][keys + ["market_price", "signal_eligible"]].rename(columns={"market_price": "put_price", "signal_eligible": "put_eligible"})
    p = calls.merge(puts, on=keys, how="outer")
    p["parity_residual"] = calculate_parity_residual(p.call_price, p.put_price, p.nifty_spot, p.strike, p.time_to_expiry, p.risk_free_rate)
    p["parity_reference_value"] = (p.nifty_spot - p.strike * np.exp(-p.risk_free_rate * p.time_to_expiry)).abs()
    p.loc[p.parity_reference_value < MIN_PARITY_PRICE, "parity_reference_value"] = np.nan
    p["parity_residual_pct"] = p.parity_residual / p.parity_reference_value * 100
    p["parity_usable"] = (p.call_price.notna() & p.put_price.notna() & p.nifty_spot.notna() & p.strike.notna() & p.time_to_expiry.notna()
                          & p.risk_free_rate.notna() & (p.call_price >= MIN_PARITY_PRICE) & (p.put_price >= MIN_PARITY_PRICE)
                          & p.call_eligible.fillna(False) & p.put_eligible.fillna(False) & p.parity_residual.notna())
    out = df.merge(p[keys + ["call_price", "put_price", "parity_residual", "parity_residual_pct", "parity_usable"]], on=keys, how="left")
    out["parity_usable"] = out.parity_usable.fillna(False)
    for column in ("parity_residual", "parity_residual_pct"):
        out[column] = out[column].replace([np.inf, -np.inf], np.nan)
    return out


def run_stages(master, daily_spot):
    """Master-schema rows of ONE session (plus the chain) to the full surface table the batch pipeline stores."""
    df = stage_pricing(master)
    df = stage_volatility(df, daily_spot)
    df = add_reference_iv_surface(df)
    df = stage_deviation(df)
    df = stage_eligibility(df)
    return stage_parity(df)


# ---------------------------------------------------------------- classifier inputs
ML_EXTRA = ("option_type_CE", "option_type_PE", "absolute_iv_surface_residual", "absolute_parity_residual", "absolute_deviation")


def classifier_frame(surface, timestamp):
    """Candidate rows at one decision bar in the ML-dataset shape. Domain = current eligibility, |deviation| >= 1, session horizon valid."""
    now = surface[surface.timestamp == pd.Timestamp(timestamp)]
    keep = now.signal_eligible.fillna(False).astype(bool) & now.reference_price_deviation.abs().ge(MIN_ABSOLUTE_DEVIATION) & session_horizon_eligible(now.timestamp).to_numpy()
    c = now[keep].copy()
    c["option_type_CE"], c["option_type_PE"] = (c.option_type == "CE").astype(int), (c.option_type == "PE").astype(int)
    c["absolute_iv_surface_residual"], c["absolute_parity_residual"], c["absolute_deviation"] = c.iv_surface_residual.abs(), c.parity_residual.abs(), c.reference_price_deviation.abs()
    core = ["nifty_spot", "strike", "time_to_expiry", "market_price", "implied_volatility", "reference_iv", "reference_theoretical_price",
            "reference_price_deviation", "delta", "gamma", "vega", "theta", "volume", "open_interest"]
    return c.replace([np.inf, -np.inf], np.nan).dropna(subset=core).reset_index(drop=True)


def model_features(candidates, surface, daily_spot, spot_5m, vix_5m):
    """The exact feature path of the frozen accuracy study: corrected features, market regime inputs and the v2 causal features."""
    from accuracy_v2.features import KEY, REQUIRED, build_v2_features
    from data_remediation_study import corrected_features
    from build_volatility_features import calculate_daily_hv
    daily = daily_spot.rename(columns={"close": "nifty_spot"}).copy()
    daily["timestamp"] = pd.to_datetime(daily.timestamp)
    daily["trade_date"] = daily.timestamp.dt.date
    hv = calculate_daily_hv(make_daily_spot(daily))
    context = train_models.build_point_in_time_context(surface)          # lags and cross-section from the session's own surface
    original = train_models._load_point_in_time_context
    train_models._load_point_in_time_context = lambda: context
    try:
        features = corrected_features(candidates, hv)
    finally:
        train_models._load_point_in_time_context = original
    features = augment(features, market_table(spot_5m, vix_5m))
    features["timestamp"] = pd.to_datetime(features.timestamp)
    v2 = build_v2_features(surface[REQUIRED])
    return features.merge(v2, on=KEY, how="left", validate="one_to_one")


class ChainFeatureService:
    """Holds one session's chain; each completed bar yields the candidates with features (and probabilities if a bundle is given)."""

    def __init__(self, daily_spot, spot_5m, vix_5m, bundle=None):
        self.daily_spot, self.spot_5m, self.vix_5m, self.bundle = daily_spot, spot_5m, vix_5m, bundle
        self.table = None
        self.session = None

    def push(self, bar_rows):
        """bar_rows: every contract's completed bar for ONE timestamp (master schema). Returns candidates at that bar."""
        bar_rows = bar_rows.copy()
        stamps = pd.to_datetime(bar_rows.timestamp).unique()
        if len(stamps) != 1:
            raise ValueError("Push exactly one completed bar (one timestamp) at a time")
        stamp = pd.Timestamp(stamps[0])
        day = stamp.date()
        if self.session != day:
            self.table, self.session = None, day                       # new session: lags and cross-section never span days
        if self.table is not None and stamp <= pd.to_datetime(self.table.timestamp).max():
            raise ValueError("Out-of-order or repeated bar")
        self.table = bar_rows if self.table is None else pd.concat([self.table, bar_rows], ignore_index=True)
        surface = run_stages(self.table, self.daily_spot)
        surface["timestamp"] = pd.to_datetime(surface.timestamp)
        candidates = classifier_frame(surface, stamp)
        if candidates.empty:
            return candidates
        features = model_features(candidates, surface, self.daily_spot, self.spot_5m, self.vix_5m)
        if self.bundle is not None:
            from accuracy_v2.experiment import predict_bundle
            features["probability"] = predict_bundle(self.bundle, features)
        return features
