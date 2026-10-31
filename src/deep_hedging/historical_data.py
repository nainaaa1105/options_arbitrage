"""Causal historical states and expanding-window convergence predictions."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from .costs import transaction_fees


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_episodes(episodes, validation_start="2026-07-01", test_start="2026-09-16"):
    v = pd.Timestamp(validation_start, tz="Asia/Kolkata")
    t = pd.Timestamp(test_start, tz="Asia/Kolkata")
    if v >= t:
        raise ValueError("Validation must precede test")
    d = episodes.copy()
    d["split"] = np.select([d.exit_time.lt(v), d.entry_time.ge(v) & d.exit_time.lt(t),
                             d.entry_time.ge(t)], ["train", "validation", "test"], default="purged")
    return d


def attach_states(episodes, paths, options, probabilities=None):
    """Join only the immediately completed candle, never its execution open."""
    keys = ["instrument_key", "timestamp"]
    if options.duplicated(keys).any():
        raise ValueError("Duplicate historical observation")
    meta = episodes[["episode_id", "instrument_key", "strike", "option_type"]]
    d = paths.merge(meta, on="episode_id", validate="many_to_one")
    d["observation_timestamp"] = d.timestamp-pd.Timedelta(minutes=5)
    cols = [*keys, "close", "implied_volatility", "time_to_expiry", "feature_available_at"]
    state = options[cols].rename(columns={"timestamp": "observation_timestamp", "close": "option_close_observed",
                                         "implied_volatility": "observed_iv", "time_to_expiry": "observed_calendar_tte"})
    d = d.merge(state, on=["instrument_key", "observation_timestamp"], how="left", validate="many_to_one")
    if d.feature_available_at.isna().any() or not d.feature_available_at.eq(d.observed_at).all():
        raise ValueError("Missing or unverified historical observation")
    d = d.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    d["pricing_state_valid"] = (np.isfinite(d.observed_iv) & d.observed_iv.gt(0)
                                 & np.isfinite(d.delta) & d.delta.abs().le(1))
    d["observed_delta_used"] = d.delta.where(d.pricing_state_valid)
    d["observed_iv"] = d.observed_iv.where(d.pricing_state_valid)
    d[["observed_delta_used", "observed_iv"]] = d.groupby("episode_id")[["observed_delta_used", "observed_iv"]].ffill()
    numeric = ["observed_iv", "observed_calendar_tte", "option_close_observed", "observed_delta_used", "future_spot_delta"]
    if not np.isfinite(d[numeric]).all().all() or (d[numeric[:3]] <= 0).any().any():
        raise ValueError("Invalid initial historical state; cannot carry a state from a future bar")
    # Preserve the pricing convention: historical IV and ACT/365 maturity
    # must stay paired, otherwise standardized moneyness changes artificially.
    # Calendar-day normalization is handled explicitly in observed_features.
    d["remaining_time"] = d.observed_calendar_tte
    d["volatility"] = d.observed_iv
    d["is_call"] = d.option_type.eq("CE")
    d = d.sort_values(["episode_id", "timestamp"]).reset_index(drop=True)
    d["recent_return"] = d.groupby("episode_id").spot_observed.transform(lambda s: np.log(s/s.shift(1))).fillna(0.)
    if probabilities is not None:
        p = probabilities.rename(columns={"timestamp": "observation_timestamp"})
        d = d.merge(p, on=["instrument_key", "observation_timestamp"], how="left", validate="many_to_one")
    if "probability" not in d:
        d["probability"] = np.nan
    d["ml_available"] = np.isfinite(d.probability)
    d["probability"] = d.probability.fillna(.5)
    if not d.probability.between(0, 1).all():
        raise ValueError("Invalid ML probability")
    return d


def prediction_blocks(dates, min_history=60, block_sessions=20, test_start="2026-09-16"):
    if min_history < 1 or block_sessions < 1:
        raise ValueError("Invalid expanding-window design")
    days = sorted(set(dates))
    boundary = pd.Timestamp(test_start).date()
    development = [d for d in days if d < boundary]
    result = []
    for i in range(min_history, len(development), block_sessions):
        result.append((development[i], development[min(i+block_sessions, len(development))-1]))
    future = [d for d in days if d >= boundary]
    if future:
        result.append((boundary, future[-1]))
    return result


def fit_predict_blocks(training, query, feature_columns, estimator, blocks, output, target="target_converged_30m"):
    """Fit a fresh estimator using labels already available before each block."""
    output.mkdir(parents=True, exist_ok=False)
    if query.duplicated(["instrument_key", "timestamp"]).any():
        raise ValueError("Duplicate prediction query")
    predictions, audit = [], []
    for index, (start, end) in enumerate(blocks):
        cutoff = pd.Timestamp(start, tz="Asia/Kolkata")
        label_end = pd.to_datetime(training.future_timestamp_30m)+pd.Timedelta(minutes=5)
        keep = training.timestamp.lt(cutoff) & label_end.lt(cutoff) & training[target].notna()
        request = query.loc[query.timestamp.dt.date.between(start, end)].copy()
        if request.empty:
            continue
        fitted = clone(estimator)
        y = training.loc[keep, target].astype(int)
        if len(y) < 100 or y.nunique() != 2:
            raise ValueError("Insufficient completed classification history")
        fitted.fit(training.loc[keep, feature_columns], y)
        probability = fitted.predict_proba(request[feature_columns])[:, 1]
        if not np.isfinite(probability).all():
            raise ValueError("Non-finite classifier probabilities")
        file = output/f"fold_{index:02d}.joblib"
        joblib.dump({"model": fitted, "feature_columns": feature_columns,
                     "training_label_end_max": str(label_end.loc[keep].max()), "prediction_start": str(start)}, file)
        result = request[["instrument_key", "timestamp"]].copy()
        result["probability"] = probability
        result["ml_fold"] = index
        result["classifier_fit_before"] = cutoff
        predictions.append(result)
        audit.append({"fold": index, "prediction_start": str(start), "prediction_end": str(end),
                      "training_rows": int(keep.sum()), "prediction_rows": len(request),
                      "maximum_training_timestamp": str(training.loc[keep, "timestamp"].max()),
                      "maximum_label_available_at": str(label_end.loc[keep].max()), "model_sha256": sha256(file)})
        print(f"ML fold {index}: {keep.sum():,} past rows -> {len(request):,} historical predictions ({start} to {end})", flush=True)
    if not predictions:
        raise ValueError("No out-of-time ML predictions")
    pd.DataFrame(audit).to_csv(output/"fold_audit.csv", index=False)
    return pd.concat(predictions, ignore_index=True), audit


def prepare_ml(stage, paths, output, test_start="2026-09-16"):
    """Reuse the established feature schema and architecture, refit past-only."""
    import train_models
    from build_volatility_features import make_daily_spot, calculate_daily_hv
    from data_remediation_study import corrected_features
    from market_regime_features import market_table, augment
    from research_targets import session_horizon_eligible

    processed = stage/"data/processed"
    evaluation = stage/"fresh_evaluation"
    frozen = json.loads((evaluation/"frozen_before_fresh_evaluation.json").read_text())
    manifest = json.loads((evaluation/"inference_manifest.json").read_text())
    bundle = joblib.load(evaluation/"endpoint_frozen_model.joblib")
    train_models.CONTEXT_FILE = processed/"options_signal_eligible.parquet"
    history = pd.concat([pd.read_parquet(processed/"ml_dataset.parquet"),
                         pd.read_parquet(processed/"fresh_ml_dataset.parquet")], ignore_index=True)
    # All labels are retained in a separate fitting table; queries do not depend
    # on whether a future label exists, which prevents outcome-availability selection.
    needed = paths[["instrument_key", "observation_timestamp"]].drop_duplicates().rename(columns={"observation_timestamp": "timestamp"})
    context = pd.read_parquet(processed/"options_parity_features.parquet")
    query = needed.merge(context, on=["instrument_key", "timestamp"], how="left", validate="one_to_one")
    domain = (query.signal_eligible.fillna(False).astype(bool)
              & query.reference_price_deviation.abs().ge(1.) & session_horizon_eligible(query.timestamp))
    query = query.loc[domain].copy()
    query["option_type_CE"] = query.option_type.eq("CE").astype(int)
    query["option_type_PE"] = query.option_type.eq("PE").astype(int)
    # Match the training schema before engineering; do not introduce high/low,
    # source metadata or label columns as new predictors.
    base_columns = [c for c in history.columns if c in query.columns]
    raw = pd.concat([history, query[base_columns]], ignore_index=True).drop_duplicates(["instrument_key", "timestamp"])
    daily = pd.read_parquet(manifest["daily_spot_input"]).rename(columns={"close": "nifty_spot"})
    daily["trade_date"] = pd.to_datetime(daily.timestamp).dt.date
    features = corrected_features(raw, calculate_daily_hv(make_daily_spot(daily)))
    profile = frozen.get("market_feature_design")
    if profile is not None:
        features = augment(features, market_table(pd.read_parquet(profile["spot_input"]), pd.read_parquet(profile["vix_input"])))
    training = features.merge(history[["instrument_key", "timestamp"]], on=["instrument_key", "timestamp"], validate="one_to_one")
    requests = features.merge(query[["instrument_key", "timestamp"]], on=["instrument_key", "timestamp"], validate="one_to_one")
    blocks = prediction_blocks(context.timestamp.dt.date, test_start=test_start)
    result, audit = fit_predict_blocks(training, requests, bundle["feature_columns"], bundle["model"], blocks, output)
    return result, audit, {"target": "30-minute endpoint deviation convergence", "feature_columns": bundle["feature_columns"],
                           "source_architecture_sha256": sha256(evaluation/"endpoint_frozen_model.joblib"),
                           "query_domain": "Current signal_eligible, absolute deviation >= 1, session horizon valid; no future-label requirement",
                           "missing_probability": "Neutral 0.5 with explicit unavailable flag; position retained"}


def pack_paths(episodes, paths, config):
    """Fixed independent episodes for training, with execution and observation arrays separate."""
    ordered = episodes.sort_values("entry_time")
    groups = {key: group.sort_values("timestamp") for key, group in paths.groupby("episode_id")}
    rows = [groups[key] for key in ordered.episode_id]
    if not rows or any(len(g) != config.horizon_minutes//config.bar_minutes+1 for g in rows):
        raise ValueError("Incomplete historical training paths")
    mapping = {"option": "option_mark", "future": "future_mark", "delta": "delta", "spot": "spot_observed",
               "observed_option": "option_close_observed", "observed_future": "future_close_observed",
               "variance": "observed_iv", "maturity": "remaining_time", "future_spot_delta": "future_spot_delta",
               "recent_return": "recent_return", "probability": "probability", "ml_available": "ml_available",
               "observed_delta_used": "observed_delta_used", "pricing_state_valid": "pricing_state_valid"}
    b = {key: np.stack([g[col].to_numpy(float) for g in rows]) for key, col in mapping.items()}
    b["variance"] **= 2
    b.update({"strike": ordered.strike.to_numpy(float), "call": ordered.option_type.eq("CE").to_numpy(),
              "option_lot_size": ordered.option_lot_size.to_numpy(int),
              "future_lot_size": np.array([g.future_lot_size.iloc[0] for g in rows], int),
              "future_tick_size": np.array([g.future_tick_size.iloc[0] for g in rows], float)})
    for asset in ("option", "future"):
        b[asset+"_stt"] = np.array([transaction_fees(asset, -1, 1., g.timestamp.iloc[0], config)["stt"] for g in rows])
    b["episode_id"] = ordered.episode_id.to_numpy(str)
    return b


def subset_batch(batch, indices):
    return {key: value[indices] for key, value in batch.items()}
