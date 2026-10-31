"""Tune candidate convergence classifiers on chronological validation data.

The selected family and probability cutoff are chosen on validation only.
The later test period is scored for diagnostics and is not used for selection.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import optuna
import joblib
from functools import lru_cache

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, log_loss, matthews_corrcoef
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLIT_DIR    = PROJECT_ROOT / "data" / "processed" / "ml_splits"
OUTPUT_DIR   = PROJECT_ROOT / "data" / "processed" / "models"
CONTEXT_FILE = PROJECT_ROOT / "data" / "processed" / "options_signal_eligible.parquet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILE = SPLIT_DIR / "train.parquet"
VALID_FILE = SPLIT_DIR / "validation.parquet"
TEST_FILE  = SPLIT_DIR / "test.parquet"

TARGET = "target_converged_30m"


# ============================================================
# FEATURE ENGINEERING CANDIDATES
# ============================================================

def build_point_in_time_context(context_df):
    """Build lag and cross-sectional predictors from the full causal surface.

    The context table contains every option row, including rows without valid
    future labels. It is observed at the current candle timestamp, so lag fields
    use only prior candles and cross-sectional fields use only that timestamp.
    """
    required = {
        "timestamp", "trade_date", "instrument_key", "market_price",
        "reference_price_deviation", "volume", "implied_volatility",
        "strike", "nifty_spot",
    }
    missing = sorted(required - set(context_df.columns))
    if missing:
        raise ValueError(f"Context table is missing required fields: {missing}")
    context = context_df[list(required)].copy()
    context["timestamp"] = pd.to_datetime(context["timestamp"])
    context = context.sort_values(["instrument_key", "trade_date", "timestamp"])
    keys = ["instrument_key", "trade_date"]
    grouped = context.groupby(keys, sort=False, observed=True)

    context["lag1_price"] = grouped["market_price"].shift(1)
    context["lag2_price"] = grouped["market_price"].shift(2)
    context["lag1_dev"] = grouped["reference_price_deviation"].shift(1)
    context["lag2_dev"] = grouped["reference_price_deviation"].shift(2)
    context["lag1_vol"] = grouped["volume"].shift(1)
    context["lag1_iv"] = grouped["implied_volatility"].shift(1)
    context["price_return_1step"] = (
        (context.market_price - context.lag1_price) / (context.lag1_price + 1e-4)
    ).clip(-1.0, 1.0)
    context["price_return_2step"] = (
        (context.market_price - context.lag2_price) / (context.lag2_price + 1e-4)
    ).clip(-1.0, 1.0)
    context["dev_change_1step"] = (
        context.reference_price_deviation - context.lag1_dev
    ).clip(-50.0, 50.0)
    context["dev_change_2step"] = (
        context.reference_price_deviation - context.lag2_dev
    ).clip(-50.0, 50.0)
    context["vol_surge_1step"] = (
        np.log1p(context.volume.clip(lower=0))
        - np.log1p(context.lag1_vol.clip(lower=0))
    ).clip(-5.0, 5.0)
    context["iv_change_1step"] = (
        context.implied_volatility - context.lag1_iv
    ).clip(-1.0, 1.0)

    context["_abs_dev"] = context.reference_price_deviation.abs()
    context["_turnover_volume"] = context.volume.clip(lower=0)
    cross = context.groupby("timestamp", sort=False, observed=True).agg(
        mkt_mean_abs_dev=("_abs_dev", "mean"),
        mkt_std_abs_dev=("_abs_dev", "std"),
        mkt_mean_iv=("implied_volatility", "mean"),
        mkt_active_strikes=("strike", "nunique"),
        mkt_spot_mean=("nifty_spot", "mean"),
        mkt_total_vol=("_turnover_volume", "sum"),
    ).reset_index()
    cross["mkt_std_abs_dev"] = cross["mkt_std_abs_dev"].fillna(0.0)
    context = context.merge(cross, on="timestamp", how="left", validate="many_to_one")
    context["mkt_dev_zscore"] = (
        (context["_abs_dev"] - context["mkt_mean_abs_dev"])
        / (context["mkt_std_abs_dev"] + 1e-4)
    ).clip(-5.0, 5.0)
    context["iv_minus_mkt_iv"] = context.implied_volatility - context.mkt_mean_iv
    context["strike_vol_share"] = (
        context.volume.clip(lower=0) / (context.mkt_total_vol + 1.0)
    ).clip(0.0, 1.0)
    feature_columns = [
        "lag1_price", "lag2_price", "lag1_dev", "lag2_dev", "lag1_vol", "lag1_iv",
        "price_return_1step", "price_return_2step", "dev_change_1step",
        "dev_change_2step", "vol_surge_1step", "iv_change_1step",
        "mkt_mean_abs_dev", "mkt_std_abs_dev", "mkt_dev_zscore", "mkt_mean_iv",
        "iv_minus_mkt_iv", "mkt_active_strikes", "mkt_spot_mean", "mkt_total_vol",
        "strike_vol_share",
    ]
    keys_frame = context[["timestamp", "instrument_key"] + feature_columns].copy()
    if keys_frame.duplicated(["timestamp", "instrument_key"]).any():
        raise ValueError("Context has duplicate instrument/timestamp rows")
    return keys_frame


@lru_cache(maxsize=1)
def _cached_point_in_time_context(path, modified_ns, size):
    return build_point_in_time_context(pd.read_parquet(path))


def _load_point_in_time_context():
    if not CONTEXT_FILE.exists():
        raise FileNotFoundError(
            f"Point-in-time feature context not found: {CONTEXT_FILE}. "
            "Build options_signal_eligible.parquet first."
        )
    signature = CONTEXT_FILE.stat()
    return _cached_point_in_time_context(str(CONTEXT_FILE), signature.st_mtime_ns, signature.st_size)


def build_features(df_dict, feature_schema_version=2):
    context_features = _load_point_in_time_context()
    processed = {}
    for split_name, df in df_dict.items():
        d = df.copy()
        ts = pd.to_datetime(d["timestamp"])
        d["timestamp"] = ts
        d = d.sort_values(["instrument_key", "timestamp"]).reset_index(drop=True)
        if feature_schema_version >= 2:
            # Derive row features after sorting; the old Series had the original
            # index and silently assigned other rows' session times.
            ts = d["timestamp"]
        
        # 1. Past Lags (strictly backward-looking within same day and instrument)
        grp_inst_day = d.groupby(["instrument_key", "trade_date"])
        d["lag1_price"] = grp_inst_day["market_price"].shift(1)
        d["lag2_price"] = grp_inst_day["market_price"].shift(2)
        d["lag1_dev"]   = grp_inst_day["reference_price_deviation"].shift(1)
        d["lag2_dev"]   = grp_inst_day["reference_price_deviation"].shift(2)
        d["lag1_vol"]   = grp_inst_day["volume"].shift(1)
        d["lag1_iv"]    = grp_inst_day["implied_volatility"].shift(1)
        
        d["price_return_1step"] = ((d["market_price"] - d["lag1_price"]) / (d["lag1_price"] + 1e-4)).clip(-1.0, 1.0)
        d["price_return_2step"] = ((d["market_price"] - d["lag2_price"]) / (d["lag2_price"] + 1e-4)).clip(-1.0, 1.0)
        d["dev_change_1step"]   = (d["reference_price_deviation"] - d["lag1_dev"]).clip(-50.0, 50.0)
        d["dev_change_2step"]   = (d["reference_price_deviation"] - d["lag2_dev"]).clip(-50.0, 50.0)
        d["vol_surge_1step"]    = (np.log1p(d["volume"].clip(lower=0)) - np.log1p(d["lag1_vol"].clip(lower=0))).clip(-5.0, 5.0)
        d["iv_change_1step"]    = (d["implied_volatility"] - d["lag1_iv"]).clip(-1.0, 1.0)
        
        # 2. Cross-Sectional Market State at timestamp t
        ts_grp = d.groupby("timestamp")
        d["mkt_mean_abs_dev"]   = ts_grp["absolute_price_deviation"].transform("mean")
        d["mkt_std_abs_dev"]    = ts_grp["absolute_price_deviation"].transform("std").fillna(0)
        d["mkt_dev_zscore"]     = ((d["absolute_price_deviation"] - d["mkt_mean_abs_dev"]) / (d["mkt_std_abs_dev"] + 1e-4)).clip(-5.0, 5.0)
        d["mkt_mean_iv"]        = ts_grp["implied_volatility"].transform("mean")
        d["iv_minus_mkt_iv"]    = d["implied_volatility"] - d["mkt_mean_iv"]
        d["mkt_active_strikes"] = ts_grp["strike"].transform("count")
        
        d["mkt_spot_mean"]      = ts_grp["nifty_spot"].transform("mean")
        d["mkt_total_vol"]      = ts_grp["volume"].transform("sum")
        d["strike_vol_share"]   = (d["volume"] / (d["mkt_total_vol"] + 1.0)).clip(0, 1)
        
        # 3. Intraday Session Dynamics
        d["intraday_minute"]    = (ts.dt.hour * 60 + ts.dt.minute - (9 * 60 + 15)).clip(lower=0)
        d["session_progress"]   = (d["intraday_minute"] / 375.0).clip(0, 1)
        d["time_to_close"]      = (375.0 - d["intraday_minute"]).clip(lower=0)
        d["sin_time"]           = np.sin(2 * np.pi * d["intraday_minute"] / 375.0)
        d["cos_time"]           = np.cos(2 * np.pi * d["intraday_minute"] / 375.0)
        d["day_of_week"]        = ts.dt.dayofweek
        d["is_opening_session"] = (d["intraday_minute"] <= 30).astype(float)
        d["is_closing_session"] = (d["time_to_close"] <= 45).astype(float)
        
        # 4. Expiry Dynamics
        d["days_to_expiry"]     = (d["time_to_expiry"] * (365 if feature_schema_version >= 2 else 252)).clip(lower=0)
        d["log_days_to_expiry"] = np.log1p(d["days_to_expiry"])
        d["is_0dte"]            = (d["days_to_expiry"] < 0.5).astype(float)
        if feature_schema_version >= 2:
            expiry_dates = pd.to_datetime(d["expiry"]).dt.date
            d["is_0dte"] = (expiry_dates == ts.dt.date).astype(float)
        d["is_near_expiry"]     = (d["days_to_expiry"] <= 2.0).astype(float)
        
        # 5. Moneyness & Strike Structure
        d["abs_log_moneyness"]  = d["log_moneyness"].abs()
        d["is_atm"]             = (d["abs_log_moneyness"] < 0.015).astype(float)
        d["is_itm"]             = np.where(d["option_type"] == "CE", d["nifty_spot"] > d["strike"], d["nifty_spot"] < d["strike"]).astype(float)
        d["otm_depth"]          = (d["abs_log_moneyness"] * 100).clip(upper=10.0)
        d["strike_to_spot"]     = d["strike"] / (d["nifty_spot"] + 1e-6)
        
        # 6. Mispricing & Relative Deviation Metrics
        mkt_price = d["market_price"].clip(lower=0.05)
        d["rel_price_dev"]      = (d["reference_price_deviation"] / mkt_price).clip(-5.0, 5.0)
        d["abs_rel_price_dev"]  = (d["absolute_price_deviation"] / mkt_price).clip(upper=5.0)
        d["dev_to_spot_bps"]    = (d["absolute_price_deviation"] / (d["nifty_spot"] + 1e-6) * 10000).clip(upper=500.0)
        d["dev_to_strike_bps"]  = (d["absolute_price_deviation"] / (d["strike"] + 1e-6) * 10000).clip(upper=500.0)
        
        # Parity Deviations
        d["abs_parity_residual"]     = d["parity_residual"].abs()
        d["abs_parity_residual_pct"] = d["parity_residual_pct"].abs().clip(upper=500.0)
        d["parity_dev_to_price"]     = (d["abs_parity_residual"] / mkt_price).clip(upper=5.0)
        
        # IV Deviations & Surface Residuals
        d["abs_iv_surface_residual"]     = d["iv_surface_residual"].abs()
        d["abs_iv_surface_residual_pct"] = d["iv_surface_residual_pct"].abs().clip(upper=500.0)
        ref_iv = d["reference_iv"].abs().clip(lower=0.01)
        d["iv_to_ref_ratio"]             = (d["implied_volatility"] / ref_iv).clip(0.1, 5.0)
        d["iv_ref_diff"]                 = d["implied_volatility"] - d["reference_iv"]
        
        # 7. Volatility Term Structure
        d["iv_minus_hv_5d_clip"]  = d["iv_minus_hv_5d"].clip(-1.0, 1.0)
        d["iv_minus_hv_10d_clip"] = d["iv_minus_hv_10d"].clip(-1.0, 1.0)
        d["iv_minus_hv_20d_clip"] = d["iv_minus_hv_20d"].clip(-1.0, 1.0)
        d["iv_hv_ratio_5d_clip"]  = d["iv_hv_ratio_5d"].clip(0.0, 10.0)
        d["iv_hv_ratio_10d_clip"] = d["iv_hv_ratio_10d"].clip(0.0, 10.0)
        d["iv_hv_ratio_20d_clip"] = d["iv_hv_ratio_20d"].clip(0.0, 10.0)
        d["hv_spread_5_20"]       = (d["historical_volatility_5d"] - d["historical_volatility_20d"]).clip(-1.0, 1.0)
        d["hv_ratio_5_20"]         = (d["historical_volatility_5d"] / (d["historical_volatility_20d"] + 1e-4)).clip(0.1, 5.0)
        
        # 8. Greeks & Option Elasticity
        d["abs_delta"] = d["delta"].abs()
        d["abs_gamma"] = d["gamma"].abs()
        d["abs_vega"]  = d["vega"].abs()
        d["abs_theta"] = d["theta"].abs()
        
        d["option_leverage"]  = ((d["abs_delta"] * d["nifty_spot"]) / mkt_price).clip(upper=100.0)
        d["gamma_per_price"]  = (d["abs_gamma"] * d["nifty_spot"] / mkt_price).clip(upper=50.0)
        d["vega_per_price"]   = (d["abs_vega"] / mkt_price).clip(upper=10.0)
        d["theta_per_price"]  = (d["abs_theta"] / mkt_price).clip(upper=10.0)
        d["gamma_vega_ratio"] = (d["abs_gamma"] / (d["abs_vega"] + 1e-5)).clip(upper=2.0)
        d["vega_x_iv_diff"]   = (d["abs_vega"] * d["iv_minus_hv_5d_clip"]).clip(-100.0, 100.0)
        
        # 9. Liquidity & Friction
        d["log_volume"]         = np.log1p(d["volume"].clip(lower=0))
        d["log_oi"]             = np.log1p(d["open_interest"].clip(lower=0))
        d["volume_oi_ratio"]    = (d["volume"].clip(lower=0) / (d["open_interest"].clip(lower=1) + 1.0)).clip(upper=10.0)
        d["estimated_turnover"] = d["volume"].clip(lower=0) * mkt_price
        d["log_turnover"]       = np.log1p(d["estimated_turnover"])
        d["liquidity_barrier"]  = (d["absolute_price_deviation"] / (d["log_volume"] + 1.0)).clip(upper=50.0)
        
        # 10. Multi-Factor Interactions
        d["dev_x_tte"]        = d["absolute_price_deviation"] * d["time_to_expiry"]
        d["dev_x_log_vol"]    = d["absolute_price_deviation"] * d["log_volume"]
        d["parity_x_iv_diff"] = d["abs_parity_residual"] * d["abs_iv_surface_residual"]
        d["is_call"]          = (d["option_type"] == "CE").astype(float)

        # Replace lags and same-time aggregates computed on the label-filtered
        # sample with values derived from the full contemporaneous option surface.
        context_columns = [
            name for name in context_features.columns
            if name not in {"timestamp", "instrument_key"}
        ]
        d = d.drop(columns=context_columns, errors="ignore").merge(
            context_features,
            on=["timestamp", "instrument_key"],
            how="left",
            validate="many_to_one",
            sort=False,
            indicator="__context_match",
        )
        if feature_schema_version >= 2 and (d["__context_match"] != "both").any():
            raise ValueError("Input rows lack contemporaneous context; rebuild the context dataset before prediction")
        d = d.drop(columns="__context_match")
        
        d = d.replace([np.inf, -np.inf], np.nan)
        d.attrs["feature_schema_version"] = feature_schema_version
        processed[split_name] = d
        
    return processed

FORBIDDEN_FEATURE_PATTERNS = (
    "target", "future", "label_valid", "converged", "deviation_reduction",
    "actual_", "timestamp", "trade_date", "instrument_key",
    "trading_symbol", "option_type", "_row_id", "lag", "price_return",
    "dev_change", "vol_surge", "iv_change", "mkt_", "strike_vol_share",
    "label_end", "endpoint", "first_passage", "sustained_two", "net_pnl", "profitable",
    "observed_path", "complete_path", "exit_", "entry_",
)
POINT_IN_TIME_CONTEXT_FEATURES = {
    "lag1_price", "lag2_price", "lag1_dev", "lag2_dev", "lag1_vol", "lag1_iv",
    "price_return_1step", "price_return_2step", "dev_change_1step",
    "dev_change_2step", "vol_surge_1step", "iv_change_1step",
    "mkt_mean_abs_dev", "mkt_std_abs_dev", "mkt_dev_zscore", "mkt_mean_iv",
    "iv_minus_mkt_iv", "mkt_active_strikes", "mkt_spot_mean", "mkt_total_vol",
    "strike_vol_share",
}
OPTUNA_TRIALS = 5
RANDOM_STATE = 42


def _feature_columns(frame, allow_point_in_time_context=False):
    """Select numeric predictors and fail closed on label/future-like names."""
    columns = []
    for name in frame.columns:
        lower = name.lower()
        if lower in {"expiry", "expiry_date", "source_file", "_original_order"}:
            continue
        if any(token in lower for token in FORBIDDEN_FEATURE_PATTERNS):
            if not (allow_point_in_time_context and name in POINT_IN_TIME_CONTEXT_FEATURES):
                continue
        if name in {"absolute_iv_surface_residual", "absolute_parity_residual", "absolute_deviation"}:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            columns.append(name)
    if not columns:
        raise ValueError("No numeric predictor columns remain after leakage filtering")
    return columns


def _make_estimator(name, trial=None, params=None):
    values = dict(params or {})
    if trial is not None:
        if name == "logistic_regression":
            values = {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
        elif name == "random_forest":
            values = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=100),
                "max_depth": trial.suggest_categorical("max_depth", [None, 8, 16]),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 20),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.8]),
            }
        elif name == "xgboost":
            values = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 7),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            }
        elif name == "lightgbm":
            values = {
                "n_estimators": trial.suggest_int("n_estimators", 150, 450, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 63),
                "max_depth": trial.suggest_int("max_depth", 3, 9),
                "min_child_samples": trial.suggest_int("min_child_samples", 15, 80),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20.0, log=True),
            }

    if name == "logistic_regression":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=values.get("C", 1.0), max_iter=1000,
                                          solver="lbfgs", random_state=RANDOM_STATE)),
        ])
    elif name == "random_forest":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1, **values)),
        ])
    elif name == "xgboost":
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", xgb.XGBClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                        tree_method="hist", eval_metric="logloss", **values)),
        ])
    elif name == "lightgbm":
        if values.get("subsample", 1.0) < 1.0:
            values.setdefault("subsample_freq", 1)
        estimator = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("model", lgb.LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                         verbosity=-1, **values)),
        ])
    else:
        raise ValueError(f"Unknown model family: {name}")
    return estimator


def _classification_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    calibration_bins = pd.DataFrame({"actual": y_true, "probability": probabilities})
    calibration_bins["bin"] = pd.cut(
        calibration_bins.probability,
        bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True,
        duplicates="drop",
    )
    calibration_bins = calibration_bins.groupby("bin", observed=False).agg(
        rows=("actual", "size"),
        observed=("actual", "mean"),
        predicted=("probability", "mean"),
    ).dropna()
    expected_calibration_error = (
        (calibration_bins.rows / max(len(y_true), 1))
        * (calibration_bins.observed - calibration_bins.predicted).abs()
    ).sum()
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
        "predicted_positive_rate": float(predictions.mean()),
        "matthews_corrcoef": matthews_corrcoef(y_true, predictions),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
        "balanced_accuracy": balanced_accuracy_score(y_true, predictions),
        "brier_score": brier_score_loss(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities, labels=[0, 1]),
        "expected_calibration_error_10bin": float(expected_calibration_error),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def _validation_thresholds(y_true, probabilities):
    """Return named decision thresholds selected using validation labels only."""
    grid = np.linspace(0.05, 0.95, 181)
    metrics = {"accuracy": [], "f1": [], "balanced_accuracy": []}
    for threshold in grid:
        predicted = probabilities >= threshold
        metrics["accuracy"].append(accuracy_score(y_true, predicted))
        metrics["f1"].append(f1_score(y_true, predicted, zero_division=0))
        metrics["balanced_accuracy"].append(
            balanced_accuracy_score(y_true, predicted)
        )
    selected = {
        f"validation_max_{metric}": float(grid[int(np.argmax(values))])
        for metric, values in metrics.items()
    }
    # Predeclare a 50% precision floor, then retain the most recall on validation.
    eligible = []
    for threshold in grid:
        predicted = probabilities >= threshold
        if predicted.any() and precision_score(y_true, predicted, zero_division=0) >= 0.50:
            eligible.append((recall_score(y_true, predicted), threshold))
    if eligible:
        selected["validation_max_recall_at_precision_0_50"] = float(
            max(eligible, key=lambda item: (item[0], item[1]))[1]
        )
    return selected


def _operating_point_table(y_true, probabilities, validation_thresholds):
    """Compare fixed cutoffs and validation-selected cutoffs on a given split."""
    policies = {
        "always_negative_baseline": (1.01, "baseline"),
        "always_positive_baseline": (0.00, "baseline"),
        "fixed_0.50": (0.50, "predeclared"),
        "fixed_0.55": (0.55, "predeclared"),
        "fixed_0.60": (0.60, "predeclared"),
        **{name: (value, "validation") for name, value in validation_thresholds.items()},
    }
    rows = []
    for policy, (threshold, selected_on) in policies.items():
        # Baselines need their own constant scores. Reusing model scores would
        # incorrectly give an always-negative classifier the model's ROC-AUC.
        scores = probabilities
        if policy == "always_negative_baseline":
            scores = np.zeros(len(y_true))
        elif policy == "always_positive_baseline":
            scores = np.ones(len(y_true))
        row = _classification_metrics(y_true, scores, threshold)
        row.update({"decision_policy": policy, "threshold_selected_on": selected_on})
        rows.append(row)
    return pd.DataFrame(rows)


def train_and_evaluate():
    """Tune on validation, select once, and evaluate the selected model on test."""
    import json

    print("TRAINING AND EVALUATING LEAKAGE-SCREENED MODELS")
    raw = {
        "train": pd.read_parquet(TRAIN_FILE),
        "validation": pd.read_parquet(VALID_FILE),
        "test": pd.read_parquet(TEST_FILE),
    }
    if any(TARGET not in frame for frame in raw.values()):
        raise ValueError(f"All splits must include {TARGET}")
    print("Engineering causal features independently within each chronological split...")
    prepared = build_features(raw)
    train_df, val_df, test_df = (prepared[k] for k in ("train", "validation", "test"))
    feature_columns = _feature_columns(train_df, allow_point_in_time_context=True)
    # Fail closed if a forbidden future/label field accidentally passes through.
    if any(
        any(token in col.lower() for token in FORBIDDEN_FEATURE_PATTERNS)
        and col not in POINT_IN_TIME_CONTEXT_FEATURES
        for col in feature_columns
    ):
        raise AssertionError("A forbidden predictor reached model training")
    y_train = train_df[TARGET].astype(int).to_numpy()
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()
    X_train = train_df[feature_columns]
    X_val = val_df[feature_columns]
    X_test = test_df[feature_columns]

    model_names = ["logistic_regression", "random_forest", "xgboost", "lightgbm"]
    validation_rows = []
    models = {}
    probabilities = {}
    studies = {}
    for name in model_names:
        print(f"\nTuning {name} on the validation split ({OPTUNA_TRIALS} trials)...")
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))

        def objective(trial):
            candidate = _make_estimator(name, trial=trial)
            candidate.fit(X_train, y_train)
            pred = candidate.predict_proba(X_val)[:, 1]
            return average_precision_score(y_val, pred)

        study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
        model = _make_estimator(name, params=study.best_params)
        model.fit(X_train, y_train)
        val_prob = model.predict_proba(X_val)[:, 1]
        models[name] = model
        probabilities[name] = val_prob
        studies[name] = study
        row = {"model": name, "validation_pr_auc": float(study.best_value),
               "best_params": json.dumps(study.best_params, sort_keys=True)}
        row.update(_classification_metrics(y_val, val_prob, 0.5))
        validation_rows.append(row)
        print(f"Validation PR-AUC: {study.best_value:.4f}")

    validation_results = pd.DataFrame(validation_rows).sort_values(
        ["roc_auc", "accuracy", "pr_auc"], ascending=False
    ).reset_index(drop=True)
    selected_name = validation_results.loc[0, "model"]
    selected_model = models[selected_name]
    val_prob = probabilities[selected_name]

    # Accuracy is the requested operating objective. Select this cutoff on
    # validation only; all fixed/reference and F1 alternatives remain reported.
    validation_thresholds = _validation_thresholds(y_val, val_prob)
    threshold_policy = "validation_max_accuracy"
    selected_threshold = validation_thresholds[threshold_policy]
    validation_operating = _classification_metrics(y_val, val_prob, selected_threshold)
    validation_cutoff_rows = []
    for cutoff in np.linspace(0.10, 0.90, 17):
        candidates = val_prob >= cutoff
        count = int(candidates.sum())
        validation_cutoff_rows.append({
            "split": "validation",
            "threshold": float(cutoff),
            "candidate_count": count,
            "candidate_coverage_pct": float(candidates.mean() * 100.0),
            "convergence_precision": float(y_val[candidates].mean()) if count else np.nan,
            "convergence_recall": float(y_val[candidates].sum() / max(y_val.sum(), 1)),
            "accuracy": float(accuracy_score(y_val, candidates.astype(int))),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, candidates.astype(int))),
            "f1": float(f1_score(y_val, candidates.astype(int), zero_division=0)),
        })

    print(f"\nSelected on validation ROC-AUC (accuracy tie-break): {selected_name}")
    print(f"Validation-selected accuracy threshold: {selected_threshold:.3f}")
    print("Scoring the held-out test split once for diagnostics (no model selection)...")
    test_prob = selected_model.predict_proba(X_test)[:, 1]
    test_metrics = _classification_metrics(y_test, test_prob, selected_threshold)
    test_metrics["model"] = selected_name
    test_metrics["test_rows"] = len(y_test)
    test_metrics["test_positive_rate"] = float(y_test.mean())
    test_metrics["always_negative_baseline_accuracy"] = float(1 - y_test.mean())
    test_metrics["decision_policy"] = threshold_policy
    test_metrics["threshold_selected_on"] = "validation"
    test_operating_points = _operating_point_table(
        y_test, test_prob, validation_thresholds
    )
    test_operating_points["test_rows"] = len(y_test)
    test_operating_points["test_positive_rate"] = float(y_test.mean())

    # Calibration summary on the final test scores (diagnostic only; no selection).
    calibration = pd.DataFrame({"probability": test_prob, "actual": y_test})
    calibration["bin"] = pd.cut(calibration.probability, bins=np.linspace(0, 1, 11),
                                include_lowest=True, duplicates="drop")
    calibration = calibration.groupby("bin", observed=False).agg(
        rows=("actual", "size"), mean_probability=("probability", "mean"),
        observed_rate=("actual", "mean")
    ).reset_index()

    validation_results.to_csv(OUTPUT_DIR / "model_validation_comparison.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(OUTPUT_DIR / "best_model_test_metrics.csv", index=False)
    test_operating_points.to_csv(
        OUTPUT_DIR / "best_model_test_operating_points.csv", index=False
    )
    pd.DataFrame(validation_cutoff_rows).to_csv(
        OUTPUT_DIR / "best_model_confidence_accuracy.csv", index=False
    )
    calibration.to_csv(OUTPUT_DIR / "best_model_test_calibration.csv", index=False)

    validation_rows = []
    for threshold in np.linspace(0.10, 0.90, 17):
        predicted = val_prob >= threshold
        count = int(predicted.sum())
        validation_rows.append({
            "split": "validation",
            "threshold": float(threshold),
            "candidate_count": count,
            "candidate_coverage_pct": float(predicted.mean() * 100.0),
            "convergence_precision": float(y_val[predicted].mean()) if count else np.nan,
            "convergence_recall": float(y_val[predicted].sum() / max(y_val.sum(), 1)),
            "accuracy": float(accuracy_score(y_val, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, predicted)),
            "f1": float(f1_score(y_val, predicted, zero_division=0)),
        })
    pd.DataFrame(validation_rows).to_csv(
        OUTPUT_DIR / "best_model_confidence_accuracy.csv", index=False
    )
    joblib.dump(selected_model, OUTPUT_DIR / "best_model.joblib")
    joblib.dump(feature_columns, OUTPUT_DIR / "best_model_feature_columns.joblib")
    with open(OUTPUT_DIR / "best_model_config.json", "w", encoding="utf-8") as stream:
        json.dump({"model": selected_name, "target": TARGET,
                   "threshold": selected_threshold, "threshold_policy": threshold_policy,
                   "validation_f1_at_selected_threshold": validation_operating["f1"],
                   "validation_accuracy_at_selected_threshold": validation_operating["accuracy"],
                   "validation_thresholds": validation_thresholds,
                   "selection_metric": "validation_roc_auc_then_accuracy",
                   "feature_schema_version": 2,
                   "validation_accuracy_at_0_50": float(validation_results.loc[0, "accuracy"]),
                   "validation_roc_auc": float(validation_results.loc[0, "roc_auc"]),
                   "validation_pr_auc": float(validation_results.loc[0, "validation_pr_auc"]),
                   "optuna_trials_per_model": OPTUNA_TRIALS,
                   "feature_count": len(feature_columns)}, stream, indent=2)

    # Keep per-family fitted models for reproducibility/comparison.
    for name, model in models.items():
        joblib.dump(model, OUTPUT_DIR / f"model_{name}.joblib")
    print("\nValidation comparison:")
    print(validation_results[["model", "validation_pr_auc", "accuracy", "precision", "recall", "f1"]].to_string(index=False))
    print("\nHeld-out test metrics (diagnostic only):")
    print(pd.Series(test_metrics).to_string())
    print(f"\nSaved selected model and evaluation artifacts to {OUTPUT_DIR}")
    return validation_results, test_metrics


def evaluate_saved_model():
    """Refresh operating-point diagnostics for the already-selected model.

    The model family and all decision thresholds remain selected using the
    validation split. The test split is only scored and never used for tuning.
    """
    import json

    raw = {
        "validation": pd.read_parquet(VALID_FILE),
        "test": pd.read_parquet(TEST_FILE),
    }
    config_path = OUTPUT_DIR / "best_model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    prepared = build_features(raw, feature_schema_version=config.get("feature_schema_version", 1))
    model = joblib.load(OUTPUT_DIR / "best_model.joblib")
    feature_columns = joblib.load(OUTPUT_DIR / "best_model_feature_columns.joblib")
    val_df, test_df = prepared["validation"], prepared["test"]
    y_val = val_df[TARGET].astype(int).to_numpy()
    y_test = test_df[TARGET].astype(int).to_numpy()
    val_prob = model.predict_proba(val_df[feature_columns])[:, 1]
    test_prob = model.predict_proba(test_df[feature_columns])[:, 1]

    validation_thresholds = _validation_thresholds(y_val, val_prob)
    operating_points = _operating_point_table(y_test, test_prob, validation_thresholds)
    operating_points["test_rows"] = len(y_test)
    operating_points["test_positive_rate"] = float(y_test.mean())
    operating_points.to_csv(OUTPUT_DIR / "best_model_test_operating_points.csv", index=False)

    config_path = OUTPUT_DIR / "best_model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    threshold_policy = config.get("threshold_policy", "validation_max_accuracy")
    selected_threshold = validation_thresholds[threshold_policy]
    selected_metrics = _classification_metrics(y_test, test_prob, selected_threshold)
    selected_metrics.update({
        "model": config["model"],
        "test_rows": len(y_test),
        "test_positive_rate": float(y_test.mean()),
        "decision_policy": threshold_policy,
        "threshold_selected_on": "validation",
        "always_negative_baseline_accuracy": float(1 - y_test.mean()),
    })
    pd.DataFrame([selected_metrics]).to_csv(
        OUTPUT_DIR / "best_model_test_metrics.csv", index=False
    )

    calibration = pd.DataFrame({"probability": test_prob, "actual": y_test})
    calibration["bin"] = pd.cut(
        calibration.probability, bins=np.linspace(0, 1, 11),
        include_lowest=True, duplicates="drop"
    )
    calibration = calibration.groupby("bin", observed=False).agg(
        rows=("actual", "size"), mean_probability=("probability", "mean"),
        observed_rate=("actual", "mean")
    ).reset_index()
    calibration.to_csv(OUTPUT_DIR / "best_model_test_calibration.csv", index=False)

    cutoff_rows = []
    for threshold in np.linspace(0.10, 0.90, 17):
        predicted = val_prob >= threshold
        count = int(predicted.sum())
        cutoff_rows.append({
            "split": "validation",
            "threshold": float(threshold),
            "candidate_count": count,
            "candidate_coverage_pct": float(predicted.mean() * 100.0),
            "convergence_precision": float(y_val[predicted].mean()) if count else np.nan,
            "convergence_recall": float(y_val[predicted].sum() / max(y_val.sum(), 1)),
            "accuracy": float(accuracy_score(y_val, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, predicted)),
            "f1": float(f1_score(y_val, predicted, zero_division=0)),
        })
    pd.DataFrame(cutoff_rows).to_csv(
        OUTPUT_DIR / "best_model_confidence_accuracy.csv", index=False
    )

    config["threshold"] = selected_threshold
    config["threshold_policy"] = threshold_policy
    config.pop("validation_f1", None)
    config["validation_f1_at_selected_threshold"] = float(
        _classification_metrics(y_val, val_prob, selected_threshold)["f1"]
    )
    config["validation_accuracy_at_selected_threshold"] = float(
        _classification_metrics(y_val, val_prob, selected_threshold)["accuracy"]
    )
    config["validation_thresholds"] = validation_thresholds
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(operating_points.to_string(index=False))
    print(f"\nAlways-negative test accuracy baseline: {1 - y_test.mean():.4f}")
    return operating_points


def promote_saved_validation_roc_winner():
    """Promote the retrained family with the best validation ROC-AUC.

    Each saved family model was fitted on the same training split. This selects
    among them using validation scores only and does not inspect the test labels.
    """
    import json

    comparison = pd.read_csv(OUTPUT_DIR / "model_validation_comparison.csv")
    winner = comparison.sort_values(
        ["roc_auc", "accuracy", "pr_auc"], ascending=False
    ).iloc[0]
    selected_name = winner["model"]
    comparison["selected_by_validation_roc"] = comparison["model"].eq(selected_name)
    comparison = comparison.sort_values(
        ["roc_auc", "accuracy", "pr_auc"], ascending=False
    )
    comparison.to_csv(OUTPUT_DIR / "model_validation_comparison.csv", index=False)
    model_path = OUTPUT_DIR / f"model_{selected_name}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    model = joblib.load(model_path)
    joblib.dump(model, OUTPUT_DIR / "best_model.joblib")

    config_path = OUTPUT_DIR / "best_model_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.update({
        "model": selected_name,
        "selection_metric": "validation_roc_auc_then_accuracy",
        "threshold_policy": "validation_max_accuracy",
        "validation_accuracy_at_0_50": float(winner["accuracy"]),
        "validation_roc_auc": float(winner["roc_auc"]),
        "validation_pr_auc": float(winner["validation_pr_auc"]),
    })
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(
        f"Promoted {selected_name} by validation ROC-AUC={winner['roc_auc']:.4f}; "
        f"accuracy@0.50={winner['accuracy']:.4f}."
    )
    return evaluate_saved_model()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-legacy-artifacts', action='store_true',
                        help='Explicitly opt into the preserved data/processed training/evaluation route.')
    parser.add_argument(
        "--evaluate-saved-model", action="store_true",
        help="Refresh validation-selected and fixed-threshold test diagnostics without retraining."
    )
    parser.add_argument(
        "--promote-validation-roc-winner", action="store_true",
        help="Select among fitted model families using validation ROC-AUC, then accuracy."
    )
    args = parser.parse_args()
    if not args.allow_legacy_artifacts:
        parser.error('Default data/processed artifacts are legacy. Use an isolated research run; pass --allow-legacy-artifacts only for an intentional legacy run.')
    if args.promote_validation_roc_winner:
        promote_saved_validation_roc_winner()
    elif args.evaluate_saved_model:
        evaluate_saved_model()
    else:
        train_and_evaluate()
