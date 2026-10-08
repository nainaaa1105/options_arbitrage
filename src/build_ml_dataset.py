from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_convergence_labels.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.parquet"
)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BUILDING ML DATASET")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

print(f"Input rows: {len(df):,}")


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    # --------------------------------------------------------
    # MARKET / CONTRACT
    # --------------------------------------------------------
    "nifty_spot",
    "strike",
    "time_to_expiry",
    "moneyness",
    "log_moneyness",

    # --------------------------------------------------------
    # OPTION TYPE
    # --------------------------------------------------------
    "option_type",

    # --------------------------------------------------------
    # MARKET PRICE
    # --------------------------------------------------------
    "market_price",

    # --------------------------------------------------------
    # IMPLIED VOLATILITY
    # --------------------------------------------------------
    "implied_volatility",
    "reference_iv",
    "iv_surface_residual",
    "iv_surface_residual_pct",

    # --------------------------------------------------------
    # FAIR VALUE / DEVIATION
    # --------------------------------------------------------
    "reference_theoretical_price",
    "reference_price_deviation",
    "reference_price_deviation_pct",
    "absolute_price_deviation",
    "absolute_price_deviation_pct",

    # --------------------------------------------------------
    # HISTORICAL VOLATILITY
    # --------------------------------------------------------
    "historical_volatility_5d",
    "historical_volatility_10d",
    "historical_volatility_20d",

    "iv_minus_hv_5d",
    "iv_minus_hv_10d",
    "iv_minus_hv_20d",

    "iv_hv_ratio_5d",
    "iv_hv_ratio_10d",
    "iv_hv_ratio_20d",

    # --------------------------------------------------------
    # GREEKS
    # --------------------------------------------------------
    "delta",
    "gamma",
    "vega",
    "theta",

    # --------------------------------------------------------
    # MARKET ACTIVITY
    # --------------------------------------------------------
    "volume",
    "open_interest",

    # --------------------------------------------------------
    # PARITY
    # --------------------------------------------------------
    "parity_residual",
    "parity_residual_pct",
]


# ============================================================
# CHECK AVAILABLE FEATURES
# ============================================================

available_features = [
    col
    for col in feature_columns
    if col in df.columns
]

missing_features = [
    col
    for col in feature_columns
    if col not in df.columns
]

print("\n" + "=" * 80)
print("FEATURE CHECK")
print("=" * 80)

print(
    f"Requested features: {len(feature_columns)}"
)

print(
    f"Available features: {len(available_features)}"
)

if missing_features:

    print("\nMissing features:")

    for col in missing_features:
        print(f"  - {col}")

    raise ValueError(
        "Required ML features are missing from the dataset."
    )


# ============================================================
# LABEL VALIDITY
# ============================================================

valid_label = (
    df["signal_eligible"].fillna(False)
    & df["label_valid_30m"].fillna(False)
)


# ============================================================
# SELECT ML DATA
# ============================================================

ml_columns = (
    [
        "timestamp",
        "trade_date",
        "expiry",
        "instrument_key",
        "trading_symbol",
    ]
    + available_features
    + [
        "converged_30m",
        "converged_60m",
        "deviation_reduction_30m",
        "deviation_reduction_60m",
        "label_valid_30m",
        "label_valid_60m",
    ]
)

ml_df = df.loc[
    valid_label,
    ml_columns
].copy()


# ============================================================
# TARGET
# ============================================================

ml_df["target_converged_30m"] = (
    ml_df["converged_30m"]
    .astype(int)
)


# ============================================================
# OPTION TYPE ENCODING
# ============================================================

ml_df["option_type_CE"] = (
    ml_df["option_type"] == "CE"
).astype(int)

ml_df["option_type_PE"] = (
    ml_df["option_type"] == "PE"
).astype(int)


# ============================================================
# ADD ABSOLUTE FEATURES
# ============================================================

ml_df["absolute_iv_surface_residual"] = (
    ml_df["iv_surface_residual"].abs()
)

ml_df["absolute_parity_residual"] = (
    ml_df["parity_residual"].abs()
)

ml_df["absolute_deviation"] = (
    ml_df["reference_price_deviation"].abs()
)


# ============================================================
# REMOVE EXTREME / INVALID VALUES
# ============================================================

numeric_columns = ml_df.select_dtypes(
    include=[np.number]
).columns

ml_df[numeric_columns] = (
    ml_df[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# FEATURE MISSINGNESS
# ============================================================

print("\n" + "=" * 80)
print("FEATURE MISSINGNESS")
print("=" * 80)

missing = (
    ml_df[available_features]
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print(
    missing[missing > 0].to_string()
)


# ============================================================
# DROP ROWS WITH MISSING CORE FEATURES
# ============================================================

core_features = [
    "nifty_spot",
    "strike",
    "time_to_expiry",
    "market_price",
    "implied_volatility",
    "reference_iv",
    "reference_theoretical_price",
    "reference_price_deviation",
    "delta",
    "gamma",
    "vega",
    "theta",
    "volume",
    "open_interest",
]


before = len(ml_df)

ml_df = ml_df.dropna(
    subset=core_features
).copy()

after = len(ml_df)

print(
    f"\nRows removed due to missing core features: "
    f"{before - after:,}"
)


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("TARGET DISTRIBUTION")
print("=" * 80)

target_counts = (
    ml_df["target_converged_30m"]
    .value_counts()
    .sort_index()
)

print(target_counts)

print("\nTarget percentage:")

print(
    ml_df["target_converged_30m"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)


# ============================================================
# DATE RANGE
# ============================================================

print("\n" + "=" * 80)
print("DATE RANGE")
print("=" * 80)

print(
    f"Start: {ml_df['timestamp'].min()}"
)

print(
    f"End:   {ml_df['timestamp'].max()}"
)

print(
    f"Trading dates: "
    f"{ml_df['trade_date'].nunique()}"
)

print(
    f"Expiries: "
    f"{ml_df['expiry'].nunique()}"
)


# ============================================================
# OPTION TYPE
# ============================================================

print("\n" + "=" * 80)
print("OPTION TYPE")
print("=" * 80)

print(
    ml_df["option_type"]
    .value_counts()
)


# ============================================================
# FINAL DATASET INFO
# ============================================================

print("\n" + "=" * 80)
print("FINAL ML DATASET")
print("=" * 80)

print(
    f"Rows:     {len(ml_df):,}"
)

print(
    f"Columns:  {len(ml_df.columns):,}"
)

print(
    f"Features: {len(available_features):,}"
)


# ============================================================
# SAVE
# ============================================================

ml_df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FILE)
print(
    f"Rows saved: {len(ml_df):,}"
)

print(
    "ML dataset construction completed successfully."
)