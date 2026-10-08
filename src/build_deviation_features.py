from pathlib import Path

import numpy as np
import pandas as pd

from black_scholes import black_scholes_price


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_reference_iv.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_deviation_features.parquet"
)

# Do not calculate percentage deviation when the
# theoretical price is effectively zero.
MIN_REFERENCE_PRICE = 1e-6


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BUILDING DEVIATION FEATURES")
print("=" * 80)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

df = pd.read_parquet(INPUT_FILE)

print(f"\nInput rows: {len(df):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "timestamp",
    "expiry",
    "option_type",
    "strike",
    "nifty_spot",
    "market_price",
    "time_to_expiry",
    "risk_free_rate",
    "reference_iv",
    "implied_volatility",
]

missing = [
    c for c in required_columns
    if c not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# REFERENCE BLACK-SCHOLES PRICE
# ============================================================

print("\nCalculating reference theoretical prices...")
print("This uses REFERENCE IV, not the option's own IV.")


def calculate_reference_price(row):

    spot = row["nifty_spot"]
    strike = row["strike"]
    T = row["time_to_expiry"]
    rate = row["risk_free_rate"]
    reference_iv = row["reference_iv"]
    option_type = row["option_type"]

    # Validate reference IV
    if not np.isfinite(reference_iv):
        return np.nan

    if reference_iv <= 0:
        return np.nan

    # Validate spot
    if not np.isfinite(spot) or spot <= 0:
        return np.nan

    # Validate strike
    if not np.isfinite(strike) or strike <= 0:
        return np.nan

    # Validate time to expiry
    if not np.isfinite(T) or T <= 0:
        return np.nan

    try:

        price = black_scholes_price(
            spot=spot,
            strike=strike,
            time_to_expiry=T,
            volatility=reference_iv,
            risk_free_rate=rate,
            option_type=option_type,
        )

        if not np.isfinite(price):
            return np.nan

        return price

    except Exception:
        return np.nan


df["reference_theoretical_price"] = df.apply(
    calculate_reference_price,
    axis=1,
)


# ============================================================
# PRICE DEVIATION
# ============================================================

print("\nCalculating pricing deviations...")

# Market price - reference theoretical price
#
# Positive:
#   market price > theoretical price
#
# Negative:
#   market price < theoretical price

df["reference_price_deviation"] = (
    df["market_price"]
    - df["reference_theoretical_price"]
)


# ============================================================
# PRICE DEVIATION %
# ============================================================

# Only calculate percentage deviation when the
# reference theoretical price is meaningfully positive.

df["reference_price_deviation_pct"] = np.where(
    df["reference_theoretical_price"] > MIN_REFERENCE_PRICE,
    (
        df["reference_price_deviation"]
        / df["reference_theoretical_price"]
    ),
    np.nan,
)

# Explicitly remove any accidental infinities.
df["reference_price_deviation_pct"] = (
    df["reference_price_deviation_pct"]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# IV SURFACE RESIDUAL
# ============================================================

df["iv_surface_residual"] = (
    df["implied_volatility"]
    - df["reference_iv"]
)


df["iv_surface_residual_pct"] = np.where(
    df["reference_iv"] > 0,
    (
        df["implied_volatility"]
        - df["reference_iv"]
    )
    / df["reference_iv"],
    np.nan,
)

# Remove accidental infinities.
df["iv_surface_residual_pct"] = (
    df["iv_surface_residual_pct"]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# ABSOLUTE DEVIATIONS
# ============================================================

df["absolute_price_deviation"] = (
    df["reference_price_deviation"]
    .abs()
)

df["absolute_price_deviation_pct"] = (
    df["reference_price_deviation_pct"]
    .abs()
)


# ============================================================
# VALIDATION OF NUMERIC FEATURES
# ============================================================

# Make sure no infinite values remain in the important
# deviation features.

deviation_columns = [
    "reference_theoretical_price",
    "reference_price_deviation",
    "reference_price_deviation_pct",
    "iv_surface_residual",
    "iv_surface_residual_pct",
    "absolute_price_deviation",
    "absolute_price_deviation_pct",
]

for column in deviation_columns:

    df[column] = (
        df[column]
        .replace([np.inf, -np.inf], np.nan)
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("DEVIATION SUMMARY")
print("=" * 80)

print(
    f"Rows: {len(df):,}"
)


valid_reference_price = (
    df["reference_theoretical_price"]
    .notna()
)

print(
    f"Valid reference prices: "
    f"{valid_reference_price.sum():,}"
)

print(
    f"Missing reference prices: "
    f"{(~valid_reference_price).sum():,}"
)


# ============================================================
# REFERENCE PRICE SUMMARY
# ============================================================

print("\nReference theoretical price:")

print(
    df["reference_theoretical_price"]
    .describe()
    .to_string()
)


# ============================================================
# ABSOLUTE PRICE DEVIATION SUMMARY
# ============================================================

print("\nPrice deviation:")

print(
    df["reference_price_deviation"]
    .describe()
    .to_string()
)


# ============================================================
# PERCENTAGE DEVIATION SUMMARY
# ============================================================

print("\nPrice deviation percentage:")

print(
    df["reference_price_deviation_pct"]
    .describe()
    .to_string()
)


# ============================================================
# EXTREME DEVIATIONS
# ============================================================

print("\nExtreme absolute deviations:")

extreme_df = df[
    np.isfinite(
        df["absolute_price_deviation_pct"]
    )
].copy()

extreme_columns = [
    "timestamp",
    "option_type",
    "strike",
    "market_price",
    "reference_iv",
    "reference_theoretical_price",
    "reference_price_deviation",
    "reference_price_deviation_pct",
    "absolute_price_deviation_pct",
]

print(
    extreme_df[
        extreme_columns
    ]
    .sort_values(
        "absolute_price_deviation_pct",
        ascending=False,
    )
    .head(10)
    .to_string(index=False)
)

# ============================================================
# DATA QUALITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("DATA QUALITY")
print("=" * 80)

print(
    "Infinite values in deviation columns:"
)

for column in deviation_columns:

    infinite_count = np.isinf(
        df[column].to_numpy(
            dtype=float
        )
    ).sum()

    print(
        f"{column}: {infinite_count:,}"
    )


print(
    "\nMissing percentage deviations:"
)

print(
    f"reference_price_deviation_pct: "
    f"{df['reference_price_deviation_pct'].isna().sum():,}"
)

print(
    f"iv_surface_residual_pct: "
    f"{df['iv_surface_residual_pct'].isna().sum():,}"
)


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("DEVIATION FEATURE BUILD COMPLETE")
print("=" * 80)

print(
    f"Saved: {OUTPUT_FILE}"
)