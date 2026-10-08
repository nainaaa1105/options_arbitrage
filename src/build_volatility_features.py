from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_volatility_features.parquet"
)

WINDOWS = [5, 10, 20]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("BUILDING CORRECT HISTORICAL VOLATILITY FEATURES")
print("=" * 80)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

df = pd.read_parquet(INPUT_FILE)

print(f"\nInput rows: {len(df):,}")


# ============================================================
# PREPARE TIMESTAMP
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

if df["timestamp"].dt.tz is None:
    df["timestamp"] = df["timestamp"].dt.tz_localize(
        "Asia/Kolkata"
    )
else:
    df["timestamp"] = df["timestamp"].dt.tz_convert(
        "Asia/Kolkata"
    )


# ============================================================
# CHECK NIFTY SPOT COLUMN
# ============================================================

if "nifty_spot" not in df.columns:
    raise ValueError(
        "Column 'nifty_spot' was not found in the dataset."
    )


# ============================================================
# BUILD DAILY NIFTY SERIES
# ============================================================

print("\nBuilding daily NIFTY spot series...")

spot_df = (
    df[
        [
            "timestamp",
            "trade_date",
            "nifty_spot",
        ]
    ]
    .dropna(subset=["nifty_spot"])
    .sort_values("timestamp")
    .copy()
)

# Make sure trade_date is a date
spot_df["trade_date"] = pd.to_datetime(
    spot_df["trade_date"]
).dt.date


# ------------------------------------------------------------
# Use the LAST NIFTY observation of each trading day.
#
# This gives us the daily closing/reference spot.
# ------------------------------------------------------------

daily_spot = (
    spot_df
    .groupby("trade_date", as_index=False)
    .last()
    .sort_values("trade_date")
    .reset_index(drop=True)
)

daily_spot = daily_spot[
    [
        "trade_date",
        "nifty_spot",
    ]
]

print(
    f"Trading days available: "
    f"{len(daily_spot):,}"
)


# ============================================================
# DAILY LOG RETURNS
# ============================================================

print("\nCalculating daily NIFTY log returns...")

daily_spot["log_return"] = np.log(
    daily_spot["nifty_spot"]
    / daily_spot["nifty_spot"].shift(1)
)


# ============================================================
# HISTORICAL VOLATILITY
# ============================================================

print("\nCalculating historical volatility...")

for window in WINDOWS:

    column = f"historical_volatility_{window}d"

    daily_spot[column] = (
        daily_spot["log_return"]
        .rolling(window=window)
        .std()
        * np.sqrt(252)
    )

    print(
        f"Created: {column}"
    )


# ============================================================
# KEEP ONLY FEATURES TO MERGE
# ============================================================

volatility_columns = [
    "trade_date"
]

for window in WINDOWS:

    volatility_columns.append(
        f"historical_volatility_{window}d"
    )


volatility_df = daily_spot[
    volatility_columns
].copy()


# ============================================================
# MERGE DAILY HV INTO OPTION DATA
# ============================================================

print("\nMerging daily volatility into option data...")

df["trade_date"] = pd.to_datetime(
    df["trade_date"]
).dt.date

df = df.merge(
    volatility_df,
    on="trade_date",
    how="left",
)


# ============================================================
# IV - HV FEATURES
# ============================================================

print("\nCalculating IV-HV features...")

for window in WINDOWS:

    hv_column = (
        f"historical_volatility_{window}d"
    )

    difference_column = (
        f"iv_minus_hv_{window}d"
    )

    ratio_column = (
        f"iv_hv_ratio_{window}d"
    )

    df[difference_column] = (
        df["implied_volatility"]
        - df[hv_column]
    )

    df[ratio_column] = np.where(
        df[hv_column] > 0,
        df["implied_volatility"]
        / df[hv_column],
        np.nan,
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("VOLATILITY FEATURE SUMMARY")
print("=" * 80)

print(
    f"Rows: {len(df):,}"
)

print(
    f"Trading days: "
    f"{df['trade_date'].nunique():,}"
)

for window in WINDOWS:

    hv_column = (
        f"historical_volatility_{window}d"
    )

    diff_column = (
        f"iv_minus_hv_{window}d"
    )

    ratio_column = (
        f"iv_hv_ratio_{window}d"
    )

    print(
        f"\n{window}-DAY HISTORICAL VOLATILITY"
    )

    print("-" * 80)

    print(
        df[hv_column]
        .describe()
        .to_string()
    )

    print(
        f"\nValid HV: "
        f"{df[hv_column].notna().sum():,}"
    )

    print(
        f"Valid IV-HV difference: "
        f"{df[diff_column].notna().sum():,}"
    )

    print(
        f"Valid IV/HV ratio: "
        f"{df[ratio_column].notna().sum():,}"
    )


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("VOLATILITY FEATURE BUILD COMPLETE")
print("=" * 80)

print(
    f"Saved: {OUTPUT_FILE}"
)