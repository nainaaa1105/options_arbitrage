from pathlib import Path

import pandas as pd
from convergence import build_future_convergence_target


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_parity_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_convergence_labels.parquet"
)


# ============================================================
# CONFIG
# ============================================================

HORIZON_30_MIN = 30
HORIZON_60_MIN = 60

# Deviation must reduce by at least 50%
CONVERGENCE_THRESHOLD = 0.50

# Ignore extremely tiny deviations when creating labels
MIN_CURRENT_DEVIATION = 1.0

# Future observation can be up to 5 minutes later
# than the exact target time.
FUTURE_TOLERANCE_MINUTES = 5


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BUILDING FUTURE CONVERGENCE LABELS")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

print(f"Input rows: {len(df):,}")


# ============================================================
# BASIC SORTING
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["instrument_key", "timestamp"]
).reset_index(drop=True)


# ============================================================
# CURRENT ABSOLUTE DEVIATION
# ============================================================

df["current_absolute_deviation"] = (
    df["reference_price_deviation"].abs()
)


# ============================================================
# FUTURE TARGET FUNCTION
# ============================================================

def create_future_target(data, horizon_minutes):
    print(f"\nCreating {horizon_minutes}-minute endpoint target...")
    return build_future_convergence_target(
        data, horizon_minutes,
        min_current_deviation=MIN_CURRENT_DEVIATION,
        tolerance_minutes=FUTURE_TOLERANCE_MINUTES,
        reduction_threshold=CONVERGENCE_THRESHOLD,
    )


# ============================================================
# CREATE 30-MINUTE LABELS
# ============================================================

labels_30 = create_future_target(
    df,
    HORIZON_30_MIN,
)


# ============================================================
# CREATE 60-MINUTE LABELS
# ============================================================

labels_60 = create_future_target(
    df,
    HORIZON_60_MIN,
)


# ============================================================
# MERGE LABELS BACK
# ============================================================

df = df.merge(
    labels_30,
    on=[
        "instrument_key",
        "timestamp",
    ],
    how="left",
)

df = df.merge(
    labels_60,
    on=[
        "instrument_key",
        "timestamp",
    ],
    how="left",
)


# ============================================================
# CLEAN TEMP COLUMN
# ============================================================

df = df.drop(
    columns=["current_absolute_deviation"]
)


# ============================================================
# 30-MINUTE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("30-MINUTE LABEL SUMMARY")
print("=" * 80)

valid_30 = df["label_valid_30m"].fillna(False)

converged_30 = (
    df["converged_30m"].fillna(False)
)

print(
    f"Valid labels:     {valid_30.sum():,}"
)

print(
    f"Converged:         {converged_30.sum():,}"
)

if valid_30.sum() > 0:

    rate_30 = (
        converged_30[valid_30].mean()
        * 100
    )

    print(
        f"Convergence rate:  {rate_30:.2f}%"
    )

    print("\nDeviation reduction:")

    print(
        df.loc[
            valid_30,
            "deviation_reduction_30m"
        ].describe()
    )


# ============================================================
# 60-MINUTE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("60-MINUTE LABEL SUMMARY")
print("=" * 80)

valid_60 = df["label_valid_60m"].fillna(False)

converged_60 = (
    df["converged_60m"].fillna(False)
)

print(
    f"Valid labels:     {valid_60.sum():,}"
)

print(
    f"Converged:         {converged_60.sum():,}"
)

if valid_60.sum() > 0:

    rate_60 = (
        converged_60[valid_60].mean()
        * 100
    )

    print(
        f"Convergence rate:  {rate_60:.2f}%"
    )

    print("\nDeviation reduction:")

    print(
        df.loc[
            valid_60,
            "deviation_reduction_60m"
        ].describe()
    )


# ============================================================
# CROSS-TAB
# ============================================================

print("\n" + "=" * 80)
print("30m vs 60m CONVERGENCE")
print("=" * 80)

valid_both = (
    valid_30
    & valid_60
)

if valid_both.sum() > 0:

    table = pd.crosstab(
        df.loc[
            valid_both,
            "converged_30m"
        ],
        df.loc[
            valid_both,
            "converged_60m"
        ],
        rownames=["Converged 30m"],
        colnames=["Converged 60m"],
    )

    print(table)


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FILE)
print(f"Rows saved: {len(df):,}")

print(
    "Future convergence labels completed successfully."
)
