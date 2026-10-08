from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_dataset.parquet"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_splits"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

TARGET = "target_converged_30m"

# Chronological proportions
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("PREPARING TIME-BASED ML SPLIT")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

print(f"Input rows: {len(df):,}")


# ============================================================
# BASIC SORTING
# ============================================================

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values(
    ["timestamp", "instrument_key"]
).reset_index(drop=True)


# ============================================================
# LEAKAGE AUDIT
# ============================================================

print()
print("=" * 80)
print("LEAKAGE AUDIT")
print("=" * 80)

# Columns that must NEVER be model features
FORBIDDEN_COLUMNS = [
    # Target
    "target_converged_30m",

    # Future / label construction
    "future_timestamp",
    "future_deviation",
    "deviation_reduction_30m",
    "label_valid_30m",
    "converged_30m",

    # 60-minute target information
    "target_converged_60m",
    "future_deviation_60m",
    "deviation_reduction_60m",
    "label_valid_60m",
]

present_forbidden = [
    col for col in FORBIDDEN_COLUMNS
    if col in df.columns
]

print("Forbidden columns present:")
if present_forbidden:
    for col in present_forbidden:
        print(f"  - {col}")
else:
    print("  None")


# ============================================================
# FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "nifty_spot",
    "strike",
    "time_to_expiry",
    "moneyness",
    "log_moneyness",
    "option_type",
    "market_price",
    "implied_volatility",
    "reference_iv",
    "iv_surface_residual",
    "iv_surface_residual_pct",
    "reference_theoretical_price",
    "reference_price_deviation",
    "reference_price_deviation_pct",
    "absolute_price_deviation",
    "absolute_price_deviation_pct",

    "historical_volatility_5d",
    "historical_volatility_10d",
    "historical_volatility_20d",

    "iv_minus_hv_5d",
    "iv_minus_hv_10d",
    "iv_minus_hv_20d",

    "iv_hv_ratio_5d",
    "iv_hv_ratio_10d",
    "iv_hv_ratio_20d",

    "delta",
    "gamma",
    "vega",
    "theta",

    "volume",
    "open_interest",

    "parity_residual",
    "parity_residual_pct",
]


missing_features = [
    col for col in FEATURE_COLUMNS
    if col not in df.columns
]

if missing_features:
    print()
    print("MISSING FEATURES:")
    for col in missing_features:
        print(f"  - {col}")

    raise ValueError("Required ML features are missing.")


print()
print(f"Feature count: {len(FEATURE_COLUMNS)}")
print("All required features are present.")


# ============================================================
# DATE-BASED SPLIT
# ============================================================

print()
print("=" * 80)
print("CHRONOLOGICAL SPLIT")
print("=" * 80)

unique_dates = np.sort(
    df["timestamp"].dt.date.unique()
)

n_dates = len(unique_dates)

train_end_idx = int(n_dates * TRAIN_RATIO)
valid_end_idx = int(
    n_dates * (TRAIN_RATIO + VALID_RATIO)
)

train_dates = unique_dates[:train_end_idx]
valid_dates = unique_dates[train_end_idx:valid_end_idx]
test_dates = unique_dates[valid_end_idx:]

train_start = train_dates[0]
train_end = train_dates[-1]

valid_start = valid_dates[0]
valid_end = valid_dates[-1]

test_start = test_dates[0]
test_end = test_dates[-1]


print(f"Total trading dates: {n_dates}")

print()
print("TRAIN")
print(f"Dates: {len(train_dates)}")
print(f"Start: {train_start}")
print(f"End:   {train_end}")

print()
print("VALIDATION")
print(f"Dates: {len(valid_dates)}")
print(f"Start: {valid_start}")
print(f"End:   {valid_end}")

print()
print("TEST")
print(f"Dates: {len(test_dates)}")
print(f"Start: {test_start}")
print(f"End:   {test_end}")


# ============================================================
# CREATE MASKS
# ============================================================

date_series = df["timestamp"].dt.date

train_mask = date_series.isin(train_dates)
valid_mask = date_series.isin(valid_dates)
test_mask = date_series.isin(test_dates)

train = df.loc[train_mask].copy()
valid = df.loc[valid_mask].copy()
test = df.loc[test_mask].copy()


# ============================================================
# SAFETY CHECK
# ============================================================

assert len(train) + len(valid) + len(test) == len(df)

assert train["timestamp"].max() < valid["timestamp"].min()
assert valid["timestamp"].max() < test["timestamp"].min()

print()
print("Chronological ordering check: PASSED")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

def print_target_distribution(name, data):
    print()
    print(f"{name}")
    print("-" * 40)

    counts = data[TARGET].value_counts().sort_index()

    print(counts)

    percentages = (
        data[TARGET]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print()
    print(percentages)


print()
print("=" * 80)
print("TARGET DISTRIBUTION BY SPLIT")
print("=" * 80)

print_target_distribution("TRAIN", train)
print_target_distribution("VALIDATION", valid)
print_target_distribution("TEST", test)


# ============================================================
# ROW COUNTS
# ============================================================

print()
print("=" * 80)
print("SPLIT SIZES")
print("=" * 80)

print(f"Train rows:      {len(train):,}")
print(f"Validation rows: {len(valid):,}")
print(f"Test rows:       {len(test):,}")
print(f"Total rows:      {len(train) + len(valid) + len(test):,}")


# ============================================================
# SAVE
# ============================================================

train_file = OUTPUT_DIR / "train.parquet"
valid_file = OUTPUT_DIR / "validation.parquet"
test_file = OUTPUT_DIR / "test.parquet"

train.to_parquet(train_file, index=False)
valid.to_parquet(valid_file, index=False)
test.to_parquet(test_file, index=False)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 80)
print("SAVED")
print("=" * 80)

print(f"Train:      {train_file}")
print(f"Validation: {valid_file}")
print(f"Test:       {test_file}")

print()
print("Time-based ML split completed successfully.")