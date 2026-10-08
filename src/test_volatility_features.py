from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_volatility_features.parquet"
)


print("=" * 80)
print("VALIDATING VOLATILITY FEATURES")
print("=" * 80)

df = pd.read_parquet(FILE)

print(f"\nRows: {len(df):,}")

required_columns = [
    "historical_volatility_5d",
    "historical_volatility_10d",
    "historical_volatility_20d",
    "iv_minus_hv_5d",
    "iv_minus_hv_10d",
    "iv_minus_hv_20d",
    "iv_hv_ratio_5d",
    "iv_hv_ratio_10d",
    "iv_hv_ratio_20d",
]

print("\nChecking required columns...")

for column in required_columns:

    if column not in df.columns:
        raise AssertionError(
            f"Missing column: {column}"
        )

    print(f"PASS: {column}")


print("\nChecking row count...")

assert len(df) == 519725

print("PASS")


print("\nChecking NIFTY spot...")

assert "nifty_spot" in df.columns

assert df["nifty_spot"].notna().all()

print("PASS")

print("\nChecking HV values...")

for column in [
    "historical_volatility_5d",
    "historical_volatility_10d",
    "historical_volatility_20d",
]:

    valid = df[column].dropna()

    assert (valid >= 0).all()

    print(
        f"PASS: {column} "
        f"valid={len(valid):,}"
    )


print("\nChecking IV-HV features...")

for column in [
    "iv_minus_hv_5d",
    "iv_minus_hv_10d",
    "iv_minus_hv_20d",
]:

    print(
        f"{column}: "
        f"{df[column].notna().sum():,} valid"
    )


print("VOLATILITY FEATURE VALIDATION PASSED")