from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_signal_eligible.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_parity_features.parquet"
)


# ============================================================
# CONFIG
# ============================================================

MIN_PARITY_PRICE = 1.0


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BUILDING PUT-CALL PARITY FEATURES")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

print(f"Input rows: {len(df):,}")


# ============================================================
# ORIGINAL DATA
# ============================================================

result = df.copy()


# ============================================================
# CALL DATA
# ============================================================

key_columns = [
    "timestamp",
    "expiry",
    "strike",
]

calls = (
    df[df["option_type"] == "CE"][
        key_columns
        + [
            "market_price",
            "nifty_spot",
            "time_to_expiry",
            "risk_free_rate",
            "signal_eligible",
        ]
    ]
    .rename(
        columns={
            "market_price": "call_price",
            "signal_eligible": "call_eligible",
        }
    )
)


# ============================================================
# PUT DATA
# ============================================================

puts = (
    df[df["option_type"] == "PE"][
        key_columns
        + [
            "market_price",
            "signal_eligible",
        ]
    ]
    .rename(
        columns={
            "market_price": "put_price",
            "signal_eligible": "put_eligible",
        }
    )
)


# ============================================================
# MERGE CALL + PUT
# ============================================================

parity = calls.merge(
    puts,
    on=key_columns,
    how="outer",
)


# ============================================================
# DISCOUNTED STRIKE
# ============================================================

parity["discounted_strike"] = (
    parity["strike"]
    * np.exp(
        -parity["risk_free_rate"]
        * parity["time_to_expiry"]
    )
)


# ============================================================
# THEORETICAL C - P
# ============================================================

parity["theoretical_call_put_difference"] = (
    parity["nifty_spot"]
    - parity["discounted_strike"]
)


# ============================================================
# OBSERVED C - P
# ============================================================

parity["observed_call_put_difference"] = (
    parity["call_price"]
    - parity["put_price"]
)


# ============================================================
# PARITY RESIDUAL
# ============================================================

parity["parity_residual"] = (
    parity["observed_call_put_difference"]
    - parity["theoretical_call_put_difference"]
)


# ============================================================
# RELATIVE PARITY RESIDUAL
# ============================================================

parity["parity_reference_value"] = (
    parity["theoretical_call_put_difference"].abs()
)

parity.loc[
    parity["parity_reference_value"] < MIN_PARITY_PRICE,
    "parity_reference_value"
] = np.nan

parity["parity_residual_pct"] = (
    parity["parity_residual"]
    / parity["parity_reference_value"]
    * 100
)


# ============================================================
# PARITY USABILITY
# ============================================================

parity["parity_usable"] = (
    parity["call_price"].notna()
    & parity["put_price"].notna()
    & parity["nifty_spot"].notna()
    & parity["strike"].notna()
    & parity["time_to_expiry"].notna()
    & parity["risk_free_rate"].notna()
    & (parity["call_price"] >= MIN_PARITY_PRICE)
    & (parity["put_price"] >= MIN_PARITY_PRICE)
    & parity["call_eligible"].fillna(False)
    & parity["put_eligible"].fillna(False)
    & parity["parity_residual"].notna()
)


# ============================================================
# PARITY FEATURES
# ============================================================

parity_features = parity[
    key_columns
    + [
        "call_price",
        "put_price",
        "parity_residual",
        "parity_residual_pct",
        "parity_usable",
    ]
].copy()


# ============================================================
# MERGE BACK
# ============================================================

result = result.merge(
    parity_features,
    on=key_columns,
    how="left",
)


# ============================================================
# CLEAN UP
# ============================================================

result["parity_usable"] = (
    result["parity_usable"]
    .fillna(False)
)

result["parity_residual"] = (
    result["parity_residual"]
    .replace([np.inf, -np.inf], np.nan)
)

result["parity_residual_pct"] = (
    result["parity_residual_pct"]
    .replace([np.inf, -np.inf], np.nan)
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("PARITY SUMMARY")
print("=" * 80)

valid_parity = result["parity_residual"].notna()

usable_parity = result["parity_usable"]

print(
    f"Rows with parity calculation: "
    f"{valid_parity.sum():,}"
)

print(
    f"Rows with usable parity: "
    f"{usable_parity.sum():,}"
)

print(
    f"Usable parity rate: "
    f"{usable_parity.mean() * 100:.2f}%"
)


# ============================================================
# RESIDUAL STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("PARITY RESIDUAL")
print("=" * 80)

if valid_parity.any():

    print(
        result.loc[
            valid_parity,
            "parity_residual"
        ].describe()
    )


# ============================================================
# USABLE PARITY RESIDUAL STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("USABLE PARITY RESIDUAL")
print("=" * 80)

if usable_parity.any():

    print(
        result.loc[
            usable_parity,
            "parity_residual"
        ].describe()
    )


# ============================================================
# PARITY PERCENTAGE
# ============================================================

valid_pct = (
    result["parity_residual_pct"].notna()
    & np.isfinite(result["parity_residual_pct"])
)

print("\n" + "=" * 80)
print("PARITY RESIDUAL PERCENTAGE")
print("=" * 80)

if valid_pct.any():

    print(
        result.loc[
            valid_pct,
            "parity_residual_pct"
        ].describe()
    )


# ============================================================
# TOP ABSOLUTE PARITY RESIDUALS
# ============================================================

print("\n" + "=" * 80)
print("TOP ABSOLUTE PARITY RESIDUALS")
print("=" * 80)

top = (
    result.loc[
        valid_parity,
        [
            "timestamp",
            "expiry",
            "strike",
            "option_type",
            "call_price",
            "put_price",
            "nifty_spot",
            "parity_residual",
        ],
    ]
    .copy()
)

top["absolute_parity_residual"] = (
    top["parity_residual"].abs()
)

top = (
    top
    .sort_values(
        "absolute_parity_residual",
        ascending=False,
    )
    .head(10)
)

print(
    top.to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

result.to_parquet(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FILE)
print(f"Rows saved: {len(result):,}")
print("Put-call parity feature construction completed successfully.")