from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "options_deviation_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "options_signal_eligible.parquet"


# ============================================================
# CONFIG
# ============================================================

MIN_MARKET_PRICE = 1.0
MIN_REFERENCE_PRICE = 1.0
MIN_TIME_TO_EXPIRY = 0.25

MIN_LOG_MONEYNESS = -0.15
MAX_LOG_MONEYNESS = 0.15

MIN_IV = 0.01
MAX_IV = 3.0

MIN_VOLUME = 1
MIN_OPEN_INTEREST = 1

# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BUILDING SIGNAL ELIGIBILITY DATASET")
print("=" * 80)

df = pd.read_parquet(INPUT_FILE)

print(f"Input rows: {len(df):,}")


# ============================================================
# BASIC VALIDITY
# ============================================================

df["valid_reference"] = (
    df["reference_iv"].notna()
    & df["reference_theoretical_price"].notna()
    & (df["reference_iv"] > 0)
    & (df["reference_theoretical_price"] >= 0)
)

df["valid_market_price"] = (
    df["market_price"].notna()
    & (df["market_price"] >= MIN_MARKET_PRICE)
)

df["valid_spot"] = (
    df["nifty_spot"].notna()
    & (df["nifty_spot"] > 0)
)

df["valid_strike"] = (
    df["strike"].notna()
    & (df["strike"] > 0)
)


# ============================================================
# TIME TO EXPIRY
# ============================================================

df["expiry_usable"] = (
    df["time_to_expiry"].notna()
    & (df["time_to_expiry"] >= MIN_TIME_TO_EXPIRY / 365.0)
)


# ============================================================
# REFERENCE PRICE USABILITY
# ============================================================

df["reference_price_usable"] = (
    df["reference_theoretical_price"].notna()
    & (df["reference_theoretical_price"] >= MIN_REFERENCE_PRICE)
)


# ============================================================
# IV USABILITY
# ============================================================

df["iv_usable"] = (
    df["implied_volatility"].notna()
    & (df["implied_volatility"] >= MIN_IV)
    & (df["implied_volatility"] <= MAX_IV)
)

df["reference_iv_usable"] = (
    df["reference_iv"].notna()
    & (df["reference_iv"] >= MIN_IV)
    & (df["reference_iv"] <= MAX_IV)
)


# ============================================================
# MONEYNESS
# ============================================================

df["moneyness_usable"] = (
    df["log_moneyness"].notna()
    & (df["log_moneyness"] >= MIN_LOG_MONEYNESS)
    & (df["log_moneyness"] <= MAX_LOG_MONEYNESS)
)


# ============================================================
# LIQUIDITY
# ============================================================

df["volume_usable"] = (
    df["volume"].fillna(0) >= MIN_VOLUME
)

df["open_interest_usable"] = (
    df["open_interest"].fillna(0) >= MIN_OPEN_INTEREST
)

df["liquid"] = (
    df["volume_usable"]
    & df["open_interest_usable"]
)


# ============================================================
# DEVIATION USABILITY
# ============================================================

df["deviation_usable"] = (
    df["reference_price_deviation"].notna()
    & df["reference_price_deviation_pct"].notna()
    & np.isfinite(df["reference_price_deviation"])
    & np.isfinite(df["reference_price_deviation_pct"])
)


# ============================================================
# OVERALL SIGNAL ELIGIBILITY
# ============================================================

df["signal_eligible"] = (
    df["valid_reference"]
    & df["valid_market_price"]
    & df["valid_spot"]
    & df["valid_strike"]
    & df["expiry_usable"]
    & df["reference_price_usable"]
    & df["iv_usable"]
    & df["reference_iv_usable"]
    & df["moneyness_usable"]
    & df["liquid"]
    & df["deviation_usable"]
)


# ============================================================
# REASON FOR INELIGIBILITY
# ============================================================

def get_reason(row):

    reasons = []

    if not row["valid_reference"]:
        reasons.append("invalid_reference")

    if not row["valid_market_price"]:
        reasons.append("low_market_price")

    if not row["valid_spot"]:
        reasons.append("invalid_spot")

    if not row["valid_strike"]:
        reasons.append("invalid_strike")

    if not row["expiry_usable"]:
        reasons.append("near_expiry")

    if not row["reference_price_usable"]:
        reasons.append("low_reference_price")

    if not row["iv_usable"]:
        reasons.append("invalid_iv")

    if not row["reference_iv_usable"]:
        reasons.append("invalid_reference_iv")

    if not row["moneyness_usable"]:
        reasons.append("extreme_moneyness")

    if not row["liquid"]:
        reasons.append("illiquid")

    if not row["deviation_usable"]:
        reasons.append("invalid_deviation")

    if not reasons:
        return "eligible"

    return "|".join(reasons)


df["eligibility_reason"] = df.apply(get_reason, axis=1)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("ELIGIBILITY SUMMARY")
print("=" * 80)

total = len(df)
eligible = int(df["signal_eligible"].sum())

print(f"Total rows:       {total:,}")
print(f"Eligible rows:    {eligible:,}")
print(f"Ineligible rows:  {total - eligible:,}")
print(f"Eligibility rate: {eligible / total * 100:.2f}%")


# ============================================================
# INDIVIDUAL FLAGS
# ============================================================

flags = [
    "valid_reference",
    "valid_market_price",
    "valid_spot",
    "valid_strike",
    "expiry_usable",
    "reference_price_usable",
    "iv_usable",
    "reference_iv_usable",
    "moneyness_usable",
    "volume_usable",
    "open_interest_usable",
    "liquid",
    "deviation_usable",
]

print("\n" + "=" * 80)
print("INDIVIDUAL FLAG COUNTS")
print("=" * 80)

for flag in flags:
    count = int(df[flag].sum())
    pct = count / total * 100
    print(f"{flag:<30} {count:>10,}  ({pct:>6.2f}%)")


# ============================================================
# INELIGIBILITY REASONS
# ============================================================

print("\n" + "=" * 80)
print("TOP INELIGIBILITY REASONS")
print("=" * 80)

reasons = (
    df.loc[~df["signal_eligible"], "eligibility_reason"]
    .value_counts()
    .head(20)
)

for reason, count in reasons.items():
    print(f"{reason:<50} {count:>10,}")


# ============================================================
# ELIGIBLE DATASET SUMMARY
# ============================================================

eligible_df = df[df["signal_eligible"]].copy()

print("\n" + "=" * 80)
print("ELIGIBLE DATASET")
print("=" * 80)

if len(eligible_df) > 0:

    print(f"Rows: {len(eligible_df):,}")

    print("\nOption type:")
    print(eligible_df["option_type"].value_counts())

    print("\nExpiry count:")
    print(eligible_df["expiry"].nunique())

    print("\nTrading dates:")
    print(eligible_df["trade_date"].nunique())

    print("\nMarket price:")
    print(eligible_df["market_price"].describe())

    print("\nReference deviation:")
    print(eligible_df["reference_price_deviation"].describe())

    print("\nIV surface residual:")
    print(eligible_df["iv_surface_residual"].describe())


# ============================================================
# SAVE
# ============================================================

df.to_parquet(OUTPUT_FILE, index=False)

print("\n" + "=" * 80)
print("SAVED")
print("=" * 80)

print(OUTPUT_FILE)
print(f"Rows saved: {len(df):,}")
print("Signal eligibility layer completed successfully.")