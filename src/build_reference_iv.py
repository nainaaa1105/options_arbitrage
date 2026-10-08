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
    / "options_volatility_features.parquet"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "options_reference_iv.parquet"
)

# Number of neighboring observations used on each side
NEIGHBOR_COUNT = 3

# Minimum valid IV observations required for interpolation
MIN_VALID_POINTS = 4


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("BUILDING REFERENCE IV SURFACE")
print("=" * 80)

print(f"Input:  {INPUT_FILE}")
print(f"Output: {OUTPUT_FILE}")

df = pd.read_parquet(INPUT_FILE)

print(f"\nInput rows: {len(df):,}")


# ============================================================
# PREPARE
# ============================================================

required_columns = [
    "timestamp",
    "trade_date",
    "option_type",
    "strike",
    "nifty_spot",
    "time_to_expiry",
    "implied_volatility",
    "iv_status",
    "market_price",
    "log_moneyness",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    [
        "timestamp",
        "option_type",
        "strike",
    ]
).reset_index(drop=True)


# ============================================================
# REFERENCE IV FUNCTION
# ============================================================

def calculate_reference_iv(group):

    group = group.copy()

    iv = group["implied_volatility"].to_numpy(
        dtype=float
    )

    strike = group["strike"].to_numpy(
        dtype=float
    )

    valid = (
        np.isfinite(iv)
        &
        (iv > 0)
        &
        (iv <= 5.0)
    )

    reference_iv = np.full(
        len(group),
        np.nan,
        dtype=float
    )

    valid_indices = np.where(valid)[0]

    if len(valid_indices) < MIN_VALID_POINTS:
        group["reference_iv"] = reference_iv
        return group

    valid_strikes = strike[valid]

    valid_ivs = iv[valid]

    # --------------------------------------------------------
    # Interpolate across strike
    # --------------------------------------------------------

    for i in range(len(group)):

        # Do not use the option's own IV directly
        other_mask = valid_indices != i

        other_indices = valid_indices[
            other_mask
        ]

        if len(other_indices) < MIN_VALID_POINTS:
            continue

        other_strikes = strike[
            other_indices
        ]

        other_ivs = iv[
            other_indices
        ]

        target_strike = strike[i]

        # Sort by distance from target
        distances = np.abs(
            other_strikes - target_strike
        )

        order = np.argsort(distances)

        selected = order[
            :min(
                NEIGHBOR_COUNT * 2,
                len(order)
            )
        ]

        x = other_strikes[selected]
        y = other_ivs[selected]

        # Need enough distinct strikes
        if len(np.unique(x)) < 2:
            continue

        # Sort x for interpolation
        sort_order = np.argsort(x)

        x = x[sort_order]
        y = y[sort_order]

        # ----------------------------------------------------
        # Linear interpolation
        # ----------------------------------------------------
        #
        # If target lies inside the neighboring strike range,
        # interpolate.
        #
        # If it is outside, use the nearest valid observation
        # only when the distance is small.
        # ----------------------------------------------------

        if (
            target_strike >= x.min()
            and target_strike <= x.max()
        ):

            reference = np.interp(
                target_strike,
                x,
                y
            )

        else:

            nearest = np.argmin(
                np.abs(x - target_strike)
            )

            reference = y[nearest]

        reference_iv[i] = reference

    group["reference_iv"] = reference_iv

    return group


# ============================================================
# APPLY BY TIMESTAMP + OPTION TYPE
# ============================================================

print(
    "\nCalculating leave-one-out reference IV..."
)

groups = []

for (timestamp, option_type), group in df.groupby(
    ["timestamp", "option_type"],
    sort=False
):
    result = calculate_reference_iv(group)

    result["timestamp"] = timestamp
    result["option_type"] = option_type

    groups.append(result)

df = pd.concat(
    groups,
    ignore_index=True
)


# ============================================================
# REFERENCE IV DIFFERENCE
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


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("REFERENCE IV SUMMARY")
print("=" * 80)

valid_reference = (
    df["reference_iv"].notna()
)

print(
    f"Valid reference IV: "
    f"{valid_reference.sum():,}"
)

print(
    f"Missing reference IV: "
    f"{(~valid_reference).sum():,}"
)

print("\nReference IV statistics:")

print(
    df["reference_iv"]
    .describe()
    .to_string()
)

print("\nIV surface residual statistics:")

print(
    df["iv_surface_residual"]
    .describe()
    .to_string()
)


# ============================================================
# SAVE
# ============================================================

df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("REFERENCE IV BUILD COMPLETE")

print(
    f"Saved: {OUTPUT_FILE}"
)