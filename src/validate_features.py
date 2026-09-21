import numpy as np
import pandas as pd

from config import FEATURE_DATASET_FILE


def main():

    print("=" * 80)
    print("VALIDATING QUANTITATIVE FEATURE DATASET")
    print("=" * 80)

    df = pd.read_parquet(
        FEATURE_DATASET_FILE
    )

    print(f"Rows: {len(df):,}")

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "timestamp",
        "expiry",
        "strike",
        "option_type",
        "nifty_spot",
        "market_price",
        "time_to_expiry",
        "implied_volatility",
        "theoretical_price",
        "delta",
        "gamma",
        "vega",
        "theta",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise RuntimeError(
            f"Missing columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    print()
    print("BASIC CHECKS")
    print("-" * 80)

    print(
        "Missing NIFTY:",
        df["nifty_spot"].isna().sum(),
    )

    print(
        "Missing market price:",
        df["market_price"].isna().sum(),
    )

    print(
        "Missing time to expiry:",
        df["time_to_expiry"].isna().sum(),
    )

    # --------------------------------------------------------
    # IV status
    # --------------------------------------------------------

    print()
    print("IV STATUS")
    print("-" * 80)

    print(
        df["iv_status"]
        .value_counts(dropna=False)
    )

    # --------------------------------------------------------
    # IV statistics
    # --------------------------------------------------------

    valid_iv = df[
        df["implied_volatility"].notna()
    ]["implied_volatility"]

    print()
    print("VALID IV")
    print("-" * 80)

    print(valid_iv.describe())

    # --------------------------------------------------------
    # Invalid extreme values
    # --------------------------------------------------------

    extreme_iv = df[
        df["implied_volatility"]
        > 5
    ]

    print()
    print(
        f"IV > 500%: {len(extreme_iv):,}"
    )

    # --------------------------------------------------------
    # Theoretical-price reconstruction error
    # --------------------------------------------------------

    valid_theoretical = df[
        df["theoretical_price"].notna()
    ].copy()

    if len(valid_theoretical) > 0:

        error = (
            valid_theoretical["market_price"]
            - valid_theoretical["theoretical_price"]
        ).abs()

        print()
        print(
            "Max market/BS reconstruction error:",
            error.max(),
        )

    # --------------------------------------------------------
    # Greek ranges
    # --------------------------------------------------------

    print()
    print("GREEK RANGES")
    print("-" * 80)

    for column in [
        "delta",
        "gamma",
        "vega",
        "theta",
    ]:

        series = df[column].dropna()

        if len(series) == 0:
            continue

        print(
            f"{column:8s} "
            f"min={series.min():.6f} "
            f"max={series.max():.6f} "
            f"mean={series.mean():.6f}"
        )

    # --------------------------------------------------------
    # Delta sanity
    # --------------------------------------------------------

    bad_delta = df[
        df["delta"].notna()
        & (
            (df["delta"] < -1.000001)
            | (df["delta"] > 1.000001)
        )
    ]

    print()
    print(
        f"Invalid delta rows: {len(bad_delta):,}"
    )

    # --------------------------------------------------------
    # Time-to-expiry sanity
    # --------------------------------------------------------

    negative_t = df[
        df["time_to_expiry"] < 0
    ]

    print(
        f"Negative T rows: {len(negative_t):,}"
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    ).sum()

    print(
        f"Duplicate contract/timestamp rows: "
        f"{duplicates:,}"
    )

    # --------------------------------------------------------
    # Hard failures
    # --------------------------------------------------------

    if len(bad_delta) > 0:
        raise RuntimeError(
            "Invalid Delta values detected."
        )

    if len(negative_t) > 0:
        raise RuntimeError(
            "Negative time-to-expiry detected."
        )

    if duplicates > 0:
        raise RuntimeError(
            "Duplicate rows detected."
        )

    if len(extreme_iv) > 0:
        print(
            "WARNING: Extreme IV values exist."
        )

    print()
    print("=" * 80)
    print("FEATURE VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()