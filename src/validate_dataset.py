from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_UPSTOX_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "upstox"
)


# ============================================================
# VALIDATE ALL EXPIRY FILES
# ============================================================

def validate_file(file):

    print()
    print("=" * 70)
    print(f"VALIDATING: {file.name}")
    print("=" * 70)

    df = pd.read_parquet(file)

    issues = []

    # --------------------------------------------------------
    # Basic checks
    # --------------------------------------------------------

    if df.empty:
        issues.append("Dataset is empty")

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest",
        "instrument_key",
        "trading_symbol",
        "expiry",
        "strike",
        "option_type",
        "lot_size",
        "nifty_spot",
        "trade_date",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        issues.append(
            f"Missing columns: {missing_columns}"
        )

    if issues:
        print("REVIEW")
        for issue in issues:
            print(f"  - {issue}")
        return False

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    duplicates = df.duplicated(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    ).sum()

    if duplicates > 0:
        issues.append(
            f"Duplicate contract/timestamp rows: "
            f"{duplicates}"
        )

    # --------------------------------------------------------
    # Missing NIFTY
    # --------------------------------------------------------

    missing_nifty = df[
        "nifty_spot"
    ].isna().sum()

    if missing_nifty > 0:
        issues.append(
            f"Missing NIFTY spot rows: "
            f"{missing_nifty}"
        )

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing_values = (
        df[required_columns]
        .isna()
        .sum()
        .sum()
    )

    if missing_values > 0:
        issues.append(
            f"Missing values in required columns: "
            f"{missing_values}"
        )

    # --------------------------------------------------------
    # Negative prices
    # --------------------------------------------------------

    negative_prices = (
        df["close"] < 0
    ).sum()

    if negative_prices > 0:
        issues.append(
            f"Negative option prices: "
            f"{negative_prices}"
        )

    # --------------------------------------------------------
    # Option types
    # --------------------------------------------------------

    option_types = set(
        df["option_type"].dropna().unique()
    )

    invalid_types = option_types - {
        "CE",
        "PE",
    }

    if invalid_types:
        issues.append(
            f"Invalid option types: "
            f"{invalid_types}"
        )

    # --------------------------------------------------------
    # Trading sessions
    # --------------------------------------------------------

    trading_sessions = (
        df["trade_date"]
        .nunique()
    )

    if trading_sessions != 5:
        issues.append(
            f"Expected 5 trading sessions, "
            f"found {trading_sessions}"
        )

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    min_date = df["trade_date"].min()
    max_date = df["trade_date"].max()

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"Contracts: "
        f"{df['instrument_key'].nunique()}"
    )

    print(
        f"Strikes: "
        f"{df['strike'].nunique()}"
    )

    print(
        f"CE rows: "
        f"{(df['option_type'] == 'CE').sum():,}"
    )

    print(
        f"PE rows: "
        f"{(df['option_type'] == 'PE').sum():,}"
    )

    print(
        f"Trading sessions: "
        f"{trading_sessions}"
    )

    print(
        f"Date range: "
        f"{min_date} → {max_date}"
    )

    print(
        f"Missing NIFTY: "
        f"{missing_nifty}"
    )

    print(
        f"Duplicate rows: "
        f"{duplicates}"
    )

    print(
        f"Negative prices: "
        f"{negative_prices}"
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if issues:

        print()
        print("REVIEW")

        for issue in issues:
            print(
                f"  - {issue}"
            )

        return False

    print()
    print("PASS")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("DATASET VALIDATION")
    print("=" * 70)

    files = sorted(
        RAW_UPSTOX_DIR.glob(
            "expiry_*.parquet"
        )
    )

    if not files:

        raise RuntimeError(
            "No expiry_*.parquet files found."
        )

    print(
        f"Files found: {len(files)}"
    )

    passed = 0
    review = 0

    results = []

    for file in files:

        success = validate_file(
            file
        )

        if success:

            passed += 1
            status = "PASS"

        else:

            review += 1
            status = "REVIEW"

        results.append(
            (
                file.name,
                status,
            )
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(
        f"{'file':35} status"
    )

    for filename, status in results:

        print(
            f"{filename:35} {status}"
        )

    print()
    print(
        f"Passed: {passed}"
    )

    print(
        f"Review: {review}"
    )

    print()
    print(
        "VALIDATION COMPLETE"
    )


if __name__ == "__main__":
    main()