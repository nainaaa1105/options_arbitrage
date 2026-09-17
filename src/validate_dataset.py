import sys
from pathlib import Path

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import RAW_UPSTOX_DIR


def validate_dataset(file_path):

    print()
    print("FILE:", file_path.name)

    df = pd.read_parquet(file_path)

    print("Rows:", len(df))

    print(
        "Contracts:",
        df["instrument_key"].nunique()
    )

    print(
        "Strikes:",
        df["strike"].nunique()
    )

    print(
        "CE:",
        (df["option_type"] == "CE").sum()
    )

    print(
        "PE:",
        (df["option_type"] == "PE").sum()
    )

    print(
        "Trading dates:",
        df["timestamp"].dt.date.nunique()
    )

    print(
        "Date range:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max()
    )

    # Missing values
    missing = df.isna().sum()

    total_missing = int(
        missing.sum()
    )

    print(
        "Total missing values:",
        total_missing
    )

    if total_missing > 0:

        for column, count in missing.items():

            if count > 0:

                print(
                    f"   {column}: {count}"
                )

    # Duplicate contract/timestamp
    duplicates = df.duplicated(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    ).sum()

    print(
        "Duplicate contract/timestamp rows:",
        duplicates
    )

    # Negative prices
    negative = (
        df["close"] < 0
    ).sum()

    print(
        "Negative close prices:",
        negative
    )

    # Zero prices
    zero = (
        df["close"] == 0
    ).sum()

    print(
        "Zero close prices:",
        zero
    )

    # NIFTY alignment
    missing_spot = (
        df["nifty_spot"].isna()
    ).sum()

    print(
        "Missing NIFTY spot:",
        missing_spot
    )

    # Contract candle counts
    counts = (
        df.groupby(
            "instrument_key"
        )["timestamp"]
        .count()
    )

    print(
        "Min candles/contract:",
        counts.min()
    )

    print(
        "Max candles/contract:",
        counts.max()
    )

    print(
        "Average candles/contract:",
        round(
            counts.mean(),
            2
        )
    )

    # Status
    passed = (
        total_missing == 0
        and duplicates == 0
        and negative == 0
        and missing_spot == 0
    )

    if passed:

        print()
        print("STATUS: ✅ PASS")

    else:

        print()
        print("STATUS: ⚠️ REVIEW REQUIRED")

    return passed


def validate_all():

    print("VALIDATE ALL EXPIRY DATASETS")

    files = sorted(
        RAW_UPSTOX_DIR.glob(
            "expiry_*.parquet"
        )
    )

    if not files:

        print(
            "No expiry parquet files found."
        )

        return

    print(
        "Files found:",
        len(files)
    )

    results = []

    for file_path in files:

        try:

            passed = validate_dataset(
                file_path
            )

            results.append(
                {
                    "file": file_path.name,
                    "status":
                        "PASS"
                        if passed
                        else "REVIEW",
                }
            )

        except Exception as e:

            print(
                "ERROR:",
                e
            )

            results.append(
                {
                    "file": file_path.name,
                    "status": "ERROR",
                }
            )

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    summary = pd.DataFrame(
        results
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        "Passed:",
        (
            summary["status"]
            == "PASS"
        ).sum()
    )

    print(
        "Review:",
        (
            summary["status"]
            != "PASS"
        ).sum()
    )

    print()
    print("VALIDATION COMPLETE")

if __name__ == "__main__":

    validate_all()