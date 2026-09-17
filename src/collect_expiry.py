import sys
from pathlib import Path

import pandas as pd



# PATH SETUP


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    TEST_EXPIRY,
    TEST_FROM_DATE,
    TEST_TO_DATE,
    RAW_UPSTOX_DIR,
)

from upstox_api import (
    get_expired_candles,
)

from prepare_expiry import (
    prepare_expiry,
)



# OPTION CANDLES → DATAFRAME


def candles_to_dataframe(
    candles,
    contract,
):
    """
    Convert Upstox option candles into DataFrame.
    """

    if not candles:
        return pd.DataFrame()

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["instrument_key"] = contract[
        "instrument_key"
    ]

    df["trading_symbol"] = contract[
        "trading_symbol"
    ]

    df["expiry"] = pd.to_datetime(
        contract["expiry"]
    )

    df["strike"] = float(
        contract["strike_price"]
    )

    df["option_type"] = contract[
        "instrument_type"
    ]

    df["lot_size"] = int(
        contract["lot_size"]
    )

    return df



# COLLECT ONE EXPIRY


def collect_expiry(
    expiry_date,
    from_date,
    to_date,
):
    print()
    print("=" * 70)
    print("COLLECT EXPIRY")
    print("=" * 70)

    print("Expiry:", expiry_date)
    print("Window:", from_date, "→", to_date)

    # --------------------------------------------------------
    # Prepare daily strike universe
    # --------------------------------------------------------

    prepared = prepare_expiry(
        expiry_date,
        from_date,
        to_date,
    )

    contracts = prepared[
        "required_contracts"
    ]

    daily_info = prepared[
        "daily_info"
    ]

    # --------------------------------------------------------
    # Download option candles
    # --------------------------------------------------------

    all_frames = []
    failed = []

    print()
    print("=" * 70)
    print("DOWNLOADING OPTION DATA")
    print("=" * 70)

    print(
        "Contracts to download:",
        len(contracts)
    )

    for index, contract in enumerate(
        contracts,
        start=1,
    ):

        instrument_key = contract[
            "instrument_key"
        ]

        symbol = contract[
            "trading_symbol"
        ]

        print(
            f"[{index}/{len(contracts)}] "
            f"{symbol}"
        )

        try:

            candles = get_expired_candles(
                instrument_key,
                from_date,
                to_date,
            )

            df = candles_to_dataframe(
                candles,
                contract,
            )

            if df.empty:

                print(
                    "   WARNING: no candles"
                )

                failed.append(
                    {
                        "instrument_key":
                            instrument_key,
                        "trading_symbol":
                            symbol,
                        "reason":
                            "no candles",
                    }
                )

                continue

            all_frames.append(df)

            print(
                "   rows:",
                len(df)
            )

        except Exception as e:

            print(
                "   FAILED:",
                e
            )

            failed.append(
                {
                    "instrument_key":
                        instrument_key,
                    "trading_symbol":
                        symbol,
                    "reason":
                        str(e),
                }
            )

    if not all_frames:
        raise RuntimeError(
            "No option data collected."
        )

    options_df = pd.concat(
        all_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    options_df = options_df.drop_duplicates(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    )

    # --------------------------------------------------------
    # Download NIFTY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DOWNLOADING NIFTY DATA")
    print("=" * 70)

    nifty_frames = []

    for day in daily_info:

        date = day["date"]

        print(
            "Downloading NIFTY:",
            date
        )

        from upstox_api import (
            get_nifty_candles
        )

        candles = get_nifty_candles(
            date
        )

        if not candles:
            print(
                "   WARNING: no NIFTY data"
            )
            continue

        df = pd.DataFrame(
            candles,
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "open_interest",
            ],
        )

        df["timestamp"] = pd.to_datetime(
            df["timestamp"]
        )

        nifty_frames.append(
            df[
                [
                    "timestamp",
                    "close",
                ]
            ].rename(
                columns={
                    "close":
                        "nifty_spot"
                }
            )
        )

    if not nifty_frames:
        raise RuntimeError(
            "No NIFTY data collected."
        )

    nifty_df = pd.concat(
        nifty_frames,
        ignore_index=True,
    )

    nifty_df = nifty_df.drop_duplicates(
        subset=["timestamp"]
    )

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("MERGING OPTION + NIFTY DATA")
    print("=" * 70)

    merged = options_df.merge(
        nifty_df,
        on="timestamp",
        how="left",
    )

    # --------------------------------------------------------
    # Daily dynamic strike filtering
    # --------------------------------------------------------

    print()
    print(
        "Applying daily dynamic strike filter..."
    )

    filtered_parts = []

    for day in daily_info:

        date = day["date"]

        selected_strikes = set(
            float(x)
            for x in day[
                "selected_strikes"
            ]
        )

        date_mask = (
            merged["timestamp"]
            .dt.strftime("%Y-%m-%d")
            == date
        )

        strike_mask = (
            merged["strike"]
            .isin(selected_strikes)
        )

        part = merged[
            date_mask & strike_mask
        ].copy()

        filtered_parts.append(part)

    final_df = pd.concat(
        filtered_parts,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    final_df = final_df.sort_values(
        [
            "timestamp",
            "strike",
            "option_type",
        ]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        RAW_UPSTOX_DIR
        / f"expiry_{expiry_date}.parquet"
    )

    final_df.to_parquet(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print(
        "Rows:",
        len(final_df)
    )

    print(
        "Contracts:",
        final_df[
            "instrument_key"
        ].nunique()
    )

    print(
        "Strikes:",
        final_df[
            "strike"
        ].nunique()
    )

    print(
        "CE rows:",
        (
            final_df["option_type"]
            == "CE"
        ).sum()
    )

    print(
        "PE rows:",
        (
            final_df["option_type"]
            == "PE"
        ).sum()
    )

    print(
        "Missing NIFTY spot:",
        final_df[
            "nifty_spot"
        ].isna().sum()
    )

    print(
        "Duplicate rows:",
        final_df.duplicated(
            subset=[
                "instrument_key",
                "timestamp",
            ]
        ).sum()
    )

    print(
        "Date range:",
        final_df["timestamp"].min(),
        "→",
        final_df["timestamp"].max(),
    )

    print(
        "Failed contracts:",
        len(failed)
    )

    if failed:

        print()
        print("FAILED CONTRACTS:")

        for item in failed:
            print(
                item["trading_symbol"],
                "→",
                item["reason"]
            )

    print()
    print("DATASET SAVED")
 

    print(
        output_file
    )

    return final_df

# TEST

if __name__ == "__main__":

    collect_expiry(
        TEST_EXPIRY,
        TEST_FROM_DATE,
        TEST_TO_DATE,
    )