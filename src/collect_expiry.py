import pandas as pd

from config import (
    UPSTOX_RAW_DIR,
    OPTION_TIMEFRAME,
)

from prepare_expiry import (
    prepare_expiry,
)

from upstox_api import (
    get_expired_candles,
)


# ============================================================
# COLLECT ONE EXPIRY
# ============================================================

def collect_expiry(
    expiry_date,
    number_of_sessions=5,
):
    """
    Collect exactly N trading sessions for one expiry.

    Dynamic strike selection is performed independently
    for every trading session.
    """

    print()
    print("=" * 80)
    print(
        f"COLLECTING EXPIRY: {expiry_date}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Prepare expiry
    # --------------------------------------------------------

    prepared = prepare_expiry(
        expiry_date,
        number_of_sessions,
    )

    trading_dates = prepared[
        "trading_dates"
    ]

    required_contracts = prepared[
        "contracts"
    ]

    daily_selection = prepared[
        "daily_selection"
    ]

    nifty_data = prepared[
        "nifty_data"
    ]

    # --------------------------------------------------------
    # Date range
    # --------------------------------------------------------

    from_date = trading_dates[0]

    to_date = trading_dates[-1]

    print()
    print(
        f"Historical window:"
        f" {from_date} → {to_date}"
    )

    # --------------------------------------------------------
    # Download option candles
    # --------------------------------------------------------

    option_frames = []

    failed_contracts = []

    print()
    print(
        f"Downloading "
        f"{len(required_contracts)} "
        f"contracts..."
    )

    for index, contract in enumerate(
        required_contracts,
        start=1,
    ):

        instrument_key = contract[
            "instrument_key"
        ]

        strike = float(
            contract["strike_price"]
        )

        option_type = contract[
            "instrument_type"
        ]

        trading_symbol = contract.get(
            "trading_symbol",
            "",
        )

        print(
            f"[{index}/"
            f"{len(required_contracts)}] "
            f"{trading_symbol}"
        )

        try:

            candles = get_expired_candles(
                instrument_key,
                from_date,
                to_date,
            )

            if not candles:

                print(
                    "  WARNING: no candles"
                )

                failed_contracts.append(
                    instrument_key
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

            if df.empty:
                continue

            df["timestamp"] = pd.to_datetime(
                df["timestamp"]
            )

            df["instrument_key"] = (
                instrument_key
            )

            df["trading_symbol"] = (
                trading_symbol
            )

            df["expiry"] = pd.to_datetime(
                expiry_date
            )

            df["strike"] = strike

            df["option_type"] = (
                option_type
            )

            df["lot_size"] = int(
                contract.get(
                    "lot_size",
                    0,
                )
            )

            option_frames.append(df)

        except Exception as e:

            print(
                f"  ERROR: {e}"
            )

            failed_contracts.append(
                instrument_key
            )

    # --------------------------------------------------------
    # Make option dataframe
    # --------------------------------------------------------

    if not option_frames:

        raise RuntimeError(
            f"No option data collected "
            f"for {expiry_date}"
        )

    options_df = pd.concat(
        option_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Remove duplicate candles
    # --------------------------------------------------------

    options_df = (
        options_df
        .drop_duplicates(
            subset=[
                "instrument_key",
                "timestamp",
            ]
        )
        .sort_values(
            [
                "instrument_key",
                "timestamp",
            ]
        )
        .reset_index(drop=True)
    )

    print()
    print(
        f"Raw option rows: "
        f"{len(options_df):,}"
    )

    print(
        f"Contracts with data: "
        f"{options_df['instrument_key'].nunique()}"
    )

    # --------------------------------------------------------
    # Build NIFTY dataframe
    # --------------------------------------------------------

    nifty_frames = []

    for date in trading_dates:

        nifty_df = nifty_data.get(
            date
        )

        if nifty_df is None:
            continue

        temp = nifty_df.copy()

        temp["trade_date"] = date

        nifty_frames.append(temp)

    if not nifty_frames:

        raise RuntimeError(
            "No NIFTY data available."
        )

    nifty_df = pd.concat(
        nifty_frames,
        ignore_index=True,
    )

    nifty_df = nifty_df[
        [
            "timestamp",
            "close",
        ]
    ].rename(
        columns={
            "close": "nifty_spot"
        }
    )

    # --------------------------------------------------------
    # Align NIFTY spot to option timestamps
    # --------------------------------------------------------

# Options data = 5-minute candles
# NIFTY data    = 1-minute candles
#
# IMPORTANT:
# The option API may contain a 15:35 candle, while
# NIFTY spot data does not have a corresponding candle.
#
# Therefore, remove option timestamps that occur after
# the last available NIFTY timestamp FOR THAT DAY.

    options_df = options_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    nifty_df = nifty_df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

# Add trading date to both datasets
    options_df["trade_date"] = (
        options_df["timestamp"]
        .dt.date
        .astype(str)
    )

    nifty_df["trade_date"] = (
        nifty_df["timestamp"]
        .dt.date
        .astype(str)
    )

# Find the last available NIFTY timestamp for each day
    last_nifty_time = (
        nifty_df
        .groupby("trade_date")["timestamp"]
        .max()
        .rename("last_nifty_timestamp")
    )

# Attach the daily cutoff to option rows
    options_df = options_df.merge(
        last_nifty_time,
        on="trade_date",
        how="left",
    )

# Keep only option candles that have an actual NIFTY
# observation available for that same trading session
    options_df = options_df[
        options_df["timestamp"]
        <= options_df["last_nifty_timestamp"]
        ].copy()

    options_df = options_df.drop(
    columns=["last_nifty_timestamp"]
    )

# Remove temporary trade_date because it will be
# recreated later
    options_df = options_df.drop(
        columns=["trade_date"]
    )

    nifty_df = nifty_df.drop(
        columns=["trade_date"]
    )

# Final timestamp alignment
    merged = pd.merge_asof(
        options_df.sort_values("timestamp"),
        nifty_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=1),
    )
# --------------------------------------------------------
# Remove option candles after the NIFTY trading session
# --------------------------------------------------------
# Upstox expired option data can contain a 15:35 candle,
# while NIFTY spot data ends before that timestamp.
#
# We do not want to invent/fill a NIFTY spot value for it.
# Therefore keep only option timestamps that can be
# matched to actual NIFTY data.

    latest_nifty_timestamp = nifty_df["timestamp"].max()

    options_df = options_df[
        options_df["timestamp"] <= latest_nifty_timestamp
    ].copy()

# --------------------------------------------------------
# Align NIFTY spot to option timestamps
# --------------------------------------------------------

    merged = pd.merge_asof(
        options_df,
        nifty_df,
        on="timestamp",
        direction="backward",
        tolerance=pd.Timedelta(minutes=1),
)
    # --------------------------------------------------------
    # Daily dynamic strike filter
    # --------------------------------------------------------

    selection_map = {}

    for item in daily_selection:

        selection_map[
            item["date"]
        ] = set(
            float(x)
            for x in item[
                "selected_strikes"
            ]
        )

    merged["trade_date"] = (
        merged["timestamp"]
        .dt.date
        .astype(str)
    )

    merged = merged[
        merged.apply(
            lambda row:
                float(row["strike"])
                in selection_map.get(
                    row["trade_date"],
                    set(),
                ),
            axis=1,
        )
    ].copy()

    # --------------------------------------------------------
    # IMPORTANT:
    # 09:15 is retained in RAW dataset.
    #
    # It will be excluded from model analysis later
    # because it was used for daily strike selection.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    merged = merged.sort_values(
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
        UPSTOX_RAW_DIR
        / f"expiry_{expiry_date}.parquet"
    )

    merged.to_parquet(
        output_file,
        index=False,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("EXPIRY VALIDATION")
    print("=" * 80)

    print(
        f"Rows: {len(merged):,}"
    )

    print(
        f"Contracts: "
        f"{merged['instrument_key'].nunique()}"
    )

    print(
        f"Strikes: "
        f"{merged['strike'].nunique()}"
    )

    print(
        f"CE: "
        f"{(merged['option_type'] == 'CE').sum()}"
    )

    print(
        f"PE: "
        f"{(merged['option_type'] == 'PE').sum()}"
    )

    print(
        f"Trading dates: "
        f"{merged['trade_date'].nunique()}"
    )

    print(
        f"Missing NIFTY spot: "
        f"{merged['nifty_spot'].isna().sum()}"
    )

    duplicates = merged.duplicated(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    ).sum()

    print(
        f"Duplicate contract/timestamp: "
        f"{duplicates}"
    )

    print(
        f"Failed contracts: "
        f"{len(failed_contracts)}"
    )

    print(
        f"Saved: {output_file}"
    )

    # --------------------------------------------------------
    # Hard validation
    # --------------------------------------------------------

    if len(trading_dates) != number_of_sessions:

        raise RuntimeError(
            "Incorrect number of trading sessions."
        )

    if merged["nifty_spot"].isna().any():

        raise RuntimeError(
            "Missing NIFTY spot detected."
        )

    if duplicates > 0:

        raise RuntimeError(
            "Duplicate contract/timestamp detected."
        )

    print()
    print(
        "EXPIRY COLLECTION PASSED"
    )

    return merged


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    collect_expiry(
        "2025-09-23",
        number_of_sessions=5,
    )