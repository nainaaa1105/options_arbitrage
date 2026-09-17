import pandas as pd
from pathlib import Path

from config import (
    TEST_EXPIRY,
    TEST_FROM_DATE,
    TEST_TO_DATE,
    RAW_UPSTOX_DIR,
    STRIKES_EACH_SIDE
)

from upstox_api import (
    get_expired_contracts,
    get_nifty_candles,
    get_expired_candles
)

from strike_selector import (
    get_atm_strike,
    select_strikes,
    filter_contracts_by_strikes
)



# HELPER FUNCTIONS


def candles_to_dataframe(candles, contract):
    """
    Convert Upstox option candles into a DataFrame.
    """

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest"
    ]

    df = pd.DataFrame(candles, columns=columns)

    # Convert timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Add contract information
    df["instrument_key"] = contract["instrument_key"]
    df["trading_symbol"] = contract["trading_symbol"]
    df["expiry"] = pd.to_datetime(contract["expiry"])
    df["strike"] = contract["strike_price"]
    df["option_type"] = contract["instrument_type"]
    df["lot_size"] = contract["lot_size"]

    return df


def get_nifty_dataframe(date):
    """
    Download one day of 1-minute NIFTY data.
    """

    candles = get_nifty_candles(date)

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest"
    ]

    df = pd.DataFrame(candles, columns=columns)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.rename(
        columns={
            "close": "nifty_spot"
        }
    )

    return df[
        [
            "timestamp",
            "nifty_spot"
        ]
    ]


def get_reference_spot(from_date):
    """
    Get the first available NIFTY spot price at 09:15.

    Used for the initial ATM selection.
    """

    df = get_nifty_dataframe(from_date)

    if df.empty:
        raise ValueError(
            f"No NIFTY data available for {from_date}"
        )

    first_timestamp = df["timestamp"].min()

    first_row = df[
        df["timestamp"] == first_timestamp
    ]

    if first_row.empty:
        raise ValueError(
            f"Could not find first NIFTY candle for {from_date}"
        )

    return float(first_row["nifty_spot"].iloc[0])


def clean_option_data(df):
    """
    Basic cleaning and sorting.
    """

    df = df.copy()

    # Remove exact duplicates
    df = df.drop_duplicates(
        subset=[
            "instrument_key",
            "timestamp"
        ]
    )

    # Sort
    df = df.sort_values(
        [
            "timestamp",
            "instrument_key"
        ]
    )

    # Reset index
    df = df.reset_index(drop=True)

    return df


def attach_nifty_spot(option_df, nifty_df):
    """
    Attach NIFTY spot to option observations
    using exact timestamps.

    Both datasets use IST timestamps.
    """

    merged = option_df.merge(
        nifty_df,
        on="timestamp",
        how="left"
    )

    return merged



# MAIN COLLECTION


def main():

    print("=" * 70)
    print("MULTI-DAY HISTORICAL OPTION COLLECTOR")
    print("=" * 70)

    print("\nExpiry:", TEST_EXPIRY)
    print("From:", TEST_FROM_DATE)
    print("To:", TEST_TO_DATE)

    
    # 1. Get expired contracts
    

    print("\n[1/6] Getting expired contracts...")

    contracts = get_expired_contracts(
        TEST_EXPIRY
    )

    print("Total contracts:", len(contracts))

    
    # 2. Get reference NIFTY spot
    

    print("\n[2/6] Getting reference NIFTY spot...")

    reference_spot = get_reference_spot(
        TEST_FROM_DATE
    )

    print("Reference NIFTY spot:", reference_spot)

    
    # 3. Find available strikes
    

    available_strikes = sorted(
        set(
            contract["strike_price"]
            for contract in contracts
            if contract.get("instrument_type") in {"CE", "PE"}
        )
    )

    
    # 4. Determine ATM and select strikes
    

    atm_strike = get_atm_strike(
        reference_spot,
        available_strikes
    )

    selected_strikes = select_strikes(
        atm_strike,
        available_strikes,
        STRIKES_EACH_SIDE
    )

    selected_contracts = filter_contracts_by_strikes(
        contracts,
        selected_strikes
    )

    print("\n[3/6] Strike selection")

    print("ATM strike:", atm_strike)
    print("Selected strikes:", len(selected_strikes))
    print("Selected contracts:", len(selected_contracts))

    ce_count = sum(
        1
        for contract in selected_contracts
        if contract["instrument_type"] == "CE"
    )

    pe_count = sum(
        1
        for contract in selected_contracts
        if contract["instrument_type"] == "PE"
    )

    print("CE:", ce_count)
    print("PE:", pe_count)

    
    # 5. Download option candles
    

    print("\n[4/6] Downloading option candles...")

    all_option_data = []

    failed = []

    total = len(selected_contracts)

    for i, contract in enumerate(
        selected_contracts,
        start=1
    ):

        print(
            f"\rDownloading {i}/{total}: "
            f"{contract['trading_symbol']}",
            end=""
        )

        try:

            candles = get_expired_candles(
                contract["instrument_key"],
                TEST_FROM_DATE,
                TEST_TO_DATE
            )

            if not candles:
                failed.append(
                    (
                        contract["instrument_key"],
                        "No candles returned"
                    )
                )
                continue

            df = candles_to_dataframe(
                candles,
                contract
            )

            all_option_data.append(df)

        except Exception as e:

            failed.append(
                (
                    contract["instrument_key"],
                    str(e)
                )
            )

    print()

    if not all_option_data:
        raise RuntimeError(
            "No option data was collected."
        )

    option_df = pd.concat(
        all_option_data,
        ignore_index=True
    )

    print(
        "Raw option rows:",
        len(option_df)
    )

    
    # 6. Download NIFTY data for each trading day
    

    print("\n[5/6] Downloading NIFTY spot data...")

    option_dates = sorted(
        option_df["timestamp"]
        .dt.strftime("%Y-%m-%d")
        .unique()
    )

    nifty_data = []

    for date in option_dates:

        print(
            f"Downloading NIFTY: {date}"
        )

        try:

            nifty_df = get_nifty_dataframe(
                date
            )

            nifty_data.append(
                nifty_df
            )

        except Exception as e:

            print(
                f"Failed NIFTY {date}: {e}"
            )

    if not nifty_data:
        raise RuntimeError(
            "No NIFTY data was collected."
        )

    nifty_df = pd.concat(
        nifty_data,
        ignore_index=True
    )

    
    # Attach NIFTY spot
    

    print("\n[6/6] Attaching NIFTY spot...")

    option_df = clean_option_data(
        option_df
    )

    final_df = attach_nifty_spot(
        option_df,
        nifty_df
    )

    
    # Final sorting
    

    final_df = final_df.sort_values(
        [
            "timestamp",
            "strike",
            "option_type"
        ]
    ).reset_index(drop=True)

    
    # Save
    

    RAW_UPSTOX_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        RAW_UPSTOX_DIR
        / f"historical_{TEST_EXPIRY}_window.parquet"
    )

    final_df.to_parquet(
        output_file,
        index=False
    )

    
    # SUMMARY
    

    print("\n")
    print("=" * 70)
    print("COLLECTION COMPLETE")
    print("=" * 70)

    print("Output:", output_file)
    print("Rows:", len(final_df))
    print(
        "Contracts:",
        final_df["instrument_key"].nunique()
    )
    print(
        "Strikes:",
        final_df["strike"].nunique()
    )
    print(
        "CE:",
        (final_df["option_type"] == "CE").sum()
    )
    print(
        "PE:",
        (final_df["option_type"] == "PE").sum()
    )

    print(
        "Date range:",
        final_df["timestamp"].min(),
        "→",
        final_df["timestamp"].max()
    )

    print(
        "Trading dates:",
        final_df["timestamp"]
        .dt.date
        .nunique()
    )

    print(
        "Missing NIFTY spot:",
        final_df["nifty_spot"].isna().sum()
    )

    print(
        "Failed contracts:",
        len(failed)
    )

    if failed:

        print("\nFailed contracts:")

        for instrument_key, reason in failed:
            print(
                instrument_key,
                "→",
                reason
            )

    print("=" * 70)


if __name__ == "__main__":
    main()