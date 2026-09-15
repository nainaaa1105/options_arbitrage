import os
import requests
import pandas as pd
from dotenv import load_dotenv

# CONFIGURATION

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

BASE_URL = "https://api.upstox.com/v2"

NIFTY_KEY = "NSE_INDEX|Nifty 50"

EXPIRY_DATE = "2025-09-23"

# Temporary range for testing.
# Later we will replace this with dynamic ATM-based selection.
MIN_STRIKE = 24000
MAX_STRIKE = 26000

TIMEFRAME = "5minute"

OUTPUT_DIR = "data/raw/upstox"


HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

# 1. GET EXPIRIES

def get_expiries():

    url = f"{BASE_URL}/expired-instruments/expiries"

    params = {
        "instrument_key": NIFTY_KEY
    }

    response = requests.get(
        url,
        params=params,
        headers=HEADERS
    )

    response.raise_for_status()

    data = response.json()

    return data["data"]

# 2. GET EXPIRED OPTION CONTRACTS

def get_expired_contracts(expiry_date):

    url = f"{BASE_URL}/expired-instruments/option/contract"

    params = {
        "instrument_key": NIFTY_KEY,
        "expiry_date": expiry_date
    }

    response = requests.get(
        url,
        params=params,
        headers=HEADERS
    )

    response.raise_for_status()

    data = response.json()

    return data["data"]

# 3. FILTER CE / PE OPTIONS

def filter_options(contracts):

    options = []

    for contract in contracts:

        if contract["instrument_type"] in ["CE", "PE"]:
            options.append(contract)

    return options

# 4. SELECT STRIKE RANGE

def select_strikes(
    contracts,
    min_strike,
    max_strike
):

    selected = []

    for contract in contracts:

        strike = contract["strike_price"]

        if min_strike <= strike <= max_strike:

            selected.append(contract)

    return selected


# 5. GET EXPIRED HISTORICAL CANDLES

def get_expired_candles(
    instrument_key,
    date
):

    url = (
        f"{BASE_URL}/expired-instruments/"
        f"historical-candle/"
        f"{instrument_key}/"
        f"{TIMEFRAME}/"
        f"{date}/"
        f"{date}"
    )

    response = requests.get(
        url,
        headers=HEADERS
    )

    response.raise_for_status()

    data = response.json()

    return data["data"]["candles"]

# 6. CONVERT CANDLES TO DATAFRAME

def candles_to_dataframe(
    candles,
    contract
):

    columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "open_interest"
    ]

    df = pd.DataFrame(
        candles,
        columns=columns
    )

    # Add contract information

    df["instrument_key"] = contract["instrument_key"]

    df["trading_symbol"] = contract["trading_symbol"]

    df["expiry"] = contract["expiry"]

    df["strike"] = contract["strike_price"]

    df["option_type"] = contract["instrument_type"]

    df["lot_size"] = contract["lot_size"]

    return df


# 7. DOWNLOAD ONE CONTRACT

def download_contract(
    contract,
    expiry_date
):

    instrument_key = contract["instrument_key"]

    candles = get_expired_candles(
        instrument_key,
        expiry_date
    )

    if not candles:

        return None

    df = candles_to_dataframe(
        candles,
        contract
    )

    return df

# 8. CLEAN DATA

def clean_data(df):

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # Convert expiry to datetime
    df["expiry"] = pd.to_datetime(
        df["expiry"]
    )

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Sort by instrument and timestamp
    df = df.sort_values(
        by=[
            "instrument_key",
            "timestamp"
        ]
    )

    # Reset index
    df = df.reset_index(
        drop=True
    )

    return df


# 9. SAVE DATA

def save_data(
    df,
    expiry_date
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        f"options_{expiry_date}.parquet"
    )

    df.to_parquet(
        output_file,
        index=False
    )

    return output_file



if __name__ == "__main__":

    print("=" * 60)
    print("NIFTY OPTIONS DATA COLLECTOR")
    print("=" * 60)

    # Check token

    if not ACCESS_TOKEN:

        raise ValueError(
            "UPSTOX_ACCESS_TOKEN not found in .env file."
        )

    print("\nAuthentication token found.")

    # Get expiries

    print("\nGetting available expiries...")

    expiries = get_expiries()

    print(
        "Number of available expiries:",
        len(expiries)
    )

    print(
        "Selected expiry:",
        EXPIRY_DATE
    )

    # Validate expiry

    if EXPIRY_DATE not in expiries:

        raise ValueError(
            f"{EXPIRY_DATE} is not available "
            "in the returned expiry list."
        )

    # Get contracts-

    print("\nGetting expired option contracts...")

    contracts = get_expired_contracts(
        EXPIRY_DATE
    )

    print(
        "Total contracts:",
        len(contracts)
    )

    # Filter CE / PE

    options = filter_options(
        contracts
    )

    print(
        "CE/PE contracts:",
        len(options)
    )

    # Select strikes

    selected = select_strikes(
        options,
        MIN_STRIKE,
        MAX_STRIKE
    )

    print(
        f"Selected contracts "
        f"({MIN_STRIKE} - {MAX_STRIKE}):",
        len(selected)
    )

    # Show selected contracts

    print("\nSelected contracts:")

    for contract in selected:

        print(
            contract["strike_price"],
            contract["instrument_type"],
            "|",
            contract["trading_symbol"]
        )

    # Download candles

    print("\n" + "=" * 60)
    print("DOWNLOADING CANDLES")
    print("=" * 60)

    all_data = []

    failed_contracts = []

    total_contracts = len(selected)

    for i, contract in enumerate(selected, start=1):

        print(
            f"[{i}/{total_contracts}] "
            f"{contract['trading_symbol']}"
        )

        try:

            df = download_contract(
                contract,
                EXPIRY_DATE
            )

            if df is None:

                print("   No candle data returned.")

                failed_contracts.append(
                    contract["instrument_key"]
                )

                continue

            print(
                f"   Candles: {len(df)}"
            )

            all_data.append(df)

        except Exception as e:

            print(
                f"   ERROR: {e}"
            )

            failed_contracts.append(
                contract["instrument_key"]
            )


    if not all_data:

        raise RuntimeError(
            "No option data was downloaded."
        )


    print("\nCombining data...")

    final_df = pd.concat(
        all_data,
        ignore_index=True
    )


    print("Cleaning data...")

    final_df = clean_data(
        final_df
    )


    output_file = save_data(
        final_df,
        EXPIRY_DATE
    )


    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        "Total rows:",
        len(final_df)
    )

    print(
        "Total columns:",
        len(final_df.columns)
    )

    print(
        "Unique contracts:",
        final_df["instrument_key"].nunique()
    )

    print(
        "CE rows:",
        (final_df["option_type"] == "CE").sum()
    )

    print(
        "PE rows:",
        (final_df["option_type"] == "PE").sum()
    )

    print(
        "Unique strikes:",
        final_df["strike"].nunique()
    )

    print(
        "Time range:",
        final_df["timestamp"].min(),
        "to",
        final_df["timestamp"].max()
    )

    print(
        "\nSaved file:",
        output_file
    )


    print(
        "\nFailed contracts:",
        len(failed_contracts)
    )

    if failed_contracts:

        print("\nFailed instrument keys:")

        for key in failed_contracts:

            print(key)

    print("\nCollection completed.")