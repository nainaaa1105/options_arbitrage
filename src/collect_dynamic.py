import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

BASE_URL = "https://api.upstox.com/v2"

NIFTY_KEY = "NSE_INDEX|Nifty 50"

TEST_DATES = [
    "2025-09-23",
    "2025-09-16",
    "2025-09-09",
    "2025-09-02",
    "2025-08-28",
]

TIMEFRAME = "5minute"

NUMBER_OF_STRIKES = 10

OUTPUT_DIR = "data/raw/upstox"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}



# 1. GET NIFTY DATA


def get_nifty_candles(date):

    url = (
    f"{BASE_URL}/historical-candle/"
    f"{NIFTY_KEY}/1minute/"
    f"{date}/"
    f"{date}"
    ) 

    response = requests.get(
        url,
        headers=HEADERS
    )

    print("NIFTY status:", response.status_code)

    response.raise_for_status()

    return response.json()["data"]["candles"]



# 2. CONVERT NIFTY TO DATAFRAME


def nifty_to_dataframe(candles):

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values("timestamp")

    return df



# 3. GET 09:15 REFERENCE SPOT


def get_reference_spot(nifty_df):

    opening_row = nifty_df[
        nifty_df["timestamp"].dt.strftime("%H:%M") == "09:15"
    ]

    if opening_row.empty:

        raise ValueError(
            "09:15 NIFTY candle not found."
        )

    spot = opening_row.iloc[0]["close"]

    return float(spot)



# 4. GET EXPIRED OPTION CONTRACTS


def get_expired_contracts(expiry_date):

    url = (
        f"{BASE_URL}/expired-instruments/"
        f"option/contract"
    )

    params = {
    "instrument_key": NIFTY_KEY,
    "expiry_date": expiry_date
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    print(
        "Expired contracts status:",
        response.status_code
    )

    response.raise_for_status()

    return response.json()["data"]



# 5. FILTER CE / PE


def filter_options(contracts):

    return [
        contract
        for contract in contracts
        if contract.get("instrument_type") in ["CE", "PE"]
    ]



# 6. FIND ATM


def find_atm_strike(contracts, spot):

    strikes = sorted(
        set(
            float(contract["strike_price"])
            for contract in contracts
            if contract.get("strike_price") is not None
        )
    )

    if not strikes:

        raise ValueError(
            "No option strikes found."
        )

    atm = min(
        strikes,
        key=lambda strike: abs(strike - spot)
    )

    return atm, strikes

# 7. SELECT ATM ± 10 STRIKES


def select_atm_strikes(
    strikes,
    atm_strike,
    number_of_strikes=10
):

    atm_index = strikes.index(atm_strike)

    start = max(
        0,
        atm_index - number_of_strikes
    )

    end = min(
        len(strikes),
        atm_index + number_of_strikes + 1
    )

    return strikes[start:end]


# 8. GET OPTION CANDLES

def get_option_candles(
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

    if response.status_code != 200:

        print(
            "Failed:",
            instrument_key,
            response.status_code
        )

        return []

    return response.json()["data"]["candles"]


# 9. OPTION CANDLES → DATAFRAME

def candles_to_dataframe(
    candles,
    contract,
    expiry_date
):

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
            "open_interest"
        ]
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["instrument_key"] = contract[
        "instrument_key"
    ]

    df["trading_symbol"] = contract.get(
        "trading_symbol"
    )

    df["expiry"] = pd.to_datetime(
        expiry_date
    )

    df["strike"] = float(
        contract["strike_price"]
    )

    df["option_type"] = contract[
        "instrument_type"
    ]

    df["lot_size"] = contract.get(
        "lot_size"
    )

    return df


# 10. ADD NIFTY SPOT TO OPTION DATA

def attach_spot(
    option_df,
    nifty_df
):

    spot_df = nifty_df[
        [
            "timestamp",
            "close"
        ]
    ].copy()

    spot_df = spot_df.rename(
        columns={
            "close": "nifty_spot"
        }
    )

    # Exact timestamp alignment
    merged = option_df.merge(
        spot_df,
        on="timestamp",
        how="left"
    )

    return merged


# 11. CLEAN DATA

def clean_data(df):

    df = df.sort_values(
        [
            "timestamp",
            "strike",
            "option_type"
        ]
    )

    df = df.drop_duplicates(
        subset=[
            "timestamp",
            "instrument_key"
        ]
    )

    return df

def collect_one_date(date):
    """
    Collect dynamic ATM option data for one historical date.
    """

    print()
    print("=" * 70)
    print(f"COLLECTING DATE: {date}")
    print("=" * 70)

    output_file = (
        f"{OUTPUT_DIR}/"
        f"dynamic_options_{date}.parquet"
    )

    
    # Resume protection
    

    if os.path.exists(output_file):

        print("Already exists:")
        print(output_file)
        print("Skipping this date.")

        return {
            "date": date,
            "status": "already_exists",
            "rows": None
        }

    
    # 1. NIFTY
    

    nifty_candles = get_nifty_candles(date)

    nifty_df = nifty_to_dataframe(
        nifty_candles
    )

    print(
        "NIFTY candles:",
        len(nifty_df)
    )

    spot = get_reference_spot(
        nifty_df
    )

    print(
        "09:15 NIFTY reference:",
        spot
    )

    
    # 2. OPTIONS
    

    contracts = get_expired_contracts(
        date
    )

    options = filter_options(
        contracts
    )

    print(
        "CE/PE contracts:",
        len(options)
    )

    
    # 3. DYNAMIC ATM
    

    atm, strikes = find_atm_strike(
        options,
        spot
    )

    print(
        "ATM strike:",
        atm
    )

    selected_strikes = select_atm_strikes(
        strikes,
        atm,
        NUMBER_OF_STRIKES
    )

    print(
        "Selected strikes:",
        selected_strikes
    )

    
    # 4. SELECT CONTRACTS
    

    selected_contracts = [
        contract
        for contract in options
        if float(contract["strike_price"])
        in selected_strikes
    ]

    print(
        "Selected contracts:",
        len(selected_contracts)
    )

    
    # 5. DOWNLOAD OPTION DATA
    

    all_data = []
    failed = []

    for i, contract in enumerate(
        selected_contracts,
        start=1
    ):

        print(
            f"[{i}/{len(selected_contracts)}]",
            contract.get("trading_symbol")
        )

        try:

            candles = get_option_candles(
                contract["instrument_key"],
                date
            )

            if not candles:

                failed.append(
                    contract["instrument_key"]
                )

                continue

            df = candles_to_dataframe(
                candles,
                contract,
                date
            )

            all_data.append(df)

        except Exception as e:

            print(
                "ERROR:",
                contract["instrument_key"],
                e
            )

            failed.append(
                contract["instrument_key"]
            )

    
    # 6. CHECK DATA
    

    if not all_data:

        print(
            "NO OPTION DATA FOR:",
            date
        )

        return {
            "date": date,
            "status": "failed",
            "rows": 0
        }

    
    # 7. COMBINE
    

    options_df = pd.concat(
        all_data,
        ignore_index=True
    )

    print(
        "Raw option rows:",
        len(options_df)
    )

    
    # 8. ATTACH NIFTY
    

    options_df = attach_spot(
        options_df,
        nifty_df
    )

    
    # 9. CLEAN
    

    options_df = clean_data(
        options_df
    )

    # 10. SAVE

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    options_df.to_parquet(
        output_file,
        index=False
    )

    print()
    print("Saved:")
    print(output_file)

    print(
        "Final rows:",
        len(options_df)
    )

    print(
        "Failed contracts:",
        len(failed)
    )

    return {
        "date": date,
        "status": "success",
        "rows": len(options_df),
        "contracts": len(selected_contracts),
        "failed": len(failed),
        "atm": atm
    }

if __name__ == "__main__":

    print("=" * 70)
    print("MULTI-DATE DYNAMIC NIFTY OPTIONS COLLECTOR")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    results = []

    for date in TEST_DATES:

        try:

            result = collect_one_date(
                date
            )

            results.append(
                result
            )

        except Exception as e:

            print()
            print(
                f"FAILED DATE: {date}"
            )

            print(
                "ERROR:",
                e
            )

            results.append(
                {
                    "date": date,
                    "status": "error",
                    "rows": 0
                }
            )

    
    # SUMMARY
    

    print()
    print("=" * 70)
    print("COLLECTION SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(
        results
    )

    print(
        results_df.to_string(
            index=False
        )
    )

    print()
    print("Dates processed:", len(results))