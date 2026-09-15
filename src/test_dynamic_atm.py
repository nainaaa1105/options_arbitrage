import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

BASE_URL = "https://api.upstox.com/v2"

NIFTY_KEY = "NSE_INDEX|Nifty 50"
EXPIRY_DATE = "2025-09-23"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

# 1. Get NIFTY candles

def get_nifty_candles():

    date = EXPIRY_DATE

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

    if response.status_code != 200:
        print(response.text)

    response.raise_for_status()

    return response.json()["data"]["candles"]

# 2. Get 09:15 NIFTY reference price

def get_opening_spot(candles):

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

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    opening_row = df[
        df["timestamp"].dt.strftime("%H:%M") == "09:15"
    ]

    if opening_row.empty:
        raise ValueError("09:15 NIFTY candle not found.")

    # Use 09:15 candle CLOSE
    spot = opening_row.iloc[0]["close"]

    return float(spot)

# 3. Get expired option contracts

def get_expired_contracts():

    url = (
        f"{BASE_URL}/expired-instruments/"
        f"option/contract"
    )

    params = {
        "instrument_key": NIFTY_KEY,
        "expiry_date": EXPIRY_DATE
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params
    )

    print("Contracts status:", response.status_code)

    if response.status_code != 200:
        print(response.text)

    response.raise_for_status()

    return response.json()["data"]


# 4. Filter CE and PE

def filter_options(contracts):

    options = []

    for contract in contracts:

        option_type = contract.get("instrument_type")

        if option_type in ["CE", "PE"]:
            options.append(contract)

    return options

# 5. Find dynamic ATM

def find_atm_strike(contracts, spot):

    strikes = sorted(
        set(
            float(contract["strike_price"])
            for contract in contracts
            if contract.get("strike_price") is not None
        )
    )

    if not strikes:
        raise ValueError("No strikes found.")

    atm_strike = min(
        strikes,
        key=lambda strike: abs(strike - spot)
    )

    return atm_strike, strikes


# 6. Select ATM ± N strikes

def select_atm_range(strikes, atm_strike, number_of_strikes=10):

    atm_index = strikes.index(atm_strike)

    start = max(
        0,
        atm_index - number_of_strikes
    )

    end = min(
        len(strikes),
        atm_index + number_of_strikes + 1
    )

    selected_strikes = strikes[start:end]

    return selected_strikes


# MAIN

if __name__ == "__main__":

    print("=" * 60)
    print("DYNAMIC ATM TEST")
    print("=" * 60)

    # Step 1
    candles = get_nifty_candles()

    print("NIFTY candles:", len(candles))

    # Step 2
    spot = get_opening_spot(candles)

    print()
    print("NIFTY 09:15 reference price:", spot)

    # Step 3
    contracts = get_expired_contracts()

    print("Total expired contracts:", len(contracts))

    # Step 4
    options = filter_options(contracts)

    print("Total CE/PE contracts:", len(options))

    # Step 5
    atm_strike, strikes = find_atm_strike(
        options,
        spot
    )

    print()
    print("ATM strike:", atm_strike)

    # Step 6
    selected_strikes = select_atm_range(
        strikes,
        atm_strike,
        number_of_strikes=10
    )

    print()
    print("Selected strikes:")
    print(selected_strikes)

    print()
    print("Number of selected strikes:",
          len(selected_strikes))

    # Step 7
    selected_contracts = [
        contract
        for contract in options
        if float(contract["strike_price"])
        in selected_strikes
    ]

    ce_count = sum(
        1
        for contract in selected_contracts
        if contract.get("instrument_type") == "CE"
    )

    pe_count = sum(
        1
        for contract in selected_contracts
        if contract.get("instrument_type") == "PE"
    )

    print()
    print("Selected contracts:", len(selected_contracts))
    print("CE contracts:", ce_count)
    print("PE contracts:", pe_count)

    print()
    print("Sample contracts:")

    for contract in selected_contracts[:10]:

        print(
            contract.get("instrument_key"),
            "|",
            contract.get("trading_symbol"),
            "| Strike:",
            contract.get("strike_price"),
            "| Type:",
            contract.get("instrument_type")
        )

    print()
