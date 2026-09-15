import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")
BASE_URL = "https://api.upstox.com/v2"
INTERVAL = "1minute"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}


def get_nifty_candles():

    if not ACCESS_TOKEN:
        raise RuntimeError(
            "UPSTOX_ACCESS_TOKEN is missing. Add it to the .env file or environment."
        )

    instrument_key = "NSE_INDEX|Nifty 50"
    from_date = "2025-09-23"
    to_date = "2025-09-23"

    url = (
        f"{BASE_URL}/historical-candle/"
        f"{instrument_key}/{INTERVAL}/"
        f"{to_date}/"
        f"{from_date}"
    )

    response = requests.get(url, headers=HEADERS, timeout=30)

    print("Status code:", response.status_code)

    if not response.ok:
        print("Response:", response.text)

    response.raise_for_status()

    data = response.json()

    return data["data"]["candles"]


if __name__ == "__main__":

    candles = get_nifty_candles()

    print("Number of candles:", len(candles))

    # Convert candles into DataFrame
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

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nLast 5 rows:")
    print(df.tail())

    print("\nNumber of candles:", len(df))

    print("\nFirst timestamp:", df["timestamp"].min())
    print("Last timestamp:", df["timestamp"].max())

    print("\nUnique dates:")
    print(df["timestamp"].dt.date.unique())

    print("\n09:15 candle:")
    print(
        df[df["timestamp"].dt.strftime("%H:%M") == "09:15"]
    )

    print("\n15:29 candle:")
    print(
        df[df["timestamp"].dt.strftime("%H:%M") == "15:29"]
    )