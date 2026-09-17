import os
import requests
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

BASE_URL = "https://api.upstox.com/v2"

HEADERS = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

instrument_key = "NSE_FO|47798|23-09-2025"

from_date = "2025-09-18"
to_date = "2025-09-23"

url = (
    f"{BASE_URL}/expired-instruments/"
    f"historical-candle/"
    f"{instrument_key}/5minute/"
    f"{to_date}/{from_date}"
)

response = requests.get(url, headers=HEADERS)

print("=" * 70)
print("MULTI-DAY EXPIRED CANDLE TEST")
print("=" * 70)

print("Status:", response.status_code)
print("URL:", url)

if response.status_code == 200:
    data = response.json()["data"]["candles"]

    print("Total candles:", len(data))

    if data:
        print("First candle:", data[0])
        print("Last candle:", data[-1])

        dates = sorted(set(c[0][:10] for c in data))

        print("\nDates returned:")
        for date in dates:
            print(date)

else:
    print("Response:")
    print(response.text)