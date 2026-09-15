import os
import requests
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

instrument_key = "NSE_FO|47851|23-09-2025"

# Test date: the expiry day itself
from_date = "2025-09-23"
to_date = "2025-09-23"

url = (
    "https://api.upstox.com/v2/expired-instruments/"
    "historical-candle/"
    f"{instrument_key}/5minute/{to_date}/{from_date}"
)

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

response = requests.get(url, headers=headers)

print("Status code:", response.status_code)

data = response.json()

print(data)