import os
import requests
from dotenv import load_dotenv

load_dotenv()

ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN")

url = "https://api.upstox.com/v2/expired-instruments/option/contract"

params = {
    "instrument_key": "NSE_INDEX|Nifty 50",
    "expiry_date": "2025-09-23"
}

headers = {
    "Authorization": f"Bearer {ACCESS_TOKEN}",
    "Accept": "application/json"
}

response = requests.get(
    url,
    params=params,
    headers=headers
)

print("Status code:", response.status_code)

data = response.json()

print(data)