import os
import requests
from dotenv import load_dotenv

load_dotenv()

access_token = os.getenv("UPSTOX_ACCESS_TOKEN")

url = "https://api.upstox.com/v2/expired-instruments/expiries"

params = {
    "instrument_key": "NSE_INDEX|Nifty 50"
}

headers = {
    "Accept": "application/json",
    "Authorization": f"Bearer {access_token}"
}

response = requests.get(
    url,
    params=params,
    headers=headers
)

print("Status code:", response.status_code)

try:
    data = response.json()
    print(data)
except Exception:
    print(response.text)