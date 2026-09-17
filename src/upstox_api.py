import time
import requests

from config import (
    BASE_URL,
    NIFTY_KEY,
    UPSTOX_ACCESS_TOKEN,
    MAX_RETRIES,
    REQUEST_DELAY_SECONDS,
)


HEADERS = {
    "Authorization": f"Bearer {UPSTOX_ACCESS_TOKEN}",
    "Accept": "application/json",
}



# GENERIC REQUEST


def make_request(url, params=None):
    """
    Make a GET request with retry handling.
    """

    if not UPSTOX_ACCESS_TOKEN:
        raise ValueError(
            "UPSTOX_ACCESS_TOKEN not found in .env"
        )

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=30,
            )

            if response.status_code == 200:
                time.sleep(REQUEST_DELAY_SECONDS)
                return response.json()

            last_error = (
                f"HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

            # Authentication / permission errors should
            # not be retried repeatedly.
            if response.status_code in [400, 401, 403]:
                raise RuntimeError(last_error)

            print(
                f"Request failed "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{last_error}"
            )

        except requests.RequestException as e:

            last_error = str(e)

            print(
                f"Request exception "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{e}"
            )

        if attempt < MAX_RETRIES:
            time.sleep(attempt)

    raise RuntimeError(
        f"Request failed after {MAX_RETRIES} attempts: "
        f"{last_error}"
    )



# EXPIRIES


def get_expiries():
    """
    Get available expired NIFTY option expiry dates.
    """

    url = (
        f"{BASE_URL}/expired-instruments/expiries"
    )

    params = {
        "instrument_key": NIFTY_KEY
    }

    data = make_request(url, params)

    return data.get("data", [])



# EXPIRED OPTION CONTRACTS


def get_expired_contracts(expiry_date):
    """
    Get all expired option contracts for a given expiry.
    """

    url = (
        f"{BASE_URL}/expired-instruments/"
        f"option/contract"
    )

    params = {
        "instrument_key": NIFTY_KEY,
        "expiry_date": expiry_date,
    }

    data = make_request(url, params)

    return data.get("data", [])



# NIFTY HISTORICAL CANDLES


def get_nifty_candles(date):
    """
    Get 1-minute NIFTY candles for a single trading day.
    """

    url = (
        f"{BASE_URL}/historical-candle/"
        f"{NIFTY_KEY}/1minute/"
        f"{date}/{date}"
    )

    data = make_request(url)

    return data.get("data", {}).get("candles", [])



# EXPIRED OPTION HISTORICAL CANDLES


def get_expired_candles(
    instrument_key,
    from_date,
    to_date,
):
    """
    Get expired option candles for a date range.

    Upstox returns candles in reverse chronological order.
    """

    url = (
        f"{BASE_URL}/expired-instruments/"
        f"historical-candle/"
        f"{instrument_key}/5minute/"
        f"{to_date}/{from_date}"
    )

    data = make_request(url)

    return data.get("data", {}).get("candles", [])



# SIMPLE TEST


if __name__ == "__main__":

    print("=" * 70)
    print("UPSTOX API TEST")
    print("=" * 70)

    expiries = get_expiries()

    print(
        "Number of expiries:",
        len(expiries)
    )

    if expiries:

        print(
            "First expiry:",
            expiries[0]
        )

        print(
            "Last expiry:",
            expiries[-1]
        )

    print("API TEST PASSED")
    