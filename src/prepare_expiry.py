import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


# Allow imports from src
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    TEST_EXPIRY,
    TEST_FROM_DATE,
    TEST_TO_DATE,
    STRIKES_EACH_SIDE,
)

from upstox_api import (
    get_expired_contracts,
    get_nifty_candles,
)

from strike_selector import (
    get_atm_strike,
    select_strikes,
)



# HELPERS


def get_nifty_dataframe(date):
    """
    Convert NIFTY candles into a DataFrame.
    """

    candles = get_nifty_candles(date)

    if not candles:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "nifty_spot",
            ]
        )

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "open_interest",
        ],
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df["nifty_spot"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    return df[
        ["timestamp", "nifty_spot"]
    ]


def get_trading_dates(
    from_date,
    to_date,
):
    """
    Generate calendar dates.

    We later keep only dates for which
    NIFTY actually returns candles.
    """

    start = datetime.strptime(
        from_date,
        "%Y-%m-%d"
    )

    end = datetime.strptime(
        to_date,
        "%Y-%m-%d"
    )

    dates = []

    current = start

    while current <= end:

        dates.append(
            current.strftime("%Y-%m-%d")
        )

        current += timedelta(days=1)

    return dates



# DAILY STRIKE PREPARATION


def prepare_expiry(
    expiry_date,
    from_date,
    to_date,
):
    print("=" * 70)
    print("PREPARE EXPIRY")
    print("=" * 70)

    print("Expiry:", expiry_date)
    print("From:", from_date)
    print("To:", to_date)

    
    # Get contracts
    

    print()
    print("Downloading expired contracts...")

    contracts = get_expired_contracts(
        expiry_date
    )

    print(
        "Total contracts:",
        len(contracts)
    )

    if not contracts:
        raise RuntimeError(
            "No contracts returned."
        )

    # Only CE / PE
    option_contracts = [
        c for c in contracts
        if c.get("instrument_type")
        in ["CE", "PE"]
    ]

    available_strikes = sorted(
        set(
            float(c["strike_price"])
            for c in option_contracts
        )
    )

    print(
        "Available strikes:",
        len(available_strikes)
    )

    print(
        "Strike range:",
        min(available_strikes),
        "to",
        max(available_strikes)
    )

    
    # Calculate daily ATM
    

    print()
    print("Calculating daily ATM...")

    trading_dates = get_trading_dates(
        from_date,
        to_date,
    )

    daily_info = []

    all_required_strikes = set()

    for date in trading_dates:

        nifty_df = get_nifty_dataframe(date)

        if nifty_df.empty:
            continue

        # First available NIFTY observation.
        #
        # We use the 09:15 candle close.
        reference_spot = float(
            nifty_df.iloc[0]["nifty_spot"]
        )

        atm = get_atm_strike(
            reference_spot,
            available_strikes,
        )

        selected = select_strikes(
            atm,
            available_strikes,
            STRIKES_EACH_SIDE,
        )

        all_required_strikes.update(
            selected
        )

        daily_info.append(
            {
                "date": date,
                "nifty_spot": reference_spot,
                "atm_strike": atm,
                "selected_strikes": selected,
            }
        )

        print(
            f"{date} | "
            f"NIFTY={reference_spot:.2f} | "
            f"ATM={atm:.0f} | "
            f"Strikes={len(selected)}"
        )

    if not daily_info:
        raise RuntimeError(
            "No trading days found."
        )

    
    # Summary
    

    union_strikes = sorted(
        all_required_strikes
    )

    print()
    print("=" * 70)
    print("PREPARATION SUMMARY")
    print("=" * 70)

    print(
        "Trading days:",
        len(daily_info)
    )

    print(
        "Daily strike target:",
        (STRIKES_EACH_SIDE * 2) + 1
    )

    print(
        "Unique strikes in UNION:",
        len(union_strikes)
    )

    print(
        "Union strike range:",
        min(union_strikes),
        "to",
        max(union_strikes),
    )

    
    # Find required contracts
    

    required_contracts = [
        c for c in option_contracts
        if float(c["strike_price"])
        in set(union_strikes)
    ]

    print(
        "Required contracts:",
        len(required_contracts)
    )

    ce_count = sum(
        1
        for c in required_contracts
        if c.get("instrument_type") == "CE"
    )

    pe_count = sum(
        1
        for c in required_contracts
        if c.get("instrument_type") == "PE"
    )

    print(
        "CE:",
        ce_count,
        "| PE:",
        pe_count,
    )

    return {
        "expiry": expiry_date,
        "from_date": from_date,
        "to_date": to_date,
        "daily_info": daily_info,
        "union_strikes": union_strikes,
        "required_contracts": required_contracts,
    }



# TEST


if __name__ == "__main__":

    result = prepare_expiry(
        TEST_EXPIRY,
        TEST_FROM_DATE,
        TEST_TO_DATE,
    )

    print()
    print("PREPARATION COMPLETE")
