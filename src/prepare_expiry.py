from datetime import datetime, timedelta

import pandas as pd

from config import (
    STRIKES_EACH_SIDE,
)

from strike_selector import (
    get_atm_strike,
    select_strikes,
    filter_contracts_by_strikes,
)

from upstox_api import (
    get_expired_contracts,
    get_nifty_candles,
)



# NIFTY DATA


def get_nifty_dataframe(date):
    """
    Download NIFTY 1-minute candles for one date.
    """

    data = get_nifty_candles(date)

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(
        data,
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

    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    # API returns reverse chronological order.
    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df



# FIND EXACT TRADING SESSIONS


def get_previous_trading_sessions(
    expiry_date,
    number_of_sessions,
):
    """
    Find exactly N trading sessions ending on expiry date.

    We move backwards one calendar day at a time.
    A date is considered a trading session only if
    NIFTY historical candles are returned.
    """

    expiry = datetime.strptime(
        expiry_date,
        "%Y-%m-%d",
    ).date()

    sessions = []

    current_date = expiry

    # Safety limit
    max_days_to_check = (
        number_of_sessions * 5
    ) + 10

    days_checked = 0

    while len(sessions) < number_of_sessions:

        date_str = current_date.isoformat()

        print(
            f"Checking NIFTY: {date_str}"
        )

        nifty_df = get_nifty_dataframe(
            date_str
        )

        if not nifty_df.empty:

            sessions.append(
                {
                    "date": date_str,
                    "nifty_df": nifty_df,
                }
            )

            print(
                f"  Trading session found: "
                f"{date_str}"
            )

        current_date -= timedelta(days=1)

        days_checked += 1

        if days_checked > max_days_to_check:

            raise RuntimeError(
                f"Could not find exactly "
                f"{number_of_sessions} trading "
                f"sessions ending at "
                f"{expiry_date}"
            )

    # We searched backwards.
    # Reverse into chronological order.
    sessions.reverse()

    return sessions



# PREPARE ONE EXPIRY


def prepare_expiry(
    expiry_date,
    number_of_sessions=5,
):
    """
    Prepare all data required for one expiry.

    Steps:

    1. Download all expired contracts.
    2. Find exactly N trading sessions.
    3. Calculate daily ATM from 09:15 NIFTY close.
    4. Select ±10 strikes around daily ATM.
    5. Take union of all required strikes.
    6. Select CE + PE contracts for union.
    """

    print()
    print("=" * 80)
    print(
        f"PREPARING EXPIRY: {expiry_date}"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # 1. Get contracts
    # --------------------------------------------------------

    contracts = get_expired_contracts(
        expiry_date
    )

    if not contracts:

        raise RuntimeError(
            f"No expired contracts found "
            f"for {expiry_date}"
        )

    print(
        f"Total available contracts: "
        f"{len(contracts)}"
    )

    available_strikes = sorted(
        set(
            float(c["strike_price"])
            for c in contracts
            if c.get("instrument_type")
            in ["CE", "PE"]
        )
    )

    print(
        f"Available strikes: "
        f"{len(available_strikes)}"
    )

    # --------------------------------------------------------
    # 2. Find exactly 5 sessions
    # --------------------------------------------------------

    sessions = get_previous_trading_sessions(
        expiry_date,
        number_of_sessions,
    )

    trading_dates = [
        session["date"]
        for session in sessions
    ]

    print()
    print(
        "EXACT TRADING SESSIONS:"
    )

    for date in trading_dates:
        print(
            f"  {date}"
        )

    if len(trading_dates) != number_of_sessions:

        raise RuntimeError(
            "Trading-session count mismatch."
        )

    # --------------------------------------------------------
    # 3. Daily ATM selection
    # --------------------------------------------------------

    daily_selection = []

    union_strikes = set()

    for session in sessions:

        date = session["date"]

        nifty_df = session["nifty_df"]

        if nifty_df.empty:
            continue

        # First candle = 09:15
        reference_spot = float(
            nifty_df.iloc[0]["close"]
        )

        atm_strike = get_atm_strike(
            reference_spot,
            available_strikes,
        )

        selected_strikes = select_strikes(
            atm_strike,
            available_strikes,
            STRIKES_EACH_SIDE,
        )

        union_strikes.update(
            selected_strikes
        )

        daily_selection.append(
            {
                "date": date,
                "nifty_spot": reference_spot,
                "atm_strike": atm_strike,
                "selected_strikes":
                    selected_strikes,
            }
        )

        print(
            f"{date} | "
            f"NIFTY={reference_spot:.2f} | "
            f"ATM={atm_strike:.0f} | "
            f"Strikes={len(selected_strikes)} | "
            f"Range="
            f"{min(selected_strikes):.0f}"
            f" → "
            f"{max(selected_strikes):.0f}"
        )

    # --------------------------------------------------------
    # 4. Union strikes
    # --------------------------------------------------------

    union_strikes = sorted(
        union_strikes
    )

    print()
    print(
        f"UNION STRIKES: "
        f"{len(union_strikes)}"
    )

    print(
        f"UNION RANGE: "
        f"{min(union_strikes):.0f}"
        f" → "
        f"{max(union_strikes):.0f}"
    )

    # --------------------------------------------------------
    # 5. Required contracts
    # --------------------------------------------------------

    required_contracts = (
        filter_contracts_by_strikes(
            contracts,
            union_strikes,
        )
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

    print()
    print(
        f"REQUIRED CONTRACTS: "
        f"{len(required_contracts)}"
    )

    print(
        f"CE: {ce_count}"
    )

    print(
        f"PE: {pe_count}"
    )

    print("=" * 80)

    return {
        "expiry_date": expiry_date,

        "trading_dates":
            trading_dates,

        "daily_selection":
            daily_selection,

        "union_strikes":
            union_strikes,

        "contracts":
            required_contracts,

        "nifty_data": {
            session["date"]:
                session["nifty_df"]
            for session in sessions
        },
    }



# TEST


if __name__ == "__main__":

    result = prepare_expiry(
        "2025-09-23",
        number_of_sessions=5,
    )

    print()
    print("=" * 80)
    print("PREPARATION COMPLETE")
    print("=" * 80)

    print(
        "Number of sessions:",
        len(result["trading_dates"]),
    )

    print(
        "Trading dates:",
        result["trading_dates"],
    )

    print(
        "Union strikes:",
        len(result["union_strikes"]),
    )

    print(
        "Required contracts:",
        len(result["contracts"]),
    )