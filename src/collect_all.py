from pathlib import Path

from config import (
    UPSTOX_RAW_DIR,
    HISTORICAL_TRADING_SESSIONS,
)

from upstox_api import (
    get_expiries,
)

from collect_expiry import (
    collect_expiry,
)



# RUN SIZE


# Change this to:
#
# 3  -> test
# 10 -> first scale
# 30 -> second scale
# ALL -> final
#
RUN_MODE = 30



# KNOWN TEST EXPIRIES


TEST_EXPIRIES = [
    "2025-09-09",
    "2025-09-16",
    "2025-09-23",
]



# SELECT EXPIRIES


def select_expiries(all_expiries):

    all_expiries = sorted(
        all_expiries
    )

    if RUN_MODE == 3:

        return TEST_EXPIRIES

    if RUN_MODE == 10:

        return all_expiries[-10:]

    if RUN_MODE == 30:

        return all_expiries[-30:]

    if RUN_MODE == "ALL":

        return all_expiries

    raise ValueError(
        "RUN_MODE must be "
        "3, 10, 30 or 'ALL'"
    )



# MAIN


def main():

    print("=" * 80)
    print("OPTIONS DATA COLLECTION")
    print("=" * 80)

    print(
        f"Historical sessions per expiry: "
        f"{HISTORICAL_TRADING_SESSIONS}"
    )

    
    # Get all available expired expiries
    

    all_expiries = get_expiries()

    if not all_expiries:

        raise RuntimeError(
            "No expired expiries returned."
        )

    all_expiries = sorted(
        all_expiries
    )

    print(
        f"Available expired expiries: "
        f"{len(all_expiries)}"
    )

    
    # Select run
    

    selected_expiries = (
        select_expiries(
            all_expiries
        )
    )

    print()
    print(
        f"Selected expiries: "
        f"{len(selected_expiries)}"
    )

    for expiry in selected_expiries:

        print(
            f"  {expiry}"
        )

    
    # Collect
    

    results = []

    for index, expiry_date in enumerate(
        selected_expiries,
        start=1,
    ):

        print()
        print("#" * 80)
        print(
            f"EXPIRY "
            f"{index}/"
            f"{len(selected_expiries)}"
            f": {expiry_date}"
        )
        print("#" * 80)

        output_file = (
            UPSTOX_RAW_DIR
            / f"expiry_{expiry_date}.parquet"
        )

        
        # Resume protection
        

        if output_file.exists():

            print(
                f"ALREADY EXISTS: "
                f"{output_file.name}"
            )

            results.append(
                {
                    "expiry":
                        expiry_date,
                    "status":
                        "already_exists",
                }
            )

            continue

        
        # Collect
        

        try:

            df = collect_expiry(
                expiry_date,
                HISTORICAL_TRADING_SESSIONS,
            )

            results.append(
                {
                    "expiry":
                        expiry_date,
                    "status":
                        "success",
                    "rows":
                        len(df),
                    "contracts":
                        df[
                            "instrument_key"
                        ].nunique(),
                    "trading_dates":
                        df[
                            "trade_date"
                        ].nunique(),
                }
            )

        except Exception as e:

            print()
            print(
                f"FAILED: "
                f"{expiry_date}"
            )

            print(
                f"ERROR: {e}"
            )

            results.append(
                {
                    "expiry":
                        expiry_date,
                    "status":
                        "failed",
                    "error":
                        str(e),
                }
            )

    
    # Summary
    

    print()
    print("=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)

    for result in results:

        print(result)

    print("COLLECTION COMPLETE")


if __name__ == "__main__":
    main()