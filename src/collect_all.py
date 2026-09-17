import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import (
    RAW_UPSTOX_DIR,
    HISTORICAL_DAYS_PER_EXPIRY,
)

from upstox_api import (
    get_expiries,
)

from collect_expiry import (
    collect_expiry,
)



# SETTINGS


# START SMALL.
# Once this works, increase to 10, then all.
NUMBER_OF_EXPIRIES = 3



# HELPERS


def get_previous_calendar_dates(
    expiry_date,
    number_of_days,
):
    """
    Generate a calendar window ending on expiry.

    We intentionally request a slightly larger calendar
    window than the number of trading sessions required.
    """

    expiry = datetime.strptime(
        expiry_date,
        "%Y-%m-%d"
    )

    start = expiry - timedelta(
        days=number_of_days * 2
    )

    return (
        start.strftime("%Y-%m-%d"),
        expiry_date,
    )



# COLLECT ALL


def collect_all():

    print("=" * 70)
    print("COLLECT ALL EXPIRIES")
    print("=" * 70)

    expiries = get_expiries()

    if not expiries:

        raise RuntimeError(
            "No expiries returned."
        )

    
    # Sort chronologically
    

    expiries = sorted(
        expiries
    )

    print(
        "Available expiries:",
        len(expiries)
    )

    
    # IMPORTANT:
    #
    # Start with only a few expiries.
    

    selected_expiries = [
    "2025-09-09",
    "2025-09-16",
    "2025-09-23",
    ]

    print(
        "Selected expiries:",
        len(selected_expiries)
    )

    print()

    results = []

    
    # Process
    

    for index, expiry in enumerate(
        selected_expiries,
        start=1,
    ):

        print()

        print(
            f"EXPIRY {index}/"
            f"{len(selected_expiries)}"
        )

        print(
            "Expiry:",
            expiry
        )

        print("=" * 70)

        output_file = (
            RAW_UPSTOX_DIR
            / f"expiry_{expiry}.parquet"
        )

        
        # Resume protection
        

        if output_file.exists():

            print(
                "Already exists — SKIPPING"
            )

            results.append(
                {
                    "expiry": expiry,
                    "status": "already_exists",
                }
            )

            continue

        try:

            from_date, to_date = (
                get_previous_calendar_dates(
                    expiry,
                    HISTORICAL_DAYS_PER_EXPIRY,
                )
            )

            df = collect_expiry(
                expiry,
                from_date,
                to_date,
            )

            results.append(
                {
                    "expiry": expiry,
                    "status": "success",
                    "rows": len(df),
                    "contracts":
                        df[
                            "instrument_key"
                        ].nunique(),
                }
            )

        except Exception as e:

            print()
            print(
                "EXPIRY FAILED:",
                e
            )

            results.append(
                {
                    "expiry": expiry,
                    "status": "failed",
                    "error": str(e),
                }
            )

    
    # Summary
    

    print()
    print("COLLECTION SUMMARY")

    summary = pd.DataFrame(
        results
    )

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print("COLLECTION COMPLETE")



if __name__ == "__main__":

    collect_all()