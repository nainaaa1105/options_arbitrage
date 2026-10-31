from pathlib import Path

from config import (
    UPSTOX_RAW_DIR,
    HISTORICAL_TRADING_SESSIONS,
)

from upstox_api import (
    get_expiries,
    UpstoxAPIError,
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


def select_expiries(all_expiries, run_mode=None):
    mode = RUN_MODE if run_mode is None else run_mode

    all_expiries = sorted(
        all_expiries
    )

    if mode == 3:

        return TEST_EXPIRIES

    if mode == 10:

        return all_expiries[-10:]

    if mode == 30:

        return all_expiries[-30:]

    if mode == "ALL":

        return all_expiries

    raise ValueError(
        "RUN_MODE must be "
        "3, 10, 30 or 'ALL'"
    )



# MAIN


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Resume expiry collection without editing source constants")
    parser.add_argument("--run-mode", choices=["3", "10", "30", "ALL"], default=str(RUN_MODE))
    parser.add_argument("--sessions-per-expiry", type=int, default=HISTORICAL_TRADING_SESSIONS)
    parser.add_argument("--from-expiry")
    parser.add_argument("--to-expiry")
    parser.add_argument("--expiries-file", type=Path, help="Optional JSON list of known expiry dates, including older gaps")
    parser.add_argument("--plan-only", action="store_true", help="Print the requested collection plan without downloading contracts/candles")
    args = parser.parse_args()
    if args.sessions_per_expiry < 1:
        parser.error("sessions-per-expiry must be positive")

    print("=" * 80)
    print("OPTIONS DATA COLLECTION")
    print("=" * 80)

    print(
        f"Historical sessions per expiry: "
        f"{args.sessions_per_expiry}"
    )

    
    # Get all available expired expiries
    

    all_expiries = json.loads(args.expiries_file.read_text()) if args.expiries_file else get_expiries()

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
        select_expiries(all_expiries, "ALL" if args.run_mode == "ALL" else int(args.run_mode))
    )
    selected_expiries = [date for date in selected_expiries
                        if (not args.from_expiry or date >= args.from_expiry)
                        and (not args.to_expiry or date <= args.to_expiry)]
    if args.plan_only:
        print(json.dumps({"expiries": selected_expiries, "sessions_per_expiry": args.sessions_per_expiry,
                          "resume_existing": True}, indent=2))
        return

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
            manifest_path = output_file.with_suffix(".manifest.json")
            existing_status = "already_exists_unverified"
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                existing_status = ("already_exists_verified" if manifest.get("all_requested_contracts_returned")
                                   else "already_exists_with_failed_contracts")

            print(
                f"ALREADY EXISTS: "
                f"{output_file.name}"
            )

            results.append(
                {
                    "expiry":
                        expiry_date,
                    "status":
                        existing_status,
                }
            )

            continue

        
        # Collect
        

        try:

            df = collect_expiry(
                expiry_date,
                args.sessions_per_expiry,
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

        except UpstoxAPIError as e:
            if e.status_code in (401, 403, 429, 200):
                raise
            results.append({"expiry": expiry_date, "status": "failed", "error": str(e)})
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
