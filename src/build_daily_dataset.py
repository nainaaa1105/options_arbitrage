import pandas as pd

from config import (
    RAW_UPSTOX_DIR,
    STRIKES_EACH_SIDE
)

from strike_selector import (
    get_atm_strike,
    select_strikes
)



# CONFIG


INPUT_FILE = (
    RAW_UPSTOX_DIR
    / "historical_2025-09-23_window.parquet"
)

OUTPUT_FILE = (
    RAW_UPSTOX_DIR
    / "historical_2025-09-23_dynamic.parquet"
)



# LOAD DATA


def load_data():

    print("=" * 70)
    print("BUILD DAILY DYNAMIC DATASET")
    print("=" * 70)

    print("\nInput:", INPUT_FILE)

    df = pd.read_parquet(INPUT_FILE)

    print("Rows:", len(df))
    print(
        "Contracts:",
        df["instrument_key"].nunique()
    )

    return df



# GET DAILY ATM


def calculate_daily_atm(df):

    daily_atm = {}

    dates = sorted(
        df["timestamp"]
        .dt.strftime("%Y-%m-%d")
        .unique()
    )

    print("\nDaily ATM:")

    for date in dates:

        day_data = df[
            df["timestamp"].dt.strftime("%Y-%m-%d")
            == date
        ]

        # First available NIFTY spot of the day
        first_row = day_data.sort_values(
            "timestamp"
        ).iloc[0]

        nifty_spot = float(
            first_row["nifty_spot"]
        )

        available_strikes = sorted(
            day_data["strike"]
            .unique()
        )

        atm = get_atm_strike(
            nifty_spot,
            available_strikes
        )

        selected = select_strikes(
            atm,
            available_strikes,
            STRIKES_EACH_SIDE
        )

        daily_atm[date] = {
            "nifty_spot": nifty_spot,
            "atm": atm,
            "selected_strikes": selected
        }

        print(
            f"{date} | "
            f"NIFTY={nifty_spot:.2f} | "
            f"ATM={atm:.0f} | "
            f"Strikes={len(selected)}"
        )

    return daily_atm



# FILTER DAILY DATA


def filter_daily_data(
    df,
    daily_atm
):

    daily_frames = []

    for date, info in daily_atm.items():

        selected_strikes = set(
            info["selected_strikes"]
        )

        mask = (
            (df["timestamp"].dt.strftime("%Y-%m-%d") == date)
            &
            (df["strike"].isin(selected_strikes))
        )

        day_df = df[mask].copy()

        daily_frames.append(day_df)

    if not daily_frames:

        raise RuntimeError(
            "No daily data was produced."
        )

    result = pd.concat(
        daily_frames,
        ignore_index=True
    )

    return result



# VALIDATE


def validate(df):

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    print("\nRows:", len(df))

    print(
        "Trading dates:",
        df["timestamp"].dt.date.nunique()
    )

    print(
        "Contracts:",
        df["instrument_key"].nunique()
    )

    print(
        "Strikes:",
        df["strike"].nunique()
    )

    print(
        "CE rows:",
        (df["option_type"] == "CE").sum()
    )

    print(
        "PE rows:",
        (df["option_type"] == "PE").sum()
    )

    print(
        "Missing NIFTY spot:",
        df["nifty_spot"].isna().sum()
    )

    print(
        "Duplicate contract/timestamp rows:",
        df.duplicated(
            subset=[
                "instrument_key",
                "timestamp"
            ]
        ).sum()
    )



# MAIN


def main():

    df = load_data()

    daily_atm = calculate_daily_atm(df)

    result = filter_daily_data(
        df,
        daily_atm
    )

    result = result.sort_values(
        [
            "timestamp",
            "strike",
            "option_type"
        ]
    ).reset_index(drop=True)

    validate(result)

    result.to_parquet(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("DATASET SAVED")
    print("=" * 70)

    print("Output:", OUTPUT_FILE)


if __name__ == "__main__":
    main()