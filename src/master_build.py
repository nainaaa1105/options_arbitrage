from pathlib import Path
import pandas as pd

from config import UPSTOX_RAW_DIR

OUTPUT_DIR = UPSTOX_RAW_DIR.parent.parent / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "master_options.parquet"


def main():

    files = sorted(UPSTOX_RAW_DIR.glob("expiry_*.parquet"))

    print("BUILDING MASTER DATASET")


    print(f"Files found: {len(files)}")

    if not files:
        raise RuntimeError("No expiry parquet files found.")

    frames = []

    for file in files:

        print(f"Reading: {file.name}")

        df = pd.read_parquet(file)

        df["source_file"] = file.name

        frames.append(df)

    master = pd.concat(
        frames,
        ignore_index=True,
    )

    # Basic cleanup

    master["timestamp"] = pd.to_datetime(
        master["timestamp"]
    )

    master["expiry"] = pd.to_datetime(
        master["expiry"]
    )

    master["strike"] = master["strike"].astype(float)

    master["nifty_spot"] = master["nifty_spot"].astype(float)

    master["option_type"] = master["option_type"].astype(str)

    
    # Remove exact duplicate contract/timestamp rows
    

    before = len(master)

    master = master.drop_duplicates(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    ).copy()

    duplicates_removed = before - len(master)

    
    # Sort
    

    master = master.sort_values(
        [
            "timestamp",
            "expiry",
            "strike",
            "option_type",
        ]
    ).reset_index(drop=True)

    # Validation
    print()
    print("MASTER DATASET")

    print(f"Rows: {len(master):,}")
    print(f"Files: {len(files)}")
    print(
        f"Contracts: "
        f"{master['instrument_key'].nunique():,}"
    )
    print(
        f"Expiries: "
        f"{master['expiry'].nunique():,}"
    )
    print(
        f"Trading dates: "
        f"{master['trade_date'].nunique():,}"
    )
    print(
        f"Strikes: "
        f"{master['strike'].nunique():,}"
    )

    print(
        f"CE rows: "
        f"{(master['option_type'] == 'CE').sum():,}"
    )

    print(
        f"PE rows: "
        f"{(master['option_type'] == 'PE').sum():,}"
    )

    print(
        f"Missing NIFTY: "
        f"{master['nifty_spot'].isna().sum():,}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed:,}"
    )

    print(
        f"Timestamp range: "
        f"{master['timestamp'].min()} "
        f"→ "
        f"{master['timestamp'].max()}"
    )
    
    # Hard validation
    if master["nifty_spot"].isna().any():
        raise RuntimeError(
            "Master dataset contains missing NIFTY spot values."
        )

    duplicates = master.duplicated(
        subset=[
            "instrument_key",
            "timestamp",
        ]
    ).sum()

    if duplicates > 0:
        raise RuntimeError(
            "Duplicate contract/timestamp rows remain."
        )

    
    # Save
    

    master.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print()
    print("MASTER DATASET BUILD PASSED")


if __name__ == "__main__":
    main()