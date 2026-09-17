import pandas as pd
import glob

files = glob.glob(
    "data/raw/upstox/dynamic_options_*.parquet"
)

print("SPOT ALIGNMENT CHECK")

for file in sorted(files):

    print()
    print("FILE:", file)

    df = pd.read_parquet(file)

    print("Rows:", len(df))

    print(
        "Missing NIFTY spot:",
        df["nifty_spot"].isna().sum()
    )

    print(
        "Missing %:",
        round(
            df["nifty_spot"].isna().mean() * 100,
            2
        )
    )

    print(
        "Unique NIFTY spot values:",
        df["nifty_spot"].nunique()
    )

    print(
        "Timestamp range:",
        df["timestamp"].min(),
        "→",
        df["timestamp"].max()
    )

print()
print("CHECK COMPLETE")