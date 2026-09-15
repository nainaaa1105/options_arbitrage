import pandas as pd

FILE = "data/raw/upstox/dynamic_options_2025-09-23.parquet"

df = pd.read_parquet(FILE)

print("=" * 60)
print("DYNAMIC DATASET CHECK")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique contracts:")
print(df["instrument_key"].nunique())

print("\nOption types:")
print(df["option_type"].value_counts())

print("\nUnique strikes:")
print(df["strike"].nunique())

print("\nStrike range:")
print(
    df["strike"].min(),
    "to",
    df["strike"].max()
)

print("\nNIFTY spot:")
print(
    df["nifty_spot"].describe()
)

print("\nFirst rows:")
print(
    df[
        [
            "timestamp",
            "nifty_spot",
            "strike",
            "option_type",
            "close"
        ]
    ].head(10)
)

print("\nLast rows:")
print(
    df[
        [
            "timestamp",
            "nifty_spot",
            "strike",
            "option_type",
            "close"
        ]
    ].tail(10)
)