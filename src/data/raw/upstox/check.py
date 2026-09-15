import pandas as pd


file_path = "options_2025-09-23.parquet"

df = pd.read_parquet(file_path)

print("=" * 60)
print("DATASET CHECK")
print("=" * 60)

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nOption types:")
print(df["option_type"].value_counts())

print("\nStrikes:")
print(
    df["strike"]
    .sort_values()
    .unique()
)

print("\nContracts:")
print(
    df["instrument_key"]
    .nunique()
)

print("\nTimestamp range:")
print(
    df["timestamp"].min(),
    "to",
    df["timestamp"].max()
)