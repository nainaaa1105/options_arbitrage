import pandas as pd

file = "data/raw/upstox/dynamic_options_2025-09-23.parquet"

df = pd.read_parquet(file)

print("=" * 70)
print("MISSING ROW CHECK")
print("=" * 70)

# Expected: 42 contracts × timestamps
expected = (
    df.groupby("instrument_key")["timestamp"]
    .nunique()
)

print("\nContract timestamp counts:")
print(expected.value_counts().sort_index())

print("\nContracts with fewer than 75 timestamps:")

bad = expected[expected < 75]

print(bad)

if len(bad) > 0:
    print("\nDetails:")

    for instrument_key in bad.index:
        contract = df[df["instrument_key"] == instrument_key]

        print("\nInstrument:", instrument_key)
        print("Symbol:", contract["trading_symbol"].iloc[0])
        print("Rows:", len(contract))
        print(
            "Timestamp range:",
            contract["timestamp"].min(),
            "→",
            contract["timestamp"].max()
        )