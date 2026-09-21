import numpy as np
import pandas as pd

from config import (
    RISK_FREE_RATE,
    FEATURE_DATASET_FILE,
    MAX_REASONABLE_IV,
)

from time_to_expiry import add_time_to_expiry

from implied_volatility import (
    implied_volatility,
)

from black_scholes import (
    black_scholes_price,
    intrinsic_value,
)

from greeks import (
    calculate_greeks,
)


def process_row(row):
    """
    Calculate IV, theoretical price and Greeks
    for one option observation.
    """

    market_price = float(row["close"])
    spot = float(row["nifty_spot"])
    strike = float(row["strike"])
    T = float(row["time_to_expiry"])
    option_type = row["option_type"]

    # --------------------------------------------------------
    # Implied volatility
    # --------------------------------------------------------

    iv, iv_status = implied_volatility(
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=T,
        risk_free_rate=RISK_FREE_RATE,
        option_type=option_type,
    )

    # --------------------------------------------------------
    # If IV failed, all IV-dependent values are NaN.
    # --------------------------------------------------------

    if iv_status != "valid":

        return pd.Series(
            {
                "implied_volatility": np.nan,
                "iv_status": iv_status,
                "theoretical_price": np.nan,
                "price_deviation": np.nan,
                "price_deviation_pct": np.nan,
                "delta": np.nan,
                "gamma": np.nan,
                "vega": np.nan,
                "theta": np.nan,
            }
        )

    # --------------------------------------------------------
    # Theoretical price
    # --------------------------------------------------------

    theoretical_price = black_scholes_price(
        spot=spot,
        strike=strike,
        time_to_expiry=T,
        volatility=iv,
        risk_free_rate=RISK_FREE_RATE,
        option_type=option_type,
    )

    # --------------------------------------------------------
    # Greeks
    # --------------------------------------------------------

    delta, gamma, vega, theta = calculate_greeks(
        spot=spot,
        strike=strike,
        time_to_expiry=T,
        volatility=iv,
        risk_free_rate=RISK_FREE_RATE,
        option_type=option_type,
    )

    # --------------------------------------------------------
    # Deviation
    # --------------------------------------------------------

    price_deviation = (
        market_price - theoretical_price
    )

    if theoretical_price > 1e-8:

        price_deviation_pct = (
            price_deviation
            / theoretical_price
        )

    else:

        price_deviation_pct = np.nan

    return pd.Series(
        {
            "implied_volatility": iv,
            "iv_status": iv_status,
            "theoretical_price": theoretical_price,
            "price_deviation": price_deviation,
            "price_deviation_pct": price_deviation_pct,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
        }
    )


def main():

    input_file = (
        FEATURE_DATASET_FILE.parent
        / "master_options.parquet"
    )

    print("=" * 80)
    print("BUILDING QUANTITATIVE FEATURE DATASET")
    print("=" * 80)

    print(f"Input:  {input_file}")
    print(f"Output: {FEATURE_DATASET_FILE}")

    # --------------------------------------------------------
    # Load master dataset
    # --------------------------------------------------------

    df = pd.read_parquet(input_file)

    print()
    print(f"Input rows: {len(df):,}")

    # --------------------------------------------------------
    # Market price
    # --------------------------------------------------------

    df["market_price"] = df["close"].astype(float)

    # --------------------------------------------------------
    # Time to expiry
    # --------------------------------------------------------

    print("Calculating time to expiry...")

    df = add_time_to_expiry(df)

    # --------------------------------------------------------
    # Intrinsic value
    # --------------------------------------------------------

    print("Calculating intrinsic value...")

    df["intrinsic_value"] = df.apply(
        lambda row: intrinsic_value(
            spot=row["nifty_spot"],
            strike=row["strike"],
            option_type=row["option_type"],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Moneyness
    # --------------------------------------------------------

    df["moneyness"] = (
        df["strike"]
        / df["nifty_spot"]
    )

    df["log_moneyness"] = np.log(
        df["strike"]
        / df["nifty_spot"]
    )

    # --------------------------------------------------------
    # Liquidity flags
    # --------------------------------------------------------

    df["zero_volume"] = (
        df["volume"] <= 0
    )

    df["zero_open_interest"] = (
        df["open_interest"] <= 0
    )

    df["zero_market_price"] = (
        df["market_price"] <= 0
    )

    # --------------------------------------------------------
    # Calculate IV + BS + Greeks
    # --------------------------------------------------------

    print()
    print("Calculating IV, theoretical price and Greeks...")
    print("This may take some time for 519k rows.")

    features = df.apply(
        process_row,
        axis=1,
    )

    df = pd.concat(
        [df, features],
        axis=1,
    )

    # --------------------------------------------------------
    # Sanity flag for IV
    # --------------------------------------------------------

    df["iv_reasonable"] = (
        df["implied_volatility"].notna()
        & (
            df["implied_volatility"]
            <= MAX_REASONABLE_IV
        )
    )

    # --------------------------------------------------------
    # Final ordering
    # --------------------------------------------------------

    preferred_columns = [
        "timestamp",
        "trade_date",
        "expiry",
        "expiry_timestamp",

        "instrument_key",
        "trading_symbol",

        "strike",
        "option_type",
        "lot_size",

        "nifty_spot",
        "market_price",

        "open",
        "high",
        "low",
        "close",

        "volume",
        "open_interest",

        "time_to_expiry",
        "risk_free_rate",

        "moneyness",
        "log_moneyness",

        "intrinsic_value",

        "implied_volatility",
        "iv_status",
        "iv_reasonable",

        "theoretical_price",

        "price_deviation",
        "price_deviation_pct",

        "delta",
        "gamma",
        "vega",
        "theta",

        "zero_volume",
        "zero_open_interest",
        "zero_market_price",

        "source_file",
    ]

    # Add risk-free rate column
    df["risk_free_rate"] = RISK_FREE_RATE

    existing_columns = [
        col
        for col in preferred_columns
        if col in df.columns
    ]

    remaining_columns = [
        col
        for col in df.columns
        if col not in existing_columns
    ]

    df = df[
        existing_columns
        + remaining_columns
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    df.to_parquet(
        FEATURE_DATASET_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FEATURE DATASET SUMMARY")
    print("=" * 80)

    print(f"Rows: {len(df):,}")

    print(
        f"Valid IV: "
        f"{(df['iv_status'] == 'valid').sum():,}"
    )

    print(
        f"IV missing: "
        f"{df['implied_volatility'].isna().sum():,}"
    )

    print()
    print("IV STATUS")
    print("-" * 80)

    print(
        df["iv_status"]
        .value_counts(dropna=False)
    )

    print()
    print("IV SUMMARY")
    print("-" * 80)

    print(
        df["implied_volatility"]
        .describe()
    )

    print()
    print("THEORETICAL PRICE SUMMARY")
    print("-" * 80)

    print(
        df["theoretical_price"]
        .describe()
    )

    print()
    print("GREEK MISSING VALUES")
    print("-" * 80)

    for column in [
        "delta",
        "gamma",
        "vega",
        "theta",
    ]:

        print(
            f"{column}: "
            f"{df[column].isna().sum():,}"
        )

    print()
    print(f"Saved: {FEATURE_DATASET_FILE}")

    print()
    print("FEATURE DATASET BUILD COMPLETE")


if __name__ == "__main__":
    main()