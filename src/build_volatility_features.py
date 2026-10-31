"""Build point-in-time historical volatility features.

Daily volatility for an observation on trading date *d* is computed only from
daily close-to-close returns through the prior completed trading date.  This is
deliberately shifted by one day before it is joined to intraday option rows.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "options_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "options_volatility_features.parquet"
WINDOWS = (5, 10, 20)
MARKET_TZ = "Asia/Kolkata"


def _as_market_date(series: pd.Series) -> pd.Series:
    """Normalize trade dates to Python dates without changing their local day."""
    return pd.to_datetime(series, errors="coerce").dt.date


def make_daily_spot(df: pd.DataFrame) -> pd.DataFrame:
    """Return one observed daily close per trading date, in timestamp order."""
    required = {"timestamp", "trade_date", "nifty_spot"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns for daily spot series: {sorted(missing)}")

    spot = df[["timestamp", "trade_date", "nifty_spot"]].copy()
    spot["timestamp"] = pd.to_datetime(spot["timestamp"], errors="coerce")
    if spot["timestamp"].dt.tz is None:
        spot["timestamp"] = spot["timestamp"].dt.tz_localize(MARKET_TZ)
    else:
        spot["timestamp"] = spot["timestamp"].dt.tz_convert(MARKET_TZ)
    spot["trade_date"] = _as_market_date(spot["trade_date"])
    spot["nifty_spot"] = pd.to_numeric(spot["nifty_spot"], errors="coerce")
    calendar = pd.DataFrame({
        "trade_date": pd.Index(spot["trade_date"].dropna().unique()).sort_values()
    })
    valid = spot.dropna(subset=["trade_date", "timestamp", "nifty_spot"])
    valid = valid.loc[valid["nifty_spot"] > 0]

    # Select the last actual timestamp row, rather than independently taking
    # the last non-null value in each column.
    last_observation = (
        valid.sort_values("timestamp")
        .groupby("trade_date", sort=True, as_index=False)
        .tail(1)
        .sort_values("trade_date")
        .reset_index(drop=True)[["trade_date", "nifty_spot"]]
    )
    # Keep sessions with no usable spot as explicit missing rows so returns
    # cannot bridge across the missing session as if it were one trading day.
    return calendar.merge(last_observation, on="trade_date", how="left")


def calculate_daily_hv(daily_spot: pd.DataFrame, windows=WINDOWS,
                       max_calendar_gap_days=7) -> pd.DataFrame:
    """Calculate annualized HV for each date using completed prior-day returns."""
    if not {"trade_date", "nifty_spot"}.issubset(daily_spot.columns):
        raise ValueError("daily_spot must contain trade_date and nifty_spot")

    daily = daily_spot.copy().sort_values("trade_date").reset_index(drop=True)
    prices = pd.to_numeric(daily["nifty_spot"], errors="coerce")
    prices = prices.where(np.isfinite(prices) & (prices > 0))
    returns = np.log(prices / prices.shift(1))
    calendar_gap = pd.to_datetime(daily["trade_date"]).diff().dt.days
    if max_calendar_gap_days is not None:
        if max_calendar_gap_days < 1:
            raise ValueError("max_calendar_gap_days must be positive")
        # This conservative guard handles missing coverage, not an exchange
        # holiday calendar. A dedicated complete daily spot history is preferred.
        long_gap = calendar_gap > max_calendar_gap_days
        returns = returns.mask(long_gap)
    else:
        long_gap = pd.Series(False, index=daily.index)

    for window in windows:
        if window < 2:
            raise ValueError("HV windows must be at least 2 daily returns")
        column = f"historical_volatility_{window}d"
        # At date d, rolling HV is lagged: no close or return from d is used.
        daily[column] = (
            returns.rolling(window=window, min_periods=window)
            .std(ddof=1)
            .shift(1)
            * np.sqrt(252.0)
        ).mask(long_gap)
    return daily[["trade_date", *(f"historical_volatility_{w}d" for w in windows)]]


def add_historical_volatility_features(
    df: pd.DataFrame, windows=WINDOWS, daily_spot=None, max_calendar_gap_days=7
) -> pd.DataFrame:
    """Attach prior-completed-day HV and IV/HV features to option rows."""
    out = df.copy()
    out["trade_date"] = _as_market_date(out["trade_date"])
    source = make_daily_spot(out) if daily_spot is None else daily_spot
    daily = calculate_daily_hv(source, windows=windows,
                               max_calendar_gap_days=max_calendar_gap_days)
    daily["trade_date"] = _as_market_date(daily["trade_date"])
    out = out.drop(
        columns=[c for c in daily.columns if c != "trade_date" and c in out.columns]
    ).merge(daily, on="trade_date", how="left", validate="many_to_one")

    for window in windows:
        hv = out[f"historical_volatility_{window}d"]
        out[f"iv_minus_hv_{window}d"] = out["implied_volatility"] - hv
        out[f"iv_hv_ratio_{window}d"] = np.where(
            hv > 0, out["implied_volatility"] / hv, np.nan
        )
    return out


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE)
    parser.add_argument("--daily-spot", type=Path,
                        help="Complete independent daily NIFTY Parquet history: timestamp, close")
    args = parser.parse_args()
    print("BUILDING POINT-IN-TIME HISTORICAL VOLATILITY FEATURES")
    print(f"Input:  {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    df = pd.read_parquet(args.input)
    daily = None
    if args.daily_spot:
        daily = pd.read_parquet(args.daily_spot).rename(columns={"close": "nifty_spot"})
        daily["timestamp"] = pd.to_datetime(daily["timestamp"])
        daily["trade_date"] = daily.timestamp.dt.date
        daily = make_daily_spot(daily)
    result = add_historical_volatility_features(df, daily_spot=daily)
    print(f"Rows: {len(result):,}; trading dates: {result['trade_date'].nunique():,}")
    for window in WINDOWS:
        col = f"historical_volatility_{window}d"
        print(f"{col}: {result[col].notna().sum():,} valid rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(args.output, index=False)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
