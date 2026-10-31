"""Match candle closes using their availability times, not bar-start labels."""

import pandas as pd


def align_option_spot_candles(options, spot, option_minutes=5, spot_minutes=1):
    """Use the spot candle that finishes at the option candle's close.

    Upstox timestamps identify candle starts. For a 09:15 five-minute option
    candle, the matching one-minute spot candle starts at 09:19. Missing exact
    observations remain missing; a stale or later quote is never substituted.
    The option timestamp remains its original bar-start identifier.
    """
    if option_minutes < 1 or spot_minutes < 1:
        raise ValueError("Candle durations must be positive")
    if "bar_minutes" in spot and not spot["bar_minutes"].eq(spot_minutes).all():
        raise ValueError("Declared spot candle duration conflicts with source provenance")
    left = options.drop(columns=[
        "nifty_spot", "spot_timestamp", "spot_available_at", "feature_available_at"
    ], errors="ignore").copy()
    right = spot[["timestamp", "close"]].copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"])
    right["timestamp"] = pd.to_datetime(right["timestamp"])
    if left.timestamp.dt.tz != right.timestamp.dt.tz:
        raise ValueError("Option and spot timestamps must use the same timezone")
    left["feature_available_at"] = left.timestamp + pd.Timedelta(minutes=option_minutes)
    right = right.rename(columns={"timestamp": "spot_timestamp", "close": "nifty_spot"})
    right["spot_available_at"] = right.spot_timestamp + pd.Timedelta(minutes=spot_minutes)
    if right.spot_available_at.duplicated().any():
        raise ValueError("Duplicate spot candle availability timestamps")
    result = left.merge(
        right, left_on="feature_available_at", right_on="spot_available_at",
        how="left", validate="many_to_one", sort=False,
    )
    same_session = result.timestamp.dt.date == result.spot_timestamp.dt.date
    result.loc[~same_session, "nifty_spot"] = float("nan")
    return result
