import pandas as pd

from config import EXPIRY_HOUR, EXPIRY_MINUTE


def add_time_to_expiry(df):
    """
    Add expiry timestamp and time_to_expiry in years.

    Both market timestamp and expiry timestamp are kept
    timezone-aware in IST.

    Expiry is treated as 15:30 IST on the expiry date.
    """

    df = df.copy()

    # --------------------------------------------------------
    # Ensure timestamp is timezone-aware
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    # Master dataset timestamps should already be IST-aware.
    # If they are somehow naive, localize them to IST.
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(
            "Asia/Kolkata"
        )
    else:
        # Normalize everything to IST.
        df["timestamp"] = df["timestamp"].dt.tz_convert(
            "Asia/Kolkata"
        )

    # --------------------------------------------------------
    # Ensure expiry is timezone-aware
    # --------------------------------------------------------

    df["expiry"] = pd.to_datetime(
        df["expiry"],
        errors="coerce",
    )

    # The expiry column represents a date, so normalize first.
    expiry_dates = df["expiry"].dt.normalize()

    # If expiry is timezone-naive, localize to IST.
    if expiry_dates.dt.tz is None:
        expiry_dates = expiry_dates.dt.tz_localize(
            "Asia/Kolkata"
        )
    else:
        expiry_dates = expiry_dates.dt.tz_convert(
            "Asia/Kolkata"
        )

    # --------------------------------------------------------
    # Build expiry timestamp
    # --------------------------------------------------------

    df["expiry_timestamp"] = (
        expiry_dates
        + pd.Timedelta(
            hours=EXPIRY_HOUR,
            minutes=EXPIRY_MINUTE,
        )
    )

    # --------------------------------------------------------
    # Calculate remaining time
    # --------------------------------------------------------

    remaining_seconds = (
        df["expiry_timestamp"]
        - df["timestamp"]
    ).dt.total_seconds()

    # --------------------------------------------------------
    # Negative values should not occur in the valid dataset.
    # Keep them at zero rather than creating negative T.
    # --------------------------------------------------------

    remaining_seconds = remaining_seconds.clip(
        lower=0
    )

    # --------------------------------------------------------
    # Convert seconds to years
    #
    # ACT/365 convention for the first version.
    # --------------------------------------------------------

    df["time_to_expiry"] = (
        remaining_seconds
        / (365.0 * 24.0 * 60.0 * 60.0)
    )

    return df