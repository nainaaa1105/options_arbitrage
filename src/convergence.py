"""Pure endpoint convergence labels, independent of file I/O."""

import numpy as np
import pandas as pd


def build_future_convergence_target(data, horizon_minutes, min_current_deviation=1.0,
                                    tolerance_minutes=5, reduction_threshold=.5):
    """Measure reduction at the first quote at/after the endpoint horizon.

    This is not a first-passage label: a temporary earlier convergence does not
    count if the deviation has widened again at the endpoint. Labels crossing
    trading sessions or expiry, or using nonfinite deviations, are invalid.
    """
    if horizon_minutes <= 0 or tolerance_minutes < 0:
        raise ValueError("Invalid horizon or tolerance")
    frame = data[["instrument_key", "timestamp", "expiry", "signal_eligible"]].copy()
    frame["timestamp"] = pd.to_datetime(frame.timestamp)
    if frame.duplicated(["instrument_key", "timestamp"]).any():
        raise ValueError("Duplicate contract/timestamp label inputs")
    frame["current_absolute_deviation"] = (
        data["current_absolute_deviation"].to_numpy()
        if "current_absolute_deviation" in data
        else data["reference_price_deviation"].abs().to_numpy()
    )
    left = frame.copy()
    left["target_time"] = left.timestamp + pd.Timedelta(minutes=horizon_minutes)
    right = frame.drop(columns="signal_eligible").rename(columns={
        "timestamp": "future_timestamp", "expiry": "future_expiry",
        "current_absolute_deviation": "future_absolute_deviation",
    })
    merged = pd.merge_asof(
        left.sort_values(["target_time", "instrument_key"]),
        right.sort_values(["future_timestamp", "instrument_key"]),
        left_on="target_time", right_on="future_timestamp", by="instrument_key",
        direction="forward", tolerance=pd.Timedelta(minutes=tolerance_minutes),
    )
    current, future = merged.current_absolute_deviation, merged.future_absolute_deviation
    valid = (
        merged.signal_eligible.fillna(False).astype(bool)
        & np.isfinite(current) & (current >= min_current_deviation)
        & np.isfinite(future) & (future >= 0)
        & merged.expiry.eq(merged.future_expiry)
        & merged.timestamp.dt.date.eq(merged.future_timestamp.dt.date)
    )
    reduction = (current - future) / current.where(current > 0)
    reduction = reduction.replace([np.inf, -np.inf], np.nan)
    suffix = f"{horizon_minutes}m"
    return pd.DataFrame({
        "instrument_key": merged.instrument_key,
        "timestamp": merged.timestamp,
        f"future_timestamp_{suffix}": merged.future_timestamp,
        f"future_deviation_{suffix}": future,
        f"deviation_reduction_{suffix}": reduction,
        f"label_valid_{suffix}": valid,
        f"converged_{suffix}": valid & (reduction >= reduction_threshold),
    })
