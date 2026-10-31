"""Decision-slot option universe: Part 1 admission rules plus the classifier's own domain."""
import numpy as np
import pandas as pd

from deep_hedging.config import HedgeConfig

DECISION_TIMES = ("09:30", "10:30", "11:30", "12:30", "13:30", "14:30")
STEP = pd.Timedelta(minutes=5)


def slot_valid(decision, config: HedgeConfig):
    """Slot-level window rule copied from the Part 1 builder."""
    entry = decision + STEP
    end = entry + pd.Timedelta(minutes=config.horizon_minutes)
    return not (end.date() != decision.date() or end.time() > pd.Timestamp("15:25").time()
                or decision.time() < pd.Timestamp("09:15").time())


def slot_frame(options, config: HedgeConfig, decision_times=DECISION_TIMES, domain=None):
    """Rows observed at decision bars with admission/domain flags.

    ``options`` is the validated Part 1 option frame. ``domain`` optionally supplies
    signal_eligible/reference_price_deviation/session_ok keyed by instrument and timestamp.
    Nothing here reads a later bar.
    """
    clock = options.timestamp.dt.strftime("%H:%M")
    d = options.loc[clock.isin(decision_times)].copy()
    d["decision_time"] = d.timestamp
    d["entry_time"] = d.timestamp + STEP
    d["end_time"] = d.entry_time + pd.Timedelta(minutes=config.horizon_minutes)
    expiry_time = pd.to_datetime(d.expiry.astype(str)).dt.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
    d["admitted"] = (np.isfinite(d.delta) & d.delta.abs().le(1) & d.close.ge(1) & d.volume.gt(0)
                     & d.open_interest.gt(0) & d.nifty_spot.gt(0) & expiry_time.gt(d.end_time)
                     & d.decision_time.map(lambda t: slot_valid(t, config)))
    if domain is not None:
        d = d.merge(domain, on=["instrument_key", "timestamp"], how="left", validate="one_to_one")
        d["in_domain"] = (d.signal_eligible.fillna(False).astype(bool) & d.reference_price_deviation.abs().ge(1.)
                          & d.session_ok.fillna(False).astype(bool))
        d["undervalued"] = d.reference_price_deviation.lt(0)
    return d
