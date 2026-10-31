"""Incremental, causal completed-bar state for the frozen paper runner.

Uses the same IV solver, Greeks and ACT/365 convention as the batch research pipeline. A state is produced only
from a COMPLETED five-minute bar; gaps, invalid prices and unsolved IV yield an explicit unavailable result, never a
carried-forward or interpolated value. Classifier context features need the full option chain and are NOT built here.
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import RISK_FREE_RATE
from greeks import calculate_greeks
from implied_volatility import implied_volatility

BAR = pd.Timedelta(minutes=5)


@dataclass(frozen=True)
class Unavailable:
    bar_start: pd.Timestamp
    reason: str


class CompletedBarStateBuilder:
    def __init__(self, contracts, risk_free_rate=RISK_FREE_RATE):
        option = contracts["option"]
        self.kind, self.strike = option["option_type"], float(option["strike"])
        self.expiry = pd.Timestamp(option["expiry_at"])
        self.r = risk_free_rate
        self._last = None          # (bar_start, spot_close)

    def push(self, bar_start, spot_close, option_close, future_close):
        """Feed one completed bar. Returns a state dict or an Unavailable record."""
        bar_start = pd.Timestamp(bar_start)
        if bar_start.tzinfo is None:
            raise ValueError("Timezone-aware bar start required")
        if self._last is not None and bar_start <= self._last[0]:
            return Unavailable(bar_start, "out_of_order_or_repeated_bar")
        available = bar_start + BAR
        contiguous = (self._last is not None and bar_start - self._last[0] == BAR
                      and bar_start.tz_convert("Asia/Kolkata").date() == self._last[0].tz_convert("Asia/Kolkata").date())
        previous_spot = self._last[1] if contiguous else None
        def usable(value):
            return isinstance(value, (int, float, np.floating)) and math.isfinite(value) and value > 0
        for name, value in (("spot", spot_close), ("option", option_close), ("future", future_close)):
            if not usable(value):
                # Never reuse an older spot across an unusable bar: the next return would span a gap.
                self._last = (bar_start, float(spot_close) if usable(spot_close) else None)
                return Unavailable(bar_start, f"invalid_{name}_close")
        self._last = (bar_start, float(spot_close))
        tte = max((self.expiry - available).total_seconds(), 0.) / (365. * 86400.)
        iv, status = implied_volatility(float(option_close), float(spot_close), self.strike, tte, self.r, self.kind)
        if status != "valid":
            return Unavailable(bar_start, f"iv_{status}")
        delta, *_ = calculate_greeks(float(spot_close), self.strike, tte, iv, self.r, self.kind)
        if not math.isfinite(delta):
            return Unavailable(bar_start, "invalid_delta")
        # Batch research uses recent_return = 0 on an episode's first observation; a gap/new session does the same.
        recent = math.log(spot_close / previous_spot) if previous_spot and previous_spot > 0 else 0.
        return {"bar_start": bar_start.isoformat(), "available_at": available.isoformat(), "spot_observed": float(spot_close),
                "option_close_observed": float(option_close), "future_close_observed": float(future_close), "delta": float(delta),
                "volatility": float(iv), "remaining_time": float(tte), "recent_return": float(recent), "signal_eligible": False}


def decision_record(event_id, kind, state):
    """Decision packet for assemble_paper_events: decided exactly when the bar completes."""
    if kind not in ("open", "hedge", "close"):
        raise ValueError("Unknown decision type")
    return {"event_id": event_id, "type": kind, "decision_at": state["available_at"], "state": state}
