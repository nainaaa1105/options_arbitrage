"""Policies receive only the state observed at the current execution boundary."""

import math
import numpy as np

STRATEGIES = ("unhedged", "static_delta", "dynamic_delta", "band_delta")


def target_lots(strategy, state, current_lots, initial, config):
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy}")
    if strategy == "unhedged":
        return 0, "unhedged"
    if strategy == "static_delta" and not initial:
        return current_lots, "static_hold"
    delta = state["delta"]
    sensitivity = state["future_spot_delta"]
    if not np.isfinite(delta) or abs(delta) > 1 or not np.isfinite(sensitivity) or sensitivity <= 0:
        return current_lots, "invalid_delta_hold"
    desired_units = -state["option_quantity"] * delta / sensitivity
    desired = desired_units / state["future_lot_size"]
    if strategy == "band_delta" and not initial:
        residual = abs(desired_units - current_lots * state["future_lot_size"])
        if residual <= config.no_trade_band * abs(state["option_quantity"]):
            return current_lots, "inside_no_trade_band"
    # Nearest executable lot; exact half-lot ties go toward zero.
    rounded = int(math.copysign(math.floor(abs(desired) + .5 - 1e-12), desired)) if desired else 0
    bounded = max(-config.max_future_lots, min(config.max_future_lots, rounded))
    return bounded, "exposure_capped" if bounded != rounded else "delta_target"
