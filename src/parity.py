"""Pure put-call parity calculation helpers."""

import numpy as np


def calculate_parity_residual(call_price, put_price, spot, strike, time_to_expiry, risk_free_rate):
    """Observed C-P less European parity value; supports scalars and arrays."""
    call_price = np.asarray(call_price, dtype=float)
    put_price = np.asarray(put_price, dtype=float)
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    time_to_expiry = np.asarray(time_to_expiry, dtype=float)
    risk_free_rate = np.asarray(risk_free_rate, dtype=float)
    discounted_strike = strike * np.exp(-risk_free_rate * time_to_expiry)
    return (call_price - put_price) - (spot - discounted_strike)
