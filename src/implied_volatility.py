import numpy as np
from scipy.optimize import brentq

from black_scholes import black_scholes_price
from config import (
    IV_MIN,
    IV_MAX,
    IV_SOLVER_XTOL,
    MIN_OPTION_PRICE,
)


def no_arbitrage_bounds(
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    option_type,
):
    """
    Calculate theoretical lower and upper bounds
    for a European option under Black-Scholes assumptions.

    Returns
    -------
    lower_bound, upper_bound
    """

    if (
        spot <= 0
        or strike <= 0
        or time_to_expiry <= 0
    ):
        return np.nan, np.nan

    discount_factor = np.exp(
        -risk_free_rate * time_to_expiry
    )

    if option_type == "CE":

        lower_bound = max(
            spot - strike * discount_factor,
            0.0,
        )

        upper_bound = spot

    elif option_type == "PE":

        lower_bound = max(
            strike * discount_factor - spot,
            0.0,
        )

        upper_bound = (
            strike * discount_factor
        )

    else:
        return np.nan, np.nan

    return lower_bound, upper_bound


def implied_volatility(
    market_price,
    spot,
    strike,
    time_to_expiry,
    risk_free_rate,
    option_type,
):
    """
    Calculate implied volatility using Brent's method.

    Returns
    -------
    iv, status
    """

    # --------------------------------------------------------
    # Basic validity
    # --------------------------------------------------------

    if not np.isfinite(market_price):
        return np.nan, "invalid_market_price"

    if not np.isfinite(spot):
        return np.nan, "invalid_spot"

    if not np.isfinite(strike):
        return np.nan, "invalid_strike"

    if not np.isfinite(time_to_expiry):
        return np.nan, "invalid_time"

    if market_price <= 0:
        return np.nan, "zero_or_negative_price"

    if spot <= 0 or strike <= 0:
        return np.nan, "invalid_parameters"

    if time_to_expiry <= 0:
        return np.nan, "expired"

    # --------------------------------------------------------
    # No-arbitrage bounds
    # --------------------------------------------------------

    lower_bound, upper_bound = no_arbitrage_bounds(
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        option_type=option_type,
    )

    if not np.isfinite(lower_bound):
        return np.nan, "invalid_bounds"

    if market_price < lower_bound:
        return np.nan, "below_lower_bound"

    if market_price > upper_bound:
        return np.nan, "above_upper_bound"

    # --------------------------------------------------------
    # Black-Scholes equation
    # --------------------------------------------------------

    def objective(volatility):

        theoretical_price = black_scholes_price(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            volatility=volatility,
            risk_free_rate=risk_free_rate,
            option_type=option_type,
        )

        return theoretical_price - market_price

    try:

        price_low = objective(IV_MIN)
        price_high = objective(IV_MAX)

        # If the market price cannot be bracketed,
        # Brent's method cannot solve it.
        if price_low * price_high > 0:

            return np.nan, "solver_not_bracketed"

        iv = brentq(
            objective,
            IV_MIN,
            IV_MAX,
            xtol=IV_SOLVER_XTOL,
            maxiter=100,
        )

    except Exception:
        return np.nan, "solver_failed"

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if not np.isfinite(iv):
        return np.nan, "invalid_iv"

    if iv <= 0:
        return np.nan, "invalid_iv"

    if iv > IV_MAX:
        return np.nan, "iv_above_limit"

    return iv, "valid"