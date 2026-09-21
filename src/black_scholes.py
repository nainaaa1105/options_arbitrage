import numpy as np
from scipy.stats import norm


def black_scholes_price(
    spot,
    strike,
    time_to_expiry,
    volatility,
    risk_free_rate,
    option_type,
):
    """
    Calculate Black-Scholes theoretical option price.

    option_type:
        CE = Call
        PE = Put
    """

    if spot <= 0:
        return np.nan

    if strike <= 0:
        return np.nan

    if time_to_expiry <= 0:
        if option_type == "CE":
            return max(spot - strike, 0.0)

        if option_type == "PE":
            return max(strike - spot, 0.0)

        return np.nan

    if volatility <= 0:
        return np.nan

    sqrt_t = np.sqrt(time_to_expiry)

    d1 = (
        np.log(spot / strike)
        + (
            risk_free_rate
            + 0.5 * volatility ** 2
        ) * time_to_expiry
    ) / (
        volatility * sqrt_t
    )

    d2 = d1 - volatility * sqrt_t

    discounted_strike = (
        strike
        * np.exp(-risk_free_rate * time_to_expiry)
    )

    if option_type == "CE":

        return (
            spot * norm.cdf(d1)
            - discounted_strike * norm.cdf(d2)
        )

    elif option_type == "PE":

        return (
            discounted_strike * norm.cdf(-d2)
            - spot * norm.cdf(-d1)
        )

    return np.nan


def intrinsic_value(
    spot,
    strike,
    option_type,
):
    """
    Calculate option intrinsic value.
    """

    if option_type == "CE":
        return max(spot - strike, 0.0)

    if option_type == "PE":
        return max(strike - spot, 0.0)

    return np.nan