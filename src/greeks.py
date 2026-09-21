import numpy as np
from scipy.stats import norm


def calculate_greeks(
    spot,
    strike,
    time_to_expiry,
    volatility,
    risk_free_rate,
    option_type,
):
    """
    Calculate Black-Scholes Greeks.

    Returns
    -------
    delta
    gamma
    vega
    theta
    """

    if (
        not np.isfinite(spot)
        or not np.isfinite(strike)
        or not np.isfinite(time_to_expiry)
        or not np.isfinite(volatility)
    ):
        return np.nan, np.nan, np.nan, np.nan

    if (
        spot <= 0
        or strike <= 0
        or time_to_expiry <= 0
        or volatility <= 0
    ):
        return np.nan, np.nan, np.nan, np.nan

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

    pdf_d1 = norm.pdf(d1)

    # --------------------------------------------------------
    # Gamma
    # Same for calls and puts.
    # --------------------------------------------------------

    gamma = (
        pdf_d1
        / (spot * volatility * sqrt_t)
    )

    # --------------------------------------------------------
    # Vega
    #
    # This is price change for a 1.00 change in volatility.
    # Divide by 100 if you want per 1 volatility percentage
    # point.
    # --------------------------------------------------------

    vega = (
        spot
        * pdf_d1
        * sqrt_t
        / 100.0
    )

    # --------------------------------------------------------
    # Delta and Theta
    # --------------------------------------------------------

    discounted_strike = (
        strike
        * np.exp(-risk_free_rate * time_to_expiry)
    )

    if option_type == "CE":

        delta = norm.cdf(d1)

        theta = (
            -(
                spot
                * pdf_d1
                * volatility
            )
            / (2.0 * sqrt_t)
            - risk_free_rate
            * discounted_strike
            * norm.cdf(d2)
        )

    elif option_type == "PE":

        delta = norm.cdf(d1) - 1.0

        theta = (
            -(
                spot
                * pdf_d1
                * volatility
            )
            / (2.0 * sqrt_t)
            + risk_free_rate
            * discounted_strike
            * norm.cdf(-d2)
        )

    else:

        return np.nan, np.nan, np.nan, np.nan

    # Convert theta from per-year to per-day.
    theta = theta / 365.0

    return delta, gamma, vega, theta