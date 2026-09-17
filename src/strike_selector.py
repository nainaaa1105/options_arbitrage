def get_atm_strike(
    nifty_spot,
    available_strikes,
):
    """
    Return the strike closest to NIFTY spot.
    """

    if not available_strikes:
        raise ValueError(
            "No available strikes."
        )

    return min(
        available_strikes,
        key=lambda x: abs(x - nifty_spot)
    )


def select_strikes(
    atm_strike,
    available_strikes,
    strikes_each_side=10,
):
    """
    Select ATM +/- N strikes.

    Example:

    ATM = 25250
    strikes_each_side = 2

    Result:
    25150, 25200, 25250, 25300, 25350
    """

    available = sorted(
        set(float(x) for x in available_strikes)
    )

    if atm_strike not in available:
        atm_strike = get_atm_strike(
            atm_strike,
            available
        )

    atm_index = available.index(
        atm_strike
    )

    start = max(
        0,
        atm_index - strikes_each_side
    )

    end = min(
        len(available),
        atm_index + strikes_each_side + 1
    )

    selected = available[start:end]

    return selected


def filter_contracts_by_strikes(
    contracts,
    selected_strikes,
):
    """
    Keep CE and PE contracts whose strike
    belongs to selected_strikes.
    """

    selected = set(
        float(x) for x in selected_strikes
    )

    result = []

    for contract in contracts:

        strike = float(
            contract["strike_price"]
        )

        option_type = contract.get(
            "instrument_type"
        )

        if (
            strike in selected
            and option_type in ["CE", "PE"]
        ):
            result.append(contract)

    return result