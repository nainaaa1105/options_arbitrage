from greeks import calculate_greeks


def main():

    delta, gamma, vega, theta = calculate_greeks(
        spot=25000,
        strike=25000,
        time_to_expiry=30 / 365,
        volatility=0.20,
        risk_free_rate=0.06,
        option_type="CE",
    )

    print("=" * 60)
    print("GREEKS TEST")
    print("=" * 60)

    print(f"Delta: {delta}")
    print(f"Gamma: {gamma}")
    print(f"Vega:  {vega}")
    print(f"Theta: {theta}")

    if not -1 <= delta <= 1:
        raise RuntimeError("Invalid Delta.")

    if gamma < 0:
        raise RuntimeError("Invalid Gamma.")

    if vega < 0:
        raise RuntimeError("Invalid Vega.")

    print()
    print("GREEKS TEST PASSED")


if __name__ == "__main__":
    main()