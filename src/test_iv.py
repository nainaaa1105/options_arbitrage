from implied_volatility import implied_volatility


def main():

    spot = 25000
    strike = 25000
    T = 30 / 365
    rate = 0.06

    # Price generated approximately from
    # the previous Black-Scholes test.
    market_price = 633.9811

    iv, status = implied_volatility(
        market_price=market_price,
        spot=spot,
        strike=strike,
        time_to_expiry=T,
        risk_free_rate=rate,
        option_type="CE",
    )

    print("=" * 60)
    print("IMPLIED VOLATILITY TEST")
    print("=" * 60)

    print(f"Market price: {market_price}")
    print(f"IV:           {iv}")
    print(f"Status:       {status}")

    if status != "valid":
        raise RuntimeError(
            "IV solver test failed."
        )

    if abs(iv - 0.20) > 0.001:
        raise RuntimeError(
            f"Expected IV approximately 0.20, got {iv}"
        )

    print()
    print("IMPLIED VOLATILITY TEST PASSED")


if __name__ == "__main__":
    main()