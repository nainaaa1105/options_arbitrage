from black_scholes import black_scholes_price


def main():

    spot = 25000
    strike = 25000
    T = 30 / 365
    volatility = 0.20
    risk_free_rate = 0.06

    call = black_scholes_price(
        spot,
        strike,
        T,
        volatility,
        risk_free_rate,
        "CE",
    )

    put = black_scholes_price(
        spot,
        strike,
        T,
        volatility,
        risk_free_rate,
        "PE",
    )

    print("Black-Scholes test")
    print("=" * 50)

    print(f"Spot:       {spot}")
    print(f"Strike:     {strike}")
    print(f"T:          {T:.6f}")
    print(f"Volatility: {volatility}")
    print(f"Rate:       {risk_free_rate}")
    print()

    print(f"Call price: {call:.4f}")
    print(f"Put price:  {put:.4f}")

    if call <= 0:
        raise RuntimeError("Call price test failed.")

    if put <= 0:
        raise RuntimeError("Put price test failed.")

    print()
    print("BLACK-SCHOLES TEST PASSED")


if __name__ == "__main__":
    main()