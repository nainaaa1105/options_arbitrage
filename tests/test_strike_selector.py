import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from upstox_api import get_expired_contracts

from strike_selector import (
    get_atm_strike,
    select_strikes,
    filter_contracts_by_strikes
)


EXPIRY_DATE = "2025-09-23"


print("STRIKE SELECTOR TEST")

# 1. Get contracts-

contracts = get_expired_contracts(EXPIRY_DATE)

print("\nTotal contracts:", len(contracts))


# 2. Inspect first contract

print("\nFirst contract:")
print(contracts[0])


# 3. Extract strikes

available_strikes = sorted(
    set(
        contract["strike_price"]
        for contract in contracts
        if contract.get("instrument_type") in {"CE", "PE"}
    )
)

print("\nNumber of available strikes:", len(available_strikes))

print(
    "Strike range:",
    min(available_strikes),
    "→",
    max(available_strikes)
)


# 4. Test ATM

test_spot = 25237.5

atm = get_atm_strike(
    test_spot,
    available_strikes
)

print("\nNIFTY spot:", test_spot)
print("ATM strike:", atm)


# 5. Select strikes

selected = select_strikes(
    atm,
    available_strikes,
    strikes_each_side=10
)

print("\nSelected strikes:")
print(selected)

print("Number of strikes:", len(selected))


# 6. Filter contracts

selected_contracts = filter_contracts_by_strikes(
    contracts,
    selected
)

print("\nSelected contracts:", len(selected_contracts))


# 7. CE / PE count

ce_count = sum(
    1
    for c in selected_contracts
    if c.get("instrument_type") == "CE"
)

pe_count = sum(
    1
    for c in selected_contracts
    if c.get("instrument_type") == "PE"
)

print("CE contracts:", ce_count)
print("PE contracts:", pe_count)


print("TEST COMPLETE")