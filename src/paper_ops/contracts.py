"""Deterministic contract selection from a dated Upstox instrument-master snapshot.

Rule (frozen in the validation protocol): from the nearest NIFTY expiry that still has at least `min_trading_days`
trading days left, take the call and put nearest the reference spot (ATM) and the call and put about `otm_fraction`
out of the money, plus the front-month future that outlives the option. Re-selection is required when fewer than
`reroll_days` trading days remain or the option has drifted more than `max_drift` from ATM. Reads real expiry dates,
so Monday/Tuesday expiry changes and holiday shifts need no special case. No network access; weekday holidays are
not known to this module, so remaining-day counts treat every weekday as a trading day (conservative for the roll).
"""
import gzip
import json
import math
from pathlib import Path

import pandas as pd

IST = "Asia/Kolkata"
UNDERLYING = "NSE_INDEX|Nifty 50"


def load_master(path):
    raw = Path(path).read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8")
    records = [r for r in json.loads(text) if r.get("segment") == "NSE_FO" and r.get("underlying_key") == UNDERLYING]
    if not records:
        raise ValueError("No NIFTY NSE_FO records in the instrument master")
    return records


def expiry_date(record):
    return pd.Timestamp(record["expiry"], unit="ms", tz="UTC").tz_convert(IST).normalize()


def trading_days_left(today, expiry):
    """Weekdays from `today` through `expiry` inclusive (holidays unknown, so this never undercounts risk)."""
    today, expiry = pd.Timestamp(today).tz_localize(None).normalize(), pd.Timestamp(expiry).tz_localize(None).normalize()
    return int(len(pd.bdate_range(today, expiry))) if expiry >= today else 0


def to_contract(record, asset_type):
    expiry = expiry_date(record) + pd.Timedelta(hours=15, minutes=30)
    out = {"instrument_key": record["instrument_key"], "asset_type": asset_type, "lot_size": int(record["lot_size"]),
           "tick_size": float(record["tick_size"]) / 100., "expiry_at": expiry.isoformat()}
    if asset_type == "OPT":
        out.update(option_type=record["instrument_type"], strike=float(record["strike_price"]))
    return out


def select_contracts(records, today, spot, min_trading_days=3, otm_fraction=.01):
    today = pd.Timestamp(today)
    options = [r for r in records if r.get("instrument_type") in ("CE", "PE")]
    expiries = sorted({expiry_date(r) for r in options})
    usable = [e for e in expiries if trading_days_left(today, e) >= min_trading_days]
    if not usable:
        raise ValueError("No expiry with enough trading days remaining")
    expiry = usable[0]
    chain = [r for r in options if expiry_date(r) == expiry]

    def nearest(kind, target):
        pool = [r for r in chain if r["instrument_type"] == kind]
        return min(pool, key=lambda r: (abs(float(r["strike_price"]) - target), float(r["strike_price"])))

    targets = {"ATM": (spot, spot), "OTM": (spot * (1 + otm_fraction), spot * (1 - otm_fraction))}
    futures = sorted((r for r in records if r.get("instrument_type") == "FUT" and expiry_date(r) >= expiry), key=expiry_date)
    if not futures:
        raise ValueError("No hedge future outlives the selected option")
    future = to_contract(futures[0], "FUT")
    selected = {}
    for label, (call_target, put_target) in targets.items():
        for kind, target in (("CE", call_target), ("PE", put_target)):
            record = nearest(kind, target)
            selected[f"{label}_{kind}"] = {"option": to_contract(record, "OPT"), "future": future,
                                           "reference_spot": float(spot), "selected_on": str(today.date()), "moneyness": label,
                                           "otm_fraction": 0. if label == "ATM" else otm_fraction,
                                           "trading_days_left": trading_days_left(today, expiry)}
    keys = [v["option"]["instrument_key"] for v in selected.values()]
    if len(set(keys)) != len(keys):
        raise ValueError("Selection produced duplicate contracts; widen the OTM distance")
    return selected


def needs_reroll(contract, today, spot, reroll_days=2, max_drift=.02):
    """Return the reasons a previously selected contract must be replaced (empty list means keep it)."""
    reasons = []
    expiry = pd.Timestamp(contract["option"]["expiry_at"])
    if trading_days_left(today, expiry) < reroll_days:
        reasons.append("fewer_than_reroll_days_remaining")
    sign = 1. if contract["option"]["option_type"] == "CE" else -1.
    signed = math.log(contract["option"]["strike"] / spot) * sign          # positive = out of the money
    if abs(signed - contract["otm_fraction"]) > max_drift:
        reasons.append("moneyness_drift")
    return reasons
