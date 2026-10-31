"""Separate option premium and futures notional transaction accounting."""

import math
import pandas as pd

from .config import HedgeConfig


def execution_price(mark, quantity, tick_size, half_spread, slippage):
    if not all(math.isfinite(x) for x in (mark, quantity, tick_size, half_spread, slippage)):
        raise ValueError("Non-finite execution input")
    if mark <= 0 or tick_size <= 0 or half_spread < 0 or slippage < 0 or quantity == 0:
        raise ValueError("Invalid execution input")
    desired = mark + math.copysign(half_spread + slippage, quantity)
    if desired <= 0:
        raise ValueError("No positive executable proxy price")
    ticks = desired / tick_size
    result = (math.ceil(ticks - 1e-9) if quantity > 0 else math.floor(ticks + 1e-9)) * tick_size
    if result <= 0:
        raise ValueError("No positive executable proxy price")
    return float(result)


def transaction_fees(asset, quantity, fill, timestamp, config: HedgeConfig):
    if asset not in ("option", "future") or not math.isfinite(quantity) or not math.isfinite(fill) or fill <= 0:
        raise ValueError("Invalid fee input")
    if quantity == 0:
        return {k: 0. for k in ("brokerage", "exchange", "sebi", "ipft", "stt", "stamp", "gst", "total")}
    turnover = abs(quantity) * fill
    ts = pd.Timestamp(timestamp)
    ts = ts.tz_localize("Asia/Kolkata") if ts.tz is None else ts.tz_convert("Asia/Kolkata")
    after = ts >= pd.Timestamp("2026-04-01", tz="Asia/Kolkata")
    if asset == "option":
        brokerage = config.brokerage_per_order
        exchange, ipft, stamp = config.option_exchange_rate, config.option_ipft_rate, config.option_stamp_rate
        stt = config.option_stt_override
        stt = (.0015 if after else .001) if stt is None else stt
    else:
        brokerage = min(config.brokerage_per_order, config.future_brokerage_cap_rate * turnover)
        exchange, ipft, stamp = config.future_exchange_rate, config.future_ipft_rate, config.future_stamp_rate
        stt = config.future_stt_override
        stt = (.0005 if after else .0002) if stt is None else stt
    fees = {"brokerage": brokerage, "exchange": turnover * exchange,
            "sebi": turnover * config.sebi_rate, "ipft": turnover * ipft,
            "stt": turnover * stt if quantity < 0 else 0.,
            "stamp": turnover * stamp if quantity > 0 else 0.}
    fees["gst"] = config.gst_rate * sum(fees[k] for k in ("brokerage", "exchange", "sebi", "ipft"))
    fees["total"] = sum(fees.values())
    return fees
