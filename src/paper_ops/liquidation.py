"""Conservative liquidation equity: close everything at the last recorded bid/ask, with fees, slippage and depth."""
import json
import math
import sqlite3
from pathlib import Path

import pandas as pd

from deep_hedging.accounting import FuturesPosition
from deep_hedging.config import HedgeConfig
from deep_hedging.costs import execution_price, transaction_fees


def _last_quoted_event(db):
    for (payload,) in db.execute("SELECT payload FROM events ORDER BY rowid DESC"):
        event = json.loads(payload)
        if isinstance(event.get("quotes"), dict) and {"option", "future"} <= set(event["quotes"]):
            return event
    return None


def liquidation_report(session_dir, now=None):
    d = Path(session_dir)
    protocol = json.loads((d / "protocol.json").read_text())
    config = HedgeConfig(**protocol["execution_config"])
    db = sqlite3.connect((d / "paper.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        row = db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        event = _last_quoted_event(db)
    finally:
        db.close()
    if row is None:
        return {"exposure": False, "mid_equity": config.initial_capital, "liquidation_equity": config.initial_capital,
                "executable": True, "note": "No state recorded yet"}
    s = json.loads(row[0])
    book = FuturesPosition(s["h"], s["average"])
    om, fm = s.get("last_option_mid"), s.get("last_future_mid")
    mid = s["cash"] + (s["q"] * om if s["q"] else 0.) + (book.unrealized(fm) if s["h"] else 0.)
    out = {"exposure": bool(s["q"] or s["h"]), "option_quantity": s["q"], "future_quantity": s["h"], "mid_equity": mid,
           "cash": s["cash"], "basis": "last accepted state marks vs last recorded quote book"}
    if not out["exposure"]:
        return {**out, "liquidation_equity": s["cash"], "executable": True, "liquidation_cost_vs_mid": s["cash"] - mid}
    if event is None:
        return {**out, "liquidation_equity": None, "executable": False, "reason": "no_quote_book_recorded"}
    received = pd.Timestamp(event["received_at"])
    cash, fees, reasons, ages = s["cash"], 0., [], {}
    stamp_now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    for asset, change in (("option", -s["q"]), ("future", -s["h"])):
        if not change:
            continue
        quote = event["quotes"][asset]
        ages[asset] = (stamp_now - pd.Timestamp(quote["quote_time"])).total_seconds()
        side_size = quote["ask_size"] if change > 0 else quote["bid_size"]
        if not all(math.isfinite(quote[k]) and quote[k] > 0 for k in ("bid", "ask", "bid_size", "ask_size")) or quote["bid"] > quote["ask"]:
            reasons.append(f"{asset}_invalid_book")
            continue
        if abs(change) > side_size:
            reasons.append(f"{asset}_insufficient_displayed_depth")
            continue
        mark = quote["ask"] if change > 0 else quote["bid"]
        fill = execution_price(mark, change, protocol["contracts"][asset]["tick_size"], 0., getattr(config, asset + "_slippage"))
        cost = transaction_fees(asset, change, fill, received, config)["total"]
        fees += cost
        cash += (-change * fill if asset == "option" else book.trade(change, fill)) - cost
    if reasons:
        return {**out, "liquidation_equity": None, "executable": False, "reason": ",".join(reasons), "quote_age_seconds": ages}
    return {**out, "liquidation_equity": cash, "executable": True, "liquidation_fees": fees, "quote_age_seconds": ages,
            "liquidation_cost_vs_mid": cash - mid}
