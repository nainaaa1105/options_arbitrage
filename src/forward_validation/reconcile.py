"""Independent re-derivation of a paper account from its fill ledger; the runner's own arithmetic is not trusted."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

from deep_hedging.accounting import FuturesPosition


def reconcile_session(session_dir, tolerance=1e-6):
    d = Path(session_dir)
    protocol = json.loads((d / "protocol.json").read_text())
    initial = protocol["execution_config"]["initial_capital"]
    db = sqlite3.connect((d / "paper.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    try:
        events = db.execute("SELECT event_id,status FROM events ORDER BY rowid").fetchall()
        marks = {k: json.loads(v) for k, v in db.execute("SELECT event_id,payload FROM equity")}
        fills: dict[str, list] = {}
        for k, v in db.execute("SELECT event_id,payload FROM fills ORDER BY rowid"):
            fills.setdefault(k, []).append(json.loads(v))
        state = json.loads(db.execute("SELECT payload FROM state WHERE id=1").fetchone()[0]) if db.execute("SELECT COUNT(*) FROM state").fetchone()[0] else None
        counts = {t: db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in ("events", "fills", "equity")}
    finally:
        db.close()
    cash, book, option_units, fees = initial, FuturesPosition(), 0, 0.
    worst_cash = worst_position = 0.
    checked = 0
    for event_id, status in events:
        if status != "accepted":
            if event_id in fills or event_id in marks:
                return {"session": d.name, "ok": False, "reason": f"rejected event {event_id} left fills or marks"}
            continue
        for fill in fills.get(event_id, []):
            change, price, cost = fill["quantity_change"], fill["fill"], fill["fees"]
            fees += cost
            if fill["asset"] == "option":
                cash -= change * price + cost
                option_units += change
            else:
                cash += book.trade(change, price) - cost
        mark = marks[event_id]
        worst_cash = max(worst_cash, abs(cash - mark["cash"]))
        worst_position = max(worst_position, abs(option_units - mark["option_quantity"]), abs(book.quantity - mark["future_quantity"]))
        checked += 1
    state_fee_error = abs(fees - state["fees"]) if state else 0.
    closing = {}
    if state:
        closing = {"cash_error": abs(cash - state["cash"]), "option_error": abs(option_units - state["q"]), "future_error": abs(book.quantity - state["h"])}
    exported = {}
    for table, filename in (("events", "events.csv"), ("fills", "fills.csv"), ("equity", "equity.csv")):
        if (d / filename).exists():
            exported[table] = len(pd.read_csv(d / filename)) == counts[table]
    worst = max([worst_cash, worst_position, state_fee_error, *closing.values()])
    ok = worst <= tolerance and all(exported.values())
    return {"session": d.name, "ok": bool(ok), "events_checked": checked, "max_cash_error": worst_cash, "max_position_error": worst_position,
            "state_fee_error": state_fee_error, "closing_errors": closing, "exports_match_database": exported, "max_error": worst}
