"""Join completed decisions to the first synchronized captured quote snapshot."""

import argparse
import json
from pathlib import Path
import pandas as pd
from forward_paper import stamp, canonical


def assemble(session, decisions, quotes_file, output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((session/"protocol.json").read_text())
    quotes = pd.read_parquet(quotes_file)
    quotes["received_at"] = pd.to_datetime(quotes.received_at, utc=True)
    groups = list(quotes.groupby("received_at", sort=True))
    events, unavailable = [], []
    for line in decisions.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        start = stamp(record["decision_at"])
        found = None
        for receipt, frame in groups:
            if not start <= receipt <= start+pd.Timedelta(seconds=protocol["max_fill_delay_seconds"]):
                continue
            packet = {}
            for asset in ("option", "future"):
                selected = frame.loc[frame.instrument_key.eq(protocol["contracts"][asset]["instrument_key"])]
                if len(selected) != 1 or not bool(selected.iloc[0].executable_quote_valid):
                    break
                row = selected.iloc[0]
                packet[asset] = {"instrument_key": row.instrument_key, "quote_time": stamp(row.quote_time).isoformat(),
                                 **{name: float(row[name]) for name in ("bid", "ask", "bid_size", "ask_size")}}
            if len(packet) != 2:
                continue
            times = [stamp(quote["quote_time"]) for quote in packet.values()]
            if (min(times)-start).total_seconds() < protocol["minimum_latency_seconds"] or (max(times)-min(times)).total_seconds() > protocol["max_quote_skew_seconds"]:
                continue
            found = {**record, "received_at": receipt.isoformat(), "quotes": packet}
            break
        if found:
            events.append(found)
        else:
            unavailable.append({"event_id": record["event_id"], "reason": "No synchronized executable snapshot after decision; no fill invented"})
    (output/"events.jsonl").write_text("\n".join(canonical(event) for event in events)+("\n" if events else ""))
    (output/"unavailable.json").write_text(json.dumps(unavailable, indent=2))
    return events, unavailable


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("session", "decisions", "quotes", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    events, skipped = assemble(args.session, args.decisions, args.quotes, args.output)
    print(f"Prepared {len(events)} packets; {len(skipped)} unavailable decisions recorded")
