"""Bounded read-only Upstox quote capture with raw snapshots and quality flags.

Use active contracts, including the tradable hedge instrument. REST snapshots
are not a reconstruction of historical bid/ask quotes or every exchange tick.
"""

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv


def parse_quote_time(value):
    if value is None or value == "":
        return pd.NaT
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return pd.to_datetime(float(value), unit="ms", utc=True)
    result = pd.Timestamp(value)
    if result.tzinfo is None:
        raise ValueError("Quote timestamps must include a timezone")
    return result.tz_convert("UTC")


def normalize_snapshot(payload, received_at, max_age_seconds=5.):
    received = parse_quote_time(received_at)
    if pd.isna(received):
        raise ValueError("A recorded receipt time is required")
    rows = []
    for key, quote in payload.get("data", {}).items():
        buy = [p for p in quote.get("depth", {}).get("buy", [])
               if p.get("quantity", 0) > 0 and p.get("price", 0) > 0]
        sell = [p for p in quote.get("depth", {}).get("sell", [])
                if p.get("quantity", 0) > 0 and p.get("price", 0) > 0]
        best_bid = max(buy, key=lambda p: p["price"]) if buy else {}
        best_ask = min(sell, key=lambda p: p["price"]) if sell else {}
        bid, ask = best_bid.get("price", np.nan), best_ask.get("price", np.nan)
        updated = parse_quote_time(quote.get("timestamp"))
        age = (received - updated).total_seconds() if pd.notna(updated) else np.nan
        if not np.isfinite(bid) or not np.isfinite(ask):
            reason = "missing_two_sided_depth"
        elif bid > ask:
            reason = "crossed_book"
        elif not np.isfinite(age):
            reason = "missing_quote_timestamp"
        elif age < 0:
            reason = "quote_after_receipt_clock_error"
        elif age > max_age_seconds:
            reason = "stale_snapshot"
        else:
            reason = "valid"
        rows.append({
            "instrument_key": quote.get("instrument_token", key),
            "quote_time": updated, "received_at": received,
            "last_trade_time": parse_quote_time(quote.get("last_trade_time")),
            "bid": bid, "ask": ask, "bid_size": best_bid.get("quantity", 0),
            "ask_size": best_ask.get("quantity", 0),
            "mid": (bid + ask) / 2 if reason == "valid" else np.nan,
            "spread": ask - bid if np.isfinite(bid + ask) else np.nan,
            "last_price": quote.get("last_price", np.nan),
            "age_seconds": age, "quality_reason": reason,
            "executable_quote_valid": reason == "valid",
        })
    return pd.DataFrame(rows)


def capture(keys, output, snapshots=60, interval_seconds=1., max_age_seconds=5., transport="project"):
    if not keys or len(keys) > 500:
        raise ValueError("Supply between 1 and 500 active instrument keys")
    if snapshots < 1 or interval_seconds < 1 or max_age_seconds < 0:
        raise ValueError("Invalid capture count, interval or maximum age")
    output.mkdir(parents=True, exist_ok=False)
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is not configured")
    query = urllib.parse.urlencode({"instrument_key": ",".join(keys)})
    request = urllib.request.Request(
        f"https://api.upstox.com/v2/market-quote/quotes?{query}",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
    )
    frames = []
    status = "incomplete"
    try:
        with (output / "raw_snapshots.jsonl").open("x", encoding="utf-8") as raw:
            for index in range(snapshots):
                if transport == "project":
                    from upstox_api import make_request
                    payload = make_request(request.full_url)
                elif transport == "stdlib":
                    try:
                        with urllib.request.urlopen(request, context=ssl.create_default_context(), timeout=30) as response:
                            payload = json.load(response)
                    except urllib.error.HTTPError as error:
                        raise RuntimeError(f"Quote capture returned HTTP {error.code}; no quotes were fabricated") from None
                else:
                    raise ValueError("transport must be project or stdlib")
                received = datetime.now(timezone.utc).isoformat()
                if payload.get("status") != "success" or not payload.get("data"):
                    raise RuntimeError("Quote response did not contain a successful nonempty snapshot")
                raw.write(json.dumps({"received_at": received, "payload": payload}) + "\n")
                raw.flush()
                frames.append(normalize_snapshot(payload, received, max_age_seconds))
                print(f"Captured snapshot {index + 1}/{snapshots}", flush=True)
                if index + 1 < snapshots:
                    time.sleep(interval_seconds)
            status = "complete"
    finally:
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(output / "quotes.parquet", index=False)
        (output / "capture_manifest.json").write_text(json.dumps({
            "status": status, "requested_snapshots": snapshots, "completed_snapshots": len(frames),
            "instrument_keys": keys, "interval_seconds": interval_seconds,
            "max_age_seconds": max_age_seconds, "finished_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys-file", type=Path, required=True, help="One active instrument key per line; include hedge futures")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshots", type=int, default=60)
    parser.add_argument("--interval-seconds", type=float, default=1.)
    parser.add_argument("--max-age-seconds", type=float, default=5.)
    parser.add_argument("--transport", choices=["project", "stdlib"], default="project")
    args = parser.parse_args()
    keys = [line.strip() for line in args.keys_file.read_text().splitlines() if line.strip()]
    capture(keys, args.output, args.snapshots, args.interval_seconds, args.max_age_seconds, args.transport)
