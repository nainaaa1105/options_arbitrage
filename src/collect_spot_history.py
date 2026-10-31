"""Collect independent historical spot candles using the project's API client.

Minute requests are split by calendar month. Daily history is useful for HV even
when the option sample has gaps. Existing output files are never overwritten.
"""

import argparse
import json
import urllib.parse
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


def request_windows(first, last, unit):
    if first > last:
        raise ValueError("from-date must not be after to-date")
    if unit == "days":
        if (last - first).days > 3650:
            raise ValueError("Daily requests must cover at most ten years")
        return [(first, last)]
    if unit != "minutes":
        raise ValueError("unit must be minutes or days")
    windows, current = [], first
    while current <= last:
        next_month = (pd.Timestamp(current).to_period("M") + 1).start_time.date()
        end = min(last, next_month - timedelta(days=1))
        windows.append((current, end))
        current = end + timedelta(days=1)
    return windows


def collect(first, last, unit, interval, output, instrument_key="NSE_INDEX|Nifty 50"):
    if output.exists():
        raise FileExistsError(f"Preserving existing output: {output}")
    if unit == "days" and interval != 1 or unit == "minutes" and not 1 <= interval <= 15:
        raise ValueError("Supported durations are daily or 1-15 minute bars")
    if last >= pd.Timestamp.now(tz="Asia/Kolkata").date():
        raise ValueError("Use completed historical dates; this command excludes the current session")
    from upstox_api import make_request
    encoded = urllib.parse.quote(instrument_key, safe="")
    windows = request_windows(first, last, unit)
    frames = []
    for start, end in windows:
        payload = make_request(f"https://api.upstox.com/v3/historical-candle/{encoded}/{unit}/{interval}/{end}/{start}")
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise RuntimeError(f"No {unit} candles returned for {start}..{end}; output was not written")
        frame = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
        if not frame.timestamp.dt.date.between(start, end).all():
            raise ValueError("API response contains dates outside the requested window")
        frames.append(frame)
        print(f"Received {len(frame):,} bars for {start}..{end}", flush=True)
    result = pd.concat(frames, ignore_index=True).sort_values("timestamp")
    if result.timestamp.duplicated().any():
        raise ValueError("Duplicate candle timestamps; reconcile the source before using history")
    result["bar_unit"] = unit
    result["bar_interval"] = interval
    if unit == "minutes":
        result["bar_minutes"] = interval
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    output.with_suffix(".manifest.json").write_text(json.dumps({
        "instrument_key": instrument_key, "unit": unit, "interval": interval,
        "from_date": str(first), "to_date": str(last), "rows": len(result),
        "source": "Upstox historical V3", "created_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-date", type=date.fromisoformat, required=True)
    parser.add_argument("--to-date", type=date.fromisoformat, required=True)
    parser.add_argument("--unit", choices=["days", "minutes"], default="days")
    parser.add_argument("--interval", type=int, default=1)
    parser.add_argument("--instrument-key", default="NSE_INDEX|Nifty 50")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    collect(args.from_date, args.to_date, args.unit, args.interval, args.output, args.instrument_key)
