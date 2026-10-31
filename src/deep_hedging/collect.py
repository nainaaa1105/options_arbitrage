"""Download additive front-month NIFTY futures history with provenance."""

import argparse
import hashlib
import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from collect_spot_history import request_windows
from diagnose_upstox import candle_quality
from upstox_api import get_expiries, get_expired_candles, make_request


def collect(first, last, output, spot_calendar=None, resume=False):
    if first > last or last >= pd.Timestamp.now(tz="Asia/Kolkata").date():
        raise ValueError("Supply an ordered range of completed historical dates")
    output.mkdir(parents=True, exist_ok=resume)
    if (output / "manifest.json").exists():
        raise FileExistsError("Preserving complete futures collection")
    expected_dates = None
    if spot_calendar is not None:
        calendar = pd.read_parquet(spot_calendar, columns=["timestamp"])
        expected_dates = set(pd.to_datetime(calendar.timestamp).dt.date)
    expiries = pd.Series(pd.to_datetime(get_expiries())).sort_values()
    monthly = expiries.groupby(expiries.dt.to_period("M")).max().dt.date.tolist()
    records, frames, requests = [], [], []
    previous = None
    for expiry in monthly:
        start = max(first, previous + timedelta(days=1)) if previous else first
        previous = expiry
        if expiry < first:
            continue
        if start > last:
            break
        payload = make_request("https://api.upstox.com/v2/expired-instruments/future/contract",
                               {"instrument_key": "NSE_INDEX|Nifty 50", "expiry_date": str(expiry)})
        contracts = [c for c in payload.get("data", []) if c.get("instrument_type") == "FUT"
                     and c.get("underlying_key") == "NSE_INDEX|Nifty 50"]
        if len(contracts) != 1:
            raise ValueError(f"Expected one front-month future for {expiry}; got {len(contracts)}")
        contract = contracts[0]
        if int(contract["lot_size"]) <= 0 or float(contract["tick_size"]) <= 0:
            raise ValueError("Missing valid futures contract units")
        # Upstox instrument metadata tick sizes are expressed in paise.
        metadata = {**contract, "tick_size_rupees": float(contract["tick_size"]) / 100.,
                    "tick_size_units": "API paise converted to rupees", "front_month_from": str(start)}
        records.append(metadata)
        cache = output / f"future_{expiry}.parquet"
        if resume and cache.exists():
            cached = pd.read_parquet(cache)
            if (cached.instrument_key.nunique() != 1 or cached.instrument_key.iloc[0] != contract["instrument_key"]
                    or not pd.to_datetime(cached.timestamp).dt.date.between(start, min(last, expiry)).all()
                    or not cached.lot_size.eq(int(contract["lot_size"])).all()
                    or not cached.tick_size.eq(metadata["tick_size_rupees"]).all()
                    or not cached.asset_type.eq("FUT").all() or not cached.bar_minutes.eq(5).all()
                    or cached.timestamp.duplicated().any()):
                raise ValueError(f"Cached contract failed resume validation: {cache}")
            frames.append(cached)
            requests.append({"expiry": str(expiry), "cached_rows": len(cached), "resume_validated": True})
            continue
        contract_frames = []
        for begin, end in request_windows(start, min(last, expiry), "minutes"):
            candles = get_expired_candles(contract["instrument_key"], str(begin), str(end))
            if not candles and expected_dates is not None and not any(begin <= day <= end for day in expected_dates):
                requests.append({"expiry": str(expiry), "from_date": str(begin), "to_date": str(end),
                                 "rows": 0, "status": "no_spot_sessions_in_window"})
                continue
            quality = candle_quality(candles, 5)
            if not quality["usable"]:
                raise ValueError(f"Invalid/empty futures candles {expiry}: {quality}")
            d = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
            d["timestamp"] = pd.to_datetime(d.timestamp).dt.tz_convert("Asia/Kolkata")
            if not d.timestamp.dt.date.between(begin, end).all():
                raise ValueError("Response contains dates outside request")
            d["instrument_key"] = contract["instrument_key"]
            d["expiry"] = pd.Timestamp(expiry)
            d["lot_size"] = int(contract["lot_size"])
            d["tick_size"] = metadata["tick_size_rupees"]
            d["trading_symbol"] = contract["trading_symbol"]
            d["asset_type"] = "FUT"
            d["bar_minutes"] = 5
            contract_frames.append(d)
            requests.append({"expiry": str(expiry), "from_date": str(begin), "to_date": str(end), **quality})
        if not contract_frames:
            raise ValueError(f"No futures data for complete front-month interval {expiry}")
        d = pd.concat(contract_frames, ignore_index=True).sort_values("timestamp")
        if d.timestamp.duplicated().any():
            raise ValueError("Duplicate front-month contract bars")
        d.to_parquet(output / f"future_{expiry}.parquet", index=False)
        frames.append(d)
        (output / "collection_progress.json").write_text(json.dumps({"status": "in_progress",
            "contracts": records, "requests": requests}, indent=2), encoding="utf-8")
        print(f"Collected {expiry}: {len(d):,} bars, lot={contract['lot_size']}", flush=True)
    if not frames:
        raise RuntimeError("No eligible futures contracts returned")
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    if combined.timestamp.duplicated().any():
        raise ValueError("Front-month windows overlap")
    destination = output / "front_month_futures.parquet"
    combined.to_parquet(destination, index=False)
    (output / "contracts.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    manifest = {"source": "Upstox expired futures", "first_requested": str(first), "last_requested": str(last),
                "created_at": datetime.now(timezone.utc).isoformat(), "rows": len(combined),
                "sessions": int(combined.timestamp.dt.date.nunique()), "contracts": len(records),
                "front_month_rule": "Monthly expiry from available expiries, verified with futures metadata; switch next calendar date",
                "metadata_lot_size_assumption": "Static per expired contract; no independent intracontract revision history",
                "tick_size_conversion": "API paise / 100 to rupees",
                "spot_calendar": str(spot_calendar.resolve()) if spot_calendar else None,
                "requests": requests, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-date", type=date.fromisoformat, required=True)
    parser.add_argument("--to-date", type=date.fromisoformat, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spot-calendar", type=Path, help="Independent spot timestamps distinguish closed sessions from missing futures responses")
    parser.add_argument("--resume", action="store_true", help="Validate and preserve previously downloaded contract files")
    args = parser.parse_args()
    collect(args.from_date, args.to_date, args.output, args.spot_calendar, args.resume)
