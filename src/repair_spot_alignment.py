"""Create a separate repaired master from authentic NIFTY spot candles.

Supply matching five-minute spot bars, or one-minute bars offset so that their
close availability matches the five-minute option close.

This does not reconstruct missing minute data or overwrite existing datasets.
After repair, all pricing, surface, label and ML datasets must be rebuilt from
the repaired master before claiming financially synchronized model results.
"""

import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from candle_alignment import align_option_spot_candles


def download_missing_spot(master_path, spot_directory, transport="project"):
    """Resume a dedicated authentic minute-candle cache without replacing files."""
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    if not token:
        raise RuntimeError("UPSTOX_ACCESS_TOKEN is not configured")
    dates = sorted(pd.to_datetime(pd.read_parquet(master_path, columns=["timestamp"]).timestamp).dt.date.unique())
    spot_directory.mkdir(parents=True, exist_ok=True)
    tls_context = ssl.create_default_context()
    for date in dates:
        destination = spot_directory / f"{date}.parquet"
        if destination.exists():
            continue
        url = f"https://api.upstox.com/v3/historical-candle/NSE_INDEX%7CNifty%2050/minutes/1/{date}/{date}"
        if transport == "project":
            # Match the authenticated client already used by collection. Do not
            # change browser signatures or fall back after an access denial.
            from upstox_api import make_request
            payload = make_request(url)
        elif transport == "stdlib":
            request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, context=tls_context, timeout=30) as response:
                    payload = json.load(response)
            except urllib.error.HTTPError as error:
                raise RuntimeError(f"Upstox returned HTTP {error.code} while fetching {date}; no data was replaced") from None
        else:
            raise ValueError("transport must be project or stdlib")
        candles = payload.get("data", {}).get("candles", [])
        if not candles:
            raise RuntimeError(f"Upstox returned no minute candles for {date}")
        frame = pd.DataFrame(candles, columns=[
            "timestamp", "open", "high", "low", "close", "volume", "open_interest"
        ])
        frame["timestamp"] = pd.to_datetime(frame.timestamp)
        frame["bar_minutes"] = 1
        if not (frame.timestamp.dt.date == date).all():
            raise ValueError(f"Unexpected dates in minute response for {date}")
        frame.sort_values("timestamp").to_parquet(destination, index=False)
        print(f"Cached authentic minute candles for {date}", flush=True)
        time.sleep(.3)


def repair(master_path, spot_path, output_path, spot_minutes=1):
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    master = pd.read_parquet(master_path)
    files = sorted(spot_path.glob("*.parquet")) if spot_path.is_dir() else [spot_path]
    if not files:
        raise FileNotFoundError(f"No authentic spot candles found under {spot_path}")
    spot = pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
    spot["timestamp"] = pd.to_datetime(spot.timestamp)
    if spot.duplicated("timestamp").any():
        inconsistent = spot.groupby("timestamp").close.nunique().gt(1).any()
        if inconsistent:
            raise ValueError("Conflicting spot closes for a timestamp")
        spot = spot.drop_duplicates("timestamp")
    repaired = align_option_spot_candles(master, spot, spot_minutes=spot_minutes)
    missing = int(repaired.nifty_spot.isna().sum())
    if missing:
        raise ValueError(f"Missing synchronized spot closes for {missing:,} option rows; supply complete spot data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_parquet(output_path, index=False)
    print(f"Saved {len(repaired):,} synchronized master rows to {output_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=root / "data/processed/master_options.parquet")
    parser.add_argument("--spot", type=Path, required=True, help="Spot candle Parquet file or directory; columns timestamp, close; declare duration with --spot-minutes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download-missing", action="store_true", help="Fetch missing authentic minute candles into --spot directory before repair")
    parser.add_argument("--spot-minutes", type=int, choices=[1, 5], default=1,
                        help="Actual source candle duration; five-minute spot exports can align directly with five-minute options")
    parser.add_argument("--transport", choices=["project", "stdlib"], default="project",
                        help="Default uses the same authenticated requests client as project collection")
    args = parser.parse_args()
    if args.download_missing:
        if args.spot_minutes != 1:
            parser.error("--download-missing currently fetches authentic one-minute candles only")
        if args.spot.exists() and not args.spot.is_dir():
            parser.error("--download-missing requires --spot to name a directory")
        download_missing_spot(args.master, args.spot, args.transport)
    repair(args.master, args.spot, args.output, args.spot_minutes)
