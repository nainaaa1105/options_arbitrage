"""Compare existing option candles with fresh authenticated API observations."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upstox_api import get_expired_candles


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    result = []
    for expiry in ("2025-09-23", "2026-02-24", "2026-09-15"):
        raw = pd.read_parquet(root / "data/raw/upstox" / f"expiry_{expiry}.parquet")
        # Deterministic liquid contract near median strike; selection uses no outcomes.
        choices = raw.loc[raw.option_type.eq("CE")].groupby("instrument_key").volume.sum()
        instrument = choices.idxmax()
        local = raw.loc[raw.instrument_key.eq(instrument)].copy()
        date = str(pd.to_datetime(local.timestamp).dt.date.max())
        local = local.loc[pd.to_datetime(local.timestamp).dt.date.astype(str).eq(date)]
        candles = get_expired_candles(instrument, date, date)
        fresh = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
        if fresh.empty:
            result.append({"expiry": expiry, "date": date, "api_rows": 0, "verified": False})
            continue
        fresh["timestamp"] = pd.to_datetime(fresh.timestamp)
        fresh.to_parquet(output / f"api_option_{expiry}.parquet", index=False)
        merged = local.merge(fresh, on="timestamp", suffixes=("_local", "_api"), validate="one_to_one")
        differences = {c: int((~np.isclose(merged[c+"_local"], merged[c+"_api"], atol=1e-7, rtol=0)).sum())
                       for c in ("open", "high", "low", "close", "volume", "open_interest")}
        result.append({"expiry": expiry, "date": date, "local_rows": len(local), "api_rows": len(fresh),
                       "matched_rows": len(merged), "field_mismatches": differences,
                       "verified": len(merged) == len(local) and not any(differences.values())})
        print(json.dumps(result[-1]), flush=True)
    (output / "raw_option_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
