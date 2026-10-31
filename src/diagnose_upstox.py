"""Read-only authenticated market-data diagnostics; never saves credentials."""

import argparse
import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import dotenv_values, load_dotenv
from http_transport import verified_session


ROOT = Path(__file__).resolve().parents[1]


def classify_response(status, payload, content_type=""):
    errors = payload.get("errors", []) if isinstance(payload, dict) else []
    codes = [str(e.get("errorCode", e.get("code", ""))) for e in errors if isinstance(e, dict)]
    if status == 200:
        return "ok" if isinstance(payload, dict) and payload.get("status") == "success" else "invalid_success_payload"
    if status == 401:
        return "authentication_rejected"
    if status == 403:
        return "provider_or_edge_access_denied"
    if status == 429:
        return "rate_limited"
    if status >= 500:
        return "provider_server_error"
    return "api_request_rejected" if codes or "json" in content_type else "non_json_http_error"


def candle_quality(candles, minutes=None):
    """Validate actual response contents instead of treating HTTP 200 as data."""
    if not candles:
        return {"rows": 0, "usable": False, "reason": "empty_candles"}
    if any(not isinstance(row, list) or len(row) != 7 for row in candles):
        return {"rows": len(candles), "usable": False, "reason": "unexpected_candle_schema"}
    d = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
    ts = pd.to_datetime(d.timestamp, errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata")
    prices = d[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    import numpy as np
    finite = np.isfinite(prices).all(axis=1)
    invalid = (~finite | prices.le(0).any(axis=1) | prices.high.lt(prices[["open", "close", "low"]].max(axis=1))
               | prices.low.gt(prices[["open", "close", "high"]].min(axis=1)))
    volume = pd.to_numeric(d.volume, errors="coerce")
    oi = pd.to_numeric(d.oi, errors="coerce")
    negatives = int((volume.lt(0) | oi.lt(0)).sum())
    invalid_activity = int((~np.isfinite(volume) | ~np.isfinite(oi)).sum())
    result = {"rows": len(d), "invalid_timestamps": int(ts.isna().sum()),
              "duplicate_timestamps": int(ts.duplicated().sum()), "invalid_ohlc": int(invalid.sum()),
              "negative_volume_or_oi": negatives,
              "invalid_volume_or_oi": invalid_activity,
              "first": str(ts.min()), "last": str(ts.max()),
              "sessions": int(ts.dt.date.nunique())}
    if minutes:
        offset = ts.dt.hour * 60 + ts.dt.minute - 555
        result["off_grid_rows"] = int((offset.mod(minutes).ne(0)).sum())
    result["usable"] = all(result[k] == 0 for k in (
        "invalid_timestamps", "duplicate_timestamps", "invalid_ohlc", "negative_volume_or_oi", "invalid_volume_or_oi"))
    if minutes:
        result["usable"] &= result["off_grid_rows"] == 0
    return result


def token_metadata(token):
    """Unverified JWT expiry metadata is diagnostic only, never authorization."""
    result = {"present": bool(token), "outer_whitespace": bool(token and token != token.strip())}
    if not token:
        return result
    try:
        part = token.split(".")[1]
        claims = json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
        expiry = float(claims["exp"])
        result.update({"unverified_expiry_utc": datetime.fromtimestamp(expiry, timezone.utc).isoformat(),
                       "unverified_expired": expiry <= datetime.now(timezone.utc).timestamp()})
    except (ValueError, KeyError, IndexError, TypeError, OverflowError):
        result["unverified_expiry_available"] = False
    return result


def run(output, date="2026-09-15"):
    output.mkdir(parents=True, exist_ok=False)
    dotenv_token = dotenv_values(ROOT / ".env").get("UPSTOX_ACCESS_TOKEN")
    inherited = os.getenv("UPSTOX_ACCESS_TOKEN")
    load_dotenv(ROOT / ".env")
    token = os.getenv("UPSTOX_ACCESS_TOKEN")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "token": token_metadata(token),
              "env_file_present": (ROOT / ".env").exists(),
              "inherited_token_overrides_different_dotenv": bool(inherited and dotenv_token and inherited != dotenv_token),
              "tls_verification": True, "requests": []}
    if token:
        session = verified_session()
        key = quote("NSE_INDEX|Nifty 50", safe="")
        endpoints = [
            ("spot_v3_5minute", f"https://api.upstox.com/v3/historical-candle/{key}/minutes/5/{date}/{date}", 5),
            ("spot_v2_1minute", f"https://api.upstox.com/v2/historical-candle/{key}/1minute/{date}/{date}", 1),
            ("expired_expiries", f"https://api.upstox.com/v2/expired-instruments/expiries?instrument_key={key}", None),
        ]
        for name, url, minutes in endpoints:
            row = {"endpoint": name}
            try:
                response = session.get(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"}, timeout=30)
                try:
                    payload = response.json()
                except ValueError:
                    payload = None
                row.update({"http_status": response.status_code,
                            "classification": classify_response(response.status_code, payload, response.headers.get("Content-Type", ""))})
                if isinstance(payload, dict):
                    row["error_codes"] = [str(e.get("errorCode", "")) for e in payload.get("errors", []) if isinstance(e, dict)]
                if row["classification"] == "ok":
                    if minutes:
                        candles = payload.get("data", {}).get("candles", [])
                        row["quality"] = candle_quality(candles, minutes)
                        if row["quality"]["usable"]:
                            frame = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "open_interest"])
                            frame["timestamp"] = pd.to_datetime(frame.timestamp)
                            frame["bar_minutes"] = minutes
                            frame.sort_values("timestamp").to_parquet(output / f"{name}.parquet", index=False)
                    else:
                        values = payload.get("data", [])
                        row["returned_expiries"] = len(values) if isinstance(values, list) else None
                report["requests"].append(row)
                print(json.dumps(row), flush=True)
                # A denied credential/client is not worked around with another endpoint.
                if response.status_code in (401, 403, 429):
                    break
            except requests.exceptions.SSLError:
                row["classification"] = "tls_certificate_verification_failed"
                report["requests"].append(row)
                break
            except requests.RequestException:
                row["classification"] = "network_or_timeout_error"
                report["requests"].append(row)
                break
    report["historical_data_received"] = any(r.get("quality", {}).get("usable") for r in report["requests"])
    (output / "api_diagnostic.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="2026-09-15")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.date)
