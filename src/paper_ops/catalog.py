"""Authoritative instrument validation against a dated Upstox instrument-master snapshot.

The snapshot is supplied as a local file (JSON or .json.gz, a list of instrument records). Nothing is downloaded here.
Tick sizes in the Upstox master are in paise; they are converted to rupees and the unit is a stated assumption that the
validator cross-checks against the tick size declared in the contract file.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path

import pandas as pd

NIFTY_INDEX_KEY = "NSE_INDEX|Nifty 50"


class CatalogError(ValueError):
    pass


def _load(path):
    raw = Path(path).read_bytes()
    text = gzip.decompress(raw).decode("utf-8") if raw[:2] == b"\x1f\x8b" else raw.decode("utf-8")
    records = json.loads(text)
    if not isinstance(records, list):
        raise CatalogError("Instrument master must be a list of records")
    return records, hashlib.sha256(raw).hexdigest()


class InstrumentCatalog:
    def __init__(self, records, sha256, as_of, tick_unit="paise"):
        if tick_unit not in ("paise", "rupees"):
            raise ValueError("tick_unit must be 'paise' or 'rupees'")
        self.as_of, self.sha256, self.tick_divisor = pd.Timestamp(as_of), sha256, 100. if tick_unit == "paise" else 1.
        if self.as_of.tzinfo is None:
            raise ValueError("Catalog as_of must be timezone-aware")
        self.records = {r["instrument_key"]: r for r in records if isinstance(r, dict) and "instrument_key" in r}
        if not self.records:
            raise CatalogError("Catalog contains no keyed instruments")

    @classmethod
    def from_file(cls, path, as_of, tick_unit="paise"):
        records, digest = _load(path)
        return cls(records, digest, as_of, tick_unit)

    def _expiry(self, record):
        value = record.get("expiry")
        if value is None:
            return None
        return pd.Timestamp(value, unit="ms", tz="UTC").tz_convert("Asia/Kolkata") if isinstance(value, (int, float)) else pd.Timestamp(value)

    def validate(self, contracts, now=None, max_catalog_age_days=2.):
        """Return a provenance report or raise CatalogError listing every problem found."""
        now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
        problems = []
        age = (now - self.as_of).total_seconds() / 86400
        if age > max_catalog_age_days or age < -1:
            problems.append(f"catalog snapshot age {age:.1f} days is outside the permitted window")
        checked = {}
        for name, kind in (("option", ("CE", "PE")), ("future", ("FUT",))):
            contract = contracts[name]
            record = self.records.get(contract["instrument_key"])
            if record is None:
                problems.append(f"{name}: instrument_key not in the dated catalog")
                continue
            if record.get("segment") != "NSE_FO":
                problems.append(f"{name}: segment {record.get('segment')!r} is not NSE_FO")
            if record.get("underlying_key") != NIFTY_INDEX_KEY and record.get("underlying_symbol") != "NIFTY":
                problems.append(f"{name}: underlying is not NIFTY")
            if record.get("instrument_type") not in kind:
                problems.append(f"{name}: instrument_type {record.get('instrument_type')!r} not in {kind}")
            if name == "option":
                if record.get("instrument_type") != contract.get("option_type"):
                    problems.append("option: CE/PE disagrees with the catalog")
                if not math.isclose(float(record.get("strike_price", float('nan'))), float(contract["strike"]), abs_tol=1e-9):
                    problems.append("option: strike disagrees with the catalog")
            if int(record.get("lot_size", -1)) != int(contract["lot_size"]):
                problems.append(f"{name}: lot_size {record.get('lot_size')} != {contract['lot_size']}")
            tick = float(record.get("tick_size", float("nan"))) / self.tick_divisor
            if not math.isclose(tick, float(contract["tick_size"]), rel_tol=0, abs_tol=1e-9):
                problems.append(f"{name}: tick_size {tick} rupees != {contract['tick_size']} (check the unit assumption)")
            expiry = self._expiry(record)
            declared = pd.Timestamp(contract["expiry_at"])
            if expiry is None or expiry.tz_convert("Asia/Kolkata").date() != declared.tz_convert("Asia/Kolkata").date():
                problems.append(f"{name}: expiry date disagrees with the catalog")
            elif declared.tz_convert("UTC") <= now:
                problems.append(f"{name}: contract already expired")
            checked[name] = {"instrument_key": contract["instrument_key"], "trading_symbol": record.get("trading_symbol"),
                             "expiry_ist": str(expiry) if expiry is not None else None}
        if not problems and pd.Timestamp(contracts["future"]["expiry_at"]) < pd.Timestamp(contracts["option"]["expiry_at"]):
            problems.append("hedge future expires before the option")
        if problems:
            raise CatalogError("; ".join(problems))
        return {"catalog_sha256": self.sha256, "catalog_as_of": str(self.as_of), "validated_at_utc": now.isoformat(),
                "contracts": checked, "tick_unit_assumption": "paise" if self.tick_divisor == 100. else "rupees"}


def freeze_validated(run_dir, contracts_file, catalog, output, strategy="dynamic_delta", mode="forward", now=None):
    """Validate against the catalog first, then register the session; provenance is stored beside the protocol."""
    from forward_paper import freeze
    contracts = json.loads(Path(contracts_file).read_text())
    report = catalog.validate(contracts, now=now)
    protocol = freeze(Path(run_dir), Path(contracts_file), Path(output), strategy, mode)
    (Path(output) / "catalog_validation.json").write_text(json.dumps(
        {**report, "protocol_sha256": hashlib.sha256((Path(output) / "protocol.json").read_bytes()).hexdigest()}, indent=2))
    return protocol, report
