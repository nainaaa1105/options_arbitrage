"""Build fixed option episodes from causal observations and exact fill bars."""

import hashlib
import numpy as np
import pandas as pd

from .config import HedgeConfig

OPTION_COLUMNS = ["timestamp", "instrument_key", "expiry", "strike", "option_type", "lot_size",
                  "open", "close", "volume", "open_interest", "nifty_spot", "delta",
                  "feature_available_at", "spot_available_at"]


def market_times(values):
    ts = pd.to_datetime(values, errors="raise")
    if ts.dt.tz is None:
        raise ValueError("Market timestamps must be timezone-aware")
    return ts.dt.tz_convert("Asia/Kolkata")


def validate_sources(options, futures, bar_minutes=5):
    missing = set(OPTION_COLUMNS).difference(options.columns)
    if missing:
        raise ValueError(f"Options missing synchronized feature fields: {sorted(missing)}")
    needed = {"timestamp", "instrument_key", "open", "close", "lot_size", "tick_size", "expiry", "asset_type", "bar_minutes"}
    if needed.difference(futures.columns):
        raise ValueError(f"Futures missing fields: {sorted(needed.difference(futures.columns))}")
    o, f = options[OPTION_COLUMNS].copy(), futures.copy()
    for d in (o, f):
        d["timestamp"] = market_times(d.timestamp)
        if d.duplicated(["instrument_key", "timestamp"]).any():
            raise ValueError("Duplicate instrument/timestamp records")
        for col in ("open", "close", "lot_size"):
            values = pd.to_numeric(d[col], errors="coerce")
            if (~np.isfinite(values) | values.le(0)).any():
                raise ValueError(f"Invalid market field: {col}")
        if d.lot_size.mod(1).ne(0).any():
            raise ValueError("Lot sizes must be integral")
    o["feature_available_at"] = market_times(o.feature_available_at)
    o["spot_available_at"] = market_times(o.spot_available_at)
    if not (o.feature_available_at.eq(o.timestamp + pd.Timedelta(minutes=bar_minutes))
            & o.spot_available_at.eq(o.feature_available_at)).all():
        raise ValueError("Unverified option/spot close availability")
    if not f.asset_type.eq("FUT").all() or f.instrument_key.str.startswith("NSE_INDEX|").any():
        raise ValueError("Executable futures required; an index is not a hedge")
    if not f.bar_minutes.eq(bar_minutes).all() or f.timestamp.duplicated().any():
        raise ValueError("Need one exact front-month future per bar")
    if (~np.isfinite(f.tick_size) | f.tick_size.le(0)).any():
        raise ValueError("Invalid futures tick size in rupees")
    o["expiry"] = pd.to_datetime(o.expiry).dt.date
    f["expiry"] = pd.to_datetime(f.expiry).dt.date
    return o.sort_values(["timestamp", "instrument_key"]), f.sort_values("timestamp")


def build_episodes(options, futures, config: HedgeConfig,
                   decision_times=("09:30", "10:30", "11:30", "12:30", "13:30", "14:30")):
    o, f = validate_sources(options, futures, config.bar_minutes)
    # Include observed hedge-only sessions as zero-trade dates; do not silently
    # omit a market day just because the option collection lacks that date.
    dates = sorted(set(o.timestamp.dt.date) | set(f.timestamp.dt.date))
    by_time = {ts: group for ts, group in o.groupby("timestamp", sort=False)}
    option_rows = o.set_index(["instrument_key", "timestamp"])
    future_rows = f.set_index("timestamp")
    records, paths, skipped = [], [], []
    last_exit = None
    step = pd.Timedelta(minutes=config.bar_minutes)
    for day in dates:
        for slot, clock in enumerate(decision_times):
            decision = pd.Timestamp(f"{day} {clock}", tz="Asia/Kolkata")
            entry = decision + step
            end = entry + pd.Timedelta(minutes=config.horizon_minutes)
            reason = None
            kind = "CE" if slot % 2 == 0 else "PE"
            if end.date() != day or end.time() > pd.Timestamp("15:25").time() or decision.time() < pd.Timestamp("09:15").time():
                reason = "outside_regular_fill_window"
            if last_exit is not None and entry < last_exit:
                reason = "overlapping_episode_schedule"
            current = by_time.get(decision)
            if current is None:
                reason = reason or "no_decision_bar"
            if reason:
                skipped.append({"decision_time": decision, "option_type": kind, "reason": reason})
                continue
            expiry_time = pd.to_datetime(current.expiry.astype(str)).dt.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=15, minutes=30)
            eligible = (current.option_type.eq(kind) & np.isfinite(current.delta) & current.delta.abs().le(1)
                        & current.close.ge(1) & current.volume.gt(0) & current.open_interest.gt(0)
                        & current.nifty_spot.gt(0) & expiry_time.gt(end))
            choices = current.loc[eligible].copy()
            if choices.empty:
                skipped.append({"decision_time": decision, "option_type": kind, "reason": "no_valid_option_at_decision"})
                continue
            choices["atm_distance"] = np.abs(np.log(choices.strike / choices.nifty_spot))
            choices = choices.loc[choices.atm_distance.le(.02)].sort_values(["expiry", "atm_distance", "strike", "instrument_key"])
            if choices.empty:
                skipped.append({"decision_time": decision, "option_type": kind, "reason": "outside_atm_universe"})
                continue
            selected = choices.iloc[0]
            key = selected.instrument_key
            grid = pd.date_range(entry, end, freq=f"{config.bar_minutes}min")
            chunk = []
            try:
                for index, execution in enumerate(grid):
                    observation = execution - step
                    state = option_rows.loc[(key, observation)]
                    mark = option_rows.loc[(key, execution)]
                    fm = future_rows.loc[execution]
                    fs = future_rows.loc[observation]
                    if fs.instrument_key != fm.instrument_key or fm.expiry < end.date():
                        raise ValueError("future_contract_roll_or_expiry")
                    if int(mark.lot_size) != int(selected.lot_size) or fm.lot_size != fs.lot_size:
                        raise ValueError("intracontract_lot_size_change")
                    if not np.isfinite(state.nifty_spot) or state.nifty_spot <= 0:
                        raise ValueError("invalid_observed_spot")
                    chunk.append({"timestamp": execution, "observed_at": state.feature_available_at,
                                  "option_mark": float(mark.open), "future_mark": float(fm.open),
                                  "delta": float(state.delta), "spot_observed": float(state.nifty_spot),
                                  "future_close_observed": float(fs.close),
                                  "future_spot_delta": float(fs.close / state.nifty_spot),
                                  "future_lot_size": int(fm.lot_size), "future_tick_size": float(fm.tick_size),
                                  "future_instrument_key": fm.instrument_key, "step": index})
            except KeyError:
                reason = "missing_exact_option_or_future_bar"
            except ValueError as exc:
                reason = str(exc)
            if reason:
                skipped.append({"decision_time": decision, "option_type": kind, "instrument_key": key, "reason": reason})
                continue
            if any(r["observed_at"] > r["timestamp"] for r in chunk):
                raise AssertionError("Future observation leaked into execution")
            identity = f"{key}|{decision}|{config.horizon_minutes}|{config.option_lots}|{config.option_side}"
            episode = hashlib.sha256(identity.encode()).hexdigest()[:20]
            records.append({"episode_id": episode, "instrument_key": key, "option_type": kind,
                            "expiry": str(selected.expiry), "strike": float(selected.strike),
                            "decision_time": decision, "entry_time": entry, "exit_time": end,
                            "option_lot_size": int(selected.lot_size), "option_lots": config.option_lots,
                            "option_quantity": config.option_side * config.option_lots * int(selected.lot_size),
                            "future_instrument_key": chunk[0]["future_instrument_key"]})
            paths.extend({"episode_id": episode, **row} for row in chunk)
            last_exit = end
    return pd.DataFrame(records), pd.DataFrame(paths), pd.DataFrame(skipped), dates
