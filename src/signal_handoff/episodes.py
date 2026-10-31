"""Build hedging episodes for externally chosen (decision_time, instrument) entries.

The path construction is a line-for-line port of ``deep_hedging.data.build_episodes``; only the
instrument choice is injected. ``verify_equivalence`` re-derives Part 1's preserved episodes.
"""
import hashlib

import numpy as np
import pandas as pd

from deep_hedging.config import HedgeConfig
from deep_hedging.data import validate_sources

STEP = pd.Timedelta(minutes=5)


def build_chosen_episodes(options, futures, config: HedgeConfig, chosen, validated=False):
    o, f = (options, futures) if validated else validate_sources(options, futures, config.bar_minutes)
    chosen = chosen.copy()
    chosen["decision_time"] = pd.to_datetime(chosen.decision_time).dt.tz_convert("Asia/Kolkata")
    option_rows = o.set_index(["instrument_key", "timestamp"])
    future_rows = f.set_index("timestamp")
    meta = o.drop_duplicates("instrument_key").set_index("instrument_key")
    records, paths, skipped = [], [], []
    last_exit = None
    for _, pick in chosen.sort_values("decision_time").iterrows():
        decision, key = pick.decision_time, pick.instrument_key
        entry = decision + STEP
        end = entry + pd.Timedelta(minutes=config.horizon_minutes)
        kind = meta.loc[key, "option_type"]
        reason = None
        if end.date() != decision.date() or end.time() > pd.Timestamp("15:25").time() or decision.time() < pd.Timestamp("09:15").time():
            reason = "outside_regular_fill_window"
        if last_exit is not None and entry < last_exit:
            reason = "overlapping_episode_schedule"
        if reason:
            skipped.append({"decision_time": decision, "instrument_key": key, "reason": reason})
            continue
        grid = pd.date_range(entry, end, freq=f"{config.bar_minutes}min")
        chunk = []
        lot_size = int(meta.loc[key, "lot_size"])
        try:
            for index, execution in enumerate(grid):
                observation = execution - STEP
                state = option_rows.loc[(key, observation)]
                mark = option_rows.loc[(key, execution)]
                fm = future_rows.loc[execution]
                fs = future_rows.loc[observation]
                if fs.instrument_key != fm.instrument_key or fm.expiry < end.date():
                    raise ValueError("future_contract_roll_or_expiry")
                if int(mark.lot_size) != lot_size or fm.lot_size != fs.lot_size:
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
            skipped.append({"decision_time": decision, "instrument_key": key, "reason": reason})
            continue
        if any(r["observed_at"] > r["timestamp"] for r in chunk):
            raise AssertionError("Future observation leaked into execution")
        identity = f"{key}|{decision}|{config.horizon_minutes}|{config.option_lots}|{config.option_side}"
        episode = hashlib.sha256(identity.encode()).hexdigest()[:20]
        strike, expiry = float(meta.loc[key, "strike"]), meta.loc[key, "expiry"]
        records.append({"episode_id": episode, "instrument_key": key, "option_type": kind, "expiry": str(expiry),
                        "strike": strike, "decision_time": decision, "entry_time": entry, "exit_time": end,
                        "option_lot_size": lot_size, "option_lots": config.option_lots,
                        "option_quantity": config.option_side * config.option_lots * lot_size,
                        "future_instrument_key": chunk[0]["future_instrument_key"]})
        paths.extend({"episode_id": episode, **row} for row in chunk)
        last_exit = end
    return pd.DataFrame(records), pd.DataFrame(paths), pd.DataFrame(skipped)
