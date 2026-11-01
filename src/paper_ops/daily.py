"""One trading day of paper operation: select/validate contracts, register accounts, build causal packets, run the service.

Live and replay share the same code path; only the market object and the clock differ. Replay sessions are registered in
a non-forward mode and therefore can never count as evidence. No orders exist anywhere in this chain.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from forward_paper import inspect_session
from quote_capture import normalize_snapshot
from .catalog import freeze_validated
from .market import INDEX_KEY
from .service import LockHeld, run_cycle
from .state_builder import CompletedBarStateBuilder, Unavailable

IST = "Asia/Kolkata"
BAR = pd.Timedelta(minutes=5)
SLOT_OPENS = ("09:35", "10:35", "11:35", "12:35", "13:35", "14:35")      # completed-bar availability times (decision bars 09:30...)
KINDS = ("open", "hedge", "hedge", "hedge", "hedge", "hedge", "close", "retry_close", "retry_close")
STRATEGIES = ("unhedged", "static_delta", "dynamic_delta", "band_delta", "adapted_no_ml")


def schedule(day):
    out = []
    for slot, clock in enumerate(SLOT_OPENS):
        start = pd.Timestamp(f"{day} {clock}", tz=IST)
        for k, kind in enumerate(KINDS):
            out.append((start + k * BAR, kind, slot))
    return sorted(out, key=lambda item: item[0])


def session_name(label, contract, strategy):
    return f"{label}_{contract['option']['instrument_key'].split('|')[1]}__{strategy}"


def register_sessions(base, selection, run_dir, catalog, mode, strategies=STRATEGIES, now=None):
    base = Path(base)
    sessions = {}
    for label, contract in selection.items():
        folder = base / "contracts"
        folder.mkdir(parents=True, exist_ok=True)
        pair = {"option": contract["option"], "future": contract["future"]}
        file = folder / f"{label}_{contract['option']['instrument_key'].split('|')[1]}.json"
        file.write_text(json.dumps(pair, indent=2))
        for strategy in strategies:
            target = base / "sessions" / session_name(label, contract, strategy)
            if not target.exists():
                freeze_validated(run_dir, file, catalog, target, strategy, mode, now=now)
            sessions[(label, strategy)] = target
    (base / "active.json").write_text(json.dumps({"registered_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "selection": selection,
                                                  "sessions": {f"{k[0]}|{k[1]}": str(v) for k, v in sessions.items()}}, indent=2, default=str))
    return sessions


def _bar(frame, start):
    row = frame[frame.timestamp == start]
    return None if row.empty else row.iloc[0]


def _log(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _session_state(folder):
    try:
        return inspect_session(folder)["state"] or {}
    except Exception:                                       # unreadable store: treated as flat; health will alert
        return {}


def run_day(base, day, market, selection, sessions, now_fn=None, sleep_fn=time.sleep, latency=1.0, bar_retries=3, quiet=False):
    base = Path(base)
    day = pd.Timestamp(day).date()
    live = getattr(market, "live", False)
    today = pd.Timestamp.now(tz=IST).date()
    builders = {label: CompletedBarStateBuilder({"option": c["option"]}) for label, c in selection.items()}
    future_key = next(iter(selection.values()))["future"]["instrument_key"]
    keys = [INDEX_KEY, future_key] + [c["option"]["instrument_key"] for c in selection.values()]
    quote_keys = keys[1:]
    unavailable, counters = base / "unavailable" / f"{day}.jsonl", {"events_sent": 0, "unavailable": 0, "cycles": 0, "lock_skips": 0}
    for T, kind, slot in schedule(day):
        if live:
            wait = (T + pd.Timedelta(seconds=latency) - pd.Timestamp.now(tz=IST)).total_seconds()
            if wait < -240:                                  # far behind schedule: do not fabricate a late decision
                _log(unavailable, {"time": T, "kind": kind, "reason": "runner_started_too_late_for_event"})
                counters["unavailable"] += 1
                continue
            if wait > 0:
                sleep_fn(wait)
        start = T - BAR
        bars = {}
        for attempt in range(bar_retries if live else 1):
            frames = {key: market.candles(key, str(day), today) for key in keys}
            bars = {key: _bar(frames[key], start) for key in keys}
            if all(b is not None for b in bars.values()):
                break
            if live:
                sleep_fn(2.0)
        received, payload = market.quotes(quote_keys, T)
        snap = normalize_snapshot(payload, received, max_age_seconds=5.).set_index("instrument_key") if payload.get("data") else pd.DataFrame()
        for label, contract in selection.items():
            option_key = contract["option"]["instrument_key"]
            spot_bar, future_bar, option_bar = bars.get(INDEX_KEY), bars.get(future_key), bars.get(option_key)
            state = Unavailable(start, "missing_completed_bar")
            if spot_bar is not None and future_bar is not None and option_bar is not None:
                state = builders[label].push(start, float(spot_bar.close), float(option_bar.close), float(future_bar.close))
            packet_quotes, reason = {}, None
            for asset, key in (("option", option_key), ("future", future_key)):
                if key not in snap.index or not bool(snap.loc[key, "executable_quote_valid"]):
                    reason = f"{asset}_quote_{snap.loc[key, 'quality_reason'] if key in snap.index else 'missing'}"
                    break
                row = snap.loc[key]
                packet_quotes[asset] = {"instrument_key": key, "quote_time": row.quote_time.isoformat(), "bid": float(row.bid), "ask": float(row.ask),
                                        "bid_size": float(row.bid_size), "ask_size": float(row.ask_size)}
            for strategy in sorted(s for (l, s) in sessions if l == label):
                folder = sessions[(label, strategy)]
                s = _session_state(folder)
                exposed = bool(s.get("q") or s.get("h"))
                event_type = "open" if kind == "open" else "close" if kind in ("close", "retry_close") else "hedge"
                if event_type == "open" and (exposed or s.get("halted")):
                    continue
                if event_type != "open" and not exposed:
                    continue
                if reason or (event_type != "close" and isinstance(state, Unavailable)):
                    _log(unavailable, {"time": T, "label": label, "strategy": strategy, "kind": kind, "reason": reason or state.reason})
                    counters["unavailable"] += 1
                    continue
                packet = {"event_id": f"{label}|{T.strftime('%Y%m%dT%H%M')}|{kind}", "type": event_type,
                          "decision_at": (T if isinstance(state, Unavailable) else pd.Timestamp(state["available_at"])).isoformat(),
                          "received_at": received.isoformat(), "quotes": packet_quotes}
                if event_type != "close":
                    packet["state"] = state
                inbox = base / "inbox" / folder.name
                inbox.mkdir(parents=True, exist_ok=True)
                with open(inbox / "events.jsonl", "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(packet, default=str) + "\n")
                counters["events_sent"] += 1
                try:
                    run_cycle(folder, inbox, now=T if not live else None)
                    counters["cycles"] += 1
                except LockHeld:
                    counters["lock_skips"] += 1
        if not quiet:
            print(f"{T:%H:%M} {kind:<11} sent={counters['events_sent']} unavailable={counters['unavailable']}", flush=True)
    return counters
