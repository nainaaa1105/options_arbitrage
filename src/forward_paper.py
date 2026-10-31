"""Persistent, quote-based paper execution. This module has no broker order API."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import sqlite3
import joblib

import numpy as np
import pandas as pd

from deep_hedging.accounting import FuturesPosition
from deep_hedging.baselines import target_lots
from deep_hedging.config import HedgeConfig
from deep_hedging.costs import transaction_fees, execution_price
from deep_hedging.historical_policy import HistoricalScalarPolicy, load_historical_policy


def stamp(value):
    ts = pd.Timestamp(value)
    if ts.tzinfo is None or pd.isna(ts):
        raise ValueError("Timezone-aware timestamps required")
    return ts.tz_convert("UTC")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def runtime_manifest():
    """Freeze imported numerical helpers as well as the entry-point file."""
    root = Path(__file__).resolve().parent
    files = [Path(__file__).resolve(), root/"research_targets.py", *sorted((root/"deep_hedging").glob("*.py"))]
    return {str(p.relative_to(root)): digest(p) for p in files}


def inspect_session(directory):
    """Read-only recovery inspection; never bypasses execution integrity gates."""
    directory=Path(directory)
    protocol=json.loads((directory/'protocol.json').read_text())
    db=sqlite3.connect((directory/'paper.sqlite').resolve().as_uri()+'?mode=ro',uri=True)
    try:
        row=db.execute('SELECT payload FROM state WHERE id=1').fetchone()
        counts=dict(db.execute('SELECT status,COUNT(*) FROM events GROUP BY status').fetchall())
    finally:
        db.close()
    return {'mode':protocol['mode'],'strategy':protocol['strategy'],'state':json.loads(row[0]) if row else None,
            'event_counts':counts,'read_only':True,'execution_authorized_by_inspection':False,
            'protocol_hash_matches':digest(directory/'protocol.json')==(directory/'protocol.sha256').read_text().strip(),
            'runner_matches':digest(__file__)==protocol['code_sha256']}


def freeze(run_dir, contracts_file, output, strategy="dynamic_delta", mode="forward"):
    if mode not in ("forward", "replay", "synthetic_demo"):
        raise ValueError("Unknown evidence mode")
    if strategy not in ("unhedged", "static_delta", "dynamic_delta", "band_delta", "adapted_ml", "adapted_no_ml"):
        raise ValueError("Unsupported paper strategy")
    contracts = json.loads(contracts_file.read_text())
    for name, kind in (("option", "OPT"), ("future", "FUT")):
        d = contracts[name]
        if d["asset_type"] != kind or not d["instrument_key"].startswith("NSE_FO|"):
            raise ValueError("Tradable NSE option/futures instrument metadata required")
        if int(d["lot_size"]) != d["lot_size"] or d["lot_size"] < 1 or not math.isfinite(d["tick_size"]) or d["tick_size"] <= 0:
            raise ValueError("Invalid contract lot/tick metadata")
        stamp(d["expiry_at"])
    if contracts["option"]["option_type"] not in ("CE", "PE") or contracts["option"]["strike"] <= 0:
        raise ValueError("Invalid option metadata")
    output.mkdir(parents=True, exist_ok=False)
    source = json.loads((run_dir/"experiment_plan.json").read_text())
    config = HedgeConfig(**source["config"])
    config = replace(config, option_tick_size=contracts["option"]["tick_size"])
    if config.option_side != 1 or config.funding_rate != 0:
        raise ValueError("Paper protocol supports premium-funded long options and zero funding only")
    selected = None
    classifier_hash = None
    if strategy.startswith("adapted_"):
        manifest = json.loads((run_dir/"frozen_before_historical_test.json").read_text())
        rel = next(name for name in manifest["checkpoint_sha256"] if Path(name).parts[0] == strategy)
        file = run_dir/rel
        if digest(file) != manifest["checkpoint_sha256"][rel]:
            raise ValueError("Historical checkpoint hash mismatch")
        shutil.copy2(file, output/"policy.pt")
        selected = digest(output/"policy.pt")
    if strategy == "adapted_ml":
        folds = pd.read_csv(run_dir/"classifier_folds/fold_audit.csv")
        latest = folds.sort_values("prediction_start").iloc[-1]
        classifier = run_dir/f"classifier_folds/fold_{int(latest['fold']):02d}.joblib"
        if digest(classifier) != latest.model_sha256:
            raise ValueError("Classifier hash mismatch")
        shutil.copy2(classifier, output/"classifier.joblib")
        classifier_hash = digest(output/"classifier.joblib")
    now = datetime.now(timezone.utc).isoformat()
    if mode == "forward" and any(stamp(contracts[name]["expiry_at"]) <= stamp(now) for name in ("option", "future")):
        raise ValueError("Forward contracts are expired")
    protocol = {"created_at_utc": now, "mode": mode, "strategy": strategy, "contracts": contracts,
                "execution_config": asdict(config), "policy_sha256": selected, "classifier_sha256": classifier_hash,
                "source_run": str(run_dir.resolve()), "source_plan_sha256": digest(run_dir/"experiment_plan.json"),
                "historical_end": source["test_end"], "max_quote_age_seconds": 3., "minimum_latency_seconds": .2,
                "max_fill_delay_seconds": 10., "max_quote_skew_seconds": 1., "max_daily_loss_fraction": .02,
                "max_total_drawdown_fraction": .10, "minimum_forward_sessions": 60,
                "execution": "Observed best ask for buys / best bid for sells plus adverse slippage and tick rounding; no partial fills",
                "model_input": "Completed-bar state; optional probability must have model provenance and domain flag; missing ML is explicit",
                "fees_and_margin": "Inherited configured research proxies, not a broker/SPAN certification",
                "code_sha256": digest(__file__), "runtime_sha256": runtime_manifest()}
    (output/"protocol.json").write_text(json.dumps(protocol, indent=2))
    (output/"protocol.sha256").write_text(digest(output/"protocol.json"))
    return protocol


class PaperSession:
    def __init__(self, directory):
        self.directory = Path(directory)
        if digest(self.directory/"protocol.json") != (self.directory/"protocol.sha256").read_text().strip():
            raise ValueError("Frozen protocol changed")
        self.p = json.loads((self.directory/"protocol.json").read_text())
        if digest(__file__) != self.p["code_sha256"]:
            raise ValueError("Runner changed since protocol freeze; register a new session")
        if self.p.get("runtime_sha256") != runtime_manifest():
            raise ValueError("Imported execution helpers changed since protocol freeze")
        self.c = HedgeConfig(**self.p["execution_config"])
        self.model = None
        self.classifier = None
        if self.p["policy_sha256"]:
            if digest(self.directory/"policy.pt") != self.p["policy_sha256"]:
                raise ValueError("Frozen policy changed")
            self.model, saved = load_historical_policy(self.directory/"policy.pt")
            for key in ("horizon_minutes", "option_lots", "max_future_lots", "bar_minutes"):
                if saved["execution_config"][key] != getattr(self.c, key):
                    raise ValueError("Policy and execution configuration mismatch")
        if self.p["classifier_sha256"]:
            if digest(self.directory/"classifier.joblib") != self.p["classifier_sha256"]:
                raise ValueError("Frozen classifier changed")
            self.classifier = joblib.load(self.directory/"classifier.joblib")
        self.db = sqlite3.connect(self.directory/"paper.sqlite")
        self.db.execute("CREATE TABLE IF NOT EXISTS events (event_id TEXT PRIMARY KEY, payload TEXT, status TEXT, detail TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS fills (event_id TEXT, payload TEXT)")
        self.db.execute("CREATE TABLE IF NOT EXISTS equity (event_id TEXT PRIMARY KEY, payload TEXT)")
        self.db.commit()

    def close(self):
        self.db.close()

    def state(self):
        row = self.db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        return json.loads(row[0]) if row else {"cash": self.c.initial_capital, "q": 0, "h": 0, "average": 0.,
            "fees": 0., "peak_equity": self.c.initial_capital, "day_start_equity": self.c.initial_capital,
            "day": None, "last_received_at": None, "entry_at": None, "episode_start_cash": self.c.initial_capital,
            "last_decision_at": None, "halted": False}

    def quote(self, asset, event, quantity=0):
        d = event["quotes"][asset]
        meta = self.p["contracts"][asset]
        if d["instrument_key"] != meta["instrument_key"]:
            raise ValueError("Wrong instrument")
        receipt, qt, decision = stamp(event["received_at"]), stamp(d["quote_time"]), stamp(event["decision_at"])
        age, delay = (receipt-qt).total_seconds(), (qt-decision).total_seconds()
        if not 0 <= age <= self.p["max_quote_age_seconds"]:
            raise ValueError("Stale or future quote")
        if not self.p["minimum_latency_seconds"] <= delay <= self.p["max_fill_delay_seconds"]:
            raise ValueError("Quote not after decision latency or excessive fill delay")
        if stamp(meta["expiry_at"]) <= receipt:
            raise ValueError("Expired instrument")
        for name in ("bid", "ask", "bid_size", "ask_size"):
            if not math.isfinite(d[name]) or d[name] <= 0:
                raise ValueError("Missing two-sided depth")
        if d["bid"] > d["ask"]:
            raise ValueError("Crossed book")
        if quantity and abs(quantity) > d["ask_size" if quantity > 0 else "bid_size"]:
            raise ValueError("Insufficient displayed depth; no fabricated partial fill")
        return d

    def process(self, event):
        payload = canonical(event)
        event_id = event["event_id"]
        # Serialize the read/modify/write cycle, not just its final INSERTs.
        self.db.execute("BEGIN IMMEDIATE")
        existing = self.db.execute("SELECT payload,status,detail FROM events WHERE event_id=?", (event_id,)).fetchone()
        if existing:
            if existing[0] != payload:
                self.db.rollback()
                raise ValueError("Conflicting duplicate event ID")
            self.db.commit()
            return {"status": existing[1], "detail": existing[2], "duplicate": True}
        original = self.state()
        self._liquidation_requested = self._risk_detected = False
        try:
            state, fills, equity = self.apply(event, original.copy())
            with self.db:
                self.db.execute("INSERT INTO state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (canonical(state),))
                self.db.executemany("INSERT INTO fills VALUES (?,?)", [(event_id, canonical(fill)) for fill in fills])
                self.db.execute("INSERT INTO equity VALUES (?,?)", (event_id, canonical(equity)))
                self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (event_id, payload, "accepted", ""))
            return {"status": "accepted", "equity": equity}
        except (ValueError, KeyError, TypeError, ArithmeticError) as error:
            detail = str(error)
            with self.db:
                if self._liquidation_requested or self._risk_detected:
                    # A failed fill must roll back accounting, not erase the
                    # risk decision. Recovery packets may only liquidate.
                    original["pending_liquidation"] = bool(original["q"]) and self._liquidation_requested
                    original["halted"] = original["halted"] or self._risk_detected
                    original["liquidation_reason"] = detail
                    self.db.execute("INSERT INTO state VALUES (1,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload", (canonical(original),))
                self.db.execute("INSERT INTO events VALUES (?,?,?,?)", (event_id, payload, "rejected", detail))
            return {"status": "rejected", "detail": detail}

    def apply(self, event, s):
        kind = event["type"]
        if kind not in ("open", "hedge", "close"):
            raise ValueError("Unknown event type")
        receipt, decision = stamp(event["received_at"]), stamp(event["decision_at"])
        if self.p["mode"] == "forward":
            now = pd.Timestamp.now(tz="UTC")
            if decision < stamp(self.p["created_at_utc"]) or decision <= stamp(self.p["historical_end"]) or receipt > now:
                raise ValueError("Not a newly observed forward event")
        local = receipt.tz_convert("Asia/Kolkata")
        if local.weekday() >= 5 or not "09:15" <= local.strftime("%H:%M") < "15:30":
            raise ValueError("Outside regular weekday session; special sessions require a new protocol")
        if s["last_received_at"] and receipt <= stamp(s["last_received_at"]):
            raise ValueError("Out-of-order event receipt")
        if decision > receipt or (s["last_decision_at"] and decision <= stamp(s["last_decision_at"])):
            raise ValueError("Invalid or repeated decision timestamp")
        if s["entry_at"] and stamp(s["entry_at"]).tz_convert("Asia/Kolkata").date() != local.date():
            raise ValueError("Unresolved overnight position; explicit reconciliation required")
        option, future = self.quote("option", event), self.quote("future", event)
        if abs((stamp(event["quotes"]["option"]["quote_time"])-stamp(event["quotes"]["future"]["quote_time"])).total_seconds()) > self.p["max_quote_skew_seconds"]:
            raise ValueError("Unsynchronized option/futures quotes")
        om, fm = (option["bid"]+option["ask"])/2, (future["bid"]+future["ask"])/2
        book = FuturesPosition(s["h"], s["average"])
        gross = s.get("gross_pnl", 0.)+s["q"]*(om-s.get("last_option_mid", om))+s["h"]*(fm-s.get("last_future_mid", fm))
        equity_before = s["cash"]+s["q"]*om+book.unrealized(fm)
        day = str(local.date())
        if day != s["day"]:
            s["day"], s["day_start_equity"] = day, equity_before
        s["peak_equity"] = max(s["peak_equity"], equity_before)
        breach = (equity_before <= s["day_start_equity"]*(1-self.p["max_daily_loss_fraction"])
                  or equity_before <= s["peak_equity"]*(1-self.p["max_total_drawdown_fraction"])
                  or s["cash"]+min(book.unrealized(fm), 0.)-abs(s["h"])*fm*self.c.future_margin_fraction < max(equity_before, 0)*(1-self.c.capital_fraction))
        s["halted"] = s["halted"] or breach
        self._risk_detected = breach
        self._liquidation_requested = breach and bool(s["q"])
        terminal = bool(s["entry_at"] and decision >= stamp(s["entry_at"])+pd.Timedelta(minutes=self.c.horizon_minutes))
        if kind == "open" and (s["q"] or s["halted"]):
            raise ValueError("Position already open or risk stop active")
        if kind != "open" and not s["q"]:
            raise ValueError("No open option episode")
        if kind == "open" and (local+pd.Timedelta(minutes=self.c.horizon_minutes)).strftime("%H:%M") >= "15:30":
            raise ValueError("Insufficient regular session for holding horizon")
        close = kind == "close" or terminal or breach or s.get("pending_liquidation", False)
        self._liquidation_requested, self._risk_detected = close, breach
        if kind == "hedge" and not close:
            elapsed = (decision-stamp(s["last_decision_at"])).total_seconds()
            if elapsed <= 0 or elapsed % (self.c.bar_minutes*60) != 0:
                raise ValueError("Hedge decisions require five-minute grid boundaries")
            s["missed_boundaries"] = s.get("missed_boundaries", 0)+int(elapsed//(self.c.bar_minutes*60))-1
        q = 0 if close else self.c.option_lots*self.p["contracts"]["option"]["lot_size"]
        initial = kind == "open"
        desired = 0
        if not close:
            state = event["state"]
            available = stamp(state["available_at"])
            if available > decision or not 0 <= (decision-available).total_seconds() <= 1:
                raise ValueError("Stale or future completed-bar state")
            if stamp(state["bar_start"])+pd.Timedelta(minutes=5) != available:
                raise ValueError("State is not a completed five-minute bar")
            if available.second or available.microsecond or available.minute % self.c.bar_minutes:
                raise ValueError("Completed state must lie on the five-minute session grid")
            for name in ("spot_observed", "option_close_observed", "future_close_observed", "volatility", "remaining_time"):
                if not math.isfinite(state[name]) or state[name] <= 0:
                    raise ValueError("Invalid observed market state")
            if not math.isfinite(state["delta"]) or abs(state["delta"]) > 1 or not math.isfinite(state["recent_return"]):
                raise ValueError("Invalid observed delta/return")
            if (self.p["contracts"]["option"]["option_type"] == "CE" and state["delta"] < 0) or (self.p["contracts"]["option"]["option_type"] == "PE" and state["delta"] > 0):
                raise ValueError("Option type and delta sign disagree")
            expected_tte = (stamp(self.p["contracts"]["option"]["expiry_at"])-available).total_seconds()/(365*86400)
            if abs(state["remaining_time"]-expected_tte) > 1/86400/365:
                raise ValueError("Maturity is inconsistent with ACT/365 contract expiry")
            ml, probability = False, .5
            from research_targets import session_horizon_eligible
            domain_time = bool(session_horizon_eligible(pd.Series([available-pd.Timedelta(minutes=5)]), self.c.horizon_minutes).iloc[0])
            if self.classifier and domain_time and state.get("signal_eligible") and abs(state.get("reference_price_deviation", 0)) >= 1 and state.get("classifier_features"):
                if stamp(self.classifier["training_label_end_max"]) >= available:
                    raise ValueError("Classifier was not fitted before the observation")
                columns = self.classifier["feature_columns"]
                frame = pd.DataFrame([{name: state["classifier_features"][name] for name in columns}], columns=columns).astype(float)
                if np.isinf(frame.to_numpy()).any():
                    raise ValueError("Non-finite classifier inputs")
                probability = float(self.classifier["model"].predict_proba(frame)[0, 1])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Invalid classifier probability")
                ml = True
            quantity = q
            if initial:
                s["episode_start_cash"] = s["cash"]
            fs = state["future_close_observed"]/state["spot_observed"]
            observed = {**state, "option_quantity": quantity, "future_lot_size": self.p["contracts"]["future"]["lot_size"],
                        "future_spot_delta": fs, "strike": self.p["contracts"]["option"]["strike"],
                        "is_call": self.p["contracts"]["option"]["option_type"] == "CE", "timestamp": decision,
                        "cash": s["cash"], "future_mark": state["future_close_observed"],
                        "future_unrealized": book.unrealized(state["future_close_observed"]),
                        "observed_delta_used": state["delta"], "pricing_state_valid": True,
                        "probability": probability if ml else .5, "ml_available": bool(ml),
                        "remaining_steps": self.c.horizon_minutes//5 if initial else max(1, int((stamp(s["entry_at"])+pd.Timedelta(minutes=self.c.horizon_minutes)-decision).total_seconds()//300))}
            current = s["h"]//observed["future_lot_size"]
            if self.model is None:
                desired = target_lots(self.p["strategy"], observed, current, initial, self.c)[0]
            else:
                desired = HistoricalScalarPolicy(self.model, s["episode_start_cash"])(observed, current, initial, self.c)[0]
        new_h = desired*self.p["contracts"]["future"]["lot_size"]
        fills = []
        for asset, change in (("option", q-s["q"]), ("future", new_h-s["h"])):
            if not change:
                continue
            quote = self.quote(asset, event, change)
            mark = quote["ask"] if change > 0 else quote["bid"]
            fill = execution_price(mark, change, self.p["contracts"][asset]["tick_size"], 0., getattr(self.c, asset+"_slippage"))
            fees = transaction_fees(asset, change, fill, receipt, self.c)
            if asset == "option":
                s["cash"] -= change*fill+fees["total"]
            else:
                s["cash"] += book.trade(change, fill)-fees["total"]
            s["fees"] += fees["total"]
            s["impact"] = s.get("impact", 0.)+change*(fill-(om if asset == "option" else fm))
            fills.append({"asset": asset, "quantity_change": change, "fill": fill, "fees": fees["total"],
                          "quote_time": quote["quote_time"], "received_at": event["received_at"], "evidence_mode": self.p["mode"]})
        s.update(q=q, h=book.quantity, average=book.average_fill)
        equity = s["cash"]+q*om+book.unrealized(fm)
        calculated = self.c.initial_capital+gross-s["fees"]-s.get("impact", 0.)
        if not np.isclose(equity, calculated, atol=1e-6, rtol=1e-10):
            raise ArithmeticError("Paper cash and marked P&L do not reconcile")
        s.update(gross_pnl=gross, last_option_mid=om, last_future_mid=fm)
        margin = abs(s["h"])*fm*self.c.future_margin_fraction
        free = s["cash"]+min(book.unrealized(fm), 0.)-margin
        if not close and (equity <= 0 or free < equity*(1-self.c.capital_fraction)):
            raise ValueError("Insufficient capital/margin; entire simulated order group rejected")
        s["entry_at"] = event["decision_at"] if initial else s["entry_at"]
        if close:
            s["entry_at"] = None
            s["pending_liquidation"] = False
        s.update(last_received_at=event["received_at"], last_decision_at=event["decision_at"])
        mark = {"received_at": event["received_at"], "equity": equity, "cash": s["cash"], "net_pnl": equity-self.c.initial_capital,
                "option_quantity": q, "future_quantity": s["h"], "margin_proxy": margin, "free_cash": free,
                "fees": s["fees"], "halted": s["halted"], "evidence_mode": self.p["mode"], "forced_close": close and kind != "close"}
        mark.update(gross_pnl=gross, execution_impact=s.get("impact", 0.), reconciliation_error=equity-calculated)
        return s, fills, mark

    def export(self):
        for table in ("events", "fills", "equity"):
            frame = pd.read_sql_query(f"SELECT * FROM {table}", self.db)
            frame.to_csv(self.directory/f"{table}.csv", index=False)
        frames = {}
        for table, filename in (("fills", "fill_ledger"), ("equity", "equity_marks")):
            rows = self.db.execute(f"SELECT event_id,payload FROM {table}").fetchall()
            frames[table] = pd.DataFrame([{"event_id": key, **json.loads(value)} for key, value in rows])
            frames[table].to_parquet(self.directory/f"{filename}.parquet", index=False)
        observed_days = 0
        last_equity = self.c.initial_capital
        if not frames["equity"].empty:
            marks = frames["equity"]
            observed_days = int(pd.to_datetime(marks.received_at, utc=True).dt.tz_convert("Asia/Kolkata").dt.date.nunique())
            last_equity = float(marks.equity.iloc[-1])
        events = pd.read_sql_query("SELECT status,detail FROM events", self.db)
        evidence = {"mode": self.p["mode"], "strategy": self.p["strategy"], "events": len(events),
                    "accepted_events": int(events.status.eq("accepted").sum()), "rejected_events": int(events.status.eq("rejected").sum()),
                    "state": self.state(), "forward_performance_established": False,
                    "observed_sessions": observed_days, "required_forward_sessions": self.p["minimum_forward_sessions"],
                    "marked_net_pnl": last_equity-self.c.initial_capital,
                    "minimum_forward_window_observed": self.p["mode"] == "forward" and observed_days >= self.p["minimum_forward_sessions"],
                    "risk_note": "Rejected close/hedge leaves inventory open; obtain valid quotes and reconcile before resuming"}
        (self.directory/"status.json").write_text(json.dumps(evidence, indent=2))
        return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--run-dir", type=Path, required=True)
    f.add_argument("--contracts", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--strategy", default="dynamic_delta")
    f.add_argument("--mode", choices=["forward", "replay", "synthetic_demo"], default="forward")
    r = sub.add_parser("consume")
    r.add_argument("--session", type=Path, required=True)
    r.add_argument("--events", type=Path, required=True)
    inspect=sub.add_parser('inspect',help='Read-only inspection of old frozen sessions without executing changed code')
    inspect.add_argument('--session',type=Path,required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.run_dir, args.contracts, args.output, args.strategy, args.mode)
    elif args.command=='inspect':
        print(json.dumps(inspect_session(args.session),indent=2))
    else:
        runner = PaperSession(args.session)
        try:
            for line in args.events.read_text().splitlines():
                if line.strip():
                    event = json.loads(line)
                    print(json.dumps({"event_id": event["event_id"], **runner.process(event)}))
            runner.export()
        finally:
            runner.close()


if __name__ == "__main__":
    main()
