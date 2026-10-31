"""Frozen, hash-chained forward-validation protocol and per-session conformance checks.

The protocol is written BEFORE any forward observation exists. It freezes the allowed code/model/config hashes, the risk
rules, the evidence policy and the numeric acceptance criteria. Sessions are registered afterwards and are checked
against it; a session that does not conform is recorded but never counts as evidence.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_ACCEPTANCE = {
    "minimum_forward_sessions": 60,               # observed trading days with accepted forward events (not a power guarantee)
    "minimum_event_coverage": 0.90,               # accepted events / events expected from the planned episodes
    "minimum_event_acceptance_rate": 0.90,        # accepted / (accepted + rejected)
    "maximum_unresolved_exposure_days": 2,             # session-days ending with inventory open; each is a recorded incident
    "maximum_reconciliation_error": 1e-6,
    "maximum_integrity_violations": 0,
    "maximum_quarantine_fraction": 0.01,
    "maximum_drawdown_fraction": 0.10,
    "economic": {"claim": "hedging reduces episode loss CVaR95 versus unhedged on identical positions",
                 "candidate": "dynamic_delta", "reference": "unhedged", "require_interval_excludes_zero": True,
                 "bootstrap_repeats": 2000, "bootstrap_seed": 91},
    "cost_efficiency": {"maximum_cost_per_unit_cvar_reduction": 0.75,
                        "note": "Mean P&L given up per episode must not exceed 0.75 of the CVaR95 reduction it buys. Part 3 historical ratio was 0.39; a ceiling of 0.75 passes 98% of simulated 60-day windows at that cost and only 2% when hedging costs twice as much (docs/PROTOCOL_SENSITIVITY.md)"},
    "ml_claim": {"candidate": "adapted_ml", "reference": "adapted_no_ml", "metric": "mean episode net P&L difference",
                 "require_interval_excludes_zero": True, "applies_only_if_ml_strategy_registered": True},
}


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def default_spec(allowed, planned_episodes_per_day=6, events_per_episode=7):
    return {"schema": 1, "objective": "paper validation of frozen hedge strategies on identical forward positions",
            "evidence_policy": {"counts_only": "mode == forward, session registered after freeze, decision_at after freeze, protocol conformance",
                                "never_counts": ["synthetic_demo", "replay", "pre-freeze registrations", "pre-freeze decisions"]},
            "plan": {"planned_episodes_per_day": planned_episodes_per_day, "events_per_episode": events_per_episode,
                     "decision_grid_minutes": 5, "holding_horizon_minutes": 30,
                     "shakedown_days": 3},
            "allowed": allowed, "acceptance": DEFAULT_ACCEPTANCE,
            "decision_rule": {"accepted": "every criterion passes with at least the minimum forward sessions",
                              "rejected": "any hard criterion fails with enough sessions, or any integrity/reconciliation/risk failure at any time",
                              "pending": "fewer than the minimum forward sessions; no performance conclusion is drawn"},
            "limits": ["Configured fee/margin proxies, not broker contract notes", "Atomic option+futures fills; leg risk not modelled",
                       "Intervals condition on observed sessions and exclude regime change"]}


def freeze_protocol(spec, output, registry):
    """Write the protocol, its hash and append a hash-chained registry entry. Refuses to overwrite."""
    output, registry = Path(output), Path(registry)
    if output.exists():
        raise FileExistsError("Preserving existing frozen protocol; register a new version with a new path")
    frozen_at = datetime.now(timezone.utc).isoformat()
    previous = None
    version = 1
    if registry.exists():
        lines = [json.loads(l) for l in registry.read_text().splitlines() if l.strip()]
        verify_chain(registry)
        previous, version = sha256_bytes(canonical(lines[-1]).encode()), lines[-1]["version"] + 1
    document = {**spec, "frozen_at_utc": frozen_at, "version": version}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True))
    entry = {"version": version, "protocol": str(output), "protocol_sha256": sha256_bytes(output.read_bytes()),
             "frozen_at_utc": frozen_at, "prev_entry_sha256": previous}
    with open(registry, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    return document


def verify_chain(registry):
    previous = None
    for number, line in enumerate((l for l in Path(registry).read_text().splitlines() if l.strip()), start=1):
        entry = json.loads(line)
        if entry["prev_entry_sha256"] != previous:
            raise ValueError(f"Registry chain broken at entry {number}")
        previous = sha256_bytes(canonical(entry).encode())
    return True


def load_protocol(path, registry):
    """Load a protocol only if its bytes match the hash-chained registry entry."""
    verify_chain(registry)
    path = Path(path)
    digest = sha256_bytes(path.read_bytes())
    entries = [json.loads(l) for l in Path(registry).read_text().splitlines() if l.strip()]
    if not any(e["protocol_sha256"] == digest and Path(e["protocol"]).resolve() == path.resolve() for e in entries):
        raise ValueError("Protocol file does not match its registry entry (modified after freeze)")
    return json.loads(path.read_text())


def session_conformance(protocol, session_dir):
    """List every way a registered paper session departs from the frozen allowed values."""
    d = Path(session_dir)
    session = json.loads((d / "protocol.json").read_text())
    allowed, problems = protocol["allowed"], []
    if sha256_bytes((d / "protocol.json").read_bytes()) != (d / "protocol.sha256").read_text().strip():
        problems.append("session protocol modified after its own freeze")
    if session["mode"] != "forward":
        problems.append(f"mode {session['mode']!r} is not forward evidence")
    if pd_ts(session["created_at_utc"]) < pd_ts(protocol["frozen_at_utc"]):
        problems.append("session registered before the validation protocol was frozen")
    if session["code_sha256"] not in allowed["code_sha256"]:
        problems.append("runner code hash not in the frozen allow-list")
    if session["runtime_sha256"] != allowed["runtime_sha256"]:
        problems.append("imported execution helpers differ from the frozen runtime manifest")
    if session["strategy"] not in allowed["strategies"]:
        problems.append(f"strategy {session['strategy']!r} not allowed")
    for key in ("max_daily_loss_fraction", "max_total_drawdown_fraction", "max_quote_age_seconds", "minimum_latency_seconds",
                "max_fill_delay_seconds", "max_quote_skew_seconds"):
        if session[key] != allowed["risk_and_quote_rules"][key]:
            problems.append(f"{key} differs from the frozen rule")
    for key, value in allowed["execution_config"].items():
        if session["execution_config"].get(key) != value:
            problems.append(f"execution_config.{key} differs from the frozen configuration")
    for name in ("policy_sha256", "classifier_sha256"):
        if session.get(name) and session[name] not in allowed[name]:
            problems.append(f"{name} not in the frozen allow-list")
    return problems


def pd_ts(value):
    import pandas as pd
    return pd.Timestamp(value).tz_convert("UTC")
