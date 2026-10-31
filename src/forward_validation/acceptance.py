"""Predeclared acceptance evaluation. Statuses: pass / fail / pending / not_applicable; overall: accepted / rejected / pending.

Shakedown days (declared in the protocol before the freeze) are the first N distinct forward days. They still count toward the
hard gates (integrity, reconciliation, drawdown, unresolved exposure) but are excluded from the day count, coverage,
acceptance rate, quarantine fraction and every economic comparison. Episodes are paired on identical positions: same option
contract and same open decision time.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .observations import open_store


def loss_cvar(pnl, alpha=.95):
    loss = -np.asarray(pnl, float)
    var = np.quantile(loss, alpha, method="higher")
    return float(var + np.maximum(loss - var, 0).mean() / (1 - alpha))


def paired_interval(candidate, reference, days, statistic, repeats, seed):
    """Day-cluster bootstrap of statistic(reference) - statistic(candidate) on identical positions."""
    candidate, reference, days = (np.asarray(x) for x in (candidate, reference, days))
    unique = np.unique(days)
    members = [np.flatnonzero(days == d) for d in unique]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        index = np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
        values.append(statistic(reference[index]) - statistic(candidate[index]))
    point = statistic(reference) - statistic(candidate)
    return point, float(np.quantile(values, .025)), float(np.quantile(values, .975))


def evaluator_manifest(root=None):
    """Hashes of the evaluation code itself; recorded in the protocol so post-freeze edits are visible."""
    folder = Path(root) if root else Path(__file__).resolve().parent
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(folder.glob("*.py"))}


def _criterion(name, requirement, observed, passed, pending=False, hard=False):
    status = "pending" if pending else ("pass" if passed else "fail")
    return {"criterion": name, "requirement": requirement, "observed": observed, "status": status, "hard": hard}


def evaluate(protocol, store_path, reconciliations=(), evaluator_root=None):
    a = protocol["acceptance"]
    plan = protocol["plan"]
    db = open_store(store_path)
    sessions = pd.read_sql_query("SELECT * FROM sessions", db)
    daily = pd.read_sql_query("SELECT * FROM daily", db)
    episodes = pd.read_sql_query("SELECT * FROM episodes", db)
    db.close()
    eligible = daily[daily.forward_eligible == 1]
    days = sorted(eligible.day.unique())
    shake = set(days[:int(plan.get("shakedown_days", 0))])
    counted = eligible[~eligible.day.isin(shake)]
    observed_days = int(counted[counted.accepted > 0].day.nunique())
    enough = observed_days >= a["minimum_forward_sessions"]
    rows = [_criterion("forward_sessions", f">= {a['minimum_forward_sessions']} counted forward trading days (after {len(shake)} shakedown days)",
                       observed_days, enough)]
    nonconforming = sessions[sessions.conforms == 0]
    forward_nonconforming = nonconforming[nonconforming["mode"] == "forward"]
    integrity = int(sum(len(json.loads(v)) for v in forward_nonconforming.violations))
    expected_hashes = protocol["allowed"].get("evaluator_sha256")
    changed = []
    if expected_hashes is not None:
        current = evaluator_manifest(evaluator_root)
        changed = sorted(n for n in set(expected_hashes) | set(current) if expected_hashes.get(n) != current.get(n))
    rows.append(_criterion("integrity_violations", f"<= {a['maximum_integrity_violations']} among forward sessions and no evaluator edits after freeze",
                           {"session_violations": integrity, "evaluator_files_changed": changed},
                           integrity + len(changed) <= a["maximum_integrity_violations"], hard=True))
    recon_error = max([r.get("max_error", 0.) for r in reconciliations], default=None)
    recon_ok = all(r["ok"] for r in reconciliations) if reconciliations else None
    rows.append(_criterion("ledger_reconciliation", f"all sessions reconcile within {a['maximum_reconciliation_error']}", recon_error,
                           bool(recon_ok) and (recon_error or 0.) <= a["maximum_reconciliation_error"], pending=not reconciliations, hard=True))
    unresolved = int(eligible.exposure_at_close.sum())
    rows.append(_criterion("unresolved_exposure_days", f"<= {a['maximum_unresolved_exposure_days']} session-days ending with inventory open (each is an incident)",
                           unresolved, unresolved <= a["maximum_unresolved_exposure_days"], hard=True))
    dd = float(eligible.drawdown_fraction.max()) if len(eligible) else None
    rows.append(_criterion("max_drawdown_fraction", f"<= {a['maximum_drawdown_fraction']}", dd,
                           dd is not None and dd <= a["maximum_drawdown_fraction"], pending=dd is None, hard=True))
    expected = len(counted) * plan["planned_episodes_per_day"] * plan["events_per_episode"]
    accepted, rejected = int(counted.accepted.sum()), int(counted.rejected.sum())
    coverage = accepted / expected if expected else None
    rows.append(_criterion("event_coverage", f">= {a['minimum_event_coverage']}", coverage,
                           coverage is not None and coverage >= a["minimum_event_coverage"], pending=not enough))
    rate = accepted / (accepted + rejected) if accepted + rejected else None
    rows.append(_criterion("event_acceptance_rate", f">= {a['minimum_event_acceptance_rate']}", rate,
                           rate is not None and rate >= a["minimum_event_acceptance_rate"], pending=not enough))
    total = int(sessions.total_events.sum()) if len(sessions) else 0
    quarantined = int(sessions.quarantined.sum()) if len(sessions) else 0
    fraction = quarantined / (total + quarantined) if total + quarantined else None
    rows.append(_criterion("quarantine_fraction", f"<= {a['maximum_quarantine_fraction']}", fraction,
                           fraction is not None and fraction <= a["maximum_quarantine_fraction"], pending=not enough))
    # Economic claims: identical positions = same option contract and same open decision time, shakedown days excluded.
    meta = sessions.set_index("session")
    pool = episodes[(episodes.forward_eligible == 1) & ~episodes.day.isin(shake)].copy()
    pool["strategy"], pool["contract"] = pool.session.map(meta.strategy), pool.session.map(meta.contract)
    pairs = {}
    for key in ("economic", "ml_claim"):
        spec = a[key]
        if key == "ml_claim" and spec.get("applies_only_if_ml_strategy_registered") and spec["candidate"] not in set(meta.strategy):
            rows.append({"criterion": "ml_incremental_pnl", "requirement": "applies only if an ML strategy is registered", "observed": None,
                         "status": "not_applicable", "hard": False})
            continue
        side = lambda name: pool[pool.strategy == name].set_index(["contract", "open_decision_at"])
        joined = side(spec["candidate"]).join(side(spec["reference"]), lsuffix="_c", rsuffix="_r", how="inner")
        label = "cvar95_reduction" if key == "economic" else "ml_incremental_pnl"
        if joined.empty or not enough:
            rows.append(_criterion(label, f"paired interval excludes zero ({spec['candidate']} vs {spec['reference']})", {"paired_episodes": len(joined)}, False, pending=True))
            continue
        bootstrap = a["economic"]
        statistic = loss_cvar if key == "economic" else np.mean
        point, lo, hi = paired_interval(joined.net_pnl_c, joined.net_pnl_r, joined.day_c, statistic, bootstrap["bootstrap_repeats"], bootstrap["bootstrap_seed"])
        if key == "ml_claim":                    # mean(candidate) - mean(reference)
            point, lo, hi = -point, -hi, -lo
        else:
            pairs["economic"] = (joined, point)
        rows.append(_criterion(label, f"95% paired interval excludes zero in the claimed direction ({spec['candidate']} vs {spec['reference']})",
                               {"paired_episodes": len(joined), "estimate": point, "lower95": lo, "upper95": hi}, lo > 0))
    efficiency = a.get("cost_efficiency")
    if efficiency:
        if "economic" in pairs:
            joined, reduction = pairs["economic"]
            cost = float(joined.net_pnl_r.mean() - joined.net_pnl_c.mean())           # mean P&L given up by hedging, per episode
            ratio = cost / reduction if reduction > 0 else None
            ok = ratio is not None and ratio <= efficiency["maximum_cost_per_unit_cvar_reduction"]
            rows.append(_criterion("cost_efficiency", f"mean P&L given up per episode / CVaR95 reduction <= {efficiency['maximum_cost_per_unit_cvar_reduction']}",
                                   {"mean_cost_per_episode": cost, "cvar95_reduction": reduction, "ratio": ratio}, ok))
        else:
            rows.append(_criterion("cost_efficiency", "reported once the CVaR comparison has enough days", None, False, pending=True))
    hard_fail = any(r["status"] == "fail" and r["hard"] for r in rows)
    any_fail = any(r["status"] == "fail" for r in rows)
    if hard_fail:
        overall = "rejected"
    elif not enough:
        overall = "pending"
    else:
        overall = "accepted" if not any_fail and not any(r["status"] == "pending" for r in rows) else ("rejected" if any_fail else "pending")
    return {"overall": overall, "observed_forward_days": observed_days, "required_forward_days": a["minimum_forward_sessions"],
            "shakedown_days_excluded": sorted(shake), "nonconforming_sessions": nonconforming.session.tolist(), "criteria": rows,
            "note": "Pending means no performance conclusion is drawn; synthetic and replay sessions never count."}
