"""Phase 9 command line: freeze a validation protocol, ingest sessions, reconcile ledgers, evaluate acceptance.

Nothing here contacts a network or a broker. `evaluate` reports PENDING until enough genuinely new forward sessions exist.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from forward_paper import digest, runtime_manifest
from deep_hedging.config import HedgeConfig
from forward_validation.acceptance import evaluate, evaluator_manifest
from forward_validation.observations import ingest_session
from forward_validation.protocol import default_spec, freeze_protocol, load_protocol
from forward_validation.reconcile import reconcile_session

RISK_AND_QUOTE_RULES = {"max_quote_age_seconds": 3., "minimum_latency_seconds": .2, "max_fill_delay_seconds": 10.,
                        "max_quote_skew_seconds": 1., "max_daily_loss_fraction": .02, "max_total_drawdown_fraction": .10}
# adapted_ml is deliberately NOT allowed: without live classifier features it silently equals adapted_no_ml
# and would fail its own ML criterion.
STRATEGIES = ["unhedged", "static_delta", "dynamic_delta", "band_delta", "adapted_no_ml"]


def build_spec(run_dir):
    """Allowed values are read from the live runner and the Part 3 frozen artifacts, not typed by hand."""
    run_dir = Path(run_dir)
    config = asdict(HedgeConfig(**json.loads((run_dir / "experiment_plan.json").read_text())["config"]))
    config.pop("option_tick_size")                       # contract-specific; validated against the instrument catalog instead
    manifest = json.loads((run_dir / "frozen_before_historical_test.json").read_text())
    policies = sorted(v for k, v in manifest["checkpoint_sha256"].items() if Path(k).parts[0].startswith("adapted_"))
    folds = pd.read_csv(run_dir / "classifier_folds/fold_audit.csv").sort_values("prediction_start")
    import forward_paper
    allowed = {"code_sha256": [digest(forward_paper.__file__)], "runtime_sha256": runtime_manifest(), "strategies": STRATEGIES,
               "execution_config": config, "risk_and_quote_rules": RISK_AND_QUOTE_RULES, "policy_sha256": policies,
               "classifier_sha256": [str(folds.iloc[-1].model_sha256)],
               "evaluator_sha256": evaluator_manifest(),
               "source_part3_run": run_dir.name}
    return default_spec(allowed)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--run-dir", type=Path, required=True)
    f.add_argument("--output", type=Path, required=True)
    f.add_argument("--registry", type=Path, required=True)
    for name in ("ingest", "reconcile", "evaluate"):
        s = sub.add_parser(name)
        s.add_argument("--protocol", type=Path, required=True)
        s.add_argument("--registry", type=Path, required=True)
        s.add_argument("--store", type=Path, required=True)
        s.add_argument("--session", type=Path, nargs="*", default=[])
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(json.dumps(freeze_protocol(build_spec(args.run_dir), args.output, args.registry), indent=2)[:600])
        return 0
    protocol = load_protocol(args.protocol, args.registry)
    if args.command == "ingest":
        for session in args.session:
            print(json.dumps(ingest_session(protocol, session, args.store)))
        return 0
    results = [reconcile_session(s) for s in args.session]
    if args.command == "reconcile":
        print(json.dumps(results, indent=2))
        return 0 if all(r["ok"] for r in results) else 2
    report = evaluate(protocol, args.store, results)
    print(json.dumps(report, indent=2, default=str))
    return {"accepted": 0, "pending": 3, "rejected": 2}[report["overall"]]


if __name__ == "__main__":
    raise SystemExit(main())
