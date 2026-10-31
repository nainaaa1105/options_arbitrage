# Implementation log: Phases 6, 8, 0b, 9 and the accuracy programme (31 October 2026)

| Work | Status | Primary documents / artifacts |
|---|---|---|
| Phase 6 GRU, ablations, ensembles | Complete. GRU not better than MLP; ensembles add nothing | `RESEARCH_PHASE_6.md`, `data/research/phase6_gru_ablations_20261031/` |
| Phase 8 ML-selected entries + hedging (Objective B, approved) | Complete. Selection precision higher; economic benefit not established; all hedged variants lose to costs | `PHASE8_SIGNAL_HANDOFF.md`, `data/research/phase8_signal_handoff_20261031/` |
| Phase 0b operational hardening | Complete in software; live feature service, catalog snapshot, hosting pending | `OPERATIONS_PHASE_0B.md` |
| Phase 9 forward validation | **Infrastructure only; forward observation PENDING** | `PHASE9_FORWARD_VALIDATION.md`, demo in `data/research/phase9_infrastructure_demo_20261031/` |
| Accuracy programme v2 | Complete. 72.68% vs 72.20%; 93-94% not attainable | `ACCURACY_V2_REPORT.md`, `data/research/accuracy_v2_20261031/` |

## New code (all additive; completed-phase files untouched except where noted)
`src/continuous_hedging/{recurrent,phase6}.py`, `src/run_phase6_research.py`, `src/report_phase6.py`; `src/signal_handoff/`;
`src/paper_ops/`; `src/forward_validation/`, `src/forward_validation_cli.py`; `src/accuracy_v2/`; `tools/`; `.github/workflows/offline-ci.yml`;
tests: `test_phase6_recurrent` (12), `test_signal_handoff` (14), `test_paper_ops` (25), `test_ops_tools` (7), `test_forward_validation` (16), `test_accuracy_v2` (12).

## Edits to pre-existing files
`src/build_project_dashboard.py` (data-derived counts, build-time lineage, all requested sessions shown); nine legacy online scripts
got `main()` guards and request timeouts (`tests/testToken.py`, `tests/get_*.py`, `tests/test_multiday_candles.py`, `tests/test_strike_selector.py`,
`src/test_dynamic_atm.py`, `src/test_nifty_data.py`, `src/test_volatility_features.py`); `README.md` pointers. `forward_paper.py` and all frozen hashes unchanged.

## Defects found by the new tests and fixed
State builder reused a stale spot across an unusable bar; health assessment raised on an unopened session store; dev_same_sign_run exceeded
available history at session start (first accuracy run was stopped and restarted); ML-claim bootstrap settings missing; analysis script NaN handling.

## Verification at the end of this work
268 offline tests across 32 modules pass with sockets blocked and an empty API token; phases 0-3 and 4/7/5 artifact registries verify unchanged.
Not run: the clean-install installer (plan printed only), the CI workflow, any live API call, any broker interaction.

## Environment change
CatBoost 1.2.10 installed with `--no-deps` into `.venv` (see `requirements/accuracy-v2.txt`). No other package changed.

## Addendum: paper-operation build (31 Oct evening)
New: `src/paper_ops/{contracts,market,daily}.py`, `src/run_paper_day.py`, `tools/{run_live_day,schedule_paper_day}.ps1`, `RUNBOOK_PAPER_OPERATIONS.md`,
`tests/test_paper_ops_contracts.py` (6), `tests/test_paper_day.py` (5). Edited (pre-freeze): `forward_validation/{acceptance,observations,protocol}.py`,
`forward_validation_cli.py` (shakedown days, incident tolerance, cost efficiency, contract-aware pairing, evaluator hash, no `adapted_ml`).
Verification: Friday replay dry run clean (20 accounts, 840 events, all reconcile); 284 offline tests pass; registries unchanged.
Downloaded: Upstox public instrument master to `data/raw/upstox/instruments/NSE_20261031.json.gz` (read-only GET). Live token check and quote/candle calls were read-only.
Not done: scheduler not registered, protocol not frozen, live in-session behaviour untested.
