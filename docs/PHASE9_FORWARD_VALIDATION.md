# Phase 9: forward paper validation. Infrastructure ready, observation pending

Implemented 31 October 2026. **The forward-observation milestone is NOT complete.** No genuine forward session exists.
This phase delivers the machinery and tests; it does not and cannot supply evidence of performance.

## What exists

| Piece | Purpose | Where |
|---|---|---|
| Frozen, hash-chained protocol | Freezes allowed runner/runtime/policy/classifier hashes, execution config, risk and quote rules, evidence policy and numeric acceptance criteria BEFORE observation. Registry entries chain by hash; edits to the protocol or reordering the registry are detected. | `forward_validation/protocol.py` |
| Conformance | A registered session counts only if mode is `forward`, it was registered after the freeze, its hashes/config/risk rules equal the frozen values and its own protocol is untampered. | same |
| Observation store | Idempotent SQLite aggregation per session/day: events, accepted/rejected with reasons, fills, fees, impact, equity, drawdown, signal-eligible events, exposure at close, per-episode P&L, quarantine counts, forward-eligibility. | `forward_validation/observations.py` |
| Independent reconciliation | Rebuilds cash, positions and fees from the fill ledger alone and compares with the runner's marks, final state and exported CSVs. A tampered fill is detected. | `forward_validation/reconcile.py` |
| Acceptance evaluator | Predeclared criteria; statuses pass / fail / pending / not_applicable; overall accepted / rejected / pending. | `forward_validation/acceptance.py` |
| CLI | `freeze`, `ingest`, `reconcile`, `evaluate` (exit codes: 0 accepted, 3 pending, 2 rejected). | `src/forward_validation_cli.py` |

## Decision rule (frozen in the protocol)

- **Pending**: fewer than 60 observed forward trading days. No performance conclusion is drawn.
- **Rejected immediately**, at any sample size: integrity violation in a registered forward session, ledger reconciliation
  failure, unresolved exposure at a day's close, or drawdown above 10%.
- **With 60+ days**: event coverage >= 90%, event acceptance rate >= 90%, quarantine fraction <= 1%, and the economic claim
  (dynamic delta lowers episode loss CVaR95 versus unhedged on identical positions, paired day-cluster 95% interval excluding
  zero). If an ML strategy is registered, its paired mean-P&L increment over the no-ML strategy must also exclude zero.
- 60 days is a minimum research window, not a statistical-power guarantee.

## What the tests establish and what they do not

`tests/test_forward_validation.py` (16 tests) establishes software behavior: tamper and chain detection, conformance rules,
idempotent ingestion, reconciliation catching tampering, and every branch of the decision rule. The "accepted" case is
built from rows constructed in memory to exercise logic; it is not market evidence. The runner will not accept forward-mode
events dated in the future, so no real-mode session can be fabricated by these tests.

`data/research/phase9_infrastructure_demo_20261031/` is a labelled synthetic run of the full chain under a **DEMO** protocol
(not the registered one): two synthetic sessions ingest and reconcile exactly, contribute 0 forward days, and the verdict is
`pending`.

## What you must supply before the forward window can start

1. **Approve the acceptance criteria** (or change them) and then run `freeze`. The real protocol has deliberately not been
   frozen yet: freezing is effectively irreversible (changes require a new chained version and a new window).
2. **Decide what is frozen for observation**: with the Phase 8 result (see `PHASE8_SIGNAL_HANDOFF.md`) the hedger-only
   strategies are the only ones that can run live today.
3. A dated **instrument-master snapshot**, the **active contracts**, an **Upstox token** with market-data access, and a machine
   that stays up through sessions. Quotes come from `quote_capture.py`; decision packets from the (not yet built)
   chain-wide live feature service for ML-assisted entries, or from `paper_ops.state_builder` for hedger-only sessions.
4. Calendar time: roughly 60 trading days after the freeze.
5. Scheduling/alert delivery and any broker-specific fee or margin evidence you want reflected in the proxies.

## Run order once ready

```powershell
.\.venv\Scripts\python.exe src\forward_validation_cli.py freeze --run-dir data\research\historical_hedging_part3_final_20261031 --output data\validation\protocol_v1.json --registry data\validation\registry.jsonl
# register sessions with forward_paper.py freeze --mode forward (or paper_ops.catalog.freeze_validated), run the service daily, then:
.\.venv\Scripts\python.exe src\forward_validation_cli.py ingest    --protocol data\validation\protocol_v1.json --registry data\validation\registry.jsonl --store data\validation\obs.sqlite --session data\paper\*
.\.venv\Scripts\python.exe src\forward_validation_cli.py evaluate  --protocol data\validation\protocol_v1.json --registry data\validation\registry.jsonl --store data\validation\obs.sqlite --session data\paper\*
```

---

# Update 31 Oct 2026 (evening): protocol v1 draft, live setup and freeze checklist

## Protocol v1 draft (changes from the first draft above; NOTHING is frozen yet)
| Item | Value |
|---|---|
| Shakedown | First **3** distinct forward days: excluded from the 60-day count, coverage, acceptance rate, quarantine and economic tests; still subject to integrity, reconciliation, drawdown and open-exposure gates |
| Open positions at day end | Up to **2** session-days tolerated as recorded incidents (was 0); a third rejects |
| New criterion | Cost efficiency: mean P&L given up per episode / CVaR95 reduction <= **0.75** (Part 3 historical ratio 0.39; chosen by simulation, see [PROTOCOL_SENSITIVITY.md](PROTOCOL_SENSITIVITY.md)) |
| Strategies | unhedged, static delta, dynamic delta, band delta, adapted-no-ML. `adapted_ml` is not allowed: it would silently equal no-ML |
| Contracts | Four per week: ATM call, ATM put, ~1% OTM call, ~1% OTM put, from the nearest expiry with at least 3 trading days left; hedge future outliving the option. Rolled when fewer than 2 trading days remain or moneyness drifts more than 2% |
| Accounts | 4 contracts x 5 strategies = 20 independent paper accounts, one shared quote request per snapshot |
| Pairing | Episodes pair on (option contract, open decision time): identical positions across strategies |
| Evaluator integrity | Hashes of `src/forward_validation/*.py` are recorded in the protocol; any later edit is an integrity failure |

## Built and verified today
`paper_ops/contracts.py` (selection and roll rule, validated against the real instrument master), `paper_ops/market.py`
(read-only live market and a clearly synthetic replay market), `paper_ops/daily.py` + `src/run_paper_day.py` (one-command day:
preflight, selection, registration, causal packets, per-session service cycles, end-of-day report), `tools/run_live_day.ps1`
and `tools/schedule_paper_day.ps1` (task registers **disabled**), `RUNBOOK_PAPER_OPERATIONS.md`.

**Dry run, Fri 30 Oct 2026 (real candles, synthetic quotes, replay mode, cannot count as evidence):** 20 accounts, 840 events
sent, 0 unavailable, 42 of 42 events accepted in every account, 0 rejected, every ledger reconciles, nothing left open.
Suite: 284 offline tests pass (network blocked); earlier registries unchanged.

## What the dry run could NOT test (check on Monday morning)
1. **Live intraday candles**: the endpoint returned zero bars on a Saturday, as expected; first real check is 09:20 Monday.
2. **Real quote depth**: the displayed best level is small (one 65-unit lot after hours). The paper runner rejects an order larger than the
   best displayed level, and the position is 4 lots (260 units). If depth is thin in session, many events will be rejected. The 3 shakedown
   days exist partly to find this. Options if it happens: fewer option lots (coarser hedging) or a different execution rule (needs a new protocol).
3. Token lifetime, rate limits and machine uptime over a full session.

## Freeze checklist (tomorrow evening)
- [ ] You approve the table above (strategies, contracts, 3 shakedown days, 2-incident tolerance, 0.75 cost ceiling).
- [ ] Machine and token plan confirmed for Monday 09:00.
- [ ] `tools\run_offline_tests.py` passes; `git status` shows no edits to frozen files since the dry run.
- [ ] Freeze: `python src\forward_validation_cli.py freeze --run-dir data\research\historical_hedging_part3_final_20261031 --output data\validation\protocol_v1.json --registry data\validation\registry.jsonl` (with `PYTHONPATH=src`).
- [ ] Monday: `tools\run_live_day.ps1` (or enable the scheduled task), watch the first hour of the log.
