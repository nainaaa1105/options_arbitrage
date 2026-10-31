# Part 4 — dashboard, frozen forward protocol and paper execution

Implemented 31 October 2026, Asia/Kolkata.

## Explanation and scope

Part 4 makes the research inspectable and adds a persistent paper-execution
boundary for newly observed data. It does not create a new performance claim.
Part 3 remains a retrospective comparison: the ML/no-ML policies had identical
test P&L and slightly underperformed dynamic delta. Valid classifier probabilities
were available at only 13 of 1,008 test hedge decisions.

The dashboard reads the final corrected Part 3 artifacts. It displays all eight
strategies, costs, tail risk, Sharpe, drawdown, interactive equity curves,
verification and ML coverage. Paper session statuses can be included, with
forward, replay and synthetic evidence modes distinguished explicitly. The HTML
contains its own data and JavaScript and needs no internet connection or CDN.

The paper runner supports six separately registered strategies: unhedged,
static/dynamic/band delta, adapted without ML and adapted with ML. Register
separate sessions on the same contracts and packets to compare their execution.
It takes completed-bar state and then a subsequent synchronized option/futures
bid/ask snapshot. Buys use the ask, sells use the bid, with adverse slippage,
tick rounding, displayed-depth checks, configured fees and margin constraints.

Its frozen protocol includes model, classifier and runner hashes, contract lot
and tick sizes, expiry, initial capital, quote age, latency and risk limits.
SQLite commits the portfolio, event audit, simulated fills and equity together.
Duplicate event IDs do not repeat fills; conflicting payloads with the same ID
fail. A rejected order group leaves the account unchanged and is recorded.

## Deliverables

| Deliverable | File |
|---|---|
| Offline interactive dashboard builder | `src/build_project_dashboard.py` |
| Frozen protocol and persistent quote-based paper runner | `src/forward_paper.py` |
| Captured-quote/completed-state packet adapter | `src/assemble_paper_events.py` |
| Stale-feed and overdue-position monitoring | `src/monitor_paper.py` |
| Six-strategy synthetic integration demonstration | `src/run_paper_demo.py` |
| Regression tests | `tests/test_forward_paper_part4.py` |
| Generated dashboard | `data/research/part4_final_dashboard_20261031.html` |
| Demonstration ledgers and protocols | `data/research/paper_part4_final_demo_20261031/` |

Each session exports `protocol.json`, hashes, `paper.sqlite`, `events.csv`,
`fills.csv`, `equity.csv`, flat `fill_ledger.parquet`, `equity_marks.parquet` and
`status.json`. Monitoring adds `monitor.json`. Exported cumulative gross P&L,
execution impact, fees and marked equity reconcile at every accepted event.
Futures notional is not treated as a cash security purchase.

## Input and execution contract

Contract JSON has `option` and `future` objects containing `instrument_key`,
`asset_type` (`OPT`/`FUT`), `lot_size`, `tick_size`, and timezone-aware `expiry_at`.
The option also needs `option_type` (`CE`/`PE`) and `strike`. Use active, verified
Upstox contract metadata; the demonstration IDs are synthetic placeholders.

Decision JSONL records have unique `event_id`, `type` (`open`, `hedge`, `close`),
and timezone-aware `decision_at`. Opening/hedging decisions also have `state`:

- `bar_start` and `available_at`: exactly a completed five-minute candle.
- `spot_observed`, `option_close_observed`, `future_close_observed`, `delta`,
  `volatility`, `remaining_time` (ACT/365 years), and `recent_return`.
- Optional `signal_eligible`, `reference_price_deviation`, and
  `classifier_features`, keyed exactly as the frozen classifier schema.

Inventory/capital features are constructed from the session account. An external
`probability` value is not trusted as a classifier result. The adapted-ML runner
computes its own probability using a copied, hash-checked past-fitted classifier
when its domain and feature requirements are met; otherwise it uses a neutral
probability and explicit unavailable state. Feature construction remains the
responsibility of the causal market-feature pipeline. The adapter does not
invent IV, Greeks, surfaces, cross-sectional context or missing live features.

`assemble_paper_events.py` joins those decisions to `quote_capture.py`'s
normalized `quotes.parquet`, selecting the first eligible shared snapshot after
the latency boundary. Unavailable decisions are written to `unavailable.json`.
The runner validates freshness/depth again. It never fills from an LTP-only
record, an index quote, a pre-decision quote or fabricated depth.

Actual event packets add `received_at` and `quotes.option` / `quotes.future`
with instrument key, quote timestamp, bid/ask and displayed bid/ask sizes.
`synthetic_events.jsonl` in the demonstration is a complete schema example.

## Risk controls and restart behavior

Only premium-funded long options and zero funding are supported in this
protocol. Hedge decisions use consecutive five-minute boundaries, whole lots,
the inherited four-futures-lot cap and configured margin/cash reserve. Entry
must allow the full horizon before the regular session ends. Contract expiry,
quote skew, stale state, stale/crossed books and insufficient depth reject the
entire simulated order group. Forward mode rejects events preceding registration
or the historical evaluation end, and receipts from the future.

At a supplied valid event, reaching the horizon, the 2% daily-loss stop, the
10% total-drawdown stop, or a margin breach triggers liquidation. Risk stops
remain latched; resume only through a reviewed new protocol. A timer cannot
create a fill: if quotes are unavailable or closing depth is insufficient,
inventory stays open, the failed close is recorded, and monitoring reports the
unresolved exposure. Feed scheduling and valid close packets are operational
requirements. No overnight settlement or holiday-calendar inference is supplied;
special sessions require an explicit new protocol.

Restarting uses the same SQLite state and immutable protocol. Consuming the same
file again is safe: existing events are returned without changing balances.
Run one consumer per session; the runner is not a concurrent order router.

## Run locally

The existing project environment and Part 2 CPU PyTorch dependency are sufficient.
No new dashboard framework or broker-order dependency is required. Use fresh
output paths; preserve previously registered evidence.

```powershell
# Explicitly synthetic smoke run, all six paper strategies.
.\.venv\Scripts\python.exe src/run_paper_demo.py --run-dir data/research/historical_hedging_part3_final_20261031 --output data/research/my_paper_demo

# Register an actual forward session with verified active contracts.
.\.venv\Scripts\python.exe src/forward_paper.py freeze --run-dir data/research/historical_hedging_part3_final_20261031 --contracts active_contracts.json --strategy adapted_ml --mode forward --output data/paper/my_forward_session

# Existing read-only Upstox collector; active_keys.txt includes both contracts.
.\.venv\Scripts\python.exe src/quote_capture.py --keys-file active_keys.txt --output data/paper/my_quotes --snapshots 60

# completed_decisions.jsonl comes from the causal completed-bar feature pipeline.
.\.venv\Scripts\python.exe src/assemble_paper_events.py --session data/paper/my_forward_session --decisions completed_decisions.jsonl --quotes data/paper/my_quotes/quotes.parquet --output data/paper/my_packets

.\.venv\Scripts\python.exe src/forward_paper.py consume --session data/paper/my_forward_session --events data/paper/my_packets/events.jsonl

.\.venv\Scripts\python.exe src/monitor_paper.py --session data/paper/my_forward_session

.\.venv\Scripts\python.exe src/build_project_dashboard.py --run-dir data/research/historical_hedging_part3_final_20261031 --output data/research/my_dashboard.html --paper-session data/paper/my_forward_session

.\.venv\Scripts\python.exe -m unittest tests.test_forward_paper_part4 -v
```

Open the generated HTML in a local browser. It is a snapshot; rebuild to refresh
paper statuses. CSV download and strategy/measure selectors work offline.

## What is verified and what remains

The synthetic demonstration exercises six real policy/runner configurations,
seven events each, complete round trips, checkpoint loading, SQLite persistence
and accounting. It uses invented quotes for functional testing and establishes
no market performance. The existing historical metrics are preserved.

All 138 regression tests passed, including 14 Part 4 tests. Source/test
compilation passed. Replaying all 42 demonstration events after reopening the
sessions produced no duplicate fills or balance changes, all six inventories
were flat, and accounting errors were below INR 0.000001. The selected Part 3
checkpoint hashes remained unchanged. Machine-readable checks are saved in
`data/research/part4_verification_20261031.json`.

No genuine later forward period has elapsed in this implementation. Active
contracts, synchronized fresh quotes, continuous completed-bar feature updates,
and a scheduled consumer/monitor must be supplied during market sessions.
The frozen protocol sets a minimum 60 observed-session research window; that
count alone does not certify performance or completeness of market coverage.
Historical SPAN margins, calibrated market impact, queue priority and partial
fills remain outside the available data. Fees/margin remain configured proxies.

The software/reporting deliverables are implemented. Forward evidence collection,
economic validation and operational readiness remain pending actual observation.
Nothing in Part 4 submits a live order or changes the Part 3 performance finding.
