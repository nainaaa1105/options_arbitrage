# Phase 0b: operational hardening of the paper-trading chain

Implemented 31 October 2026. Scope: software around the frozen paper runner. No broker order API exists anywhere in the
repository, nothing here opens a network connection, and the existing ledger format, `forward_paper.py` and every frozen
protocol hash are unchanged (all additions live in new packages `src/paper_ops/` and `tools/`).

## What was added, mapped to the audit

| Audit item | Deliverable | Where |
|---|---|---|
| A04 authoritative contracts | Validate both legs against a dated Upstox instrument-master snapshot (key, segment, NIFTY underlying, CE/PE/FUT type, strike, lot, tick, expiry date, expiry ordering, snapshot age). Provenance is stored beside the protocol; nothing is registered if any check fails. | `paper_ops/catalog.py` |
| A05/A17 causal state | `CompletedBarStateBuilder`: IV, delta, ACT/365 maturity, recent return from COMPLETED five-minute bars with the same solver/Greeks code as the batch pipeline. Gaps, bad prices, out-of-order bars and unsolved IV give an explicit `Unavailable`; nothing is carried forward. | `paper_ops/state_builder.py` |
| A09 malformed input | Per-record envelope validation; bad JSON, `NaN/Infinity`, missing fields, empty IDs and conflicting duplicate IDs are quarantined (deduplicated by line hash) and later close packets still run. | `paper_ops/quarantine.py` |
| A12 liquidation equity | Closing every leg at the last recorded bid/ask with slippage, fees and displayed-depth checks. Reports `None` / `executable: false` when the book cannot absorb the close. | `paper_ops/liquidation.py` |
| A10 health | Integrity, risk latch, pending liquidation, overdue exposure, stale feed while exposed (market hours only), rejection rate, missed boundaries, quarantine count, unexecutable liquidation. Deduplicated local alert log. Unreadable session databases become a critical alert, not an exception. | `paper_ops/health.py` |
| A08/A11 orchestration | Bounded single-writer cycle: inbox files, quarantine-safe consume, export, health, alerts, heartbeat. Idempotent (changed files re-read, duplicates are no-ops), retry-limited on a locked store, stops on integrity failure, never runs unbounded. | `paper_ops/service.py` |
| A13/A14 dashboard | Session count derived from data; lineage recomputed at build time (frozen checkpoint hashes, artifact IDs, build timestamp); the saved verification file is labelled "as of its creation"; every requested paper session is shown, including missing/stale ones and their alerts. | `src/build_project_dashboard.py` |
| A18 test safety | `tools/run_offline_tests.py` screens modules by AST for import-time network/credential/data calls, blocks sockets and blanks the token. The nine legacy online/data scripts got `main()` guards and request timeouts. | `tools/`, `tests/`, `src/test_*.py` |
| A19 reproducibility | CI workflow and `tools/clean_install_check.py` (venv, pinned installs, `pip check`, offline suite). | `.github/workflows/offline-ci.yml` |

Lock design note: staleness is age based. Probing a pid with `os.kill(pid, 0)` is unsafe on Windows (it terminates the process).

## Commands

```powershell
# One bounded service cycle over an inbox of event files (JSONL packets from assemble_paper_events.py)
.\.venv\Scripts\python.exe src\paper_ops\service.py --session data\paper\my_session --inbox data\paper\inbox --cycles 1

# Register a session only after validating its contracts against a dated instrument-master snapshot (Python API)
#   from paper_ops.catalog import InstrumentCatalog, freeze_validated

# Offline tests with sockets blocked; --list shows eligibility decisions
.\.venv\Scripts\python.exe tools\run_offline_tests.py
.\.venv\Scripts\python.exe tools\clean_install_check.py --print-plan     # downloads packages only without --print-plan
```

## Verification

- Fault-injection tests (`tests/test_paper_ops.py`, 25): malformed and conflicting records, close packet after garbage,
  idempotent re-consumption, restart after partial processing equals a single pass, lock contention and stale takeover,
  second-writer refusal, tampered protocol, stale feed/overdue exposure only when exposed, thin book makes liquidation
  unexecutable, alert deduplication, catalog mismatch matrix, live-vs-batch state parity, gap handling, prefix invariance,
  and acceptance of builder output by the frozen runner.
- Two real defects were found by these tests and fixed: the state builder reused a spot from before an unusable bar when
  computing the next return, and `assess` raised on an unopened session database instead of alerting.
- `tools/run_offline_tests.py` ran 231 tests across 29 modules with sockets blocked and 0 modules excluded.

## What is NOT done (and why)

- **Classifier context features in real time.** Probabilities need lags and cross-sectional statistics over the whole
  option chain. The builder supplies hedge state only; ML-assisted entries cannot be produced live yet. Needs a
  chain-wide incremental feature service with batch/live parity tests on recorded chains.
- **Instrument-master snapshot.** The validator is tested on fixtures. A real dated Upstox master must be supplied by you (it is
  public; this code never downloads it). The paise tick-size unit is an assumption the validator cross-checks against each contract.
- **Clean-install run.** `clean_install_check.py` was exercised only in `--print-plan` mode because a real run downloads
  large wheels. Run it on the target machine or in CI. The CI workflow was not executed here; tests that need locally generated
  `data/research` artifacts will fail or skip on a fresh clone.
- **Unattended hosting, scheduling and alert delivery.** Alerts are local files. A scheduler (Task Scheduler/cron) and any
  notification channel need your machine and your permission.
- **Leg risk, partial fills, broker margin.** Option and futures fills remain atomic with configured proxies.
