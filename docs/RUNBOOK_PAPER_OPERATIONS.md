# Runbook: daily paper operation (read-only market data, simulated fills, no orders)

Nothing in this chain can place an order. It reads Upstox candles and quotes and records simulated fills in local SQLite files.

## Before the first live day (once)
1. Instrument master present and under 1.5 days old: `data/raw/upstox/instruments/NSE_<date>.json.gz` (the preflight fails otherwise).
2. A valid `UPSTOX_ACCESS_TOKEN` in `.env`. Access tokens are short-lived, so refresh it every morning (verify the exact lifetime in your Upstox account).
3. Validation protocol frozen (`PHASE9_FORWARD_VALIDATION.md`). Sessions registered **before** the freeze never count as evidence.
4. Scheduler registered but disabled: `powershell -File tools\schedule_paper_day.ps1`. Enable with `-Enable` only after you have reviewed the dry run.

## Every trading morning (by 09:00)
1. Machine on, charged or plugged in, sleep disabled until 15:45.
2. Refresh the token in `.env`.
3. Quick check: `.\.venv\Scripts\python.exe src\run_paper_day.py live --base data\paper\live` prints `PREFLIGHT FAILED` with reasons if the token, master or network is wrong. (The scheduled task runs the same command through `tools\run_live_day.ps1`.)
4. Look at `data\paper\logs\<today>.log` at about 09:40: you should see one line per five-minute event with `sent=` increasing.

## During the day
- First open at 09:35, then six one-hour episodes, last close 15:05; retries until 15:20 if a close fails.
- Do not edit files under `src\forward_paper.py`, `src\deep_hedging\`, `src\research_targets.py`, `src\forward_validation\`: they are frozen and an edit invalidates the evidence.
- Fixes inside `src\paper_ops\`, the scheduler and this runbook are allowed at any time.

## End of day (after 15:45)
1. The log ends with `exit=0`. `data\paper\live\daily_report_<date>.json` shows `all_reconcile: true` and `any_open_exposure: false`.
2. Review alerts: each session writes `health.json` and `alerts.jsonl`.
3. `data\paper\live\unavailable\<date>.jsonl` lists every event skipped for a missing bar or unusable quote. A few is normal; many means a data problem.
4. Weekly: ingest, reconcile and evaluate with `forward_validation_cli.py` (see `PHASE9_FORWARD_VALIDATION.md`). Result stays `pending` until enough days exist.

## Exit codes and what to do
| Code | Meaning | Action |
|---|---|---|
| 0 | Ran, reconciled, nothing open | none |
| 2 | Preflight failed (token, network, stale master, weekend) | fix the stated cause; start late only if before about 09:30, otherwise skip the day |
| 3 | Ran but reconciliation failed or exposure is open | stop; do not start tomorrow until `daily_report` is understood |

## Alerts
| Alert | Meaning | Action |
|---|---|---|
| `integrity_failure` | frozen protocol or runner hash changed | stop; never edit a frozen file; restore it from git or register new sessions under a new protocol version |
| `pending_liquidation` / `overdue_open_position` | a close failed | start the runner again during market hours; it retries closes; if the market is closed the exposure stays recorded and counts as an incident |
| `stale_feed_while_exposed` | no event for more than about five minutes with a position open | check network and token |
| `cannot_liquidate_on_displayed_book` | recorded depth cannot absorb the close | informational for paper; counts toward quote-quality review |
| `quarantined_input` | malformed or conflicting packet set aside | inspect `session\quarantine.jsonl` |
| `risk_stop_active` | daily loss, drawdown or margin stop latched | the account stays halted by design |

## Contract roll
Contracts persist in `data\paper\live\active.json` and are reused until fewer than 2 trading days remain to expiry or the option drifts more than 2% from its intended moneyness; then four new contracts and their accounts are registered and the old ones simply stop trading.

## Stopping
Close the window or stop the task (`Stop-ScheduledTask NiftyPaperDay`). Open simulated positions stay recorded; restart the runner during market hours to let it close them. Disable the schedule with `tools\schedule_paper_day.ps1 -Remove`.
