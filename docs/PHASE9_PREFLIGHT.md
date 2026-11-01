# Phase 9 pre-flight: approval sheet and Monday checklist

Status on Sunday 1 November 2026: infrastructure verified, protocol **not frozen**, no forward day recorded. Nothing below is performance evidence.

## 1. Decisions to approve before the freeze

Freezing is effectively irreversible: any change needs a new chained protocol version and a new observation window.

| Item | Proposed | Why | Your answer |
|---|---|---|---|
| Strategies | unhedged, static delta, dynamic delta, band delta, adapted-no-ML | `adapted_ml` would silently equal no-ML; ML entries are not part of this protocol | approve / change |
| Contracts | 4 per week: ATM call, ATM put, ~1% OTM call, ~1% OTM put, nearest expiry with at least 3 trading days left | matches the validated historical design | approve / change |
| Shakedown | first 3 forward days excluded from the 60-day count and from coverage, acceptance and economic tests (integrity, reconciliation, drawdown and exposure gates still apply) | finds depth and data problems without burning the window | approve / change |
| Open positions at day end | up to 2 session-days tolerated as recorded incidents; a third rejects | a failed close should not void the window on one bad day | approve / change |
| Cost-efficiency ceiling | P&L given up per episode / CVaR95 reduction <= 0.75 | historical ratio 0.39; chosen by power simulation ([PROTOCOL_SENSITIVITY.md](PROTOCOL_SENSITIVITY.md)) | approve / change |
| Minimum window | 60 observed forward trading days, then pass / fail / pending | a minimum research window, not a power guarantee | approve / change |

Facts that bear on the decision (see [CARRY_AND_PNL_CHECKS.md](CARRY_AND_PNL_CHECKS.md)):
- Gross price moves on long entries are statistically zero and a round trip costs about INR 430-450. This protocol tests hedging, not an entry edge, and it does not claim a profitable strategy.
- Depth is untested. The runner rejects an order larger than the best displayed level, and the position is 4 lots (260 units). Thin depth can cause many rejections. The 3 shakedown days exist partly to find this. The remedies (fewer lots or a new execution rule) need a new protocol version.

## 2. Checked on Sunday 1 November

| Check | Result |
|---|---|
| Live token and network (`UpstoxMarket().quotes`) | passed (the only preflight problem reported was "weekend") |
| Instrument master | refreshed to `NSE_20261101.json.gz`; `run_paper_day.py` now uses the newest dated snapshot by default. The old file would have been about 1.5 days old at 09:00 Monday, the preflight limit. |
| Replay dry run with the new master (30 Oct, ATM call, 5 strategies) | 210 events sent, 0 unavailable, 42 of 42 accepted per account, all ledgers reconcile, nothing open |
| Offline suite | 309 tests pass, network blocked |
| Frozen files | none edited; the registry test passes |

## 3. Before the freeze

- [ ] You approve section 1.
- [ ] `git status` shows no edits under `src/forward_paper.py`, `src/deep_hedging/`, `src/research_targets.py`, `src/forward_validation/`.
- [ ] `tools\run_offline_tests.py` passes.
- [ ] Freeze (PowerShell, from the repo root):
  `$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe src\forward_validation_cli.py freeze --run-dir data\research\historical_hedging_part3_final_20261031 --output data\validation\protocol_v1.json --registry data\validation\registry.jsonl`
- [ ] Sessions registered before the freeze never count, so register forward sessions only afterwards.

## 4. Monday 2 November (before 09:00)

- [ ] Confirm that Monday 2 November is a normal NSE trading day (check the exchange holiday list).
- [ ] Machine on and plugged in, sleep disabled until 15:45, stable network.
- [ ] Refresh `UPSTOX_ACCESS_TOKEN` in `.env` (tokens are short-lived).
- [ ] Preflight: `.\.venv\Scripts\python.exe src\run_paper_day.py live --base data\paper\live` prints `PREFLIGHT FAILED` with reasons if anything is wrong. (On a weekend it will always report the weekend.)
- [ ] Start with `tools\run_live_day.ps1` (or enable the scheduled task after reviewing a dry run).
- [ ] At about 09:40 check `data\paper\logs\<date>.log`: one line per 5-minute event with `sent=` rising.
- [ ] First open is 09:35. If the first live intraday candles come back empty, stop and read the log before retrying.

## 5. After the first day (after 15:45)

- [ ] Log ends with `exit=0`.
- [ ] `data\paper\live\daily_report_<date>.json`: `all_reconcile: true`, `any_open_exposure: false`.
- [ ] `data\paper\live\unavailable\<date>.jsonl`: a few entries are normal, many mean a data problem.
- [ ] Count rejected events by reason; depth rejections are the main thing the shakedown must reveal.

## 6. What can still go wrong

| Risk | Effect | Mitigation |
|---|---|---|
| Token expires mid-session | quotes fail, events unavailable | refresh each morning; runner retries bars, skips events rather than fabricating |
| Machine sleeps or loses network | missed events; with an open position, an overdue close | disable sleep; restart the runner within market hours, it retries closes until 15:20 |
| Thin displayed depth | many rejected orders | shakedown days; changing the rule needs a new protocol version |
| Rate limiting (HTTP 429), seen on the historical API | missing bars | one shared quote request per snapshot already limits calls; watch the unavailable log |
| Calendar time | 60 trading days is about 3 months | nothing shortens it |
