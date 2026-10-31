# System audit — 31 October 2026

## Verdict

The repository is a substantial historical research implementation with working
pricing, causal feature engineering, classifiers, hedge accounting, neural
policies, reports and paper-execution primitives. It is not yet a reliable
unattended forward-paper system or an economically validated trading strategy.
Earlier statements that Parts 3/4 were implemented describe their software
scope; they must not be read as an assertion that all edge cases are correct.
This audit reproduced operational defects missed by the existing tests.

No application code, weights, original datasets or registered paper protocols
were changed in this audit. Temporary fixtures were used for fault injection.
The report and audit evidence/logs are the only new repository artifacts.

Owner tags: **[ME]** means code/research work I can implement; **[YOU]** means
account access, preferences, approvals or operation you control; **[BOTH]**
means I implement the tooling and you supply decisions/access or run future
market observation. A research improvement is not a promised performance gain.

## Verified evidence

- The explicit offline regression suite passed: **138 tests**. The suite misses
  the fault-injection cases below. Generic discovery was deliberately avoided:
  some legacy files execute authenticated HTTP calls at import time.
- Every Python file under `src/` and `tests/` parsed successfully.
- `pip check` reported no broken requirements in the current environment.
  This does not prove reproducibility in a newly installed environment.
- Expanded option features: **941,949 rows, 60 expiries, 283 option dates**;
  zero duplicate instrument/timestamp keys; zero mismatches between recorded
  feature/spot availability and candle-start plus five minutes. This checks
  stored alignment fields, not independent vendor accuracy of every price.
- **6,324 option rows have zero open interest**. Do not remove whole dates or
  relabel missing liquidity as good data without investigating provider fields.
- Nine relative local Markdown hyperlink targets in root reports/README exist.
  No missing local targets were found in that scope. Inline code paths, prior
  chat links, remote HTTP availability and all browser interactions were not
  certified by this check.
- `.env` is ignored and was not returned by `git ls-files .env`; token values
  were not printed. This is not an exhaustive secret-history scan.
- Live Upstox authentication was not re-tested in this audit. Prior successful
  collections are evidence of historical operation, not current token validity.

Evidence: `data/research/system_audit_tests_20261031.log` and
`data/research/system_audit_probes_20261031.json`.

## Model and economic results

The later-period classifier artifact scores 12,466 rows over 28 sessions:

| Target | Accuracy | ROC-AUC | Precision | Recall | Majority accuracy |
|---|---:|---:|---:|---:|---:|
| Endpoint convergence | 72.20% | 0.7825 | 62.91% | 52.88% | 64.49% |
| First passage | 86.53% | 0.9144 | 85.57% | 97.60% | 71.42% |
| Two-bar sustained | 75.71% | 0.8433 | 70.79% | 73.78% | 57.13% |

Source: `data/research/expanded_20261030/fresh_evaluation/fresh_metrics.csv`.
First passage is an easier/different target with a 71.42% positive rate; its
scores cannot be substituted for endpoint accuracy. Earlier 93%+ selective
scores describe a restricted subset, not all opportunities or trade win rates.
The previously observed test periods no longer provide pristine confirmation.

The corrected historical hedge test has 168 fixed long-option episodes across
28 sessions and INR 5,000,000 initial capital per sequential account:

| Strategy | Net P&L | Episode loss CVaR 95% |
|---|---:|---:|
| Unhedged | +80,827.85 | 11,096.73 |
| Dynamic delta | -345,002.28 | 4,555.48 |
| Adapted without ML | -346,647.44 | 4,568.57 |
| Adapted with ML | -346,647.44 | 4,568.57 |

The adapted strategy's gross P&L is approximately INR 182, fees INR 311,833,
and execution impact INR 34,996. Cost drag dominates this experiment. Lower
tail loss is a risk reduction, not evidence of positive expected return.
The identical ML/no-ML results establish no incremental ML benefit.

## Confirmed defects and weak controls

Priority P0 blocks unattended paper operation; P1 is high priority; P2 is
maintenance/reporting. Reproduced means exercised with temporary synthetic
inputs; static means established from the implementation and needs a dedicated
regression case before a repair is considered complete.

| ID | Priority / evidence | Finding and consequence | Correction | Owner |
|---|---|---|---|---|
| A01 | P0, reproduced | A loss-stop event attempts closing, fails for insufficient displayed depth, and rolls back the `halted` state along with the trade. The account remains exposed with `halted=false`; a later recovered quote can allow an ordinary hedge. | Persist the risk latch independently of unsuccessful fills; record a pending-liquidation state and permit only risk-reducing recovery actions. Test failed-close/restart/recovery sequences. | **[ME]** |
| A02 | P1, reproduced | After accepting the initial event, skipping one five-minute boundary makes subsequent normal hedge events fail the exact-next-boundary check. Rejections do not advance that boundary, so normal recovery cannot occur before terminal close. | Separate decision-grid validation from missed-event handling; use an explicit stale/recovery state, allow the next valid grid boundary, and retain the missed interval in the audit. | **[ME]** |
| A03 | P1, static | Code integrity freezes only `forward_paper.py`. Changes to imported policy, accounting, cost or feature helpers can change behavior without tripping that hash. Editing the frozen file itself also prevents even the current monitor from loading the old session. | Freeze a dependency/environment manifest; version execution adapters; provide a read-only recovery/inspection path independent of current runner hash. | **[ME]** |
| A04 | P1, static | Contract registration validates structure, not an authoritative active instrument catalog. Incorrect underlying, lot size, tick or expiry can be supplied with an `NSE_FO` prefix. | Validate both legs against a dated Upstox instrument master; freeze its provenance, matching underlying and permitted units. | **[BOTH]** |
| A05 | P1, static | Paper state checks timestamp freshness but does not prove how supplied IV, Greeks and classifier features were built. A claimed availability timestamp is not lineage. Option-type/delta sign consistency and proper session-grid alignment are not fully checked. | Produce state internally from synchronized bars/context; validate call/put delta ranges, bar grid, expiry, feature schema and provenance. | **[ME]** |
| A06 | P1, static | ML domain checks omit the full session-horizon constraint used in training. A late intraday hedge can supply a nominally eligible 30-minute convergence feature after that horizon no longer fits before close. | Reuse the shared target-domain validator at every probability decision; log explicit unavailability. | **[ME]** |
| A07 | P1, static | The paper runner accepts inherited horizon settings but computes later `remaining_steps` using a hard-coded 30 minutes. | Restrict the protocol to validated 30-minute checkpoints or remove the constant and separately validate matching 60-minute models. | **[ME]** |
| A08 | P1, static | SQLite commits each event atomically, but the account is read before acquiring a transaction spanning the read/modify/write cycle. Two consumers can calculate from the same old account. Documentation says one consumer, but code does not enforce it. | Enforce a single-writer lock or use a serialized transactional read/update with state-version conflict detection. | **[ME]** |
| A09 | P1, static | Invalid JSON, missing `event_id`, or non-finite values can fail before the per-event rejection handler; one bad record can abort the consume batch and prevent later close packets from being reached. | Validate envelopes per line, quarantine malformed input, persist diagnostics and define continuation behavior for open positions. | **[ME]** |
| A10 | P1, static | Unavailable adapter decisions are written separately but do not enter the paper account's event/health stream. Monitor health can therefore omit missed decisions and quote failures. | Route adapter failures/heartbeats through a shared health channel; show pending actions, rejected closes and coverage gaps. | **[ME]** |
| A11 | P1, static | The full live chain is not implemented as one continuously running process. REST quote capture, completed-state construction, packet assembly, consuming and monitoring are separate commands. | Implement a bounded service/scheduler, bar-completion coordination, retries, heartbeat, crash recovery and local alert delivery. | **[BOTH]** |
| A12 | P1, static | Paper equity uses mid marks; it is not liquidation equity. Stops can see a healthier account than the executable bid/ask liquidation value. | Report both mid-marked and conservative liquidation equity; define which risk controls use each, including closing fees and depth. | **[ME]** |
| A13 | P2, static | Dashboard reads saved `verification.json` and displays that hashes were verified, without recomputing them for the data it just loaded. It also hard-codes the 28-session narrative. | Revalidate lineage or label verification as an as-of saved check; derive session counts and add generation time/artifact IDs. | **[ME]** |
| A14 | P2, static | Dashboard silently ignores requested paper directories missing `status.json`, and does not display `monitor.json` alerts. Thus a missing or stale session can appear absent rather than unhealthy. | Display every requested session, missing files, status age, overdue exposure and current monitor alarms. | **[ME]** |
| A15 | P2, confirmed docs | README introduction still says recovery is blocked by HTTP 403; later paragraphs describe successful repaired collection. | Reconcile the current status and retain historical failure notes in dated reports only. | **[ME]** |
| A16 | P1, confirmed routing | Legacy signal/training commands default to `data/processed` artifacts, while corrected research artifacts are isolated under `expanded_20261030`. Default commands can therefore use older context/weights. | Add a canonical run registry and mandatory matching model/data/schema IDs; explicit promotion after validation, preserving old artifacts. | **[ME]** |
| A17 | P1, static | `predict_research_model.py` uses a fixed saved historical context path. New future rows fail schema-v2 context matching unless the full causal context is extended. | Add rolling full-surface feature context, daily HV/VIX updates, and incremental batch/live parity tests. | **[ME]** |
| A18 | P2, confirmed repo | Legacy test scripts make API calls at import time, some without timeouts. Generic test discovery may use credentials or hang. No checked-in CI/packaging configuration was found at the inspected root paths. | Isolate online diagnostics, add main guards/timeouts, define offline discovery/CI and credential-free fixtures. | **[ME]** |
| A19 | P2, confirmed dependencies | Current packages satisfy requirements, but base dependencies use broad minimum versions. The CPU Torch file does not lock the full environment. | Create a tested reproducible lock/environment manifest and a clean-install validation job. | **[ME]** |

Primary source locations: `src/forward_paper.py:154` (event processing),
`:176` (state transitions), `:217` (boundary rule), `:260` (horizon constant),
`src/monitor_paper.py:10`, `src/assemble_paper_events.py:10`,
`src/build_project_dashboard.py:12`, and `src/predict_research_model.py:18`.

The temporary zero-delta probe in `system_audit_probes_20261031.json` was accepted;
zero delta alone can be legitimate. It is **not**, by itself, proof of a sign
validation bug. The missing explicit option-type/sign cross-check is the static
finding in A05.

## Research limitations and performance weak spots

| ID | Finding | Best next experiment / remedy | Owner |
|---|---|---|---|
| R01 | Endpoint accuracy 72.20% and AUC 0.7825 do not meet the requested 93–94% whole-population benchmark. There is no demonstrated route to guarantee that level fairly. | Use purged chronological validation, compare simple baselines, calibration and learning curves; report coverage and uncertainty. Freeze all changes before evaluating genuinely later data. | **[ME]** for implementation; **[BOTH]** for later evidence |
| R02 | The convergence target predicts relative deviation reduction, not net profit or hedge improvement. First-passage targets can count a temporary reduction even if the exit outcome is poor. | Predeclare a cost-aware endpoint/P&L or residual-risk objective, derived from the same causal executable book. Evaluate alongside the unchanged original target; do not select a target by final-test accuracy. | **[BOTH]** |
| R03 | Only 13/1,008 test hedge decisions have valid probabilities; nearest-ATM fixed exposures differ from the classifier's eligible-deviation population. | Choose between a general-purpose liability hedger and a convergence-selected entry strategy. For the latter, predeclare the eligible entry universe and compare all hedgers on identical selected liabilities. Preserve domain restrictions. | **[BOTH]** |
| R04 | Both adapted policies have identical test P&L; superior neural/ML hedging is unproven. | Use more representative training scenarios, regularized adaptation and controlled ablations; compare against optimized no-trade-band and cost-aware conventional baselines. Do not retrain repeatedly against the same test period. | **[ME]** |
| R05 | Costs dominate gross edge in the hedge experiment. | Decompose taxes, brokerage and turnover; validate them against contract notes; tune rebalance bands/frequency on development data and compare risk-adjusted cost savings. Avoid removing real costs to improve a score. | **[BOTH]** |
| R06 | GBM/Heston simulations are stylized and uncalibrated; historical IV/maturity/basis differ from their training support. | Calibrate training-only distributions, include volatility/basis jumps and transaction-cost stresses, track out-of-distribution rates and define an abstain/fallback policy. | **[ME]** |
| R07 | BS/LOO reference prices are relative model estimates, not proof of risk-free arbitrage. Neighbor prices and put-call parity can reflect asynchronous or noisy quotes and carry assumptions. | Compare forward/Black-76 or constrained surface references on development data; validate parity using matched observed quotes and carry, and quantify model uncertainty. | **[ME]**; **[BOTH]** for real quote/carry inputs |
| R08 | Only 28 test sessions / 168 hedge episodes; approximately nine episodes drive the 95% tail. Large negative annualized hedge Sharpes are possible with persistent cost losses and low daily variation; they do not alone prove a formula bug. | Show raw daily returns and confidence intervals, collect more sessions, and avoid long-run extrapolation from these annualized ratios. | **[BOTH]** |
| R09 | Market test periods were reviewed over multiple development rounds; chronological splitting cannot restore an untouched holdout. | Register the protocol now and collect a genuinely later period. Target 60 observed sessions as a minimum research window, not a performance guarantee. | **[BOTH]** |
| R10 | Zero OI, incomplete chains, missing days and selection on complete historical paths can affect the sampled population. Existing alignment metadata is consistent but not an independent full-price audit. | Audit coverage by date/expiry/strike and rejection reason; sample raw API reconciliation; retain missing-data and zero-trade-session reporting. Correct provider misinterpretation only with evidence. | **[ME]** tools; **[BOTH]** API evidence |
| R11 | OHLC fills, deterministic F/S sensitivity and notional margin fractions remain approximations. Full option/future execution is treated atomically in paper simulation, although real legs may fill separately. | Obtain bid/ask observations and broker margin evidence; model leg risk, partial fills, depth depletion and basis risk; report sensitivity before any execution deployment. | **[BOTH]** |
| R12 | No genuine forward performance has been collected by Part 4; the six-strategy 42-event demo is synthetic. | Start observation after hardening A01–A12, maintain clocks/uptime, review missing observations and accrue later-session evidence. | **[BOTH]** |

## What you need to do

1. **[YOU]** Choose the product objective: hedge existing long-option exposure,
   or choose convergence-driven option entries and then hedge them. Also specify
   risk limits, capital, permitted contracts and whether shorts are excluded.
2. **[YOU]** Maintain Upstox authorization/subscriptions and access to verified
   active instruments; keep credentials in `.env`, never in chat or reports.
3. **[YOU]** Supply the broker's actual charges/margin evidence if you want
   claims closer to your account's executable economics.
4. **[YOU]** Provide an always-on machine/network and approve any eventual
   service scheduling, external alerts or deployment. No broker-order permission
   is assumed by this audit.
5. **[BOTH]** Run and review future sessions. I can build/operate bounded tools
   while active in a session; future market time and reliable hosting cannot be
   manufactured by tuning or synthetic data.

## Recommended remaining-work order

| Order | Deliverable | Owner | Completion evidence |
|---|---|---|---|
| 1 | Fix failed-close risk latch and missed-boundary recovery; add adversarial event-sequence tests | **[ME]** | Failure/restart/recovery tests with exposure retained and safe behavior |
| 2 | Harden schema, metadata, code lineage, concurrency and input failures | **[ME/BOTH]** | Deterministic replay, single-writer enforcement, active-master checks |
| 3 | Repair dashboard health/status reporting, README and canonical artifact routing | **[ME]** | Link/CLI smoke tests and browser-level dashboard tests |
| 4 | Decide entry-versus-hedging objective and freeze experiment design | **[YOU+BOTH]** | Written target/universe/capital/risk/validation protocol |
| 5 | Implement continuous causal feature/context feed and supervised paper orchestration | **[ME]** | Recorded quotes → completed features → inference → ledger → alerts integration test |
| 6 | Calibrate pricing/simulation/cost baselines using development data only | **[ME/BOTH]** | Walk-forward research report with untouched later evaluation reserved |
| 7 | Collect and evaluate genuinely later quote-based paper sessions | **[BOTH]** | Coverage, costs, risk and paired confidence intervals, plus reconciliation |
| 8 | Decide deployment based on evidence rather than headline accuracy | **[YOU]** decision, **[ME]** evidence | Explicit acceptance/rejection criteria and operational readiness review |

## Audit boundaries

This pass inventories the repository, checks all Python syntax, reruns the
explicit offline suite, reviews the principal pricing/ML/hedging/paper/dashboard
paths and reproduces critical paper failure modes. It does not execute every
legacy collector, retrain every model, certify the vendor's raw prices, verify
all remote links, or perform a live broker/margin test. No claim of exhaustive
bug absence follows from this report. Confirmed findings should be repaired
before expanding unattended operation.
