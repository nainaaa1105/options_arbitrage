# Data, target design, pricing and training remediation

Update: [ACCURACY_RECOVERY_20261030.md](ACCURACY_RECOVERY_20261030.md) records the
subsequent successful authenticated connection, certificate repair, authentic
spot repair, expiry backfill and retraining. Access/data-repair blockers below
describe the earlier audit date and are superseded by that investigation.

Prepared 27 October 2026, Asia/Kolkata.

## What the audit establishes

Yes: the historical spot/option mismatch is a **coding error in the join**, not
proof that Upstox returned bad candles. Valid responses can still be combined
incorrectly. Upstox labels each candle with its start time; the `close` belongs
to the end of that candle's interval. This convention is documented for both
[spot historical candles](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/)
and [expired option candles](https://upstox.com/developer/api-documentation/get-expired-historical-candle-data/).

| Record labelled 09:15 | Price represented | Available after |
|---|---|---|
| Five-minute option close | Last option trade in 09:15–09:20 bar | 09:20 |
| One-minute spot close | Last index value in 09:15–09:16 bar | 09:16 |
| Correct one-minute spot match, labelled 09:19 | Last index value in 09:19–09:20 bar | 09:20 |
| Equally valid five-minute spot match, labelled 09:15 | Index close of the matching five-minute bar | 09:20 |

The collector previously matched the first two records on 09:15. The corrected
join matches **close availability times**. You can repair with authentic
five-minute spot candles matching the option interval; one-minute data are an
alternative, not a requirement. `bar_minutes` provenance is checked when supplied.
The result keeps the original option bar identifier plus `feature_available_at`,
`spot_timestamp` and `spot_available_at` for auditing.

Matching candle windows does not make the option's last transaction a fresh
executable quote. A thin option may last trade earlier within the bar. Bid/ask
timestamps and sizes are needed to address that separate microstructure issue.

You confirmed that this project's `.env` works for your API calls. This agent's
secure V3 request still returned HTTP 403 with a client-signature blocking
message. That is not evidence that your token or local connection is invalid.
Repair and quote tools now default to the **same authenticated project client**
used by collection. They stop on access denial; they do not spoof browser
signatures, disable TLS verification or invent data. Offline repair also accepts
exported Parquet files from your working connection.

## What was completed in this pass

| Area | Implemented | What remains |
|---|---|---|
| Alignment | Exact close-time joins, one- or five-minute spot inputs, separate repair output, duration provenance | Retrieve authentic spot history locally, then rebuild derived data |
| Issue 4: data coverage | Coverage audit; configurable expiry/date/session collection; manifests; raw overwrite protection; independent daily/5-minute spot collector | Additional real option sessions/regimes and any unavailable older history |
| Volatility history | Long-gap return invalidation, missing-history flags in research, support for independent complete daily spot history | Obtain that daily history; rebuild active HV features |
| Issue 5: drift | Eight rolling/expanding/weighting/input policies compared on purged folds; drift diagnostics and session uncertainty | Monitor and retrain on new matured data; no permanent solution to nonstationarity |
| Issue 6: quote detail | Bounded quote recorder; raw snapshots; bid/ask sizes; timestamps; stale/crossed/missing-book flags; capture manifest | Actually record quotes and instrument metadata; obtain historical quotes if needed |
| Issue 11: outcomes | Exact reason audit; session-horizon gates in signal/action inference; complete-path labels; nullable missing outcomes | Special-session calendar support; survival targets only if intentionally extending the problem |
| Target design | Endpoint, sampled first passage, two-close sustained, continuous reduction, quantile targets and cashflow-label functions | Select the economic objective and validate on synchronized data/new dates |
| Pricing | Black–76, target-strike-excluded parity forwards, robust leave-one-out variance smoothing and static-shape diagnostics | Real forward/mid inputs; constrained cross-expiry surface; consistent IV/Greeks rebuild |
| Modeling | New research candidates, training-only policy selection, fixed cutoff comparisons and majority baselines | No candidate is promoted based on an attractive validation score |

Active raw files and model weights were not replaced by these experiments.
Research candidates are separate, and all current scores remain provisional
because the historical alignment repair has not been completed.

## Issue 4 — too few independent observations and incomplete time coverage

The master has 519,725 rows, 1,994 contracts, 33 expiries and 156 observed sessions.
The model split has 109 training, 23 validation and 24 historical test sessions.
Those are not hundreds of thousands of independent market experiments: many
rows share spot moves, expiry effects and volatility shocks.

The audit found **no observed sessions between 23 September 2025 and 18 February
2026**, a 148-calendar-day jump. The saved raw files contain three September
2025 test expiries and 30 later expiries. This pattern is consistent with the
collection selection logic; it does not show that Upstox had no data in between.

Changes made:

- `collect_all.py` now accepts `--run-mode ALL`, expiry bounds, session count,
  a local expiry manifest and `--plan-only`. Existing raw files remain protected.
- New collection manifests record requested/observed contracts, failed contract
  keys, sessions and alignment provenance. Existing files without manifests are
  explicitly labelled unverified, rather than assumed complete.
- `collect_expiry.py` validates session coverage before saving and refuses to
  overwrite an existing raw file. Contract failures remain observable.
- `collect_spot_history.py` collects independent daily or minute spot history,
  splits minute requests by calendar month and preserves interval provenance.
- Offline pricing/audits no longer require an API token merely to import config;
  actual API calls still require authentication.

Complete this by collecting more **dates and regimes**, not merely more strikes
from the same dates. Cover the existing gap where your entitlement and instrument
history permit it, preserve actual listing dates, and extend beyond the last
observed session, 15 September 2026. Keep a genuinely unused later period for
confirmation after target and model choices are frozen. If an expiry endpoint
does not return older dates, use a verified expiry manifest and check that the
provider actually supports those expired keys. Do not infer unavailable contracts.

Longer histories per expiry are useful for hedging research, but requesting 20
sessions does not guarantee a particular weekly contract was listed for all 20.
The collector now exposes missing coverage instead of silently calling it complete.

### Why the gap matters for HV

The old daily-HV calculation could treat a return spanning the 148-day coverage
gap as one daily return. The new conservative fallback invalidates returns over
calendar gaps longer than seven days and waits for enough completed observations
again. That threshold is a missing-coverage guard, not an exchange holiday calendar.

| Training feature | Previously missing rows | Gap-aware missing rows |
|---|---:|---:|
| 5-day HV | 2,225 | 5,818 |
| 10-day HV | 3,800 | 9,741 |
| 20-day HV | 8,709 | 15,345 |

More missing values are preferable to fabricated daily returns. The full solution
is an independent, complete **daily NIFTY close history**, including dates where
no options were collected, with enough prior history for warmup. The builder
accepts it through `--daily-spot`. Daily returns remain shifted to use only prior
completed sessions. HV annualization by 252 trading sessions is appropriate;
the earlier 252/365 bug concerned converting ACT/365 expiry fractions into days,
which is a different calculation.

## Issue 5 — changing market distributions

The observed positive rate changes from 40.23% in training to 35.86% in validation
and 30.10% in the previously inspected test period. Feature drift is also visible
in spot levels and volatility variables. This establishes distribution change;
it does not identify a particular macroeconomic cause.

I tested these policies using the same regularized LightGBM model, corrected
feature timing, gap-aware HV, three expanding date folds, a full-session gap and
purging through the future candle's close availability. Thresholds were fixed at
0.50 for comparison. No test-split file was opened for model development.

| Policy | Mean internal ROC-AUC |
|---|---:|
| Expanding, excluding raw index/strike levels | **0.7394** |
| Expanding, uniform observations | 0.7387 |
| Recent weighting plus that input exclusion | 0.7343 |
| Equal total weight per session | 0.7332 |
| Recency weighting, half-life 40 sessions | 0.7327 |
| Rolling 60 sessions | 0.7317 |
| Recency weighting, half-life 20 sessions | 0.7316 |
| Rolling 40 sessions | 0.7260 |

The selected policy removes only `nifty_spot`, `strike` and `mkt_spot_mean` as raw
index levels; it retains moneyness, relative features and option prices. It is
not a claim that every input is scale invariant. Its score advantage is small.
Heavier recency weighting and shorter histories reduced useful information here,
so they were not forced into the chosen model.

On the existing 7,803-row validation block, the selected policy had **70.00%
accuracy, 0.7524 ROC-AUC, 0.6029 PR-AUC, 0.5670 F1 and 0.1898 Brier score**.
The comparable expanding baseline had 69.47% accuracy and 0.7513 ROC-AUC. Recall
slipped slightly; every metric did not improve simultaneously. Session-bootstrap
intervals are approximately 68.11–71.85% accuracy and 0.7241–0.7796 ROC-AUC.

To complete operational drift handling: log feature missingness, train-defined
distribution bins, matured label rates and recent calibration; compare candidate
models by whole sessions; retrain using only outcomes already available at the
training timestamp; and promote against predeclared risk/coverage requirements.
PSI/KS outputs here are descriptive, not automatic significance tests on
independent rows. A changing regime can be managed, not permanently removed.

## Issue 11 — missing outcomes: the actual cause in this dataset

This audit corrects the earlier broad suspicion of missing-data bias. The
current candidate set has **no unexpected missing future candles** at these
horizons. Missing endpoints are explained by the session close.

| Quantity | Rows |
|---|---:|
| All rows | 519,725 |
| Signal eligible | 395,554 |
| Eligible but below the target's one-point deviation floor | 322,079 |
| Eligible with deviation at least one point | 73,475 |
| Observed 30-minute outcomes | 67,274 |
| 30-minute session-end censored outcomes | 6,201 |
| Observed 60-minute outcomes | 61,305 |
| 60-minute session-end censored outcomes | 12,170 |

Thus most of the reduction to the ML dataset is **intentional target filtering**,
followed by session-end censoring. It is not evidence that the API randomly lost
hundreds of thousands of outcomes. The current train/validation/test row counts
sum to 67,274; boundary purging did not explain that historical sample reduction.

The signal and actionable-trade paths now enforce enough remaining session time
for a completed five-minute signal bar plus the holding period. Under the current
15:30 close convention, the last eligible bar start is **14:55 for 30 minutes**
and **14:25 for 60 minutes**. A missing outcome stays nullable; it is never changed
to “did not converge.” The new path targets require all expected sampled closes.

For this same-day strategy, the cutoff addresses the observed censoring cause.
For special/short sessions, provide the actual dated exchange session calendar.
If you want overnight holdings or time-to-convergence instead, define that as a
different problem and use explicit right-censoring/survival methods; do not fill
overnight gaps or pretend the market was continuously executable.

## A better target design

The correct target follows the intended action. A higher score on a different
label is not automatically a better model for the original problem.

Let `D(t) = market_price(t) - reference_price(t)`. The three implemented targets
use the same complete future paths and the same features/folds:

1. **Endpoint:** `|D(t+30)| <= 0.5 |D(t)|`. Useful if you deliberately hold until
   the endpoint. It does not answer “ever converge within 30 minutes.”
2. **Sampled first passage:** that condition holds at any observed close after
   entry, up to 30 minutes. This matches a literal “within 30 minutes” question.
   It does not infer intrabar touches or guarantee an order can capture them.
3. **Two-close sustained:** the condition holds at two consecutive sampled
   closes. It reduces dependence on a single transient snapshot. It does not
   prove the deviation stayed below the threshold between those snapshots.

All comparisons used the same 50,278 training and 7,803 validation observations,
the training-selected policy, a fixed 0.50 cutoff and no test-based selection.

| Target | Validation positive rate | Training-majority baseline accuracy | Model accuracy | ROC-AUC | Balanced accuracy |
|---|---:|---:|---:|---:|---:|
| Endpoint | 35.86% | 64.14% | 70.00% | 0.752 | 66.65% |
| Sampled first passage | 74.20% | 74.20% | **86.70%** | **0.888** | 75.17% |
| Two-close sustained | 43.39% | 56.61% | **72.83%** | **0.823** | 73.61% |

First passage had 85.41% precision, 98.98% recall and 51.37% specificity. Its
high positive base rate and weak negative specificity must accompany the 86.70%
headline. Sustained convergence is a useful intermediate research target when
transient deviations are a concern. Neither target establishes trading profit.
They have not replaced the active project's label merely to raise accuracy.

### Magnitude and uncertainty

A continuous forecast retains information discarded by binary labels. I tested
`log((future_absolute_deviation + 0.05) / (current_absolute_deviation + 0.05))`.
Its validation MAE was **0.7805**, versus **0.8987** for the training-median
constant forecast, about a 13.2% reduction. Lower values forecast stronger
convergence; positive values forecast widening.

The new 10th/90th percentile forecasts improved pinball loss over training-quantile
baselines. Their nominal 80% interval covered **77.55%** of validation observations,
so uncertainty was still somewhat underestimated. These intervals concern the
deviation ratio, not monetary hedging loss, and have no universal coverage guarantee.

### Recommended economic target for the eventual system

For trade selection, use continuous **net cashflow/P&L after costs**, with
`P(net P&L > 0)` as a supporting classification output. For a static hedge:

```text
net P&L = option_side * option_quantity * (option_exit_fill - option_entry_fill)
        + signed_hedge_quantity * (hedge_exit_fill - hedge_entry_fill)
        - remaining transaction costs
```

`research_targets.executable_cashflow_targets` implements and validates that
accounting. It requires supplied fills, quantities, costs and timing; an index
identifier is rejected as a tradable hedge. It distinguishes measured costs
from configured proxies. Spread already included in bid/ask fills must not be
subtracted twice. Replaying the existing 845 proxy-backtest trades reproduced
their net cashflows to numerical precision; that checks accounting, not real
execution quality. The cost-aware model cannot yet be trained honestly from
missing historical quotes or hedge fills.

For deep hedging, train a hedge policy to reduce the distribution/tail risk of
hedging loss and costs relative to delta hedging. A convergence classifier can
be an auxiliary input; its accuracy is not a prerequisite or a proof of hedge
quality. This is consistent with the risk-and-frictions objective in
[Deep Hedging](https://arxiv.org/abs/1802.03042). A static hedge label is only a
bridge to that sequential cashflow problem.

## Can changing pricing methodology help?

Yes, by making the benchmark financially more coherent. It must be selected by
pricing consistency and independent validation, not whichever benchmark makes
labels easiest. Pricing an option with its own solved IV and calling the result
fair value is circular; the reference must continue to exclude the target.

The recommended sequence is:

1. Synchronize completed option/spot windows and use quote freshness where available.
2. Use good bid/ask mids with spreads and sizes instead of assuming the last trade
   is an executable mid.
3. Use a consistent forward/carry convention. Black–76 with an observed matching
   forward, or a target-strike-excluded robust parity forward, can incorporate
   carry more coherently than zero-dividend spot pricing. A weekly option may
   require careful expiry interpolation; do not assume a matching future exists.
4. Fit total variance `w = IV^2 * T` against log forward moneyness within each
   contemporaneous expiry. Exclude the target option, and exclude its strike pair
   if that pair otherwise contributes to the forward estimate.
5. Add explicit vertical-spread, butterfly/convexity and cross-expiry checks.
   A smooth curve alone does not establish absence of arbitrage.
6. Recompute IV, prices, Greeks and labels consistently. Spot delta and futures
   delta are different sensitivities; do not mix conventions in the hedge.

Implemented research code includes Black–76, leave-target-strike-out parity
forwards, robust local quadratic total-variance fits and static price diagnostics.
Tests confirm the Black–76/Black–Scholes equivalence under identical carry, target
exclusion and expiry isolation.

On 1,504 common training-snapshot prices, the smoother reduced vertical-spread
violations from **44 to 34**, and convexity violations from **76 to 65**. Violations
remain, so it is not certified arbitrage free and was not promoted as the pricing
engine. The comparison also uses the still-provisional candle data.

A production constrained SVI/SSVI fit needs sufficient good strikes/maturities,
proper forward inputs, explicit constraints and diagnostics. The distinction
between a convenient parameterization and an arbitrage-free calibration is
central to [Gatheral and Jacquier's SVI work](https://arxiv.org/abs/1204.0646).
That full constrained calibration has not been implemented here.

## Issue 6 — bid/ask, freshness and executable inputs

OHLC cannot uniquely reveal historical bid/ask quotes, sizes or order-book state.
Training cannot recover those facts. Historical option candles and live quotes
are different products; today's quote response does not backfill last year's book.

`quote_capture.py` now records bounded batches of active instruments with:

- Original authenticated response snapshots and receipt timestamps.
- Instrument key, feed update time and separate last-trade time.
- Best available positive-size bid/ask, size, mid, spread and age.
- Explicit missing-depth, crossed-book, stale-snapshot and clock-error flags.
- A capture manifest, including partial completion if retrieval fails.

The adapter never substitutes LTP for a missing two-sided mid. It supports the
documented [Upstox Full Market Quotes](https://upstox.com/developer/api-documentation/get-full-market-quote/)
snapshot format. A bounded REST recorder is implemented and tested; a production
continuous recorder should use the [V3 market feed](https://upstox.com/developer/api-documentation/v3/get-market-data-feed/)
with reconnect handling, decoded messages and observable feed gaps.

To complete this issue, record actual quotes through your working connection
and preserve an instrument-master snapshot with expiry, lot size and tick size.
Include the tradable hedge instrument; the cash index itself is not a fillable
hedge. Add order-size/depth, latency, partial-fill, short-margin and paired-quote
synchronization checks. Valid-looking snapshots still do not guarantee fills.
If historical quotes are unavailable, retain clearly labelled execution proxies
and stress their spreads/slippage; do not describe them as observed execution.

## Other training issues: completed work and limits

| Issue | Current evidence and action |
|---|---|
| Overfitting | Regularized learners, chronological purging, schema checks and session intervals are in place. The new validation intervals remain broad; no proof that future overfitting is eliminated. |
| Calibration | A separate calibration/decision-block experiment was already run. Sigmoid calibration did not improve decision-block log loss and was not forced into the candidate. New quantile forecasts also show residual undercoverage. More data and independent recalibration checks are needed. |
| Class imbalance | Every new target comparison reports its positive rate and a training-majority baseline. High recall, selective accuracy and profitable trades remain distinct quantities. |
| Missing information | Causal full-surface lags exist; independent daily history and quote capture are now supported. Future realized costs/fills and alternative target columns are explicitly rejected as predictors. |
| Tuning | More model/weight policies were tested, but a larger search cannot supply missing market information. The exposed validation block is development evidence, not a pristine final benchmark. |

## Reproduction and completion commands

Run from the repository root in your working API environment. Existing outputs
are protected where the tools create staged data; choose a new filename for a
new download. Do not overwrite authentic raw option files to repair spot fields.

### 1. Obtain coherent spot inputs

The simplest option uses spot candles with the same five-minute interval:

```powershell
python src/collect_spot_history.py --from-date 2025-09-03 --to-date 2026-10-26 --unit minutes --interval 5 --output data/raw/upstox/nifty_5minute_to_20261026.parquet
python src/collect_spot_history.py --from-date 2025-06-01 --to-date 2026-10-26 --unit days --interval 1 --output data/raw/upstox/nifty_daily_to_20261026.parquet
```

The collector splits minute requests by month, matching the documented retrieval
limit. Daily history begins earlier to provide HV warmup. These authenticated
downloads were not completed by this agent because of its network access block.

### 2. Expand option coverage deliberately

```powershell
python src/collect_all.py --run-mode ALL --from-expiry 2025-09-24 --to-expiry 2026-10-26 --sessions-per-expiry 20 --plan-only
python src/collect_all.py --run-mode ALL --from-expiry 2025-09-24 --to-expiry 2026-10-26 --sessions-per-expiry 20
```

Review availability/listing dates and the plan first. Older keys may require an
explicit verified expiry JSON via `--expiries-file`. Existing files are skipped;
an old five-session file is not silently expanded. Extending those existing
expiries requires a separate collection version and deliberate reconciliation.

### 3. Stage and repair the master

A staged union of the existing 33 raw files has already been written to
`data/repaired/master_union_20261027.parquet` with 519,725 rows. It is a raw union,
not a repaired dataset. If you collect additional files, build a newly named union:

```powershell
python src/master_build.py --output data/repaired/master_union_after_new_collection.parquet
python src/repair_spot_alignment.py --master data/repaired/master_union_after_new_collection.parquet --spot data/raw/upstox/nifty_5minute_to_20261026.parquet --spot-minutes 5 --output data/repaired/master_options_aligned.parquet
```

For the current union, pass `data/repaired/master_union_20261027.parquet` instead.
The alternative `--download-missing --transport project` repairs from one-minute
caches using the same project client. Supplied one-minute exports use `--spot-minutes 1`.

### 4. Activate only after backup, then rebuild every dependent calculation

```powershell
$remediationBackup = Join-Path (Resolve-Path data).Path ('backup_before_aligned_rebuild_' + (Get-Date -Format yyyyMMdd_HHmmss))
Copy-Item -LiteralPath data/processed -Destination $remediationBackup -Recurse
Copy-Item -LiteralPath data/repaired/master_options_aligned.parquet -Destination data/processed/master_options.parquet -Force
python src/build_features.py
python src/build_volatility_features.py --daily-spot data/raw/upstox/nifty_daily_to_20261026.parquet
python src/build_reference_iv.py
python src/build_deviation_features.py
python src/build_signal_eligibility.py
python src/build_parity_features.py
python src/build_convergence_labels.py
python src/build_ml_dataset.py
python src/prepare_ml_split.py
```

Do not run the default `master_build.py` immediately after activating the repaired
master: it would rebuild from the old raw joins and undo that activation. The
existing default labels remain the endpoint baseline; alternative research targets
are generated separately rather than silently changing the business definition.

### 5. Re-run independent development, then evaluate genuinely new dates

```powershell
python src/data_remediation_study.py
python src/robust_model_research.py --trials 8 --threads 4
```

Freeze the chosen pricing method, target, features, training policy, calibration
and decision rule before using a new final period. Keep matured labels and
availability times in every fold audit. Report full-sample and selective results,
coverage, class precision, calibration, costs and hedge-loss benchmarks.

### 6. Start authentic execution-data capture

Create `active_instrument_keys.txt` with one valid active instrument key per line,
including the hedge instrument, then run during the intended trading session:

```powershell
python src/quote_capture.py --keys-file active_instrument_keys.txt --output data/raw/quotes_capture_20261027 --snapshots 600 --interval-seconds 1 --transport project
```

The directory must be new. Ten minutes of snapshots is only an ingestion test,
not enough training history or a complete tick feed. Archive instrument metadata
and the market-session calendar alongside the data.

## Evidence and verification

All 52 focused offline regression tests passed in the project `.venv`, and
Python compilation passed for `src/` and `tests/`. The system default Python
does not have the project dependencies; activate `.venv` before running the
commands below or in the README. Legacy API smoke scripts are separate from
the offline regression suite and were not executed as part of these checks.

The final isolated run is
`data/processed/research/remediation_20261026_190037/` (directory timestamp is UTC).
It contains outcome reason rows, monthly coverage, corrected daily HV, feature
drift, CV policies, validation metrics, target comparisons, continuous/quantile
results, pricing diagnostics and separate candidate models. Its `run_provenance.json`
marks the data-quality status and that no production model was promoted.

The report complements `ML_REAUDIT.md`; the older report's minute-only repair
description is superseded by the valid matching-five-minute alternative here.
The previous selective 93.30% result remains a subset result, not a claim that
the entire predictor has 93–94% accuracy. The new target comparisons are different
tasks and do not retroactively improve the original endpoint benchmark.
