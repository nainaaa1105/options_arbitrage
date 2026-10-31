# Accuracy recovery and source-data investigation — 30 October 2026

## What this work can and cannot establish

The requested benchmark is 93–94% accuracy on every eligible observation from
a genuinely later period, without leakage or test-based selection. It has not
been established by the earlier project results. A high score after rejecting
most observations is a selective score and must disclose coverage, positive
precision, negative precision, counts and independent-session support.

Changing the target can create a useful new task, but cannot improve the score
of the original task retroactively. In particular, endpoint convergence,
first-passage convergence, sustained convergence and profitable trading are
different outcomes. No method can guarantee a chosen accuracy before observing
unseen outcomes. We have not established a statistical upper bound on the
predictability of this market.

## API connection: verified and repaired

The token was present in `.env`, had no outer whitespace and was not being
overridden by a different inherited token. An unverified JWT expiry inspection
did not indicate expiry; successful authenticated responses, rather than that
metadata, establish that the tested requests were authorized.

The ordinary requests certificate path initially failed TLS verification.
`http_transport.py` now supplies the operating system's trust store through a
verified SSL context. Certificate-chain and hostname checks remain enabled.
The shared authenticated client and diagnostic use this transport. No `verify=False`,
browser-signature spoofing, alternate credentials or access-control workaround
was used.

Actual authenticated responses then returned:

| Request | HTTP | Actual contents |
|---|---:|---|
| NIFTY V3 five-minute candles, 15 September | 200 | 75 valid bars |
| NIFTY V2 one-minute candles, same date | 200 | 375 valid bars |
| Expired NIFTY expiries | 200 | 109 expiry dates |

The diagnostic validates payload status, candle schema, timestamp duplication,
OHLC consistency, activity fields and expected bar grid. HTTP 200 alone is no
longer accepted as proof of success by `upstox_api.make_request`. Collection
aborts on authentication/access/rate-limit or malformed-success failures rather
than silently iterating through every contract/expiry.

The three sampled original option sessions were checked against fresh API
responses: 225 stored bars matched exactly on all OHLC, volume and OI fields.
The newest sampled response also contained two post-session bars; matching the
75 stored regular-session bars does not assert that the provider response has
only 75 records. This sample supports the stored values, not an exhaustive
verification of every historical record.

Evidence: `data/processed/research/api_diagnostic_20261030_system_trust/` and
`data/processed/research/option_api_verification_20261030/`.

Upstox documents the [historical candle interfaces](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data/)
and [expired option candle interface](https://upstox.com/developer/api-documentation/get-expired-historical-candle-data/).
The prior report's access-blocked status describes the earlier environment and
is superseded by this successful connection test.

## Candle alignment was a real coding error

A candle labelled 09:15 starts at 09:15. Its close represents the end of its
interval. The old join paired a five-minute option close with a one-minute
spot close having the same start label. Correctly matching close availability
uses either the 09:19 one-minute spot candle or the 09:15 five-minute spot candle.

Authentic five-minute NIFTY history was downloaded, and every original row was
repaired. The original option prices were preserved.

| Measurement on original 519,725 rows | Result |
|---|---:|
| Spot values changed | 515,903 |
| Median absolute spot correction | 7.40 index points |
| 95th percentile absolute correction | 30.95 points |
| Maximum absolute correction | 200.95 points |
| Rows with exact matched close availability | 519,725 |
| Common valid old/new convergence outcomes | 58,561 |
| Changed outcomes within that common valid set | 4,951 (8.45%) |

Additionally, eligibility and the one-point deviation floor changed which rows
have valid targets. The old and repaired validation populations therefore differ;
their aggregate scores are not a controlled same-row before/after comparison.
Original valid 30-minute targets numbered 67,274; repaired targets number 66,691.

Valid IV observations increased from 476,919 to 478,873. Below-lower-bound
observations fell from 42,723 to 39,397; 1,386 expiry-close observations are now
correctly marked expired because pricing uses the bar's close availability.
Remaining bound violations do not by themselves prove bad API data: stale last
trades, asynchronous observations and the pricing assumptions also matter.

The original master audit found zero duplicate contract/timestamp keys, invalid
OHLC rows, missing/nonpositive spot prices, negative activity values and
nonpositive lot sizes. It found 1,024 zero-volume rows, 24 zero-OI rows and ten
contract/sessions with fewer than 75 bars. An unchanged adjacent close occurred
on 2.80% of all rows; that is a descriptive observation, not proof of a stale
executable quote.

Evidence: `data/research/rebuild_audit_final_20261030/rebuild_audit.json`.
The earlier `rebuild_audit_20261030` diagnostic counted non-null stored labels;
the final audit correctly restricts comparison to labels valid on both versions.

## Coverage and volatility history

All missing expired weekly instruments in the requested range were collected
without replacing existing raw expiry files. There are 27 additional expiry
files: 21 in the old coverage gap and six after the prior September endpoint.

| Dataset | Rows | Expiries | Observed sessions |
|---|---:|---:|---:|
| Original master | 519,725 | 33 | 156 |
| Added records | 422,224 | 27 | 127 distinct added dates |
| Combined synchronized master | 941,949 | 60 | 283 |

The combined dataset has 3,555 contracts. All 941,949 rows have synchronized spot
closes after repair. Per-expiry manifests expose requested/returned contracts,
failed requests and interval/alignment provenance; a manifest's partial-contract
flag must still be inspected before assuming every intended contract traded.

Complete downloaded daily NIFTY history, 351 records from June 2025 through
29 October 2026, now supplies HV. The missing option months are no longer treated
as consecutive observed daily closes. HV continues to use completed prior-day
returns and 252-session annualization. ACT/365 is retained for expiry time.

Also downloaded: authentic five-minute spot and India VIX history. Market
feature experiments use exact timestamp matches, completed bars, no future
fill, return-gap checks and within-session rolling volatility. Prefix-invariance
tests ensure future bars cannot change earlier feature values.

## Model development and fair evaluation

There are 119,940 usable ML rows across the combined data. The historical
development dataset ends on 15 September and contains 107,474 rows. Its
chronological split is 79,792 training, 13,745 validation and 13,937 historical
test rows. The later period contains 12,466 endpoint candidates. It is excluded
from hyperparameter, feature-policy and threshold selection.

The work includes:

1. Recomputed IV, Greeks, HV, leave-one-out reference IV, theoretical prices,
   eligibility, parity and convergence labels from synchronized observations.
2. Twenty Optuna trials per XGBoost and LightGBM family on internal purged
   chronological folds. Random Forest, logistic and equal-ensemble comparisons
   use the same internal protocol. The equal ensemble won the original
   repaired-data internal CV comparison.
3. Separate calibration and decision blocks; preprocessing fitted on fitting
   observations only; purging includes future candle close availability.
4. Eight historical weighting/window policies, including recency weighting and
   removal of absolute index-level inputs. These are selected using training
   folds, not fresh outcomes.
5. A final predeclared market-input comparison: the frozen ensemble versus
   regularized LightGBM with spot, VIX, or both input groups. Selection uses
   expanded-history training folds only. All compared profiles are reported.
6. Endpoint, first-passage and sustained-two-bar targets on complete observed
   paths. Missing outcomes remain missing. Alternative targets are reported
   separately and never relabelled as original-target improvements.
7. Continuous deviation and quantile forecasts, majority/prior baselines,
   calibration diagnostics, session-level uncertainty and selective coverage.

After the design is frozen, the fresh evaluator refits on completed history
through 15 September and evaluates later observations once. It saves model
hashes and the frozen configuration before opening the fresh labelled dataset.
The predeclared cutoff is 0.50. Selective margins are fixed beforehand and are
diagnostics, not a license to choose the best fresh-period margin afterward.

Training-period performance is reported separately from fresh performance.
Chronological validation reduces leakage; it cannot prove absence of all
overfitting or establish stability across every future market regime.

## Remaining causes and corrections

| Cause | Current status / correction |
|---|---|
| Connection/certificate failure | Fixed and verified with real authenticated responses. |
| Spot/option timing mismatch | Repaired in staged original and expanded datasets; all dependent quantities rebuilt. |
| Missing option-history months | Available expiries backfilled; partial-contract manifests remain observable. |
| Corrupted HV across coverage gap | Independent daily history replaces sparse option-date history. |
| Earlier time-feature sorting and expiry-feature errors | Corrected feature schema retained; new fits use it. Legacy weights were preserved. |
| Near-session-end missing outcomes | Eligibility/censoring is explicit; missing outcomes are not fabricated negatives. |
| Class imbalance and prior drift | Majority baselines, PR-AUC, specificity, balanced accuracy, calibration and chronological CV reported. |
| Limited explicit market-regime inputs | Spot/VIX input experiment uses historical folds; no fresh-period feature selection. |
| High correlation among observations | Whole-session splits/purges and session bootstrap; 941,949 rows are not that many independent market regimes. |
| Last-trade candle prices versus executable prices | Still unresolved historically. Need timestamped bid/ask and depth/fill evidence. |
| Simplified carry/rates/reference surface | Forward/robust-surface research exists, but this run preserves the original target methodology for comparability. No arbitrage-free or profitability claim. |
| Threshold-selected apparent accuracy | Full coverage and selective scores remain separate; both class precisions and coverage required. |
| Repeatedly inspected old holdout | Kept as historical diagnostics. Later collected dates supply the fresh evaluation. |

The original gate requiring at least 0.25 calendar days to expiry excludes most
of expiry-day trading after the opening bars. This is a policy choice that
restricts applicability; changing it changes the eligible population and needs
its own frozen evaluation. Regular-session horizon rules also require explicit
exchange-calendar treatment before supporting special evening sessions.

## What would justify a useful 93–94% claim

For full coverage, the model would need to achieve that accuracy on later data
under the frozen original target, with nontrivial minority-class performance,
adequate session support and an uncertainty interval. The evidence collected
here does not make a chosen threshold or a larger search budget a guarantee.

A selective system can abstain when uncertain, but must publish coverage and
accuracy together. Positive convergence precision must be checked separately;
many correct non-convergence rejections do not establish a high-quality trading
entry engine. A target or policy chosen after seeing these fresh results needs
another new period for confirmation.

The highest-value missing information for the trading objective is timestamped
option bid/ask, depth, synchronized executable hedge prices and actual costs.
Upstox's [full quote interface](https://upstox.com/developer/api-documentation/get-full-market-quote/)
provides snapshots with depth; these are different inputs from historical OHLC.
The recorder is available, but historical quote data cannot be reconstructed
from the downloaded candles. Collect across multiple regimes, rather than
training on a few minutes of snapshots.

For deep hedging, use expected net hedged P&L and downside/risk forecasts as
primary economic outputs, with convergence as an auxiliary feature or target.
The [Deep Hedging paper](https://arxiv.org/abs/1802.03042) explicitly addresses
hedging under transaction costs and other frictions. This classification
experiment does not establish improved net P&L, Sharpe or drawdown.

## Reproducibility and artifacts

All earlier active processed datasets and model weights were preserved during
research. New raw files are additive. Staged model bundles and predictions must
be used with their matching feature builders, regime inputs and data context;
do not copy one weight file into the old active pipeline without those inputs.

- API verification: `data/processed/research/api_diagnostic_20261030_system_trust/`.
- Option replay check: `data/processed/research/option_api_verification_20261030/`.
- Repaired original-history run: `data/research/aligned_20261030/`.
- Added-expiry rebuild: `data/research/added_20261030/`.
- Expanded-history run: `data/research/expanded_20261030/`.
- Feature study: `data/research/expanded_20261030/regime_study/`.
- Source snapshots, input hashes, logs and resume history accompany these runs.

Scalar and batch pricing were compared on sampled real observations and
invalid-input cases. Reference-IV optimization matched the previous calculation
on 100 sampled complete surface groups, including method labels. The optimized
implementations avoid unnecessarily repeating scalar calculations and retaining
wide copies of every surface group; they do not introduce a new pricing model.

## Actual fresh-period results

Evaluation completed on 12,466 observations across 28 sessions, using the fixed
0.50 cutoff. The selected design is regularized LightGBM with spot/VIX inputs
and the historical CV-selected relative-input policy. There are six later
expiry cycles; session bootstrap intervals do not establish independence across
cycles or stability across every future regime.

| Target | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Balanced accuracy | Brier | Log loss | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Original endpoint | 72.20% | 62.91% | 52.88% | 0.5746 | 0.7825 | 0.6443 | 67.86% | 0.1783 | 0.5207 | 0.0185 |
| First passage | 86.53% | 85.57% | 97.60% | 0.9119 | 0.9144 | 0.9582 | 78.24% | 0.0960 | 0.3088 | 0.0198 |
| Sustained two bars | 75.71% | 70.79% | 73.78% | 0.7226 | 0.8433 | 0.7918 | 75.47% | 0.1604 | 0.4766 | 0.0209 |

Training-majority baseline accuracy is respectively 64.49%, 71.42% and 57.13%.
These figures are not trade win rates. First passage crosses 0.90 ROC-AUC, but
does not reach 93–94% full-coverage accuracy. Its high PR-AUC is interpreted
against its 71.42% positive prevalence, not against a balanced-class baseline.

Session-bootstrap 95% intervals:

- Endpoint: accuracy 70.86–73.89%, ROC-AUC 0.7683–0.7952.
- First passage: accuracy 83.93–88.70%, ROC-AUC 0.9004–0.9290.
- Sustained: accuracy 74.36–77.41%, ROC-AUC 0.8300–0.8596.

Endpoint training accuracy was 72.45%, versus 72.20% fresh accuracy. This small
gap is reassuring for this run, but is not proof that overfitting is impossible.
First-passage training accuracy was 88.61%, versus 86.53% fresh accuracy.

### Selective diagnostics: a possible route, with explicit limits

The full predeclared margin grid is saved. Selected rows below illustrate
different coverage/accuracy tradeoffs, not policies selected on training data
that have now been confirmed independently.

| Target | Fixed probability bands | Coverage | Accepted accuracy | Positive precision | Positive decisions |
|---|---|---:|---:|---:|---:|
| Endpoint | ≤0.20 or ≥0.80 | 23.66% | 93.66% | 87.10% | 93 |
| First passage | ≤0.20 or ≥0.80 | 76.98% | 93.47% | 92.32% | 7,878 |
| First passage | ≤0.10 or ≥0.90 | 60.99% | 96.36% | 95.60% | 6,021 |
| Sustained two bars | ≤0.10 or ≥0.90 | 18.51% | 98.35% | 94.66% | 337 |

For first passage at the fixed 0.10/0.90 bands, negative precision is 99.24%.
That is a promising selective classifier on a different target. It abstains on
39.01% of rows, and accepted rows include both positive entries and negative
rejections. The 6,021 positive decisions are overlapping option observations,
not 6,021 independent executable trades.

Those positive decisions capture 5,756 of the period's 8,903 true first-passage
outcomes (64.65%). Abstention therefore has a material opportunity cost. The
10,000-draw trading-session bootstrap gives 95% intervals of 95.42–97.23% for
accepted accuracy and 94.39–96.70% for positive precision. Even the exploratory
Bonferroni-adjusted percentile lower bounds across the 60 reported diagnostic
fields are 94.75% and 93.51%, respectively. This evidence remains conditional on
this period, six expiry cycles and the bootstrap assumptions; it is not a
finite-sample or future-regime guarantee.

The original endpoint's 99.40% selective score at 13.45% coverage issued **zero
positive decisions**; it is not a useful high-accuracy convergence entry engine.
This is why overall selective accuracy alone is insufficient.

All these bands were predeclared, but choosing a deployment policy from the
fresh results makes this period development evidence for that choice. Freeze
the chosen target/bands now and confirm them on a new later period; do not keep
adjusting them against these outcomes. The separate selective uncertainty
report covers every grid point and includes an exploratory multiplicity
adjustment. None of these observations establishes a future accuracy guarantee
or profitable trading after costs.

Fresh results: `data/research/expanded_20261030/fresh_evaluation/fresh_metrics.csv`.
Predictions, models, the pre-evaluation freeze file, fitting metrics, session
intervals and the complete selective grid are alongside it. All 59 focused
regression tests passed, and source/test compilation passed.

The old active model and processed artifacts remain preserved. The new run is
self-contained research output; its fitted bundles include feature columns,
schema/profile metadata and matching source-data paths. This avoids silently
running corrected weights against the old context or relabelling earlier
performance reports as newly confirmed results.

## Run the new frozen models

From the repository root, using feature rows constructed from the matching run:

```powershell
.\.venv\Scripts\python.exe src/predict_research_model.py --run-dir data/research/expanded_20261030 --input data/research/expanded_20261030/data/processed/fresh_ml_dataset.parquet --target first_passage --output data/research/expanded_20261030/my_first_passage_predictions.parquet
```

Use `--target endpoint` for the original task, or `--target sustained_two_bars`
for the sustained alternative. The output path must be new. The helper reads
the fitted feature list and the matching spot/VIX/daily-HV/context inputs; the
ordinary prediction output uses 0.50 and does not silently activate a selective
trading policy. Refresh the causal market context before scoring later dates.

A 250-row replay removed all nine outcome/future columns and reproduced the
saved first-passage probabilities exactly (maximum absolute error 0). This
checks that inference does not require future target columns.

`research_policy_candidate.json` records the proposed first-passage 0.10/0.90
policy as **selected after reviewing fresh diagnostics**, for subsequent
confirmation. It does not modify the old active classifier or place orders.
