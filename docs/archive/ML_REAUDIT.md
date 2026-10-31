# ML re-audit — 26 October 2026

The re-audit found defects missed by the earlier checks. Corrected feature code,
collector alignment, reproducible research and regression tests are now present.
Existing historical pricing/label/model results remain provisional until the
spot-close alignment is repaired using authentic one-minute NIFTY candles.

## Findings and corrections

| Finding | Correction and verification |
|---|---|
| Session features used a timestamp Series captured before sorting the rows. | Schema 2 derives time features from the sorted rows. 49,561 of 50,278 training rows had mismatched session features under the legacy code. Regression tests check input-order and batch-size invariance. |
| A substring filter excluded every name containing `expiry`, including four valid predictors. | Raw expiry metadata is excluded explicitly while `time_to_expiry`, `days_to_expiry`, `log_days_to_expiry` and `is_near_expiry` are retained. |
| Expiry engineering multiplied ACT/365 year fractions by 252. | Schema 2 uses 365, and zero-day expiry is determined by the actual expiry date. |
| Five-minute option closes were paired with the one-minute spot close sharing the candle-start timestamp. | `candle_alignment.py` joins exact close availability times. The 09:15 option bar matches the 09:19 spot minute; both close at 09:20. Missing matching minutes remain missing. Collection now caches authentic minute candles. |
| Expiry time for newly aligned rows was calculated at candle start. | `time_to_expiry.py` uses `feature_available_at` when that provenance is present. |
| The split-size assertion failed whenever purging removed any rows. | The accounting assertion includes the purged-row count. |
| Constant-class baseline report rows reused model probabilities and therefore displayed the model's AUC and Brier score. | Baseline rows now use their own constant scores. The existing baseline CSV was corrected, with its previous version preserved under the research output directory. |
| Label code did not explicitly enforce same-session and finite-deviation validity. | A pure, tested label function now does. Recomputing 30- and 60-minute endpoints over all 519,725 rows found zero differences from the stored future timestamps, validity flags and targets. |
| Context features could be silently imputed when an input row was absent from the context table. | Schema 2 rejects missing context keys. Context caching also refreshes when its source file changes. |
| LightGBM `subsample` could have no effect with the default sampling frequency. | New fits enable sampling frequency when subsampling is requested. |

Upstox documents its timestamps as the **start** of the candle timeframe:
[Historical Candle Data](https://upstox.com/developer/api-documentation/get-historical-candle-data/).

The target remains **endpoint convergence at 30–35 minutes**, not first passage
at any moment within 30 minutes. It was not changed to make the classification
task easier. The same distinction applies to the 60-minute endpoint.

## Experiment and results

Run directory: `data/processed/research/audit_20261026_172925/`.

- Eight Optuna trials each for XGBoost and LightGBM, plus Random Forest,
  Logistic Regression and an equal-weight boosted-tree ensemble.
- Tuning used three expanding folds with one complete session of gap plus label
  purging, inside the earliest 76 training sessions.
- A further 16 sessions fitted sigmoid calibration. Another 17 sessions selected
  calibration and decision thresholds. The 23-session validation block was then
  scored after those decisions were frozen.
- Selection criterion was mean internal ROC-AUC minus 0.25 times its fold SD.
  The ensemble won with mean 0.7296 and SD 0.0021.
- A predeclared raw comparison refitted that ensemble on all 109 training
  sessions. Calibration was not selected because it failed to improve decision
  block log loss. No test-split file was read by this experiment.

| Validation metric | Full-training raw ensemble, fixed 0.50 | Separate calibration/decision experiment |
|---|---:|---:|
| Accuracy | 69.73% | 68.47% |
| Precision | 58.18% | 56.50% |
| Recall | 55.43% | 52.54% |
| F1 | 0.5677 | 0.5444 |
| ROC-AUC | 0.7530 | 0.7372 |
| PR-AUC | 0.6032 | 0.5701 |
| Balanced accuracy | 66.58% | 64.96% |
| Brier score | 0.1898 | 0.1953 |
| Log loss | 0.5479 | 0.5602 |

Session-bootstrap 95% intervals for the full-training candidate were
67.68–71.64% accuracy and 0.7260–0.7805 ROC-AUC. These are development results:
the validation period has been reviewed before, and the source spot alignment
still requires repair. They do not establish a pristine final performance score.

## What the 93–94% benchmark means here

Predeclared fixed bands were evaluated, with coverage and each predicted class
reported independently. These were not selected using validation labels.

| Accepted probabilities | Coverage | Accepted rows | Selective accuracy | Positive predictions | Positive precision |
|---|---:|---:|---:|---:|---:|
| ≤0.20 or ≥0.80 | 19.89% | 1,552 | 93.30% | 108 | 77.78% |
| ≤0.10 or ≥0.90 | 12.51% | 976 | 99.18% | 5 | 60.00% |

The first result is 93.30% on a subset, not 93.30% on all options. Its selective
accuracy session-bootstrap interval is 90.17–95.41%; the positive-precision
interval is 62.75–92.63%. High selective accuracy is largely driven by confident
non-convergence predictions. No decision-block policy qualified for 93% or 94%
precision for **both** predicted classes with the required sample/session support.

## Historical data repair blocker

The full original minute series is absent from the saved raw dataset. Secure
read-only requests to both V2 and V3 returned HTTP 403; the V3 response states
that the site owner blocked the client's browser signature. This is an access
block, not evidence of an expired token. No certificate checking was
disabled, and no historical spot values were fabricated or interpolated.

When authorized historical-data access is restored, run:

```powershell
python src/repair_spot_alignment.py --spot data/raw/upstox/nifty_1minute --download-missing --output data/repaired/master_options.parquet
```

The downloader resumes cached days. The repair refuses to overwrite an output
or accept missing exact minute closes. After reviewing the separate repaired
master, back up active processed outputs, activate the repaired master and rebuild
pricing, IV/Greeks, volatility, reference pricing, labels, splits and models in the
README dependency order. Do not mix corrected spots with old derived labels.

## Saved-model compatibility

Feature schema 2 is the corrected training default. Existing saved artifacts
without a schema version continue to use schema 1 through the signal generator,
evaluation utility and classifier entry point. Their validation predictions were
verified unchanged: accuracy 0.6990900936 and ROC-AUC 0.7528678612 at cutoff 0.50.
The new research candidates are separate artifacts, tagged schema 2 and marked
with the unresolved data-quality status. They were not promoted automatically.

Reproduce the research with:

```powershell
python src/robust_model_research.py --trials 8 --threads 4
```

This creates a new run directory containing audit, trials, folds, regime metrics,
predictions, session confidence intervals, selective coverage and candidate models.

Verification: all 34 focused regression tests pass and Python compilation passes.
The repair CLI help and active saved-model compatibility checks also passed.

27 October update: [DATA_MODEL_REMEDIATION_REPORT.md](DATA_MODEL_REMEDIATION_REPORT.md)
adds the measured coverage/censoring audit, alternative target experiments,
drift policies, quote ingestion and staged recovery commands. Authentic matching
five-minute spot bars also repair alignment; one-minute data are not mandatory.
The new candidates are provisional and do not replace the active saved model.
