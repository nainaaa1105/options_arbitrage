# Accuracy programme v2: result and verdict

Run: `data/research/accuracy_v2_20261031/` (protocol, CV tables, frozen manifest, final model, fresh predictions, regimes,
importances, bottleneck analysis). Code: `src/accuracy_v2/`, tests: `tests/test_accuracy_v2.py` (12). Target is unchanged: 30-minute
**endpoint** convergence. Existing datasets, baseline models and frozen experiments were not modified.

## Verdict
**93-94% overall accuracy was NOT achieved and is not attainable on this target with this data.** The best reproducible result is
**72.68% accuracy (AUC 0.788)** on the post-cutoff rows, +0.48 points over the existing 72.20%, a difference whose 95%
interval includes zero. No leakage, relabelling, majority prediction or sample exclusion was used to look better.

## Honest status of the evaluation data
All selection (features, models, hyperparameters, threshold) used history up to 15 Sep 2026 only, in 5 chronological blocks with a
one-session embargo and label-end purging. The model was hashed and frozen before the post-cutoff rows (12,466 rows, 28 sessions)
were read, and they were evaluated once. **Those dates were already reviewed while developing the baseline, so this is not a
pristine holdout.** Only sessions collected after this run can confirm any figure.

## Baseline versus improved
| | Baseline (frozen) | Improved (stacked ensemble) |
|---|---:|---:|
| Dev blocks V4-V5 AUC (mean) | 0.7620 | 0.7720 |
| Dev blocks V4-V5 accuracy at 0.5 | 71.65% | 72.07% (majority 66.2%) |
| Post-cutoff accuracy at 0.5 | 72.20% | **72.68%** (majority 64.49%) |
| Post-cutoff accuracy at frozen threshold 0.53 | n/a | 72.49% |
| Balanced accuracy | 67.86% | 67.83% |
| Precision / recall / F1 | 62.9% / 52.9% / 0.575 | 64.6% / 51.1% / 0.571 |
| ROC-AUC / PR-AUC | 0.7825 / 0.6443 | 0.7884 / 0.6605 |
| Brier / log loss | 0.1783 / 0.5207 | 0.1755 / 0.5136 |

Paired session-cluster bootstrap (improved minus baseline): accuracy +0.48 points (95% -0.13 to +1.10, not significant);
AUC +0.0059 (95% +0.0011 to +0.0109, significant but tiny). Confusion matrix at 0.5 (TN/FP/FN/TP): 6,797 / 1,242 / 2,164 / 2,263.
Class distribution: 35.5% positive post-cutoff, 36.8% in development.

## What contributed
- Chosen model: stacking (logistic meta-learner fitted only on earlier blocks' out-of-fold scores) of tuned LightGBM, XGBoost,
  CatBoost and ExtraTrees on the baseline features plus 41 new causal features. Tuned LightGBM: 284 trees, lr 0.0229, 37 leaves,
  depth 6, min child 43, lambda 6.09, colsample 0.35, subsample 0.57 (Optuna, 20 trials, blocks V1-V3 only).
- Gain by feature group (LightGBM, same hyperparameters): **deviation dynamics carry essentially all of it** (AUC 0.7695 vs 0.7613 base on V4-V5).
  Price dynamics, staleness, IV dynamics, smile, smile dynamics and cross-section groups: no gain on their own (0.7607-0.7624).
- Model families all land together (AUC 0.767-0.772). Stacking and blending add 0.0001-0.001 AUC over the best single model.
- Class balancing lowers accuracy (68.3% vs 72.0%); calibrated/thresholded variants do not help accuracy.
- Highest-importance new features: absdev_frac_of_day_max, dev_same_sign_run, absdev_mom3, dev_roll_std6, absdev_mom6.
- Impossible here: cross-expiry term structure (the collection keeps about 5 sessions per weekly expiry, so one expiry exists per timestamp).

## Why accuracy is capped
- 15.0% of labels lie within 0.25 index points of the 50%-reduction cutoff and 31.7% within 0.5 points (tick 0.05), so close to a third of
  the target is decided by price moves of the size of the spread and last-trade jitter.
- Median starting gap is 1.9 points. For gaps of 1.5 points or less (about a third of rows) only 28% converge, and accuracy there is no
  better than the majority rate plus a few points. Positive rate rises with gap size (34%, 39%, 45%, 48%, 66%).
- Different model families and about 40 added features converge to the same AUC, which points at label noise rather than model capacity.
- The data contain candle OHLC only, with no bid/ask, depth or trade-direction information, which is what would predict snap-back.
- No slice (gap size, staleness, liquidity, days to expiry, time of day) reaches 93%; best lift over majority is a few points.

## Selective accuracy versus coverage (post-cutoff, always read with coverage)
| Margin | Coverage | Accuracy | Positive calls | Positive precision | Negative precision |
|---|---:|---:|---:|---:|---:|
| 0.00 | 100% | 72.7% | 3,505 | 64.6% | 75.9% |
| 0.10 | 76.5% | 78.3% | 2,150 | 70.7% | 80.5% |
| 0.20 | 53.2% | 84.4% | 1,063 | 79.7% | 85.4% |
| 0.30 | 31.1% | 91.5% | 276 | 88.8% | 91.7% |
| 0.35 | 22.4% | 94.9% | 75 | 93.3% | 95.0% |
| 0.40 | 15.8% | 98.4% | 2 | n/a | 98.4% |
Beyond about 22% coverage the accepted rows are almost all confident "will not converge" calls. At the 0.4 margin only 1.7% of accepted rows are
positives, so the majority rate there is already 98.3% and the 98.4% is almost no lift. At the 0.35 margin the 94.9% (75 positive calls over
28 sessions) is comparable to, not better than, the earlier 93.66% at 23.7% coverage, and it is not a general entry signal.

## Evidence against leakage
Prefix-invariance tests on real data (features for earlier rows are bit-identical when all later data is removed, including a
mid-session cut); hand-checked synthetic features; row-order invariance; forbidden-name guard on every predictor; chronological
purged folds with label-end purging and a one-session embargo asserted in tests; shuffled-label AUC stays near 0.5; thresholds
and the stacking meta-learner fitted only on earlier blocks' out-of-fold scores; hashed freeze before the post-cutoff read;
deterministic LightGBM/XGBoost fits (ExtraTrees reproducible to 1e-12).

## Reproduce
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -W ignore -m accuracy_v2.experiment --output data\research\my_accuracy_run --trials 20
.\.venv\Scripts\python.exe -W ignore -m accuracy_v2.analysis --run-dir data\research\my_accuracy_run --output data\research\my_accuracy_run\analysis
```
`pip install -r requirements/accuracy-v2.txt` adds CatBoost (installed without dependencies). The saved bundle is
`final_model.joblib` (members, columns, threshold, calibrator). It is a new, separate artifact; downstream signal and hedging code still use the old frozen models.

## What could legitimately move the number
Timestamped bid/ask and depth (to price snap-back), a larger gap floor or a different target (needs your approval, and would be a
different task), and more independent sessions. None of these were done without approval.
