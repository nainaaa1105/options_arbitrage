# Earlier-period holdout for the frozen classifier

The classifier was trained and tuned on sessions from September 2025 to September 2026. To test it on data it never saw, 49 earlier weekly expiries were collected (sessions from 26 September 2024 to 2 September 2025) and run through the same pipeline stages (features, volatility, reference surface, deviation, eligibility, parity, labels, ML dataset). The frozen model bundle was then applied **once**, with no refitting, no threshold change and no feature change (`tools/older_regime_evaluate.py`; outputs in `docs/results/older_regime/`).

**What this is and is not.** The test period is earlier in time than the training period, so this checks regime transfer (a different year, different volatility levels), not forward prediction. The model, features and hyperparameters were chosen on later data, so the result says nothing about how a model trained before September 2024 would have done. Genuine forward evidence still needs the Phase 9 observation window.

## Result

| Measure | Earlier period | Later period (frozen test) |
|---|---|---|
| Rows / sessions | 111,900 / 227 | 12,466 / 28 |
| Positive rate | 38.7% | 35.5% |
| Model accuracy at 0.5 | **70.5%** (session-cluster 95% interval 69.9% to 71.1%) | 72.7% |
| Majority-class accuracy | 61.3% | 64.5% |
| Gain over majority | **+9.2 points** (interval +8.5 to +9.9) | +8.2 points |
| ROC AUC | **0.773** | 0.788 |

- The result holds in every month, from AUC 0.75 (November 2024 to January 2025) to 0.80 (July 2025), and accuracy is above the majority rate by 6 to 12 points in each month (`by_month.csv`).
- The performance is a little lower than on the later period but of the same size, which is what regime transfer without overfitting looks like. It does not support a 93-94% accuracy claim: the best honest figure remains about 70-73%.
- Accuracy here is the accuracy of the 30-minute convergence label. As shown in [CARRY_AND_PNL_CHECKS.md](CARRY_AND_PNL_CHECKS.md), predicting convergence does not make a long entry profitable after costs.
