# Results summary

One page with the headline numbers of every research track. Each row names the report that holds the detail; the older dated
reports are in [archive/](archive/README.md). All numbers are retrospective until the forward test runs
([PHASE9_FORWARD_VALIDATION.md](PHASE9_FORWARD_VALIDATION.md)). Nothing here establishes a profitable strategy.

## 1. Data and the timing defect
| Item | Result |
|---|---|
| Dataset | 941,949 five-minute option candles, 60 weekly expiries, 283 sessions (Sep 2025 to Oct 2026); spot, futures, India VIX |
| Defect found | Five-minute option closes were paired with the one-minute spot candle that shares the same *start* label, not the one that closes at the same instant |
| Size of the error | 515,903 of 519,725 spot values changed (median 7.4 index points, 95th percentile 31); 4,951 of 58,561 comparable labels (8.45%) flipped |
| Response | Aligned join by close time, full rebuild of features, labels and models in an isolated run ([archive/ML_REAUDIT.md](archive/ML_REAUDIT.md), [archive/ACCURACY_RECOVERY_20261030.md](archive/ACCURACY_RECOVERY_20261030.md)) |

## 2. Convergence prediction (will a pricing gap halve within 30 minutes?)
| Target | Accuracy on later rows | Majority baseline | ROC-AUC |
|---|---:|---:|---:|
| Endpoint (primary) | **72.68%** (earlier model 72.20%) | 64.49% | 0.788 |
| First passage (easier, 71% positives) | 86.53% | 71.42% | 0.914 |
| Two-bar sustained | 75.71% | 57.13% | 0.843 |

The endpoint gain over the earlier model is +0.48 points (95% interval -0.13 to +1.10). A 93-94% target was investigated and ruled out:
15% of labels sit within 0.25 index points of the cut-off and 32% within 0.5. Selective accuracy of 94.9% exists only at 22% coverage with 75
positive calls. Detail: [ACCURACY_V2_REPORT.md](ACCURACY_V2_REPORT.md).

## 3. Entry selection and hedging on real data
| Question | Result |
|---|---|
| Do classifier-chosen entries converge more often? | Yes: 67.1% vs 47.5% random and 50.3% largest-gap (760 paired entries) |
| Does that become profit? | No detectable difference vs random after fees; dynamic-delta total -2.15M over 760 episodes ([PHASE8_SIGNAL_HANDOFF.md](PHASE8_SIGNAL_HANDOFF.md)) |
| Does hedging pay for itself? (168 positions, 28 sessions) | Unhedged +81k with worst-5% loss 11.1k; dynamic delta -345k with 4.6k. Tail risk falls by 59%, fees exceed the gain ([archive/PART3_HISTORICAL_ML_HEDGING.md](archive/PART3_HISTORICAL_ML_HEDGING.md)) |
| Does the classifier help hedging decisions? | No: identical P&L with and without it; valid probabilities on only 13 of 1,008 decisions |

## 4. Neural hedging in simulation
| Track | Result |
|---|---|
| Whole-lot policies on GBM/Heston ([archive/PART2_DEEP_HEDGING.md](archive/PART2_DEEP_HEDGING.md)) | Cut worst-case loss ~75-78% vs unhedged but do not beat delta baselines |
| Continuous policies, cost grid ([../RESEARCH_PHASES_4_7_5.md](../RESEARCH_PHASES_4_7_5.md)) | Loss CVaR95 0.00993 vs delta 0.01181 (-15.9%) at cost 0.001; a tuned no-trade band has the better mean P&L |
| Rounding to whole lots | A one-lot book carries about 5x the continuous CVaR; the rounded conventional hedge gets within about 1% of continuous at 40 lots, while the trained whole-lot policy stays about 17% above and did not consistently beat rounding |
| Robustness | Neural vs delta reduction: 12.8% (GBM), 6.2% (Heston), 9.1% (bootstrap); replacing observed variance with a trailing estimate raises Heston CVaR95 about 10% |
| GRU, ablations, ensembles ([RESEARCH_PHASE_6.md](RESEARCH_PHASE_6.md)) | GRU 1.8% worse than the MLP (interval 0.00013 to 0.00024); ensembles add nothing |

## 5. Paper trading and validation machinery
| Item | Status |
|---|---|
| Full-session replay, 20 accounts, real candles with synthetic quotes | 840 events, every account reconciles, nothing left open |
| Tamper detection | Editing the frozen protocol is rejected; one altered fill is caught as a 130-rupee error |
| Tests | 293 offline tests, network blocked ([OPERATIONS_PHASE_0B.md](OPERATIONS_PHASE_0B.md)) |
| Forward test | Protocol drafted and analysed ([PROTOCOL_SENSITIVITY.md](PROTOCOL_SENSITIVITY.md)); **not frozen, not started** |
