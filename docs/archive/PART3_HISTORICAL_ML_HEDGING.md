# Part 3 — historical ML-assisted neural hedging

Implemented 31 October 2026, Asia/Kolkata. The authoritative experiment is
`data/research/historical_hedging_part3_final_20261031/`.

## Purpose and explanation

Part 1 established historical NIFTY option/futures episodes and conventional
hedging baselines. Part 2 trained whole-lot neural hedgers on stylized GBM and
Heston simulations. Part 3 transfers those policies to historical observations,
adds a small historical adaptation, and tests whether the existing convergence
classifier contributes useful information to hedging.

The convergence classifier answers whether a pricing deviation will shrink.
The hedge policy answers how many futures lots to hold against an existing
option. Those are different decisions. A convergence probability is an input
to the hedge policy; it is not a probability of a profitable hedge, a directional
price forecast, or an instruction to buy an option.

All eight strategies receive the same fixed long-option episodes. The experiment
does not select favorable option entries or reject low-confidence positions.
This isolates the value of the hedge and the incremental ML information.

## Implementation and actual workflow

1. Verify the Part 1 input hashes and load its exact option/futures episodes.
   Keep the four-option-lot long exposure, actual contract lot/tick metadata,
   30-minute horizon, five-minute decisions and INR 5,000,000 initial capital.
2. Join the immediately completed option candle to each execution boundary.
   Use observed spot, option close, futures close, IV, delta and maturity.
   Preserve ACT/365 maturity paired with its IV in standardized moneyness;
   normalize the calendar-day feature explicitly. No historical option prices
   or Greeks are recalculated from the simulation pricing model.
3. Reuse the established endpoint-classifier architecture and feature schema.
   Clone and refit it on expanding past data, after a 60-session burn-in and
   in 20-session blocks. Both the predictor timestamp and its label's availability
   must precede the prediction block. Keep the final classifier fixed from
   16 September 2026 through the end of the experiment.
4. Generate probabilities only within the original training domain: current
   signal eligibility, at least one point of absolute deviation, and a valid
   remaining session horizon. A query does not require an observed future label.
   Outside that domain use probability 0.5 and an explicit unavailable flag;
   retain the position and its realized outcome.
5. Freeze the Heston network. Train only an 85-parameter linear residual over
   the 14 inventory/market features plus probability and availability. Compare
   the full version with a matched version masking the two ML inputs to zero.
   Both have the same architecture, optimizer, training budgets and seed set.
6. Fit the residuals on episodes before 1 July 2026. Select the epoch and seed
   using validation episode loss CVaR over 1 July–15 September only. Include
   epoch zero as a fallback. Use seeds 11/22/33, eight epochs, 16 updates/epoch,
   128 sampled training episodes/update, learning rate 0.0003 and an L2 anchor.
   Optimize discrete whole-lot actions using the CVaR score-function objective,
   learned VaR threshold and an action-independent band-delta control variate.
7. Write the frozen checkpoint manifest, then run the later historical test.
   Compare unhedged, static delta, dynamic delta, band delta, frozen GBM,
   frozen Heston, adapted without ML, and adapted with ML.
8. Produce sequential ledgers, daily equity including zero-trade sessions,
   metrics, paired session-bootstrap intervals, monthly aggregates, a doubled
   execution-friction sensitivity and independent checkpoint replay.

Training minibatches reset capital for independent risk episodes; reported
historical portfolios compound their own sequential cash balances. All strategies
share a conservative admission check based on the lowest remaining capital,
so a capital rejection applies to every strategy. Entry, hedge action capacity,
costs, margin rules and liquidation conventions are matched.

## Causality and accounting checks

The opening prices used for simulated fills do not enter policy observations.
At the initial boundary, option funding is estimated from the known close and
configured adverse execution model. Subsequent account features mark inventory
at completed closes. The actual opening fill affects realized accounting and
execution-time solvency constraints, then becomes known for later decisions.

Historical STT uses each episode's date, including the existing engine's
1 April 2026 schedule change. Futures notional is not purchased or sold as a
cash security. Realized futures P&L, option premium, fees, impact and funding
reconcile to the ending cash account. Positive unrealized futures gains are
not assumed immediately withdrawable.

There are 21 missing IV/delta path rows in the original full Part 1 universe.
They are retained. Features can carry forward only earlier valid pricing state
within the same episode; an invalid current state holds the existing hedge.
There is no backward fill, future interpolation or favorable-outcome deletion.
An unavailable initial state fails closed rather than borrowing a later state.

## Results and interpretation

See the authoritative run's `RESULTS.md`, `test/strategy_comparison.csv` and
`test/paired_comparisons.csv` for all measured values. The test contains
168 common episodes over 28 observed sessions, starting 16 September 2026.
Training has 846 episodes and validation has 324 episodes after burn-in.

The original nearest-ATM sampling universe seldom satisfies the convergence
target's minimum-deviation rule. Only 13 of 1,008 nonterminal test decisions
have an eligible ML probability (1.29%). This is a substantive coverage limit,
not a reason to manufacture probabilities outside the classifier's domain.
`secondary_evaluation/coverage_and_transfer_audit.csv` measures this coverage,
invalid-state holds, and states outside the nominal simulation training range.

The conclusions must distinguish reduction in tail risk, incremental ML value,
and aggregate net profitability. Similar or identical ML/no-ML outcomes do not
establish a benefit from the classifier. Positive unhedged P&L on this short
sample does not establish that leaving an option unhedged is a superior future
risk strategy. Costs can overwhelm the benefits of frequent futures hedging.

The final corrected run produced INR -346,647.44 net P&L for both adapted
policies, versus INR -345,002.28 for dynamic delta. Their episode loss CVaR
was INR 4,568.57, versus INR 4,555.48 for dynamic delta and INR 11,096.73
unhedged. Incremental ML benefit and neural superiority were not established.

All 124 regression tests passed, including 18 Part 3 tests. Compilation and
frozen artifact hashes passed verification. All four neural policies matched
the scalar accounting replay exactly. See the final run's `verification.json`.

The full report includes gross/net P&L, return, win rate, episode standard
deviation, loss VaR/CVaR, worst episode, daily Sharpe/Sortino, daily and intraday
drawdown, costs, turnover, margin, forced exits, writeoffs, delta exposure and
reconciliation error. Annualized daily ratios based on 28 sessions are unstable
descriptive statistics, not long-run performance estimates. Approximately nine
test episodes determine the 95% tail; confidence intervals remain important.

## Research status and limitations

This is retrospective chronological research. The market dates and previous
classification/backtest results have already been inspected. The classifier
architecture was selected earlier in project development; expanding-window fits
do not undo that historical design selection. Model fits and hedge selection
exclude later labels, but this is not a pristine prospective holdout.

The first Part 3 run is preserved at
`data/research/historical_hedging_part3_20261031/`. A final audit found that its
new transfer adapter changed maturity units without preserving IV-squared times
maturity. The final run corrects that normalization, adds an explicit regression
test, and repeats the same architecture, seeds, budgets and split protocol.
The original run is diagnostic only. No claim of an untouched test set follows
from rerunning the correction.

The simulation models were not calibrated to the historical NIFTY volatility
surface or futures basis. Historical maturity and volatility may lie outside
the simulation training range. Observed futures/spot sensitivity remains the
Part 1 deterministic-carry approximation. No quote depth, partial fills, market
impact calibration, historical SPAN portfolio margins, broker fills or live
orders are available in this experiment. Fees/spreads/margin are the explicitly
configured research proxies; the friction sensitivity is hypothetical.

Before a deployment-quality claim, Part 4 should establish forward paper trading,
quote-based execution assumptions, operational monitoring and a genuinely later
evaluation period. Changing the option entry universe to match the convergence
target would be a separate, predeclared strategy experiment; all candidate
hedgers must still receive the same positions in that experiment.

## Deliverables

| Deliverable | Location |
|---|---|
| Walk-forward classifier and causal states | `src/deep_hedging/historical_data.py` |
| Frozen transfer and ML/no-ML residual policies | `src/deep_hedging/historical_policy.py` |
| CVaR historical adaptation and learning curves | `src/deep_hedging/historical_train.py`; per-ablation output directories |
| Common-account comparisons and uncertainty | `src/deep_hedging/historical_evaluate.py` |
| Reproducible experiment | `src/run_historical_hedging.py`; `experiment_plan.json`, source snapshot and frozen manifest |
| Standalone saved-policy replay | `src/predict_historical_hedge.py` |
| Frozen-policy stress and coverage diagnostics | `src/evaluate_historical_hedging.py` |
| Trade and equity audit trails | `test/transaction_ledger.parquet`, `marked_equity.parquet`, `episode_results.parquet`, `daily_equity.csv` |
| Metrics and paired confidence intervals | `test/strategy_comparison.csv`, `test/paired_comparisons.csv` |
| Automated safety/regression checks | `tests/test_historical_hedging_part3.py` |

## Commands

Run from the repository root using the existing project environment and the
optional CPU PyTorch dependency installed in Part 2. Use new output directories.

```powershell
.\.venv\Scripts\python.exe src/run_historical_hedging.py --output data/research/my_part3_run

.\.venv\Scripts\python.exe src/predict_historical_hedge.py --run-dir data/research/my_part3_run --output data/research/my_part3_replay

.\.venv\Scripts\python.exe src/evaluate_historical_hedging.py --run-dir data/research/my_part3_run --output data/research/my_part3_run/secondary_evaluation

.\.venv\Scripts\python.exe -m unittest tests.test_historical_hedging_part3 tests.test_deep_hedging_part2 tests.test_hedging_part1 -v
```

The risk-aware, cost-constrained objective is motivated by the original
[Deep Hedging paper](https://arxiv.org/abs/1802.03042). This project's discrete
policy-gradient implementation and historical results are its own experiment;
the paper's reported superiority is not evidence of superiority here.
