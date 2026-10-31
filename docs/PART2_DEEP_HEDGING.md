# Part 2 — PyTorch deep hedging on GBM and Heston simulations

Implemented 31 October 2026, Asia/Kolkata.

## Outcome and scope

Part 2 is implemented, trained and evaluated. It adds a neural futures-hedging
policy, model-consistent simulated option markets, CVaR optimization, independent
simulation test banks and stress comparisons. Existing historical/ML artifacts
are preserved.

The measured outcome is important: **the selected neural policies reduce tail
loss relative to an unhedged option, but do not outperform the strong delta
baselines on the primary test scenarios**. The implementation is complete;
superior neural hedging performance is not established. CVaR updates also failed
to improve on the behavior-cloning warm start in this training budget.

These are stylized, uncalibrated simulations, not historical NIFTY performance,
live returns or an integration of the earlier convergence classifier. Part 3
must supply the historical comparison and walk-forward ML inputs.

## What the policy learns

Each episode starts with a fixed four-lot option position. Every five minutes,
the policy chooses the magnitude of its futures hedge from zero to four whole
lots. Hedge direction offsets the option's directional exposure. It cannot
select a different option or avoid the option liability to manufacture a lower
risk score. Both positions are closed at the end of 30 minutes.

The neural policy is a two-hidden-layer, 32-unit tanh network implemented in
PyTorch. Its 14 causal state inputs describe moneyness, maturity, current
volatility, option delta/type, time remaining, current hedge inventory,
delta target, the latest observed return, free cash, margin, transaction costs,
contract units and accumulated portfolio P&L. They use deterministic scaling,
not normalization fitted on validation or test paths.

The logits have a disclosed delta-informed prior. Training begins with behavior
cloning of the no-trade-band baseline, then uses a categorical score-function
policy gradient. This is not a claim that the network learns pricing/hedging
from raw prices without Greek information.

Actions are discrete during training. There is no fractional-contract training
result subsequently presented as executable whole-lot performance. Margin
projection and forced liquidation use the same rules as Part 1.

## CVaR objective

Let net episode P&L be Pi and loss be L = -Pi. Training minimizes:

`CVaR_alpha(L) = min_eta [eta + E(max(L - eta, 0)) / (1-alpha)]`, with alpha=0.95.

This targets the worst five-percent loss tail. It is not classification accuracy
or a profit-maximization objective. Loss includes fees, execution impact and
terminal hedge liquidation.

The final trainer learns the VaR threshold eta jointly with the policy. Since
fills, discrete orders and fees are not differentiable, actor gradients use the
sampled categorical action log-probabilities. A fixed baseline hedge on the
same simulated path supplies an action-independent control variate. Future
prices enter the training reward/control-variate calculation, not the actor's
observations. Prefix-invariance tests check the distinction.

Training samples actions; evaluation uses the deterministic argmax policy.
Validation uses that same deterministic evaluation rule. The gap between
stochastic optimization and deterministic execution remains a limitation.

## Simulated markets and option valuation

| Market | Path generation | Option values and delta |
|---|---|---|
| GBM | Log-Euler/exact constant-volatility increments, zero drift | Black–Scholes, same variance and clock |
| Heston | Correlated spot/variance shocks, full-truncation Euler with one-minute internal steps | Heston Fourier pricing and model-consistent spot delta |

Rates and carry are zero, so simulated futures are priced at spot. This is a
synthetic zero-carry futures market, not a claim that an index can be traded.
The drift choice prevents a directional-return forecast from being confused
with hedging skill. [Drift and deep hedging](https://arxiv.org/abs/2111.07844)
discusses that distinction.

Simulation and valuation use one coherent trading-year clock: 252 sessions of
375 minutes. This differs from the historical pricing pipeline's ACT/365
convention; these time conventions are not mixed within a simulated experiment.
At the 30-minute endpoint, options retain positive time to expiry and are
marked to model value, not incorrectly replaced with expiry payoffs.

Starting index levels range from 21,000 to 26,000; strikes vary around the
initial level, calls/puts are sampled, initial volatility ranges from 12–35%
and remaining maturity from 1–7 trading days. These are chosen simulation
ranges, not estimates calibrated to NIFTY history. Variance is assumed observed
in the simulated state; real-market estimation error is not tested here.

Heston uses a shared normalized price/delta grid, evaluated using the same
oracle for the neural policy and conventional baselines. Direct Fourier checks
on 256 sampled states found maximum interpolation errors of:

| Scenario | Option price error per unit | Delta error |
|---|---:|---:|
| Nominal Heston | INR 0.496 | 0.000899 |
| Parameter-shift Heston | INR 0.303 | 0.000933 |

The grid does not silently extrapolate. Its variance range corresponds to
6–65% instantaneous volatility; out-of-grid paths raise an error rather than
being selectively discarded. Pricing is approximate, and small performance
differences should be interpreted with that numerical error in mind.

## Training and test protocol

The final run is `data/research/deep_hedging_part2_final_20261031/`.

- Three policy seeds per market: 11, 22 and 33.
- Sixty behavior-cloning warm-start updates per seed.
- Twelve CVaR epochs, 32 minibatch updates per epoch, 256 paths per batch.
- 98,304 fresh risk-training paths per seed; 589,824 across the six runs,
  in addition to warm-start and threshold-initialization samples.
- 2,048 validation paths per market, shared across seed comparisons.
- Epoch and seed chosen only by validation CVaR after CVaR updates.
- 8,192 independent test paths for each nominal market.
- 4,096 additional Heston parameter-shift paths.
- Predeclared doubled-cost tests reuse each nominal test bank for paired stress comparisons.

An initial exploratory trainer used noisy minibatch threshold estimates and
performed poorly on validation. It completed and scored its initial test bank
during diagnostics; that bank is considered exposed and remains archived under
`data/research/deep_hedging_part2_20261031/`.

The revised trainer uses a learned threshold and a path-specific control variate.
Its authoritative evaluation uses new seeds, declared before testing:
1,090,001 for GBM, 1,190,001 for Heston and 1,290,001 for the parameter shift.
The new bank was generated only after all final model choices were frozen in
`frozen_before_test.json`. It was not used for further performance tuning.

GBM selected seed 22, epoch 1; Heston selected seed 11, epoch 1. All seed results,
validation curves and behavior-cloning diagnostics are retained. Selecting an
early checkpoint is not evidence that later optimization converged.

## Actual results

All amounts below are INR **per independent simulated episode**, net of the
configured execution costs. Lower positive loss CVaR is better.

| Market | Strategy | Mean net P&L | Loss CVaR 95% |
|---|---|---:|---:|
| GBM | Unhedged | -478.53 | 31,227.05 |
| GBM | Static delta | -2,236.45 | 6,178.40 |
| GBM | Dynamic delta | -2,433.52 | 6,136.62 |
| GBM | No-trade band | -2,239.66 | 6,178.40 |
| GBM | Selected neural policy | -2,381.32 | 6,911.70 |
| Heston | Unhedged | -322.11 | 30,060.20 |
| Heston | Model-consistent static delta | -2,238.29 | 7,097.47 |
| Heston | Model-consistent dynamic delta | -2,410.92 | 7,142.31 |
| Heston | Model-consistent no-trade band | -2,241.81 | 7,097.47 |
| Heston | Black–Scholes dynamic delta | -2,420.69 | 6,860.72 |
| Heston | Selected neural policy | -2,388.64 | 7,550.24 |

Relative to unhedged positions, the neural policy reduces loss CVaR by about
77.9% under GBM and 74.9% under Heston. Conventional hedging does better on that
metric. Against model-consistent dynamic delta, neural loss CVaR is higher by
INR 775.07 under GBM and INR 407.93 under Heston. Paired bootstrap 95% intervals
for these differences are respectively [603.71, 957.50] and [230.76, 577.23].

The small mean-P&L differences versus dynamic delta have intervals containing
zero. They do not establish a reliable expected-return advantage. Negative
average P&L is expected in this zero-drift, fixed-position experiment once costs
are included; this is not a synthetic arbitrage-profit exercise.

The Heston parameter shift increases initial volatility, shortens maturity,
changes leverage correlation, mean reversion and volatility of volatility. The
neural policy's loss CVaR there is INR 8,439.13, versus INR 7,420.28 for dynamic
delta. Doubled-cost results are included in the complete comparison. The stress
tests do not establish superior neural robustness.

The selected warm-start validation CVaR values were 6,318.77 (GBM) and 7,491.34
(Heston); the best post-CVaR values were 6,825.72 and 7,719.72. This directly shows
that CVaR optimization did not improve the warm start in this budget. More
iterations alone are not guaranteed to fix it. Credit assignment, gradient
variance, stochastic/greedy mismatch and policy design need investigation on
training/validation data, followed by another new confirmation bank.

No Sharpe ratio or random-path-order drawdown is reported: independent Monte
Carlo paths have no justified daily calendar or chronological portfolio order.
Historical Sharpe/drawdown belong to the Part 3 walk-forward experiment.

## Execution and numerical checks

The batched environment duplicates the Part 1 accounting rules for efficient
training. Validation replays the selected policy and conventional baselines
through the original scalar engine on each scenario. All checked P&L values
matched exactly; the audit is `scalar_replay_checks.json`.

Tests cover GBM/Heston reproducibility, independent seeds, martingale moments,
put-call parity, the constant-volatility limit, Heston delta finite differences,
pricing-grid error, integer actions, future-state isolation, scalar/batch P&L,
margin exits, CVaR signs, learned-threshold gradients, checkpoint reload and
actual CVaR parameter updates.

Costs and margin remain the configured Part 1 proxies. No official historical
SPAN, quote execution, liquidity or latency model has been introduced. The
hypothetical doubled-cost scenario is not a statutory fee schedule.

## Deliverables and files

| Deliverable | Implementation / artifact |
|---|---|
| GBM/Heston market and option pricing | `src/deep_hedging/simulation.py`; saved pricing grids and path banks |
| Whole-lot batched environment | `batch_environment.py`, checked against `engine.py` |
| PyTorch policy and safe reload | `policy.py`; six `policy_seed_*.pt` files |
| CVaR training | `objectives.py`, `train.py`; per-seed learning curves and warm checkpoints |
| Frozen independent comparisons | `evaluate_simulation.py`, `part2.py`; metrics, path P&L and paired intervals |
| Inference replay | `src/predict_simulated_hedge.py` |
| Regression checks | `tests/test_deep_hedging_part2.py` |
| Reproduction/provenance | experiment plan, source snapshot, model hashes and environment versions |

Primary result files are `RESULTS.md`, `simulation_comparison.csv` and
`paired_comparisons.csv`. Every scenario also has per-policy execution arrays,
paired path-level P&L and all random-seed metrics. The selected GBM checkpoint is
`gbm/policy_seed_22.pt`; Heston is `heston/policy_seed_11.pt`.

## Commands

Use the project environment. CPU PyTorch 2.15.0 was installed and tested; base
requirements remain separate from this optional dependency.

```powershell
.\.venv\Scripts\python.exe -m pip install --timeout 180 -r requirements/deep-hedging-lock.txt

.\.venv\Scripts\python.exe src/run_deep_hedging.py --output data/research/my_deep_hedging_run

.\.venv\Scripts\python.exe -m unittest tests.test_deep_hedging_part2 tests.test_hedging_part1 -v
```

Outputs must be new directories. Training budget flags are visible in `--help`.
The Python API also accepts a matching 60-minute execution/simulation setup;
the reported fitted models were trained for 30 minutes.

Replay a fitted model without training:

```powershell
.\.venv\Scripts\python.exe src/predict_simulated_hedge.py --checkpoint data/research/deep_hedging_part2_final_20261031/gbm/policy_seed_22.pt --paths data/research/deep_hedging_part2_final_20261031/gbm_test_paths.npz --output data/research/my_gbm_policy_replay
```

The original [Deep Hedging framework](https://arxiv.org/abs/1802.03042) motivates
cost- and risk-aware hedging. This implementation uses a discrete policy-gradient
approach and does not reproduce every architecture or reported result in that
paper. The next stage must not advertise historical ML-assisted superiority
until it is actually implemented and validated.

## Final verification

All 106 regression tests passed, including 19 Part 2 tests. Source and test
compilation passed. All six frozen checkpoint hashes match their pre-test
manifest. Scalar accounting replays matched the batched environment, and a
separate reload of the selected GBM policy reproduced all 8,192 saved test
executions exactly. Machine-readable checks are saved in
`data/research/deep_hedging_part2_final_20261031/verification.json`.
