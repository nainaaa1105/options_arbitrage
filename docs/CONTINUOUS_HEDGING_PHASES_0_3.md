# Continuous deep hedging — phases 0a through 3

## Scope and two-track contract

The continuous research track is a separate package, `src/continuous_hedging/`.
It does not replace the historical whole-lot futures implementation or import
its collectors/classifier modules. All old data, weights and prior reports are
preserved. Phases 4–9 are not part of this delivery.

| Convention | Continuous research | Preserved executable research |
|---|---|---|
| Position | Short one European call or put through expiry | Existing four-lot long-option 30-minute experiments |
| Hedge asset | Synthetic tradable underlying, continuous units | Actual recorded NIFTY futures, integer lots |
| Time | One year fraction shared by pricing and simulation, initial 20/252 | Historical ACT/365 features / explicit intraday ledger |
| Rates/drift | Both zero | Configured historical carry/fee assumptions |
| Premium | BS inception price, identical for every strategy on that path | Actual option entry proxy / supplied quote |
| Exit | Pay liability at expiry and liquidate all underlying | Close option and futures at holding-period exit |
| Learning | Differentiable P&L backpropagation | Categorical whole-lot policy gradients |
| Scope | Controlled validation; normalized strike 1 currency | Contract sizing, fees, margin and historical tests |

This research is neither an implementation of Indian futures settlement nor
an assertion that a continuous policy is executable without a rounding bridge.

For B paths and N rebalancing intervals, spots have shape `[B,N+1]`, chosen
hedges `[B,N]`, and scalar-per-path outputs `[B]`. A hedge at t observes the
current state and earns only the subsequent move. Initial and terminal hedge
inventory are zero outside those N held positions.

The zero-rate short-option P&L is

`premium - payoff + sum(h_t * (S_(t+1)-S_t)) - trading_costs`.

Position changes include the initial purchase/sale and final liquidation.
Costs support proportional notional, absolute half-spread and fixed per-trade
charges in evaluation. A separate stock/cash ledger independently purchases
and sells notionals and reconciles against the gain-based P&L. Exact fixed
trade-count costs are nondifferentiable; backprop training rejects them rather
than silently claiming to optimize a nonexistent trade-count gradient.

## Phase 0a corrections and preservation

- Failed paper liquidation now persists pending-liquidation and risk-halt flags
  independently of rejected fills. A recovered packet closes existing exposure;
  a risk halt remains latched after closing.
- Missed five-minute boundaries can recover at later valid boundaries and are
  counted. Off-grid/repeated decisions remain invalid.
- Paper account read/modify/write starts a SQLite immediate transaction.
- New protocols hash imported numerical helpers as well as the runner.
- Checkpoint horizon/action configuration is checked; the hard-coded 30-minute
  remaining-step calculation was replaced by the configured horizon.
- Completed-state grid, option/delta sign and classifier session-domain checks
  were added. These are validation checks, not a full live-feature lineage system.
- Read-only `forward_paper.py inspect` can inspect old immutable sessions without
  authorizing execution under changed code. Old runner hashes are not rewritten.
- Legacy training/signal CLI defaults require explicit opt-in. Custom legacy
  model CLIs need matching context; frozen research bundles use their dedicated
  inference command. Programmatic APIs remain available for explicit legacy use.
- Frozen inference checks model hashes. `research_registry.py` records canonical
  corrected artifact hashes and installed versions, and validates them before
  registered inference. `research_runs.json` identifies the separate tracks.
- README recovery status and the Vega-unit comment were corrected; numerical
  Greek values were not changed by that comment correction.

The initial registry is `data/research/phases0_3_baseline_20261031.json`.
It preserves reference hashes before subsequent source changes. The final
delivery registry will record the finalized code; historical weight/data hashes
must agree. `requirements/research-lock.txt` records the current environment;
the CPU Torch wheel uses the separately documented PyTorch CPU index. This is
an environment snapshot, not a claim that a clean-machine install was executed.

Remaining operations work from the audit is explicitly outside offline phase
validation: authoritative active-contract catalog checks, continuous feature
construction and scheduling, malformed-input quarantine, liquidation-equity
risk reporting, alerts, calendar/overnight recovery, broker-specific margins,
and full dashboard health hardening. No unattended service is started.

## Phases 2 and 3 learning and baselines

The policy is a bounded shared-weight MLP with two 32-unit tanh layers. It
observes standardized moneyness, remaining time fraction, volatility, absolute
BS delta, call/put identity, previous hedge magnitude and the cost parameter.
Actions are continuous underlying units bounded in magnitude by 1.2. The short
call uses nonnegative hedge inventory; the short put uses nonpositive inventory.

Two disclosed initializations are compared:

- `flat`: initial hedge magnitude 0.5, without the analytic-delta skip prior.
- `delta`: explicit analytic delta prior plus a trainable residual.

Both observe BS delta as an input. Successful flat-model training is evidence
that the P&L training loop learns an appropriate mapping, not independent
discovery of Black–Scholes theory. The initial delta-prior policy is a diagnostic
and valid checkpoint-selection fallback, never evidence of learning by itself.

With zero frictions the experiment uses the Markov state and masks the inventory
input, allowing vectorized evaluation. With costs, inventory is carried through
the differentiable rollout. Fresh paths are sampled every optimizer update.
Adam, gradient clipping and exponential learning-rate decay are used.

The objectives are normalized RU CVaR with a learned training threshold, and
stable log-sum-exp entropic risk with inverse-currency aversion 100. Validation
uses exact empirical CVaR, rather than the fitted threshold, so checkpoint
selection cannot improve merely by moving the threshold. Empirical CVaR uses
fractional tail mass and is reported at 95% and 99%.

Baselines are no hedge, discrete analytic delta, constant no-trade band and
Whalley–Wilmott. Both bands trade to the nearest band edge. Constant bandwidth
and WW coefficients are selected from explicit validation grids, separately for
each objective, then frozen. An additional WW comparison fixes its coefficient
to the entropic objective's aversion to distinguish utility matching from tuning.

For proportional rate c, option gamma Gamma and price S, the zero-rate WW
half-width used here is `(3*c*Gamma^2*S/(2*a))^(1/3)`. It is a small-cost,
exponential-utility approximation. It is not asserted to be optimal for finite
rebalancing, CVaR or fixed charges. Reference: the original-author
[Whalley–Wilmott paper](https://users.ox.ac.uk/~ofrcinfo/file_links/mf_papers/1999mf09.pdf)
and the implementing research library's
[formula documentation](https://pfnet-research.github.io/pfhedge/generated/pfhedge.nn.WhalleyWilmott.html).

The phase-3 foundation compares zero costs and one positive proportional-cost
case (0.001). This is **not** the multi-level cost-grid/no-trade-band experiment
reserved for Phase 4.

## Validation protocol and numerical acceptance

Models use seeds 11/22/33. The validation market bank has seed 35001 and 4,096
paths. Independent primary testing uses seed 85001 and 16,384 paths; the nested
frequency bank uses seed 95001. No independent test bank is generated until
the selected frictionless model for each initialization/objective passes:

- Mean absolute delta error at most 0.06.
- P&L standard deviation at most 1.20 times the same-grid analytic-delta value.

These are declared numerical tolerances at 32 steps, not an exact continuous-time
theorem or a promise that delta minimizes finite-grid CVaR. Analytic delta
frequency validation uses the same 256-step paths subsampled to 8/16/32/64/128/256
steps, and requires lower dispersion at the finest than at the coarsest grid.
Individual paths need not improve monotonically. Fine-grid observations never
enter the coarser hedge's state before their sampled timestamp.

All seeds and validation curves are retained. Final seed/checkpoint selection
uses validation objective only. Models, baseline settings, source and experiment
configurations are frozen before independent testing. Reports include mean/std,
CVaR95/99, worst P&L, costs, turnover, delta errors, reconciliation and paired
mean-P&L intervals. Synthetic paths have no invented calendar Sharpe/drawdown.

The first 12-epoch run at `continuous_phases1_3_20261031` failed the flat-model
dispersion gates and generated no test banks. It remains a development record.
The extended run keeps the same architecture, tolerances, objective, validation
bank and split protocol, but provides flat initialization 48 training epochs.
Delta-initialized policies retain 12 epochs. This validation-driven budget
amendment occurred before independent testing; it is not test-set retuning.
That second sigmoid-output run still failed the flat entropic dispersion gate
and generated no independent test bank. The next run,
`continuous_phases1_3_projected_20261031`, uses projected bounded position
outputs instead of sigmoid logits. It keeps the same observations, losses,
seeds, validation bank and tolerances. This removes the need for the network
to learn the saturating inverse-sigmoid shape near delta 0/1. Flat initialization
still starts at magnitude 0.5 and learns solely from P&L, without delta labels.
Output parameterization is versioned in new checkpoints. Initial development
checkpoints lacking that field are loaded only through their hash-verified
as-run source snapshot, so sigmoid weights cannot silently become projected
weights. Unknown or missing architecture provenance fails closed.

## Files and reproduction

| Deliverable | Location |
|---|---|
| Independent research modules | `src/continuous_hedging/` |
| Experiment entry point | `src/run_continuous_hedging.py` |
| Registry and model/data preservation | `src/research_registry.py`, `research_runs.json` |
| Accounting/causality/gradient/baseline tests | `tests/test_continuous_hedging.py` |
| Recovery and risk-latch tests | `tests/test_forward_paper_part4.py` |
| Registry and legacy routing tests | `tests/test_research_registry.py` |
| Development run and curves | `data/research/continuous_phases1_3_20261031/` |
| Extended experiment | `data/research/continuous_phases1_3_extended_20261031/` |
| Projected-output experiment | `data/research/continuous_phases1_3_projected_20261031/` |

```powershell
.\.venv\Scripts\python.exe src/run_continuous_hedging.py --output data/research/my_continuous_phases1_3

.\.venv\Scripts\python.exe -m unittest tests.test_continuous_hedging tests.test_research_registry tests.test_forward_paper_part4 -v

.\.venv\Scripts\python.exe src/research_registry.py verify --file data/research/phases0_3_baseline_20261031.json

.\.venv\Scripts\python.exe src/predict_research_model.py --registry data/research/phases0_3_baseline_20261031.json --input data/research/expanded_20261030/data/processed/fresh_ml_dataset.parquet --output data/research/my_verified_predictions.parquet

.\.venv\Scripts\python.exe src/forward_paper.py inspect --session data/research/paper_part4_final_demo_20261031/adapted_ml
```

Every run requires a new output directory. Test failures preserve their logs
and artifacts. No resume command silently reuses an independent test bank for
selection. Read `RESULTS.md`, `validation_gates.csv`, `verification.json` and
the frozen manifest from the actual successful run before making a result claim.

Final delivery verification: **162 tests passed**. All 18 continuous checkpoints
were independently reloaded and exactly reproduced their saved positions and
P&L across all 16,384 primary test paths. All 18 protected pre-existing artifacts
match the initial baseline hashes, and registered classifier inference exactly
matches all 12,466 preserved predictions. Both failed development runs generated
no independent test banks. Final source/artifact/environment registration is
`data/research/phases0_3_delivery_registry_20261031.json`; the machine-readable
checklist is `data/research/phases0_3_delivery_verification_20261031.json`.

## Completed experiment results

The authoritative run is `continuous_phases1_3_projected_20261031`. All four
validation and independent test model gates passed. On 16,384 independent
primary test paths, selected models' delta MAE/std-relative-to-analytic-delta are:

| Objective | Initialization | Delta MAE | P&L std / delta std |
|---|---|---:|---:|
| CVaR | Flat | 0.02009 | 1.1115 |
| CVaR | Delta | 0.02136 | 1.1099 |
| Entropic | Flat | 0.00989 | 1.0322 |
| Entropic | Delta | 0.00635 | 1.0099 |

These satisfy the predeclared numerical gates at 32 intervals. They do not
establish exact recovery in the mathematical continuous-time limit. The CVaR
policies accept a larger P&L standard deviation in exchange for a smaller
sampled loss tail. Mean returns are near zero, consistent with the zero-drift
frictionless setting, rather than a discovered arbitrage.

On the separate common-path frequency bank, analytic delta P&L std falls from
0.006144 (8 intervals) to 0.001159 (256). Frozen neural policies also improve,
but retain approximation/finite-training-objective error: at 256 intervals the
CVaR flat model has std about 0.001917. Do not claim it equals the asymptotic
analytic solution based on the coarser-grid gate.

In the single positive-cost foundation case, the selected CVaR policy has
CVaR95 0.009769 versus delta 0.011532, with std 0.003726 versus delta 0.003436.
The risk/cost tradeoff is measured in normalized currency per short option;
these are not NIFTY returns or evidence that every metric improves. The full
cost grid and no-trade-band interpretation remain Phase 4 work.

Validation figures and a frozen-policy frequency table are in the successful
run's `validation_figures/` directory. `report_continuous_validation.py` reloads
frozen policies to reproduce them and verifies checkpoint hashes. Independent
primary paths are reused across cost scenarios; they are not counted as
independent new observations for each model. The primary and frequency banks
contain 32,768 distinct simulated paths in total.
