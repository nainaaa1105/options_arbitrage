# NIFTY options system: consolidated status and latest implementation plan

As of 31 October 2026. This report consolidates the canonical run registry,
saved evaluation tables, implementation reports and delivery verification.
The Phase 4/7/5 artifact registry was reverified while preparing this report.
No new model training or market performance is asserted by this document.

## 1. Current position

The project is an implemented options-pricing/convergence research pipeline,
a historical option/futures hedge comparison, an offline dashboard with paper
execution primitives, and a controlled continuous deep-hedging research suite.

In the **latest revised implementation plan**, completed stages are:

**0a → 1 → 2 and 3 → 4 → 7 → 5.**

Phase 4 completed the minimum viable research project. Phase 7 and Phase 5
added granularity and robustness studies. **Phase 6 is next.** Phases 8, 0b
and 9 remain. This does not mean every numbered phase through 7 is complete.

The research implementation is substantial and its latest offline checks pass.
There is no established profitable trading edge, general 93–94% endpoint
accuracy, consistent historical ML-hedging advantage, or unattended production
readiness.

## 2. What the system does and why it matters

The pricing pipeline asks whether an option is priced unusually relative to a
statistical volatility/pricing reference, then estimates whether that deviation
will shrink. The hedge engine asks how much underlying/futures exposure to
hold against an option liability or position while accounting for costs and
bad outcomes. These are related but different decisions.

Potential uses are an options research workbench, candidate screening,
model-risk monitoring, and comparison of the cost of hedging against its
protection from tail losses. These are demonstrated research capabilities.
Executable arbitrage, positive after-cost expectancy, and business savings in
a live account remain hypotheses requiring suitable market evidence.

## 3. Architecture and data flow

```text
Upstox option/spot candles + instrument metadata
  -> expiry collection, strike selection and raw Parquet storage
  -> synchronized master/feature datasets
  -> pricing, IV, Greeks, historical/reference volatility
  -> deviations, parity residuals and signal eligibility
  -> causal ML features and endpoint/first-passage/sustained labels
  -> chronological purged fitting, calibration and frozen predictions
  -> candidate trades and OHLC backtesting/risk reports

Recorded option/futures candles + past-only classifier predictions
  -> fixed historical hedge episodes
  -> conventional and whole-lot neural hedgers
  -> transaction ledger, equity, costs and risk comparisons

Independent continuous research package
  -> GBM/Heston/bootstrap paths + matched short-option liability
  -> continuous/learned-band/categorical policies
  -> frozen common-path comparisons, robustness, plots and replays

Saved research outputs -> offline HTML dashboard
Completed states + synchronized quotes -> packet adapter -> SQLite paper ledger
                                                       -> status/monitor outputs
```

The last market-to-paper chain currently consists of separate commands. It is
not yet a hardened continuously running feature/execution/monitoring service.
The newer continuous models have not been automatically connected to that
older paper runner.

Core locations:

| Area | Main implementation |
|---|---|
| Collection | `src/upstox_api.py`, `collect_all.py`, `collect_expiry.py`, spot/futures collectors |
| Pricing and reference features | pricing/Greeks modules and `build_*features.py`, `build_reference_iv.py` |
| Labels and classifier | `build_convergence_labels.py`, `research_targets.py`, split/training/research/inference scripts |
| Signal/backtest/risk | `generate_signals.py`, `generate_trade_signals.py`, `backtest.py`, `risk_analysis.py` |
| Historical/discrete hedge engine | `src/deep_hedging/` and Part 1–3 runner/evaluation scripts |
| Continuous research | `src/continuous_hedging/`, Phase 4/7/5 runners, reporting and verification scripts |
| Paper/dashboard | `forward_paper.py`, `quote_capture.py`, `assemble_paper_events.py`, `monitor_paper.py`, `build_project_dashboard.py` |
| Provenance | `research_runs.json`, `research_registry.py`, frozen manifests and as-run source snapshots |

The Python stack includes pandas/NumPy, Parquet, SciPy, scikit-learn, XGBoost,
LightGBM, Optuna, PyTorch, SQLite and Matplotlib. The dashboard is self-contained
HTML/JavaScript. Environment snapshots exist; a clean-machine automated install
has not been certified.

## 4. Data, pricing and classifier implementation

The corrected expanded option feature dataset contains **941,949 rows, 60
expiries and 283 option dates**. The initial 519,725-row description refers to
an older dataset. Older weights/data are preserved separately and must not be
mixed with corrected feature context.

Historical futures collection covers 14 monthly contracts, 21,355 bars and 284
observed dates. A futures session without corresponding option observations
remains a zero-trade date. Coverage/rejection records retain missing or invalid
inputs rather than inventing complete paths. An earlier audit found 6,324
option rows with zero reported OI; that is a data-quality/eligibility issue,
not proof that the market had no open interest.

Implemented pricing/feature capabilities include European Black–Scholes, IV
inversion, Delta/Gamma/Vega/Theta, moneyness, maturity, daily historical
volatility, IV–HV comparisons, leave-one-out neighboring-strike reference IV,
theoretical-price deviations, IV residuals and put-call parity residuals.
Reference prices are model-dependent relative benchmarks rather than proof
of risk-free arbitrage.

The historical spot/option candle alignment error was repaired in the isolated
research rebuild. Availability timestamps represent completed observations.
Daily HV uses past completed sessions. Chronological splits purge labels that
cross boundaries; preprocessing fits on training data. Original base features
are supplemented by causal engineered features with saved per-run schemas.

Classifier code supports Logistic Regression, Random Forest, XGBoost,
LightGBM, tuning, ensemble configurations, calibration, operating-point
analysis and coverage-aware selective prediction. The preserved later-period
frozen evaluation contains **12,466 rows over 28 sessions**:

| Target | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|
| Endpoint | 72.20% | 62.91% | 52.88% | 0.5746 | 0.7825 | 0.6443 | 0.1783 |
| First passage | 86.53% | 85.57% | 97.60% | 0.9119 | 0.9144 | 0.9582 | 0.0960 |
| Two-bar sustained | 75.71% | 70.79% | 73.78% | 0.7226 | 0.8433 | 0.7918 | 0.1604 |

Endpoint measures the specified horizon outcome; first passage can count a
temporary convergence before it. They are different targets and class balances.
At a predeclared ±0.30 endpoint confidence margin, selective accuracy is
**93.66% at 23.66% coverage**, with 93 positive predictions and 87.10% positive
precision. This is not 93.66% accuracy over all opportunities, a trade win rate,
or a guaranteed deployment result. The older 93.45% figure belongs to a
different historical result and should not replace the current evaluation.

These dates have been reviewed during development. Their chronological
ordering does not make them a pristine prospective holdout today. The latest
hedging extensions did not retrain or improve the classifier.

## 5. Original Parts 1–4: separate from the new phase numbering

| Original part | Implemented scope | What the evidence does not establish |
|---|---|---|
| Part 1 | Historical futures/option episodes, real contract metadata, accounting, conventional baselines, costs, margin proxies, equity and risk | Exact exchange margins or demonstrated executable OHLC fills |
| Part 2 | PyTorch categorical whole-lot hedgers, GBM/Heston simulation, model-consistent valuation, CVaR optimization and independent simulation tests | Consistent superiority over strong delta baselines |
| Part 3 | Historical policy transfer/adaptation, past-fitted classifier probabilities, matched ML/no-ML ablation, ledgers and reports | Incremental convergence-ML benefit or positive net strategy expectancy |
| Part 4 | Offline interactive dashboard, frozen paper protocols, bid/ask packet processing, SQLite persistence, monitoring and synthetic integration demo | A fully hardened unattended service or genuine forward trading performance |

The historical Part 3 test has 168 fixed long-option episodes over 28 sessions,
with ₹5,000,000 initial capital per separate sequential account:

| Strategy | Net P&L | Episode loss CVaR95 |
|---|---:|---:|
| Unhedged | +₹80,827.85 | ₹11,096.73 |
| Dynamic delta | −₹345,002.28 | ₹4,555.48 |
| Adapted without ML | −₹346,647.44 | ₹4,568.57 |
| Adapted with ML | −₹346,647.44 | ₹4,568.57 |

Hedging reduces tail exposure but incurs substantial costs. ML/no-ML results
are identical, so an incremental ML contribution is not established. Only
**13 of 1,008** test hedge decisions had valid classifier probabilities. The
classifier's eligible-deviation universe poorly overlaps the fixed nearest-ATM
hedge universe. This is a key reason Phase 8 requires an objective/universe
decision rather than simply wiring a probability into every hedge.

Historical Sharpe, Sortino, drawdown, daily returns, turnover, fees, margin
usage and session-bootstrap comparisons exist. Annualized ratios from 28
sessions are unstable descriptions, not long-run forecasts. Profitable
unhedged observations do not establish a superior risk-adjusted strategy.

## 6. Latest revised implementation plan and completion status

| Execution order | Phase | Work | Status |
|---|---|---|---|
| 1 | 0a | Results-affecting fixes, provenance protection and research freeze | Complete within the agreed offline scope |
| 2 | 1 | Explicit two-track experiment contract and accounting tests | Complete |
| 3 | 2 | Differentiable continuous learning and frictionless validation | Complete |
| 3, parallel research scope | 3 | Conventional, entropic and Whalley–Wilmott baselines | Complete |
| 4 | 4 | Multi-cost grid, inventory/hold diagnostics, plots and central report | Complete; minimum viable research milestone passed |
| 5 | 7 | Matched continuous-to-whole-lot rounding bridge | Complete |
| 6 | 5 | Heston, bootstrap, cross-world misspecification, alpha/frequency/stress sweeps | Complete |
| 7 | 6 | GRU, controlled ablations and ensemble diagnostics | Not implemented in the new continuous track |
| 8 | 8 | Objective-defined signal handoff and integrated comparison | Not complete; older Part 3 integration is not this phase |
| 9 | 0b | Remaining operational hardening | Partial foundations; substantial work remains |
| 10 | 9 | Genuinely later forward validation and readiness decision | Not completed |

The completed continuous track uses a short European call/put through expiry,
normalized strike 1, synthetic tradable underlying and explicit frictions.
The historical track holds recorded long-option positions for an intraday
horizon and hedges with integer futures lots. Their liabilities, clocks,
prices and units are not interchangeable. Phase 7 bridges granularity under a
matched continuous contract; it does not make the synthetic model production
ready for Indian futures by itself.

### Completed stage details

**0a:** Fixed failed-close risk-latch persistence, pending-liquidation recovery,
missed-boundary recovery, transactional account read/update, helper code hashes,
horizon/config validation and classifier session-domain checks. Added read-only
inspection, canonical artifact routing, registry checks and environment snapshots.
This addresses selected earlier audit defects; the old audit is not a current
list of wholly unfixed issues.

**1:** Defined the two tracks, sign conventions, timing, positions, opening/final
costs and independent cash reconciliation. Tests cover causality and accounting.

**2:** Implemented bounded MLP policies trained through differentiable P&L,
fresh training paths, learned CVaR thresholds and stable entropic risk.
Flat and delta-prior initializations passed independent declared frictionless
tolerances. Flat models still observe delta as a feature; this is not a claim
of discovering Black–Scholes without analytic information. Failed development
gate runs are retained and did not generate their independent test banks.

**3:** Added unhedged, analytic delta, constant no-trade-band and
Whalley–Wilmott comparisons; validation-only tuning and utility-matched controls.
Whalley–Wilmott remains a small-cost exponential-utility benchmark, not a
universal CVaR optimum.

**4:** Completed four proportional cost rates, two objectives, MLP and explicit
band families, three seeds, 48 checkpoints, hold diagnostics, conditional paired
intervals and PNG/PDF reports. Both independent MVP gates passed. At cost
0.001, selected MLP CVaR95 is 0.009926 versus delta 0.011809: **15.9% lower**.
The conventional tuned band has better mean P&L than that MLP. The explicit band
architecture imposes its hold structure; the MLP did not show an exact hold
interval at the reported diagnostic state.

**7:** Completed closed-loop rounding for books of 1/4/10/40 identical option
lots, matched conventional baselines, 12 categorical risk-training fits and 12
imitation-only checkpoints. The next decision observes executed inventory.

| Book size | Continuous CVaR95 | Rounded teacher | Selected categorical |
|---|---:|---:|---:|
| 1 | 0.009841 | 0.051248 | 0.049961 |
| 4 | 0.009841 | 0.015537 | 0.015803 |
| 10 | 0.009841 | 0.010943 | 0.011368 |
| 40 | 0.009841 | 0.009935 | 0.011517 |

The granularity gap decreases with book size. Categorical training does not
consistently beat rounding; its one-lot CVaR advantage has an interval crossing
zero. There is no cross-strike diversification effect in this book experiment.

**5:** Completed 11 cases/33 models, three worlds, a 3×3 transfer matrix,
alpha 0.50/0.75/0.90/0.95/0.99, 5/20/80-step frequency comparisons, parameter
and execution stresses, source/target-informed conventional controls and an
observable-volatility sensitivity.

| Matching evaluation world | Neural CVaR95 | Delta CVaR95 | Tuned constant band | Neural reduction vs delta |
|---|---:|---:|---:|---:|
| GBM | 0.007382 | 0.008463 | 0.008163 | 12.8% |
| Heston | 0.009630 | 0.010269 | 0.010070 | 6.2% |
| Bootstrap | 0.007981 | 0.008782 | 0.007498 | 9.1% |

Neural improvement over delta is not universal improvement over all baselines.
Bootstrap transfer to other worlds degrades. Heston uses training-only
moment-proxy parameters, Fourier initial premiums and observed variance in
the main experiment. Replacing latent variance with a training-only initial
prior and strictly trailing RV raises its policy's CVaR95 about **10.3%**.
Bootstrap libraries have 266/54/30 train/validation/test daily returns; its
thousands of simulations are not thousands of independent market histories.
Its BS reference premium is not claimed arbitrage-free. Confidence intervals
condition on the model/library and exclude estimation uncertainty.

Low-volatility BS numerical cancellation was corrected, initial Heston pricing
and simulation consistency were audited, and old warm-start score metadata
was reconciled without changing frozen checkpoint weights.

## 7. Remaining implementation and acceptance plan

These deliverables detail the remaining scope in the already chosen order.
Exact budgets and acceptance thresholds must be written before the next
independent evaluation; observed old test results must not select them.

### Next: Phase 6 — memory, ablations and ensembles

1. Add a causal GRU hedge policy with explicit inventory/hidden-state handling,
   checkpoint schema, deterministic reload and episode resets.
2. Compare MLP and GRU on identical contracts, costs, information and path banks,
   with disclosed parameter counts, training budgets and seeds.
3. Predeclare ablations for analytic priors, features, inventory/memory and
   initialization. Retain strong conventional and explicit-band controls.
4. Evaluate seed diversity, action/P&L correlation and validation-defined
   ensembles. Run ensemble decisions closed-loop; specify rounding separately.
5. Freeze selections before new independent simulation tests. Report all seeds,
   cost/tail-risk trade-offs, latency, curves and uncertainty.

Completion requires causal/reload/accounting tests and a reproducible comparison.
It does not require GRU/ensembles to win. Owner: **assistant** implementation;
user only needed for any new product constraints or external resources.

### Then: Phase 8 — objective-defined signal handoff

1. Decide whether the product hedges existing liabilities or selects
   convergence-driven option entries and hedges those entries.
2. Specify target, long/short permissions, entry universe, capital, costs,
   horizon, risk limits and missing-probability behavior.
3. Produce causal classifier/context outputs with eligibility, availability,
   schema and model IDs. Do not feed in-sample probabilities to the hedger.
4. Compare probability-aware and masked/no-probability policies against delta
   and tuned cost-aware controls on identical admitted positions.
5. Report coverage, availability, net economics and incremental hedging effect;
   preserve the original convergence-target evaluation separately.

Completion requires a documented contract, integration/replay tests and an
untouched-later evaluation plan. The old Part 3 result is evidence to improve
the design, not proof that this phase is finished. Owner: **user + assistant**
for the objective; **assistant** for implementation.

### Then: Phase 0b — operations hardening

- Dated authoritative active-contract validation: underlying, expiry, units,
  tick size and contract provenance.
- Incremental synchronized feature/surface/HV context, batch/live parity and
  explicit missing/stale/out-of-domain fallbacks.
- Per-record malformed-input quarantine and continuity of close/risk events.
- Conservative liquidation equity, fees/depth-aware risk controls and explicit
  treatment of partial fills/leg risk.
- Shared health events, missing/stale session visibility and monitor alarms.
- Dashboard data-derived dates/counts and current lineage verification.
- Bounded orchestration, retry/restart recovery, session calendars, heartbeats,
  local alerts and explicit service ownership.
- Offline-safe test discovery/CI, guarded online diagnostics, dependency locks
  and a clean-install check.

Completion requires fault-injection tests for crashes, duplicate/malformed
events, stale data, missing bars, failed closes, concurrent consumption and
restart recovery. Selected risk-latch/concurrency/code-hash foundations already
exist; this is the remaining hardening, not a request to rebuild them blindly.
Owner: **assistant** code; **user** account permissions/hosting; **both** actual
broker charges/margins and executable-market assumptions.

### Finally: Phase 9 — genuinely later forward validation

1. Register and freeze the chosen protocol before its observation window.
2. Collect new quotes/completed states and run matched paper accounts.
3. Track uptime, missing observations, domain coverage, rejected fills,
   exposure, fees, liquidation equity, P&L and tail risk.
4. Reconcile and compare strategies with appropriate uncertainty. A suggested
   initial research window is 60 observed sessions, not a guarantee of adequate
   statistical power or acceptable performance.
5. Apply predefined operational/economic acceptance criteria. Decide whether
   more research, revised development, or a separate deployment proposal is
   warranted. A revision needs a new later confirmation window.

Owner: **both**. The assistant can build the tools; market time, authorization,
subscriptions, machine uptime and an eventual deployment decision require the
user. No broker orders or unattended service are currently authorized/started
by these research deliveries.

## 8. Metrics and interpretation

| Metric | Meaning and practical limit |
|---|---|
| Accuracy | Fraction of labels predicted correctly; compare against class balance and report coverage |
| Precision/recall/F1 | Positive-signal correctness, positive capture and their harmonic balance; not trade profitability |
| ROC-AUC | Ranking across thresholds; not an operating cutoff or economic utility |
| PR-AUC | Positive-class precision/recall summary; depends on prevalence |
| Brier/calibration | Probability quality; lower Brier is better, assessed on the same target/population |
| Selective accuracy | Accuracy only on accepted predictions; always state rejection/coverage and positive counts |
| Net P&L | Gains/losses after the specified costs and accounting assumptions |
| CVaR95/99 | Average worst 5%/1% loss tail; lower loss is better, distinct from average return |
| P&L dispersion | Variation in hedged outcomes; can worsen while a selected tail improves |
| Turnover/costs | Trading intensity and economic friction, including opening/final hedge trades |
| Sharpe/Sortino | Historical daily risk-adjusted return summaries; unstable in short samples |
| Drawdown | Peak-to-trough equity decline on a real chronological account |

The Monte Carlo research does not invent Sharpe/drawdown by arbitrarily
ordering independent simulated episodes into a calendar. Historical account
metrics and synthetic normalized risk metrics remain separate.

## 9. Verification and evidence locations

The last delivery reports **182 passing offline regression tests**, **105
frozen checkpoint files** from 93 risk-training fits plus 12 warm-only saves,
**124 saved execution banks replayed exactly**, 12 additional stress rows
checked, and **40 earlier protected artifacts unchanged**. These counts refer
to the latest extension delivery, not every archived model in the repository.
The consolidated artifact registry was reverified for this report; the full
test suite was not rerun merely to produce documentation.

Canonical paths are in `research_runs.json`. Principal evidence:

- `data/research/expanded_20261030/fresh_evaluation/fresh_metrics.csv`
- `data/research/expanded_20261030/fresh_evaluation/fresh_selective_metrics.csv`
- `data/research/historical_hedging_part3_final_20261031/RESULTS.md`
- `data/research/continuous_phases1_3_projected_20261031/`
- `data/research/phase4_cost_grid_20261031/report/MVP_RESEARCH_REPORT.md`
- `data/research/research_extension_delivery_20261031/phase7/report/ROUNDING_REPORT.md`
- `data/research/research_extension_delivery_20261031/phase5_corrected/report_final/ROBUSTNESS_REPORT.md`
- `data/research/research_extension_delivery_20261031/delivery_verification.json`
- `data/research/phases4_7_5_delivery_registry_20261031.json`

Earlier dated reports document their original stage, not today's total status.
In particular, the old audit predates the selected 0a repairs, and the original
0–3 report's statement that Phase 4 is future work was true at that delivery.
This document consolidates rather than rewrites their frozen historical record.
