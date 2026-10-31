# Continuous deep-hedging research: Phases 4, 7 and 5

The requested order is Phase 4 (minimum viable research project), Phase 7
(whole-lot bridge), then Phase 5 (robustness). These experiments extend the
continuous **short-option** research track. They do not reuse the old
long-option hedge weights under a different liability, change the convergence
classifier, or promote a model into paper/live execution.

## Phase 4: cost experiment and minimum viable project

Run: `data/research/phase4_cost_grid_20261031`.
Central report: [MVP_RESEARCH_REPORT.md](data/research/phase4_cost_grid_20261031/report/MVP_RESEARCH_REPORT.md).

- Four proportional costs: 0, 0.0002, 0.001 and 0.005 per traded notional.
- CVaR95 and entropic objectives; unconstrained residual MLP and explicit
  learned-band policies; three seeds each: **48 fitted checkpoints**.
- Baselines: unhedged, analytic BS delta, validation-tuned constant band,
  validation-tuned Whalley–Wilmott, and utility-matched Whalley–Wilmott.
- Independent zero-cost validation gates, fixed-state inventory response,
  exact/approximate hold-region diagnostics, common-path cost comparisons,
  all-seed results and paired conditional confidence intervals.
- Static PNG/PDF figures: cost/risk trade-offs, position response, P&L
  distributions and validation learning curves.

Both independent zero-cost MLP gates passed. On the cost=0.001 experiment,
selected MLP CVaR95 is 0.009926 versus 0.011809 for delta (about 15.9% lower).
The tuned constant band has better mean P&L than that MLP. Lower tail risk
does not imply every metric improves. Values are normalized per option with
strike 1; these numbers are neither rupee profits nor classification accuracy.

`mean_net_pnl` is premium minus terminal liability plus hedge gains minus all
specified friction. `pnl_std` measures dispersion of that hedge outcome.
`cvar95`/`cvar99` average the worst 5%/1% of losses, including fractional tail
weights when the empirical sample does not divide evenly. Lower tail loss is
better. Turnover measures traded underlying units/notional, including opening
and final liquidation. Entropic risk is a separate exponential-utility
criterion. A strategy can improve one of these measures while worsening another.

The explicit band learns wider bounds at the diagnostic state as cost rises.
Its structure was imposed by architecture. The MLP has no exact hold interval
on that diagnostic grid; the report does not claim that it discovered one.

## Phase 7: matched whole-lot bridge

Run: `data/research/research_extension_delivery_20261031/phase7`.
Report: [ROUNDING_REPORT.md](data/research/research_extension_delivery_20261031/phase7/report/ROUNDING_REPORT.md).

Books have 1, 4, 10 or 40 equal-sized option lots. Hedge lot units equal option
lot units. Each path's book repeats the same option contract; there is no
cross-strike diversification benefit in this experiment. Actions are integer
hedge lots divided by book size, with half ties
toward zero and a capacity of floor(1.2 × book size). The rounded policy observes
its **executed rounded inventory** at the next decision. All strategies share
liability, premium, paths, horizon and proportional costs.

The continuous teacher is frozen from the Phase 4 cost=0.001 CVaR MLP validation
selection. Each book also has a newly trained categorical CVaR policy, with
three seeds, learned tail threshold, an action-independent delta control
variate, and an explicitly disclosed 24-update imitation warm start.
Imitation-only checkpoints and all learning curves are retained: **12 risk-trained
and 12 warm-start checkpoints**. Policy selection uses validation only.

| Book size | Continuous CVaR95 | Rounded teacher | Selected categorical |
|---|---:|---:|---:|
| 1 | 0.009841 | 0.051248 | 0.049961 |
| 4 | 0.009841 | 0.015537 | 0.015803 |
| 10 | 0.009841 | 0.010943 | 0.011368 |
| 40 | 0.009841 | 0.009935 | 0.011517 |

Rounding approaches the continuous result as book granularity improves.
Categorical learning does not consistently beat the rounded teacher. In the
one-lot case its CVaR advantage has a conditional confidence interval crossing
zero. The comparison includes architecture and optimization-budget differences;
it does not isolate an algorithmic causal effect. Per-option normalization
prevents a larger book from appearing better merely because its units changed.

Supplemental fixed-action sensitivities double/quintuple proportional cost and
add spread/fixed-order friction. The fixed fee is charged per book order and
divided by book size for per-option reporting. These are normalized stress
parameters, not official NIFTY tax, brokerage, margin or execution estimates.

## Phase 5: robustness contract

Canonical corrected run:
`data/research/research_extension_delivery_20261031/phase5_corrected`.
Final report: [ROBUSTNESS_REPORT.md](data/research/research_extension_delivery_20261031/phase5_corrected/report_final/ROBUSTNESS_REPORT.md).

Daily data are read without alteration from
`data/raw/upstox/nifty_daily_to_20261029.parquet`. Calibration uses returns
ending 30 June 2026; validation is 1 July–15 September; the empirical test
library begins 16 September. Parameters and source hashes are saved before
training. These historical dates were previously inspected and are not a
pristine prospective market holdout.

- GBM, Heston and three-day moving-block bootstrap worlds; three seeds each.
- Full 3 × 3 cross-world evaluation, with source-validation-selected policies
  and source-tuned controls. Separately report stronger target-validation-tuned
  conventional benchmarks.
- GBM CVaR alpha sweep: 0.50, 0.75, 0.90, 0.95 and 0.99 on common test paths.
- GBM/Heston frequency sweep: 5, 20 and 80 intervals over the same 20 trading
  days. Each policy is trained separately and evaluated on nested common paths.
  No synthetic intraday interpolation is attributed to daily historical data.
- Frozen-policy volatility/Heston-parameter/uncentered-bootstrap stresses.
- Observable-volatility sensitivity: replace simulated latent Heston variance
  with a strictly trailing realized-volatility proxy, holding premium and
  payoff fixed. Until five returns are observed, initialize volatility from
  the training-range RMS, not the simulated latent initial variance. No model
  is refitted on these outcomes. This observability correction was recorded
  in `supplemental_protocol_v2.json` before the Phase 5 test freeze; the original
  protocol is retained.
- Eleven cases, three seeds each: **33 trained checkpoints**. All candidate
  weights and baseline choices freeze before test simulations are generated.

Heston parameters are capped realized-variance moment proxies, not a uniquely
identified fit to option prices. Initial premiums use an audited Fourier grid
and assume zero variance risk premium. BS delta/gamma at current volatility
are proxy features/baselines. The main Heston experiment observes simulated
variance; its information advantage is explicit. Bootstrap initial premiums
use a common BS reference quote and are not claimed arbitrage-free prices.

Bootstrap confidence intervals condition on a fixed return library. Thousands
of resampled paths do not create thousands of independent historical days or
measure parameter-estimation uncertainty. No Sharpe ratio is manufactured by
putting unrelated Monte Carlo paths into an artificial calendar order.

The underlying empirical libraries contain **266 training, 54 validation and
30 test daily returns**. The independent Fourier audit's maximum interpolation
errors are 6.37e−7 (base) and 5.51e−7 (stress). A separate antithetic Monte Carlo
check compares premiums with payoffs at six fixed states for each parameter
set. The largest paired mean call-price change between 320 and 1,280 steps is
4.06e−6 (base) and 1.14e−5 (stress); these are observed numerical diagnostics,
not universal error bounds.

### Actual Phase 5 results

Each row uses its own source-validation-selected policy. No test-world winner
is promoted from the 3 × 3 matrix. Lower loss CVaR95 is better.

| Evaluation world | Matching neural CVaR95 | Delta CVaR95 | Tuned constant band | Neural reduction vs delta |
|---|---:|---:|---:|---:|
| GBM | 0.007382 | 0.008463 | 0.008163 | 12.8% |
| Heston | 0.009630 | 0.010269 | 0.010070 | 6.2% |
| Bootstrap | 0.007981 | 0.008782 | 0.007498 | 9.1% |

The matching-world neural-versus-delta CVaR differences have conditional 95%
paired intervals wholly below zero. They do not include calibration/library
uncertainty or multiplicity correction across every experiment. The bootstrap
constant band beats the matching neural policy on both CVaR95 and mean P&L.
Bootstrap-trained policies also degrade when transferred to GBM/Heston. The
study therefore does not support universal neural-policy superiority.

On the GBM alpha sweep, training alpha 0.50/0.75/0.90/0.95/0.99 gives test
CVaR95 of 0.008199/0.007795/0.007399/0.007382/0.007615. CVaR99 is lowest for
the alpha=0.99 candidate (0.010242), while alpha=0.95 is lower at CVaR95.
These diagnostics do not trigger alpha reselection from test outcomes.

Increasing decisions from 5 to 80 reduces GBM neural CVaR95 from 0.012494 to
0.005945, but increases normalized mean cost from roughly 0.00134 to 0.00218.
The finer grid is not a free improvement: it exchanges higher turnover/cost for
smaller hedging errors. Frequency comparisons use separate common-path banks,
so their 20-step values differ slightly from the main-world test values.

Removing access to Heston latent variance raises the Heston-trained policy's
CVaR95 from 0.009630 to 0.010625 (about 10.3%). Proxy-delta CVaR95 on the same
information is 0.011543. The policy still improves that benchmark's tail loss,
but the oracle-information main result is optimistic relative to this proxy.
This is a frozen-policy information stress, not a new observable-state-trained
model or an actual market backtest. All matching-world neural mean P&Ls remain
negative after costs in these normalized hedging experiments.

## Corrections and preservation

The low-volatility audit detected rare float32 negative BS premiums, with
minimum −1.74e−8 on 100,000 audited paths. Premiums now use direct call/put
formulas in float64 with erfc and financial bounds, then cast back to the path
dtype. Greeks retain their original computation for hedge checkpoint
compatibility. Phase 4 and Phase 7 as-run source snapshots, paths and results
are preserved; their hedge actions replay against the saved banks.

The queued original Phase 5 worker was stopped after Phase 7 completed, before
its training/test evaluation. Its setup-only `phase5` directory is preserved
for audit. `phase5_corrected` reruns the same predeclared specification with the
pricing correction, rather than changing parameters in response to test scores.

Early warm-start checkpoint headers inherited the final trained policy's
validation score. Their immutable epoch-zero curves contain the correct warm
score. The loader reconciles this metadata, retains the original header value,
and does not change weights, risk-policy selection or evaluation results.
Future warm checkpoints write the correct score directly.

Before these extensions, the continuous risk objectives and comparisons were
implemented through Phase 3. The added experiments preserve that earlier
delivery and do not replace its as-run evidence with results generated by a
newer pricing implementation.

## Reproduction and checks

Run from the repository root using the project virtual environment. Each
command requires a **new** output directory and preserves existing artifacts.

```powershell
python src/run_phase4_research.py --output NEW_PHASE4_RUN
python src/run_phase7_research.py --phase4 PHASE4_RUN --output NEW_PHASE7_RUN
python src/run_phase5_research.py --daily DAILY_PARQUET --output NEW_PHASE5_RUN
python src/report_extended_research.py phase4 --run-dir PHASE4_RUN --output NEW_REPORT
python src/report_extended_research.py phase7 --run-dir PHASE7_RUN --output NEW_REPORT
python src/report_extended_research.py phase5 --run-dir PHASE5_RUN --output NEW_REPORT
python src/verify_extended_research.py phase4 --run-dir PHASE4_RUN --output NEW_VERIFICATION.json
python src/verify_extended_research.py phase7 --run-dir PHASE7_RUN --output NEW_VERIFICATION.json
python src/verify_extended_research.py phase5 --run-dir PHASE5_RUN --output NEW_VERIFICATION.json
python src/evaluate_extension_sensitivity.py bridge --run-dir PHASE7_RUN --output NEW_SENSITIVITY
python src/evaluate_extension_sensitivity.py observable --run-dir PHASE5_RUN --output NEW_SENSITIVITY
python src/audit_extension_pricing.py --run-dir PHASE5_RUN --output NEW_NUMERICAL_AUDIT.json
```

`experiment_plan.json`, as-run `source_snapshot`, `frozen_before_test.json`,
saved test banks/executions, full CSV comparisons and `verification.json`
provide the experiment trail. `full_replay_verification.json` independently
reloads all checkpoint executions and checks the stock/cash reconciliation.

Final checks: **182 regression tests pass**, **105 frozen checkpoint files**
(93 risk-training fits plus 12 imitation-only checkpoints) retain their hashes,
and **124 saved execution banks replay exactly**. Twelve additional stress-table
rows and the warm-only diagnostics reconcile. Maximum stock/cash discrepancy
across the saved-bank checks is 3.58e−7 normalized units. All **40 previously
protected artifacts** still match the earlier registry. No classifier was
retrained or deployed. Machine-readable delivery evidence is saved in
[delivery_verification.json](data/research/research_extension_delivery_20261031/delivery_verification.json).

Remaining later phases are GRU/ablations/ensembles (6), objective-defined signal
handoff (8), operations hardening and genuinely forward validation (0b/9).
These experiments establish a research implementation, not real-world
profitability or production readiness.

Method references: [Deep Hedging](https://arxiv.org/abs/1802.03042) motivates
training hedge policies against risk under transaction costs. The
[Whalley–Wilmott implementation documentation](https://pfnet-research.github.io/pfhedge/generated/pfhedge.nn.WhalleyWilmott.html)
states the small-cost, exponential-utility benchmark used here. This project
does not assert its CVaR-trained neural policies reproduce that asymptotic
optimum or an exact continuous-time hedge.
