# NIFTY Options Research Platform: full system report

Status date: 1 November 2026. Everything below describes the repository as committed (`06aa66d`). Numbers are retrospective unless a section says otherwise; the forward test (Phase 9) has **not** started.

How to read this: Parts 1-3 teach the problem and the architecture. Part 4 walks through every component in the order data flows, always answering *what it does*, *why it is built that way* and *what it does not show*. Part 5 collects the results. Part 6 explains the "why" behind the surprising outcomes. Part 7 lists the loopholes. Part 8 is the plan for the open work. Part 9 is the interview pitch. Part 10 is a one-page cheat sheet.

---

## Part 1. The one-paragraph story

NIFTY 50 index options are priced by a model (Black-Scholes). Every option also trades at a market price that is slightly "off" from what the surrounding strikes imply. The project asks four questions in order: (1) can we measure that gap reliably, (2) can a classifier predict that the gap will close within 30 minutes, (3) does trading on that prediction (with or without hedging) make money after real Indian costs, and (4) can we build an experiment, with rules fixed in advance, that can answer (3) honestly on data nobody has seen. The answers so far: (1) yes, after finding and fixing a data-timing bug that corrupted 8.45% of labels; (2) yes, the classifier has real signal (72.7% vs 64.5% for guessing, AUC 0.79; 70.5% vs 61.3% on a separate earlier year); (3) no, round-trip costs of roughly INR 430-450 per trade swamp a gross price move that is statistically zero; (4) built and tested, frozen-protocol machinery ready, observation window not started.

So the system's real product is not a trading strategy. It is a **trustworthy way of finding out that a strategy does or does not work**, with a negative result that is documented rather than hidden.

---

## Part 2. Domain primer: the ideas the whole system rests on

### 2.1 Options in one page
- A **call** option gives the right to buy NIFTY at a fixed **strike** on the **expiry** date; a **put** gives the right to sell. NIFTY has weekly expiries (Thursday in the data).
- Its price (the **premium**) depends on spot, strike, time left, interest rate and expected volatility. Of these, only volatility is not directly observed.
- **Lot size**: contracts trade in lots of fixed unit size (75 historically; 65 in the live instrument master). The project holds 4 lots per position.
- **ATM / OTM / ITM**: at-the-money (strike near spot), out-of-the-money, in-the-money.

### 2.2 Black-Scholes and implied volatility (IV)
- Black-Scholes gives a fair price from spot, strike, time, rate and volatility. The project uses the European form with a **constant 6% risk-free rate** (`config.py`, `black_scholes.py`).
- **Implied volatility** is the volatility that makes the model price equal the market price. It is found numerically (root-finding, Brent method) in `implied_volatility.py`. `vector_pricing.py` is a fast batch version and a test proves it equals the scalar version.
- **Why IV matters here**: option prices are quoted in rupees but compared across strikes in volatility terms. If a strike's IV sits away from its neighbours' IV, its price looks "off".

### 2.3 Greeks
Sensitivities of the option price: **delta** (to spot), **gamma** (delta's change), **vega** (to volatility), **theta** (to time). Delta is what a hedger neutralises with futures. Greeks are also model inputs to the classifier (`greeks.py`).

### 2.4 The reference surface and the "deviation"
- For each timestamp, expiry and option type, the system estimates a smooth IV from the *other* strikes (**leave-one-out**) by interpolating in log-moneyness (`build_reference_iv.py`). Extrapolation at the edges is capped at 0.03 log-moneyness; weak or distant estimates stay missing.
- The **reference price** is the Black-Scholes price at that reference IV. The **deviation** is `market price - reference price`. Negative means the option looks cheap versus its neighbours (this is the side the project trades: long only).
- Honest status of this idea: it is a *statistical benchmark*, not proof of arbitrage. A deviation can be noise, a stale quote, or a genuine mispricing that is not tradable after the spread.

### 2.5 The label (what the classifier predicts)
`convergence.py`: "30 minutes later, has the absolute deviation fallen by at least 50%?" (and the starting deviation is at least 1 index point). This is an **endpoint** label (looks at the value at the 30-minute mark), not "touched at any time" (first-passage). Endpoint is stricter and closer to what a trader holding for 30 minutes actually experiences.

### 2.6 Hedging
Buying an option exposes you to spot moves. A **delta hedge** holds futures against the option's delta to cancel that exposure; rebalancing as delta changes costs brokerage, taxes and spread every time. A **no-trade band** only rebalances when delta drifts far enough, trading hedge quality for lower cost. **CVaR95** is the average loss in the worst 5% of outcomes: the risk measure the hedging research tries to reduce.

### 2.7 Why a "frozen protocol" at all
If you test many ideas on the same history, one will look good by luck. A **frozen protocol** writes the pass/fail rules, the allowed code (by hash) and the model *before* new data exists, in a hash-chained file. Later edits are detected. That is what makes the final result believable.

---

## Part 3. System architecture

### 3.1 Layers and data flow

```text
                         ┌────────────────────────────────────────────┐
  Upstox REST API v2     │ 1. COLLECTION (read-only)                   │
  expired + live candles │ upstox_api, collect_all/expiry, quote_capture│
  instrument master      └───────────────┬────────────────────────────┘
                                         │ raw expiry Parquet (+ manifests)
                         ┌───────────────▼────────────────────────────┐
                         │ 2. MASTER DATASET + TIMING ALIGNMENT         │
                         │ master_build, candle_alignment, repair_*     │
                         └───────────────┬────────────────────────────┘
                                         │ master_options.parquet
              ┌──────────────────────────▼───────────────────────────────────┐
              │ 3. FEATURE / LABEL PIPELINE (one stage = one script, Parquet) │
              │ pricing+IV+Greeks → historical vol → reference IV surface →    │
              │ deviation → eligibility → put-call parity → labels → ML set    │
              └──────────────┬───────────────────────────────┬───────────────┘
                             │                               │ same stages as pure functions
                 ┌───────────▼──────────┐          ┌─────────▼───────────────┐
                 │ 4. MODELS            │          │ 4b. chain_features.py    │
                 │ accuracy_v2: 41 causal│          │ bar-by-bar live service  │
                 │ features, 6 families, │◄─────────┤ (parity-tested vs batch) │
                 │ purged CV, stack      │          └──────────────────────────┘
                 └───────────┬──────────┘
                             │ probabilities
      ┌──────────────────────▼───────────────────────────────────────────────┐
      │ 5. DECISION + HEDGING RESEARCH                                        │
      │ signal_handoff (ML vs random/max-dev/ATM) · deep_hedging (GBM/Heston,  │
      │ historical, costs, engine) · continuous_hedging (MLP/GRU, CVaR, lots)  │
      └──────────────────────┬───────────────────────────────────────────────┘
                             │ candidate policies + cost model
      ┌──────────────────────▼───────────────────────────────────────────────┐
      │ 6. PAPER TRADING (no orders can be placed)                            │
      │ forward_paper (frozen account) · paper_ops (contracts, state builder,  │
      │ daily runner, liquidation, health, single-writer service, scheduler)   │
      └──────────────────────┬───────────────────────────────────────────────┘
                             │ session folders: ledgers, events, reports
      ┌──────────────────────▼───────────────────────────────────────────────┐
      │ 7. FORWARD VALIDATION (frozen)                                        │
      │ protocol (hash chain) · observations (SQLite) · reconcile · acceptance │
      └──────────────────────────────────────────────────────────────────────┘
   Cross-cutting: tests (309 offline, network blocked), CI (GitHub Actions),
   ruff + mypy, registries of hashes in data/research, docs, project page.
```

### 3.2 Design principles (the "why" behind the structure)
1. **One stage, one script, one Parquet file.** Each stage can be rerun and inspected alone. Cost: many files. Benefit: you can point at the exact table where any number came from.
2. **Never overwrite.** Runs go to new dated folders with a source snapshot and hashes (`research_registry.py`, `run_aligned_research.py`). Old results stay reproducible.
3. **Causal by construction.** Every feature at time *t* uses only information available at *t*. This is *tested* (features must not change when all later data is deleted).
4. **Read-only toward the broker.** The code cannot place orders. There is no order endpoint in the code path.
5. **Pre-registration.** Entry rules, thresholds and acceptance criteria are fixed (and hashed) before the data that tests them exists.
6. **Negative results are results.** Reports state failures with the same prominence as successes.
7. **Frozen modules.** `forward_paper.py`, `research_targets.py`, `deep_hedging/*`, `forward_validation/*` are hash-pinned by protocols and registries. Editing them invalidates evidence, so they are exempt from lint and refactoring.

### 3.3 Repository map
| Folder | Contents |
|---|---|
| `src/` | 84 top-level modules plus packages `accuracy_v2`, `continuous_hedging`, `deep_hedging` (frozen), `forward_validation` (frozen), `paper_ops`, `signal_handoff` |
| `tests/` | 30 test modules, 309 tests, run with network blocked |
| `tools/` | offline test runner, clean-install check, scheduler scripts, demo and diagnostic scripts, parity and replay checks |
| `docs/` | reports, runbook, results summary, study guide, this report; `docs/results/` holds small evidence tables; `docs/archive/` holds dated reports |
| `data/` | local only (not in git): raw candles, features, models, run folders, paper sessions |
| `requirements*/` | base list plus optional sets (PyTorch CPU, CatBoost, reporting) |

---

## Part 4. Component deep dive

### 4.1 Collection (`upstox_api.py`, `http_transport.py`, `collect_all.py`, `collect_expiry.py`, `prepare_expiry.py`, `strike_selector.py`, `collect_spot_history.py`, `collect_hedging_futures.py`, `quote_capture.py`)
- **What**: pulls 5-minute option candles for expired weekly contracts (and the underlying spot, futures and India VIX) from the Upstox API; for live use, pulls current candles and best bid/ask quotes.
- **How**: per expiry, it chooses strikes around the spot (`strike_selector`), downloads each contract's candles for the last N sessions before expiry, and writes one Parquet plus a manifest (rows, contracts, failures). It never overwrites existing raw files.
- **Why this shape**: the API throttles (HTTP 429 happened during the older-expiry collection), so collection is resumable and writes manifests so you can prove completeness.
- **Limits**: candles only; **no historical bid/ask or depth**. This one fact is behind many of the system's limits.

### 4.2 Master dataset and the timing defect (`master_build.py`, `candle_alignment.py`, `repair_spot_alignment.py`)
- **What**: unions expiry files into `master_options.parquet` with columns for option OHLCV, open interest, `nifty_spot`, `spot_timestamp`, `feature_available_at`, `spot_available_at`.
- **The bug (the project's best story)**: a candle is stamped with its **start** time. A 09:15 five-minute option candle *closes* at 09:20. It had been paired with the one-minute spot candle also stamped 09:15 (closes 09:16), not the spot that closes at 09:20. Result: spot was typically wrong by a few index points (median 7.4, 95th percentile 31; 515,903 of 519,725 spot values changed), which corrupted IV, deviations and labels: **4,951 of 58,561 comparable labels (8.45%) flipped**.
- **The fix**: `candle_alignment.py` (37 lines) joins by *close time*, and every dataset, feature and model was rebuilt in an isolated run. Option candles now carry `feature_available_at = start + 5 min`: the earliest moment the data could be known live.
- **Why it matters for the pitch**: it shows you question your own data, quantify the error and rebuild, instead of patching.

### 4.3 Pricing, IV and Greeks (`build_features.py`, `black_scholes.py`, `implied_volatility.py`, `vector_pricing.py`, `greeks.py`, `time_to_expiry.py`)
- Time to expiry in years from the bar's availability time to expiry close (15:30). Moneyness = strike/spot and its log.
- IV by Brent root-finding; batch version vectorised; Greeks computed from the solved IV.
- Rows with zero price/volume/open interest are flagged, IV above 5.0 is treated as unreasonable.
- **Carry assumption**: constant 6%. Checked in `docs/CARRY_AND_PNL_CHECKS.md`: futures-implied carry has median 6.2%; re-running the chain at 3%/4.5%/7.5%/9% changes only 5-9% of labels, because IV and the reference surface are computed under the same rate so most of an error cancels.

### 4.4 Historical volatility (`build_volatility_features.py`)
5/10/20-day close-to-close volatility through the **previous completed** trading day only (never today), from an independent daily spot file. Used for IV-minus-HV features.

### 4.5 Reference IV surface, deviation, eligibility, parity (`build_reference_iv.py`, `build_deviation_features.py`, `build_signal_eligibility.py`, `build_parity_features.py`)
- **Reference IV** as in 2.4. **Deviation** columns: absolute and percentage, IV surface residual.
- **Eligibility** (`signal_eligible`): valid reference, price at least 1, time to expiry at least 0.25 day, IV between 1% and 300%, log-moneyness within ±0.15, volume and open interest at least 1, deviation finite. These filters exist because the others produce garbage (illiquid, expired, extreme strikes).
- **Put-call parity residual**: for a call and put with the same strike and expiry at the same time, `C - P` should equal `S - K·e^(-rT)`. The residual is a second, independent mispricing signal and a model feature.

### 4.6 Labels and ML dataset (`build_convergence_labels.py`, `convergence.py`, `build_ml_dataset.py`, `prepare_ml_split.py`)
- Label as in 2.5; the future candle may be up to five minutes after the nominal 30-minute horizon; missing future bars give no label (never filled).
- ML dataset rows = eligible, labelled, `|deviation| >= 1`.
- **Splits are chronological by whole trading date.** Label end times are carried so any training sample whose label window crosses into the validation block is **purged**, plus a one-session embargo.

### 4.7 Models and the accuracy programme (`train_models.py`, `walk_forward_benchmark.py`, `robust_model_research.py`, `data_remediation_study.py`, `accuracy_v2/*`)
- **Families**: logistic regression, random forest, extra trees, XGBoost, LightGBM, CatBoost (optional); small Optuna searches; final model is a **logistic stack** over members, with a decision threshold (0.53) chosen from validation data only.
- **accuracy_v2**: 41 additional causal features (deviation dynamics, staleness of quotes, smile fits, run-lengths of same-sign deviation), purged folds, a single frozen evaluation on later data (`frozen_before_fresh.json` is written before the later period is opened).
- **Result and why it is capped** (Part 6.1): 72.68% accuracy vs 64.49% majority, AUC 0.788; the earlier model got 72.20%. The 93-94% goal was investigated and ruled out.
- **Regime check** (`docs/OLDER_REGIME_HOLDOUT.md`): the frozen bundle applied once to 227 earlier sessions: 70.5% vs 61.3%, AUC 0.773, positive in every month.
- **Honesty flags**: the later period had been reviewed across earlier iterations, so it is not a pristine holdout; the earlier-year test checks regime transfer, not forward prediction.

### 4.8 Live feature service (`chain_features.py`)
- **What**: the batch stages (pricing → volatility → reference IV → deviation → eligibility → parity → v2 features) re-expressed as pure functions plus `ChainFeatureService.push(bar)` which takes one completed bar of the whole chain and returns candidate options with the frozen model's probability.
- **How it stays correct ("parity by construction")**: features are causal, so rerunning the batch functions on the session's table of completed bars gives, for the newest bar, exactly what the batch pipeline would store. Verified: stage outputs differ from stored datasets by exactly 0.0 on a recorded session; 136 model features matched on two bars; probabilities match stored predictions to about 1e-16 over three replayed sessions at roughly 1.5 s per bar (`tools/chain_parity_check.py`, `tools/live_replay_check.py`).
- **Status**: built and verified, **not connected** to the paper loop (the loop trades fixed contracts chosen before the day; dynamic entries would be a new mode and a new protocol).

### 4.9 Entry selection (`signal_handoff/*`)
Phase 8 asks: do classifier-chosen entries beat simple rules? Six decision slots a day, one option per slot, bought at the next bar's open, held 30 minutes, 4 lots. Rules: **ML** (highest probability above tau 0.5), and three controls in the same slots: **random** (seeded), **largest deviation**, **nearest ATM**. Selections were hashed *before* any P&L existed.
Findings: ML entries converge more often (67.1% vs 47.5% random, 50.3% largest-gap, n=760 paired), but this does not become better P&L after fees. Convergence of a gap is not option profit: the trade pays the option's own move minus spread, slippage and tax.

### 4.10 Hedging engine and costs (`deep_hedging/*` frozen, `continuous_hedging/*`)
- **Ledger** (`engine.py`, `accounting.py`): self-financing; cash, option and futures positions, margin; every trade pays costs.
- **Cost model** (`costs.py`, `config.py`): option half-spread 0.15 + slippage 0.15 (futures 0.10 + 0.10), 0.05 tick rounding, brokerage INR 20 per order, exchange, SEBI and IPFT charges, stamp duty, 18% GST on fees, date-dependent STT.
- **Baselines**: unhedged, static delta, dynamic delta, band delta.
- **Neural policies**: MLP/GRU policies trained with CVaR objectives on simulated GBM and Heston paths, then tested historically. Lot rounding (whole lots) is modelled explicitly.
- **Findings** (Part 5): hedging cuts the worst-5% loss by 59% but costs more than it saves; neural policies cut CVaR 6-16% vs delta in simulation but a tuned no-trade band has the better mean P&L; GRU adds nothing.

### 4.11 Paper trading (`forward_paper.py` frozen, `paper_ops/*`, `run_paper_day.py`)
- **Account** (`forward_paper.py`): quote checks (staleness, crossed book, depth), simulated fills at bid/ask with depth rule, risk latch (daily loss, drawdown, margin stops).
- **Contracts** (`paper_ops/contracts.py`): four contracts per week (ATM call, ATM put, ~1% OTM call, ~1% OTM put) from the nearest expiry with at least 3 trading days left, rolled when under 2 days remain or moneyness drifts over 2%. Validated against a dated instrument-master snapshot (`catalog.py`).
- **Accounts**: 4 contracts × 5 strategies = 20 independent paper accounts per day, one shared quote request per 5-minute snapshot.
- **State builder** (`state_builder.py`): builds decision packets from completed bars only (never partial bars), with a stale-spot guard.
- **Daily runner** (`daily.py`): schedule (open 09:35, six one-hour episodes, last close 15:05, retries to 15:20), preflight (token, master age, weekend), per-session service cycles, end-of-day report.
- **Service** (`service.py`): single-writer lock per session; **quarantine** for malformed or conflicting packets; **liquidation** logic and **health** alerts (stale feed while exposed, overdue open position, etc.).
- **Operations**: Windows scheduler scripts (task registered disabled), runbook with exit codes 0/2/3.

### 4.12 Forward validation (`forward_validation/*`, `forward_validation_cli.py`)
- **Protocol** (hash chain): allowed runner/runtime/policy hashes, execution config, risk and quote rules, numeric acceptance criteria. Editing it, or reordering the registry, is detected on load. A session counts only if registered **after** the freeze in `forward` mode with matching hashes.
- **Observations** (SQLite): idempotent per session-day aggregation of events, accepted/rejected with reasons, fills, fees, equity, drawdown, exposure at close, quarantine counts, forward eligibility.
- **Reconciliation**: rebuilds cash, positions and fees *from the fill ledger alone* and compares with what the runner recorded. A single altered fill is caught as a 130-rupee error (`tools/demo_tamper_check.py`).
- **Acceptance** (`acceptance.py`): **pending** below 60 forward days; **rejected immediately** on any integrity violation, reconciliation failure, unresolved exposure beyond tolerance, or drawdown above 10%; with 60+ days: event coverage ≥ 90%, acceptance ≥ 90%, quarantine ≤ 1%, CVaR95 reduction versus unhedged significant (paired day-cluster 95% interval excludes zero), and cost efficiency (P&L given up per episode / CVaR reduction) ≤ 0.75; exit codes 0 accepted, 3 pending, 2 rejected.
- **Shakedown**: first 3 distinct forward days are excluded from coverage and economic tests (integrity gates still apply). Up to 2 open-at-close session-days are tolerated as incidents.

### 4.13 Added checks after the main results
| Check | File | Finding |
|---|---|---|
| Does a constant 6% carry create fake mispricing? | `tools/carry_diagnostic.py`, `tools/rate_sensitivity.py` | No: futures-implied carry 6.2%; label agreement 91-97% across a 3-9% rate grid |
| Does anything predict **net profit**? | `accuracy_v2/pnl_experiment.py` | No: profit classifier AUC 0.505; convergence classifier's picks do no better than all candidates |
| How much of the loss is cost? | `tools/cost_decomposition.py` | Almost all: gross mean -39 (history) and +142 (later) INR, both statistically zero; cost 431-448 per round trip |
| Is the model transferable to another year? | `tools/older_regime_evaluate.py` | Yes: 70.5% vs 61.3%, AUC 0.773 over 227 earlier sessions |
| Can a clean clone run the pipeline? | `tools/synthetic_demo.py` | Yes, on invented data (proves the code runs; says nothing about markets) |

### 4.14 Engineering and tooling
- **Tests**: 309 `unittest` tests; the runner blocks sockets and blanks the token; data-dependent tests skip themselves on a clean clone. Categories: unit, accounting identities, leakage and prefix-invariance, fault injection, tamper detection, end-to-end on synthetic data.
- **CI** (`.github/workflows/offline-ci.yml`): Python 3.12 and 3.13 matrix, pip cache, ruff, mypy, then the offline suite.
- **Static checks**: ruff (syntax and pyflakes), mypy on `paper_ops`, `forward_validation`, `signal_handoff`; frozen modules intentionally not linted.
- **Reproducibility**: run folders with source snapshot and hashes; registries pin source files by path and hash.
- **Presentation**: project page (`build_project_dashboard.py`), demo scripts, README, results summary, study guide.

---

## Part 5. Technology stack

| Layer | Tools | Why this choice |
|---|---|---|
| Language | Python 3.12/3.13 | the quant/ML ecosystem; CI-tested on both |
| Data | pandas, NumPy, PyArrow (Parquet) | columnar files, fast and typed; each stage inspectable |
| Numerics | SciPy (root-finding, normal CDF) | Black-Scholes and IV |
| ML | scikit-learn, XGBoost, LightGBM, CatBoost (optional), Optuna, joblib | tabular data suits boosted trees; small tuned searches for reproducibility |
| Deep learning | PyTorch (CPU) | MLP/GRU hedging policies; optional install |
| HTTP | requests, python-dotenv | Upstox REST API v2; token in `.env` (never committed) |
| Storage | Parquet (datasets), JSON/JSONL (events, protocols), SQLite (observations), joblib (models) | simple, portable, no server |
| Operations | PowerShell scripts, Windows Task Scheduler | the target machine is Windows; scheduled task registered disabled |
| Quality | unittest, coverage (48% of lines; low because runner/report scripts only run on real data), ruff, mypy | |
| CI/CD | GitHub Actions | offline, credential-free |
| Integrity | SHA-256 hash chains and registries | tamper-evident evidence |
| Reporting | matplotlib, generated HTML project page | |

---

## Part 6. Results and the "why" behind them

### 6.1 Why the accuracy stops near 72-73%
- The label compares a *last-traded price* with a *smoothed surface* 30 minutes later and asks whether the gap halved. About 15% of labels sit within 0.25 index points of the cut-off and 32% within 0.5, which is the size of a bid/ask spread. The data has no bid/ask, so those rows are decided by tick noise no model can see.
- Six model families and ~40 added features all plateau at the same AUC. When different models stall at the same score, the limit is in the *information*, not the model.
- Selective accuracy of 94.9% exists only at 22% coverage with 75 positive calls: impressive-sounding, not usable.
- Hence the honest ceiling is about 70-73%. A claim of 93-94% would require either leakage or relabelling the target.

### 6.2 Why entries that "converge more" still lose money
Convergence of the deviation is a statement about distance to a model price. A long option P&L is the option's own price move minus spread, slippage, brokerage, charges and STT. Decomposition (45,624 historical and 5,184 later trades): mean gross -39 and +142 INR (both intervals span zero) vs cost 431-448 INR. The cost is a fixed toll per round trip; the gross edge is nil.

### 6.3 Why hedging cuts risk but loses money
Per position (168 positions, 28 sessions): unhedged +81k total with worst-5% loss 11.1k; dynamic delta -345k with 4.6k. Tail risk falls 59%. The price: about INR 2.5k of mean P&L given up per position for a 6.5k reduction in tail loss, a ratio of 0.39. Every five-minute rebalance pays costs; the hedge is genuinely protective but expensive at this fee level. A no-trade band reduces rebalances and has better mean P&L than the neural policy, so a simple baseline beat the sophisticated one.

### 6.4 Why the classifier does not help hedging
Same P&L with and without it, and valid probabilities existed on only 13 of 1,008 decisions: the hedging decisions rarely land where the classifier has an opinion.

### 6.5 Result table
| Track | Headline |
|---|---|
| Data | 941,949 five-minute candles, 60 weekly expiries, 283 sessions (Sep 2025-Oct 2026); plus 111,900 rows over 227 sessions in the earlier year |
| Timing bug | 8.45% of labels flipped; full rebuild |
| Convergence classifier | 72.68% vs 64.49% majority, AUC 0.788; earlier year 70.5% vs 61.3%, AUC 0.773 |
| Entry selection | ML 67.1% converge vs 47.5% random; no P&L benefit after fees |
| Hedging | tail risk -59%; fees exceed the gain |
| Net profit prediction | none (AUC 0.505) |
| Cost split | cost 431-448 INR per round trip; gross edge ≈ 0 |
| Paper machinery | 20 accounts, 840 events in a full-session replay, all reconcile, nothing open |
| Tests | 309 offline tests passing; ruff and mypy clean |
| Forward test | protocol drafted and sensitivity-analysed; **not frozen, not started** |

---

## Part 7. Loopholes and weaknesses (candid)

### 7.1 Scientific
| # | Weakness | Why it matters | What is done / could be done |
|---|---|---|---|
| S1 | **No demonstrated edge.** Every tested rule loses after costs | The headline "so what" is negative | Stated plainly; forward test and cost-aware variants are the only routes |
| S2 | **No pristine holdout.** The later period was reviewed across iterations | Reported accuracy may be slightly optimistic through selection | Disclosed; earlier-year holdout added (checks regime transfer); forward test is the real confirmation |
| S3 | **Earlier-year test is backward in time** | It shows transfer, not forecasting | Disclosed in the holdout doc |
| S4 | **Thin history** (about one year of main data plus one earlier year; two regimes) | Rare-regime behaviour untested (crashes, event days) | Collect more years if the API allows; forward test adds live regimes |
| S5 | **Model-dependent reference price** (constant rate, no forwards or dividends) | "Mispricing" may be partly model error | Carry sensitivity shows small effect; forward-implied rebuild is optional follow-up |
| S6 | **Label noise floor** (no bid/ask) | Caps accuracy at ~72% | Needs recorded quote data (see Part 8) |
| S7 | **Multiple testing** across many experiments | Some "significant" intervals could be luck | Pre-registration and the forward test; additional variants must be pre-registered |

### 7.2 Execution and cost realism
| # | Weakness | Detail |
|---|---|---|
| E1 | Fills at candle opens with an assumed spread | True spread and queue effects unknown until live bid/ask is recorded |
| E2 | **Order-book depth untested.** The runner rejects orders larger than the best displayed level; a position is 4 lots (260 units at lot size 65) | After hours, depth was one lot; many rejections possible in session |
| E3 | Short options excluded (no historical SPAN margin) | Strategy space is long-only |
| E4 | Costs are configured proxies | Brokerage and tax rates change; must match the real broker |
| E5 | No latency model | A 5-minute bar-based design hides sub-minute effects |

### 7.3 Validation and governance
| # | Weakness | Detail |
|---|---|---|
| V1 | Forward test not started | No clean evidence exists yet |
| V2 | Protocol parameters (0.75 ceiling, 3 shakedown days, 2-incident tolerance, contract set) not yet approved | Needs your decision before the freeze |
| V3 | Single developer, no independent review | Blind spots remain |
| V4 | Frozen modules exempt from lint | They are less clean by design |
| V5 | 60 days is an operational window, not a power guarantee | Statistical power analysis shows tail-risk claims resolve sooner, economic claims may not |

### 7.4 Operations
| # | Weakness | Detail |
|---|---|---|
| O1 | Token is short-lived and refreshed manually each morning | A missed refresh loses a day |
| O2 | Single Windows machine | Sleep, network loss or reboot interrupts sessions; closes are retried until 15:20 |
| O3 | API throttling (HTTP 429 seen) | Missing bars; the runner skips events rather than fabricating |
| O4 | ML entries not live | The service is verified in replay but not wired to the paper loop |

### 7.5 Engineering
| # | Weakness | Detail |
|---|---|---|
| G1 | Coverage 48% | Runner and report scripts run only on real data |
| G2 | Flat `src/` layout | Cannot be changed without breaking path-pinned registries and documented commands |
| G3 | Many scripts of mixed ages | Some legacy routes require explicit opt-in flags |

---

## Part 8. How the open work will be completed

### 8.1 Dependency order
```text
approve protocol ─► freeze ─► shakedown (3 days) ─► forward window (60 days) ─► evaluate
        │                         │
        │                         └─► depth findings ─► (maybe) new protocol version
        └─► record bid/ask (in parallel, from day 1) ─► real cost model ─► cost-aware variants (new pre-registered experiments)
live ML mode (new protocol) ◄─ only after Phase 9 baseline is running
```

### 8.2 Work items
| # | Item | Steps | Done when | Time |
|---|---|---|---|---|
| 1 | **Approve and freeze Phase 9** | You review `docs/PHASE9_PREFLIGHT.md`; confirm hashes unchanged; run the `freeze` command; commit the protocol file | `protocol_v1.json` and registry exist; evaluator exit code 3 (pending) | 1 hour |
| 2 | **Shakedown** | Refresh token, run `tools\run_live_day.ps1` each trading day; read the daily report; log rejects by reason | 3 days with all ledgers reconciling; depth rejection rate known | 3 trading days |
| 3 | **Depth decision** | If rejections are high: choose fewer lots, or a new execution rule under a **new protocol version** (never edit v1) | Rejection rate acceptable or v2 frozen | 1-3 days after shakedown |
| 4 | **60-day forward window** | Daily run; weekly `ingest`, `reconcile`, `evaluate` | Verdict accepted/rejected (or still pending with reasons) | ~3 months |
| 5 | **Record real bid/ask and depth** | The paper chain already stores quotes; add a compact quote archive | At least 4 weeks of quotes; empirical spread distribution replaces the 0.15+0.15 assumption | Parallel to 4 |
| 6 | **Cost-aware variants (new experiments)** | Pre-register a small set (e.g. larger-gap threshold, longer hold, profit target, liquid strikes only); select on history only; confirm on forward data | Gross edge reliably above ~INR 450 per trade, or a documented null | After item 5 |
| 7 | **Live ML entries** | New dynamic-session mode: push each completed bar into `ChainFeatureService`, pass candidates to the existing `ml_choice`, register sessions on demand; own protocol, own tests, default off | Paper run matches replay probabilities; frozen under protocol v2 | 1-2 weeks of work, after item 4 is running |
| 8 | **More history** | Retry collection of earlier expiries (before Sep 2024) in small batches with pauses; rerun the one-shot evaluator | Third regime evaluated | Hours to days (API-limited) |
| 9 | **Carry from forwards (optional)** | Replace constant 6% with futures-implied carry if live data shows drift | Rebuild and compare labels; accept only if it changes results materially | 1-2 days |
| 10 | **Independent review** | Ask a peer to run `tools/clean_install_check.py` and review one module; keep notes | Written feedback incorporated | 1 week |
| 11 | **Hygiene** | Tests for runner scripts; token-refresh helper; scheduled-task alerting | Coverage and ops checklist improved | Ongoing |

### 8.3 Decision tree for the forward verdict
- **Accepted**: the hedging claim (tail-risk reduction at acceptable cost) holds out of sample under frozen rules. Claim is limited to hedging, not to alpha.
- **Rejected**: integrity, reconciliation or economics failed. Report the reason; the platform still stands as a validation framework.
- **Pending**: fewer than 60 days or insufficient evidence. Extend the window under the same protocol; never loosen the criteria.

---

## Part 9. How to pitch it in an interview

### 9.1 The positioning (one sentence)
> "I built a research platform that tests whether a trading idea survives real costs and honest validation, and it told me the idea does not, which is exactly the answer a desk wants *before* it puts capital behind it."

Do **not** pitch it as a money-making strategy. Interviewers who know markets will probe that, and you have the evidence that it is not. The strength is the *judgement and rigour*.

### 9.2 What the "business impact" really is
Frame impact as **decision quality, capital protection, risk and operational trust**, with your own numbers:
1. **A bad strategy was stopped before it cost money.** Hedged ML-selected entries lost INR 2.15M over 760 paper episodes (about INR 2,834 each, win rate ~10%). Naive "convergence" accuracy looked good (72.7%); only cost decomposition showed gross edge ≈ 0 vs INR 430-450 toll per trade. That is the financial value of rigorous testing.
2. **Data quality caught before it propagated.** A one-line join error corrupted 8.45% of labels. Finding and quantifying it (and rebuilding) is the same skill that prevents silent P&L misstatements in production.
3. **Priced the cost of risk protection.** Hedging reduced the worst-5% loss by 59% at about INR 2.5k mean P&L per position, a cost-efficiency ratio of 0.39. A desk can compare that with the price of other protection.
4. **Built tamper-evident controls.** Hash-chained protocol, independent ledger reconciliation (an altered fill caught as a 130-rupee error), no-order architecture. This maps to model risk management, audit and compliance requirements.
5. **Reproducibility.** 309 offline tests with the network blocked, CI on two Python versions, run folders that never overwrite, and a synthetic demo so anyone can run it. That is what makes research reviewable and handoverable.

Who values this: quant research and strategy teams, model validation and risk teams, systematic trading firms, brokerage/fintech data science teams, execution/TCA teams, any ML role where leakage and evaluation honesty matter.

### 9.3 The 30-second version
"I built an end-to-end platform for NIFTY options: data collection, pricing, a classifier that predicts whether a mispricing closes, hedging research, a paper-trading engine, and a pre-registered forward test. The classifier has real signal, 72.7% against 64.5% for guessing, but after Indian transaction costs there is no edge. I found and fixed a timing bug that corrupted 8.45% of labels, and I designed the validation so I cannot fool myself."

### 9.4 The 2-minute version (structure: problem → what I built → what I found → what I'd do next)
1. **Problem** (15 s): "Option prices sometimes deviate from what neighbouring strikes imply. Does that deviation predict something tradable?"
2. **Build** (30 s): "Pipeline from API candles to a leave-one-out volatility surface to a causal-feature classifier with purged time-ordered validation; a cost-accurate hedging engine; 20-account paper trading with independent reconciliation."
3. **Findings** (45 s): "Signal is real and stable across two years (AUC 0.79 and 0.77). But gross price moves after entry are zero and costs are INR 430-450 per round trip, so every rule loses. Hedging cuts tail loss 59% but costs more than it saves."
4. **Judgement** (30 s): "I tried to hit 93% accuracy as a stress test and showed it is impossible because a third of labels are decided by tick noise. I found my own timing bug. The next step is a 60-day forward test with frozen rules."

### 9.5 The 5-minute demo (reuse `docs/DEMO_SCRIPT.md`)
Project page → accuracy and noise-floor explanation → entry selection vs controls → run the 25-second paper-day replay → run the tamper demo → engineering/limits → forward test status. Keep one terminal ready; practise the commands.

### 9.6 Tailoring by interviewer
| Interviewer | Lead with | Go deep on | Avoid |
|---|---|---|---|
| Quant / researcher | label design, leave-one-out surface, purged CV, cost decomposition | why gross edge is zero; label noise floor; multiple testing | claiming alpha |
| Data scientist / ML | leakage controls, prefix-invariance tests, stack and threshold selection on validation only | why 93% is impossible; calibration; selective accuracy vs coverage | accuracy without baseline |
| Risk / model validation | frozen protocol, reconciliation, tamper demo, quarantine | what the protocol protects against; incident tolerance | glossing over the unfrozen status |
| Software / platform | architecture, hash-pinned modules, CI, single-writer service, offline tests | parity by construction for the live feature service | over-explaining finance |
| Product / fintech | "a tool that tells you not to trade", cost transparency for users | how costs destroy small edges; honest metrics | jargon |

### 9.7 Questions to expect (short answers)
- **"Is it profitable?"** No. The classifier selects better entries (67% vs 47% convergence) but gross price moves are zero and costs are INR 430-450 per trade. That result is the finding.
- **"Why not 93%?"** A third of labels are within the bid/ask spread of the cut-off; the data has no bid/ask. Six model families plateau at the same AUC. Only different data or a different target could change that.
- **"How do you know there's no leakage?"** Purge plus embargo, features tested to be unchanged when later data is removed, shuffled labels give AUC near 0.5, thresholds fixed on earlier data only.
- **"You looked at the test period; is it a holdout?"** No, and I say so. I then evaluated the frozen model once on a never-inspected earlier year and the forward test is the confirmation.
- **"What was your hardest bug?"** The candle timing defect: spot paired by start time instead of close time. Quantified, fixed, rebuilt.
- **"What would you do differently?"** Record bid/ask from the start; pre-register earlier; avoid reviewing the test period more than once.
- **"What would change your conclusion?"** Narrower real spreads, or a fee-aware entry rule that shows a gross edge above ~INR 450 per trade in the forward data.
- **"Why public?"** The code and the negative results are the portfolio piece; no credentials or market data are in it.
- **"What's left?"** Freeze and run the 60-day forward test, record quotes, then pre-registered cost-aware variants and a live ML mode.

### 9.8 Stories to have ready (STAR)
1. **Timing bug**: Situation (accuracy plateau, odd spot values) → Task (find cause) → Action (compare spot timestamps, quantify, write alignment, rebuild everything in an isolated run) → Result (8.45% of labels flipped, results restated, 37-line fix with tests).
2. **93% target**: Situation (pressure for high accuracy) → Action (stress-tested six families and 40 features, measured the noise floor) → Result (documented why it is impossible, refused to inflate it).
3. **Negative result**: Situation (classifier works) → Action (decomposed P&L into gross and cost) → Result (found the edge is zero, saved effort chasing it).

### 9.9 What not to say
"Beats the market", "profitable", "arbitrage found", "93%". Also do not claim the forward test is running or that results are confirmatory.

### 9.10 Rehearsal plan
1. Say the 30-second version aloud until it takes under 40 seconds.
2. Run the replay and tamper demos twice each.
3. Memorise the ten numbers in Part 10.
4. Practise three questions you are afraid of (profitability, holdout, 93%).
5. Prepare one question for them: "How do you decide when a research idea is dead?"

---

## Part 10. Cheat sheet

| Fact | Number |
|---|---|
| Main dataset | 941,949 candles, 60 weekly expiries, 283 sessions |
| Earlier-year holdout | 111,900 rows, 227 sessions |
| Timing bug | 8.45% of labels (4,951 of 58,561) |
| Accuracy | 72.68% vs 64.49% majority; AUC 0.788 (earlier year: 70.5% vs 61.3%; AUC 0.773) |
| Prior model | 72.20% (gain +0.48 points, interval -0.13 to +1.10) |
| Entry selection | 67.1% vs 47.5% random (760 paired entries) |
| Hedging tail effect | worst-5% loss 11.1k → 4.6k (-59%) |
| Cost per round trip | INR 431-448 |
| Gross edge | -39 (history), +142 (later); intervals span zero |
| Profit classifier AUC | 0.505 |
| Tests | 309 offline, network blocked |
| Paper machinery | 20 accounts, 840 events in a full-session replay, all reconcile |
| Forward window | 60 days after 3 shakedown days; cost ceiling 0.75; status: not started |

*Everything in this document is a description of research software. Nothing in it is investment advice or evidence of a profitable strategy.*
