# NIFTY Options Research Platform

Pricing, convergence prediction, hedging experiments and a frozen forward-validation protocol for NIFTY index options,
built on Upstox market data. **Research system: no profitable trading edge is claimed, and no orders are ever placed.**

## At a glance

| | |
|---|---|
| Data | 941,949 five-minute option candles, 60 weekly expiries, 283 sessions, plus spot, futures and India VIX |
| Prediction | 72.7% accuracy on later data vs 64.5% for always guessing the common outcome (AUC 0.788). A 93-94% target was investigated and ruled out: about a third of labels sit within half an index point of the cut-off, so tick-level noise caps the score ([report](docs/ACCURACY_V2_REPORT.md)) |
| Entry selection | Classifier picks converge 67.1% of the time vs 47.5% for random picks among the same candidates, but this does not become a detectable profit after costs ([Phase 8](docs/PHASE8_SIGNAL_HANDOFF.md)) |
| Hedging | Delta hedging cuts worst-case loss by more than half but fees exceed the gain; a GRU policy is no better than an MLP ([Phase 6](docs/RESEARCH_PHASE_6.md)) |
| Rigor | Fixed a spot/option timing bug (changed 8.45% of labels); purged time-ordered validation; leakage tests; 284 offline tests with the network blocked |
| Evidence integrity | Hash-chained validation protocol, independent ledger reconciliation, tamper detection ([Phase 9](docs/PHASE9_FORWARD_VALIDATION.md)) |
| Forward test | Protocol drafted, **not yet frozen**; needs about 60 trading days of real quotes. Until then every number here is retrospective |

Small result tables and figures are in [docs/results/](docs/results). Large data, models and run folders are not in git.

### Repository layout
```text
src/            pipeline, hedging research, paper trading and validation code (module map: src/README.md)
tests/          284 offline tests (unit, accounting, leakage, fault injection, end to end on synthetic data)
tools/          offline test runner, clean-install check, scheduler scripts, demo scripts
docs/           reports, runbook and results index (docs/README.md); docs/results/ holds small evidence tables
requirements/   optional dependency sets (PyTorch CPU, reporting, CatBoost); base list is requirements.txt
data/           local only: raw candles, features, models and run folders (not committed)
```

### Run the tests (no API token needed; a few tests that need locally generated research artifacts skip or report visibly on a fresh clone)
```powershell
python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt
$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe tools\run_offline_tests.py     # sockets blocked, empty token
```
Paper-trading replay and live operation need a stored dataset or an Upstox token; see [DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md) and
[RUNBOOK_PAPER_OPERATIONS.md](docs/RUNBOOK_PAPER_OPERATIONS.md). Copy `.env.example` to `.env` for live data.

---

## Original pipeline notes

This repository is a historical research pipeline for NIFTY index options. It builds point-in-time pricing and volatility features, labels 30- and 60-minute deviation convergence, trains a 30-minute classifier, creates candidate trades, and evaluates them using a conservative OHLC execution model.

The [26 October 2026 re-audit](docs/ML_REAUDIT.md) identified a historical spot/option candle-close alignment problem. The subsequent [recovery investigation](docs/ACCURACY_RECOVERY_20261030.md) obtained authentic matching data and rebuilt a separate synchronized research pipeline. Older active artifacts remain preserved and must not be mixed with the corrected research models/context. The audit also documents the coverage limits of selective accuracy.

It is a research project, not a live broker or order execution system. No accuracy or profitability is guaranteed.

## Pipeline

```text
Upstox expiry candles + NIFTY spot
  -> raw expiry Parquet files
  -> master_options.parquet
  -> options_features.parquet
  -> options_volatility_features.parquet
  -> options_reference_iv.parquet
  -> options_deviation_features.parquet
  -> options_signal_eligible.parquet
  -> options_parity_features.parquet
  -> options_convergence_labels.parquet
  -> ml_dataset.parquet + chronological splits
  -> validation-tuned models + test evaluation
  -> prediction candidates -> next-bar backtest -> daily risk report
```

## Setup

Accuracy programme v2 (new causal features, model search, single frozen evaluation; 93-94% not attainable) is in
[ACCURACY_V2_REPORT.md](docs/ACCURACY_V2_REPORT.md).

Phases 6, 8, 0b and 9 (GRU/ablations/ensembles, ML-selected-entry handoff, operational hardening, forward-validation
infrastructure) are documented in [IMPLEMENTATION_LOG_PHASES_6_9.md](docs/IMPLEMENTATION_LOG_PHASES_6_9.md),
[OPERATIONS_PHASE_0B.md](docs/OPERATIONS_PHASE_0B.md) and [PHASE9_FORWARD_VALIDATION.md](docs/PHASE9_FORWARD_VALIDATION.md).
Phase 9 forward validation is **pending** genuine new market observations.

The continuous research extension through phases 0a–3 is documented in
[CONTINUOUS_HEDGING_PHASES_0_3.md](docs/CONTINUOUS_HEDGING_PHASES_0_3.md).
It is separate from the preserved executable whole-lot experiments.
`research_runs.json` identifies the canonical corrected research locations;
`research_registry.py` freezes/verifies their hashes before inference.
Legacy `data/processed` training/signal CLIs now require explicit
`--allow-legacy-artifacts`; custom legacy-format models require matching context.

Part 4 adds an offline results dashboard and persistent quote-based paper
execution: see [PART4_DASHBOARD_AND_PAPER_TRADING.md](docs/PART4_DASHBOARD_AND_PAPER_TRADING.md).
Frozen protocols, depth/latency checks, restart-safe ledgers and monitoring are
implemented. Synthetic smoke runs are labelled; genuine forward validation is pending.

Part 3 historical ML-assisted hedging is implemented: see
[PART3_HISTORICAL_ML_HEDGING.md](docs/PART3_HISTORICAL_ML_HEDGING.md) for expanding-window
classifier inputs, matched ML/no-ML policy adaptation, common-position historical
comparisons, costs, risk metrics and replay. These are retrospective research
results; incremental ML benefit and deployment quality must be measured separately.

Part 2 is implemented and trained: see [PART2_DEEP_HEDGING.md](docs/PART2_DEEP_HEDGING.md)
for the CPU PyTorch policy, GBM/Heston simulation, CVaR objective and independent
test results. The neural policies reduce risk versus unhedged positions but do
not beat the strong delta baselines in the primary simulation tests.

Part 1 hedging foundation is implemented: see
[PART1_HEDGING_FOUNDATION.md](docs/PART1_HEDGING_FOUNDATION.md) for futures collection,
the self-financing ledger, four conventional baselines, tests and historical
results. Part 3 adds the historical ML-assisted hedging and delta comparison.

Latest investigation: [ACCURACY_RECOVERY_20261030.md](docs/ACCURACY_RECOVERY_20261030.md).
The authenticated API connection now works with verified system trust, missing
expiries have been backfilled, and a separate synchronized 941,949-row research
pipeline has been rebuilt. Use the model and feature context from the same run;
the old active weights are preserved and are not interchangeable with new fits.

Use Python 3.10 or later and install the packages in `requirements.txt` into the project environment. Set `UPSTOX_ACCESS_TOKEN` in `.env` before collection. Offline feature building, research and configuration imports do not require an API token; authenticated requests do.

Read [DATA_MODEL_REMEDIATION_REPORT.md](docs/DATA_MODEL_REMEDIATION_REPORT.md) before rebuilding the data. It documents the candle alignment error, safe staged repair, coverage audit, target experiments and provisional validation results. Authentic matching five-minute spot candles or correctly offset one-minute spot candles can repair alignment. Research candidates remain separate from the active model.

## Reproduce the pipeline

Run each command from the repository root. Collection can be skipped when raw expiry files already exist. `collect_all.py` supports `--run-mode`, expiry bounds, session count and `--plan-only`, and preserves expiry files already present. The commands below describe the standard pipeline; follow the staged repair instructions in the remediation report for the currently misaligned historical data.

```powershell
python src/collect_all.py
python src/master_build.py
python src/build_features.py
python src/build_volatility_features.py
python src/build_reference_iv.py
python src/build_deviation_features.py
python src/build_signal_eligibility.py
python src/build_parity_features.py
python src/build_convergence_labels.py
python src/build_ml_dataset.py
python src/prepare_ml_split.py
python src/walk_forward_benchmark.py
python src/train_models.py --allow-legacy-artifacts
python src/generate_signals.py --allow-legacy-artifacts
python src/generate_trade_signals.py --allow-legacy-artifacts
python src/backtest.py
python src/risk_analysis.py
python src/show_eval_metrics.py
```

The signal generator reads the probability cutoff saved by training from validation data. Do not choose thresholds from test results. To run focused regression tests without installing `pytest`:

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_point_in_time_features tests.test_ml_evaluation_safety tests.test_backtest_accounting tests.test_risk_analysis tests.test_financial_calculations tests.test_candle_alignment tests.test_robust_model_research tests.test_convergence_labels tests.test_data_remediation -v
```

## Quantitative assumptions

- Historical volatility uses daily close-to-close returns through the **previous completed trading day**. The rolling windows require a full history before producing a value.
- Reference IV is grouped by timestamp, expiry, and option type. Each target contract is left out of its own estimate. Interior values use the nearest strike neighbors in log-moneyness; edge extrapolation is limited to 0.03 log-moneyness. Insufficient or distant estimates remain missing.
- Pricing uses European Black–Scholes with the configured constant risk-free rate. The reference surface is a statistical benchmark; a residual is not proof of executable arbitrage.
- The 30-minute label requires at least a 50% reduction in absolute reference-price deviation at the endpoint. The future candle can be up to five minutes after the nominal horizon. This is not an any-time/first-passage convergence label. The 60-minute endpoint label is retained for analysis; the current classifier target is 30 minutes.
- Splits are chronological by whole trading date. Label end timestamps are carried into the split audit, and samples that cross a split boundary are purged. Preprocessing pipelines fit their imputers and scalers on training data only.
- Model families are Logistic Regression, Random Forest, XGBoost, and LightGBM. Optuna searches a small configured trial budget (`OPTUNA_TRIALS` in `src/train_models.py`) against validation PR-AUC within each family. The family is selected by validation ROC-AUC (accuracy tie-break), and the operating threshold is selected by validation accuracy. The selected model is scored on the later test split only for diagnostics. Because this test period has been reviewed across multiple project iterations, it is no longer a pristine final holdout; use newly collected later dates for confirmatory performance claims.
- `walk_forward_benchmark.py` compares a predeclared set of regularized models using three expanding, date-based folds inside the training period. It purges training labels crossing each fold boundary, uses only point-in-time lags and full-surface context features, and writes an experimental model separately from `best_model.joblib`. Its terminal validation score is a historical diagnostic; after prior iterations, only newly collected later dates can serve as a pristine final holdout.
- Classification accuracy, selective predictions, convergence precision, trade win rate, and net profitability are different measures. The evaluation reports a fixed 0.50 reference cutoff, predeclared 0.55/0.60 cutoffs, validation-selected accuracy/F1/balanced-accuracy cutoffs, and an always-negative baseline. Every test row is an evaluation of a cutoff chosen in advance or from validation; test metrics must not be used to retune the model. See `best_model_test_operating_points.csv`.

## Execution and cost assumptions

The data has 5-minute OHLC candles, not bid/ask quotes or order-book depth. A signal is formed after its candle and enters at the next contract candle's open. Exit uses the first candle open at or after the requested holding horizon. Missing entry or exit bars are skipped and counted; no flat-price fallback is invented.

The backtest applies configurable half-spread and slippage assumptions per leg, brokerage per order, exchange premium-turnover charges, SEBI turnover fees, buyer-side stamp duty, GST on brokerage and exchange/SEBI fees, and date-dependent STT. Defaults are documented in `src/backtest.py`; update them for the relevant broker, historical date, and contract note. Exchange schedules and statutory rates can change.

Long positions are constrained by premium-funded capital, a capital fraction, maximum concurrent positions, and maximum lots. Short-option trades are skipped by default because this data set has no historical SPAN margin series. They can only be enabled by explicitly supplying a conservative `--short-margin-per-lot` proxy; that proxy is not an official margin calculation.

Risk returns use realized daily P&L divided by prior-day equity and include zero-trade dates from the market calendar. Sharpe and Sortino are annualized from daily returns using 252 sessions and the configured risk-free rate.

## Outputs

Processed files are written under `data/processed/`. Before the correction rebuild, the prior processed data, models, predictions, and reports were copied to `data/backup_pre_corrections_20261026/`. Raw files in `data/raw/` are left unchanged.

The terminal reporting script reads saved model metrics and risk outputs; it does not provide an interactive dashboard. No broker connection or live order placement is implemented.

## Known limitations

- Upstox historical candles do not provide bid/ask execution prices or historical margin requirements.
- Black–Scholes uses a constant configured rate and does not model discrete dividends, volatility jumps, or market impact.
- Backtest fills at candle opens are a transparent approximation; queue priority, partial fills, latency, and market impact are unavailable.
- Hyperparameter search is intentionally small for reproducibility and runtime. Results should be repeated across additional chronological periods before drawing strategy conclusions.
- Some legacy modules and API smoke scripts remain in the repository; not every file is part of the active pipeline.

## Continuous deep-hedging research extensions

The isolated continuous short-option research track now includes the Phase 4
cost-grid minimum viable project, Phase 7 whole-lot rounding bridge and Phase 5
robustness experiments. See [RESEARCH_PHASES_4_7_5.md](RESEARCH_PHASES_4_7_5.md)
for the experiment contracts, actual results, corrections, deliverables and
reproduction commands. Canonical directories are recorded in `research_runs.json`.

- [Minimum viable project report](data/research/phase4_cost_grid_20261031/report/MVP_RESEARCH_REPORT.md)
- [Whole-lot bridge report](data/research/research_extension_delivery_20261031/phase7/report/ROUNDING_REPORT.md)
- [Robustness report](data/research/research_extension_delivery_20261031/phase5_corrected/report_final/ROBUSTNESS_REPORT.md)

These reports use synthetic paths and explicit normalized costs. Lower hedge
tail loss is distinct from classification accuracy or trading profit. Frozen
models, as-run source snapshots, common test-path banks, all-seed comparisons,
paired conditional intervals, accounting replays and static PNG/PDF figures
remain available. These runs do not promote a production model or establish
prospective real-market performance.
