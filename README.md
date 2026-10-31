# NIFTY Options Research Platform

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg) ![Status: research](https://img.shields.io/badge/status-research-lightgrey.svg)

Pricing, convergence prediction, hedging experiments and a frozen forward-validation protocol for NIFTY index options,
built on Upstox market data. **Research system: no profitable trading edge is claimed, and no orders are ever placed.**

## At a glance

| | |
|---|---|
| Data | 941,949 five-minute option candles, 60 weekly expiries, 283 sessions, plus spot, futures and India VIX |
| Prediction | 72.7% accuracy on later data vs 64.5% for always guessing the common outcome (AUC 0.788). A 93-94% target was investigated and ruled out: about a third of labels sit within half an index point of the cut-off, so tick-level noise caps the score ([report](docs/ACCURACY_V2_REPORT.md)) |
| Entry selection | Classifier picks converge 67.1% of the time vs 47.5% for random picks among the same candidates, but this does not become a detectable profit after costs ([Phase 8](docs/PHASE8_SIGNAL_HANDOFF.md)) |
| Hedging | Delta hedging cuts worst-case loss by more than half but fees exceed the gain; a GRU policy is no better than an MLP ([Phase 6](docs/RESEARCH_PHASE_6.md)) |
| Rigor | Fixed a spot/option timing bug (changed 8.45% of labels); purged time-ordered validation; leakage tests; 290 offline tests with the network blocked |
| Evidence integrity | Hash-chained validation protocol, independent ledger reconciliation, tamper detection ([Phase 9](docs/PHASE9_FORWARD_VALIDATION.md)) |
| Forward test | Protocol drafted, **not yet frozen**; needs about 60 trading days of real quotes. Until then every number here is retrospective |

Small result tables and figures that back these claims are in [docs/results/](docs/results). Large data, models and run folders are not in git.

## Repository layout

```text
src/            pipeline, hedging research, paper trading and validation code (module map: src/README.md)
tests/          290 offline tests (unit, accounting, leakage, fault injection, end to end on synthetic data)
tools/          offline test runner, clean-install check, scheduler scripts, demo scripts
docs/           reports, runbook and results index (docs/README.md); docs/results/ holds small evidence tables
requirements/   optional dependency sets (PyTorch CPU, reporting, CatBoost); base list is requirements.txt
data/           local only: raw candles, features, models and run folders (not committed)
```

## Quick start

```powershell
git clone https://github.com/nainaaa1105/options_arbitrage.git
cd options_arbitrage
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe tools\run_offline_tests.py          # network blocked, empty token, no data needed
```

On a fresh clone the tests that need locally generated market data skip themselves; everything else runs. Hedging research
needs the optional CPU PyTorch set (`requirements/deep-hedging.txt`, installation note inside the file). Live data needs an
Upstox token: copy `.env.example` to `.env` and fill it in. The code only reads market data.

A 25-second paper-trading replay (stored real candles, synthetic quotes) and a tamper-detection demo are described in
[docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md); daily operation is in [docs/RUNBOOK_PAPER_OPERATIONS.md](docs/RUNBOOK_PAPER_OPERATIONS.md).
Both need data and an Upstox token that are not in this repository.

## Documentation

The full index is [docs/README.md](docs/README.md). The research tracks, in the order they were built:

| Track | Read |
|---|---|
| Data defect and rebuild (spot/option candle timing) | [ML_REAUDIT](docs/ML_REAUDIT.md), [DATA_MODEL_REMEDIATION_REPORT](docs/DATA_MODEL_REMEDIATION_REPORT.md), [ACCURACY_RECOVERY](docs/ACCURACY_RECOVERY_20261030.md) |
| Hedging foundation, neural hedgers, ML-assisted hedging | [Part 1](docs/PART1_HEDGING_FOUNDATION.md), [Part 2](docs/PART2_DEEP_HEDGING.md), [Part 3](docs/PART3_HISTORICAL_ML_HEDGING.md) |
| Continuous-time policies, cost grid, rounding, robustness | [Phases 0a-3](docs/CONTINUOUS_HEDGING_PHASES_0_3.md), [Phases 4/7/5](RESEARCH_PHASES_4_7_5.md), [reports with figures](docs/results/continuous_hedging) |
| GRU, ablations, ensembles | [Phase 6](docs/RESEARCH_PHASE_6.md) |
| ML-selected entries plus hedging | [Phase 8](docs/PHASE8_SIGNAL_HANDOFF.md) |
| Accuracy programme | [ACCURACY_V2_REPORT](docs/ACCURACY_V2_REPORT.md) |
| Dashboard, paper trading, operations, forward validation | [Part 4](docs/PART4_DASHBOARD_AND_PAPER_TRADING.md), [Phase 0b](docs/OPERATIONS_PHASE_0B.md), [Phase 9](docs/PHASE9_FORWARD_VALIDATION.md) |
| Audits and status | [System audit](docs/SYSTEM_AUDIT_20261031.md), [Status and roadmap](docs/SYSTEM_STATUS_AND_ROADMAP_20261031.md), [Implementation log](docs/IMPLEMENTATION_LOG_PHASES_6_9.md) |

`research_runs.json` records the canonical corrected research run locations. Older artifacts under `data/processed` belong to the
pre-correction pipeline and must not be mixed with the corrected models and feature context; the legacy training and signal
commands require an explicit `--allow-legacy-artifacts`. Some dated reports (for example `RESEARCH_PHASES_4_7_5.md`) link to
local run folders that are not committed, because their text is pinned by artifact hashes; the committed copies are under `docs/results/`.

## Pipeline

```text
Upstox expiry candles + NIFTY spot
  -> raw expiry Parquet files -> master_options.parquet
  -> options_features (IV, Greeks)      -> options_volatility_features (historical volatility)
  -> options_reference_iv (leave-one-out surface) -> options_deviation_features
  -> options_signal_eligible -> options_parity_features -> options_convergence_labels
  -> ml_dataset + chronological splits -> validation-tuned models + evaluation
  -> candidate trades -> next-bar backtest -> daily risk report
```

Reproducing the standard pipeline (from the repository root, with `PYTHONPATH=src`). Collection can be skipped when raw expiry
files exist. `collect_all.py` supports `--run-mode`, expiry bounds, session count and `--plan-only`, and never overwrites existing
raw files. Follow the staged repair in the remediation report if you rebuild from the original misaligned data.

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

The signal generator reads the probability cutoff saved by training from validation data. Do not choose thresholds from test results.
The corrected research route (isolated, hash-recorded runs) is `run_aligned_research.py`, `merge_research_components.py`,
`evaluate_fresh_period.py` and the `accuracy_v2` package; see the reports above.

## Quantitative assumptions

- Historical volatility uses daily close-to-close returns through the **previous completed trading day**. The rolling windows require a full history before producing a value.
- Reference IV is grouped by timestamp, expiry, and option type. Each target contract is left out of its own estimate. Interior values use the nearest strike neighbors in log-moneyness; edge extrapolation is limited to 0.03 log-moneyness. Insufficient or distant estimates remain missing.
- Pricing uses European Black-Scholes with the configured constant risk-free rate. The reference surface is a statistical benchmark; a residual is not proof of executable arbitrage.
- The 30-minute label requires at least a 50% reduction in absolute reference-price deviation at the endpoint. The future candle can be up to five minutes after the nominal horizon. This is not an any-time/first-passage convergence label. The 60-minute endpoint label is retained for analysis; the current classifier target is 30 minutes.
- Splits are chronological by whole trading date. Label end timestamps are carried into the split audit, and samples that cross a split boundary are purged. Preprocessing pipelines fit their imputers and scalers on training data only.
- Model families are Logistic Regression, Random Forest, XGBoost, LightGBM (and CatBoost and ExtraTrees in the accuracy programme). Optuna searches a small configured trial budget against validation metrics within each family. The operating threshold is selected on validation data only. Because the later test period has been reviewed across several project iterations, it is no longer a pristine holdout; use newly collected later dates for confirmatory performance claims.
- `walk_forward_benchmark.py` compares a predeclared set of regularized models using three expanding, date-based folds inside the training period, purging training labels that cross each fold boundary, and writes an experimental model separately from `best_model.joblib`.
- Classification accuracy, selective predictions, convergence precision, trade win rate, and net profitability are different measures. Selective accuracy is only meaningful alongside its coverage and the number of positive calls.

## Execution and cost assumptions

The data has 5-minute OHLC candles, not bid/ask quotes or order-book depth. A signal is formed after its candle and enters at the next contract candle's open. Exit uses the first candle open at or after the requested holding horizon. Missing entry or exit bars are skipped and counted; no flat-price fallback is invented.

The backtest applies configurable half-spread and slippage assumptions per leg, brokerage per order, exchange premium-turnover charges, SEBI turnover fees, buyer-side stamp duty, GST on brokerage and exchange/SEBI fees, and date-dependent STT. Defaults are documented in `src/backtest.py`; update them for the relevant broker, historical date, and contract note. Exchange schedules and statutory rates can change.

Long positions are constrained by premium-funded capital, a capital fraction, maximum concurrent positions, and maximum lots. Short-option trades are skipped by default because this data set has no historical SPAN margin series. They can only be enabled by explicitly supplying a conservative `--short-margin-per-lot` proxy; that proxy is not an official margin calculation.

Risk returns use realized daily P&L divided by prior-day equity and include zero-trade dates from the market calendar. Sharpe and Sortino are annualized from daily returns using 252 sessions and the configured risk-free rate.

## Known limitations

- Upstox historical candles do not provide bid/ask execution prices or historical margin requirements. The paper-trading chain records real bid/ask only from the day it is switched on.
- Black-Scholes uses a constant configured rate and does not model discrete dividends, volatility jumps, or market impact.
- Backtest fills at candle opens are a transparent approximation; queue priority, partial fills, latency, and market impact are unavailable.
- Hyperparameter searches are intentionally small for reproducibility and runtime; results should be repeated across additional chronological periods before drawing strategy conclusions.
- Machine-learning entries cannot yet run in live paper trading: the classifier needs a real-time option-chain feature service that is not built.
- The forward-validation protocol is drafted but not frozen, and displayed order-book depth on the paper position size is untested until live sessions run.

## Disclaimer

This is research software provided for educational and demonstration purposes. Nothing in this repository is investment advice,
and no result here establishes that any strategy is profitable. Do not trade real money on the basis of this code.

## License

Released under the [MIT License](LICENSE).
