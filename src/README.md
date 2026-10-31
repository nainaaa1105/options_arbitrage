# Source map

All modules are run from the repository root with `PYTHONPATH=src`. Modules marked **frozen** are hash-pinned by paper-trading
protocols and the artifact registries: do not edit them after a protocol is frozen (`forward_paper.py`, `research_targets.py`,
`deep_hedging/`, `forward_validation/`).

| Area | Modules |
|---|---|
| **Collection (Upstox, read-only)** | `upstox_api`, `http_transport`, `collect_all`, `collect_expiry`, `prepare_expiry`, `strike_selector`, `collect_spot_history`, `collect_hedging_futures`, `quote_capture`, `master_build`, `diagnose_upstox`, `verify_raw_api`, `repair_spot_alignment`, `candle_alignment` |
| **Pricing** | `black_scholes`, `implied_volatility`, `greeks`, `vector_pricing`, `time_to_expiry`, `pricing_research` (Black-76, robust surface fits), `parity`, `convergence` |
| **Feature and label pipeline** | `build_features`, `build_volatility_features`, `build_reference_iv`, `build_deviation_features`, `build_signal_eligibility`, `build_parity_features`, `build_convergence_labels`, `build_ml_dataset`, `prepare_ml_split`, `market_regime_features`, `research_targets` (**frozen**) |
| **Models and evaluation** | `train_models` (legacy route), `walk_forward_benchmark`, `robust_model_research`, `data_remediation_study`, `regime_feature_study`, `refit_frozen_design`, `evaluate_fresh_period`, `summarize_selective_uncertainty`, `merge_research_components`, `run_aligned_research`, `predict_research_model`, `arbitrage_classifier`, `research_registry`, `audit_rebuild`; package `accuracy_v2/` (new features, model search, bottleneck analysis) |
| **Signals and backtest (legacy route)** | `generate_signals`, `generate_trade_signals`, `backtest`, `risk_analysis`, `show_eval_metrics` |
| **Hedging research** | packages `deep_hedging/` (**frozen**, historical and simulated hedging), `continuous_hedging/` (differentiable policies, MLP/GRU, cost grid, rounding, robustness), `signal_handoff/` (ML-selected entries); entry points `run_hedging_baselines`, `run_deep_hedging`, `run_historical_hedging`, `run_continuous_hedging`, `run_phase4_research`, `run_phase5_research`, `run_phase6_research`, `run_phase7_research`, `run_signal_handoff`; `predict_*_hedge`, `evaluate_historical_hedging`, `*_extended_research`, `evaluate_extension_sensitivity`, `audit_extension_pricing`, `report_continuous_validation`, `report_phase6` |
| **Paper trading and validation** | `forward_paper` (**frozen**), `assemble_paper_events`, `monitor_paper`, `run_paper_demo`, `run_paper_day`, `build_project_dashboard`; packages `paper_ops/` (contracts, state builder, quarantine, liquidation equity, health, service, daily runner), `forward_validation/` (**frozen**, protocol, observations, reconciliation, acceptance) and `forward_validation_cli` |
| **Data checks** | `validate_dataset`, `validate_features` |

Run folders are written to `data/research/<name>/` with a source snapshot and hashes, and are never overwritten. Tests are in `../tests/`, tooling in `../tools/`.
