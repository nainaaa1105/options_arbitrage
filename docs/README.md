# Documentation index

Start with the [project README](../README.md). Reports are grouped by what they answer. Dated reports record the state of the
project on that date; later reports supersede earlier ones where they disagree.

## Results and verdicts
| Document | Question it answers |
|---|---|
| [ACCURACY_V2_REPORT.md](ACCURACY_V2_REPORT.md) | Can classification accuracy reach 93-94%? What was tried, what helped, why it is capped |
| [PHASE8_SIGNAL_HANDOFF.md](PHASE8_SIGNAL_HANDOFF.md) | Does ML-selected entry add value once hedged and costed? |
| [RESEARCH_PHASE_6.md](RESEARCH_PHASE_6.md) | Does a GRU or an ensemble hedge better than an MLP? |
| [PART3_HISTORICAL_ML_HEDGING.md](PART3_HISTORICAL_ML_HEDGING.md) | Historical test of ML-assisted neural hedging |
| [PART2_DEEP_HEDGING.md](PART2_DEEP_HEDGING.md) | Neural hedgers trained on GBM and Heston simulations |
| [PART1_HEDGING_FOUNDATION.md](PART1_HEDGING_FOUNDATION.md) | Historical futures data, accounting and conventional hedges |
| [CONTINUOUS_HEDGING_PHASES_0_3.md](CONTINUOUS_HEDGING_PHASES_0_3.md) | Continuous-time policy research track, phases 0a-3 |
| [../RESEARCH_PHASES_4_7_5.md](../RESEARCH_PHASES_4_7_5.md) | Cost grid, rounding and robustness (pinned by the artifact registry, so it stays at the repo root) |

## Data quality and audits
| Document | Topic |
|---|---|
| [ML_REAUDIT.md](ML_REAUDIT.md) | The spot/option candle-close timing defect and its correction |
| [DATA_MODEL_REMEDIATION_REPORT.md](DATA_MODEL_REMEDIATION_REPORT.md) | Coverage, target design, pricing and drift remedies |
| [ACCURACY_RECOVERY_20261030.md](ACCURACY_RECOVERY_20261030.md) | API repair, data rebuild and the corrected classifier |
| [SYSTEM_AUDIT_20261031.md](SYSTEM_AUDIT_20261031.md) | Defects found in the first end-to-end audit |
| [SYSTEM_STATUS_AND_ROADMAP_20261031.md](SYSTEM_STATUS_AND_ROADMAP_20261031.md) | Consolidated status and roadmap at that date |

## Operating the paper-trading system
| Document | Topic |
|---|---|
| [PHASE9_FORWARD_VALIDATION.md](PHASE9_FORWARD_VALIDATION.md) | Frozen protocol, acceptance rules, freeze checklist |
| [OPERATIONS_PHASE_0B.md](OPERATIONS_PHASE_0B.md) | Operational hardening and what is still missing |
| [RUNBOOK_PAPER_OPERATIONS.md](RUNBOOK_PAPER_OPERATIONS.md) | Daily checklist, exit codes, alerts, recovery |
| [PART4_DASHBOARD_AND_PAPER_TRADING.md](PART4_DASHBOARD_AND_PAPER_TRADING.md) | Dashboard and the persistent paper ledger |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | A 5-minute walkthrough with Q&A |
| [IMPLEMENTATION_LOG_PHASES_6_9.md](IMPLEMENTATION_LOG_PHASES_6_9.md) | Change log for phases 6, 8, 0b, 9 and the accuracy programme |

Small result tables and figures that back these reports are in [results/](results/). Large data and models are not committed.
