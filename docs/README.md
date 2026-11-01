# Documentation index

Start with the [project README](../README.md), then these two:

| Document | Read it to |
|---|---|
| [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) | See every track's headline numbers on one page |
| [STUDY_GUIDE.md](STUDY_GUIDE.md) | Understand the system well enough to explain and defend it |

## Current reports
| Document | Question it answers |
|---|---|
| [ACCURACY_V2_REPORT.md](ACCURACY_V2_REPORT.md) | Can classification accuracy reach 93-94%? What helped, and why it is capped |
| [PHASE8_SIGNAL_HANDOFF.md](PHASE8_SIGNAL_HANDOFF.md) | Does ML-selected entry add value once hedged and costed? |
| [RESEARCH_PHASE_6.md](RESEARCH_PHASE_6.md) | Does a GRU or an ensemble hedge better than an MLP? |
| [../RESEARCH_PHASES_4_7_5.md](../RESEARCH_PHASES_4_7_5.md) | Cost grid, whole-lot rounding and robustness (pinned by the artifact registry, so it stays at the repo root) |

## Forward validation and operation
| Document | Topic |
|---|---|
| [PHASE9_FORWARD_VALIDATION.md](PHASE9_FORWARD_VALIDATION.md) | Frozen protocol, acceptance rules, freeze checklist |
| [PROTOCOL_SENSITIVITY.md](PROTOCOL_SENSITIVITY.md) | How likely the criteria are to pass or fail, and why the thresholds are set where they are |
| [CARRY_AND_PNL_CHECKS.md](CARRY_AND_PNL_CHECKS.md) | Whether the fixed 6% carry creates fake mispricing, and whether any classifier predicts net profit |
| [OLDER_REGIME_HOLDOUT.md](OLDER_REGIME_HOLDOUT.md) | The frozen classifier applied once to 227 earlier sessions it never saw |
| [PHASE9_PREFLIGHT.md](PHASE9_PREFLIGHT.md) | Approval sheet, freeze checklist and Monday checklist for the forward window |
| [OPERATIONS_PHASE_0B.md](OPERATIONS_PHASE_0B.md) | Operational hardening and what is still missing |
| [RUNBOOK_PAPER_OPERATIONS.md](RUNBOOK_PAPER_OPERATIONS.md) | Daily checklist, exit codes, alerts, recovery |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | A 5-minute walkthrough with Q&A |

## Evidence and history
- [results/](results/): small tables and figures behind the reports ([accuracy_v2](results/accuracy_v2), [phase6_gru](results/phase6_gru),
  [phase8_signal_handoff](results/phase8_signal_handoff), [continuous_hedging](results/continuous_hedging),
  [protocol_sensitivity](results/protocol_sensitivity), [phase9_infrastructure](results/phase9_infrastructure), [paper_dry_run](results/paper_dry_run)).
- [archive/](archive/README.md): eleven dated stage reports (data defect, hedging parts 1-4, audits, change log), kept as a record.

Large data, models and run folders are not committed.
