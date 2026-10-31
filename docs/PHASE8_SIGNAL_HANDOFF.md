# Phase 8: ML-selected entries and hedging (Objective B, approved)

Run: `data/research/phase8_signal_handoff_20261031/` (full tables in its `PHASE8_REPORT.md`). Retrospective research:
the test dates were inspected in earlier phases, so nothing here is confirmatory.

## Contract
Long options only. Candidate = Part 1 admission rules + classifier domain (signal_eligible, |deviation| >= 1, session horizon) +
undervalued (negative market-minus-reference deviation). Six decision slots per day, at most one option per slot, bought at the next
bar's open, held 30 minutes, 4 lots, Part 1 costs/margin/capital, four hedgers. ML rule: highest out-of-time endpoint-convergence
probability >= tau (0.5 primary; 0.6/0.7 sensitivity). Controls trade only in the slots where ML fires: seeded random candidate,
largest |deviation|, and a nearest-ATM no-signal option of the same type. Selections were hashed and frozen before any P&L existed.

## Verified mechanics
Injected-choice builder reproduces all 1,680 Part 1 episodes and 11,760 path rows exactly; probabilities come only from classifiers
fitted before the decision bar; controls never trade outside ML slots; accounting error 2.4e-9; no shorts. Coverage after the
60-session burn-in: 1,338 slots, 1,113 with a candidate (about 3 per slot); ML fires in 760 slots at tau 0.5 (91 in the test period).

## Results (tau = 0.5, 760 paired episodes)
| Question | Finding |
|---|---|
| Does ML pick entries that converge more? | Yes: 67.1% converged vs 47.5% random, 50.3% max-deviation, 44.1% across all candidates (17.6% for ATM no-signal, n=17 labelled). Same ordering in development, validation and test. |
| Does that become better P&L vs random picks? | Not detectably: paired mean difference -485 (unhedged, CI -1,199..265) and +137 (dynamic delta, CI -51..353). |
| vs largest-deviation rule, dynamic delta | ML higher: +219 per episode (CI 81..369; test +471, CI 27..923), but validation interval includes zero. Fees are lower (2,625 vs 3,003), so this is at least partly a cost/turnover effect, not proven selection skill. |
| vs nearest-ATM, hedged | ML lower by about 930 per episode (CI -1,084..-759), driven by much larger hedge costs on the options ML prefers. |
| Do hedged ML entries make money? | No. Dynamic-delta total net P&L is -2.15M over 760 episodes (-2,834 each; win rate about 10%); unhedged -450k. Fees (about 2,600 per episode) dominate, as in Part 3. |

## Conclusions allowed by the evidence
- The classifier carries real information about which mispriced options converge (precision gap is large and consistent).
- Incremental economic benefit of ML selection is **not established**: the one interval excluding zero in ML's favor (vs largest
  deviation, hedged) is plausibly cost-driven and not stable across periods; ML is significantly worse than a no-signal ATM
  rule once hedged. Profitability is **not established**; hedging costs exceed any observed edge.
- Convergence of a deviation is not option P&L: the label measures price-to-reference distance, the trade pays the option's own
  move minus spread, slippage and taxes.

## Limits and next experiments
OHLC fills, no quote depth; one-position-per-slot sequential account; the probability model, tau and rules were fixed in advance but
earlier phases already inspected these dates. Natural next steps (each needs its own predeclared protocol and later data):
unhedged or lightly-hedged ML entries with cost-aware exit rules, a fee-aware selection score, and recorded bid/ask to price the
real spread. The probability-aware neural hedger (Part 3 style) was **not** re-run on these entries.
