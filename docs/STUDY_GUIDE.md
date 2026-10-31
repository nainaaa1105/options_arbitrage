# Study guide: be able to defend every part

Goal: after one careful read and the ten-minute exercises at the end, you can explain what each part does, why it was built that
way, where it lives in the code, and what it does not show. If you cannot explain something below in your own words, say so in an
interview and describe how you would find out. That is a good answer; bluffing is not.

## The 60-second story
NIFTY index options are priced by a model. Where a quoted price sits away from a smoothed fair value, the gap might close, so I built a
classifier to predict that, tested whether acting on it makes money after hedging and costs, and set up a forward test with rules fixed
in advance. The classifier carries real signal (72.7% vs 64.5%), but fees consume everything: the honest result is no profitable edge.
Most of the work went into making sure that conclusion can be trusted.

## Ten ideas you must be able to explain
| Idea | In your own words | Where |
|---|---|---|
| Implied volatility (IV) | The volatility that makes the Black-Scholes price equal the market price; solved numerically | `implied_volatility.py` (Brent root-finder), `vector_pricing.py` (fast batch version, tested equal to the scalar one) |
| Reference price and deviation | Each option's IV is compared with an interpolation of its neighbours' IVs, leaving itself out. Deviation = market price minus the price at that reference IV | `build_reference_iv.py`, `build_deviation_features.py` |
| The label | "Did the absolute deviation fall by at least half 30 minutes later?" Endpoint, not first-passage | `convergence.py` |
| Candle timing defect | A candle is stamped with its start. A 09:15 option bar closes at 09:20, so it must pair with the spot bar that also closes at 09:20 | `candle_alignment.py` |
| Leakage control | Training labels must have ended before the validation block starts, plus a one-session gap; features are tested to be unchanged when all later data is removed | `robust_model_research.purged_folds`, `tests/test_accuracy_v2.py` |
| Selective accuracy | Accuracy only where the model is confident. Meaningless without coverage and the count of positive calls | `evaluate_fresh_period.py`, `accuracy_v2/experiment.py` |
| Delta hedging | Hold futures to offset the option's price sensitivity. Rebalancing costs money every time | `deep_hedging/baselines.py`, `engine.py`, `costs.py` |
| CVaR95 | Average loss in the worst 5% of outcomes; the risk the neural policies were trained to reduce | `continuous_hedging/objectives.py` |
| Frozen protocol | Criteria, allowed code hashes and models written to a hash-chained file before any forward data exists, so they cannot be moved later | `forward_validation/protocol.py` |
| Independent reconciliation | Rebuild cash, positions and fees from the fill ledger alone and compare with what the runner recorded | `forward_validation/reconcile.py` |

## Files worth knowing (everything else hangs off these)
| File | Why |
|---|---|
| `src/candle_alignment.py` | The bug fix; 37 lines |
| `src/convergence.py` | The label definition |
| `src/accuracy_v2/features.py` | 41 causal features: deviation dynamics, staleness, smile fits |
| `src/accuracy_v2/experiment.py` | Purged folds, model search, freeze, single evaluation |
| `src/deep_hedging/engine.py` | Self-financing hedging ledger with fees and margin |
| `src/continuous_hedging/recurrent.py` | GRU and MLP policies behind one interface; closed-loop ensembles |
| `src/signal_handoff/selection.py` | ML entry rule and the three non-ML controls |
| `src/forward_paper.py` | Paper account: quote checks, fills, risk latch (hash-frozen) |
| `src/paper_ops/daily.py` | One trading day end to end; `service.py` for the single-writer cycle |
| `src/forward_validation/acceptance.py` | The predeclared pass/fail rules |

## Questions you should expect, with the short answer
1. **Why is the accuracy only 72.7%?** Label noise: about a third of labels are decided by price moves smaller than half an index point, and the data has no bid/ask.
2. **Did you try to reach 93%?** Yes, as a stress test. Six model families and about 40 new features all plateau at the same AUC, so more modelling cannot fix it.
3. **How do you know there is no leakage?** Purged time-ordered folds, prefix-invariance tests on real data, shuffled-label sanity check, thresholds fitted only on earlier out-of-fold scores.
4. **Why does hedging lose money?** Every five-minute rebalance pays brokerage, taxes and spread; in these runs fees were larger than the loss they avoided.
5. **Why not just delta hedge less often?** A no-trade band does better on mean P&L than the neural policy; it is a stronger baseline than it looks.
6. **What did the GRU add?** Nothing. The problem is Markov in the inputs I give it, so memory has nothing to remember.
7. **What is the frozen protocol protecting against?** Moving the goalposts after seeing results.
8. **What does "reconciles" mean?** An independent replay of the fills reproduces the account's cash and positions to the rupee.
9. **Why 60 days?** Operational and regime coverage. Statistically the tail-risk claim is decided far sooner ([PROTOCOL_SENSITIVITY.md](PROTOCOL_SENSITIVITY.md)).
10. **What would change your conclusion?** Real bid/ask and depth showing a spread smaller than assumed, or a fee-aware entry rule that survives the forward test.
11. **What are the biggest weaknesses?** No pristine holdout, a model-dependent reference price (constant rate, no forward), thin sample, and untested live execution depth.
12. **What did you not build?** Real-time classifier features, so ML entries cannot run live yet.

## Ten-minute exercises (run these once)
```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe tools\run_offline_tests.py --list           # what runs offline and why
.\.venv\Scripts\python.exe -m unittest tests.test_candle_alignment -v  # the bug fix, tested
.\.venv\Scripts\python.exe tools\demo_tamper_check.py                  # evidence-chain demo (needs local data)
```
Then open `src/candle_alignment.py` and explain the join aloud; open `docs/results/accuracy_v2/fresh_selective.csv` and explain why the
98% row is not impressive.

## What to volunteer before you are asked
No profitable edge; no pristine holdout; the reference price depends on a simple pricing model; execution realism is untested until the forward test runs.
