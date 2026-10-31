# Phase 6: causal GRU, controlled ablations, seed diversity and ensembles

Run `data/research/phase6_gru_ablations_20261031/` (report with figures in `report/`). Synthetic zero-rate short European
option, proportional cost 0.001 unless stated. Validation bank seed 735001, independent test bank seed 785001 (16,384 paths,
generated after freeze, shared across costs/objectives). Intervals are paired path bootstraps conditional on the GBM model and the
fitted networks; they exclude training-seed uncertainty. **Differences below are candidate minus reference loss CVaR95, so a
positive number means the candidate is worse.**

## What was built
`src/continuous_hedging/recurrent.py` (causal `GRUStep`, matched `MLPStep`, closed-loop `Ensemble`, deterministic checkpoint
schema), `phase6.py` (runner), `src/report_phase6.py`, `tests/test_phase6_recurrent.py` (12 tests). Hidden state starts at zero
for every path, is updated from the current state only, never crosses rows or chunks. MLP and GRU share the same 7 inputs, the
per-seed training-path stream, optimizer, budget (12 epochs; 48 for flat-initialized variants), objectives, costs and bank.
Parameter counts: GRU(16) 1,217, MLP 1,345, GRU(32) 3,969. Phase 4/7/5 code and artifacts were not modified.

## Findings (cost 0.001, CVaR objective, validation-selected seed)
| Question | Result |
|---|---|
| Learned vs conventional | MLP loss CVaR95 0.00999 vs delta 0.01188 (-15.9%) and tuned constant band 0.01131 (-11.7%); all intervals exclude zero. But the tuned band has the better mean P&L (-0.00185 vs -0.00217). Same pattern as Phase 4. |
| GRU vs MLP | **GRU slightly worse**: +0.00018 (95% 0.00013..0.00024), about 1.8%. No sign of a memory advantage. |
| Does memory matter inside the GRU? | Yes, mildly: resetting the hidden state each step is worse (+0.00009, 0.00006..0.00014). It still does not make the GRU beat the memoryless MLP. |
| Capacity | GRU(32) beats GRU(16) (-0.00009, -0.00013..-0.00006), yet stays level with the MLP. |
| Analytic delta prior | Flat MLP worse than delta-prior MLP (+0.00008); flat GRU not distinguishable (-0.00008, -0.00016..0.00001). Flat networks learn a good policy from P&L alone but they still observe delta. |
| Delta as an input (flat init) | Removing it hurts: MLP +0.00017 (0.00006..0.00029), GRU +0.00126 (0.00108..0.00141). Neither network rediscovers delta well from moneyness, time and volatility within this budget. |
| Inventory input | Helps the MLP (+0.00009 without it); GRU shows no difference (memory substitutes for it). |
| Ensembles | Seed means: no gain (MLP +0.00002, GRU +0.00001, intervals include zero). Cross-family mean worse than the best single (+0.00008). Seed-to-seed diversity is tiny (position correlation 0.9999, P&L correlation 0.99), so there is little to average away. |
| Entropic objective | Same ordering: learned policies beat delta and Whalley-Wilmott on entropic risk; the learned-band architecture does not beat the tuned constant band. |
| Latency (CPU, 2 threads, per 1,024 paths) | MLP 107 ms, GRU 87 ms, GRU seed-mean 102 ms, cross-family 251 ms (relative indication only). |

## Conclusion
Scientifically valid comparison, no GRU superiority: a recurrent policy adds nothing here because the problem is Markov in
(moneyness, time, volatility, inventory), and the hedging risk is dominated by cost and discretization rather than
unobserved history. Ensembling is not warranted. The learned policies' CVaR advantage over delta is real in this simulation but
comes with higher mean cost than a tuned band; it is not evidence about NIFTY.

## Verification
20 selected checkpoints reload and reproduce positions to 2.4e-7; chunked inference identical; accounting reconciles to 2.7e-7;
frozen checkpoints unchanged after the test bank; zero-cost gates (delta MAE <= 0.06, P&L std <= 1.2x delta) pass for MLP and
GRU under both objectives. The Phase 4 frozen MLPs were also scored on the new bank (`phase4_frozen_mlp_on_new_bank.csv`).

## Limits
GBM with observed constant volatility per path; three seeds; one cost level for ablations; selection and test share the GBM
specification, so these results say nothing about misspecification (Phase 5 covers robustness for the MLP only).
