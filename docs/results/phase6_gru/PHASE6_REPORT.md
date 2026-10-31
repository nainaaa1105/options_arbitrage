# Phase 6 report: GRU, ablations and ensembles

Run `phase6_gru_ablations_20261031`. All figures read the frozen test bank (seed 785001, 16,384 paths); checkpoint hashes verified unchanged. Synthetic zero-rate short-option world; no market profitability is implied.

## Cost 0, cvar objective (validation-selected seed per family)

| strategy | parameter_count | mean_net_pnl | pnl_std | cvar95 | cvar99 | mean_cost | mean_turnover_units | delta_mae |
|---|---|---|---|---|---|---|---|---|
| unhedged |  | -0.00028 | 0.04063 | 0.11191 | 0.16071 | 0.00000 | 0.000 | 0.4999 |
| delta |  | -0.00006 | 0.00322 | 0.00842 | 0.01345 | 0.00000 | 2.228 | 0.0000 |
| constant_band |  | -0.00005 | 0.00327 | 0.00838 | 0.01326 | 0.00000 | 2.017 | 0.0092 |
| whalley_wilmott |  | -0.00006 | 0.00322 | 0.00842 | 0.01345 | 0.00000 | 2.228 | 0.0000 |
| ww_utility_matched |  | -0.00006 | 0.00322 | 0.00842 | 0.01345 | 0.00000 | 2.228 | 0.0000 |
| mlp_delta_33 | 1345 | -0.00006 | 0.00355 | 0.00734 | 0.01113 | 0.00000 | 2.188 | 0.0215 |
| gru_delta_33 | 1217 | -0.00007 | 0.00350 | 0.00749 | 0.01159 | 0.00000 | 2.226 | 0.0196 |
| band_delta_33 | 1378 | -0.00006 | 0.00350 | 0.00763 | 0.01171 | 0.00000 | 2.092 | 0.0208 |
| mlp_single |  | -0.00006 | 0.00355 | 0.00734 | 0.01113 | 0.00000 | 2.188 | 0.0215 |
| gru_single |  | -0.00007 | 0.00350 | 0.00749 | 0.01159 | 0.00000 | 2.226 | 0.0196 |
| mlp_seed_mean |  | -0.00006 | 0.00355 | 0.00736 | 0.01118 | 0.00000 | 2.196 | 0.0217 |
| gru_seed_mean |  | -0.00007 | 0.00353 | 0.00745 | 0.01149 | 0.00000 | 2.224 | 0.0211 |
| cross_family_mean |  | -0.00006 | 0.00354 | 0.00740 | 0.01132 | 0.00000 | 2.209 | 0.0213 |

## Cost 0, entropic objective (validation-selected seed per family)

| strategy | parameter_count | mean_net_pnl | pnl_std | cvar95 | cvar99 | mean_cost | mean_turnover_units | delta_mae |
|---|---|---|---|---|---|---|---|---|
| unhedged |  | -0.00028 | 0.04063 | 0.11191 | 0.16071 | 0.00000 | 0.000 | 0.4999 |
| delta |  | -0.00006 | 0.00322 | 0.00842 | 0.01345 | 0.00000 | 2.228 | 0.0000 |
| constant_band |  | -0.00005 | 0.00327 | 0.00838 | 0.01326 | 0.00000 | 2.017 | 0.0092 |
| whalley_wilmott |  | -0.00006 | 0.00322 | 0.00842 | 0.01345 | 0.00000 | 2.228 | 0.0000 |
| ww_utility_matched |  | -0.00006 | 0.00322 | 0.00842 | 0.01345 | 0.00000 | 2.228 | 0.0000 |
| mlp_delta_11 | 1345 | -0.00007 | 0.00325 | 0.00816 | 0.01299 | 0.00000 | 2.213 | 0.0059 |
| gru_delta_11 | 1217 | -0.00007 | 0.00325 | 0.00827 | 0.01319 | 0.00000 | 2.216 | 0.0054 |
| band_delta_11 | 1378 | -0.00006 | 0.00325 | 0.00841 | 0.01336 | 0.00000 | 2.062 | 0.0071 |
| mlp_single |  | -0.00007 | 0.00325 | 0.00816 | 0.01299 | 0.00000 | 2.213 | 0.0059 |
| gru_single |  | -0.00007 | 0.00325 | 0.00827 | 0.01319 | 0.00000 | 2.216 | 0.0054 |
| mlp_seed_mean |  | -0.00007 | 0.00325 | 0.00809 | 0.01288 | 0.00000 | 2.214 | 0.0063 |
| gru_seed_mean |  | -0.00007 | 0.00325 | 0.00819 | 0.01306 | 0.00000 | 2.215 | 0.0055 |
| cross_family_mean |  | -0.00007 | 0.00325 | 0.00814 | 0.01297 | 0.00000 | 2.215 | 0.0058 |

## Cost 0.001, cvar objective (validation-selected seed per family)

| strategy | parameter_count | mean_net_pnl | pnl_std | cvar95 | cvar99 | mean_cost | mean_turnover_units | delta_mae |
|---|---|---|---|---|---|---|---|---|
| unhedged |  | -0.00028 | 0.04063 | 0.11191 | 0.16071 | 0.00000 | 0.000 | 0.4999 |
| delta |  | -0.00229 | 0.00351 | 0.01188 | 0.01718 | 0.00223 | 2.228 | 0.0000 |
| constant_band |  | -0.00185 | 0.00374 | 0.01131 | 0.01624 | 0.00180 | 1.807 | 0.0216 |
| whalley_wilmott |  | -0.00165 | 0.00413 | 0.01200 | 0.01716 | 0.00161 | 1.612 | 0.0264 |
| ww_utility_matched |  | -0.00146 | 0.00496 | 0.01345 | 0.01914 | 0.00144 | 1.444 | 0.0404 |
| mlp_delta_33 | 1345 | -0.00217 | 0.00376 | 0.00999 | 0.01366 | 0.00212 | 2.121 | 0.0296 |
| gru_delta_33 | 1217 | -0.00223 | 0.00372 | 0.01018 | 0.01410 | 0.00216 | 2.164 | 0.0287 |
| band_delta_33 | 1378 | -0.00205 | 0.00375 | 0.01019 | 0.01416 | 0.00199 | 1.992 | 0.0294 |
| mlp_single |  | -0.00217 | 0.00376 | 0.00999 | 0.01366 | 0.00212 | 2.121 | 0.0296 |
| gru_single |  | -0.00223 | 0.00372 | 0.01018 | 0.01410 | 0.00216 | 2.164 | 0.0287 |
| mlp_seed_mean |  | -0.00218 | 0.00374 | 0.01001 | 0.01377 | 0.00213 | 2.128 | 0.0292 |
| gru_seed_mean |  | -0.00223 | 0.00372 | 0.01019 | 0.01411 | 0.00217 | 2.168 | 0.0292 |
| cross_family_mean |  | -0.00221 | 0.00372 | 0.01008 | 0.01390 | 0.00214 | 2.146 | 0.0291 |

## Cost 0.001, entropic objective (validation-selected seed per family)

| strategy | parameter_count | mean_net_pnl | pnl_std | cvar95 | cvar99 | mean_cost | mean_turnover_units | delta_mae |
|---|---|---|---|---|---|---|---|---|
| unhedged |  | -0.00028 | 0.04063 | 0.11191 | 0.16071 | 0.00000 | 0.000 | 0.4999 |
| delta |  | -0.00229 | 0.00351 | 0.01188 | 0.01718 | 0.00223 | 2.228 | 0.0000 |
| constant_band |  | -0.00185 | 0.00374 | 0.01131 | 0.01624 | 0.00180 | 1.807 | 0.0216 |
| whalley_wilmott |  | -0.00165 | 0.00413 | 0.01200 | 0.01716 | 0.00161 | 1.612 | 0.0264 |
| ww_utility_matched |  | -0.00146 | 0.00496 | 0.01345 | 0.01914 | 0.00144 | 1.444 | 0.0404 |
| mlp_delta_11 | 1345 | -0.00205 | 0.00367 | 0.01075 | 0.01522 | 0.00199 | 1.995 | 0.0204 |
| gru_delta_33 | 1217 | -0.00202 | 0.00361 | 0.01087 | 0.01556 | 0.00197 | 1.972 | 0.0180 |
| band_delta_22 | 1378 | -0.00172 | 0.00396 | 0.01141 | 0.01620 | 0.00167 | 1.668 | 0.0293 |
| mlp_single |  | -0.00205 | 0.00367 | 0.01075 | 0.01522 | 0.00199 | 1.995 | 0.0204 |
| gru_single |  | -0.00202 | 0.00361 | 0.01087 | 0.01556 | 0.00197 | 1.972 | 0.0180 |
| mlp_seed_mean |  | -0.00205 | 0.00366 | 0.01075 | 0.01523 | 0.00199 | 1.994 | 0.0201 |
| gru_seed_mean |  | -0.00203 | 0.00363 | 0.01081 | 0.01542 | 0.00197 | 1.975 | 0.0193 |
| cross_family_mean |  | -0.00204 | 0.00364 | 0.01076 | 0.01531 | 0.00198 | 1.983 | 0.0196 |

## Paired contrasts (primary cost, CVaR objective)

| candidate | reference | role | cvar95_difference | cvar95_lower95 | cvar95_upper95 | mean_pnl_difference | verdict |
|---|---|---|---|---|---|---|---|
| gru_delta | mlp_delta | architecture | 0.000184 | 0.000130 | 0.000238 | -0.000052 | candidate higher CVaR95 (interval excludes 0) |
| mlp_seed_mean | mlp_single | ensemble | 0.000022 | -0.000006 | 0.000048 | -0.000011 | no detectable CVaR95 difference |
| gru_seed_mean | gru_single | ensemble | 0.000015 | -0.000010 | 0.000040 | -0.000004 | no detectable CVaR95 difference |
| cross_family_mean | mlp_single | ensemble | 0.000084 | 0.000045 | 0.000127 | -0.000032 | candidate higher CVaR95 (interval excludes 0) |
| gru_delta_h32 | gru_delta | ablation | -0.000094 | -0.000132 | -0.000057 | 0.000030 | candidate lower CVaR95 (interval excludes 0) |
| gru_delta_reset | gru_delta | ablation | 0.000094 | 0.000057 | 0.000142 | 0.000005 | candidate higher CVaR95 (interval excludes 0) |
| gru_delta_noinv | gru_delta | ablation | -0.000007 | -0.000033 | 0.000018 | 0.000011 | no detectable CVaR95 difference |
| mlp_delta_noinv | mlp_delta | ablation | 0.000086 | 0.000044 | 0.000135 | -0.000047 | candidate higher CVaR95 (interval excludes 0) |
| mlp_flat | mlp_delta | ablation | 0.000077 | 0.000039 | 0.000114 | -0.000013 | candidate higher CVaR95 (interval excludes 0) |
| gru_flat | gru_delta | ablation | -0.000078 | -0.000161 | 0.000012 | 0.000080 | no detectable CVaR95 difference |
| mlp_flat_nodelta | mlp_flat | ablation | 0.000173 | 0.000063 | 0.000286 | -0.000036 | candidate higher CVaR95 (interval excludes 0) |
| gru_flat_nodelta | gru_flat | ablation | 0.001258 | 0.001081 | 0.001407 | -0.000059 | candidate higher CVaR95 (interval excludes 0) |

## Seed diversity

| cost | objective | spec | cvar95_mean | cvar95_std | mean_abs_position_gap | position_correlation | pnl_correlation |
|---|---|---|---|---|---|---|---|
| 0.0 | cvar | mlp_delta | 0.00738 | 0.00004 | 0.00449 | 0.99997 | 0.99417 |
| 0.0 | cvar | gru_delta | 0.00747 | 0.00003 | 0.00454 | 0.99996 | 0.99459 |
| 0.0 | cvar | band_delta | 0.00770 | 0.00009 | 0.00597 | 0.99995 | 0.99164 |
| 0.0 | entropic | mlp_delta | 0.00810 | 0.00006 | 0.00400 | 0.99998 | 0.99596 |
| 0.0 | entropic | gru_delta | 0.00819 | 0.00007 | 0.00223 | 0.99999 | 0.99838 |
| 0.0 | entropic | band_delta | 0.00841 | 0.00002 | 0.00338 | 0.99999 | 0.99659 |
| 0.001 | cvar | mlp_delta | 0.01005 | 0.00007 | 0.00598 | 0.99993 | 0.99127 |
| 0.001 | cvar | gru_delta | 0.01022 | 0.00005 | 0.00588 | 0.99994 | 0.99150 |
| 0.001 | cvar | band_delta | 0.01027 | 0.00010 | 0.00527 | 0.99996 | 0.99409 |
| 0.001 | cvar | gru_delta_h32 | 0.01017 | 0.00008 | 0.00618 | 0.99993 | 0.99016 |
| 0.001 | cvar | gru_delta_reset | 0.01030 | 0.00003 | 0.00550 | 0.99996 | 0.99357 |
| 0.001 | cvar | gru_delta_noinv | 0.01026 | 0.00009 | 0.00485 | 0.99996 | 0.99408 |
| 0.001 | cvar | mlp_delta_noinv | 0.01012 | 0.00005 | 0.00459 | 0.99996 | 0.99514 |
| 0.001 | cvar | mlp_flat | 0.01015 | 0.00007 | 0.00606 | 0.99992 | 0.99127 |
| 0.001 | cvar | gru_flat | 0.01014 | 0.00004 | 0.00667 | 0.99987 | 0.98744 |
| 0.001 | cvar | mlp_flat_nodelta | 0.01024 | 0.00001 | 0.01234 | 0.99959 | 0.96446 |
| 0.001 | cvar | gru_flat_nodelta | 0.01153 | 0.00016 | 0.01206 | 0.99961 | 0.97305 |
| 0.001 | entropic | mlp_delta | 0.01078 | 0.00003 | 0.00603 | 0.99995 | 0.99216 |
| 0.001 | entropic | gru_delta | 0.01085 | 0.00001 | 0.00738 | 0.99995 | 0.98743 |
| 0.001 | entropic | band_delta | 0.01131 | 0.00010 | 0.00495 | 0.99995 | 0.99524 |

## Inference cost

| strategy | members | rollout_ms_per_1024_paths |
|---|---|---|
| mlp_single | 1 | 107.0 |
| gru_single | 1 | 87.3 |
| mlp_seed_mean | 3 | 258.8 |
| gru_seed_mean | 3 | 102.0 |
| cross_family_mean | 6 | 250.8 |

## Verification

```json
{
  "numerical_gates": [
    {
      "objective": "cvar",
      "spec": "mlp_delta",
      "delta_mae": 0.021481024101376534,
      "std_over_delta": 1.1020351741266095,
      "passed": true
    },
    {
      "objective": "cvar",
      "spec": "gru_delta",
      "delta_mae": 0.019568271934986115,
      "std_over_delta": 1.0848411410606449,
      "passed": true
    },
    {
      "objective": "entropic",
      "spec": "mlp_delta",
      "delta_mae": 0.005911371670663357,
      "std_over_delta": 1.007145102832815,
      "passed": true
    },
    {
      "objective": "entropic",
      "spec": "gru_delta",
      "delta_mae": 0.005366487428545952,
      "std_over_delta": 1.0071200085149161,
      "passed": true
    }
  ],
  "numerical_gates_pass": true,
  "replay_max_abs_error": 2.384185791015625e-07,
  "chunk_invariance_max_abs_error": 2.384185791015625e-07,
  "max_reconciliation_error": 2.6728957891464233e-07,
  "checkpoints_replayed": 20,
  "frozen_checkpoints_unchanged": true,
  "test_retuning": false,
  "validation_seed": 735001,
  "test_seed": 785001,
  "independent_of_phase4_7_5_seeds": true,
  "same_test_paths_across_cost_levels_and_objectives": true,
  "real_market_profitability_established": false
}
```

## Claim rules (fixed before fitting)

- **architecture**: GRU is called better than MLP only if the paired CVaR95 difference upper95 < 0 at the primary cost and cvar objective; worse if lower95 > 0; otherwise no detectable difference
- **ablation**: A factor matters only if variant-minus-reference paired CVaR95 lower95 > 0 (variant worse) or upper95 < 0 (variant better)
- **ensemble**: An ensemble helps only if its paired CVaR95 difference to the validation-selected single model of the same family has upper95 < 0
- **numerical_gate**: Zero-cost delta-prior MLP and GRU: delta MAE <= 0.06 and P&L std <= 1.2 x analytic delta (diagnostic, not performance)
- **intervals**: IID synthetic paths conditional on the GBM specification and fitted models; not real-market confidence and exclude training-seed uncertainty
- **scope**: Synthetic zero-rate short European option; no NIFTY margin, basis or tax claim; observed constant volatility per path
