# Phase 8: ML-selected entries and hedging (Objective B)

Retrospective research; all figures use configured OHLC execution and margin proxies. Test-period dates were inspected in earlier phases, so nothing here is confirmatory evidence.

## Contract

- **objective**: B: ML-selected long entries then hedge
- **universe**: Part 1 admission + classifier domain (signal_eligible, |dev|>=1, session horizon) + undervalued (negative market-minus-reference deviation); long options only
- **entry**: At most one option per decision slot; buy at the next bar's open; 30-minute hold; same Part 1 execution/cost/margin engine
- **ml_rule**: Highest out-of-time endpoint-convergence probability >= tau at the decision bar; ties larger |deviation| then instrument key
- **controls**: {'random': 'seeded uniform among eligible undervalued candidates', 'max_deviation': 'largest |deviation| among the same candidates', 'atm_nearest': "Part 1 no-signal nearest-expiry/ATM option of the ML trade's option type"}
- **matching**: Controls trade only in slots where the ML rule fires; unmatched slots are skipped by every rule
- **status_of_test**: Retrospective: these dates were inspected in earlier phases; confirmatory evidence is Phase 9 only
- **claim_rule**: ML selection adds value only if the slot-paired date-cluster interval of (ML - control) excludes zero for the same hedger in a period

## Coverage

| period | slots_after_burn_in | slots_with_candidates | mean_candidates_per_slot | candidates_with_probability | candidates | ml_slots_tau50 | ml_slots_tau60 | ml_slots_tau70 |
|---|---|---|---|---|---|---|---|---|
| development | 846 | 712 | 3.41 | 2429 | 2429 | 511 | 364 | 193 |
| validation | 324 | 259 | 2.75 | 711 | 711 | 158 | 98 | 41 |
| test | 168 | 142 | 3.25 | 462 | 462 | 91 | 63 | 28 |

## Did selected entries converge? (original 30-minute endpoint label)

| selection | period | selected | labelled | converged_rate |
|---|---|---|---|---|
| all_eligible_candidates | all | 3602 | 3602 | 0.441 |
| atm_nearest_tau50 | all | 760 | 17 | 0.176 |
| atm_nearest_tau50 | development | 511 | 9 | 0.222 |
| atm_nearest_tau50 | test | 91 | 2 | 0.000 |
| atm_nearest_tau50 | validation | 158 | 6 | 0.167 |
| max_deviation_tau50 | all | 760 | 760 | 0.503 |
| max_deviation_tau50 | development | 511 | 511 | 0.538 |
| max_deviation_tau50 | test | 91 | 91 | 0.484 |
| max_deviation_tau50 | validation | 158 | 158 | 0.399 |
| ml_tau50 | all | 760 | 760 | 0.671 |
| ml_tau50 | development | 511 | 511 | 0.658 |
| ml_tau50 | test | 91 | 91 | 0.703 |
| ml_tau50 | validation | 158 | 158 | 0.696 |
| random_tau50 | all | 760 | 760 | 0.475 |
| random_tau50 | development | 511 | 511 | 0.479 |
| random_tau50 | test | 91 | 91 | 0.429 |
| random_tau50 | validation | 158 | 158 | 0.487 |

## Episode economics at tau=0.5

| selection | hedger | period | episodes | net_pnl_sum | net_pnl_mean | win_rate | fees_mean | loss_cvar95 |
|---|---|---|---|---|---|---|---|---|
| atm_nearest_tau50 | band_delta | all | 760 | -1,456,413.02 | -1,916.33 | 0.089 | 1,616.94 | 4,762.68 |
| max_deviation_tau50 | band_delta | all | 760 | -2,252,649.15 | -2,964.01 | 0.076 | 2,885.02 | 7,619.33 |
| ml_tau50 | band_delta | all | 760 | -2,142,157.88 | -2,818.63 | 0.071 | 2,550.85 | 7,070.01 |
| random_tau50 | band_delta | all | 760 | -2,180,013.92 | -2,868.44 | 0.070 | 2,675.67 | 7,449.40 |
| atm_nearest_tau50 | band_delta | development | 511 | -941,413.68 | -1,842.30 | 0.104 | 1,471.02 | 4,921.69 |
| max_deviation_tau50 | band_delta | development | 511 | -1,404,997.80 | -2,749.51 | 0.086 | 2,591.45 | 7,532.25 |
| ml_tau50 | band_delta | development | 511 | -1,356,277.58 | -2,654.16 | 0.072 | 2,321.86 | 7,238.01 |
| random_tau50 | band_delta | development | 511 | -1,365,250.95 | -2,671.72 | 0.072 | 2,396.96 | 7,416.38 |
| atm_nearest_tau50 | band_delta | test | 91 | -169,044.77 | -1,857.63 | 0.077 | 1,855.35 | 4,447.25 |
| max_deviation_tau50 | band_delta | test | 91 | -286,672.42 | -3,150.25 | 0.110 | 3,385.75 | 7,804.26 |
| ml_tau50 | band_delta | test | 91 | -259,473.26 | -2,851.35 | 0.121 | 2,927.57 | 6,893.61 |
| random_tau50 | band_delta | test | 91 | -274,305.96 | -3,014.35 | 0.121 | 3,234.85 | 7,508.77 |
| atm_nearest_tau50 | band_delta | validation | 158 | -345,954.57 | -2,189.59 | 0.051 | 1,951.55 | 4,414.62 |
| max_deviation_tau50 | band_delta | validation | 158 | -560,978.93 | -3,550.50 | 0.025 | 3,546.10 | 7,716.19 |
| ml_tau50 | band_delta | validation | 158 | -526,407.04 | -3,331.69 | 0.038 | 3,074.49 | 6,542.65 |
| random_tau50 | band_delta | validation | 158 | -540,457.01 | -3,420.61 | 0.032 | 3,255.03 | 7,322.40 |
| atm_nearest_tau50 | dynamic_delta | all | 760 | -1,446,736.26 | -1,903.60 | 0.087 | 1,643.56 | 4,771.75 |
| max_deviation_tau50 | dynamic_delta | all | 760 | -2,320,014.03 | -3,052.65 | 0.071 | 3,003.40 | 7,813.38 |
| ml_tau50 | dynamic_delta | all | 760 | -2,153,581.69 | -2,833.66 | 0.068 | 2,624.57 | 7,006.73 |
| random_tau50 | dynamic_delta | all | 760 | -2,257,519.42 | -2,970.42 | 0.064 | 2,778.88 | 7,688.87 |
| atm_nearest_tau50 | dynamic_delta | development | 511 | -934,935.20 | -1,829.62 | 0.102 | 1,502.08 | 4,925.28 |
| max_deviation_tau50 | dynamic_delta | development | 511 | -1,434,812.82 | -2,807.85 | 0.080 | 2,704.39 | 7,598.34 |
| ml_tau50 | dynamic_delta | development | 511 | -1,360,327.31 | -2,662.09 | 0.070 | 2,406.33 | 7,036.07 |
| random_tau50 | dynamic_delta | development | 511 | -1,394,239.16 | -2,728.45 | 0.068 | 2,501.54 | 7,366.67 |
| atm_nearest_tau50 | dynamic_delta | test | 91 | -165,846.49 | -1,822.49 | 0.066 | 1,903.27 | 4,520.91 |
| max_deviation_tau50 | dynamic_delta | test | 91 | -301,243.98 | -3,310.37 | 0.099 | 3,522.30 | 8,452.88 |
| ml_tau50 | dynamic_delta | test | 91 | -258,345.28 | -2,838.96 | 0.110 | 3,006.03 | 7,355.70 |
| random_tau50 | dynamic_delta | test | 91 | -287,606.07 | -3,160.51 | 0.099 | 3,333.58 | 8,773.86 |
| atm_nearest_tau50 | dynamic_delta | validation | 158 | -345,954.57 | -2,189.59 | 0.051 | 1,951.55 | 4,414.62 |
| max_deviation_tau50 | dynamic_delta | validation | 158 | -583,957.22 | -3,695.93 | 0.025 | 3,671.58 | 8,024.55 |
| ml_tau50 | dynamic_delta | validation | 158 | -534,909.10 | -3,385.50 | 0.038 | 3,110.69 | 6,657.96 |
| random_tau50 | dynamic_delta | validation | 158 | -575,674.19 | -3,643.51 | 0.032 | 3,356.40 | 7,604.06 |
| atm_nearest_tau50 | unhedged | all | 760 | -341,169.93 | -448.91 | 0.396 | 126.79 | 12,431.78 |
| max_deviation_tau50 | unhedged | all | 760 | -349,275.68 | -459.57 | 0.463 | 296.72 | 23,652.11 |
| ml_tau50 | unhedged | all | 760 | -450,408.37 | -592.64 | 0.418 | 265.72 | 20,774.02 |
| random_tau50 | unhedged | all | 760 | -82,066.63 | -107.98 | 0.470 | 277.39 | 21,132.83 |
| atm_nearest_tau50 | unhedged | development | 511 | -291,578.81 | -570.60 | 0.397 | 127.68 | 13,211.96 |
| max_deviation_tau50 | unhedged | development | 511 | -165,508.46 | -323.89 | 0.468 | 305.15 | 23,812.07 |
| ml_tau50 | unhedged | development | 511 | -368,007.94 | -720.17 | 0.427 | 265.93 | 22,129.52 |
| random_tau50 | unhedged | development | 511 | -1,597.94 | -3.13 | 0.497 | 276.02 | 21,516.26 |
| atm_nearest_tau50 | unhedged | test | 91 | -26,987.21 | -296.56 | 0.451 | 124.85 | 14,050.11 |
| max_deviation_tau50 | unhedged | test | 91 | -44,384.56 | -487.74 | 0.516 | 322.46 | 29,165.83 |
| ml_tau50 | unhedged | test | 91 | -51,432.44 | -565.19 | 0.429 | 279.91 | 23,106.06 |
| random_tau50 | unhedged | test | 91 | 50,379.92 | 553.63 | 0.484 | 310.80 | 26,066.57 |
| atm_nearest_tau50 | unhedged | validation | 158 | -22,603.91 | -143.06 | 0.361 | 125.04 | 7,242.22 |
| max_deviation_tau50 | unhedged | validation | 158 | -139,382.66 | -882.17 | 0.418 | 254.63 | 16,827.74 |
| ml_tau50 | unhedged | validation | 158 | -30,968.00 | -196.00 | 0.386 | 256.89 | 11,872.25 |
| random_tau50 | unhedged | validation | 158 | -130,848.61 | -828.16 | 0.373 | 262.57 | 14,995.62 |

## Slot-paired ML minus control at tau=0.5

| control | hedger | period | paired_slots | mean_diff_ml_minus_control | lower95 | upper95 | ml_better_share | verdict |
|---|---|---|---|---|---|---|---|---|
| random | unhedged | all | 760 | -484.66 | -1,198.65 | 265.01 | 0.337 | no detectable difference |
| random | unhedged | development | 511 | -717.05 | -1,705.90 | 251.92 | 0.339 | no detectable difference |
| random | unhedged | validation | 158 | 632.16 | -448.81 | 1,630.99 | 0.354 | no detectable difference |
| random | unhedged | test | 91 | -1,118.82 | -3,103.48 | 945.31 | 0.297 | no detectable difference |
| random | dynamic_delta | all | 760 | 136.76 | -50.49 | 352.58 | 0.367 | no detectable difference |
| random | dynamic_delta | development | 511 | 66.36 | -167.21 | 380.07 | 0.360 | no detectable difference |
| random | dynamic_delta | validation | 158 | 258.01 | -22.21 | 556.59 | 0.367 | no detectable difference |
| random | dynamic_delta | test | 91 | 321.55 | -23.25 | 684.19 | 0.407 | no detectable difference |
| max_deviation | unhedged | all | 760 | -133.07 | -829.19 | 575.27 | 0.272 | no detectable difference |
| max_deviation | unhedged | development | 511 | -396.28 | -1,222.18 | 425.62 | 0.245 | no detectable difference |
| max_deviation | unhedged | validation | 158 | 686.17 | -830.66 | 2,389.27 | 0.361 | no detectable difference |
| max_deviation | unhedged | test | 91 | -77.45 | -3,051.91 | 2,952.24 | 0.275 | no detectable difference |
| max_deviation | dynamic_delta | all | 760 | 218.99 | 81.12 | 369.09 | 0.333 | ML higher |
| max_deviation | dynamic_delta | development | 511 | 145.76 | -5.21 | 312.40 | 0.311 | no detectable difference |
| max_deviation | dynamic_delta | validation | 158 | 310.43 | -24.08 | 627.54 | 0.373 | no detectable difference |
| max_deviation | dynamic_delta | test | 91 | 471.41 | 27.11 | 922.64 | 0.385 | ML higher |
| atm_nearest | unhedged | all | 760 | -143.73 | -390.21 | 110.63 | 0.483 | no detectable difference |
| atm_nearest | unhedged | development | 511 | -149.57 | -484.63 | 177.42 | 0.497 | no detectable difference |
| atm_nearest | unhedged | validation | 158 | -52.94 | -457.19 | 341.63 | 0.449 | no detectable difference |
| atm_nearest | unhedged | test | 91 | -268.63 | -1,161.18 | 529.84 | 0.462 | no detectable difference |
| atm_nearest | dynamic_delta | all | 760 | -930.06 | -1,083.94 | -758.72 | 0.203 | ML lower |
| atm_nearest | dynamic_delta | development | 511 | -832.47 | -1,035.70 | -581.20 | 0.235 | ML lower |
| atm_nearest | dynamic_delta | validation | 158 | -1,195.91 | -1,366.46 | -1,041.32 | 0.089 | ML lower |
| atm_nearest | dynamic_delta | test | 91 | -1,016.47 | -1,366.10 | -687.28 | 0.220 | ML lower |

## Verification

```json
{
  "part1_equivalence": {
    "part1_episodes": 1680,
    "part1_path_rows": 11760,
    "rebuild_skipped": 0
  },
  "builder_reproduces_part1_exactly": true,
  "max_reconciliation_error": 2.3856046027503908e-09,
  "probabilities_from_past_fit_only": true,
  "controls_within_ml_slots": true,
  "selections_frozen_before_outcomes": {
    "frozen_at": "2026-10-31T12:54:54.352797+00:00",
    "selections_sha256": "ea674ec6147b3b7a2bfe4dd84fe6bcd2530da315602afc4fea8a30bb82f404e1",
    "rows": 6188,
    "outcomes_computed_before_freeze": false
  },
  "shorts_used": false,
  "primary_tau_all_period_verdicts": {
    "random/unhedged": "no detectable difference",
    "random/static_delta": "no detectable difference",
    "random/dynamic_delta": "no detectable difference",
    "random/band_delta": "no detectable difference",
    "max_deviation/unhedged": "no detectable difference",
    "max_deviation/static_delta": "no detectable difference",
    "max_deviation/dynamic_delta": "ML higher",
    "max_deviation/band_delta": "ML higher",
    "atm_nearest/unhedged": "no detectable difference",
    "atm_nearest/static_delta": "ML lower",
    "atm_nearest/dynamic_delta": "ML lower",
    "atm_nearest/band_delta": "ML lower"
  },
  "profitability_established": false
}
```
