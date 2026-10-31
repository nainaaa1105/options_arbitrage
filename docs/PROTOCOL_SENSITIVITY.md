# Acceptance criteria: sensitivity and power

Question: if the forward test ran, how likely is each predeclared criterion to pass or fail, and are the thresholds sensible?
Method (`tools/protocol_sensitivity.py`): take the 168 historical test positions of the Part 3 experiment (28 sessions, 6 per session,
unhedged vs dynamic delta), resample whole days with replacement to build simulated forward windows of 30 to 120 days, and apply the
criteria with the same day-cluster bootstrap the real evaluator uses. 200 simulated windows per cell. Full table:
[results/protocol_sensitivity/power.csv](results/protocol_sensitivity/power.csv).

**Read these as optimistic.** Every simulated window is drawn from one 28-day period, so regime change is not represented.

## Findings

| Scenario (60 forward days) | P(CVaR95 reduction interval excludes zero) | Median cost / CVaR reduction | P(ratio ≤ 0.5) | P(≤ 0.75) | P(≤ 1.0) |
|---|---:|---:|---:|---:|---:|
| Hedging works as it did historically | 1.00 | 0.37 | 0.85 | 0.98 | 0.995 |
| Hedging has half the historical effect | 1.00 | 0.28 | 1.00 | 1.00 | 1.00 |
| Hedging costs twice as much | 0.97 | 1.33 | 0.00 | 0.02 | 0.17 |
| Hedging does nothing (noise only) | 0.00 | n/a | 0.00 | 0.00 | 0.00 |

1. **The tail-risk criterion is decided early and is nearly certain to pass.** At 30 days it already passes even when hedging has only half
   its historical strength, and never when it has none. So 60 days is not chosen for statistical power; it is chosen for operational and regime
   coverage (a few expiry cycles, different volatility days). A pass confirms what Part 3 already showed: hedging cuts tail loss.
2. **The cost-efficiency criterion is the informative one.** Historically hedging gave up 0.39 of rupee-mean per rupee of tail reduction.
   A ceiling of 1.0 would pass 99.5% of the time at that cost but also 17% of the time when costs double. **A ceiling of 0.75 passes 98%
   at historical cost and only 2% when costs double**, so it is now the draft default. A ceiling of 0.5 would wrongly fail the
   historical case 15% of the time.
3. **A false pass is very unlikely.** With no hedging effect, neither criterion passed in any of the 200 simulated windows at any length.

## Effective sample size
Episodes within a day are nearly independent in the Part 3 data (intraclass correlation 0.027, design effect 1.13, so 168 episodes behave
like about 148). The forward test is different: its four contracts and five strategies all track the same underlying on the same day,
so their outcomes are strongly correlated. The independent unit is therefore **the trading day**, and the evaluator already resamples
whole days. Read a 60-day window as 60 independent observations, not 1,440 episodes per strategy.

## Choices this supports
| Parameter | Draft value | Basis |
|---|---|---|
| Counted forward days | 60 (after 3 shakedown days) | Operational and regime coverage; power is not the constraint |
| Cost ceiling | 0.75 | Table above |
| Open-position incidents tolerated | 2 session-days | An operational allowance, not statistically derived; adjust if you prefer |

## Limits
One historical period, one hedger (dynamic delta), no regime shifts, and no correlation across contracts modelled beyond treating days as units.
Re-run `tools/protocol_sensitivity.py` after the first weeks of real data; the real cost ratio replaces the historical 0.37.
