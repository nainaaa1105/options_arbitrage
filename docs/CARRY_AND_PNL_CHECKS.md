# Two checks on the pricing assumption and the target

Both checks were run after the main results and change no frozen artifact. Raw outputs: `docs/results/carry_diagnostic/`, `docs/results/rate_sensitivity/`, `docs/results/pnl_target/`.

## 1. Does the constant 6% carry rate create fake mispricing?

The reference price uses a fixed 6% annual carry. If the true carry differs, every option's reference price shifts and "mispricing" could be an artefact.

**Measured carry** (`tools/carry_diagnostic.py`, 239 sessions):

| Source | Median | p10 – p90 |
|---|---|---|
| Futures-implied | 6.2% | 1.4% – 9.6% |
| Put-call-parity-implied | 7.4% | -2.4% – 22.5% |

Futures are the cleaner estimate and sit on the assumed 6%. Parity is noisier because it uses option mid-prices with spread; its median is close to the assumption, and its spread reflects noise rather than a real rate change.

**Sensitivity** (`tools/rate_sensitivity.py`, 30 sessions, 100,599 rows): the whole chain (implied volatility, leave-one-out reference surface, reference price, deviation, 30-minute label) was recomputed at other rates.

| Assumed rate | Deviation correlation | Sign agreement | Label agreement |
|---|---|---|---|
| 3.0% | 0.934 | 94.5% | 93.0% |
| 4.5% | 0.977 | 97.0% | 95.3% |
| 7.5% | 0.975 | 96.9% | 94.8% |
| 9.0% | 0.919 | 94.0% | 91.4% |

**Conclusion.** The deviation signal is robust: a 3-point error in the rate changes about 5% of labels. The reason is that implied volatility and the reference surface are both computed under the same rate, so most of a rate error cancels. No rebuild of the dataset from forward prices is needed. Residual risk: a short-lived carry shock (event days) is not covered by a 30-session sample.

## 2. Does anything predict net profit, not just convergence?

`src/accuracy_v2/pnl_experiment.py` computes the real net P&L of buying every undervalued candidate at the next bar's open and selling 30 minutes later, with the Part 3 hedging cost model (half-spread and slippage 0.15 each, 0.05 tick, brokerage, exchange charges, STT, 4 lots). A classifier trained on "net P&L > 0" with the same purged chronological blocks is compared with taking all candidates and with the convergence classifier.

Mean net P&L per trade, INR, day-cluster bootstrap 95% interval:

| Population | Mean | 95% interval |
|---|---|---|
| All candidates, history blocks 3-5 | -516 | -732 to -271 |
| Profit classifier, top 20% | -346 | -979 to +316 |
| Convergence classifier, top 20% (blocks 4-5) | -616 | -1,071 to -152 |
| All candidates, blocks 4-5 | -438 | -614 to -256 |
| Later period, all candidates | -306 | -950 to +346 |
| Later period, profit classifier (frozen first) | +376 | -717 to +1,435 |

Later-period profit AUC: 0.505.

**Conclusion.**
- Buying every undervalued option and exiting after 30 minutes loses money after costs; in history the interval excludes zero.
- The profit classifier shows no skill: its selected-minus-all difference has an interval that spans zero in every period (later period: +682, -815 to +2,252), and AUC is at chance.
- The convergence classifier's picks do no better than taking everything (-182 vs all, interval -560 to +161). Predicting that a gap closes does not mean the position pays after costs.
- The later period was reviewed during earlier work, so it is retrospective, not clean forward evidence.

This is why the project's honest claim stays "a validated research platform with a pre-registered forward test", not "a profitable strategy". The forward test measures exactly this question on data nobody has seen.

## 3. How much of the loss is cost, and do bigger gaps help?

`tools/cost_decomposition.py` splits the same round trip (every undervalued candidate, bought at the next open and sold 30 minutes later) into the gross price move and the cost, by size of the initial mispricing. It is descriptive: no model, no selection. Output: `docs/results/cost_decomposition/`.

| Period | Trades | Mean gross | Gross 95% interval | Mean cost | Mean net |
|---|---|---|---|---|---|
| History | 45,624 | -39 | -200 to +138 | 431 | -470 |
| Later period | 5,184 | +142 | -508 to +804 | 448 | -306 |

- **The loss is the cost.** A round trip costs about INR 430-450 (spread, slippage, brokerage, charges, STT on 4 lots). The gross price move is statistically zero in both periods, so the net loss is almost exactly the cost.
- **Bigger gaps do not rescue it.** In history, gaps of 4-16 points show a positive gross mean (+234 and +371), but the intervals span zero, and the later period contradicts them (gaps of 8-16 points: -961). No bucket has gross above the roughly INR 450 cost reliably.
- **What this means for tuning.** Raising the gap threshold, holding longer or targeting profit instead of convergence would each need a gross edge well above INR 450 per trade, and the data shows no gross edge at all. These variants could be pre-registered as later experiments, but there is no evidence yet that they would work. Real execution costs (resting orders, narrower spreads) can only be measured from the live bid/ask chain, which starts on the first live day.
