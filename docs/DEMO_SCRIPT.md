# Demo script and Q&A (5 minutes, with a 2-minute cut)

Purpose: show an engineer's judgement, not a trading claim. Lead with what is true and verified; volunteer the limits yourself.

## Before you start (2 minutes, once)
```powershell
cd <repo>; $env:PYTHONPATH="src"
# Warm up the replay so the first run is not cold (25 seconds). Uses stored real candles for a past Friday with SYNTHETIC quotes.
.\.venv\Scripts\python.exe src\run_paper_day.py replay --date 2026-10-30 --only ATM_CE --strategies unhedged dynamic_delta --base data\paper\demo_$(Get-Date -Format HHmmss)
```
Have these open in tabs: the project page, `ACCURACY_V2_REPORT.md`, `PHASE8_SIGNAL_HANDOFF.md`, `docs/results/`.

## The 5-minute version
| Time | Show | Say |
|---|---|---|
| 0:00 | Project page, top | "NIFTY options research platform: it prices options, predicts whether a mispricing will close, tests hedging with and without ML, and runs a forward test whose rules are frozen in advance. It is research; I do not claim a profitable edge." |
| 0:45 | Pipeline row, then the accuracy plot | "Best model is 72.7% against 64.5% for guessing the common outcome. I tried to reach 93% as a stress test and measured why it is impossible: a third of labels sit within half an index point of the cut-off, so tick noise caps the score." |
| 1:30 | Selection plot and hedging table | "The classifier does pick better entries: 67% close versus 47% for random. But after fees nothing is profitable, and hedging cuts the worst case by more than half while costing more than it saves. I report that plainly." |
| 2:15 | Terminal: the replay command from above (25 s) | "This runs a full paper-trading day from stored real candles with synthetic quotes: open, hedge every five minutes, close, with fees. Both accounts reconcile. Unhedged lost about 1.5k, dynamic delta lost about 21k, of which 15k was fees." |
| 3:15 | Terminal: `python tools\demo_tamper_check.py` (30 s) | "Evidence integrity: the forward-test rules are hash-chained. Lowering the bar from 60 to 3 days is rejected on load. Editing one simulated fill by half a rupee is caught by an independent ledger replay as a 130-rupee error." |
| 4:00 | Page: engineering and limits sections | "I found my own timing bug: option and spot candles were paired by start time rather than close time. Fixing it changed 8.45% of labels and I rebuilt everything. 309 tests run with the network blocked." |
| 4:40 | Page: forward test | "Next step is a 60-day forward test on live quotes with predeclared pass or fail criteria. It starts with three shakedown days. Today it is not frozen, so every figure here is retrospective." |

## The 2-minute cut
Project page top and accuracy plot (45 s), the replay run (25 s plus talking), tamper demo (30 s), one-sentence close: "Real signal, no edge after costs, and a forward test designed so I cannot fool myself."

## Likely questions and straight answers
- **Is it profitable?** No. The classifier selects better entries, but hedged results lose to fees in every experiment. That result is the finding.
- **Why 72.7% and not 93%?** The target compares a last-traded price to a smoothed surface 30 minutes later. About 15% of labels are within 0.25 index points of the cut-off and 32% within 0.5, which is the size of the bid-ask spread. Different models and about 40 extra features all plateau at the same AUC, so it is label noise, not model capacity. Only bid/ask depth data or a different target could change that.
- **How do you know there is no leakage?** Training labels must end before each validation block starts with a one-session embargo; feature code is tested to give identical values when all later data is removed; shuffled labels give an AUC near 0.5; thresholds are fitted only on earlier out-of-fold scores.
- **You already looked at the test period. Is it a holdout?** No, and I say so. Results on those dates are retrospective. The forward test is the confirmatory step, with the rules frozen first.
- **Does the forward test exist yet?** The software, protocol draft and a full-session dry run exist and are verified. It has not started, because I want the shakedown findings (especially order-book depth) before freezing.
- **What would you do with more time or data?** Record real bid/ask and depth for a few weeks, build the live option-chain feature service so ML entries can run in paper trading, and test a fee-aware entry rule.
- **What did you get wrong along the way?** Candle timing (fixed, documented), term-structure features that turned out impossible with one expiry per timestamp (dropped), a stale-spot bug in the live feature builder and a run-length feature bug (both caught by tests and fixed).
- **Why a public repo?** The code and the negative results are the portfolio piece. No credentials or market data are in it.

## Do not say
"Beats the market", "93%", "profitable", "arbitrage opportunity found". None are supported.
