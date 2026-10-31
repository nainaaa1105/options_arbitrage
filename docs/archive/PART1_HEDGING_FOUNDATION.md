# Part 1 — historical hedge data, accounting and delta baselines

Implemented 31 October 2026, Asia/Kolkata. Existing ML models and processed ML
datasets are preserved. This part does not implement PyTorch, GBM/Heston training,
CVaR optimization or an ML-versus-delta performance claim; those are Parts 2–3.

## What this part does

The engine holds a fixed option position and compares four ways of managing its
underlying exposure. It measures what hedging changes before a neural policy is
introduced. A hedge can reduce the size of bad outcomes while costing more than
it earns. Both effects must remain visible.

At a decision candle's close, the engine observes the existing option delta,
spot and front-month futures close. It enters or rebalances at the next bar's
open. For example, a candle labelled 09:30 finishes at 09:35; its state can be
used at the 09:35 open, not the 09:30 open. There is no additional latency in
this OHLC proxy, so these are research fills rather than demonstrated fills.

The scheduled holding period is 30 minutes, with five-minute decisions. The
60-minute alternative is supported. Both legs close at the scheduled terminal
open, unless the common risk rules require an earlier margin/cash exit.

## Deliverables

| Deliverable | Implementation / evidence |
|---|---|
| Historical futures collection | `src/collect_hedging_futures.py`, `src/deep_hedging/collect.py` |
| Validated contracts and price files | `data/raw/upstox/hedging_futures_20261030/` |
| Synchronized fixed episodes | `src/deep_hedging/data.py`; `episode_universe.parquet`, `episode_paths.parquet` |
| Self-financing accounting | `src/deep_hedging/accounting.py`, `engine.py`, `costs.py` |
| Four conventional baselines | `src/deep_hedging/baselines.py` |
| Marked equity and comparison reports | `src/deep_hedging/metrics.py`, `run.py`; final result directory below |
| Financial regression checks | `tests/test_hedging_part1.py` |
| Reproduction and assumptions | This document, run manifest, source snapshot and collection manifests |

The final result directory is `data/research/hedging_part1_complete_20261031/`.
Its principal files are:

- `RESULTS.md`: readable comparison.
- `baseline_comparison.csv`: complete performance/cost/risk metrics.
- `episode_results.parquet`: reconciled result for each position and strategy.
- `transaction_ledger.parquet`: signed quantities, fills, fees and realized futures P&L.
- `marked_equity.parquet`: cash, option value, futures unrealized P&L, margin and net delta at every mark.
- `daily_equity.csv`: daily equity/returns, including zero-trade dates.
- `paired_comparison.csv`: same-episode differences with trading-session bootstrap intervals.
- `data_rejections.csv`, `capital_rejections.csv`: observable exclusions.
- `manifest.json`, `source_snapshot/`: assumptions, input hashes and code version.

## Historical data and observed coverage

Collected 14 monthly NIFTY futures contracts, covering 3 September 2025 through
27 October 2026: 21,355 bars across 284 observed dates. The front-month rule
switches on the calendar date after the previous monthly expiry. Each candidate
monthly expiry is verified against Upstox's futures metadata; an unverified
instrument or nonpositive lot/tick size causes failure.

Upstox metadata tick sizes are converted from paise to rupees. Every collected
futures close matches the converted contract tick grid. Actual returned lot
sizes are used; they are not hardcoded as one quantity across the sample.
Static metadata per expired contract remains an assumption if an intracontract
unit revision is absent from the provider's history.

Requests are broken at calendar-month boundaries. An empty request window is
accepted only when the independent spot calendar also has no session there.
An empty response for a known spot session remains an error. Interrupted
collection can resume after verifying existing contract files. Complete
collections and existing outputs are not overwritten.

The options input is the synchronized 941,949-row research feature dataset.
It has 283 observed dates. There is a futures session on 28 January 2026 without
a corresponding option session in this dataset. It is retained as a zero-trade
date, not fabricated into an option path.

On 6 and 7 October 2025, all sampled options have zero **reported** OI, despite
many having positive reported volume. This fails the declared OI gate; it does
not prove that the market had zero actual open interest. The shortened/special
21 October session also does not supply the fixed regular-session episodes.
The rejection file makes each scheduled omission observable.

## Episode selection and fair comparison

The default sampling uses decision candle starts at 09:30, 10:30, 11:30, 12:30,
13:30 and 14:30. CE and PE alternate by slot. The nearest available expiry and
then the nearest ATM strike are selected using only observations at that time.
Price, reported volume/OI, initial delta, expiry and moneyness gates are explicit.
There is no convergence-probability or future-profit selection.

The reference position is four option lots, long by default. Reference initial
capital is INR 5,000,000 (50 lakh); this is a configurable research account, not
a live capital allocation. Futures exposure is bounded to four whole lots.
The default future margin proxy is 25% of notional with a 5% equity cash buffer.
Short options require an explicit separate margin-per-lot proxy.

All four strategies receive identical option entries and scheduled exits. The
shared account is sequential: positions do not overlap. Every strategy must
pass the same conservative initial capital reserve; rejection by this common
gate removes the episode from all four. Later margin-driven forced exits remain
in the results and are not removed as inconvenient losses.

This yields 1,680 common simulated episodes. The episode/path identity is
verified when reusing an execution ledger, along with identical accounting,
cost and policy code. This allows a calendar-only reporting correction without
rerunning unchanged executions. The final manifest records that reuse.

## Baseline definitions

| Baseline | Behaviour |
|---|---|
| Unhedged | Holds the option; no futures orders |
| Static delta | Establishes a rounded delta hedge at entry and holds it until exit |
| Dynamic delta | Recomputes the target at each five-minute boundary |
| Band delta | Starts hedged, then trades only outside the declared no-trade band |

The option delta is the existing Black–Scholes spot delta. Future sensitivity
to spot is approximated by observed F/S under deterministic carry. Hence desired
future units are `-option_units * option_delta / (F/S)`, then converted to the
actual futures lot size. This does not claim a learned stochastic-basis hedge.

Targets are rounded to the nearest whole lot, with exact half-lot ties toward
zero. Invalid later deltas cause an inventory hold and a recorded event, not
retrospective deletion of the entire episode.

The band is 0.25 of the option's units: one lot-equivalent for a four-lot position
when contract multipliers match. An initial development run used 0.10, which
fell below the integer-lot rounding dead zone and reproduced dynamic hedging.
The default was changed for that structural reason, not optimized over P&L.
Both development runs remain available. These historical baseline results are
development evidence, not a newly untouched test of a tuned trading strategy.

## Cash and risk accounting

Option purchases/sales change cash by signed quantity times fill price. Futures
positions do not purchase the contract notional. Futures realized gains/losses
affect cash when exposure closes; unrealized P&L contributes to marked equity.
Positive unrealized futures gains are not assumed available to finance margin
intraday. Margin reserves constrain free cash but are not charged as expenses.

Every rebalance and terminal close incurs its asset-specific charges. Costs
include brokerage, exchange fees, SEBI/IPFT, sell-side STT, buy-side stamp duty,
GST and adverse spread/slippage with tick rounding. Futures use notional turnover;
options use premium turnover. STT uses the applicable pre-/post-1-April-2026
schedule. Other charge rates remain explicit configured assumptions across
the historical period, not a reconstruction of every historical contract note.

Sources checked: [NSE STT schedule](https://www.nseindia.in/static/products-services/equity-derivatives-securities-transaction-tax),
[Upstox brokerage/charges](https://upstox.com/brokerage-charges/) and
[expired futures metadata](https://upstox.com/developer/api-documentation/get-expired-future-contracts/).

Spread/slippage are represented in fills and execution impact; they are not
subtracted twice. Optional funding is charged once over elapsed holding time;
the default funding and risk-free cash-return assumptions are zero.

For each closed episode the engine verifies:

`cash change = option gross P&L + futures gross P&L - fees - execution impact - funding`.

Both inventories must finish at zero. If a terminal long option has no positive
bid under the adverse fill proxy, it is conservatively written off at zero,
flagged as a non-executed writeoff and retained as a loss. This occurred in seven
episodes per baseline. It is not a claim of a fill at zero or proof that the
remaining OHLC fills were executable.

Daily risk uses marked equity on the union of observed option/futures dates,
including zero-trade dates. Drawdown includes initial capital as the first peak.
Reports include episode loss VaR/CVaR, standard deviation, daily Sharpe/Sortino,
daily/intraday drawdown, costs, turnover, margin, forced exits and reconciliation.

## What the benchmark establishes

Read `RESULTS.md` and `baseline_comparison.csv` for the measured results. The
first experiments show a substantial reduction in tail loss from futures
hedging, alongside substantially worse aggregate net P&L because of turnover
and futures costs. This is a risk/cost tradeoff, not evidence of arbitrage profit.

These are fixed long-option sampling experiments. They do not measure the
existing convergence classifier's profitability and are not ML hedging results.
The Part 2 policy will consume causal state and the same accounting machinery.
Part 3 must compare that policy on identical positions and execution rules,
including an ablation with and without walk-forward ML probabilities.

## Reproduce

From the repository root with the project environment:

```powershell
.\.venv\Scripts\python.exe src/collect_hedging_futures.py --from-date 2025-09-03 --to-date 2026-10-27 --spot-calendar data/raw/upstox/nifty_5minute_to_20261029.parquet --output data/raw/upstox/my_futures_collection

.\.venv\Scripts\python.exe src/run_hedging_baselines.py --futures data/raw/upstox/hedging_futures_20261030/front_month_futures.parquet --output data/research/my_hedging_part1_run

.\.venv\Scripts\python.exe -m unittest tests.test_hedging_part1 -v
```

Add `--horizon 60` for the longer horizon. Pass `--config settings.json` for
explicit `HedgeConfig` overrides, such as capital, margin and costs. Use a new
output directory for each run. Low-capital accounts can legitimately fail the
common whole-lot collateral gate; the engine does not invent fractional futures
or silently remove the capital requirement.

## Remaining limitations for later parts

- No historical bid/ask/depth fills, queue priority, partial fills or extra latency.
- No official historical SPAN/combined portfolio-margin reconstruction.
- No concurrent multi-option netting, overnight settlement or expiry exercise.
- Fixed-cost schedules/proxies require broker/date-specific reconciliation.
- Source gaps, reported-zero OI and conservative writeoffs remain observable.
- No PyTorch policy, simulation training, ML-assisted hedge or live orders yet.
