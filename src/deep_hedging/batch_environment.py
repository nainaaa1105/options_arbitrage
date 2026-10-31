"""Vectorized, discrete accounting matched against the Part 1 scalar ledger.

No gradients pass through fills/fees/rounding. Categorical policy gradients are
used instead. All proposed actions are whole futures lots during training.
"""

import numpy as np

from .costs import transaction_fees


class BatchEnvironment:
    def __init__(self, batch, config, timestamp="2026-10-01 09:35"):
        self.batch, self.c = batch, config
        self.option = np.asarray(batch["option"], float)
        self.future = np.asarray(batch["future"], float)
        self.delta = np.asarray(batch["delta"], float)
        self.n, columns = self.option.shape
        self.steps = columns-1
        if self.steps != config.horizon_minutes//config.bar_minutes or self.future.shape != self.option.shape:
            raise ValueError("Batch horizon/shape differs from execution configuration")
        if not np.isfinite(self.option).all() or not np.isfinite(self.future).all() or (self.option < 0).any() or (self.future <= 0).any():
            raise ValueError("Invalid simulated prices")
        self.lot = np.broadcast_to(batch.get("future_lot_size", 65), (self.n,)).astype(int)
        option_lot = np.broadcast_to(batch.get("option_lot_size", self.lot), (self.n,)).astype(int)
        if (self.lot <= 0).any() or (option_lot <= 0).any():
            raise ValueError("Invalid lot sizes")
        self.tick = np.broadcast_to(batch.get("future_tick_size", .10), (self.n,)).astype(float)
        self.original_q = config.option_side*config.option_lots*option_lot
        self.q = np.zeros(self.n, dtype=int)
        self.h = np.zeros(self.n, dtype=int)
        self.average = np.zeros(self.n)
        self.cash = np.full(self.n, config.initial_capital, dtype=float)
        self.alive = np.ones(self.n, dtype=bool)
        self.fees = np.zeros(self.n)
        self.impact = np.zeros(self.n)
        self.funding = np.zeros(self.n)
        self.future_gross = np.zeros(self.n)
        self.funding_base = np.zeros(self.n)
        self.end_option = self.option[:, 0].copy()
        self.future_orders = np.zeros(self.n, dtype=int)
        self.forced = np.zeros(self.n, dtype=int)
        self.writeoffs = np.zeros(self.n, dtype=int)
        self.margin_constrained = np.zeros(self.n, dtype=int)
        self.max_margin = np.zeros(self.n)
        self.min_free_cash = np.full(self.n, config.initial_capital, dtype=float)
        self.actions = np.zeros((self.n, self.steps), dtype=int)
        self.stt = {asset: transaction_fees(asset, -1, 1., timestamp, config)["stt"] for asset in ("option", "future")}
        self.i = -1

    def fees_for(self, asset, change, fill):
        c = self.c
        turnover = abs(change)*fill
        brokerage = np.full(self.n, c.brokerage_per_order) if asset == "option" else np.minimum(c.brokerage_per_order, c.future_brokerage_cap_rate*turnover)
        taxable = brokerage+turnover*(getattr(c, asset+"_exchange_rate")+c.sebi_rate+getattr(c, asset+"_ipft_rate"))
        total = taxable*(1+c.gst_rate)+turnover*(np.where(change < 0, self.stt[asset], 0.)
                     +np.where(change > 0, getattr(c, asset+"_stamp_rate"), 0.))
        return np.where(change != 0, total, 0.)

    def fill(self, asset, change, mark):
        tick = self.c.option_tick_size if asset == "option" else self.tick
        friction = getattr(self.c, asset+"_half_spread")+getattr(self.c, asset+"_slippage")
        desired = mark+np.sign(change)*friction
        return np.where(change > 0, np.ceil(desired/tick-1e-9), np.floor(desired/tick+1e-9))*tick

    def option_trade(self, change, mark, closing=False):
        fill = self.fill("option", change, mark)
        unavailable = (change != 0) & (fill <= 0)
        if np.any(unavailable & ((change > 0) | (not closing))):
            raise ValueError("Unexecutable initial option proxy")
        self.writeoffs += unavailable.astype(int)
        fill = np.where(unavailable, 0., fill)
        fees = self.fees_for("option", np.where(unavailable, 0, change), fill)
        self.cash -= change*fill+fees
        self.q += change
        self.fees += fees
        self.impact += change*(fill-mark)

    def preview(self, desired_lots):
        change = desired_lots*self.lot-self.h
        fill = self.fill("future", change, self.future[:, self.i])
        if np.any((change != 0) & (fill <= 0)):
            raise ValueError("Unexecutable futures proxy")
        fees = self.fees_for("future", change, fill)
        new_h = self.h+change
        same = (self.h == 0) | (self.h*change >= 0)
        weighted = np.divide(abs(self.h)*self.average+abs(change)*fill, abs(new_h),
                             out=np.zeros(self.n), where=new_h != 0)
        realized = np.where(same, 0., np.minimum(abs(self.h), abs(change))*np.sign(self.h)*(fill-self.average))
        new_average = np.where(same, weighted, np.where(new_h == 0, 0., np.where(self.h*new_h < 0, fill, self.average)))
        return new_h, new_average, self.cash+realized-fees, fees, change*(fill-self.future[:, self.i]), change

    def account(self, h=None, average=None, cash=None):
        h = self.h if h is None else h
        average = self.average if average is None else average
        cash = self.cash if cash is None else cash
        future_mark = self.future[:, self.i]
        unrealized = h*(future_mark-average)
        equity = cash+self.q*self.option[:, self.i]+unrealized
        margin = abs(h)*future_mark*self.c.future_margin_fraction
        short_margin = np.where(self.q < 0, self.c.option_lots*self.c.short_option_margin_per_lot, 0.)
        free = cash+np.minimum(unrealized, 0.)-margin-short_margin
        reserve = np.maximum(equity, 0.)*(1-self.c.capital_fraction)
        return equity, margin+short_margin, free, reserve

    def commit(self, desired_lots, mask):
        desired_lots = np.where(mask, desired_lots, self.h//self.lot)
        h, avg, cash, fees, impact, change = self.preview(desired_lots)
        self.h, self.average, self.cash = h, avg, cash
        self.fees += fees
        self.impact += impact
        self.future_orders += (change != 0).astype(int)

    def prepare(self, i):
        if i != self.i+1:
            raise ValueError("Simulation must advance sequentially")
        self.i = i
        if i:
            self.future_gross += self.h*(self.future[:, i]-self.future[:, i-1])
            charge = self.funding_base*self.c.funding_rate*self.c.bar_minutes/(365*1440.)
            self.cash -= charge
            self.funding += charge
        else:
            self.option_trade(self.original_q, self.option[:, 0])
        equity, _, free, reserve = self.account()
        breach = self.alive & ((equity <= 0) | (free < reserve-1e-8))
        terminal = self.alive & (breach | (i == self.steps))
        self.forced += breach.astype(int)
        if terminal.any():
            self.commit(np.zeros(self.n, int), terminal)
            self.option_trade(np.where(terminal, -self.q, 0), self.option[:, i], closing=True)
            self.end_option = np.where(terminal, self.option[:, i], self.end_option)
            self.alive &= ~terminal
        return self.alive.copy()

    def act(self, desired_lots):
        desired = np.asarray(desired_lots)
        if desired.shape != (self.n,) or not np.isfinite(desired).all() or np.any(desired != np.round(desired)) or np.any(abs(desired) > self.c.max_future_lots):
            raise ValueError("Need bounded whole-lot actions")
        current = self.h//self.lot
        chosen = current.copy()
        found = ~self.alive
        for reduction in range(self.c.max_future_lots+1):
            candidates = np.sign(desired)*np.maximum(abs(desired)-reduction, 0)
            h, avg, cash, _, _, _ = self.preview(np.where(self.alive, candidates, current))
            equity, _, free, reserve = self.account(h, avg, cash)
            feasible = ~found & (equity > 0) & (free >= reserve-1e-8)
            chosen = np.where(feasible, candidates, chosen).astype(int)
            found |= feasible
        self.margin_constrained += (self.alive & (chosen != desired)).astype(int)
        self.commit(chosen, self.alive)
        self.actions[:, self.i] = self.h//self.lot

    def finish(self):
        _, margin, free, _ = self.account()
        self.max_margin = np.maximum(self.max_margin, margin)
        self.min_free_cash = np.minimum(self.min_free_cash, free)
        self.funding_base = np.maximum(self.q, 0)*self.option[:, 0]+margin

    def results(self):
        if self.alive.any() or np.any(self.q != 0) or np.any(self.h != 0):
            raise AssertionError("Unliquidated positions")
        option_gross = self.original_q*(self.end_option-self.option[:, 0])
        gross = option_gross+self.future_gross
        pnl = self.cash-self.c.initial_capital
        error = pnl-(gross-self.fees-self.impact-self.funding)
        if np.max(abs(error)) > 1e-6:
            raise AssertionError("Vector cash-flow reconciliation failed")
        return {"net_pnl": pnl, "gross_pnl": gross, "fees": self.fees, "execution_impact": self.impact,
                "funding": self.funding, "future_orders": self.future_orders, "forced_exits": self.forced,
                "writeoffs": self.writeoffs, "margin_constrained": self.margin_constrained,
                "peak_margin": self.max_margin, "minimum_free_cash": self.min_free_cash,
                "reconciliation_error": error, "actions": self.actions.copy()}


def baseline_actions(environment, strategy):
    if strategy not in ("unhedged", "static_delta", "dynamic_delta", "band_delta", "bs_dynamic"):
        raise ValueError(f"Unknown simulation baseline: {strategy}")
    e, i, c = environment, environment.i, environment.c
    current = e.h//e.lot
    if strategy == "unhedged":
        return np.zeros(e.n, dtype=int)
    if strategy == "static_delta" and i:
        return current.copy()
    delta = e.batch["bs_delta"][:, i] if strategy == "bs_dynamic" else e.delta[:, i]
    desired_units = -e.original_q*delta  # Simulated zero carry implies F/S = 1.
    raw = desired_units/e.lot
    rounded = (np.sign(raw)*np.floor(abs(raw)+.5-1e-12)).astype(int)
    desired = np.clip(rounded, -c.max_future_lots, c.max_future_lots)
    if strategy == "band_delta" and i:
        hold = abs(desired_units-e.h) <= c.no_trade_band*abs(e.original_q)
        desired = np.where(hold, current, desired)
    desired = np.where(np.isfinite(delta), desired, current)
    return desired


def rollout_baseline(batch, config, strategy):
    e = BatchEnvironment(batch, config)
    for i in range(e.steps+1):
        active = e.prepare(i)
        if i < e.steps and active.any():
            e.act(baseline_actions(e, strategy))
        e.finish()
    return e.results()
