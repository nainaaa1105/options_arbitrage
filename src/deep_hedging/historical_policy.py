"""Simulation transfer and small historical residuals using completed-bar state."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .batch_environment import BatchEnvironment
from .costs import execution_price, transaction_fees
from .policy import FEATURE_NAMES, HedgePolicy, feature_values

HISTORICAL_FEATURES = (*FEATURE_NAMES, "convergence_probability_centered", "convergence_available")


class HistoricalEnvironment(BatchEnvironment):
    def __init__(self, batch, config):
        super().__init__(batch, config)
        self.stt = {asset: np.asarray(batch[asset+"_stt"]) for asset in ("option", "future")}


def observed_features(spot, strike, maturity, variance, delta, sensitivity, call, remaining,
                      current_lots, cash, unrealized, option_close, future_close, q, lot,
                      recent_return, roundtrip, initial_capital, config):
    equity = cash+q*option_close+unrealized
    margin = abs(current_lots*lot)*future_close*config.future_margin_fraction
    short = np.where(q < 0, config.option_lots*config.short_option_margin_per_lot, 0.)
    free = cash+np.minimum(unrealized, 0.)-margin-short
    values = feature_values(spot, strike, maturity, variance, delta/sensitivity, call, remaining,
                            current_lots, free, q, lot,
                            {"return": recent_return, "roundtrip": roundtrip, "pnl": equity-initial_capital}, config)
    # The simulation trainer's day count is 252; historical pricing is ACT/365.
    # Keep IV^2*T intact for moneyness, and supply calendar days explicitly.
    values[:, 1] = np.clip(np.broadcast_to(np.asarray(maturity)*365/10., np.shape(spot)), -8., 8.)
    return values


def batch_features(e):
    b, i, c = e.batch, e.i, e.c
    of, ff = b["observed_option"][:, i], b["observed_future"][:, i]
    cash = e.cash.copy()
    if i == 0:
        # The first option and futures orders are simultaneous. Estimate option
        # funding from its observed close, not the as-yet-unknown opening fill.
        fill = e.fill("option", e.original_q, of)
        cash = c.initial_capital-e.original_q*fill-e.fees_for("option", e.original_q, fill)
    unrealized = e.h*(ff-e.average)
    roundtrip = e.fees_for("future", e.lot, ff)+e.fees_for("future", -e.lot, ff)
    roundtrip += 2*e.lot*(c.future_half_spread+c.future_slippage)
    x = observed_features(b["spot"][:, i], b["strike"], b["maturity"][:, i], b["variance"][:, i],
                          b["observed_delta_used"][:, i], b["future_spot_delta"][:, i], b["call"], (e.steps-i)/e.steps,
                          e.h/e.lot, cash, unrealized, of, ff, e.original_q, e.lot,
                          b["recent_return"][:, i], roundtrip, c.initial_capital, c)
    return np.column_stack([x, b["probability"][:, i]-.5, b["ml_available"][:, i]]).astype(np.float32)


class HistoricalPolicy(nn.Module):
    """Frozen Part 2 policy plus a regularized 85-parameter linear residual."""
    def __init__(self, base, use_ml):
        super().__init__()
        self.base, self.use_ml = base, bool(use_ml)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.residual = nn.Linear(len(HISTORICAL_FEATURES), base.max_lots+1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def forward(self, features):
        x = features if self.use_ml else torch.cat([features[:, :len(FEATURE_NAMES)], torch.zeros_like(features[:, -2:])], dim=1)
        return self.base(x[:, :len(FEATURE_NAMES)])+self.residual(x)


def logits_for(model, features):
    return model(features if isinstance(model, HistoricalPolicy) else features[:, :len(FEATURE_NAMES)])


def rollout_historical(batch, config, model=None, stochastic=False, baseline=None):
    from .baselines import target_lots
    e = HistoricalEnvironment(batch, config)
    logp, entropy = torch.zeros(e.n), torch.zeros(e.n)
    for i in range(e.steps+1):
        active = e.prepare(i)
        if i < e.steps and active.any():
            if baseline is not None:
                desired = np.array([target_lots(baseline, {"delta": batch["delta"][j, i],
                                    "future_spot_delta": batch["future_spot_delta"][j, i],
                                    "option_quantity": e.original_q[j], "future_lot_size": e.lot[j]},
                                    int(e.h[j]//e.lot[j]), i == 0, config)[0] for j in range(e.n)])
            else:
                distribution = Categorical(logits=logits_for(model, torch.from_numpy(batch_features(e))))
                magnitude = distribution.sample() if stochastic else distribution.logits.argmax(dim=1)
                if stochastic:
                    mask = torch.from_numpy(active.astype(np.float32))
                    logp = logp+distribution.log_prob(magnitude)*mask
                    entropy = entropy+distribution.entropy()*mask
                sign = -np.sign(e.original_q*np.where(batch["call"], 1., -1.)).astype(int)
                desired = magnitude.detach().numpy()*sign
                desired = np.where(batch["pricing_state_valid"][:, i], desired, e.h//e.lot)
            e.act(desired)
        e.finish()
    return e.results(), logp, entropy


class HistoricalScalarPolicy:
    def __init__(self, model, starting_cash):
        self.model, self.starting_cash = model, starting_cash

    def features(self, state, current_lots, initial, config):
        s, c = state, config
        q, lot = s["option_quantity"], s["future_lot_size"]
        future_close, option_close = s["future_close_observed"], s["option_close_observed"]
        cash = s["cash"]
        if initial:
            fill = execution_price(option_close, q, c.option_tick_size, c.option_half_spread, c.option_slippage)
            cash = self.starting_cash-q*fill-transaction_fees("option", q, fill, s["timestamp"], c)["total"]
        unrealized = s["future_unrealized"]+current_lots*lot*(future_close-s["future_mark"])
        fees = sum(transaction_fees("future", sign*lot, future_close, s["timestamp"], c)["total"] for sign in (-1, 1))
        roundtrip = fees+2*lot*(c.future_half_spread+c.future_slippage)
        x = observed_features(np.array([s["spot_observed"]]), s["strike"], s["remaining_time"], s["volatility"]**2,
                              s["observed_delta_used"], s["future_spot_delta"], s["is_call"], s["remaining_steps"]/(c.horizon_minutes/c.bar_minutes),
                              current_lots, cash, unrealized, option_close, future_close, q, np.array([lot]),
                              s["recent_return"], roundtrip, self.starting_cash, c)
        return np.column_stack([x, [s["probability"]-.5], [float(s["ml_available"])] ]).astype(np.float32)

    def __call__(self, state, current_lots, initial, config):
        if not state["pricing_state_valid"]:
            return current_lots, "invalid_delta_hold"
        features = torch.from_numpy(self.features(state, current_lots, initial, config))
        with torch.no_grad():
            magnitude = int(logits_for(self.model, features).argmax(dim=1).item())
        direction = -int(np.sign(state["option_quantity"]*(1 if state["is_call"] else -1)))
        return magnitude*direction, "historical_neural_ml" if getattr(self.model, "use_ml", False) else "historical_neural_no_ml"


def save_historical_policy(path, model, config, metadata):
    torch.save({"state_dict": model.state_dict(), "use_ml": model.use_ml,
                "feature_names": list(HISTORICAL_FEATURES), "max_lots": model.base.max_lots,
                "hidden": model.base.hidden, "execution_config": config.to_dict(), **metadata}, path)


def load_historical_policy(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["feature_names"] != list(HISTORICAL_FEATURES):
        raise ValueError("Historical feature schema mismatch")
    model = HistoricalPolicy(HedgePolicy(saved["max_lots"], saved["hidden"]), saved["use_ml"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved
