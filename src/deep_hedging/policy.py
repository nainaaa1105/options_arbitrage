"""Small inventory-aware categorical policy; its actions are whole lots."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from .batch_environment import BatchEnvironment
from .simulation import TRADING_MINUTES

FEATURE_NAMES = ("standardized_moneyness", "maturity_days_scaled", "volatility_scaled",
                 "absolute_delta", "is_call", "remaining_horizon", "current_hedge_magnitude",
                 "delta_target_magnitude", "recent_standardized_return", "free_cash_fraction",
                 "margin_per_lot_fraction", "roundtrip_cost_over_risk", "lot_ratio", "portfolio_pnl_over_risk")


def feature_values(spot, strike, maturity, variance, delta, call, remaining,
                   current_lots, free_cash, q, lot, cost, config):
    sigma = np.sqrt(np.maximum(variance, 1e-8))
    values = (np.log(spot/strike)/np.maximum(sigma*np.sqrt(maturity), 1e-5),
              maturity*252/10., sigma/.4, abs(delta), np.asarray(call, float), remaining,
              abs(current_lots)/config.max_future_lots,
              abs(q*delta/lot)/config.max_future_lots,
              cost["return"]/np.maximum(sigma*np.sqrt(config.bar_minutes/TRADING_MINUTES), 1e-6),
              free_cash/config.initial_capital, lot*spot*config.future_margin_fraction/config.initial_capital,
              cost["roundtrip"]/np.maximum(abs(q)*spot*sigma*np.sqrt(config.horizon_minutes/TRADING_MINUTES), 1.),
              abs(q)/(lot*config.option_lots),
              cost["pnl"]/np.maximum(abs(q)*spot*sigma*np.sqrt(config.horizon_minutes/TRADING_MINUTES), 1.))
    result = np.column_stack([np.broadcast_to(v, np.shape(spot)) for v in values]).astype(np.float32)
    result = np.clip(result, -8., 8.)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite policy state")
    return result


def current_features(environment):
    e, i, c, b = environment, environment.i, environment.c, environment.batch
    spot = b["spot"][:, i]
    recent = np.log(spot/b["spot"][:, i-1]) if i else np.zeros(e.n)
    equity, _, free, _ = e.account()
    roundtrip = (e.fees_for("future", e.lot, spot)+e.fees_for("future", -e.lot, spot)
                 +2*e.lot*(c.future_half_spread+c.future_slippage))
    return feature_values(spot, b["strike"], b["maturity"][:, i], b["variance"][:, i],
                          b["delta"][:, i], b["call"], (e.steps-i)/e.steps,
                          e.h/e.lot, free, e.original_q, e.lot,
                          {"return": recent, "roundtrip": roundtrip, "pnl": equity-c.initial_capital}, c)


class HedgePolicy(nn.Module):
    def __init__(self, max_lots=4, hidden=32):
        super().__init__()
        if max_lots < 1:
            raise ValueError("Neural policy needs positive action capacity")
        self.max_lots, self.hidden = max_lots, hidden
        self.network = nn.Sequential(nn.Linear(len(FEATURE_NAMES), hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, max_lots+1))
        nn.init.normal_(self.network[-1].weight, std=.01)
        nn.init.zeros_(self.network[-1].bias)
        self.register_buffer("action_magnitudes", torch.arange(max_lots+1, dtype=torch.float32))

    def forward(self, features):
        # Transparent delta-informed prior; trainable residual logits learn
        # inventory/cost/risk-dependent departures from it.
        target = features[:, 7:8]*self.max_lots
        prior = -1.5*(self.action_magnitudes[None, :]-target)**2
        return self.network(features)+prior


def rollout_policy(batch, config, policy, stochastic=False):
    e = BatchEnvironment(batch, config)
    log_probability = torch.zeros(e.n)
    entropy = torch.zeros(e.n)
    for i in range(e.steps+1):
        active = e.prepare(i)
        if i < e.steps and active.any():
            features = torch.from_numpy(current_features(e))
            logits = policy(features)
            distribution = Categorical(logits=logits)
            action = distribution.sample() if stochastic else logits.argmax(dim=1)
            if stochastic:
                mask = torch.from_numpy(active.astype(np.float32))
                log_probability = log_probability+distribution.log_prob(action)*mask
                entropy = entropy+distribution.entropy()*mask
            direction = -np.sign(e.original_q*np.where(batch["call"], 1., -1.)).astype(int)
            e.act(action.detach().numpy()*direction)
        e.finish()
    return e.results(), log_probability, entropy


def load_policy(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["feature_names"] != list(FEATURE_NAMES):
        raise ValueError("Checkpoint state schema differs from policy implementation")
    model = HedgePolicy(saved["max_lots"], saved["hidden"])
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, saved
