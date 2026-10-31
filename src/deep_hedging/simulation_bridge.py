"""Replay simulated paths through the original scalar Part 1 execution engine."""

import numpy as np
import pandas as pd

from .engine import run_episode
from .policy import feature_values
from .costs import transaction_fees


def episode_frame(batch, index, config):
    steps = batch["future"].shape[1]-1
    start = pd.Timestamp("2026-10-01 09:35", tz="Asia/Kolkata")
    ts = pd.date_range(start, periods=steps+1, freq=f"{config.bar_minutes}min")
    lot = int(np.broadcast_to(batch.get("future_lot_size", 65), (len(batch["future"]),))[index])
    option_lot = int(np.broadcast_to(batch.get("option_lot_size", lot), (len(batch["future"]),))[index])
    path = pd.DataFrame({"timestamp": ts, "observed_at": ts, "option_mark": batch["option"][index],
                         "future_mark": batch["future"][index], "delta": batch["delta"][index],
                         "future_spot_delta": 1., "future_lot_size": lot,
                         "future_tick_size": float(batch.get("future_tick_size", .1)),
                         "future_instrument_key": "SIM_FUTURE", "spot_observed": batch["spot"][index],
                         "volatility": np.sqrt(batch["variance"][index]), "remaining_time": batch["maturity"][index],
                         "strike": batch["strike"][index], "is_call": bool(batch["call"][index]),
                         "recent_return": np.r_[0., np.diff(np.log(batch["spot"][index]))]})
    ep = {"episode_id": f"sim_{index}", "instrument_key": "SIM_OPTION", "entry_time": ts[0], "exit_time": ts[-1],
          "option_quantity": config.option_side*config.option_lots*option_lot,
          "option_lot_size": option_lot, "option_lots": config.option_lots}
    return ep, path


class ScalarTorchPolicy:
    def __init__(self, model):
        self.model = model

    def __call__(self, state, current_lots, initial, config):
        import torch
        s = state
        spot = np.array([s["spot_observed"]])
        lot = np.array([s["future_lot_size"]])
        observed_future = s["future_mark"]
        fee = sum(transaction_fees("future", sign*s["future_lot_size"], observed_future, s["timestamp"], config)["total"]
                  for sign in (-1, 1))
        cost = {"return": s["recent_return"],
                "roundtrip": fee+2*s["future_lot_size"]*(config.future_half_spread+config.future_slippage),
                "pnl": s["equity"]-config.initial_capital}
        features = feature_values(spot, s["strike"], s["remaining_time"], s["volatility"]**2,
                                  s["delta"], s["is_call"], s["remaining_steps"]/(config.horizon_minutes/config.bar_minutes),
                                  current_lots, s["free_cash"], s["option_quantity"], lot, cost, config)
        with torch.no_grad():
            magnitude = int(self.model(torch.from_numpy(features)).argmax(dim=1).item())
        direction = -int(np.sign(s["option_quantity"]*(1 if s["is_call"] else -1)))
        return direction*magnitude, "neural_policy"


def scalar_replay(batch, config, strategy, model=None, count=16):
    values = []
    for index in range(min(count, len(batch["future"]))):
        ep, path = episode_frame(batch, index, config)
        policy = ScalarTorchPolicy(model) if model is not None else None
        result, _, _ = run_episode(ep, path, strategy, config.initial_capital, config, policy=policy)
        values.append(result["net_pnl"])
    return np.array(values)
