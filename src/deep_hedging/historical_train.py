"""Small residual adaptation; train/validation only, with an epoch-zero fallback."""

import copy
from dataclasses import asdict, dataclass
import json

import numpy as np
import pandas as pd
import torch

from .historical_data import subset_batch
from .historical_policy import HistoricalPolicy, rollout_historical, save_historical_policy
from .metrics import empirical_loss_risk
from .objectives import cvar_objective, score_function_loss


@dataclass(frozen=True)
class HistoricalTraining:
    seeds: tuple = (11, 22, 33)
    epochs: int = 8
    updates_per_epoch: int = 16
    batch_size: int = 128
    learning_rate: float = .0003
    residual_penalty: float = .02
    risk_scale: float = 20000.
    alpha: float = .95

    def __post_init__(self):
        if self.epochs < 1 or self.updates_per_epoch < 1 or self.batch_size < 2 or not self.seeds:
            raise ValueError("Invalid historical training budget")
        if self.learning_rate <= 0 or self.residual_penalty < 0 or self.risk_scale <= 0 or not 0 < self.alpha < 1:
            raise ValueError("Invalid historical objective")


def train_ablation(base, training_batch, validation_batch, config, spec, output, use_ml):
    output.mkdir(parents=True, exist_ok=False)
    with torch.no_grad():
        reference = rollout_historical(training_batch, config, baseline="band_delta")[0]["net_pnl"]
    records, candidates = [], []
    for seed in spec.seeds:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed+4000)
        model = HistoricalPolicy(copy.deepcopy(base), use_ml)
        with torch.no_grad():
            initial_pnl = rollout_historical(training_batch, config, model, stochastic=True)[0]["net_pnl"]
            validation_pnl = rollout_historical(validation_batch, config, model)[0]["net_pnl"]
        eta = torch.nn.Parameter(torch.tensor(float(np.quantile(-initial_pnl/spec.risk_scale, spec.alpha))))
        optimizer = torch.optim.Adam([{"params": model.residual.parameters()}, {"params": [eta], "lr": .002}], lr=spec.learning_rate)
        best = empirical_loss_risk(validation_pnl, spec.alpha)[1]
        best_state, best_epoch = copy.deepcopy(model.state_dict()), 0
        records.append({"seed": seed, "epoch": 0, "validation_cvar": best, "validation_mean_pnl": float(validation_pnl.mean()), "gradient_norm": 0.})
        for epoch in range(1, spec.epochs+1):
            gradients = []
            for _ in range(spec.updates_per_epoch):
                idx = rng.integers(0, len(reference), spec.batch_size)
                result, logp, entropy = rollout_historical(subset_batch(training_batch, idx), config, model, stochastic=True)
                actor, _ = score_function_loss(result["net_pnl"], logp, entropy, spec.risk_scale,
                                                spec.alpha, .0001, eta=eta, reference_pnl=reference[idx])
                threshold = cvar_objective(-torch.tensor(result["net_pnl"], dtype=torch.float32)/spec.risk_scale, spec.alpha, eta)
                penalty = spec.residual_penalty*sum(p.square().sum() for p in model.residual.parameters())
                optimizer.zero_grad()
                objective = actor+threshold+penalty
                if not torch.isfinite(objective):
                    raise ArithmeticError("Non-finite historical objective")
                objective.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.residual.parameters(), 2.)
                eta.grad.clamp_(-5., 5.)
                optimizer.step()
                gradients.append(float(gradient))
            with torch.no_grad():
                validation_pnl = rollout_historical(validation_batch, config, model)[0]["net_pnl"]
            value = empirical_loss_risk(validation_pnl, spec.alpha)[1]
            records.append({"seed": seed, "epoch": epoch, "validation_cvar": value,
                            "validation_mean_pnl": float(validation_pnl.mean()), "gradient_norm": float(np.mean(gradients)),
                            "learned_var_rupees": float(eta.detach())*spec.risk_scale})
            if value < best:
                best, best_epoch, best_state = value, epoch, copy.deepcopy(model.state_dict())
            print(f"Historical {'ML' if use_ml else 'no-ML'} seed={seed} epoch={epoch}: validation CVaR={value:.2f}", flush=True)
        model.load_state_dict(best_state)
        file = output/f"policy_seed_{seed}.pt"
        metadata = {"seed": seed, "best_epoch": best_epoch, "validation_cvar": float(best), "training_spec": asdict(spec),
                    "selection": "Minimum validation episode CVaR, including the unchanged simulation-policy fallback"}
        save_historical_policy(file, model, config, metadata)
        candidates.append({**metadata, "checkpoint": file.name})
    pd.DataFrame(records).to_csv(output/"learning_curves.csv", index=False)
    selected = min(candidates, key=lambda row: (row["validation_cvar"], row["best_epoch"], row["seed"]))
    selection = {"selected": selected, "candidates": candidates, "uses_ml": use_ml}
    (output/"selection.json").write_text(json.dumps(selection, indent=2))
    return selection
