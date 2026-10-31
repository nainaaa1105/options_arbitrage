"""Behavior-cloned warm start followed by discrete CVaR policy gradients."""

from dataclasses import dataclass, asdict
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from .batch_environment import BatchEnvironment, baseline_actions, rollout_baseline
from .metrics import empirical_loss_risk
from .objectives import score_function_loss, cvar_objective
from .policy import HedgePolicy, current_features, rollout_policy, FEATURE_NAMES
from .simulation import simulate


@dataclass(frozen=True)
class TrainingSpec:
    seeds: tuple = (11, 22, 33)
    epochs: int = 12
    updates_per_epoch: int = 32
    batch_size: int = 256
    warm_updates: int = 60
    validation_paths: int = 2048
    test_paths: int = 8192
    stress_paths: int = 4096
    learning_rate: float = .0005
    warm_learning_rate: float = .002
    alpha: float = .95
    risk_scale: float = 20000.
    hidden: int = 32
    threads: int = 2

    def __post_init__(self):
        if self.epochs < 1 or self.updates_per_epoch < 1 or self.batch_size < 32 or self.warm_updates < 0:
            raise ValueError("Invalid training budget")
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("Use distinct policy seeds")
        if min(self.validation_paths, self.test_paths, self.stress_paths) < 128:
            raise ValueError("Need at least 128 validation/test paths")
        if not 0 < self.alpha < 1 or self.risk_scale <= 0 or self.learning_rate <= 0 or self.threads < 1:
            raise ValueError("Invalid optimizer/risk settings")


def warm_start(model, optimizer, spec, oracle, config, training, rng):
    last = 0.
    for _ in range(training.warm_updates):
        batch = simulate(spec, training.batch_size, rng=rng, oracle=oracle)
        env = BatchEnvironment(batch, config)
        features, labels = [], []
        for i in range(env.steps+1):
            active = env.prepare(i)
            if i < env.steps and active.any():
                teacher = baseline_actions(env, "band_delta")
                features.append(current_features(env)[active])
                labels.append(abs(teacher[active]))
                env.act(teacher)
            env.finish()
        x = torch.from_numpy(np.concatenate(features))
        y = torch.from_numpy(np.concatenate(labels)).long()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        last = float(loss.detach())
    return last


def fit_seed(spec, oracle, config, training, validation, seed, output):
    torch.manual_seed(seed)
    rng = np.random.default_rng(1000+seed+(100000 if spec.model == "heston" else 0))
    model = HedgePolicy(config.max_future_lots, training.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=training.warm_learning_rate)
    warm_loss = warm_start(model, optimizer, spec, oracle, config, training, rng)
    with torch.no_grad():
        warm_pnl = rollout_policy(validation, config, model)[0]["net_pnl"]
    warm_cvar = empirical_loss_risk(warm_pnl, training.alpha)[1]
    print(f"{spec.model} seed={seed}: warm-start validation CVaR={warm_cvar:.2f}", flush=True)
    warm_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for group in optimizer.param_groups:
        group["lr"] = training.learning_rate
    initial_batch = simulate(spec, training.batch_size*4, rng=rng, oracle=oracle)
    with torch.no_grad():
        initial_pnl = rollout_policy(initial_batch, config, model, stochastic=True)[0]["net_pnl"]
    eta = torch.nn.Parameter(torch.tensor(float(np.quantile(-initial_pnl/training.risk_scale, training.alpha))))
    optimizer.add_param_group({"params": [eta], "lr": .005})
    best, best_epoch, best_state = float("inf"), None, None
    history = []
    initial_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, training.epochs+1):
        model.train()
        objectives, train_cvars, gradient_norms = [], [], []
        for _ in range(training.updates_per_epoch):
            batch = simulate(spec, training.batch_size, rng=rng, oracle=oracle)
            result, logp, entropy = rollout_policy(batch, config, model, stochastic=True)
            reference = rollout_baseline(batch, config, "band_delta")["net_pnl"]
            optimizer.zero_grad()
            entropy_weight = .001*(1-epoch/training.epochs)
            loss, risk = score_function_loss(result["net_pnl"], logp, entropy, training.risk_scale,
                                              training.alpha, entropy_weight, eta=eta, reference_pnl=reference)
            threshold_loss = cvar_objective(-torch.as_tensor(result["net_pnl"], dtype=torch.float32)/training.risk_scale,
                                           training.alpha, eta=eta)
            if not torch.isfinite(loss):
                raise ArithmeticError("Non-finite CVaR policy objective")
            (loss+threshold_loss).backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            if not torch.isfinite(grad):
                raise ArithmeticError("Non-finite policy gradient")
            eta.grad.clamp_(-5., 5.)
            optimizer.step()
            objectives.append(float(loss.detach()))
            train_cvars.append(float(risk)*training.risk_scale)
            gradient_norms.append(float(grad))
        model.eval()
        with torch.no_grad():
            validation_result = rollout_policy(validation, config, model)[0]
        value = empirical_loss_risk(validation_result["net_pnl"], training.alpha)[1]
        history.append({"seed": seed, "epoch": epoch, "score_function_objective": np.mean(objectives),
                        "sampled_training_cvar": np.mean(train_cvars), "mean_gradient_norm_before_clip": np.mean(gradient_norms),
                        "greedy_validation_cvar": value, "greedy_validation_mean_pnl": float(validation_result["net_pnl"].mean()),
                        "validation_mean_future_orders": float(validation_result["future_orders"].mean())})
        history[-1]["learned_var_threshold_rupees"] = float(eta.detach())*training.risk_scale
        if value < best:
            best, best_epoch = value, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(f"{spec.model} seed={seed} epoch={epoch}/{training.epochs}: validation CVaR={value:.2f}", flush=True)
    changed = float(sum((model.state_dict()[k]-initial_state[k]).abs().sum().item()
                        for k in initial_state if initial_state[k].is_floating_point()))
    if changed <= 0:
        raise AssertionError("No CVaR-training parameter update occurred")
    checkpoint = {"state_dict": best_state, "max_lots": config.max_future_lots, "hidden": training.hidden,
                  "feature_names": list(FEATURE_NAMES), "seed": seed, "best_epoch": best_epoch,
                  "validation_cvar": float(best), "warm_validation_cvar": float(warm_cvar),
                  "execution_config": config.to_dict(), "market_spec": asdict(spec),
                  "training_spec": asdict(training), "torch_version": str(torch.__version__),
                  "algorithm": "Discrete CVaR score-function policy gradient; learned eta; action-independent path control variate; whole lots",
                  "cvar_parameter_l1_change_from_warm_start": changed,
                  "selection": "Minimum greedy validation CVaR among post-CVaR-training checkpoints only"}
    path = output / f"policy_seed_{seed}.pt"
    torch.save(checkpoint, path)
    torch.save({**checkpoint, "state_dict": warm_state, "best_epoch": 0,
                "algorithm": "Behavior-cloning-only diagnostic, not the selected CVaR checkpoint"}, output / f"warm_seed_{seed}.pt")
    pd.DataFrame(history).to_csv(output / f"learning_curve_seed_{seed}.csv", index=False)
    return {"seed": seed, "checkpoint": path.name, "validation_cvar": float(best), "best_epoch": best_epoch,
            "warm_validation_cvar": float(warm_cvar), "warm_cross_entropy": warm_loss,
            "parameter_change_after_warm_start": changed}


def train_market(spec, oracle, config, training, output, validation_seed):
    output.mkdir(parents=True, exist_ok=False)
    validation = simulate(spec, training.validation_paths, seed=validation_seed, oracle=oracle)
    np.savez_compressed(output / "validation_paths.npz", **validation)
    trials = [fit_seed(spec, oracle, config, training, validation, seed, output) for seed in training.seeds]
    winner = min(trials, key=lambda row: (row["validation_cvar"], row["seed"]))
    result = {"market": asdict(spec), "training": asdict(training), "execution": config.to_dict(),
              "validation_seed": validation_seed, "trials": trials, "selected": winner,
              "selection": "Validation CVaR only; test paths not generated or opened during fitting"}
    (output / "training_selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
