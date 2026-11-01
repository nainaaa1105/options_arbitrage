"""Causal recurrent (GRU) and matched memoryless hedge policies for the continuous short-option track.

Every policy here maps the *current* observable state to the next bounded
position through ``step(x, state)``. The hidden state starts at zero for every
path, is updated from the current state only, and is never shared across rows,
chunks or episodes. Existing Phase 4 policies are wrapped, not modified.
"""
import copy
from dataclasses import asdict, dataclass
import time

import pandas as pd
import torch
from torch import nn

from .accounting import pnl
from .evaluation import metrics
from .objectives import risk
from .policies import ContinuousPolicy, FEATURES, features
from .market import simulate

SCHEMA = 'phase6_step_policy'
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PolicySpec:
    name: str
    family: str                  # 'mlp' or 'gru'
    initialization: str = 'delta'
    hidden: int = 32
    drop: tuple = ()             # FEATURES zeroed at the network input (prior and call sign untouched)
    reset_memory: bool = False   # GRU only: hidden state zeroed before every step (memory ablation)

    def __post_init__(self):
        if self.family not in ('mlp', 'gru'):
            raise ValueError('Unknown policy family')
        if self.initialization not in ('delta', 'flat'):
            raise ValueError('Unknown initialization')
        if self.hidden < 1:
            raise ValueError('Hidden width must be positive')
        if any(name not in FEATURES for name in self.drop):
            raise ValueError('Unknown feature in ablation mask')
        if self.reset_memory and self.family != 'gru':
            raise ValueError('Only a GRU has memory to reset')


def feature_mask(drop):
    unknown = [name for name in drop if name not in FEATURES]
    if unknown:
        raise ValueError(f'Unknown features: {unknown}')
    return torch.tensor([0. if name in drop else 1. for name in FEATURES])


def _bounded(raw, x, max_position, initialization):
    # The analytic prior and the call/put sign use the unmasked state: they are
    # disclosed structure, not learned inputs. Only the network input is ablated.
    prior = x[..., 3] if initialization == 'delta' else torch.full_like(x[..., 3], .5)
    magnitude = (prior + raw).clamp(0, max_position)
    return torch.where(x[..., 4] > .5, magnitude, -magnitude)


class MLPStep(nn.Module):
    """Phase 4 projected MLP behind the step interface; identical when the mask is all ones."""

    def __init__(self, initialization='delta', max_position=1.2, hidden=32, drop=()):
        super().__init__()
        self.initialization, self.max_position, self.hidden = initialization, max_position, hidden
        self.core = ContinuousPolicy(initialization, max_position, hidden)
        self.register_buffer('mask', feature_mask(drop))

    def initial_state(self, n, dtype=torch.float32):
        return None

    def step(self, x, state):
        raw = self.core.network(x * self.mask).squeeze(-1)
        return _bounded(raw, x, self.max_position, self.initialization), None


class GRUStep(nn.Module):
    """GRUCell with a zero-initialized linear head; epoch zero equals its prior exactly."""

    def __init__(self, initialization='delta', max_position=1.2, hidden=16, drop=(), reset_memory=False):
        super().__init__()
        self.initialization, self.max_position, self.hidden = initialization, max_position, hidden
        self.reset_memory = bool(reset_memory)
        self.cell = nn.GRUCell(len(FEATURES), hidden)
        self.head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.register_buffer('mask', feature_mask(drop))

    def initial_state(self, n, dtype=torch.float32):
        return torch.zeros(n, self.hidden, dtype=dtype)

    def step(self, x, state):
        previous = torch.zeros_like(state) if self.reset_memory else state
        h = self.cell(x * self.mask, previous)
        return _bounded(self.head(h).squeeze(-1), x, self.max_position, self.initialization), h


class LegacyStep:
    """Adapter for frozen Phase 4 ContinuousPolicy/BandPolicy objects (never retrained here)."""

    def __init__(self, model):
        self.model = model
        self.hidden = getattr(model, 'hidden', 0)

    def initial_state(self, n, dtype=torch.float32):
        return None

    def step(self, x, state):
        return self.model(x), None

    def eval(self):
        self.model.eval()
        return self

    def parameters(self):
        return self.model.parameters()


class Ensemble:
    """Mean-position ensemble. All members observe the ENSEMBLE's executed previous position."""

    def __init__(self, members):
        if not members:
            raise ValueError('Ensemble needs members')
        self.members = list(members)

    def initial_state(self, n, dtype=torch.float32):
        return [m.initial_state(n, dtype) for m in self.members]

    def step(self, x, states):
        outputs, new = [], []
        for member, state in zip(self.members, states):
            position, next_state = member.step(x, state)
            outputs.append(position)
            new.append(next_state)
        return torch.stack(outputs).mean(0), new


def parameter_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def build(spec, max_position):
    if spec.family == 'mlp':
        return MLPStep(spec.initialization, max_position, spec.hidden, spec.drop)
    return GRUStep(spec.initialization, max_position, spec.hidden, spec.drop, spec.reset_memory)


def rollout(model, batch, config):
    """Closed-loop sequential positions [paths, steps]; each path starts with fresh state."""
    n, dtype = len(batch['spot']), batch['spot'].dtype
    previous = torch.zeros(n, dtype=dtype)
    state = model.initial_state(n, dtype)
    outputs = []
    for i in range(config.steps):
        previous, state = model.step(features(batch, config, previous, i), state)
        outputs.append(previous)
    return torch.stack(outputs, dim=1)


def infer(model, batch, config, chunk=1024):
    values = []
    with torch.no_grad():
        for start in range(0, len(batch['spot']), chunk):
            values.append(rollout(model, {k: v[start:start + chunk] for k, v in batch.items()}, config))
    return torch.cat(values)


def latency_ms(model, batch, config, repeats=5):
    sample = {k: v[:1024] for k, v in batch.items()}
    with torch.no_grad():
        rollout(model, sample, config)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            rollout(model, sample, config)
            times.append((time.perf_counter() - start) * 1000)
    return float(sorted(times)[len(times) // 2])


def fit_recurrent(config, fit_config, validation, objective, spec, seed, output, sampler=None, extra_metadata=None):
    """Same optimizer, objective, selection rule and per-seed path stream as ``training.fit``."""
    if config.fixed_cost:
        raise ValueError('Exact fixed trade-count costs are nondifferentiable; use proportional/spread costs')
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 7000)
    model = build(spec, config.max_position)
    epochs = fit_config.flat_epochs if spec.initialization == 'flat' else fit_config.epochs
    eta = torch.nn.Parameter(torch.tensor(.1))
    groups = [{'params': model.parameters(), 'lr': fit_config.learning_rate}]
    if objective == 'cvar':
        groups.append({'params': [eta], 'lr': .01})
    optimizer = torch.optim.Adam(groups)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.95)
    best, best_state, best_epoch, best_eta = float('inf'), None, None, None
    first_state = copy.deepcopy(model.state_dict())
    records = []
    for epoch in range(epochs + 1):
        gradients = []
        if epoch:
            model.train()
            for _ in range(fit_config.updates):
                batch = (simulate(config, fit_config.batch, generator=generator) if sampler is None
                         else sampler(config, fit_config.batch, generator=generator))
                result = pnl(batch, rollout(model, batch, config), config)
                loss = risk(result['net_pnl'], config, objective, eta)
                if not torch.isfinite(loss):
                    raise ArithmeticError('Non-finite recurrent-study loss')
                optimizer.zero_grad()
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                if not torch.isfinite(grad):
                    raise ArithmeticError('Non-finite recurrent-study gradient')
                if objective == 'cvar':
                    eta.grad.clamp_(-10, 10)
                optimizer.step()
                gradients.append(float(grad))
            scheduler.step()
        model.eval()
        with torch.no_grad():
            h = rollout(model, validation, config)
            result = pnl(validation, h, config)
            score = float(risk(result['net_pnl'], config, objective))
            row = {'epoch': epoch, 'validation_objective': score,
                   **metrics(result, h, validation['delta'], config.entropic_aversion),
                   'gradient_norm': sum(gradients) / max(len(gradients), 1), 'eta_normalized': float(eta.detach())}
        records.append(row)
        if score < best:
            best, best_epoch, best_state, best_eta = score, epoch, copy.deepcopy(model.state_dict()), float(eta.detach())
        print(f'c={config.cost_rate:g} {objective}/{spec.name} seed={seed} epoch={epoch}: risk={score:.4f}, '
              f'delta MAE={row["delta_mae"]:.4f}', flush=True)
    model.load_state_dict(best_state)
    change = sum(float((model.state_dict()[k] - v).abs().sum()) for k, v in first_state.items())
    spec_dict = asdict(spec)
    spec_dict['drop'] = list(spec.drop)
    metadata = {'schema': SCHEMA, 'schema_version': SCHEMA_VERSION, 'spec': spec_dict, 'family': spec.family,
                'market': asdict(config), 'fit': asdict(fit_config), 'objective': objective, 'seed': seed,
                'epoch_budget': epochs, 'best_epoch': best_epoch, 'validation_objective': best,
                'eta_normalized': best_eta, 'selected_parameter_l1_change': change,
                'parameter_count': parameter_count(model), 'features': list(FEATURES),
                'selection': 'Minimum validation objective, including epoch-zero prior fallback',
                'torch_version': str(torch.__version__)}
    if extra_metadata:
        metadata['experiment_context'] = extra_metadata
    file = output / f'{spec.name}_{objective}_{seed}.pt'
    torch.save({'state_dict': best_state, **metadata}, file)
    pd.DataFrame(records).to_csv(output / f'{spec.name}_{objective}_{seed}_curve.csv', index=False)
    return model, {**metadata, 'checkpoint': file.name}


def load_step_policy(file):
    """Load a Phase 6 checkpoint or a frozen Phase 4 checkpoint behind the step interface."""
    saved = torch.load(file, map_location='cpu', weights_only=True)
    if saved.get('features') != list(FEATURES):
        raise ValueError('Feature schema mismatch')
    if saved.get('schema') == SCHEMA:
        if saved.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('Unsupported Phase 6 checkpoint version')
        raw = saved['spec']
        spec = PolicySpec(**{**raw, 'drop': tuple(raw['drop'])})
        model = build(spec, saved['market']['max_position'])
        model.load_state_dict(saved['state_dict'])
        model.eval()
        return model, saved
    from .training import load
    legacy, saved = load(file)
    return LegacyStep(legacy), saved
