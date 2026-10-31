"""CVaR objectives and score-function learning for discrete costly hedges."""

import torch


def cvar_objective(loss, alpha=.95, eta=None):
    if loss.ndim != 1 or len(loss) == 0 or not 0 < alpha < 1:
        raise ValueError("Need a nonempty loss vector and alpha in (0,1)")
    if eta is None:
        eta = torch.quantile(loss.detach(), alpha, interpolation="higher")
    return eta+torch.relu(loss-eta).mean()/(1-alpha)


def score_function_loss(pnl, log_probability, entropy, risk_scale, alpha=.95, entropy_weight=.001,
                        eta=None, reference_pnl=None):
    loss = -torch.as_tensor(pnl, dtype=torch.float32)/risk_scale
    eta = torch.quantile(loss.detach(), alpha, interpolation="higher") if eta is None else eta.detach()
    tail_cost = eta+torch.relu(loss-eta)/(1-alpha)
    if len(loss) < 2:
        raise ValueError("Policy-gradient minibatches require at least two paths")
    # Leave-one-out baseline is independent of this path's sampled actions.
    if reference_pnl is None:
        baseline = (tail_cost.sum()-tail_cost)/(len(tail_cost)-1)
    else:
        # This path-specific hedge is independent of sampled policy actions.
        # It is a training control variate, never an actor observation.
        reference_loss = -torch.as_tensor(reference_pnl, dtype=torch.float32)/risk_scale
        baseline = eta+torch.relu(reference_loss-eta)/(1-alpha)
    advantage = (tail_cost-baseline).detach()
    objective = (advantage*log_probability).mean()-entropy_weight*entropy.mean()
    return objective, cvar_objective(loss, alpha).detach()
