import math
import torch


def cvar(loss, alpha=.95, eta=None):
    if loss.ndim!=1 or not len(loss) or not torch.isfinite(loss).all() or not 0<alpha<1:
        raise ValueError("Finite losses and alpha in (0,1) required")
    if eta is None:
        # An exact empirical RU minimizer, including fractional tail mass.
        eta = loss.detach().sort().values[max(0,math.ceil(len(loss)*alpha)-1)]
    return eta+torch.relu(loss-eta).mean()/(1-alpha)


def entropic(loss, aversion):
    if loss.ndim!=1 or not len(loss) or aversion<=0 or not torch.isfinite(loss).all():
        raise ValueError("Invalid entropic inputs")
    return (torch.logsumexp(aversion*loss,dim=0)-math.log(len(loss)))/aversion


def risk(pnl, config, objective, eta=None):
    if objective=="cvar":
        return cvar(-pnl/config.risk_scale,config.alpha,eta)
    if objective=="entropic":
        return entropic(-pnl,config.entropic_aversion)/config.risk_scale
    raise ValueError("Unknown risk objective")
