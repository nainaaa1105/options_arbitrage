import torch
from torch import nn

FEATURES = ("standardized_moneyness","remaining_fraction","volatility_scaled","absolute_delta",
            "is_call","previous_magnitude","cost_scaled")


def features(batch, config, previous=None, step=None):
    if step is None:
        spot,t,delta = batch["spot"][:,:-1],batch["maturity"][:,:-1],batch["delta"]
        sigma,call = batch.get('volatility_path',batch['sigma'][:,None].expand_as(batch['spot']))[:,:-1],batch["call"][:,None]
        previous = torch.zeros_like(spot)
    else:
        spot,t,delta = batch["spot"][:,step],batch["maturity"][:,step],batch["delta"][:,step]
        sigma,call = (batch['volatility_path'][:,step] if 'volatility_path' in batch else batch['sigma']),batch["call"]
    m = torch.log(spot/config.strike)/(sigma*torch.sqrt(t.clamp_min(1e-12)))
    values = torch.broadcast_tensors(m.clamp(-8,8),t/config.maturity,sigma/.2,delta.abs(),call.float(),
                                    previous.abs()/config.max_position,torch.full_like(spot,config.cost_rate/.001))
    result=torch.stack(values,dim=-1)
    if not torch.isfinite(result).all():
        raise ValueError("Invalid continuous feature state")
    return result


class ContinuousPolicy(nn.Module):
    def __init__(self, initialization="flat", max_position=1.2, hidden=32, parameterization="projected"):
        super().__init__()
        if initialization not in ("flat","delta"):
            raise ValueError("Unknown initialization")
        if parameterization not in ('projected','sigmoid'):
            raise ValueError('Unknown continuous output parameterization')
        self.parameterization=parameterization
        self.initialization,self.max_position,self.hidden=initialization,max_position,hidden
        self.network=nn.Sequential(nn.Linear(len(FEATURES),hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self,x):
        if self.parameterization=='sigmoid':
            fraction=x[...,3]/self.max_position if self.initialization=='delta' else torch.full_like(x[...,3],.5/self.max_position)
            magnitude=self.max_position*torch.sigmoid(torch.logit(fraction.clamp(1e-6,1-1e-6))+self.network(x).squeeze(-1))
            return torch.where(x[...,4]>.5,magnitude,-magnitude)
        prior=x[...,3] if self.initialization=="delta" else torch.full_like(x[...,3],.5)
        # Projection gives an exact bounded continuous position without the
        # sigmoid's vanishing derivative near delta 0/1. No delta labels enter
        # training: the flat model learns only through its self-financing P&L.
        magnitude=(prior+self.network(x).squeeze(-1)).clamp(0,self.max_position)
        return torch.where(x[...,4]>.5,magnitude,-magnitude)


class BandPolicy(nn.Module):
    """Explicit no-trade structure; widths are learned, not a claimed MLP discovery."""
    requires_inventory=True
    parameterization='learned_band'
    initialization='delta'

    def __init__(self,max_position=1.2,hidden=32):
        super().__init__()
        self.max_position,self.hidden=max_position,hidden
        self.network=nn.Sequential(nn.Linear(len(FEATURES),hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,2))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        with torch.no_grad():self.network[-1].bias[1]=torch.log(torch.expm1(torch.tensor(.5)))

    def bounds(self,x):
        # Market-state band boundaries must not move with previous inventory.
        state=torch.cat([x[...,:5],torch.zeros_like(x[...,5:6]),x[...,6:]],dim=-1)
        raw=self.network(state)
        center=(x[...,3]+.1*torch.tanh(raw[...,0])).clamp(0,self.max_position)
        width=(.1*torch.nn.functional.softplus(raw[...,1])).clamp_max(.5)
        return (center-width).clamp_min(0),(center+width).clamp_max(self.max_position)

    def forward(self,x):
        lower,upper=self.bounds(x)
        previous=x[...,5]*self.max_position
        magnitude=torch.maximum(torch.minimum(previous,upper),lower)
        return torch.where(x[...,4]>.5,magnitude,-magnitude)


def positions(model,batch,config):
    if config.cost_rate==0 and config.fixed_cost==0 and config.half_spread==0 and not getattr(model,'requires_inventory',False):
        # In the frictionless Markov experiment inventory is irrelevant. This
        # vectorization also explicitly masks the inventory feature to zero.
        return model(features(batch,config))
    previous=torch.zeros(len(batch["spot"]))
    output=[]
    for i in range(config.steps):
        previous=model(features(batch,config,previous,i))
        output.append(previous)
    return torch.stack(output,dim=1)
