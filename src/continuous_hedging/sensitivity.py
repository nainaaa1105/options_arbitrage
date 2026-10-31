"""Predeclared frozen-action execution stresses and causal volatility-observation stress."""
from dataclasses import replace
import torch
from .market import bs


def observable_variance_proxy(batch,config,min_history=5):
    if min_history<2:raise ValueError('Need at least two returns for sample volatility')
    result={k:v.clone() for k,v in batch.items()}
    returns=torch.diff(torch.log(batch['spot']),dim=1)
    vols=[]
    # Initialization must not reveal simulated latent variance. Use only the
    # fixed training-calibrated volatility range until past returns are enough.
    initial=((config.sigma_low**2+config.sigma_high**2)/2)**.5
    for i in range(config.steps+1):
        if i<min_history:vols.append(torch.full_like(batch['sigma'],initial))
        else:vols.append(returns[:,max(0,i-20):i].std(1)*((config.steps/config.maturity)**.5))
    result['volatility_path']=torch.stack(vols,dim=1).clamp_min(.01)
    result['sigma']=result['volatility_path'][:,0]
    _,delta,gamma=bs(batch['spot'],config.strike,batch['maturity'],result['volatility_path'],batch['call'][:,None])
    result['delta'],result['gamma']=delta[:,:-1],gamma[:,:-1]
    return result


def execution_stresses(config,book):
    return {
        'proportional_x2':replace(config,cost_rate=config.cost_rate*2),
        'proportional_x5':replace(config,cost_rate=config.cost_rate*5),
        'spread_and_fixed_order':replace(config,half_spread=.0001,fixed_cost=.00002/book)}
