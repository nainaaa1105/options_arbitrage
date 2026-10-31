"""Whole-lot categorical CVaR gradients on the same short liability as continuous learning."""
import copy
from pathlib import Path
from dataclasses import asdict
import pandas as pd
import torch
from torch import nn
from torch.distributions import Categorical
from .policies import features,FEATURES
from .market import simulate
from .accounting import pnl
from .objectives import cvar
from .rounding import rounded_policy,rounded_baseline


class LotPolicy(nn.Module):
    def __init__(self,book,max_position=1.2,hidden=32):
        super().__init__();self.book,self.max_position,self.hidden=book,max_position,hidden
        self.actions=int(book*max_position+1e-9)+1
        self.network=nn.Sequential(nn.Linear(len(FEATURES),hidden),nn.Tanh(),nn.Linear(hidden,hidden),nn.Tanh(),nn.Linear(hidden,self.actions))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)
        self.register_buffer('grid',torch.arange(self.actions,dtype=torch.float32))

    def forward(self,x):return self.network(x)-2*(self.grid-x[...,3:4]*self.book)**2


def rollout(model,batch,c,stochastic=False):
    previous=torch.zeros(len(batch['spot']));out=[];logp=torch.zeros_like(previous);entropy=torch.zeros_like(previous)
    for i in range(c.steps):
        dist=Categorical(logits=model(features(batch,c,previous,i)))
        action=dist.sample() if stochastic else dist.logits.argmax(-1)
        if stochastic:logp=logp+dist.log_prob(action);entropy=entropy+dist.entropy()
        previous=action.detach().float()/model.book*torch.where(batch['call'],1.,-1.)
        out.append(previous)
    h=torch.stack(out,dim=1)
    return pnl(batch,h,c),h,logp,entropy


def fit_lot(c,settings,validation,teacher,book,seed,folder):
    torch.manual_seed(seed);g=torch.Generator().manual_seed(9000+seed)
    model=LotPolicy(book,c.max_position);optimizer=torch.optim.Adam(model.parameters(),lr=.002)
    # Disclosed imitation warm start; no future path state enters teacher actions.
    for _ in range(24):
        b=simulate(c,settings.batch,generator=g)
        with torch.no_grad():target=rounded_policy(teacher,b,c,book)
        x=[];previous=torch.zeros(len(b['spot']))
        for i in range(c.steps):x.append(features(b,c,previous,i));previous=target[:,i]
        logits=model(torch.stack(x,dim=1));labels=(target.abs()*book).round().long()
        loss=torch.nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten())
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
    warm=copy.deepcopy(model.state_dict())
    initial=simulate(c,settings.batch,generator=g)
    with torch.no_grad():initial_pnl=rollout(model,initial,c,True)[0]['net_pnl']
    eta=nn.Parameter(torch.quantile(-initial_pnl/c.risk_scale,c.alpha))
    optimizer=torch.optim.Adam([{'params':model.parameters(),'lr':.0005},{'params':[eta],'lr':.005}])
    best=float('inf');best_state=None;best_epoch=0;records=[]
    for epoch in range(settings.epochs+1):
        gradients=[]
        if epoch:
            for _ in range(settings.updates):
                b=simulate(c,settings.batch,generator=g)
                result,h,logp,entropy=rollout(model,b,c,True)
                with torch.no_grad():reference=pnl(b,rounded_baseline(b,c,book,'delta'),c)['net_pnl']
                threshold=eta.detach();losses=-result['net_pnl']/c.risk_scale;ref=-reference/c.risk_scale
                advantage=(torch.relu(losses-threshold)-torch.relu(ref-threshold))/(1-c.alpha)
                objective=(advantage.detach()*logp).mean()-.0001*entropy.mean()+cvar(losses,c.alpha,eta)
                optimizer.zero_grad();objective.backward();gradient=torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
                if not torch.isfinite(gradient):raise ArithmeticError('Nonfinite categorical gradient')
                eta.grad.clamp_(-5,5);optimizer.step();gradients.append(float(gradient))
        with torch.no_grad():v=rollout(model,validation,c)[0]['net_pnl'];score=float(cvar(-v,c.alpha))
        records.append({'epoch':epoch,'validation_cvar':score,'mean_pnl':float(v.mean()),'gradient_norm':sum(gradients)/max(len(gradients),1)})
        if score<best:best,best_epoch,best_state=score,epoch,copy.deepcopy(model.state_dict())
        print(f'Lot book={book} seed={seed} epoch={epoch}: validation CVaR={score:.6f}',flush=True)
    metadata={'book':book,'max_position':c.max_position,'hidden':model.hidden,'features':list(FEATURES),
              'config':asdict(c),'settings':asdict(settings),'seed':seed,'validation_cvar':best,'best_epoch':best_epoch,
              'warm_updates':24,'training_budget_note':'Same risk-training path count as continuous; additional imitation warm start disclosed',
              'algorithm':'Categorical whole-lot score-function CVaR with learned eta and action-independent rounded-delta control variate'}
    torch.save({'state_dict':best_state,**metadata},folder/f'policy_seed_{seed}.pt')
    torch.save({'state_dict':warm,**metadata,'best_epoch':0,
                'validation_cvar':records[0]['validation_cvar'],'checkpoint_stage':'imitation_only'},folder/f'warm_seed_{seed}.pt')
    pd.DataFrame(records).to_csv(folder/f'curve_seed_{seed}.csv',index=False)
    return metadata


def load_lot(file):
    saved=torch.load(file,map_location='cpu',weights_only=True)
    # First bridge runs froze warm weights with the final policy's score in the
    # metadata header. Recover the actual epoch-zero score from its saved curve;
    # preserve checkpoint bytes and do not change validation model selection.
    file=Path(file)
    if file.name.startswith('warm_seed_') and 'checkpoint_stage' not in saved:
        curve=file.with_name(f"curve_seed_{saved['seed']}.csv")
        if not curve.exists():raise ValueError('Unversioned warm checkpoint requires its epoch-zero curve')
        score=float(pd.read_csv(curve).query('epoch == 0').iloc[0].validation_cvar)
        saved={**saved,'original_header_validation_cvar':saved['validation_cvar'],
               'validation_cvar':score,'checkpoint_stage':'imitation_only',
               'metadata_correction':'Recovered warm score from immutable epoch-zero curve; weights unchanged'}
    if saved['features']!=list(FEATURES):raise ValueError('Lot feature schema mismatch')
    model=LotPolicy(saved['book'],saved['max_position'],saved['hidden']);model.load_state_dict(saved['state_dict']);model.eval()
    return model,saved
