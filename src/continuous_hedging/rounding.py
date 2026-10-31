"""Matched-liability synthetic lot bridge; no Indian broker/margin claim."""
import torch
from .policies import features
from .baselines import ww_width


def round_units(h,book,max_position=1.2):
    if int(book)!=book or book<1:raise ValueError('Book size must be a positive integer')
    # Exact half-lot ties go toward zero.
    magnitude=torch.ceil(h.abs()*book-.5).clamp(0,int(max_position*book+1e-9))/book
    return h.sign()*magnitude


def rounded_policy(model,batch,c,book):
    previous=torch.zeros(len(batch['spot']));output=[]
    for i in range(c.steps):
        previous=round_units(model(features(batch,c,previous,i)),book,c.max_position)
        output.append(previous)
    return torch.stack(output,dim=1)


def rounded_baseline(batch,c,book,name,width=0.,aversion=None):
    if name not in ('unhedged','delta','constant_band','whalley_wilmott'):raise ValueError('Unknown rounded baseline')
    previous=torch.zeros(len(batch['spot']));out=[]
    widths=ww_width(batch['spot'][:,:-1],batch['gamma'],c.cost_rate,aversion or c.entropic_aversion) if name=='whalley_wilmott' else torch.full_like(batch['delta'],width)
    for i in range(c.steps):
        target=batch['delta'][:,i]
        if name=='unhedged':target=torch.zeros_like(previous)
        if name in ('constant_band','whalley_wilmott'):
            target=torch.maximum(torch.minimum(previous,target+widths[:,i]),target-widths[:,i])
        previous=round_units(target,book,c.max_position);out.append(previous)
    return torch.stack(out,dim=1)
