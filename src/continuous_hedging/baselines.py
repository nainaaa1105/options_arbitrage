"""Band baselines trade to the nearest band edge, never automatically to its center."""
import torch


def ww_width(spot,gamma,cost,aversion):
    if cost<0 or aversion<=0 or not torch.isfinite(spot).all() or not torch.isfinite(gamma).all():
        raise ValueError("Invalid Whalley-Wilmott inputs")
    # Small proportional-cost, zero-rate, exponential-utility approximation.
    return (1.5*cost*spot*gamma.square()/aversion).clamp_min(0).pow(1/3)


def baseline(batch,config,name,width=0.,aversion=None):
    if name not in ("unhedged","delta","constant_band","whalley_wilmott") or width<0:
        raise ValueError("Unknown baseline or invalid band")
    if name=="unhedged":return torch.zeros_like(batch["delta"])
    if name=="delta":return batch["delta"].clamp(-config.max_position,config.max_position)
    widths=torch.full_like(batch["delta"],width) if name=="constant_band" else ww_width(
        batch["spot"][:,:-1],batch["gamma"],config.cost_rate,aversion or config.entropic_aversion)
    previous=torch.zeros_like(batch["delta"][:,0])
    output=[]
    for i in range(config.steps):
        delta=batch["delta"][:,i]
        previous=torch.maximum(torch.minimum(previous,delta+widths[:,i]),delta-widths[:,i])
        previous=previous.clamp(-config.max_position,config.max_position)
        output.append(previous)
    return torch.stack(output,dim=1)
