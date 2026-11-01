import numpy as np
from .objectives import cvar, entropic


def metrics(result,positions=None,delta=None,aversion=100.):
    x=result["net_pnl"].detach().double()
    out={"paths":len(x),"mean_net_pnl":float(x.mean()),"pnl_std":float(x.std()),
         "cvar95":float(cvar(-x,.95)),"cvar99":float(cvar(-x,.99)),
         "entropic_risk":float(entropic(-x,aversion)),"worst_pnl":float(x.min()),
         "mean_cost":float(result["cost"].mean()),"mean_turnover_units":float(result["turnover_units"].mean()),
         "mean_turnover_notional":float(result["turnover_notional"].mean()),"mean_trades":float(result["trades"].float().mean()),
         "max_reconciliation_error":float(result["reconciliation_error"].abs().max())}
    if positions is not None and delta is not None:
        out["delta_mae"]=float((positions-delta).abs().mean())
        out["delta_rmse"]=float((positions-delta).square().mean().sqrt())
        out["last_step_delta_mae"]=float((positions[:,-1]-delta[:,-1]).abs().mean())
    return out


def paired_intervals(a,b,repeats=500,seed=891):
    a,b=np.asarray(a),np.asarray(b)
    rng=np.random.default_rng(seed)
    differences=[];tails=[]
    def tail(pnl):
        loss=-np.asarray(pnl,dtype=float)
        index=int(np.ceil(len(loss)*.95))-1
        threshold=np.partition(loss,index)[index]
        return threshold+np.maximum(loss-threshold,0).mean()/.05
    for _ in range(repeats):
        idx=rng.integers(0,len(a),len(a))
        differences.append(float((a[idx]-b[idx]).mean()))
        tails.append(float(tail(a[idx])-tail(b[idx])))
    return {"mean_pnl_difference":float((a-b).mean()),"lower95":float(np.quantile(differences,.025)),
            "upper95":float(np.quantile(differences,.975)),"cvar95_difference":float(tail(a)-tail(b)),
            "cvar95_lower95":float(np.quantile(tails,.025)),"cvar95_upper95":float(np.quantile(tails,.975))}
