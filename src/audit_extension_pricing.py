"""Independent Heston premium/terminal-payoff consistency and discretization audit."""
import argparse,json,math
from pathlib import Path
import numpy as np
import torch
from continuous_hedging.worlds import HestonPremium
from continuous_hedging.experiment_utils import dump


def audit(parameters,maturity,seed,pairs=8192):
    # Antithetic pairs, not individual dependent legs, define MC standard errors.
    states=[(s,v) for s in (.95,1.,1.05) for v in (parameters['sigma_low']**2,parameters['sigma_high']**2)]
    s0=torch.tensor([s for s,v in states],dtype=torch.float64)[:,None].expand(-1,pairs*2)
    v0=torch.tensor([v for s,v in states],dtype=torch.float64)[:,None].expand_as(s0)
    fine_s,coarse_s=s0.clone(),s0.clone();fine_v,coarse_v=v0.clone(),v0.clone()
    generator=torch.Generator().manual_seed(seed);dt=maturity/1280
    k,theta,xi,rho=(parameters[name] for name in ('kappa','theta','xi','rho'))
    def advance(s,v,z,interval):
        vp=v.clamp_min(0);zv=rho*z[0]+math.sqrt(1-rho*rho)*z[1]
        return s*torch.exp(-.5*vp*interval+torch.sqrt(vp*interval)*z[0]),v+k*(theta-vp)*interval+xi*torch.sqrt(vp*interval)*zv
    accumulated=torch.zeros(2,*s0.shape,dtype=torch.float64)
    for i in range(1280):
        z=torch.randn(2,len(states),pairs,generator=generator,dtype=torch.float64)
        z=torch.cat((z,-z),dim=2);accumulated+=z
        fine_s,fine_v=advance(fine_s,fine_v,z,dt)
        if i%4==3:
            coarse_s,coarse_v=advance(coarse_s,coarse_v,accumulated/2,dt*4);accumulated.zero_()
    def estimate(x):
        paired=(x[:,:pairs]+x[:,pairs:])/2
        return paired.mean(1).numpy(),(paired.std(1)/math.sqrt(pairs)).numpy()
    fine,se_fine=estimate((fine_s-1).clamp_min(0));coarse,se_coarse=estimate((coarse_s-1).clamp_min(0))
    gap,se_gap=estimate((coarse_s-1).clamp_min(0)-(fine_s-1).clamp_min(0))
    martingale,se_martingale=estimate(coarse_s-s0)
    oracle=HestonPremium(parameters,maturity,parameters['sigma_low'],parameters['sigma_high'])
    price=oracle.direct(np.array([s for s,v in states]),np.array([v for s,v in states]),True)
    records=[]
    for i,(s,v) in enumerate(states):
        # Predeclared diagnostic threshold includes sampling error. This is not
        # a parameter calibration or a selection of policy based on test P&L.
        passed=abs(fine[i]-price[i])<=5*se_fine[i]+2e-5 and abs(martingale[i])<=5*se_martingale[i]+2e-5
        if not passed:raise ArithmeticError(f'Pricing/simulation consistency failed at {(s,v)}')
        records.append({'spot':s,'variance':v,'fourier_premium':float(price[i]),
            'production_320_price':float(coarse[i]),'production_mc_se':float(se_coarse[i]),
            'fine_1280_price':float(fine[i]),'fine_mc_se':float(se_fine[i]),
            'paired_coarse_minus_fine':float(gap[i]),'paired_gap_mc_se':float(se_gap[i]),
            'spot_martingale_error':float(martingale[i]),'spot_martingale_mc_se':float(se_martingale[i]),'pass':True})
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserving numerical audit')
    torch.set_num_threads(2)
    plan=json.loads((args.run_dir/'experiment_plan.json').read_text());t=plan['config']['maturity']
    result={name:audit(plan[key],t,seed) for name,key,seed in (('base','calibration',951731),('stress','stress_parameters',951732))}
    dump(args.output,{'purpose':'Independent numerical correctness audit; no policy fitting or selection',
        'precision':'Float64 auditor, float32 production paths','antithetic_pairs_per_state':8192,
        'production_substeps':320,'fine_substeps':1280,'tolerance':'Fine price within 5 MC SE + 2e-5 of Fourier; spot martingale within 5 SE + 2e-5',
        'put_consistency':'Follows call audit plus tested zero-rate put-call parity and spot martingale',
        'results':result,'all_pass':True})
    print('Heston Fourier/payoff and discretization audit passed for base and stress parameters',flush=True)
