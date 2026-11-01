"""Predeclared cost grid, exact/approximate hold-region diagnostics and MVP gate."""
import argparse
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import torch
from .config import ResearchConfig,FitConfig
from .market import simulate,bs
from .training import fit,load
from .policies import features
from .accounting import pnl
from .baselines import baseline
from .evaluation import metrics,paired_intervals
from .run import select_baselines
from .experiment_utils import begin,freeze,dump,save_batch,infer,save_execution

COSTS=(0.,.0002,.001,.005)


def hold_diagnostic(model,c):
    prev=torch.linspace(0,c.max_position,601)
    n=len(prev)
    s=torch.ones(n,2)*c.strike;t=torch.full((n,2),c.maturity/2);sigma=torch.full((n,),.2)
    _,d,g=bs(s,c.strike,t,sigma[:,None],torch.ones(n,1,dtype=torch.bool))
    b={'spot':s,'maturity':t,'sigma':sigma,'call':torch.ones(n,dtype=torch.bool),'delta':d[:,:1],'gamma':g[:,:1]}
    with torch.no_grad():
        x=features(b,c,prev,0);chosen=model(x)
        bounds=model.bounds(x) if hasattr(model,'bounds') else None
    tolerance=.005
    exact=(chosen-prev).abs()<=1e-6
    approximate=(chosen-prev).abs()<=tolerance
    def width(mask):
        indices=torch.where(mask)[0]
        return float(prev[indices[-1]]-prev[indices[0]]) if len(indices)>1 else 0.
    row={'cost':c.cost_rate,'epsilon':tolerance,'exact_hold_width':width(exact),'approximate_hold_width':width(approximate),
         'diagnostic':'Fixed S=K=1, sigma=.2, half remaining maturity, call; grid [0,1.2]',
         'structural_band_prior':bounds is not None}
    if bounds is not None:row.update(lower=float(bounds[0][0]),upper=float(bounds[1][0]),learned_width=float(bounds[1][0]-bounds[0][0]))
    return row,pd.DataFrame({'previous':prev.numpy(),'chosen':chosen.numpy(),'delta':float(d[0,0]),'exact_hold':exact.numpy(),'approximate_hold':approximate.numpy()})


def run(output,settings):
    begin(output,{'phase':4,'costs':COSTS,'settings':asdict(settings),'market':asdict(ResearchConfig()),
        'families':['mlp','band'],'objectives':['cvar','entropic'],'validation_seed':135001,'test_seed':185001,
        'hold_diagnostic_epsilon':.005,'claim_rule':'Band architecture imposes no-trade structure; MLP hold region is measured, never assumed',
        'paired_bootstrap':'IID synthetic paths conditional on the GBM specification; not real-market confidence',
        'mvp_gate':'Prior phases verified plus independent zero-cost MLP MAE<=.06 and std/delta<=1.20; no forced cost advantage'})
    selections={};base_choices={};diagnostics=[]
    for index,cost in enumerate(COSTS):
        c=ResearchConfig(cost_rate=cost);validation=simulate(c,settings.validation_paths,seed=135001)
        for objective in ('cvar','entropic'):
            choice,search=select_baselines(validation,c,objective);base_choices[f'{index}_{objective}']=choice
            folder=output/f'cost_{index}_{objective}';folder.mkdir();search.to_csv(folder/'baseline_search.csv',index=False)
            for family in ('mlp','band'):
                sub=folder/family;sub.mkdir();candidates=[]
                for seed in settings.seeds:
                    model,metadata=fit(c,settings,validation,objective,'delta',seed,sub,family=family)
                    candidates.append(metadata)
                winner=min(candidates,key=lambda r:(r['validation_objective'],r['seed']))
                selections[f'{index}_{objective}_{family}']={'selected':winner,'all_seeds':candidates,
                    'path':str((sub/winner['checkpoint']).relative_to(output))}
    freeze(output,selections,baselines=base_choices)
    rows=[];paired=[];gates=[];replay={}
    for index,cost in enumerate(COSTS):
        c=ResearchConfig(cost_rate=cost);batch=simulate(c,settings.test_paths,seed=185001)
        save_batch(output/f'test_paths_cost_{index}.npz',batch)
        ref=pnl(batch,baseline(batch,c,'delta'),c)
        for objective in ('cvar','entropic'):
            choice=base_choices[f'{index}_{objective}']
            for name in ('unhedged','delta','constant_band','whalley_wilmott'):
                h=baseline(batch,c,name,width=choice['constant_band'],aversion=choice['whalley_wilmott'])
                rows.append({'cost':cost,'objective':objective,'family':'baseline','strategy':name,**metrics(pnl(batch,h,c),h,batch['delta'])})
            h=baseline(batch,c,'whalley_wilmott',aversion=c.entropic_aversion)
            rows.append({'cost':cost,'objective':objective,'family':'baseline','strategy':'ww_utility_matched',**metrics(pnl(batch,h,c),h,batch['delta'])})
            for family in ('mlp','band'):
                selection=selections[f'{index}_{objective}_{family}']
                for candidate in selection['all_seeds']:
                    file=output/Path(selection['path']).parent/candidate['checkpoint'];model,_=load(file)
                    h=infer(model,batch,c);result=pnl(batch,h,c);m=metrics(result,h,batch['delta'])
                    selected=candidate['seed']==selection['selected']['seed']
                    rows.append({'cost':cost,'objective':objective,'family':family,'strategy':f'{family}_{candidate["seed"]}','selected':selected,**m})
                    save_execution(file.with_name(file.stem+'_execution.npz'),result,h)
                    small={k:v[:128] for k,v in batch.items()};again,_=load(file)
                    replay[str(file.relative_to(output))]=float((infer(again,small,c)-h[:128]).abs().max())
                    if selected:
                        paired.append({'cost':cost,'objective':objective,'family':family,'reference':'delta',**paired_intervals(result['net_pnl'].numpy(),ref['net_pnl'].numpy())})
                        diagnostic,curve=hold_diagnostic(model,c)
                        diagnostics.append({'objective':objective,'family':family,**diagnostic})
                        curve.to_csv(file.parent/'selected_position_response.csv',index=False)
                        if not cost and family=='mlp':gates.append({'objective':objective,'passed':m['delta_mae']<=.06 and m['pnl_std']/float(ref['net_pnl'].std())<=1.20})
    pd.DataFrame(rows).to_csv(output/'cost_comparison.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_comparisons.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'hold_regions.csv',index=False)
    if max(replay.values())>1e-6:raise AssertionError('Checkpoint replay mismatch')
    dump(output/'verification.json',{'zero_cost_gates':gates,'mvp_gates_pass':all(g['passed'] for g in gates),
        'replay_errors':replay,'test_retuning':False,'training_only_baseline_selection':False,
        'baseline_selection':'Validation only','unique_primary_test_paths':settings.test_paths,
        'same_paths_reused_across_cost_levels':True,'real_market_profitability_established':False})
    print('Phase 4 complete; MVP gates:',gates,flush=True)
    if not all(g['passed'] for g in gates):raise RuntimeError('MVP independent gate failed; preserve test result without retuning')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12);args=parser.parse_args()
    run(args.output,FitConfig(epochs=args.epochs))
