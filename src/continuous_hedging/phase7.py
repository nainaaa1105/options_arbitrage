"""Continuous -> closed-loop rounded -> matched categorical policy bridge."""
import argparse,json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import torch
from .config import ResearchConfig,FitConfig
from .market import simulate
from .training import load
from .lot_training import fit_lot,load_lot,rollout
from .rounding import rounded_policy,rounded_baseline
from .accounting import pnl
from .evaluation import metrics,paired_intervals
from .objectives import cvar
from .experiment_utils import begin,freeze,dump,sha,save_batch,infer,save_execution

BOOKS=(1,4,10,40)


def run(output,parent,settings):
    parent_frozen=json.loads((parent/'frozen_before_test.json').read_text())
    choice=parent_frozen['selections']['2_cvar_mlp'];teacher_file=parent/choice['path']
    if sha(teacher_file)!=parent_frozen['checkpoint_sha256'][choice['path']]:raise ValueError('Teacher checkpoint changed')
    if not json.loads((parent/'verification.json').read_text())['mvp_gates_pass']:raise ValueError('Phase 4 gate not passed')
    teacher,saved=load(teacher_file);c=ResearchConfig(**saved['market'])
    begin(output,{'phase':7,'books':BOOKS,'config':asdict(c),'settings':asdict(settings),'teacher':str(teacher_file.resolve()),
        'teacher_sha256':sha(teacher_file),'validation_seed':235001,'test_seed':285001,
        'comparison':'Same short call/put liability, premium, paths, costs, horizon and action bounds; per-option normalized metrics',
        'scope':'Synthetic underlying lots only, no NIFTY margin/basis/taxes claim; original long-option checkpoints are not misused',
        'rounding':'Closed-loop on executed previous inventory; half ties toward zero; no open-loop rounding shortcut',
        'attribution':'Rounding gap is paired; PG-vs-rounded gap also includes training method/architecture and imitation-budget differences'})
    validation=simulate(c,settings.validation_paths,seed=235001);selections={};baselines={}
    for book in BOOKS:
        folder=output/f'book_{book}';folder.mkdir();candidates=[]
        for seed in settings.seeds:candidates.append(fit_lot(c,settings,validation,teacher,book,seed,folder))
        selections[str(book)]={'selected':min(candidates,key=lambda r:(r['validation_cvar'],r['seed'])),'all_seeds':candidates}
        rows=[]
        for name,grid in (('constant_band',(0.,.01,.025,.05,.1,.2,.3)),('whalley_wilmott',(10.,50.,100.,200.,500.))):
            for value in grid:
                h=rounded_baseline(validation,c,book,name,width=value if name=='constant_band' else 0.,aversion=value if name=='whalley_wilmott' else None)
                rows.append({'baseline':name,'parameter':value,'validation_cvar':float(cvar(-pnl(validation,h,c)['net_pnl'],c.alpha))})
        table=pd.DataFrame(rows);table.to_csv(folder/'baseline_search.csv',index=False)
        baselines[str(book)]={name:float(g.sort_values(['validation_cvar','parameter']).iloc[0].parameter) for name,g in table.groupby('baseline')}
    freeze(output,selections,baselines=baselines,teacher_sha256=sha(teacher_file))
    batch=simulate(c,settings.test_paths,seed=285001);save_batch(output/'test_paths.npz',batch)
    continuous_h=infer(teacher,batch,c);continuous=pnl(batch,continuous_h,c)
    rows=[];paired=[];replays={}
    for book in BOOKS:
        folder=output/f'book_{book}'
        rows.append({'book':book,'strategy':'continuous_teacher','normalization':'per option',**metrics(continuous,continuous_h,batch['delta'])})
        with torch.no_grad():rounded_h=rounded_policy(teacher,batch,c,book)
        rounded=pnl(batch,rounded_h,c);save_execution(folder/'rounded_teacher_execution.npz',rounded,rounded_h)
        rows.append({'book':book,'strategy':'rounded_teacher',**metrics(rounded,rounded_h,batch['delta'])})
        paired.append({'book':book,'strategy':'rounded_teacher','reference':'continuous_teacher',**paired_intervals(rounded['net_pnl'].numpy(),continuous['net_pnl'].numpy())})
        for name in ('unhedged','delta','constant_band','whalley_wilmott'):
            h=rounded_baseline(batch,c,book,name,width=baselines[str(book)]['constant_band'],aversion=baselines[str(book)]['whalley_wilmott'])
            rows.append({'book':book,'strategy':'rounded_'+name,**metrics(pnl(batch,h,c),h,batch['delta'])})
        for seed in settings.seeds:
            model,_=load_lot(folder/f'policy_seed_{seed}.pt')
            with torch.no_grad():result,h,_,_=rollout(model,batch,c)
            selected=seed==selections[str(book)]['selected']['seed']
            rows.append({'book':book,'strategy':f'categorical_{seed}','selected':selected,**metrics(result,h,batch['delta'])})
            save_execution(folder/f'categorical_{seed}_execution.npz',result,h)
            reload,_=load_lot(folder/f'policy_seed_{seed}.pt')
            with torch.no_grad():again=rollout(reload,{k:v[:128] for k,v in batch.items()},c)[1]
            replays[f'{book}_{seed}']=float((again-h[:128]).abs().max())
            if selected:paired.append({'book':book,'strategy':f'categorical_{seed}','reference':'rounded_teacher',**paired_intervals(result['net_pnl'].numpy(),rounded['net_pnl'].numpy())})
            warm,_=load_lot(folder/f'warm_seed_{seed}.pt')
            with torch.no_grad():w,wh,_,_=rollout(warm,batch,c)
            rows.append({'book':book,'strategy':f'imitation_only_{seed}',**metrics(w,wh,batch['delta'])})
    pd.DataFrame(rows).to_csv(output/'rounding_comparison.csv',index=False);pd.DataFrame(paired).to_csv(output/'paired_comparisons.csv',index=False)
    if max(replays.values())>1e-6:raise AssertionError('Lot replay mismatch')
    dump(output/'verification.json',{'whole_lot_replay_errors':replays,'teacher_unchanged':sha(teacher_file)==parent_frozen['checkpoint_sha256'][choice['path']],
         'test_retuning':False,'unique_test_paths':settings.test_paths,'matched_liability_and_premium':True,
         'all_metrics_per_option':True,'nifty_real_execution_claim':False})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--phase4',type=Path,required=True)
    args=parser.parse_args();run(args.output,args.phase4,FitConfig())
