"""Replay every saved extension execution and independently recheck its cash ledger."""
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from continuous_hedging.config import ResearchConfig
from continuous_hedging.training import load
from continuous_hedging.lot_training import load_lot,rollout
from continuous_hedging.rounding import rounded_policy
from continuous_hedging.market import coarsen
from continuous_hedging.accounting import pnl
from continuous_hedging.baselines import baseline
from continuous_hedging.evaluation import metrics
from continuous_hedging.experiment_utils import load_batch,infer,sha,dump


def verify(phase,run,output):
    if output.exists():raise FileExistsError('Preserving verification output')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    frozen=json.loads((run/'frozen_before_test.json').read_text())
    plan=json.loads((run/'experiment_plan.json').read_text())
    for name,digest in frozen['checkpoint_sha256'].items():
        if sha(run/name)!=digest:raise AssertionError(f'Changed weights: {name}')
    for name,digest in plan['source_sha256'].items():
        if sha(run/'source_snapshot'/name)!=digest:raise AssertionError(f'Changed as-run source: {name}')
    records=[];warm_headers=[];stress_rows_checked=0
    def compare(file,batch,c,h,book=None):
        result=pnl(batch,h,c)
        with np.load(file) as saved:
            errors={name:float(np.max(np.abs(saved[name].astype(float)-value.detach().numpy().astype(float))))
                    for name,value in {'positions':h,**result}.items()}
        # Ledger subtraction amplifies otherwise negligible float32 roundoff.
        if max(errors.values())>2e-6:raise AssertionError(f'Replay mismatch: {file}: {errors}')
        if book is not None and float((h*book-(h*book).round()).abs().max())>1e-5:
            raise AssertionError('Non-whole-lot action')
        records.append({'execution':str(file.relative_to(run)),'paths':len(h),
                        'max_error':max(errors.values()),'cash_error':float(result['reconciliation_error'].abs().max())})
    if phase=='phase4':
        for key,s in frozen['selections'].items():
            i=int(key.split('_')[0]);batch=load_batch(run/f'test_paths_cost_{i}.npz')
            folder=(run/s['path']).parent
            for candidate in s['all_seeds']:
                file=folder/candidate['checkpoint'];model,meta=load(file);c=ResearchConfig(**meta['market'])
                compare(file.with_name(file.stem+'_execution.npz'),batch,c,infer(model,batch,c))
    elif phase=='phase7':
        c=ResearchConfig(**plan['config']);batch=load_batch(run/'test_paths.npz')
        teacher_file=Path(plan['teacher'])
        if sha(teacher_file)!=plan['teacher_sha256']:raise AssertionError('Changed teacher')
        teacher,_=load(teacher_file)
        table=pd.read_csv(run/'rounding_comparison.csv')
        for book in plan['books']:
            folder=run/f'book_{book}'
            with torch.no_grad():h=rounded_policy(teacher,batch,c,book)
            compare(folder/'rounded_teacher_execution.npz',batch,c,h,book)
            for seed in plan['settings']['seeds']:
                model,_=load_lot(folder/f'policy_seed_{seed}.pt')
                with torch.no_grad():h=rollout(model,batch,c)[1]
                compare(folder/f'categorical_{seed}_execution.npz',batch,c,h,book)
                warm,metadata=load_lot(folder/f'warm_seed_{seed}.pt')
                with torch.no_grad():values=rollout(warm,batch,c)[0]['net_pnl']
                expected=table.loc[table.book.eq(book)&table.strategy.eq(f'imitation_only_{seed}'),'mean_net_pnl'].iloc[0]
                if abs(float(values.double().mean())-expected)>1e-8:raise AssertionError('Warm diagnostic mismatch')
                warm_headers.append({'book':book,'seed':seed,'epoch_zero_validation_cvar':metadata['validation_cvar'],
                    'original_header_validation_cvar':metadata.get('original_header_validation_cvar'),
                    'weights_unchanged':True})
    else:
        banks={world:load_batch(run/f'test_{world}.npz') for world in ('gbm','heston','bootstrap')}
        for case,s in frozen['selections'].items():
            c=ResearchConfig(**s['config']);folder=(run/s['path']).parent
            for candidate in s['all_seeds']:
                model,_=load(folder/candidate['checkpoint']);seed=candidate['seed']
                if case.startswith('world_'):
                    for world,b in banks.items():compare(folder/f'test_{world}_{seed}.npz',b,c,infer(model,b,c))
                if case=='world_gbm' or case.startswith('alpha_'):
                    b=banks['gbm'];compare(folder/f'alpha_execution_{seed}.npz',b,c,infer(model,b,c))
                if case in ('world_gbm','world_heston') or case.startswith('frequency_'):
                    b=coarsen(load_batch(run/f"frequency_test_{s['world']}_80.npz"),c.steps)
                    compare(folder/f'frequency_execution_{seed}.npz',b,c,infer(model,b,c))
        pools={name:pd.read_parquet(run/f'{name}_daily_returns.parquet') for name in ('train','validation','test')}
        if not pools['train'].date.max()<pools['validation'].date.min()<=pools['validation'].date.max()<pools['test'].date.min():
            raise AssertionError('Return library chronology failed')
        for world,b in banks.items():
            intrinsic=torch.where(b['call'],b['spot'][:,0]-1,1-b['spot'][:,0]).clamp_min(0)
            if (b['premium']<intrinsic-1e-7).any():raise AssertionError('Premium bounds failed')
        table=pd.read_csv(run/'robustness_comparison.csv')
        for name in ('gbm_high_vol','heston_shift','bootstrap_uncentered'):
            b=load_batch(run/f'stress_{name}.npz');c=ResearchConfig(**plan['config'])
            if name!='bootstrap_uncentered':
                c=replace(c,sigma_low=plan['stress_parameters']['sigma_low'],sigma_high=plan['stress_parameters']['sigma_high'])
            for world in ('gbm','heston','bootstrap','baseline'):
                if world=='baseline':h=baseline(b,c,'delta')
                else:
                    model,_=load(run/frozen['selections']['world_'+world]['path']);h=infer(model,b,c)
                actual=metrics(pnl(b,h,c))
                expected=table.loc[table.experiment.eq('stress')&table.test_world.eq(name)&table.training_world.eq(world)].iloc[0]
                for metric in ('mean_net_pnl','pnl_std','cvar95','cvar99','mean_cost'):
                    if abs(actual[metric]-expected[metric])>1e-7:raise AssertionError(f'Stress table mismatch: {name}/{world}/{metric}')
                stress_rows_checked+=1
    dump(output,{'phase':phase,'run':str(run.resolve()),'checkpoint_count':len(frozen['checkpoint_sha256']),
        'execution_count':len(records),'checkpoint_hashes_pass':True,'as_run_source_hashes_pass':True,
        'full_test_path_replay_pass':True,'max_replay_error':max(r['max_error'] for r in records),
        'max_cash_reconciliation_error':max(r['cash_error'] for r in records),'executions':records,
        'warm_metadata_reconciliation':warm_headers,'stress_metric_rows_replayed':stress_rows_checked})
    print(f'{phase}: {len(records)} full-bank executions replayed; all checks passed',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['phase4','phase7','phase5'])
    p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.phase,a.run_dir,a.output)
