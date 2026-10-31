"""Supplemental, predeclared sensitivity evaluations; no fitting or policy selection."""
import argparse,json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from continuous_hedging.config import ResearchConfig
from continuous_hedging.training import load
from continuous_hedging.experiment_utils import load_batch,infer,dump,sha
from continuous_hedging.sensitivity import execution_stresses,observable_variance_proxy
from continuous_hedging.accounting import pnl
from continuous_hedging.rounding import rounded_baseline
from continuous_hedging.baselines import baseline
from continuous_hedging.evaluation import metrics


def bridge(run,output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    plan=json.loads((run/'experiment_plan.json').read_text());frozen=json.loads((run/'frozen_before_test.json').read_text())
    teacher_file=Path(plan['teacher'])
    if sha(teacher_file)!=plan['teacher_sha256']:raise ValueError('Teacher changed')
    c=ResearchConfig(**plan['config']);batch=load_batch(run/'test_paths.npz');teacher,_=load(teacher_file)
    continuous=infer(teacher,batch,c);records=[]
    for book in plan['books']:
        selected=frozen['selections'][str(book)]['selected']['seed'];folder=run/f'book_{book}'
        with np.load(folder/'rounded_teacher_execution.npz') as values:rounded=torch.from_numpy(values['positions'])
        with np.load(folder/f'categorical_{selected}_execution.npz') as values:categorical=torch.from_numpy(values['positions'])
        actions={'continuous_teacher':continuous,'rounded_teacher':rounded,'categorical':categorical,
                 'rounded_delta':rounded_baseline(batch,c,book,'delta')}
        for name,stress in execution_stresses(c,book).items():
            for strategy,h in actions.items():records.append({'book':book,'stress':name,'strategy':strategy,**metrics(pnl(batch,h,stress),h,batch['delta'])})
    pd.DataFrame(records).to_csv(output/'execution_sensitivity.csv',index=False)
    dump(output/'scope.json',{'frozen_actions':True,'model_refit':False,'fixed_fee_per_book_order':.00002,
         'per_option_fixed_fee':'.00002/book','spread_per_underlying_unit':.0001,
         'interpretation':'Execution-cost sensitivity with original actions held fixed; not a separately optimized strategy'})


def observable(run,output):
    output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    plan=json.loads((run/'experiment_plan.json').read_text());frozen=json.loads((run/'frozen_before_test.json').read_text())
    c=ResearchConfig(**plan['config']);batch=load_batch(run/'test_heston.npz');proxy=observable_variance_proxy(batch,c)
    records=[]
    for world in ('gbm','heston','bootstrap'):
        file=run/frozen['selections']['world_'+world]['path'];model,_=load(file)
        for info,b in (('latent_variance',batch),('trailing_realized_proxy',proxy)):
            h=infer(model,b,c);records.append({'training_world':world,'information':info,**metrics(pnl(b,h,c),h,b['delta'])})
    h=baseline(proxy,c,'delta');records.append({'training_world':'baseline','information':'trailing_realized_proxy',**metrics(pnl(proxy,h,c))})
    pd.DataFrame(records).to_csv(output/'volatility_observation_sensitivity.csv',index=False)
    dump(output/'scope.json',{'premium_and_payoff_unchanged':True,'state':'Training-range RMS volatility until five returns, then trailing <=20-return sample volatility',
        'latent_initial_variance_used':False,
        'sensitivity_source_sha256':sha(Path(__file__).resolve().parent/'continuous_hedging/sensitivity.py'),
        'future_returns_used_in_state':False,'model_refit':False,'interpretation':'Changing information regime, not a pure change of Heston dynamics'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['bridge','observable'])
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.phase](args.run_dir,args.output)
