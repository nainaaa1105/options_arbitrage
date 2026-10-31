"""Standalone phase-2 validation figures and frozen-policy frequency diagnostics."""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'data/research/.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from continuous_hedging.config import ResearchConfig
from continuous_hedging.market import simulate,coarsen
from continuous_hedging.training import load
from continuous_hedging.policies import positions
from continuous_hedging.accounting import pnl
from continuous_hedging.baselines import baseline
from continuous_hedging.evaluation import metrics


def report(run,output):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    plan=json.loads((run/'experiment_plan.json').read_text())
    frozen=json.loads((run/'frozen_before_test.json').read_text())
    from research_registry import sha256
    for name,value in frozen['checkpoint_sha256'].items():
        if sha256(run/name)!=value:raise ValueError('Frozen checkpoint changed')
    c=ResearchConfig(**plan['markets']['frictionless'])
    source=np.load(run/'frictionless/test_paths.npz')
    batch={key:torch.from_numpy(source[key]) for key in source.files}
    analytic=pnl(batch,baseline(batch,c,'delta'),c)['net_pnl'].numpy()
    selections=frozen['selections']
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    models={}
    rng=np.random.default_rng(302)
    for i,init in enumerate(('flat','delta')):
        selected=selections['frictionless_cvar_'+init]['selected']
        model,_=load(run/'frictionless'/selected['checkpoint'])
        models[init]=model
        with torch.no_grad():h=positions(model,batch,c)
        expected=batch['delta'].numpy().ravel();actual=h.numpy().ravel()
        idx=rng.choice(len(actual),min(4000,len(actual)),replace=False)
        axes[i].scatter(expected[idx],actual[idx],s=3,alpha=.18)
        axes[i].plot([-1,1],[-1,1],color='black',linewidth=1)
        axes[i].set(xlabel='Analytic delta (underlying units)',ylabel='Learned hedge (underlying units)',
                    title=f'{init} initialization · test MAE {abs(actual-expected).mean():.4f}')
    fig.suptitle('Frictionless GBM numerical validation · short option to expiry')
    fig.savefig(output/'delta_recovery.png',dpi=160);fig.savefig(output/'delta_recovery.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    ax.hist(analytic,bins=100,density=True,histtype='step',label='Analytic delta')
    for init,model in models.items():
        with torch.no_grad():v=pnl(batch,positions(model,batch,c),c)['net_pnl'].numpy()
        ax.hist(v,bins=100,density=True,histtype='step',label=f'CVaR policy / {init}')
    ax.set(xlabel='Net P&L (normalized currency per short option)',ylabel='Density',title='Independent frictionless replication-error distributions')
    ax.legend();fig.savefig(output/'replication_errors.png',dpi=160);plt.close(fig)
    fine_config=replace(c,steps=256)
    fine=simulate(fine_config,plan['fit']['test_paths'],seed=plan['seeds']['frequency'])
    rows=[]
    for steps in (8,16,32,64,128,256):
        current=replace(c,steps=steps);b=coarsen(fine,steps)
        rows.append({'steps':steps,'strategy':'analytic_delta',**metrics(pnl(b,baseline(b,current,'delta'),current))})
        for init,model in models.items():
            outputs=[]
            with torch.no_grad():
                for start in range(0,len(b['spot']),512):
                    chunk={key:value[start:start+512] for key,value in b.items()}
                    outputs.append(positions(model,chunk,current))
            h=torch.cat(outputs)
            rows.append({'steps':steps,'strategy':'cvar_'+init,**metrics(pnl(b,h,current),h,b['delta'])})
    table=pd.DataFrame(rows);table.to_csv(output/'frozen_policy_frequency.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for name,group in table.groupby('strategy'):
        ax.plot(group.steps,group.pnl_std,marker='o',label=name)
    ax.set(xlabel='Rebalancing intervals to expiry',ylabel='P&L standard deviation',title='Matched GBM paths · frozen models, no frequency retuning',xscale='log',yscale='log')
    ax.legend();fig.savefig(output/'frequency_validation.png',dpi=160);plt.close(fig)
    (output/'scope.json').write_text(json.dumps({'models_refit':False,'frequency_paths_seed':plan['seeds']['frequency'],
        'interpretation':'Frozen finite-resolution models may retain approximation error as the analytic benchmark improves; no asymptotic optimality claim',
        'checkpoint_hashes_verified':True},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report(args.run_dir,args.output)
