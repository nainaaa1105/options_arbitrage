"""Phases 1–3: validate continuous learning and cost-aware baseline foundations."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib,json,shutil
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from .config import ResearchConfig,FitConfig
from .market import simulate,coarsen
from .accounting import pnl
from .baselines import baseline
from .policies import positions
from .objectives import risk
from .training import fit,load
from .evaluation import metrics,paired_intervals


def digest(file):return hashlib.sha256(Path(file).read_bytes()).hexdigest()


def select_baselines(batch,config,objective):
    rows=[]
    for name,grid in (('constant_band',(0.,.01,.025,.05,.1,.2,.3)),('whalley_wilmott',(10.,50.,100.,200.,500.))):
        for parameter in grid:
            h=baseline(batch,config,name,width=parameter if name=='constant_band' else 0.,aversion=parameter if name=='whalley_wilmott' else None)
            score=float(risk(pnl(batch,h,config)['net_pnl'],config,objective))
            rows.append({'baseline':name,'parameter':parameter,'objective':objective,'validation_score':score})
    table=pd.DataFrame(rows)
    return {name:float(group.sort_values(['validation_score','parameter']).iloc[0].parameter) for name,group in table.groupby('baseline')},table


def run(output,fit_config):
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    shutil.copytree(Path(__file__).parent,output/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    configs={'frictionless':ResearchConfig(),'proportional_cost':ResearchConfig(cost_rate=.001)}
    plan={'created_at':datetime.now(timezone.utc).isoformat(),'fit':asdict(fit_config),'markets':{k:asdict(v) for k,v in configs.items()},
          'experiments':'Phase 2 frictionless validation; Phase 3 one positive-cost foundational comparison, not Phase 4 cost sweep',
          'initializations':{'frictionless':['flat','delta'],'proportional_cost':['delta']},'objectives':['cvar','entropic'],
          'seeds':{'validation':35001,'test':85001,'frequency':95001},
          'gates':{'selected_frictionless_delta_mae_max':fit_config.delta_mae_tolerance,'selected_frictionless_std_over_delta_max':fit_config.std_ratio_tolerance,
                   'frequency':'Fine-grid analytic delta P&L std below coarse-grid std on common zero-cost GBM paths'},
          'pricing':'Zero rates and drift, short one European option through expiry; same BS premium per path for every hedger',
          'limitations':'Stylized synthetic underlying units; no NIFTY lots, margin, taxes or calibrated real-market claim',
          'source_sha256':{p.name:digest(p) for p in (output/'source_snapshot').glob('*.py')}}
    (output/'experiment_plan.json').write_text(json.dumps(plan,indent=2))
    models,selections,baseline_choices,gate_rows={},{},{},[]
    for world,config in configs.items():
        folder=output/world;folder.mkdir()
        validation=simulate(config,fit_config.validation_paths,seed=35001)
        delta_std=float(pnl(validation,baseline(validation,config,'delta'),config)['net_pnl'].std())
        for objective in ('cvar','entropic'):
            choice,table=select_baselines(validation,config,objective)
            baseline_choices[world+'_'+objective]=choice
            table.to_csv(folder/f'{objective}_baseline_search.csv',index=False)
            for initialization in (('flat','delta') if world=='frictionless' else ('delta',)):
                candidates=[]
                for seed in fit_config.seeds:
                    budget=replace(fit_config,epochs=fit_config.flat_epochs) if initialization=='flat' else fit_config
                    model,metadata=fit(config,budget,validation,objective,initialization,seed,folder)
                    models[(world,objective,initialization,seed)]=model;candidates.append(metadata)
                selected=min(candidates,key=lambda row:(row['validation_objective'],row['seed']))
                selections[world+'_'+objective+'_'+initialization]={'selected':selected,'all_seeds':candidates}
                if world=='frictionless':
                    model=models[(world,objective,initialization,selected['seed'])]
                    with torch.no_grad():
                        h=positions(model,validation,config);m=metrics(pnl(validation,h,config),h,validation['delta'])
                    passed=m['delta_mae']<=fit_config.delta_mae_tolerance and m['pnl_std']/delta_std<=fit_config.std_ratio_tolerance
                    gate_rows.append({'objective':objective,'initialization':initialization,'delta_mae':m['delta_mae'],
                                      'std_over_delta':m['pnl_std']/delta_std,'passed':passed})
    pd.DataFrame(gate_rows).to_csv(output/'validation_gates.csv',index=False)
    if not all(row['passed'] for row in gate_rows):
        raise RuntimeError('Validation gate failed; independent test banks were NOT generated. Inspect validation_gates.csv.')
    frozen={'frozen_at':datetime.now(timezone.utc).isoformat(),'selections':selections,'baselines':baseline_choices,
            'checkpoint_sha256':{str(p.relative_to(output)):digest(p) for p in output.glob('*/*.pt')},'test_generated_before_freeze':False}
    (output/'frozen_before_test.json').write_text(json.dumps(frozen,indent=2))
    records,paired,gates=[],[],[]
    for world,config in configs.items():
        folder=output/world
        batch=simulate(config,fit_config.test_paths,seed=85001)
        np.savez_compressed(folder/'test_paths.npz',**{k:v.numpy() for k,v in batch.items()})
        delta_result=pnl(batch,baseline(batch,config,'delta'),config)
        for objective in ('cvar','entropic'):
            choice=baseline_choices[world+'_'+objective]
            for name in ('unhedged','delta','constant_band','whalley_wilmott'):
                h=baseline(batch,config,name,width=choice['constant_band'],aversion=choice['whalley_wilmott'])
                result=pnl(batch,h,config)
                records.append({'world':world,'objective':objective,'strategy':name,**metrics(result,h,batch['delta'],config.entropic_aversion)})
            h=baseline(batch,config,'whalley_wilmott',aversion=config.entropic_aversion)
            records.append({'world':world,'objective':objective,'strategy':'whalley_wilmott_utility_matched',**metrics(pnl(batch,h,config),h,batch['delta'],config.entropic_aversion)})
            for initialization in (('flat','delta') if world=='frictionless' else ('delta',)):
                selection=selections[world+'_'+objective+'_'+initialization]['selected']
                for seed in fit_config.seeds:
                    model,saved=load(folder/f'{objective}_{initialization}_{seed}.pt')
                    with torch.no_grad():h=positions(model,batch,config)
                    result=pnl(batch,h,config);m=metrics(result,h,batch['delta'],config.entropic_aversion)
                    selected=seed==selection['seed']
                    records.append({'world':world,'objective':objective,'strategy':f'neural_{initialization}_{seed}','selected':selected,**m})
                    np.savez_compressed(folder/f'{objective}_{initialization}_{seed}_execution.npz',positions=h.numpy(),**{k:v.detach().numpy() for k,v in result.items()})
                    if selected:
                        paired.append({'world':world,'objective':objective,'initialization':initialization,**paired_intervals(result['net_pnl'].detach().numpy(),delta_result['net_pnl'].numpy())})
                        if world=='frictionless':gates.append({'gate':objective+'_'+initialization,'passed':m['delta_mae']<=fit_config.delta_mae_tolerance and m['pnl_std']/float(delta_result['net_pnl'].std())<=fit_config.std_ratio_tolerance,
                                                             'delta_mae':m['delta_mae'],'std_over_delta':m['pnl_std']/float(delta_result['net_pnl'].std())})
    frequency=[]
    fine_config=replace(configs['frictionless'],steps=256)
    fine=simulate(fine_config,fit_config.test_paths,seed=95001)
    for n in (8,16,32,64,128,256):
        config=replace(fine_config,steps=n);batch=coarsen(fine,n)
        h=baseline(batch,config,'delta');m=metrics(pnl(batch,h,config))
        frequency.append({'steps':n,**m})
    gates.append({'gate':'frequency_std_declines_coarse_to_fine','passed':frequency[-1]['pnl_std']<frequency[0]['pnl_std']})
    pd.DataFrame(frequency).to_csv(output/'frequency_sanity.csv',index=False)
    pd.DataFrame(records).to_csv(output/'test_comparison.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_comparisons.csv',index=False)
    final={'test_gates':gates,'all_test_gates_pass':all(row['passed'] for row in gates),'no_test_retuning':True,
           'phase4_cost_grid_not_run':True,'units':'Normalized cash currency with strike 1; per one short option, not INR returns'}
    (output/'verification.json').write_text(json.dumps(final,indent=2))
    report=['# Continuous hedging phases 1–3','',final['units'],'','Validation-selected models and baselines; independent test banks.','',
            '| World | Objective | Strategy | Mean P&L | Std | CVaR95 | CVaR99 | Delta MAE |','|---|---|---|---:|---:|---:|---:|---:|']
    for r in records:
        if r.get('selected') or r['strategy'] in ('delta','constant_band','whalley_wilmott','whalley_wilmott_utility_matched'):
            report.append(f"| {r['world']} | {r['objective']} | {r['strategy']} | {r['mean_net_pnl']:.6f} | {r['pnl_std']:.6f} | {r['cvar95']:.6f} | {r['cvar99']:.6f} | {r.get('delta_mae',0):.5f} |")
    report+=['',f"All predeclared test gates passed: {final['all_test_gates_pass']}",'','Whalley–Wilmott is a small-cost exponential-utility approximation, not an asserted CVaR optimum.',
             'The non-prior model still observes BS delta as an input; recovery is a numerical validation, not an independent discovery of BS theory.',
             'These experiments do not establish a cost-grid/no-trade-band result, recurrent policy, or real-market advantage.']
    (output/'RESULTS.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(json.dumps(final,indent=2),flush=True)
    if not final['all_test_gates_pass']:raise RuntimeError('Independent test gate failed. Preserve results; do not retune against this bank.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=12)
    parser.add_argument('--flat-epochs',type=int,default=48)
    parser.add_argument('--updates',type=int,default=24)
    parser.add_argument('--batch',type=int,default=1024)
    args=parser.parse_args()
    run(args.output,FitConfig(epochs=args.epochs,flat_epochs=args.flat_epochs,updates=args.updates,batch=args.batch))
