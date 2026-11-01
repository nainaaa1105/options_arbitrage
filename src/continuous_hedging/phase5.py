"""Training-only market proxies, held-out-library bootstrap and cross-world robustness."""
import argparse
from dataclasses import asdict,replace
from pathlib import Path
import numpy as np
import pandas as pd
from .config import ResearchConfig,FitConfig
from .worlds import daily_returns,split_returns,calibrate,HestonPremium,WorldSampler
from .market import coarsen
from .training import fit,load
from .baselines import baseline
from .accounting import pnl
from .objectives import cvar
from .evaluation import metrics,paired_intervals
from .run import select_baselines
from .experiment_utils import begin,freeze,dump,sha,save_batch,infer,save_execution

ALPHAS=(.5,.75,.9,.95,.99)


def audit_oracle(oracle,parameters):
    rng=np.random.default_rng(570)
    s=rng.uniform(.9,1.1,256);v=rng.uniform(parameters['sigma_low'],parameters['sigma_high'],256)**2;call=rng.random(256)>.5
    error=float(np.max(abs(oracle.marks(s,v,call)-oracle.direct(s,v,call))))
    if error>2e-4:raise ArithmeticError('Initial Heston premium interpolation exceeds audit tolerance')
    return {'states':256,'max_initial_price_error':error,'tolerance':.0002,'normalized_strike':1.}


def run(output,daily_file,settings):
    pools=split_returns(daily_returns(daily_file));parameters=calibrate(pools['train'])
    c=ResearchConfig(steps=20,maturity=20/252,cost_rate=.001,sigma_low=parameters['sigma_low'],sigma_high=parameters['sigma_high'])
    stress_p={**parameters,'sigma_low':parameters['sigma_low']*1.5,'sigma_high':parameters['sigma_high']*1.5,
              'theta':parameters['theta']*1.5,'kappa':parameters['kappa']*.5,'xi':parameters['xi']*1.5,'rho':max(-.9,parameters['rho']-.2)}
    begin(output,{'phase':5,'settings':asdict(settings),'config':asdict(c),'worlds':['gbm','heston','bootstrap'],
        'calibration':parameters,'daily_source':str(daily_file.resolve()),'daily_sha256':sha(daily_file),
        'return_pools':{name:{'rows':len(d),'start':str(d.date.min()),'end':str(d.date.max())} for name,d in pools.items()},
        'bootstrap_block_days':3,'bootstrap_prefix_days':20,'bootstrap_center':'Training log moment only, unchanged on validation/test libraries',
        'alphas':ALPHAS,'alpha_world':'GBM','frequency_steps':[5,20,80],'frequency_worlds':['gbm','heston'],
        'validation_seed':335001,'test_seed':385001,'frequency_test_seed':395001,'stress_parameters':stress_p,
        'information':'GBM constant sigma; Heston observed simulated variance (oracle assumption); bootstrap strictly past synthetic 20-day realized volatility',
        'premiums':'GBM BS; Heston audited Fourier initial premium assuming zero variance-risk premium; bootstrap common BS reference quote, not claimed arbitrage-free',
        'delta_benchmark':'BS proxy using current volatility in all worlds; not an assertion of Heston-optimal delta',
        'historical_status':'Retrospective source dates already inspected; bootstrap paths are synthetic, not independent real trading histories',
        'confidence':'Path-bootstrap intervals conditional on calibrated model/empirical library, exclude source-library estimation uncertainty'})
    oracle=HestonPremium(parameters,c.maturity,c.sigma_low,c.sigma_high);oracle.save(output/'heston_initial_premium_grid.npz')
    stress_oracle=HestonPremium(stress_p,c.maturity,stress_p['sigma_low'],stress_p['sigma_high']);stress_oracle.save(output/'stress_initial_premium_grid.npz')
    dump(output/'pricing_audits.json',{'base':audit_oracle(oracle,parameters),'stress':audit_oracle(stress_oracle,stress_p)})
    for name,d in pools.items():d.to_parquet(output/f'{name}_daily_returns.parquet',index=False)
    def factory(world,split):return WorldSampler(world,parameters,pools[split]['return'].to_numpy() if world=='bootstrap' else None,oracle=oracle)
    validations={world:factory(world,'validation')(c,settings.validation_paths,seed=335001+i*10000) for i,world in enumerate(('gbm','heston','bootstrap'))}
    cases=[]
    for world in ('gbm','heston','bootstrap'):cases.append((f'world_{world}',world,c))
    for alpha in ALPHAS:
        if alpha!=.95:cases.append((f'alpha_{alpha:g}','gbm',replace(c,alpha=alpha)))
    for world in ('gbm','heston'):
        for steps in (5,80):cases.append((f'frequency_{world}_{steps}',world,replace(c,steps=steps)))
    selections={};baseline_choices={}
    for case,world,config in cases:
        print(f'Phase 5 case {case}: {world}, steps={config.steps}, alpha={config.alpha}',flush=True)
        folder=output/case;folder.mkdir();sampler=factory(world,'train')
        validation=validations[world] if config.steps==20 else factory(world,'validation')(config,settings.validation_paths,seed=335001+('heston'==world)*10000)
        candidates=[]
        choice,search=select_baselines(validation,config,'cvar');search.to_csv(folder/'baseline_search.csv',index=False);baseline_choices[case]=choice
        for seed in settings.seeds:
            model,metadata=fit(config,settings,validation,'cvar','delta',seed,folder,sampler=sampler,
                extra_metadata={'world':world,'case':case,'calibration_training_end':parameters['training_end'],'bootstrap_library_split':'train' if world=='bootstrap' else None})
            candidates.append(metadata)
        winner=min(candidates,key=lambda row:(row['validation_objective'],row['seed']))
        selections[case]={'world':world,'config':asdict(config),'selected':winner,'all_seeds':candidates,'path':str((folder/winner['checkpoint']).relative_to(output))}
    freeze(output,selections,baselines=baseline_choices,calibration=parameters,daily_sha256=sha(daily_file))
    rows=[];paired=[];baselines=[];replay={};banks={}
    for i,world in enumerate(('gbm','heston','bootstrap')):
        batch=factory(world,'test')(c,settings.test_paths,seed=385001+i*10000);banks[world]=batch;save_batch(output/f'test_{world}.npz',batch)
        reference=pnl(batch,baseline(batch,c,'delta'),c)
        choice=baseline_choices['world_'+world]
        for name in ('unhedged','delta','constant_band','whalley_wilmott'):
            h=baseline(batch,c,name,width=choice['constant_band'],aversion=choice['whalley_wilmott'])
            baselines.append({'world':world,'strategy':name,'parameters_selected_on':'target-world validation',**metrics(pnl(batch,h,c),h,batch['delta'])})
        for training_world in ('gbm','heston','bootstrap'):
            case='world_'+training_world;selection=selections[case]
            source_choice=baseline_choices[case]
            h=baseline(batch,c,'constant_band',width=source_choice['constant_band'])
            rows.append({'experiment':'cross_world','training_world':training_world,'test_world':world,'strategy':'source_tuned_constant_band',**metrics(pnl(batch,h,c),h,batch['delta'])})
            for candidate in selection['all_seeds']:
                file=output/case/candidate['checkpoint'];model,_=load(file);h=infer(model,batch,c);result=pnl(batch,h,c)
                selected=candidate['seed']==selection['selected']['seed']
                rows.append({'experiment':'cross_world','training_world':training_world,'test_world':world,'strategy':f'neural_{candidate["seed"]}','selected':selected,**metrics(result,h,batch['delta'])})
                save_execution(output/case/f'test_{world}_{candidate["seed"]}.npz',result,h)
                if selected:paired.append({'experiment':'cross_world','training_world':training_world,'test_world':world,**paired_intervals(result['net_pnl'].numpy(),reference['net_pnl'].numpy())})
    for alpha in ALPHAS:
        case='world_gbm' if alpha==.95 else f'alpha_{alpha:g}';selection=selections[case];config=ResearchConfig(**selection['config'])
        batch=banks['gbm']
        for candidate in selection['all_seeds']:
            file=output/case/candidate['checkpoint'];model,_=load(file);h=infer(model,batch,config);result=pnl(batch,h,config)
            rows.append({'experiment':'alpha','training_world':'gbm','test_world':'gbm','alpha':alpha,'strategy':f'neural_{candidate["seed"]}',
                         'selected':candidate['seed']==selection['selected']['seed'],'objective_cvar':float(cvar(-result['net_pnl'],alpha)),**metrics(result,h,batch['delta'])})
            save_execution(output/case/f'alpha_execution_{candidate["seed"]}.npz',result,h)
    for i,world in enumerate(('gbm','heston')):
        fine_config=replace(c,steps=80);fine=factory(world,'test')(fine_config,settings.test_paths,seed=395001+i*10000)
        save_batch(output/f'frequency_test_{world}_80.npz',fine)
        for steps in (5,20,80):
            config=replace(c,steps=steps);batch=coarsen(fine,steps)
            case=f'world_{world}' if steps==20 else f'frequency_{world}_{steps}';selection=selections[case]
            ref=pnl(batch,baseline(batch,config,'delta'),config)
            rows.append({'experiment':'frequency','training_world':world,'test_world':world,'steps':steps,'strategy':'delta',**metrics(ref)})
            for candidate in selection['all_seeds']:
                file=output/case/candidate['checkpoint'];model,_=load(file);h=infer(model,batch,config);result=pnl(batch,h,config)
                rows.append({'experiment':'frequency','training_world':world,'test_world':world,'steps':steps,'strategy':f'neural_{candidate["seed"]}',
                             'selected':candidate['seed']==selection['selected']['seed'],**metrics(result,h,batch['delta'])})
                save_execution(output/case/f'frequency_execution_{candidate["seed"]}.npz',result,h)
    stresses={
        'gbm_high_vol':(WorldSampler('gbm',stress_p),replace(c,sigma_low=stress_p['sigma_low'],sigma_high=stress_p['sigma_high'])),
        'heston_shift':(WorldSampler('heston',stress_p,oracle=stress_oracle),replace(c,sigma_low=stress_p['sigma_low'],sigma_high=stress_p['sigma_high'])),
        'bootstrap_uncentered':(WorldSampler('bootstrap',parameters,pools['test']['return'].to_numpy(),recenter=False),c)}
    for i,(name,(sampler,config)) in enumerate(stresses.items()):
        batch=sampler(config,settings.test_paths,seed=415001+i*10000);save_batch(output/f'stress_{name}.npz',batch)
        for source_world in ('gbm','heston','bootstrap'):
            selection=selections['world_'+source_world];model,_=load(output/selection['path']);h=infer(model,batch,config);result=pnl(batch,h,config)
            rows.append({'experiment':'stress','training_world':source_world,'test_world':name,'strategy':'selected_neural',**metrics(result,h,batch['delta'])})
        h=baseline(batch,config,'delta');rows.append({'experiment':'stress','training_world':'baseline','test_world':name,'strategy':'delta',**metrics(pnl(batch,h,config))})
    for case,selection in selections.items():
        model,_=load(output/selection['path']);config=ResearchConfig(**selection['config']);world=selection['world']
        b=factory(world,'test')(config,128,seed=645001)
        first=infer(model,b,config);again,_=load(output/selection['path']);replay[case]=float((first-infer(again,b,config)).abs().max())
    pd.DataFrame(rows).to_csv(output/'robustness_comparison.csv',index=False);pd.DataFrame(baselines).to_csv(output/'target_world_baselines.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_comparisons.csv',index=False)
    if max(replay.values())>1e-6:raise AssertionError('World checkpoint replay failed')
    dump(output/'verification.json',{'replay_errors':replay,'case_count':len(cases),'trained_checkpoints':len(cases)*len(settings.seeds),
        'training_calibration_end':parameters['training_end'],'bootstrap_train_val_test_libraries_disjoint':True,
        'no_test_retuning':True,'nominal_test_paths_per_world':settings.test_paths,
        'empirical_test_source_days':len(pools['test']),'bootstrap_paths_are_not_independent_real_histories':True,
        'conditional_confidence_only':True,'continuous_time_or_real_world_optimality_claim':False})


def main():
    root=Path(__file__).resolve().parents[2]
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--daily',type=Path,default=root/'data/raw/upstox/nifty_daily_to_20261029.parquet')
    args=parser.parse_args();run(args.output,args.daily,FitConfig(batch=2048))
