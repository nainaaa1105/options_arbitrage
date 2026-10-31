"""Reproduce Phase 4/7/5 figures and central research reports from frozen outputs."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'data/research/.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator,NullLocator,ScalarFormatter
import numpy as np
import pandas as pd
import torch
from continuous_hedging.config import ResearchConfig
from continuous_hedging.experiment_utils import load_batch,sha,dump
from continuous_hedging.accounting import pnl
from continuous_hedging.baselines import baseline,ww_width
from continuous_hedging.market import bs


def save(fig,folder,name):
    fig.savefig(folder/(name+'.png'),dpi=160);fig.savefig(folder/(name+'.pdf'));plt.close(fig)


def checked_manifest(run):
    frozen=json.loads((run/'frozen_before_test.json').read_text())
    for name,value in frozen['checkpoint_sha256'].items():
        if sha(run/name)!=value:raise ValueError(f'Checkpoint changed: {name}')
    return frozen


def selected(frame):return frame.loc[frame['selected'].eq(True)] if 'selected' in frame else frame


def markdown_table(frame,columns):
    lines=['| '+' | '.join(columns)+' |','|'+'|'.join(['---']*len(columns))+'|']
    for _,r in frame.iterrows():
        values=[f'{r[c]:.6g}' if isinstance(r[c],(float,np.floating)) else str(r[c]) for c in columns]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def phase4_report(run,output):
    output.mkdir(parents=True,exist_ok=False);frozen=checked_manifest(run)
    frame=pd.read_csv(run/'cost_comparison.csv');holds=pd.read_csv(run/'hold_regions.csv');paired=pd.read_csv(run/'paired_comparisons.csv')
    costs=sorted(frame.cost.unique());chosen=frame.loc[frame.family.eq('baseline')|frame['selected'].eq(True)].copy()
    chosen['label']=np.where(chosen.family.eq('baseline'),chosen.strategy,chosen.family)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    metrics=['cvar95','cvar99','mean_net_pnl','pnl_std','mean_cost','mean_turnover_notional']
    for ax,metric in zip(axes.flat,metrics):
        for label,g in chosen.loc[chosen.objective.eq('cvar')&chosen.label.ne('unhedged')].groupby('label'):
            g=g.sort_values('cost');ax.plot(g.cost,g[metric],marker='o',label=label)
        ax.set(xlabel='Proportional rate per traded notional',ylabel=metric)
    axes[0,0].legend(fontsize=8);fig.suptitle('Independent GBM cost grid · normalized per-option P&L and costs')
    save(fig,output,'cost_risk_tradeoff')
    fig,axes=plt.subplots(2,4,figsize=(17,8),layout='constrained')
    for row,family in enumerate(('mlp','band')):
        for index,cost in enumerate(costs):
            ax=axes[row,index];selection=frozen['selections'][f'{index}_cvar_{family}']
            file=run/selection['path'];curve=pd.read_csv(file.parent/'selected_position_response.csv')
            c=ResearchConfig(cost_rate=cost);delta=float(curve.delta.iloc[0]);_,_,gamma=bs(torch.tensor([1.]),1.,c.maturity/2,.2,True)
            a=frozen['baselines'][f'{index}_cvar']['whalley_wilmott'];width=float(ww_width(torch.tensor([1.]),gamma,cost,a))
            ww=np.clip(curve.previous.to_numpy(),delta-width,delta+width)
            ax.plot(curve.previous,curve.chosen,label='Learned policy');ax.plot(curve.previous,ww,'--',label=f'WW (a={a:g})')
            ax.plot(curve.previous,curve.previous,':',color='gray',label='Hold');ax.axhline(delta,color='black',linewidth=.6,label='BS delta')
            ax.set(xlabel='Previous hedge',ylabel='Next hedge',title=f'{family}, c={cost:g}')
    axes[0,0].legend(fontsize=8);fig.suptitle('Fixed market state · band structure is explicit for band family; MLP hold behavior is measured')
    save(fig,output,'position_vs_previous')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for index,cost in enumerate(costs):
        batch=load_batch(run/f'test_paths_cost_{index}.npz');c=ResearchConfig(cost_rate=cost);ax=axes.flat[index]
        values={'delta':pnl(batch,baseline(batch,c,'delta'),c)['net_pnl'].numpy()}
        for family in ('mlp','band'):
            selection=frozen['selections'][f'{index}_cvar_{family}'];file=run/selection['path']
            with np.load(file.with_name(file.stem+'_execution.npz')) as data:values[family]=data['net_pnl']
        bins=np.linspace(min(v.min() for v in values.values()),max(v.max() for v in values.values()),100)
        for label,v in values.items():ax.hist(v,bins=bins,density=True,histtype='step',label=label)
        ax.set(xlabel='Net P&L per option',ylabel='Density',title=f'c={cost:g}');ax.legend(fontsize=8)
    save(fig,output,'pnl_distributions')
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for i,objective in enumerate(('cvar','entropic')):
        for j,family in enumerate(('mlp','band')):
            ax=axes[i,j]
            for index,cost in enumerate(costs):
                selection=frozen['selections'][f'{index}_{objective}_{family}'];file=run/selection['path']
                curve=pd.read_csv(file.with_name(file.stem+'_curve.csv'))
                ax.plot(curve.epoch,curve.validation_objective,label=f'c={cost:g}')
            ax.set(xlabel='Epoch',ylabel='Validation risk / scale',title=f'{objective} / {family}');ax.legend(fontsize=8)
    save(fig,output,'learning_curves')
    seed_stats=frame.loc[frame.family.ne('baseline')].groupby(['cost','objective','family']).agg(
        seeds=('strategy','size'),mean_cvar95=('cvar95','mean'),seed_std_cvar95=('cvar95','std'),mean_pnl=('mean_net_pnl','mean')).reset_index()
    seed_stats.to_csv(output/'seed_summary.csv',index=False)
    lines=['# Minimum viable continuous deep-hedging research project','',
        'Phases 0a–4: accounting and frictionless validation foundations, followed by a frozen cost experiment. This report is synthetic research, not market returns.','',
        '## Experiment contract','',
        'Short one European call/put to expiry; zero drift/rates; normalized strike 1; 32 trading intervals over 20/252 years; initial spot 0.9–1.1 and sigma 0.15–0.30. Premium and terminal liability are identical for all strategies on each path. Opening and final hedge liquidation costs are included.','',
        'Four proportional rates: 0, 0.0002, 0.001, 0.005. MLP and explicit learned-band policies use CVaR95 and entropic objectives, three seeds each. Adam training uses fresh paths. Baseline and checkpoint choices use validation only; all weights were frozen before generating test simulations.','',
        'There are 16,384 unique primary test paths, reused across cost cases for pairing; they are not 65,536 independent market histories. All seed results, objective-specific baseline searches, hashes and curves are retained. Paired path-bootstrap intervals condition on this GBM specification.','',
        '## Central results','',markdown_table(chosen.loc[chosen.objective.eq('cvar')],['cost','label','mean_net_pnl','pnl_std','cvar95','cvar99','mean_cost','mean_turnover_notional']),
        '', '![Risk/cost tradeoffs](cost_risk_tradeoff.png)','',
        '## Inventory response and no-trade interpretation','',
        'The band family explicitly projects inventory into learned market-state bounds. Its no-trade structure is an architectural assumption, not proof that an unconstrained MLP discovered a band. The MLP is evaluated both for exact holds (1e-6 tolerance) and an approximate operational hold region (0.005 underlying units). Width growth is measured and may fail monotonicity; it is not an acceptance criterion. The diagnostic uses S=K=1, sigma=0.2, half remaining maturity and a call. WW curves use validation-selected coefficients and remain a small-cost exponential-utility benchmark rather than a universal CVaR optimum.','',
        markdown_table(holds.loc[holds.objective.eq('cvar')],['cost','family','exact_hold_width','approximate_hold_width']),
        '', '![Position response](position_vs_previous.png)','',
        '## P&L distributions and training','', '![Distributions](pnl_distributions.png)','', '![Curves](learning_curves.png)','',
        '## Uncertainty and interpretation','',markdown_table(paired.loc[paired.objective.eq('cvar')],['cost','family','mean_pnl_difference','lower95','upper95','cvar95_difference','cvar95_lower95','cvar95_upper95']),
        '', 'Lower loss CVaR is better, higher mean P&L is better. Objectives can trade lower tail losses against larger overall dispersion or costs. Neither a favorable point estimate nor a monotone width chart proves a real-market advantage. Seed variation is saved separately from path-bootstrap uncertainty.','',
        '## MVP gate and limits','',f"Independent zero-cost MLP gates: {json.loads((run/'verification.json').read_text())['zero_cost_gates']}",
        '', 'Prior phases independently verified cash reconciliation, future-perturbation causality, gradients, pricing and frictionless numerical tolerances. Finite-grid CVaR policy deviations from analytic delta are possible; the models do not claim exact continuous-time optimality. No Sharpe ratio is manufactured from arbitrarily ordered Monte Carlo paths. Historical tests, operational issues and prospective validation retain their existing limitations.','',
        '## Reproduction','', '`python src/run_phase4_research.py --output NEW_DIRECTORY`',
        '', '`python src/report_extended_research.py phase4 --run-dir RUN --output NEW_REPORT_DIRECTORY`','',
        'Inspect experiment_plan.json, frozen_before_test.json, cost_comparison.csv, paired_comparisons.csv, hold_regions.csv and verification.json in the experiment directory.']
    (output/'MVP_RESEARCH_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    dump(output/'report_provenance.json',{'checkpoint_hashes_revalidated':True,'model_selection_performed':False,'run':str(run.resolve()),'frozen_manifest_sha256':sha(run/'frozen_before_test.json')})


def phase7_report(run,output):
    output.mkdir(parents=True,exist_ok=False);checked_manifest(run)
    frame=pd.read_csv(run/'rounding_comparison.csv');paired=pd.read_csv(run/'paired_comparisons.csv')
    seeds=frame.loc[frame.strategy.str.startswith(('categorical_','imitation_only_'))].copy()
    seeds['stage']=np.where(seeds.strategy.str.startswith('categorical_'),'risk_trained','imitation_only')
    seeds.groupby(['book','stage']).agg(seeds=('strategy','size'),mean_cvar95=('cvar95','mean'),
        seed_std_cvar95=('cvar95','std'),mean_pnl=('mean_net_pnl','mean')).reset_index().to_csv(output/'seed_summary.csv',index=False)
    chosen=frame.loc[~frame.strategy.str.startswith(('categorical_','imitation_only_'))|frame['selected'].eq(True)].copy()
    chosen['label']=np.where(chosen.strategy.str.startswith('categorical_'),'categorical_selected',chosen.strategy)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,metric in zip(axes.flat,('cvar95','cvar99','pnl_std','mean_net_pnl','mean_cost','mean_turnover_notional')):
        for name,g in chosen.groupby('label'):
            g=g.sort_values('book');ax.plot(g.book,g[metric],marker='o',label=name)
        ax.set(xlabel='Equal-sized option lots in book',ylabel=metric,xscale='log')
    axes[0,0].legend(fontsize=7);fig.suptitle('Matched liability and costs · all values normalized per option, not book-total return')
    save(fig,output,'rounding_bridge')
    lines=['# Phase 7 matched rounding bridge','',
        'The continuous teacher, closed-loop rounded teacher, newly trained categorical policies and conventional baselines share the same short-option liability and common test paths. This is a synthetic granularity experiment; the older long-option/historical checkpoints are not reused for an incompatible liability.','',
        'Books contain 1, 4, 10 and 40 equal-sized option lots. Underlying units per hedge lot equal option units per option lot; normalized action grids have spacing 1/book. Half-lot ties go toward zero and upper capacity is floor(1.2*book)/book. This boundary effect is part of executable granularity.','',
        'Rounded policies observe their actual rounded previous inventory. Continuous execution trajectories are not rounded open-loop after the fact. All quantities, premium and costs normalize to one option, avoiding misleading improvement from changing book size. Total normalized book P&L can be obtained by multiplying per-option results by book size.','',
        'Categorical policies receive an explicit imitation warm start from the frozen continuous teacher, then CVaR policy-gradient training with a learned threshold. An imitation-only diagnostic is retained. The combined residual difference includes architecture and optimization effects plus the additional warm-start budget; it is not a causal estimate of algorithm quality alone. Model/seed selection uses validation only.','',
        markdown_table(chosen,['book','label','mean_net_pnl','pnl_std','cvar95','mean_cost']),
        '', '![Rounding](rounding_bridge.png)','',markdown_table(paired,['book','strategy','reference','mean_pnl_difference','cvar95_difference','cvar95_lower95','cvar95_upper95']),
        '', 'These results exclude NIFTY basis, brokerage, SPAN margins, market depth and partial fills. They isolate granularity and matched-learning effects before any historical execution extension.',
        '', '`python src/run_phase7_research.py --phase4 PHASE4_RUN --output NEW_DIRECTORY`']
    (output/'ROUNDING_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def phase5_report(run,output):
    output.mkdir(parents=True,exist_ok=False);checked_manifest(run)
    frame=pd.read_csv(run/'robustness_comparison.csv');base=pd.read_csv(run/'target_world_baselines.csv')
    frame.loc[frame.strategy.str.startswith('neural_')].groupby(
        ['experiment','training_world','test_world','alpha','steps'],dropna=False).agg(
        seeds=('strategy','size'),mean_cvar95=('cvar95','mean'),seed_std_cvar95=('cvar95','std'),
        mean_pnl=('mean_net_pnl','mean')).reset_index().to_csv(output/'seed_summary.csv',index=False)
    plan=json.loads((run/'experiment_plan.json').read_text())
    cross=selected(frame.loc[frame.experiment.eq('cross_world')]).copy()
    worlds=['gbm','heston','bootstrap'];matrix=np.array([[float(cross.loc[cross.training_world.eq(a)&cross.test_world.eq(b),'cvar95'].iloc[0]) for b in worlds] for a in worlds])
    fig,ax=plt.subplots(figsize=(8,6),layout='constrained');image=ax.imshow(matrix,cmap='viridis')
    for i in range(3):
        for j in range(3):
            shade=(matrix[i,j]-matrix.min())/max(float(matrix.max()-matrix.min()),1e-12)
            ax.text(j,i,f'{matrix[i,j]:.5f}',ha='center',va='center',color='black' if shade>.55 else 'white')
    ax.set(xticks=range(3),xticklabels=worlds,yticks=range(3),yticklabels=worlds,xlabel='Evaluation world',ylabel='Training world',title='Selected policies · loss CVaR95 (lower is better)')
    fig.colorbar(image,ax=ax);save(fig,output,'cross_world_cvar')
    alpha=selected(frame.loc[frame.experiment.eq('alpha')]).sort_values('alpha')
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,metric in zip(axes,('mean_net_pnl','cvar95','cvar99')):ax.plot(alpha.alpha,alpha[metric],marker='o');ax.set(xlabel='Training CVaR alpha',ylabel=metric)
    fig.suptitle('GBM alpha sweep · common independent test paths; tail levels are not utility coefficients');save(fig,output,'alpha_sweep')
    freq=frame.loc[frame.experiment.eq('frequency')];freq=freq.loc[freq.strategy.eq('delta')|freq['selected'].eq(True)].copy()
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for ax,metric in zip(axes.flat,('pnl_std','cvar95','mean_cost','mean_turnover_notional')):
        for world in ('gbm','heston'):
            for kind in ('delta','neural'):
                g=freq.loc[freq.test_world.eq(world)&(freq.strategy.eq('delta') if kind=='delta' else freq.strategy.str.startswith('neural_'))].sort_values('steps')
                ax.plot(g.steps,g[metric],marker='o',label=world+'/'+kind)
        ax.set(xlabel='Intervals over the same 20 trading days',ylabel=metric,xscale='log')
        ax.xaxis.set_major_locator(FixedLocator([5,20,80]));ax.xaxis.set_major_formatter(ScalarFormatter());ax.xaxis.set_minor_locator(NullLocator())
    axes[0,0].legend(fontsize=8);save(fig,output,'frequency_sweep')
    stress=frame.loc[frame.experiment.eq('stress')]
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for ax,(name,g) in zip(axes,stress.groupby('test_world')):
        ax.bar(g.training_world,g.cvar95);ax.set(title=name,ylabel='Loss CVaR95');ax.tick_params(axis='x',rotation=25)
    save(fig,output,'stress_tests')
    lines=['# Phase 5 robustness and misspecification report','',
        'All calibration uses daily NIFTY observations ending 30 June 2026. Validation uses 1 July–15 September; the empirical test-return library begins 16 September. These historical dates were already inspected in earlier research. The new sampled paths are synthetic and do not restore a pristine real-market holdout.','',
        '## Clock and calibration','',
        'The nominal horizon is 20 trading days with 20 daily steps. GBM volatility ranges come from training-only 20-day realized volatility. Heston uses a capped realized-variance AR(1) moment proxy, not an identified option-surface fit; pricing assumes zero variance risk premium. The estimated parameters are recorded without overriding an unexpected sign to fit a market narrative.','',
        '```json',json.dumps(plan['calibration'],indent=2),'```','',
        'GBM uses BS premiums. Heston initial premiums use an audited Fourier grid; terminal payoff is exact. BS delta/gamma using observed current variance are proxy hedge features/baselines, not asserted optimal Heston hedges. Heston variance is directly observed in this simulated experiment, an oracle-information assumption.','',
        'Bootstrap samples three-day blocks within disjoint return libraries, uses a synthetic past 20-day prefix for strictly backward-looking realized volatility, and subtracts the training-only log moment. It is not an arbitrage-free model or a genuine 20-day historical trade backtest. Its initial premium is an explicitly shared BS reference quote. Held-out library drift/dependence may differ from training.','',
        '## Source-library coverage','',markdown_table(pd.DataFrame([{'split':name,**value} for name,value in plan['return_pools'].items()]),['split','rows','start','end']),
        '', 'Thousands of bootstrap simulations do not create thousands of independent historical return observations. Path-bootstrap confidence intervals condition on the fixed empirical library and exclude its estimation uncertainty.','',
        '## Cross-world comparison','',markdown_table(cross,['training_world','test_world','mean_net_pnl','pnl_std','cvar95','cvar99','mean_cost']),
        '', '![Matrix](cross_world_cvar.png)','',
        'Each model is selected only on its own source-world validation. Source-tuned constant-band controls are retained in the comparison file; target-world validation-tuned baselines are reported separately as stronger target-informed benchmarks. No model is selected using the evaluation world’s test outcomes.','',
        markdown_table(base,['world','strategy','mean_net_pnl','pnl_std','cvar95']),
        '', '## Alpha and rebalancing sweeps','',markdown_table(alpha,['alpha','mean_net_pnl','pnl_std','cvar95','cvar99','objective_cvar']),
        '', '![Alpha](alpha_sweep.png)','', '![Frequency](frequency_sweep.png)','',
        'Frequency models are trained separately at 5, 20 and 80 intervals; evaluation uses nested common finest-grid paths. Daily bootstrap is not interpolated into invented intraday observations, so the fine-frequency comparison is limited to GBM/Heston.','',
        '## Stress tests','', '![Stress](stress_tests.png)','',
        'The predeclared shifts increase volatility, alter Heston dynamics and remove bootstrap recentering. Policies remain frozen. Failure or degradation is evidence about robustness, not a reason to change the held-out bank.','',
        '## Reproduction','', '`python src/run_phase5_research.py --daily DAILY_PARQUET --output NEW_DIRECTORY`','',
        'Read pricing_audits.json, return-pool files, experiment_plan.json, frozen_before_test.json, robustness_comparison.csv and verification.json for provenance. No prospective profitability, calibrated real-market uncertainty or global optimality is claimed.']
    sensitivity=run/'observation_sensitivity/volatility_observation_sensitivity.csv'
    if sensitivity.exists():
        observed=pd.read_csv(sensitivity)
        lines.extend(['','## Volatility observability sensitivity','',
            'Frozen policies are re-evaluated on the same Heston paths after replacing latent variance by a training-range RMS prior until five past returns are observed, then trailing realized volatility. Initial latent variance is never used. Premium and terminal liability remain unchanged. This is a pre-test protocol correction, retained in supplemental_protocol_v2.json; no policy was refitted.',
            '',markdown_table(observed,['training_world','information','mean_net_pnl','pnl_std','cvar95','cvar99']),
            '', 'These are information-regime diagnostics, not a pure dynamics comparison. The main Heston policy has higher tail loss when latent variance is unavailable. The proxy-delta baseline uses the same proxy information.'])
    (output/'ROBUSTNESS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',choices=['phase4','phase7','phase5'])
    parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();globals()[args.phase+'_report'](args.run_dir,args.output)
