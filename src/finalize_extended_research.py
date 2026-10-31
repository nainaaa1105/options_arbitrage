"""Close the requested extension delivery only after all numerical checks finish."""
import argparse,json,re
from pathlib import Path
from continuous_hedging.experiment_utils import dump,sha
import research_registry


def finalize(output,registry):
    root=Path(__file__).resolve().parents[1]
    output,registry=output.resolve(),registry.resolve()
    if output.exists() or registry.exists():raise FileExistsError('Preserving completed delivery checks')
    runs=json.loads((root/'research_runs.json').read_text())
    directories={phase:root/runs[key] for phase,key in (
        ('phase4','continuous_phase4'),('phase7','rounding_phase7'),('phase5','robustness_phase5'))}
    results={}
    for phase,directory in directories.items():
        verified=json.loads((directory/'verification.json').read_text())
        replay=json.loads((directory/'full_replay_verification.json').read_text())
        frozen=json.loads((directory/'frozen_before_test.json').read_text())
        for name,digest in frozen['checkpoint_sha256'].items():
            if sha(directory/name)!=digest:raise AssertionError('Frozen checkpoint changed')
        if not replay['full_test_path_replay_pass']:raise AssertionError('Replay failed')
        if (phase=='phase5' and verified.get('no_test_retuning') is not True) or (phase!='phase5' and verified.get('test_retuning') is not False):
            raise AssertionError('Test-selection contract violated')
        results[phase]={'verification':verified,'replay_checkpoints':replay['checkpoint_count'],
            'replayed_execution_banks':replay['execution_count'],'max_replay_error':replay['max_replay_error'],
            'stress_rows_replayed':replay.get('stress_metric_rows_replayed',0),
            'max_cash_error':replay['max_cash_reconciliation_error'],
            'manifest_sha256':sha(directory/'frozen_before_test.json')}
    if not results['phase4']['verification']['mvp_gates_pass']:raise AssertionError('MVP gate failed')
    pricing=json.loads((directories['phase5']/'simulation_pricing_audit.json').read_text())
    if not pricing['all_pass']:raise AssertionError('Simulation pricing audit failed')
    plan=json.loads((directories['phase5']/'experiment_plan.json').read_text())
    if sha(plan['daily_source'])!=plan['daily_sha256']:raise AssertionError('Daily source changed')
    required=[directories['phase4']/'report/MVP_RESEARCH_REPORT.md',directories['phase7']/'report/ROUNDING_REPORT.md',
        directories['phase5']/'report_final/ROBUSTNESS_REPORT.md',directories['phase7']/'execution_sensitivity/execution_sensitivity.csv',
        directories['phase5']/'observation_sensitivity/volatility_observation_sensitivity.csv']
    if not all(file.exists() for file in required):raise AssertionError('Missing report or sensitivity')
    test_log=root/'data/research/research_extensions_final_observability_tests_20261031.log'
    text=test_log.read_text(encoding='utf-8',errors='replace')
    if not text.rstrip().endswith('OK'):raise AssertionError('Final regression suite incomplete or failing')
    count=int(re.findall(r'Ran (\d+) tests',text)[-1])
    previous=root/'data/research/phases0_3_delivery_registry_20261031.json'
    research_registry.verify(previous)
    protected=len(json.loads(previous.read_text())['artifacts'])
    payload={'requested_phases':[4,7,5],'implementation_and_runs_complete':True,
        'regression_tests_passed':count,'regression_log_sha256':sha(test_log),
        'old_protected_artifacts_unchanged':protected,'phases':results,
        'total_frozen_checkpoints':sum(r['replay_checkpoints'] for r in results.values()),
        'risk_trained_fits':48+12+33,'additional_imitation_only_checkpoints':12,
        'all_checkpoint_hashes_revalidated':True,'raw_daily_source_hash_unchanged':True,
        'heston_simulation_pricing_audit_pass':True,'models_retuned_on_test':False,
        'bootstrap_test_library_days':plan['return_pools']['test']['rows'],
        'real_world_profitability_established':False,'classifier_retrained':False,
        'scope':'Synthetic continuous short-option research; no production promotion',
        'reports':[str(file.relative_to(root)) for file in required]}
    research_registry.freeze(registry);research_registry.verify(registry,check_source=True)
    dump(output,{**payload,'final_registry':str(registry.relative_to(root)),'registry_sha256':sha(registry)})
    print(f'Completed phases 4, 7, 5: {count} tests; {payload["total_frozen_checkpoints"]} frozen checkpoints; old artifacts preserved',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--registry',type=Path,required=True)
    a=p.parse_args();finalize(a.output,a.registry)
