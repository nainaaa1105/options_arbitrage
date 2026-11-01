"""Phase 6: causal GRU vs matched MLP, controlled ablations, seed diversity and closed-loop ensembles.

Protocol (written into experiment_plan.json before any model is fitted):
  * identical market, objective, costs, optimizer, update budget and per-seed training-path stream
    for every learned family; inventory is fed back in a sequential closed-loop rollout for all of them;
  * selection by validation objective only; everything is frozen before the independent test bank exists;
  * a claim of superiority needs a paired interval that excludes zero; otherwise "no detectable difference".
"""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .accounting import pnl
from .baselines import baseline
from .config import FitConfig, ResearchConfig
from .evaluation import metrics, paired_intervals
from .experiment_utils import begin, dump, freeze, save_batch, save_execution, sha
from .market import simulate
from .objectives import risk
from .recurrent import (Ensemble, PolicySpec, fit_recurrent, infer, latency_ms, load_step_policy, parameter_count)
from .run import select_baselines
from .training import fit

COSTS = (0., .001)
PRIMARY_COST_INDEX = 1
OBJECTIVES = ('cvar', 'entropic')
VALIDATION_SEED = 735001
TEST_SEED = 785001
GATE_MAE, GATE_STD = .06, 1.20

MLP = PolicySpec('mlp_delta', 'mlp', 'delta', 32)
GRU = PolicySpec('gru_delta', 'gru', 'delta', 16)
BAND = 'band_delta'
CORE = (MLP, GRU)

# name -> (spec, reference spec name, factor under test)
ABLATIONS = {
    'gru_delta_h32': (PolicySpec('gru_delta_h32', 'gru', 'delta', 32), 'gru_delta', 'recurrent capacity'),
    'gru_delta_reset': (PolicySpec('gru_delta_reset', 'gru', 'delta', 16, reset_memory=True), 'gru_delta', 'memory'),
    'gru_delta_noinv': (PolicySpec('gru_delta_noinv', 'gru', 'delta', 16, drop=('previous_magnitude',)), 'gru_delta', 'inventory input'),
    'mlp_delta_noinv': (PolicySpec('mlp_delta_noinv', 'mlp', 'delta', 32, drop=('previous_magnitude',)), 'mlp_delta', 'inventory input'),
    'mlp_flat': (PolicySpec('mlp_flat', 'mlp', 'flat', 32), 'mlp_delta', 'analytic delta prior'),
    'gru_flat': (PolicySpec('gru_flat', 'gru', 'flat', 16), 'gru_delta', 'analytic delta prior'),
    'mlp_flat_nodelta': (PolicySpec('mlp_flat_nodelta', 'mlp', 'flat', 32, drop=('absolute_delta',)), 'mlp_flat', 'delta feature'),
    'gru_flat_nodelta': (PolicySpec('gru_flat_nodelta', 'gru', 'flat', 16, drop=('absolute_delta',)), 'gru_flat', 'delta feature'),
}

CLAIM_RULES = {
    'architecture': 'GRU is called better than MLP only if the paired CVaR95 difference upper95 < 0 at the primary cost and cvar objective; worse if lower95 > 0; otherwise no detectable difference',
    'ablation': 'A factor matters only if variant-minus-reference paired CVaR95 lower95 > 0 (variant worse) or upper95 < 0 (variant better)',
    'ensemble': 'An ensemble helps only if its paired CVaR95 difference to the validation-selected single model of the same family has upper95 < 0',
    'numerical_gate': f'Zero-cost delta-prior MLP and GRU: delta MAE <= {GATE_MAE} and P&L std <= {GATE_STD} x analytic delta (diagnostic, not performance)',
    'intervals': 'IID synthetic paths conditional on the GBM specification and fitted models; not real-market confidence and exclude training-seed uncertainty',
    'scope': 'Synthetic zero-rate short European option; no NIFTY margin, basis or tax claim; observed constant volatility per path',
}


def candidate_names():
    return ('mlp_single', 'gru_single', 'mlp_seed_mean', 'gru_seed_mean', 'cross_family_mean')


def evaluate(step_model, batch, c):
    h = infer(step_model, batch, c)
    result = pnl(batch, h, c)
    return h, result, metrics(result, h, batch['delta'], c.entropic_aversion)


def train_cell(output, folder, c, settings, validation, objective, specs, include_band):
    """Fit every spec x seed for one (cost, objective) cell and choose the best seed by validation only."""
    chosen = {}
    plan = [(s.name, s) for s in specs] + ([(BAND, None)] if include_band else [])
    for name, spec in plan:
        sub = folder / name
        sub.mkdir()
        candidates = []
        for seed in settings.seeds:
            if spec is None:
                _, meta = fit(c, settings, validation, objective, 'delta', seed, sub, family='band')
            else:
                _, meta = fit_recurrent(c, settings, validation, objective, spec, seed, sub)
            candidates.append(meta)
        winner = min(candidates, key=lambda r: (r['validation_objective'], r['seed']))
        chosen[name] = {'selected': winner, 'all_seeds': candidates,
                        'path': str((sub / winner['checkpoint']).relative_to(output))}
    return chosen


def build_candidates(output, chosen, validation, c, objective):
    """Validation-defined single/ensemble candidates for the core families."""
    members = {}
    for spec in CORE:
        files = [output / Path(chosen[spec.name]['path']).parent / m['checkpoint'] for m in chosen[spec.name]['all_seeds']]
        members[spec.family] = [(f, load_step_policy(f)[0]) for f in files]
    singles = {}
    for spec in CORE:
        file = output / chosen[spec.name]['path']
        singles[spec.family] = (file, load_step_policy(file)[0])
    sets = {'mlp_single': [singles['mlp']], 'gru_single': [singles['gru']],
            'mlp_seed_mean': members['mlp'], 'gru_seed_mean': members['gru'],
            'cross_family_mean': members['mlp'] + members['gru']}
    definition, models = {}, {}
    for name in candidate_names():
        parts = sets[name]
        model = parts[0][1] if len(parts) == 1 else Ensemble([m for _, m in parts])
        h = infer(model, validation, c)
        score = float(risk(pnl(validation, h, c)['net_pnl'], c, objective))
        definition[name] = {'members': [str(f.relative_to(output)) for f, _ in parts], 'validation_objective': score}
        models[name] = model
    family_choice = {}
    for family, (single, mean) in {'mlp': ('mlp_single', 'mlp_seed_mean'), 'gru': ('gru_single', 'gru_seed_mean')}.items():
        family_choice[family] = mean if definition[mean]['validation_objective'] < definition[single]['validation_objective'] else single
    overall = min(candidate_names(), key=lambda n: (definition[n]['validation_objective'], candidate_names().index(n)))
    return definition, models, {'by_family': family_choice, 'overall': overall}


def correlation(a, b):
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')


def run(output, settings, costs=COSTS, ablations=True):
    primary_cost = costs[min(PRIMARY_COST_INDEX, len(costs) - 1)]
    begin(output, {
        'phase': 6, 'costs': list(costs), 'objectives': list(OBJECTIVES), 'primary_cost': primary_cost,
        'settings': asdict(settings), 'market': asdict(ResearchConfig()),
        'core_specs': [asdict(s) for s in CORE],
        'ablations': {k: {'spec': asdict(v[0]), 'reference': v[1], 'factor': v[2]} for k, v in ABLATIONS.items()} if ablations else {},
        'controls': ['unhedged', 'delta', 'constant_band', 'whalley_wilmott', 'ww_utility_matched', BAND],
        'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED,
        'matched': 'Same features, closed-loop inventory feedback, per-seed training-path stream, update budget, optimizer, objective, costs and bank for every family',
        'parameter_note': 'GRU hidden 16 (1217 params) vs MLP 2x32 (1345 params); GRU hidden 32 is a disclosed capacity ablation',
        'ensemble': 'Mean position; every member observes the ensemble executed inventory; no rounding',
        'selection': 'Validation objective only; ties to lower seed / simpler candidate; test generated after freeze',
        'claim_rules': CLAIM_RULES})
    selections, baseline_choices, ensemble_defs, models_by_cell = {}, {}, {}, {}
    banks = {}
    for index, cost in enumerate(costs):
        c = ResearchConfig(cost_rate=cost)
        validation = simulate(c, settings.validation_paths, seed=VALIDATION_SEED)
        for objective in OBJECTIVES:
            key = f'{index}_{objective}'
            choice, search = select_baselines(validation, c, objective)
            baseline_choices[key] = choice
            folder = output / f'cost_{index}_{objective}'
            folder.mkdir()
            search.to_csv(folder / 'baseline_search.csv', index=False)
            specs = list(CORE)
            if ablations and index == PRIMARY_COST_INDEX and objective == 'cvar':
                specs += [v[0] for v in ABLATIONS.values()]
            chosen = train_cell(output, folder, c, settings, validation, objective, specs, include_band=True)
            for name, value in chosen.items():
                selections[f'{key}/{name}'] = value
            definition, models, picked = build_candidates(output, chosen, validation, c, objective)
            ensemble_defs[key] = {'candidates': definition, 'selected': picked}
            models_by_cell[key] = models
    frozen = freeze(output, selections, baselines=baseline_choices, ensembles=ensemble_defs, test_seed=TEST_SEED)

    rows, paired, diversity, gates, latency, phase4 = [], [], [], [], [], []
    replay, chunk_check, reconciliation = {}, {}, []
    pnls = {}
    for index, cost in enumerate(costs):
        c = ResearchConfig(cost_rate=cost)
        batch = simulate(c, settings.test_paths, seed=TEST_SEED)
        save_batch(output / f'test_paths_cost_{index}.npz', batch)
        banks[index] = batch
        ref = pnl(batch, baseline(batch, c, 'delta'), c)
        for objective in OBJECTIVES:
            key = f'{index}_{objective}'
            choice = baseline_choices[key]
            for name in ('unhedged', 'delta', 'constant_band', 'whalley_wilmott'):
                h = baseline(batch, c, name, width=choice['constant_band'], aversion=choice['whalley_wilmott'])
                result = pnl(batch, h, c)
                m = metrics(result, h, batch['delta'])
                rows.append({'cost': cost, 'objective': objective, 'family': 'baseline', 'strategy': name, 'selected': False, **m})
                pnls[(key, name)] = result['net_pnl'].numpy()
                reconciliation.append(m['max_reconciliation_error'])
            h = baseline(batch, c, 'whalley_wilmott', aversion=c.entropic_aversion)
            rows.append({'cost': cost, 'objective': objective, 'family': 'baseline', 'strategy': 'ww_utility_matched', 'selected': False,
                         **metrics(pnl(batch, h, c), h, batch['delta'])})
            spec_names = [s.name for s in CORE] + [BAND]
            if ablations and index == PRIMARY_COST_INDEX and objective == 'cvar':
                spec_names += list(ABLATIONS)
            for name in spec_names:
                selection = selections[f'{key}/{name}']
                family = 'band' if name == BAND else (CORE[0].family if name.startswith('mlp') else 'gru')
                role = 'core' if name in (MLP.name, GRU.name, BAND) else 'ablation'
                seed_positions, seed_pnls, seed_cvar = [], [], []
                for candidate in selection['all_seeds']:
                    file = output / Path(selection['path']).parent / candidate['checkpoint']
                    model, _ = load_step_policy(file)
                    h, result, m = evaluate(model, batch, c)
                    selected = candidate['seed'] == selection['selected']['seed']
                    rows.append({'cost': cost, 'objective': objective, 'family': family, 'role': role, 'strategy': f'{name}_{candidate["seed"]}',
                                 'spec': name, 'seed': candidate['seed'], 'selected': selected, 'parameter_count': parameter_count(model),
                                 'validation_objective': candidate['validation_objective'], 'best_epoch': candidate['best_epoch'], **m})
                    save_execution(file.with_name(file.stem + '_execution.npz'), result, h)
                    reconciliation.append(m['max_reconciliation_error'])
                    seed_positions.append(h.numpy())
                    seed_pnls.append(result['net_pnl'].numpy())
                    seed_cvar.append(m['cvar95'])
                    if selected:
                        pnls[(key, name)] = result['net_pnl'].numpy()
                        fresh, _ = load_step_policy(file)
                        subset = {k: v[:128] for k, v in batch.items()}
                        replay[str(file.relative_to(output))] = float((infer(fresh, subset, c) - h[:128]).abs().max())
                        chunk_check[str(file.relative_to(output))] = float((infer(fresh, batch, c, chunk=777) - h).abs().max())
                        if cost == 0 and name in (MLP.name, GRU.name):
                            gates.append({'objective': objective, 'spec': name, 'delta_mae': m['delta_mae'],
                                          'std_over_delta': m['pnl_std'] / float(ref['net_pnl'].std()),
                                          'passed': m['delta_mae'] <= GATE_MAE and m['pnl_std'] / float(ref['net_pnl'].std()) <= GATE_STD})
                if len(seed_positions) > 1:
                    pairs = [(i, j) for i in range(len(seed_positions)) for j in range(i + 1, len(seed_positions))]
                    diversity.append({
                        'cost': cost, 'objective': objective, 'spec': name, 'seeds': len(seed_positions),
                        'cvar95_mean': float(np.mean(seed_cvar)), 'cvar95_std': float(np.std(seed_cvar, ddof=1)),
                        'cvar95_min': float(np.min(seed_cvar)), 'cvar95_max': float(np.max(seed_cvar)),
                        'mean_abs_position_gap': float(np.mean([np.abs(seed_positions[i] - seed_positions[j]).mean() for i, j in pairs])),
                        'position_correlation': float(np.nanmean([correlation(seed_positions[i], seed_positions[j]) for i, j in pairs])),
                        'pnl_correlation': float(np.nanmean([correlation(seed_pnls[i], seed_pnls[j]) for i, j in pairs]))})
            definition = ensemble_defs[key]['candidates']
            picked = ensemble_defs[key]['selected']
            for name in candidate_names():
                h, result, m = evaluate(models_by_cell[key][name], batch, c)
                rows.append({'cost': cost, 'objective': objective, 'family': 'ensemble', 'role': 'core', 'strategy': name, 'spec': name,
                             'selected': name == picked['overall'], 'selected_by_family': name in picked['by_family'].values(),
                             'validation_objective': definition[name]['validation_objective'], 'members': len(definition[name]['members']), **m})
                pnls[(key, name)] = result['net_pnl'].numpy()
                reconciliation.append(m['max_reconciliation_error'])
                if index == PRIMARY_COST_INDEX and objective == 'cvar':
                    latency.append({'strategy': name, 'members': len(definition[name]['members']),
                                    'rollout_ms_per_1024_paths': latency_ms(models_by_cell[key][name], batch, c),
                                    'note': 'CPU, 2 threads, 32 steps; relative indication only'})

            def pair(a, b, role, factor=''):
                paired.append({'cost': cost, 'objective': objective, 'role': role, 'factor': factor, 'candidate': a, 'reference': b,
                               **paired_intervals(pnls[(key, a)], pnls[(key, b)])})
            for name in (MLP.name, GRU.name, BAND):
                for reference in ('delta', 'constant_band', 'whalley_wilmott'):
                    pair(name, reference, 'vs_conventional')
            pair(GRU.name, MLP.name, 'architecture', 'memory architecture')
            pair('mlp_seed_mean', 'mlp_single', 'ensemble', 'seed mean')
            pair('gru_seed_mean', 'gru_single', 'ensemble', 'seed mean')
            best_single = 'mlp_single' if definition['mlp_single']['validation_objective'] <= definition['gru_single']['validation_objective'] else 'gru_single'
            pair('cross_family_mean', best_single, 'ensemble', 'cross-family mean')
            pair(picked['overall'], 'delta', 'validation_selected_vs_delta')
            if ablations and index == PRIMARY_COST_INDEX and objective == 'cvar':
                for name, (_, reference, factor) in ABLATIONS.items():
                    pair(name, reference, 'ablation', factor)
        # Frozen Phase 4 MLP checkpoints on this new bank, if the canonical run is present.
        runs = Path(__file__).resolve().parents[2] / 'research_runs.json'
        if runs.exists():
            import json
            mapping = json.loads(runs.read_text()).get('continuous_phase4')
            if mapping:
                folder = runs.parent / mapping
                frozen4 = json.loads((folder / 'frozen_before_test.json').read_text())
                from .training import load as load_phase4
                from .experiment_utils import infer as phase4_infer
                phase4_index = [0., .0002, .001, .005].index(cost) if cost in (0., .0002, .001, .005) else None
                if phase4_index is not None:
                    for objective in OBJECTIVES:
                        selection = frozen4['selections'][f'{phase4_index}_{objective}_mlp']
                        file = folder / selection['path']
                        if sha(file) != frozen4['checkpoint_sha256'][selection['path']]:
                            raise ValueError('Frozen Phase 4 checkpoint changed')
                        # Phase 4's own inference route (inventory masked when frictionless), not the Phase 6 closed loop.
                        legacy, _ = load_phase4(file)
                        h = phase4_infer(legacy, batch, c)
                        result = pnl(batch, h, c)
                        m = metrics(result, h, batch['delta'], c.entropic_aversion)
                        phase4.append({'cost': cost, 'objective': objective, 'strategy': 'phase4_frozen_mlp', 'source': mapping, **m})
    pd.DataFrame(rows).to_csv(output / 'comparison.csv', index=False)
    pd.DataFrame(paired).to_csv(output / 'paired_comparisons.csv', index=False)
    pd.DataFrame(diversity).to_csv(output / 'seed_diversity.csv', index=False)
    pd.DataFrame(latency).to_csv(output / 'latency.csv', index=False)
    pd.DataFrame(phase4).to_csv(output / 'phase4_frozen_mlp_on_new_bank.csv', index=False)
    changed = [p for p, v in frozen['checkpoint_sha256'].items() if sha(output / p) != v]
    if changed:
        raise AssertionError(f'Frozen checkpoints changed after test: {changed[:3]}')
    if max(replay.values()) > 1e-6 or max(chunk_check.values()) > 1e-6:
        raise AssertionError('Checkpoint replay or chunk-invariance mismatch')
    dump(output / 'verification.json', {
        'numerical_gates': gates, 'numerical_gates_pass': all(g['passed'] for g in gates),
        'replay_max_abs_error': max(replay.values()), 'chunk_invariance_max_abs_error': max(chunk_check.values()),
        'max_reconciliation_error': max(reconciliation), 'checkpoints_replayed': len(replay),
        'frozen_checkpoints_unchanged': True, 'test_retuning': False,
        'validation_seed': VALIDATION_SEED, 'test_seed': TEST_SEED, 'independent_of_phase4_7_5_seeds': True,
        'same_test_paths_across_cost_levels_and_objectives': True, 'real_market_profitability_established': False})
    print('Phase 6 complete; numerical gates:', gates, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--flat-epochs', type=int, default=48)
    parser.add_argument('--updates', type=int, default=24)
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--validation-paths', type=int, default=4096)
    parser.add_argument('--test-paths', type=int, default=16384)
    parser.add_argument('--seeds', type=int, nargs='+', default=[11, 22, 33])
    parser.add_argument('--no-ablations', action='store_true')
    args = parser.parse_args()
    settings = FitConfig(seeds=tuple(args.seeds), epochs=args.epochs, flat_epochs=args.flat_epochs, updates=args.updates,
                         batch=args.batch, validation_paths=args.validation_paths, test_paths=args.test_paths)
    run(args.output, settings, ablations=not args.no_ablations)


if __name__ == '__main__':
    main()
