"""Phase 6 tables, figures and a mechanical claim-rule summary from a completed, frozen run."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'data/research/.matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from research_registry import sha256

INK, MUTED, GRID = '#0b0b0b', '#52514e', '#dcdbd6'
COLOR = {'mlp': '#2a78d6', 'gru': '#eb6834', 'band': '#1baf7a', 'baseline': '#8d8c86', 'ensemble': '#6250d6'}
PRIMARY_KEY = '1_cvar'


def family_of(name):
    if name.startswith('mlp'):
        return 'mlp'
    if name.startswith('gru'):
        return 'gru'
    if name.startswith('band'):
        return 'band'
    if name.endswith('mean') or name.endswith('single'):
        return 'ensemble'
    return 'baseline'


def style(ax):
    ax.grid(axis='x', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED)


def verdict(row, better_when_negative=True):
    if row.cvar95_upper95 < 0:
        return 'candidate lower CVaR95 (interval excludes 0)'
    if row.cvar95_lower95 > 0:
        return 'candidate higher CVaR95 (interval excludes 0)'
    return 'no detectable CVaR95 difference'


def md(frame, columns, formats=None):
    formats = formats or {}
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    for _, r in frame.iterrows():
        cells = []
        for c in columns:
            v = r[c]
            f = formats.get(c)
            cells.append(f.format(v) if f and pd.notna(v) and not isinstance(v, str) else ('' if pd.isna(v) else str(v)))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def report(run, output):
    output.mkdir(parents=True, exist_ok=False)
    frozen = json.loads((run / 'frozen_before_test.json').read_text())
    for name, value in frozen['checkpoint_sha256'].items():
        if sha256(run / name) != value:
            raise ValueError(f'Frozen checkpoint changed: {name}')
    plan = json.loads((run / 'experiment_plan.json').read_text())
    comparison = pd.read_csv(run / 'comparison.csv')
    paired = pd.read_csv(run / 'paired_comparisons.csv')
    diversity = pd.read_csv(run / 'seed_diversity.csv')
    latency = pd.read_csv(run / 'latency.csv')
    verification = json.loads((run / 'verification.json').read_text())
    costs = plan['costs']
    sections = ['# Phase 6 report: GRU, ablations and ensembles', '',
                f'Run `{run.name}`. All figures read the frozen test bank (seed {plan["test_seed"]}, '
                f'{int(comparison.paths.iloc[0]):,} paths); checkpoint hashes verified unchanged. '
                'Synthetic zero-rate short-option world; no market profitability is implied.', '']
    # Table 1: headline at each cost/objective, selected rows only.
    cols = ['strategy', 'parameter_count', 'mean_net_pnl', 'pnl_std', 'cvar95', 'cvar99', 'mean_cost', 'mean_turnover_units', 'delta_mae']
    fmt = {'mean_net_pnl': '{:.5f}', 'pnl_std': '{:.5f}', 'cvar95': '{:.5f}', 'cvar99': '{:.5f}', 'mean_cost': '{:.5f}',
           'mean_turnover_units': '{:.3f}', 'delta_mae': '{:.4f}', 'parameter_count': '{:.0f}'}
    for index, cost in enumerate(costs):
        for objective in ('cvar', 'entropic'):
            sub = comparison[(comparison.cost == cost) & (comparison.objective == objective)]
            keep = sub[(sub.family == 'baseline') | (sub.selected.astype(bool) & (sub.family != 'ensemble')) | sub.family.eq('ensemble')]
            keep = keep[keep.role.isna() | keep.role.eq('core') | keep.family.eq('ensemble')] if 'role' in keep else keep
            sections += [f'## Cost {cost:g}, {objective} objective (validation-selected seed per family)', '', md(keep, cols, fmt), '']
    # Figure 1: primary-cell CVaR95 with seed dots.
    primary = comparison[(comparison.cost == costs[-1]) & (comparison.objective == 'cvar')]
    names = ['delta', 'constant_band', 'whalley_wilmott', 'band_delta', 'mlp_delta', 'gru_delta',
             'mlp_seed_mean', 'gru_seed_mean', 'cross_family_mean']
    unhedged = float(primary[primary.strategy == 'unhedged'].cvar95.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 4.4), layout='constrained')
    y = np.arange(len(names))[::-1]
    for yy, name in zip(y, names):
        fam = family_of(name)
        if name in primary.strategy.values:
            v = float(primary[primary.strategy == name].cvar95.iloc[0])
        else:
            seeds = primary[primary.spec == name]
            v = float(seeds[seeds.selected.astype(bool)].cvar95.iloc[0])
            ax.scatter(seeds.cvar95, [yy] * len(seeds), s=26, facecolor='none', edgecolor=COLOR[fam], linewidth=1.2, zorder=2)
        ax.scatter([v], [yy], s=60, color=COLOR[fam], edgecolor='white', linewidth=1, zorder=3)
        ax.text(v, yy + .32, f'{v:.5f}', ha='center', fontsize=8, color=INK)
    ax.set_yticks(y, names)
    ax.set_xlabel('Test CVaR95 of net P&L loss (lower is better); open dots = other seeds', color=MUTED)
    ax.set_title(f'CVaR95 at cost {costs[-1]:g}, CVaR objective (unhedged = {unhedged:.4f}, off scale)', loc='left', color=INK, fontsize=10)
    style(ax)
    fig.savefig(output / 'fig1_cvar95_primary.png', dpi=160)
    plt.close(fig)
    # Figure 2: validation curves, MLP vs GRU.
    fig, ax = plt.subplots(figsize=(8, 4.2), layout='constrained')
    folder = run / f'cost_{len(costs) - 1}_cvar'
    for spec in ('mlp_delta', 'gru_delta'):
        fam = family_of(spec)
        for k, file in enumerate(sorted((folder / spec).glob('*_curve.csv'))):
            curve = pd.read_csv(file)
            ax.plot(curve.epoch, curve.validation_objective, color=COLOR[fam], linewidth=1.6, alpha=.9,
                    label=spec if k == 0 else None, linestyle=['-', '--', ':'][k % 3])
    ax.set_xlabel('Epoch (0 = analytic-delta prior)', color=MUTED)
    ax.set_ylabel('Validation CVaR objective', color=MUTED)
    ax.set_title('Validation curves (line style = seed)', loc='left', color=INK)
    ax.legend(frameon=False)
    style(ax)
    ax.grid(axis='y', color=GRID, linewidth=.6)
    fig.savefig(output / 'fig2_validation_curves.png', dpi=160)
    plt.close(fig)
    # Figure 3: paired CVaR95 differences with 95% intervals (architecture, ensemble, ablation) at the primary cell.
    p = paired[(paired.cost == costs[-1]) & (paired.objective == 'cvar') & paired.role.isin(['architecture', 'ensemble', 'ablation'])].copy()
    p['label'] = p.candidate + ' - ' + p.reference
    fig, ax = plt.subplots(figsize=(8.6, .45 * len(p) + 1.6), layout='constrained')
    yy = np.arange(len(p))[::-1]
    for y0, (_, r) in zip(yy, p.iterrows()):
        fam = family_of(r.candidate)
        ax.plot([r.cvar95_lower95, r.cvar95_upper95], [y0, y0], color=COLOR[fam], linewidth=2)
        ax.scatter([r.cvar95_difference], [y0], s=36, color=COLOR[fam], edgecolor='white', linewidth=1, zorder=3)
    ax.axvline(0, color=INK, linewidth=.9)
    ax.set_yticks(yy, p.label)
    ax.set_xlabel('Paired CVaR95 difference (candidate - reference); < 0 = smaller loss tail', color=MUTED)
    ax.set_title('Architecture, ensemble and ablation contrasts, 95% paired bootstrap', loc='left', color=INK)
    style(ax)
    fig.savefig(output / 'fig3_paired_contrasts.png', dpi=160)
    plt.close(fig)
    # Figure 4: seed diversity (per-seed CVaR95 spread) at the primary cell.
    d = diversity[(diversity.cost == costs[-1]) & (diversity.objective == 'cvar')]
    fig, ax = plt.subplots(figsize=(8, .4 * len(d) + 1.6), layout='constrained')
    yy = np.arange(len(d))[::-1]
    for y0, (_, r) in zip(yy, d.iterrows()):
        fam = family_of(r.spec)
        ax.plot([r.cvar95_min, r.cvar95_max], [y0, y0], color=COLOR[fam], linewidth=3, solid_capstyle='round')
        ax.scatter([r.cvar95_mean], [y0], s=26, color='white', edgecolor=INK, zorder=3)
    ax.set_yticks(yy, d.spec)
    ax.set_xlabel('Test CVaR95 across seeds (range; dot = mean)', color=MUTED)
    ax.set_title('Seed-to-seed variation', loc='left', color=INK)
    style(ax)
    fig.savefig(output / 'fig4_seed_spread.png', dpi=160)
    plt.close(fig)
    sections += ['## Paired contrasts (primary cost, CVaR objective)', '',
                 md(p.assign(verdict=[verdict(r) for _, r in p.iterrows()]),
                    ['candidate', 'reference', 'role', 'cvar95_difference', 'cvar95_lower95', 'cvar95_upper95',
                     'mean_pnl_difference', 'verdict'],
                    {k: '{:.6f}' for k in ('cvar95_difference', 'cvar95_lower95', 'cvar95_upper95', 'mean_pnl_difference')}), '',
                 '## Seed diversity', '', md(diversity, ['cost', 'objective', 'spec', 'cvar95_mean', 'cvar95_std', 'mean_abs_position_gap',
                                                          'position_correlation', 'pnl_correlation'],
                                              {k: '{:.5f}' for k in ('cvar95_mean', 'cvar95_std', 'mean_abs_position_gap', 'position_correlation', 'pnl_correlation')}), '',
                 '## Inference cost', '', md(latency, ['strategy', 'members', 'rollout_ms_per_1024_paths'], {'rollout_ms_per_1024_paths': '{:.1f}'}), '',
                 '## Verification', '', '```json', json.dumps({k: v for k, v in verification.items()}, indent=2), '```', '',
                 '## Claim rules (fixed before fitting)', ''] + [f'- **{k}**: {v}' for k, v in plan['claim_rules'].items()]
    (output / 'PHASE6_REPORT.md').write_text('\n'.join(sections) + '\n', encoding='utf-8')
    print('Report written to', output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.run, args.output)
