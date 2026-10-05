"""Readable tables and a figure for the completed shared-difficulty experiment."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/shared-difficulty-mpl')
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(out):
    summary = pd.read_csv(out/'summary.csv')
    diag = pd.read_csv(out/'selected_chain_summary.csv')
    construction = pd.read_csv(out/'construction_diagnostics.csv')
    run = __import__('json').loads((out/'run.json').read_text())
    keys = ['dataset', 'target']
    gains = summary.pivot(index=keys, columns='condition', values='gain_pp')
    maximum = summary[summary.condition == 'max'].set_index(keys).w
    monotone = (gains['own'] >= gains['half']) & (gains['half'] >= gains['max'])
    first_stage = diag[(diag.population == 'test') & (diag.depth_limit == 3) & (diag.stage == 0)]
    downstream = first_stage.pivot(index=keys, columns='condition', values='A_downstream')
    rhs = first_stage.pivot(index=keys, columns='condition', values='eq10_rhs')
    downstream_rises = (downstream['own'] < downstream['half']) & (downstream['half'] < downstream['max'])
    rhs_falls = (rhs['own'] > rhs['half']) & (rhs['half'] > rhs['max'])
    lines = ['# Shared-difficulty results', '',
        f'Completed {len(summary)} cells with {run["splits"]} matched splits per cell. '
        'All cells use eight models and 500 normalized assigned budget positions. '
        'One paired score draw is used. No manuscript files were edited.', '',
        f'The held-out S3 minus S2 gain declines from w=0 to the maximum mixture in '
        f'{int((gains["max"] < gains["own"]).sum())} of {len(gains)} dataset-target comparisons. '
        f'It declines at both mixture steps in {int(monotone.sum())} of {len(gains)} comparisons.', '',
        'For the first stage of selected S3 policies, downstream AUROC rises at both '
        f'mixture steps in {int(downstream_rises.sum())} of {len(gains)} comparisons, while '
        f'equation 10 RHS falls at both steps in {int(rhs_falls.sum())} of {len(gains)}. '
        'Thus the stage diagnostic follows the proposed mechanism more consistently than '
        'the optimized depth gap. At AUROC 0.9, all four datasets with positive baseline '
        'depth gains have negative gains at maximum w. LiveCodeBench remains negative.', '',
        '| Dataset | Target AUROC | w=0 gain | w=0.5 gain | Maximum-w gain | Maximum w |',
        '|---|---:|---:|---:|---:|---:|']
    for index, row in gains.iterrows():
        lines.append(f'| {index[0]} | {index[1]:.1f} | {row["own"]:+.3f} | {row["half"]:+.3f} | '
                     f'{row["max"]:+.3f} | {maximum.loc[index]:.2f} |')
    lines += ['', 'Gains are percentage points, integrated over assigned budgets and averaged '
        'over splits. Negative held-out gaps can occur even though S3 contains S2 and the '
        'calibration gap is nonnegative. Split SDs and realized test cost gaps are in summary.csv.', '',
        '## Selected-chain equation 10 diagnostics', '',
        'All values below use held-out queries reaching the specified stage. AUROCs are '
        'budget-weighted means across selected policies. Equation 10 RHS is expressed in '
        'percentage points and already includes reach probability. Coverage is the fraction '
        'of budget positions with that nonterminal stage, averaged over splits. Undefined '
        'AUROCs are omitted from their means. These values describe selected chains, which '
        'can change across mixtures.', '']
    for limit, stage in ((2, 0), (3, 0), (3, 1)):
        lines += [f'### S{limit}, stage {stage+1}', '',
            '| Dataset | Target | Mixture | A_i | A_downstream | Equation 10 RHS (pp) | Coverage |',
            '|---|---:|---|---:|---:|---:|---:|']
        cell = diag[(diag.population == 'test') & (diag.depth_limit == limit) & (diag.stage == stage)]
        for dataset in gains.index.get_level_values(0).unique():
            for target in (.8, .9):
                for condition in ('own', 'half', 'max'):
                    selected = cell[(cell.dataset == dataset) & (cell.target == target) & (cell.condition == condition)]
                    if selected.empty:
                        continue
                    r = selected.iloc[0]
                    lines.append(f'| {dataset} | {target:.1f} | {condition} | {r.A_i:.3f} | '
                        f'{r.A_downstream:.3f} | {100*r.eq10_rhs:+.3f} | {r.policy_budget_mass:.3f} |')
    error = (construction.full_auroc-construction.target).abs().max()
    lines += ['', '## Interpretation and limits', '',
        f'The largest full-sample AUROC calibration error is {error:.6f}. '
        'The w=0 scores are identical to the original experiment. New mixtures hold that '
        'target\'s residual noise fixed and allow Pearson correlations to change. The largest '
        f'absolute correlation change is {construction.correlation_max_error.max():.3f}. '
        'This tests the shared-difficulty construction with changing cross-model score '
        'dependence, rather than isolating downstream AUROC while holding that dependence fixed.', '',
        'Equation 10 predicts gain over random escalation for a frozen continuation policy. '
        'It does not guarantee monotonic optimized depth gains. Agreement between the '
        'downstream-ranking diagnostics and depth gains is empirical evidence for this '
        'mechanism, conditional on this construction and score draw.', '',
        'The maximum mixture is a grid maximum under the documented conservative feasibility '
        'rule. Difficulty is discrete, and a small own-label term can break its ties. '
        'A maximum of 0.99 should not be described as pure difficulty.', '',
        'Scores use full-sample correctness labels. Held-out evaluation applies to policy '
        'selection, not learning a deployable scorer. These are one-draw results. Overlapping '
        'split variation is not independent replication. No significance claims are made.', '']
    (out/'INTERPRETATION.md').write_text('\n'.join(lines))

    datasets = list(gains.index.get_level_values(0).unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(16, 6.3), sharex=True, sharey='row')
    for col, dataset in enumerate(datasets):
        for target, color in ((.8, '#246eb9'), (.9, '#b65035')):
            cell = summary[(summary.dataset == dataset) & (summary.target == target)].set_index('condition').loc[['own', 'half', 'max']]
            axes[0, col].plot(cell.w, cell.gain_pp, 'o-', color=color, label=f'AUROC {target}')
            d = diag[(diag.dataset == dataset) & (diag.target == target) &
                (diag.population == 'test') & (diag.depth_limit == 3) & (diag.stage == 0)].set_index('condition').loc[['own', 'half', 'max']]
            axes[1, col].plot(d.w, 100*d.eq10_rhs, 'o-', color=color)
        axes[0, col].set_title(dataset)
        axes[0, col].axhline(0, color='#999999', linewidth=.7)
        axes[1, col].set_xlabel('Shared-difficulty weight w')
        for ax in axes[:, col]:
            ax.grid(alpha=.2)
            ax.set_xticks([0, .5, 1])
    axes[0, 0].set_ylabel('Held-out S3 minus S2 (pp)')
    axes[1, 0].set_ylabel('S3 first-stage equation 10 RHS (pp)')
    axes[0, 0].legend()
    fig.suptitle(f'Shared difficulty at matched correctness AUROC\nOne score draw, {run["splits"]} matched splits, all eight models', fontsize=12)
    fig.tight_layout()
    fig.savefig(out/'shared_difficulty.png', dpi=170)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/shared_difficulty_depth'))
    report(parser.parse_args().out)
