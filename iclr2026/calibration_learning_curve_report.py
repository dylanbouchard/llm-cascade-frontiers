"""Render the fixed-test learning curve and accompanying manuscript table."""
from pathlib import Path
import json
from manuscript_sources import ROOT, result_path, file_record, validate_learning
from calibration_learning_curve import aggregate
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = ROOT / 'iclr/results/calibration_learning_curve'
PAPER = ROOT / 'iclr'
NAMES = dict(mmlu='MMLU', triviaqa='TriviaQA', math_hard='MATH', simpleqa='SimpleQA', livecodebench='LiveCodeBench')


def load_current():
    pieces, provenance = [], []
    for dataset in NAMES:
        records = []
        for split in range(50):
            path = result_path(dataset, 'calibration_learning_curve') / dataset / f'split_{split:02d}.csv'
            exact_path = result_path(dataset, 'exact_heldout_five') / dataset / path.name
            manifest = json.loads(path.with_suffix('.json').read_text())
            exact = json.loads(exact_path.with_suffix('.json').read_text())
            validate_learning(manifest, exact, dataset, split)
            frame, baseline = pd.read_csv(path), pd.read_csv(exact_path)
            assert len(frame) == 2500 and set(frame.dataset) == {dataset} and set(frame.split) == {split}
            assert set(frame.fraction) == {.1, .25, .5, .75, 1.}
            for _, group in frame.groupby('fraction'):
                group = group.sort_values('budget_index')
                np.testing.assert_array_equal(group.budget_index, np.arange(500))
                np.testing.assert_allclose(group.budget, manifest['budgets'], rtol=0, atol=1e-15)
            full = frame[frame.fraction == 1.].sort_values('budget_index')
            baseline = baseline.sort_values('budget_index')
            for depth in (2, 3):
                for metric in ('accuracy', 'cost', 'cal_accuracy', 'cal_cost', 'calls'):
                    ref = 'mean_calls' if metric == 'calls' else metric
                    np.testing.assert_allclose(full[f'd{depth}_{metric}'], baseline[f'd{depth}_{ref}'], rtol=0, atol=1e-12)
            records.append(frame)
            provenance.append(dict(dataset=dataset, split=split, data_sha256=manifest['data_sha256'],
                files=[file_record(p) for p in (path, path.with_suffix('.json'), exact_path, exact_path.with_suffix('.json'))]))
        pieces.append(aggregate(pd.concat(records, ignore_index=True)))
    means = pd.concat(pieces, ignore_index=True)
    metrics = [c for c in means if c not in ('dataset', 'split', 'fraction', 'n_cal', 'n_test')]
    summary = means.groupby(['dataset', 'fraction', 'n_cal', 'n_test'])[metrics].agg(
        ['mean', lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
    labels = {'mean': 'mean', '<lambda_0>': 'p10', '<lambda_1>': 'p90'}
    summary.columns = [f'{a}_{labels[b]}' for a, b in summary.columns]
    return summary.reset_index(), means, provenance


def main():
    data, means, provenance = load_current()
    OUT.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT/'summary.csv', index=False)
    means.to_csv(OUT/'split_budget_means.csv', index=False)
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.25))
    metrics = [('test_gain_pp', 'Test-set accuracy', 'Difference (percentage points)'),
               ('cal_gain_pp', 'Calibration-set accuracy', 'Difference (percentage points)'),
               ('cost_gain_pct', 'Test-set cost', 'Difference / assigned budget (%)')]
    for ax, (metric, title, ylabel) in zip(axes, metrics):
        for (dataset, name), color, marker in zip(NAMES.items(), plt.cm.tab10.colors, ['o','s','^','D','v']):
            part = data[data.dataset == dataset].sort_values('n_cal')
            x = part.n_cal.to_numpy()
            ax.plot(x, part[metric+'_mean'], label=name, color=color, marker=marker, markersize=3, lw=1.3)
            ax.fill_between(x, part[metric+'_p10'], part[metric+'_p90'], color=color, alpha=.10, linewidth=0)
        ax.axhline(0, color='.35', linestyle=':', linewidth=.8)
        ax.set(title=title, xlabel='Calibration-set examples', ylabel=ylabel)
        ax.grid(alpha=.15)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, frameon=False, bbox_to_anchor=(.5, -.01))
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(PAPER/'figures/figA7_cal_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(OUT/'learning_curve.png', dpi=180, bbox_inches='tight')
    lines = [r'\begin{table}[htbp]', r'\centering', r'\scriptsize',
        r'\caption{Fixed-test learning-curve diagnostics, averaged within each seed and then across 50 seeds. Coverage is the percentage of the original 500 budgets feasible for both classes at every size. Infeasibility uses the original grid. Other diagnostics use the common feasible grid. Each paired entry reports $\leq2$ / $\leq3$ models. Overshoot frequency counts realized cost above budget, and magnitude averages $100\max(C-B,0)/B$, including zeros. Calls are expected model calls per query.}',
        r'\label{tab:cal_sensitivity}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
        r'Dataset & $n_{\rm cal}$ & Coverage & Infeasible & Overshoot & Magnitude & Calls \\',
        r' & & (\%) & (\%) & (\%) & (\%) & \\', r'\midrule']
    for dataset, name in NAMES.items():
        for i, (_, row) in enumerate(data[data.dataset == dataset].sort_values('n_cal').iterrows()):
            def pair(metric, digits=1):
                return ' / '.join(f'{row[f"d{d}_{metric}_mean"]:.{digits}f}' for d in (2, 3))
            lines.append(f'{name if i == 0 else ""} & {int(row.n_cal)} & {100*row.coverage_mean:.1f} & '+
                pair('infeasible_pct')+' & '+pair('overshoot_pct')+' & '+pair('positive_overshoot_pct',2)+' & '+pair('calls',2)+r' \\')
        if dataset != 'livecodebench':
            lines.append(r'\addlinespace')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (PAPER/'tables/table_calibration_learning_curve.tex').write_text('\n'.join(lines)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(generator=file_record(Path(__file__).resolve()), source_map=file_record(ROOT/'manuscript_sources.py'), inputs=provenance, outputs=[file_record(PAPER/'tables/table_calibration_learning_curve.tex'), file_record(PAPER/'figures/figA7_cal_sensitivity.pdf')]), indent=2)+'\n')
    print(data[['dataset','n_cal','n_test','test_gain_pp_mean','cal_gain_pp_mean','cost_gain_pct_mean','coverage_mean']].to_string(index=False))


if __name__ == '__main__':
    main()
