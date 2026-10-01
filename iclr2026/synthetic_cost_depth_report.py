"""Summarize completed synthetic price cells, keeping oracle and held-out separate."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def report(out, require_complete=False):
    rows = []
    audit = []
    expected = set()
    fitted = []
    scorers = set()
    for path in sorted(out.glob('*/manifest.json')):
        meta = json.loads(path.read_text())
        dataset = meta['dataset']
        scorers.add(meta['scorer'])
        fitted.append(dict(dataset=dataset, **{k: meta['fitted_beta'][k] for k in ('a', 'b', 'sse', 'rmse')}))
        expected.update(out/dataset/s/'oracle.csv' for s in meta['settings'])
        expected.update(out/dataset/s/f'split_{i:02d}.csv'
                        for s in meta['settings'] for i in range(meta['splits']))
        for mode, checks in meta['audit'].items():
            for setting, current in checks['settings'].items():
                audit.append(dict(dataset=meta['dataset'], mode=mode, setting=setting,
                    obs_low=checks['observed']['low'], obs_high=checks['observed']['high'],
                    obs_lower=checks['observed']['anchors'][0], obs_upper=checks['observed']['anchors'][1],
                    new_lower=current['anchors'][0], new_upper=current['anchors'][1],
                    **{k: current[k] for k in ('budgets_byte_identical', 'order_identical', 'low_identical', 'high_identical')}))
    pd.DataFrame(audit).to_csv(out/'invariant_audit.csv', index=False)
    pd.DataFrame(fitted).to_csv(out/'fitted_beta.csv', index=False)
    missing = sorted(str(p.relative_to(out)) for p in expected if not p.exists())
    (out/'completion.json').write_text(json.dumps(dict(expected=len(expected),
        completed=len(expected)-len(missing), missing=missing), indent=2)+'\n')
    if require_complete and (missing or not expected):
        raise ValueError(f'Incomplete run: {len(missing)} cells missing of {len(expected)} expected')
    for path in sorted(out.glob('*/*/*.csv')):
        f = pd.read_csv(path)
        if 'd3_accuracy' not in f:
            continue
        manifest = json.loads(path.with_suffix('.json').read_text())
        if manifest['results_sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'Result checksum mismatch: {path}')
        if len(f) != 500 or not np.isfinite(f.select_dtypes('number')).all().all():
            raise ValueError(f'Invalid budget results: {path}')
        np.testing.assert_allclose(f.budget, manifest['budgets'], rtol=1e-12, atol=1e-15)
        row = dict(dataset=f.dataset.iloc[0], setting=f.setting.iloc[0],
                   mode='oracle' if f['mode'].iloc[0] == 'oracle' else 'confirmation',
                   split=f['mode'].iloc[0])
        for depth in (1, 2, 3):
            row[f's{depth}_accuracy_pp'] = 100 * np.trapz(f[f'd{depth}_accuracy'], f.fraction)
            row[f's{depth}_cost_per_1000'] = 1000 * np.trapz(f[f'd{depth}_cost'], f.fraction)
            row[f's{depth}_overshoot_fraction'] = (f[f'd{depth}_overshoot'] > 1e-12).mean()
        row['s3_minus_s2_pp'] = row['s3_accuracy_pp'] - row['s2_accuracy_pp']
        row['s2_gain_pp'] = row['s2_accuracy_pp'] - row['s1_accuracy_pp']
        row['s3_gain_pp'] = row['s3_accuracy_pp'] - row['s1_accuracy_pp']
        row['s3_minus_s2_cost_per_1000'] = row['s3_cost_per_1000'] - row['s2_cost_per_1000']
        row['depth3_fraction'] = np.trapz((f.d3_selected_depth == 3).astype(float), f.fraction)
        rows.append(row)
    if rows:
        frame = pd.DataFrame(rows)
        frame.to_csv(out/'cell_integrals.csv', index=False)
        summary = frame.groupby(['dataset', 'mode', 'setting']).agg(
            cells=('split', 'count'), s3_minus_s2_pp=('s3_minus_s2_pp', 'mean'),
            s2_gain_pp=('s2_gain_pp', 'mean'), s3_gain_pp=('s3_gain_pp', 'mean'),
            s3_minus_s2_cost_per_1000=('s3_minus_s2_cost_per_1000', 'mean'),
            p10_pp=('s3_minus_s2_pp', lambda x: x.quantile(.1)),
            p90_pp=('s3_minus_s2_pp', lambda x: x.quantile(.9)),
            depth3_fraction=('depth3_fraction', 'mean'))
        summary.to_csv(out/'summary.csv')
        print(summary.to_string())
        if not missing:
            plot(summary, out)
            lines = ['# Synthetic cost depth results', '',
                f'Completed {len(expected)} of {len(expected)} planned cells across {len(fitted)} datasets.', '',
                f'Confidence scorer: {", ".join(sorted(scorers))}.', '',
                'All nine specified Beta price structures and OBS were evaluated with full-sample '
                'exact S1/S2/S3 selection. Every setting also used 50 matched calibration/test splits. '
                'Confidence scores, correctness, and token counts were fixed. Input and output prices '
                'received the same per-model multiplier and per-query costs were recomputed.', '',
                'The selected protocol preserves the Beta construction and recomputes each setting’s '
                'budget anchors and calibration cost order. Dollar budgets and split-specific order '
                'can differ from OBS. The 500 normalized budget fractions and split seeds are unchanged.', '',
                'The table reports S3 minus S2 accuracy, integrated over normalized budget fractions '
                'using the trapezoidal rule and expressed in percentage points. Held-out values are '
                'means across the 50 test splits. Policies are selected on calibration data '
                'and frozen before test evaluation.', '',
                '| Dataset | Test OBS (pp) | Test S-- (pp) | Test S++ (pp) |',
                '| --- | ---: | ---: | ---: |']
            for dataset in summary.index.get_level_values('dataset').unique():
                heldout = summary.loc[(dataset, 'confirmation')].s3_minus_s2_pp
                lines.append(f'| {dataset} | '
                             f'{heldout["OBS"]:.4f} | {heldout["S--"]:.4f} | {heldout["S++"]:.4f} |')
            lines += ['', 'The figure shows mean test-set gains for all nine Beta settings. '
                'The symmetric path uses log(a+b), and the asymmetric path uses log(a/b). '
                'The U result is identical in both panels. A diamond shows each dataset’s OBS '
                'test gain at its fitted parameter position on the nearer path. Nearness is '
                'Euclidean distance in (log(a), log(b)) space to the plotted parameter paths. '
                'All five observed fits lie nearer the symmetric path. '
                'S1/S2/S3 denote nested deterministic threshold-cascade classes '
                'of depth at most one, two, and three drawn from the eight-model pool.', '',
                '![Synthetic cost results](cost_depth.png)', '',
                'Detailed outputs include `summary.csv`, `cell_integrals.csv`, `fitted_beta.csv`, '
                '`invariant_audit.csv`, and `completion.json`. Each cell has a price and frozen-policy '
                'manifest saved before test replay. The report verifies CSV checksums, finite values, '
                '500 budget rows, and agreement with the saved budget vectors.', '']
            (out/'REPORT.md').write_text('\n'.join(lines))


def path_distance(point, path):
    """Euclidean distance to a piecewise-linear parameter path in log space."""
    distances = []
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        t = np.clip(np.dot(point-start, delta) / np.dot(delta, delta), 0., 1.)
        distances.append(np.linalg.norm(point - (start + t*delta)))
    return min(distances)


def plot(summary, out):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/synthetic-cost-mpl')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    datasets = ['mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench']
    datasets = [d for d in datasets if d in summary.index.get_level_values('dataset')]
    titles = dict(mmlu='MMLU', triviaqa='TriviaQA', math_hard='MATH',
                  simpleqa='SimpleQA', livecodebench='LiveCodeBench')
    from matplotlib.lines import Line2D
    symmetric = ['S--', 'S-', 'U', 'S+', 'S++']
    asymmetric = ['L++', 'L+', 'U', 'R+', 'R++']
    sym_ab = np.array([[.25,.25], [.5,.5], [1.,1.], [2.,2.], [4.,4.]])
    asym_ab = np.array([[1.,4.], [1.,2.], [1.,1.], [2.,1.], [4.,1.]])
    xs = [np.log(sym_ab.sum(axis=1)), np.log(asym_ab[:,0]/asym_ab[:,1])]
    paths = [np.log(sym_ab), np.log(asym_ab)]
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharey=True)
    plot_rows = []
    for j, dataset in enumerate(datasets):
        # Same ordering and tab10 colors as calibration_learning_curve_report.py.
        color = plt.cm.tab10.colors[j]
        heldout = summary.loc[(dataset, 'confirmation')]
        if not (heldout.reindex(['OBS']+symmetric+asymmetric).cells == 50).all():
            raise ValueError(f'{dataset}: plot requires 50 test splits at every setting')
        for panel, settings in enumerate((symmetric, asymmetric)):
            values = heldout.reindex(settings).s3_minus_s2_pp.to_numpy()
            axes[panel].plot(xs[panel], values, '-o', color=color, lw=1.8, ms=4,
                             label=titles[dataset])
            for setting, x, y in zip(settings, xs[panel], values):
                plot_rows.append(dict(dataset=dataset, panel=panel, setting=setting,
                                      x=float(x), test_gain_pp=float(y)))
        meta = json.loads((out/dataset/'manifest.json').read_text())
        a, b = (meta['fitted_beta'][k] for k in ('a', 'b'))
        distances = [path_distance(np.log([a,b]), path) for path in paths]
        panel = int(np.argmin(distances))
        x = np.log(a+b) if panel == 0 else np.log(a/b)
        y = heldout.loc['OBS', 's3_minus_s2_pp']
        axes[panel].scatter([x], [y], marker='D', s=64, color=color,
                            edgecolor='white', linewidth=1.1, zorder=5)
        plot_rows.append(dict(dataset=dataset, panel=panel, setting='OBS', x=float(x),
                              test_gain_pp=float(y), a=a, b=b,
                              symmetric_distance=float(distances[0]),
                              asymmetric_distance=float(distances[1])))
    for panel, ax in enumerate(axes):
        labels = symmetric if panel == 0 else asymmetric
        ax.set_xticks(xs[panel], [f'{x:.2f}\n{s.replace("-", "−")}' for x,s in zip(xs[panel],labels)])
        ax.axhline(0, color='#777777', lw=.8)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.margins(x=.08)
    axes[0].set(title='Symmetric price structure', xlabel=r'$\log(a+b)$')
    axes[1].set(title='Asymmetric price structure', xlabel=r'$\log(a/b)$')
    for ax, left, right in ((axes[0], 'Barbell', 'Interior-dense'),
                            (axes[1], 'Cheap-dense', 'Expensive-dense')):
        ax.text(0, 1.015, left, transform=ax.transAxes, ha='left', fontsize=9, color='.4')
        ax.text(1, 1.015, right, transform=ax.transAxes, ha='right', fontsize=9, color='.4')
        ax.title.set_y(1.08)
    axes[0].set_ylabel('Mean test-set $S_3 - S_2$ accuracy difference (pp)\nBudget-integrated, averaged over 50 splits')
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], marker='D', color='.4', linestyle='None', markersize=6))
    labels.append('Observed pool (fitted)')
    fig.legend(handles, labels, loc='lower center', ncol=6, frameon=False, fontsize=9)
    targets = {json.loads((out/d/'manifest.json').read_text()).get('target_auroc') for d in datasets}
    if targets != {None}:
        if len(targets) != 1 or None in targets:
            raise ValueError('Cannot combine different confidence targets in one figure')
        fig.suptitle(f'Synthetic confidence, target AUROC {next(iter(targets)):.1f}', fontsize=12)
    fig.tight_layout(rect=(0, .075, 1, .94 if targets != {None} else 1), w_pad=2.2)
    pd.DataFrame(plot_rows).to_csv(out/'plot_data.csv', index=False)
    fig.savefig(out/'cost_depth.png', dpi=200)
    fig.savefig(out/'cost_depth.pdf')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/synthetic_cost_depth'))
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    report(args.out, args.require_complete)
