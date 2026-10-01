"""Render observed/synthetic frontier comparisons from validated frozen caches.

Run .venv/bin/python iclr/frontier_figure.py from the repository root.
"""
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/frontier-figure-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manuscript_sources import DATASETS, EXPECTED_MODELS, result_path
from cascade_gain_table import SCORERS
PAPER = ROOT / 'iclr'
NAMES = ('MMLU', 'TriviaQA', 'MATH', 'SimpleQA', 'LiveCodeBench')
LABELS = ('Mean token negentropy', 'Min token negentropy', 'Probability margin',
          'Min probability', 'Length-normalized probability', 'Logistic regression ensemble',
          'Response-level logistic regression')
COLORS = ('#60636a', '#147878', '#d99218', '#6854a3', '#be5369')
X = np.linspace(0, 1, 500)


def read(path):
    return pd.read_csv(path), json.loads(path.with_suffix('.json').read_text())


def load_scores():
    cases = [(s, label, 5) for s, label in zip(SCORERS, LABELS)]
    cases += [(f'auroc_{t:.1f}', f'Synthetic confidence scores (AUROC = {t:.1f})', 4)
              for t in (.5, .6, .7, .8, .9)]
    data = {key: {} for key, _, _ in cases}
    for dataset in DATASETS:
        stacks = {key: [] for key, _, _ in cases}
        for split in range(50):
            name = f'split_{split:02d}.csv'
            base, ref = read(result_path(dataset, 'exact_heldout_five') / dataset / name)
            for key, _, depth in cases:
                if key == SCORERS[0]:
                    f, m = base, ref
                elif key.startswith('auroc_'):
                    f, m = read(ROOT / 'results/simulated_signal_depth_four' / dataset / key / name)
                    assert m['target_auroc'] == float(key[6:])
                else:
                    f, m = read(ROOT / 'experiments/scorer_s5_20260916/results' / dataset / key / name)
                    assert m['scorer'] == key
                assert set(m['pool']) == EXPECTED_MODELS
                assert m['dataset'] == dataset and m['seed'] == 42 + split
                for field in ('cal_idx', 'test_idx'):
                    assert m[field] == ref[field], (dataset, key, split, field)
                assert not set(m['cal_idx']) & set(m['test_idx'])
                np.testing.assert_allclose(f.fraction, X, rtol=0, atol=1e-14)
                np.testing.assert_allclose(f.budget, base.budget, rtol=0, atol=1e-14)
                np.testing.assert_allclose(f.budget, m['budgets'], rtol=0, atol=1e-14)
                if 'd1_accuracy' in f:
                    np.testing.assert_allclose(f.d1_accuracy, base.d1_accuracy, atol=1e-12, rtol=0)
                values = [base.d1_accuracy.to_numpy()]
                for d in range(2, depth + 1):
                    assert (f[f'd{d}_cal_cost'] <= f.budget + 1e-12).all()
                    values.append(f[f'd{d}_accuracy'].to_numpy())
                a = 100 * np.array(values)
                assert np.isfinite(a).all() and ((a >= 0) & (a <= 100)).all()
                stacks[key].append(a)
        for key in stacks:
            data[key][dataset] = np.stack(stacks[key])
        print(f'Validated all 12 score cases for {dataset}', flush=True)
    return cases, data


def row(axes, data, gain=False, show_titles=True):
    for ax, dataset, name in zip(axes, DATASETS, NAMES):
        a = data[dataset]
        for d in range(3 if gain else 1, a.shape[1] + 1):
            values = a[:, d-1] - a[:, 1] if gain else a[:, d-1]
            y = values.mean(axis=0) if gain else np.median(values, axis=0)
            ax.plot(X, y, color=COLORS[d-1], lw=1.15, label=f'$S_{d}$')
            if gain:
                lo, hi = np.quantile(values, [.1, .9], axis=0)
                ax.fill_between(X, lo, hi, color=COLORS[d-1], alpha=.12)
        if show_titles:
            ax.set_title(name, fontsize=10, pad=7)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .5, 1])
        ax.tick_params(labelsize=8)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 5, 10]))
        if gain:
            ax.axhline(0, color='#999999', lw=.6)
    axes[0].set_ylabel('Mean gain vs. $S_2$ (pp)' if gain else 'Median test accuracy (%)')


def canvas(titles, side_titles=False):
    fig = plt.figure(figsize=(10, 5.2), layout='constrained')
    subfigs = fig.subfigures(2, 1)
    axes = []
    for subfig, title in zip(subfigs, titles):
        if title is None:
            pass
        elif side_titles:
            subfig.supylabel(title, fontsize=10, fontweight='medium')
        else:
            subfig.suptitle(title, fontsize=11, fontweight='bold')
        axes.append(subfig.subplots(1, 5))
    return fig, axes


def main(main_only=False):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    cases, data = load_scores()
    fig, axes = canvas([None, None])
    row_labels = ('Observed scores', 'Synthetic scores')
    # Synthetic target: AUROC = 0.9 (stated in the caption).
    for index, (axs, key) in enumerate(zip(axes, [SCORERS[0], 'auroc_0.9'])):
        row(axs, data[key], show_titles=index == 0)
        axs[0].set_ylabel(r'$\bf{' + row_labels[index].replace(' ', r'\ ') + '}$\nMedian accuracy (%)')
        axs[0].legend(fontsize=8, ncol=2, loc='lower right')
    # Match scales within each dataset so the score intervention is visible.
    for top, bottom in zip(*axes):
        limits = [top.get_ylim(), bottom.get_ylim()]
        for ax in (top, bottom):
            ax.set_ylim(min(l[0] for l in limits), max(l[1] for l in limits))
    for ax in axes[1]:
        ax.set_xlabel('Assigned budget fraction')
    fig.savefig(PAPER / 'figures/fig_exact_frontiers.pdf')
    plt.close(fig)
    if main_only:
        return
    tex = [r'\subsection{Frontiers Across Observed and Synthetic Confidence Scores}',
           r'\label{app:score-frontiers}',
           'The following figures report each of the seven observed confidence scorers and five synthetic AUROC targets. '
           'All use the eight-model pool, 50 matched calibration-test splits, 500 assigned budget positions, and exhaustive threshold search. '
           'Observed scores are evaluated through $S_5$ and synthetic scores through $S_4$. '
           'The standalone baseline $S_1$ is shared across score choices. '
           'Split percentiles describe sensitivity to the split, not confidence intervals.']
    for key, title, depth in cases:
        fig, axes = canvas([title, 'Paired accuracy differences relative to $S_2$'])
        row(axes[0], data[key])
        row(axes[1], data[key], gain=True)
        axes[0][0].legend(fontsize=8, ncol=2, loc='lower right')
        for ax in axes[1]:
            ax.set_xlabel('Assigned budget fraction')
        filename = 'fig_frontiers_' + key.replace('.', '') + '.pdf'
        fig.savefig(PAPER / 'figures' / filename)
        plt.close(fig)
        tex += [r'\begin{figure}[p]', r'\centering',
                r'\includegraphics[width=\linewidth]{figures/' + filename + '}',
                r'\caption{' + title + '. Top row shows median test accuracy for $S_1$ through $S_' + str(depth) +
                '$. Bottom row shows mean paired test accuracy differences relative to $S_2$, with 10th--90th split-percentile bands. '
                'Policies are selected on calibration data and evaluated at their assigned budget fractions.}',
                r'\end{figure}']
    tex.append(r'\FloatBarrier')
    (PAPER / 'sections/score_frontiers_appendix.tex').write_text('\n\n'.join(tex) + '\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-only', action='store_true',
                        help='Render only the main frontier comparison.')
    main(main_only=parser.parse_args().main_only)
