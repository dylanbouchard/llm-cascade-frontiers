"""Regenerate appendix tables from the complete five-draw experiment."""
from pathlib import Path
import numpy as np
import pandas as pd
from synthetic_score_draws import summarize, OUT

PAPER = Path(__file__).resolve().parent / 'iclr'
DATASETS = (('mmlu', 'MMLU'), ('triviaqa', 'TriviaQA'), ('math_hard', 'MATH'),
            ('simpleqa', 'SimpleQA'), ('livecodebench', 'LiveCodeBench'))


def main():
    integrals = pd.read_csv(OUT / 'split_integrals.csv')
    expected = {(d, t) for d, _ in DATASETS for t in (.8, .9)}
    if set(zip(integrals.dataset, integrals.target_auroc)) != expected:
        raise ValueError('Expected all five datasets and both AUROC targets')
    draws, across = summarize(integrals, draws=5, splits=50)
    pd.testing.assert_frame_equal(draws, pd.read_csv(OUT / 'per_draw_summary.csv'))
    pd.testing.assert_frame_equal(across, pd.read_csv(OUT / 'across_draw_summary.csv'))
    draws = draws.set_index(['dataset', 'target_auroc']).sort_index()
    across = across.set_index(['dataset', 'target_auroc'])
    means = [r'\begin{tabular}{llrrrrrrr}', r'\toprule',
             r'& & \multicolumn{5}{c}{Mean gain for each draw} & \multicolumn{2}{c}{Across draws} \\',
             r'\cmidrule(lr){3-7}\cmidrule(lr){8-9}',
             r'Dataset & AUROC & 1 & 2 & 3 & 4 & 5 & Mean & SD \\', r'\midrule']
    splits = [r'\begin{tabular}{llrrrrr}', r'\toprule',
              r'& & \multicolumn{5}{c}{SD across splits within each draw} \\',
              r'\cmidrule(lr){3-7}',
              r'Dataset & AUROC & 1 & 2 & 3 & 4 & 5 \\', r'\midrule']
    for index, (dataset, name) in enumerate(DATASETS):
        if index:
            means.append(r'\addlinespace[2pt]')
            splits.append(r'\addlinespace[2pt]')
        for target in (.8, .9):
            cell = draws.loc[(dataset, target)].sort_values('draw')
            np.testing.assert_array_equal(cell.draw, np.arange(5))
            summary = across.loc[(dataset, target)]
            label = name if target == .8 else ''
            gain_values = [f'${x:+.3f}$' for x in cell.mean_gain_pp]
            sd_values = [f'${x:.3f}$' for x in cell.split_sd_pp]
            means.append(' & '.join([label, f'{target:.1f}'] + gain_values
                         + [f'${summary.mean_gain_pp:+.3f}$', f'${summary.across_draw_sd_pp:.3f}$']) + r' \\')
            splits.append(' & '.join([label, f'{target:.1f}'] + sd_values) + r' \\')
    for filename, lines in [('table_synthetic_draw_means.tex', means),
                            ('table_synthetic_draw_splits.tex', splits)]:
        lines += [r'\bottomrule', r'\end{tabular}']
        (PAPER / 'tables' / filename).write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
