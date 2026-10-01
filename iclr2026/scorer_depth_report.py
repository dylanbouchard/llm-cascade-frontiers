"""Generate appendix tables from complete scorer-depth results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scorer_depth_compute import ALL_SCORERS, DATASETS, report

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'results/scorer_depth'
PAPER = ROOT/'iclr'
LABELS = ['Mean negentropy', 'Min negentropy', r'Prob.\ margin', 'Min probability',
          r'Length-norm.\ probability', 'Correctness-trained token-prob.', 'Correctness-trained embedding']
NAMES = ['MMLU', 'TriviaQA', 'MATH', 'SimpleQA', 'LiveCodeBench']


def main():
    # Verify matched splits and assigned budgets across scorers before reporting.
    by_budget = []
    baseline_matches = []
    for dataset in DATASETS:
        for split in range(50):
            reference = None
            for scorer in ALL_SCORERS:
                path = OUT/dataset/scorer/f'split_{split:02d}.csv'
                manifest = json.loads(path.with_suffix('.json').read_text())
                f = pd.read_csv(path)
                signature = [manifest[k] for k in ('cal_idx', 'test_idx', 'budgets', 'pool')]
                if reference is None:
                    reference = signature
                assert signature == reference
                assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
                np.testing.assert_allclose(f.fraction, np.linspace(0, 1, 500), atol=1e-14)
                np.testing.assert_allclose(f.budget, manifest['budgets'], rtol=0, atol=1e-15)
                assert (f.d3_cal_accuracy >= f.d2_cal_accuracy-1e-12).all()
                assert (f.d2_cal_cost <= f.budget+1e-12).all()
                assert (f.d3_cal_cost <= f.budget+1e-12).all()
                by_budget.append(f.assign(accuracy_pp=100*(f.d3_accuracy-f.d2_accuracy),
                                          cost_per_1000=1000*(f.d3_cost-f.d2_cost)))
                if scorer == 'mean_token_negentropy':
                    oldpath = ROOT/'results/exact_heldout_depth'/dataset/path.name
                    old = json.loads(oldpath.with_suffix('.json').read_text())
                    match = all(old[k] == manifest[k] for k in ('cal_idx', 'test_idx', 'pool'))
                    baseline_matches.append(dict(dataset=dataset, split=split, matched=match))
                    if match:
                        original = pd.read_csv(oldpath)
                        for depth in (2, 3):
                            for metric in ('accuracy', 'cost', 'cal_accuracy', 'cal_cost'):
                                np.testing.assert_allclose(f[f'd{depth}_{metric}'], original[f'd{depth}_{metric}'], atol=1e-12)
    pd.DataFrame(baseline_matches).to_csv(OUT/'baseline_split_matches.csv', index=False)
    frame = pd.concat(by_budget, ignore_index=True)
    frame.groupby(['dataset', 'scorer', 'fraction'])[['accuracy_pp', 'cost_per_1000']].agg(
        ['mean', 'median', lambda x: x.quantile(.1), lambda x: x.quantile(.9)]).to_csv(OUT/'by_budget.csv')
    s = report(OUT, DATASETS, 50, ALL_SCORERS)
    lines = [r'\begin{table}[htbp]', r'\centering', r'\scriptsize',
        r'\caption{Scorer robustness of the exact $S_3$ versus $S_2$ comparison. '
        r'Top panel reports mean test-set accuracy differences in percentage points, '
        r'with 10th--90th split percentiles in brackets. Bottom panel reports mean '
        r'realized-cost differences in dollars per 1,000 queries. Differences integrate '
        r'over 500 assigned budget positions and average equally over 50 splits. '
        r'Positive values favor $S_3$ in accuracy and indicate greater cost.}',
        r'\label{tab:scorer-depth}', r'\resizebox{\linewidth}{!}{%', r'\begin{tabular}{lccccc}',
        r'\toprule', 'Scorer & '+' & '.join(NAMES)+r' \\', r'\midrule',
        r'\multicolumn{6}{c}{Accuracy difference (percentage points)} \\', r'\midrule']
    for scorer, label in zip(ALL_SCORERS, LABELS):
        cells = []
        for dataset in DATASETS:
            r = s.loc[(dataset, scorer)]
            cells.append(r"\shortstack{" + f"${r.accuracy_pp:+.3f}$" + r" \\ " + f"$[{r.p10_pp:+.3f}, {r.p90_pp:+.3f}]$" + "}")
        lines.append(label+' & '+' & '.join(cells)+r' \\')
    lines += [r'\midrule', r'\multicolumn{6}{c}{Realized-cost difference (\$/1,000 queries)} \\', r'\midrule']
    for scorer, label in zip(ALL_SCORERS, LABELS):
        lines.append(label+' & '+' & '.join(f"${s.loc[(d, scorer), 'cost_per_1000']:+.5f}$" for d in DATASETS)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    (PAPER/'tables/table_scorer_depth.tex').write_text('\n'.join(lines)+'\n')
    cells = s.reset_index()
    lo, hi = cells.loc[cells.accuracy_pp.idxmin()], cells.loc[cells.accuracy_pp.idxmax()]
    text = (f"Across the 35 scorer--dataset combinations, the mean budget-integrated "
        f"test-set accuracy difference ranges from ${lo.accuracy_pp:+.3f}$ to "
        f"${hi.accuracy_pp:+.3f}$ percentage points. ")
    labels = dict(zip(ALL_SCORERS, ['mean token negentropy', 'minimum token negentropy', 'probability margin', 'minimum token probability', 'sequence probability', 'the log-probability ensemble', 'the response-aware scorer']))
    names = dict(zip(DATASETS, NAMES))
    text += (f"The largest positive difference occurs for {labels[hi.scorer]} on "
             f"{names[hi.dataset]}, with a 10th--90th split-percentile range of "
             f"$[{hi.p10_pp:+.3f}, {hi.p90_pp:+.3f}]$ percentage points. ")
    (OUT/'range.txt').write_text(text+'\n')
    (PAPER/'tables/scorer_depth_findings.tex').write_text(text+'\n')
    (OUT/'REPORT.md').write_text('# Scorer robustness results\n\n'+text+'\n\n'
        'See summary.csv for all scorer and dataset cells, including paired costs and split percentiles. '
        'The comparison covers exact S3 versus S2 and does not test deeper classes.\n')


if __name__ == '__main__':
    main()
