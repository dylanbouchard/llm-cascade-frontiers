"""Refresh the main table's cascade-gain panel from matched eight-model caches.

Run from the repository root with .venv/bin/python iclr/cascade_gain_table.py.
Existing depth-gain rows are preserved.
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manuscript_sources import DATASETS, EXPECTED_MODELS, result_path

SCORERS = ('mean_token_negentropy', 'min_token_negentropy', 'probability_margin',
           'min_probability', 'sequence_probability', 'logreg_ensemble', 'response_logreg')
NAMES = ('MMLU', 'TriviaQA', 'MATH', 'SimpleQA', 'LiveCodeBench')


def main():
    rows = []
    for dataset, name in zip(DATASETS, NAMES):
        gains = [[] for _ in SCORERS]
        for split in range(50):
            baseline = result_path(dataset, 'exact_heldout_five') / dataset / f'split_{split:02d}.csv'
            single = pd.read_csv(baseline)
            reference = json.loads(baseline.with_suffix('.json').read_text())
            assert set(reference['pool']) == EXPECTED_MODELS
            assert len(single) == 500
            for index, scorer in enumerate(SCORERS):
                path = result_path(dataset, 'scorer_depth') / dataset / scorer / baseline.name
                frame = pd.read_csv(path)
                manifest = json.loads(path.with_suffix('.json').read_text())
                assert manifest['dataset'] == dataset and manifest['scorer'] == scorer
                assert manifest['seed'] == reference['seed'] == 42 + split
                for key in ('pool', 'cal_idx', 'test_idx'):
                    assert manifest[key] == reference[key], (dataset, scorer, split, key)
                assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
                for column in ('fraction', 'budget'):
                    np.testing.assert_allclose(frame[column], single[column], rtol=0, atol=1e-14)
                if index == 0:
                    np.testing.assert_allclose(frame.d2_accuracy, single.d2_accuracy, rtol=0, atol=1e-12)
                gains[index].append(100 * np.trapz(frame.d2_accuracy - single.d1_accuracy, single.fraction))
        cells = [f'${np.mean(values):+.3f}$' for values in gains]
        rows.append(name + r' & $S_2-S_1$ & ' + ' & '.join(cells) + r' \\')
    path = ROOT / 'iclr/tables/table_exact_depth.tex'
    text = path.read_text()
    header, body = text.split('\\midrule\n', 1)
    # Also accept an already refreshed table, making repeated runs idempotent.
    if r'\textbf{Depth gain}' in body:
        body = body.split(r'\textbf{Depth gain}} \\' + '\n', 1)[1]
    panel = [r'\multicolumn{9}{l}{\textbf{Cascade gain}} \\', *rows,
             r'\midrule', r'\multicolumn{9}{l}{\textbf{Depth gain}} \\']
    path.write_text(header + '\\midrule\n' + '\n'.join(panel) + '\n' + body)
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
