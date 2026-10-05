"""Reproduce the two added AUROC rows using cached pair/split eligibility.

No threshold search. Writes only a new output directory. The reference cache
determines eligible pairs, while the Diff-01 manifests determine training/test
rows. Retain these source identities when comparing against the paper snapshot.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scorer_depth_compute import prepare, BASE_SCORERS
from next_model_diff01 import fit_edges
from next_model_embedding_diff01 import fit_embedding_edges
from continuation_benefit_compute import predict
from manuscript_sources import DATASETS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair-cache', type=Path, required=True,
                        help='Directory containing dataset/results.parquet for all five datasets')
    parser.add_argument('--out', type=Path, default=Path('results/diff01_benefit_auroc'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summaries = []
    for dataset in DATASETS:
        raw, costs, rows, embeddings, _ = prepare(dataset, response=True)
        reference = pd.read_parquet(args.pair_cache/dataset/'results.parquet')
        records = []
        for split in range(50):
            path = Path('results/next_model_diff01_full')/dataset/'diff01_ridge'/f'split_{split:02d}.json'
            manifest = json.loads(path.read_text())
            ci, ti = [np.searchsorted(rows, manifest[k]) for k in ('cal_idx', 'test_idx')]
            np.testing.assert_array_equal(rows[ci], manifest['cal_idx'])
            np.testing.assert_array_equal(rows[ti], manifest['test_idx'])
            assert not set(ci) & set(ti)
            for arm in ('diff01_ridge', 'embedding_diff01_ridge'):
                xs = {m: embeddings[m] if arm.startswith('embedding') else df[BASE_SCORERS].to_numpy(float)
                      for m, df in raw.items()}
                cal = {m: dict(x=xs[m][ci], y=df.correct.to_numpy(float)[ci], c=costs[m][ci])
                       for m, df in raw.items()}
                _, edges = (fit_embedding_edges(cal, arm) if arm.startswith('embedding') else fit_edges(cal, arm))
                for pair in reference.loc[reference.split == split, 'pair'].drop_duplicates():
                    a, b = pair.split('→')
                    y = (raw[b].correct.to_numpy()[ti] > raw[a].correct.to_numpy()[ti]).astype(int)
                    score = -predict(xs[a][ti], edges[a, b])
                    auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else .5
                    records.append(dict(dataset=dataset, scorer=arm, pair=pair, split=split, benefit_auroc=auc))
        frame = pd.DataFrame(records)
        frame.to_csv(args.out/f'{dataset}.csv', index=False, mode='x')
        medians = frame.groupby(['scorer', 'pair']).benefit_auroc.median()
        for scorer, value in medians.groupby(level='scorer').mean().items():
            summaries.append(dict(dataset=dataset, scorer=scorer, benefit_auroc=value))
    pd.DataFrame(summaries).to_csv(args.out/'summary.csv', index=False, mode='x')


if __name__ == '__main__':
    main()
