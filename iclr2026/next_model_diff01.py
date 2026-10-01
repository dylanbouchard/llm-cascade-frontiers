"""Exact held-out S2/S3 using next-model Diff-01 confidence scores.

Standalone experiment. Refuses existing output directories. No manuscript writes.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/next-model-diff01-mpl')
import argparse
import hashlib
import json
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from continuation_benefit_compute import fit_linear, predictor, predict, evaluate, cutoffs
from scorer_depth_compute import prepare, BASE_SCORERS
from fig2_compute import make_split, SEED
from exact_heldout_depth import FRACTIONS, encode
from oracle_depth_analysis import score_ranks, prefix_1d, prefix_2d
from voi_compute import DATASETS

ARMS = ('diff01_ridge', 'correctness_ridge', 'mean_token_negentropy')
SOURCES = ('next_model_diff01.py', 'test_next_model_diff01.py', 'continuation_benefit_compute.py',
           'scorer_depth_compute.py', 'fig2_compute.py', 'exact_heldout_depth.py',
           'oracle_depth_analysis.py', 'optuna_frontier.py', 'voi_compute.py')


def fit_edges(cal, arm):
    models = sorted(cal, key=lambda m: (cal[m]['c'].mean(), m))
    edges = {}
    own = {}
    for m in models:
        if not np.isin(cal[m]['y'], [0, 1]).all():
            raise ValueError('Correctness must be binary')
        if arm == 'correctness_ridge':
            own[m] = predictor(*fit_linear(cal[m]['x'], cal[m]['y']))
        elif arm == 'mean_token_negentropy':
            own[m] = dict(coef=[1., 0., 0., 0., 0.], intercept=0.)
        elif arm != 'diff01_ridge':
            raise ValueError(arm)
    for a, b in combinations(models, 2):
        edges[a, b] = (predictor(*fit_linear(cal[a]['x'], cal[b]['y']-cal[a]['y']), sign=-1.)
                       if arm == 'diff01_ridge' else own[a])
    return models, edges


def select(cal, budgets, arm):
    models, edges = fit_edges(cal, arm)
    n = len(cal[models[0]]['y'])
    scores = {edge: predict(cal[edge[0]]['x'], fit) for edge, fit in edges.items()}
    ranks = {e: score_ranks(s) for e, s in scores.items()}
    cuts = {e: cutoffs(s) for e, s in scores.items()}
    best = np.full(n+1, np.inf)
    policies = [None]*(n+1)
    counts, frozen = {}, {}

    def add(seq, cost, quality, start=0):
        c = np.asarray(cost).ravel()/n
        q = np.rint(np.asarray(quality).ravel()).astype(int)
        counts[str(len(seq))] += len(c)
        # Policies exceeding the largest budget cannot be selected.
        feasible = c <= budgets[-1]+1e-12
        indices = np.flatnonzero(feasible)
        local = np.full(n+1, np.inf)
        np.minimum.at(local, q[indices], c[indices])
        changed = np.flatnonzero(local < best)
        if not len(changed):
            return
        eligible = indices[c[indices] == local[q[indices]]]
        witness = np.full(n+1, len(c), int)
        np.minimum.at(witness, q[eligible], eligible)
        for value in changed:
            idx = int(witness[value])
            stages = []
            if len(seq) == 2:
                edge = tuple(seq[:2])
                stages = [dict(predictor=edges[edge], threshold=float(cuts[edge][idx]))]
            elif len(seq) == 3:
                ia, ib = np.unravel_index(idx, np.shape(cost))
                for edge, rank in ((tuple(seq[:2]), start+ia), (tuple(seq[1:]), ib)):
                    stages.append(dict(predictor=edges[edge], threshold=float(cuts[edge][rank])))
            policies[value] = dict(models=list(seq), stages=stages,
                                   cal_cost=float(local[value]), cal_quality=float(value/n))
        best[changed] = local[changed]

    for depth in (1, 2, 3):
        counts[str(depth)] = 0
        for seq in combinations(models, depth):
            a = seq[0]
            c, q = cal[a]['c'].sum(), cal[a]['y'].sum()
            if depth == 1:
                add(seq, [c], [q])
                continue
            b = seq[1]
            ra, na = ranks[a, b]
            cb = c + prefix_1d(ra, na, cal[b]['c'])
            qb = q + prefix_1d(ra, na, cal[b]['y']-cal[a]['y'])
            if depth == 2:
                add(seq, cb, qb)
                continue
            z = seq[2]
            rb, nb = ranks[b, z]
            cc = prefix_2d(ra, na, rb, nb, cal[z]['c'])
            qq = prefix_2d(ra, na, rb, nb, cal[z]['y']-cal[b]['y'])
            for start in range(0, na+1, 128):
                stop = min(start+128, na+1)
                add(seq, cb[start:stop, None]+cc[start:stop], qb[start:stop, None]+qq[start:stop], start)
        frozen[str(depth)] = []
        for budget in budgets:
            feasible = np.flatnonzero(best <= budget+1e-12)
            if not len(feasible):
                raise ValueError('No feasible policy')
            frozen[str(depth)].append(policies[int(feasible[-1])])
    return frozen, counts


def run(dataset, out, splits):
    raw, costs, rows, _, fingerprint = prepare(dataset, response=False)
    if len(raw) != 8:
        raise ValueError(f'{dataset}: expected 8 models')
    print(f'{dataset}: loaded {len(rows)} complete rows and eight models', flush=True)
    for split in range(splits):
        cal_idx, test_idx = make_split(len(rows), next(iter(raw.values())).correct.values, SEED+split)
        def sliced(idx):
            return {m: dict(x=df[BASE_SCORERS].to_numpy(dtype=float)[idx],
                            y=df.correct.to_numpy(dtype=float)[idx], c=costs[m][idx]) for m, df in raw.items()}
        cal = sliced(cal_idx)
        low = min(cal, key=lambda m: (cal[m]['c'].mean(), m))
        high = min(cal, key=lambda m: (-cal[m]['y'].mean(), cal[m]['c'].mean(), m))
        budgets = cal[low]['c'].mean()+FRACTIONS*(cal[high]['c'].mean()-cal[low]['c'].mean())
        for arm in ARMS:
            start = time.perf_counter()
            frozen, counts = select(cal, budgets, arm)
            for depth in ('1', '2', '3'):
                unique = {json.dumps(encode(p), sort_keys=True): p for p in frozen[depth]}
                for p in unique.values():
                    np.testing.assert_allclose(evaluate(p, cal)[:2], [p['cal_cost'], p['cal_quality']], atol=1e-12)
            dest = out/dataset/arm
            dest.mkdir(parents=True, exist_ok=True)
            manifest = dict(dataset=dataset, arm=arm, split=split, seed=SEED+split, data_sha256=fingerprint,
                pool=list(raw), features=BASE_SCORERS, cal_idx=rows[cal_idx].tolist(), test_idx=rows[test_idx].tolist(),
                budgets=budgets.tolist(), frozen=frozen, candidate_counts=counts)
            with (dest/f'split_{split:02d}.json').open('x') as handle:
                json.dump(encode(manifest), handle, allow_nan=False)
            # Freeze selection before exposing any held-out rows to evaluation.
            test, memo, records = sliced(test_idx), {}, []
            for j, budget in enumerate(budgets):
                r = dict(dataset=dataset, arm=arm, split=split, fraction=FRACTIONS[j], budget=budget)
                for depth in ('1', '2', '3'):
                    p = frozen[depth][j]
                    key = json.dumps(encode(p), sort_keys=True)
                    if key not in memo:
                        memo[key] = evaluate(p, test)
                    c, q, calls = memo[key]
                    r.update({f's{depth}_{k}': v for k, v in dict(cost=c, accuracy=q, mean_calls=calls,
                        depth=len(p['models']), cal_cost=p['cal_cost'], cal_accuracy=p['cal_quality'], overshoot=c-budget).items()})
                if r['s3_cal_accuracy'] < r['s2_cal_accuracy']-1e-12:
                    raise AssertionError('Nested calibration classes violated')
                records.append(r)
            pd.DataFrame(records).to_csv(dest/f'split_{split:02d}.csv', index=False, mode='x')
            print(f'{dataset} {split+1}/{splits} {arm}: {time.perf_counter()-start:.1f}s', flush=True)


def report(out, datasets, splits):
    records = []
    for d in datasets:
        for arm in ARMS:
            for split in range(splits):
                f = pd.read_csv(out/d/arm/f'split_{split:02d}.csv')
                r = dict(dataset=d, arm=arm, split=split)
                for depth in (1, 2, 3):
                    r[f's{depth}_accuracy'] = f[f's{depth}_accuracy'].mean()
                    r[f's{depth}_cost'] = f[f's{depth}_cost'].mean()
                    r[f's{depth}_gain_over_s1_pp'] = 100*(f[f's{depth}_accuracy']-f.s1_accuracy).mean()
                    r[f's{depth}_overshoot_fraction'] = (f[f's{depth}_overshoot'] > 1e-12).mean()
                r['s3_minus_s2_pp'] = 100*(f.s3_accuracy-f.s2_accuracy).mean()
                r['s3_minus_s2_cost_per_1000'] = 1000*(f.s3_cost-f.s2_cost).mean()
                r['s3_depth3_fraction'] = (f.s3_depth == 3).mean()
                records.append(r)
    frame = pd.DataFrame(records)
    frame.to_csv(out/'split_summary.csv', index=False, mode='x')
    frame.groupby(['dataset', 'arm']).mean(numeric_only=True).to_csv(out/'summary.csv', mode='x')
    comparisons = []
    for d in datasets:
        new = frame[(frame.dataset == d)&(frame.arm == 'diff01_ridge')].set_index('split')
        for ref in ('correctness_ridge', 'mean_token_negentropy'):
            old = frame[(frame.dataset == d)&(frame.arm == ref)].set_index('split')
            for depth in (2, 3):
                delta = 100*(new[f's{depth}_accuracy']-old[f's{depth}_accuracy'])
                comparisons.append(dict(dataset=d, reference=ref, depth=depth, accuracy_pp=delta.mean(),
                    p10_pp=delta.quantile(.1), p90_pp=delta.quantile(.9),
                    cost_per_1000=1000*(new[f's{depth}_cost']-old[f's{depth}_cost']).mean()))
    pd.DataFrame(comparisons).to_csv(out/'comparisons.csv', index=False, mode='x')
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--splits', type=int, default=50)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--out', type=Path, default=Path('results/next_model_diff01'))
    args = parser.parse_args()
    if not 1 <= args.splits <= 50:
        parser.error('splits must be between 1 and 50')
    args.out.mkdir(parents=True, exist_ok=False)
    sources = args.out/'executed_source'
    sources.mkdir()
    for source in SOURCES:
        shutil.copy2(source, sources/source)
    (args.out/'provenance.json').write_text(json.dumps(dict(splits=args.splits, datasets=args.datasets,
        source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in SOURCES},
        regression='standardized ridge, alpha=1, unpenalized intercept, no clipping',
        target='U_next-U_current, negative prediction as score',
        citation='https://arxiv.org/abs/2307.02764'), indent=2))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run, d, args.out, args.splits) for d in args.datasets]
        for job in jobs:
            job.result()
    report(args.out, args.datasets, args.splits)
