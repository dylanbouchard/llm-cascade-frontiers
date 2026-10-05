"""Exact calibration search over nested depth 1/2/3 classes, then frozen test.

Run from the repository root. No model calls and no manuscript writes.
"""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from fig2_compute import load_full, make_split, slice_data, SCORER, SEED
from oracle_depth_analysis import prefix_1d, prefix_2d, score_ranks, DATASETS
from optuna_frontier import simulate_cascade

VERSION = 'exact-heldout-depth-v1'
FRACTIONS = np.linspace(0., 1., 500)


def exact_select(cal, budgets):
    """Enumerate all calibration routing partitions, retaining exact witnesses.

    Binary accuracy permits lossless reduction to the cheapest policy at each
    integer number correct. Ties retain enumeration order (shallower first).
    Threshold r is unique_score[r], with explicit never/always endpoints.
    """
    models = sorted(cal, key=lambda m: (cal[m]['costs'].mean(), m))
    n = len(cal[models[0]]['correct'])
    for d in cal.values():
        if not np.isin(d['correct'], [0, 1]).all():
            raise ValueError('Exact accuracy reduction requires binary correctness')
    ranks = {m: score_ranks(cal[m]['scores']) for m in models}
    cuts = {m: np.r_[-np.inf, np.unique(cal[m]['scores'])[1:], np.inf] for m in models}
    best_cost = np.full(n + 1, np.inf)
    best_policy = [None] * (n + 1)
    counts = {1: 0, 2: 0, 3: 0}
    frozen = {}

    def add(seq, costs, quality, start=0):
        c = np.asarray(costs).ravel() / n
        q = np.rint(np.asarray(quality).ravel()).astype(int)
        counts[len(seq)] += len(c)
        local = np.full(n + 1, np.inf)
        np.minimum.at(local, q, c)
        improved = np.flatnonzero(local < best_cost)
        if not len(improved):
            return
        eligible = np.flatnonzero(c == local[q])
        witness = np.full(n + 1, len(c), dtype=int)
        np.minimum.at(witness, q[eligible], eligible)
        for value in improved:
            idx = int(witness[value])
            if len(seq) == 1:
                taus = []
            elif len(seq) == 2:
                taus = [float(cuts[seq[0]][idx])]
            else:
                a, b = np.unravel_index(idx, np.shape(costs))
                taus = [float(cuts[seq[0]][start+a]), float(cuts[seq[1]][b])]
            best_policy[value] = {'models': list(seq), 'thresholds': taus,
                                  'cal_cost': float(local[value]), 'cal_quality': value/n}
        best_cost[improved] = local[improved]

    for depth in (1, 2, 3):
        for seq in combinations(models, depth):
            a = seq[0]
            c = cal[a]['costs'].sum()
            q = cal[a]['correct'].sum()
            if depth == 1:
                add(seq, [c], [q])
                continue
            b = seq[1]
            ra, na = ranks[a]
            cb = c + prefix_1d(ra, na, cal[b]['costs'])
            qb = q + prefix_1d(ra, na, cal[b]['correct']-cal[a]['correct'])
            if depth == 2:
                add(seq, cb, qb)
                continue
            rb, nb = ranks[b]
            z = seq[2]
            cc = prefix_2d(ra, na, rb, nb, cal[z]['costs'])
            qq = prefix_2d(ra, na, rb, nb, cal[z]['correct']-cal[b]['correct'])
            for start in range(0, na+1, 128):
                stop = min(start+128, na+1)
                add(seq, cb[start:stop, None]+cc[start:stop],
                    qb[start:stop, None]+qq[start:stop], start)
        selected = []
        for budget in budgets:
            feasible = np.flatnonzero(best_cost <= budget + 1e-12)
            selected.append(best_policy[int(feasible[-1])] if len(feasible) else None)
        frozen[str(depth)] = selected
    return frozen, counts


def evaluate_policy(policy, data):
    if policy is None:
        return (np.nan,)*3
    seq = policy['models']
    taus = [(-np.inf if t == 'never' else np.inf if t == 'always' else float(t))
            for t in policy['thresholds']]
    cost, quality = simulate_cascade(seq, taus, data)
    active = np.ones(len(data[seq[0]]['scores']), dtype=bool)
    calls = active.astype(int)
    for m, tau in zip(seq[:-1], taus):
        active &= data[m]['scores'] < tau
        calls += active
    return cost, quality, float(calls.mean())


def encode(obj):
    """Strict JSON with explicit infinity sentinels for deployable endpoints."""
    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [encode(v) for v in obj]
    if isinstance(obj, float) and np.isinf(obj):
        return 'always' if obj > 0 else 'never'
    return obj


def run_dataset(dataset, out, splits):
    raw, costs = load_full(dataset)
    if not raw:
        raise ValueError(f'No data for {dataset}')
    prompts = next(iter(raw.values()))['prompt'].values
    valid = np.ones(len(prompts), dtype=bool)
    for m, df in raw.items():
        if not np.array_equal(prompts, df['prompt'].values):
            raise ValueError('Misaligned query rows')
        valid &= np.isfinite(df[SCORER]) & np.isfinite(df['correct']) & np.isfinite(costs[m])
        if (costs[m] < 0).any():
            raise ValueError('Negative costs')
    rows = np.flatnonzero(valid)
    raw = {m: df.iloc[rows].reset_index(drop=True) for m, df in raw.items()}
    costs = {m: np.asarray(c)[rows] for m, c in costs.items()}
    digest = hashlib.sha256()
    for m, df in raw.items():
        digest.update(m.encode())
        digest.update(pd.util.hash_pandas_object(df[['prompt', SCORER, 'correct']], index=False).values.tobytes())
        digest.update(costs[m].tobytes())
    fingerprint = digest.hexdigest()
    dest = out / dataset
    dest.mkdir(parents=True, exist_ok=True)
    for split in range(splits):
        path = dest / f'split_{split:02d}.csv'
        manifest_path = path.with_suffix('.json')
        if path.exists():
            old = json.loads(manifest_path.read_text())
            if old['version'] != VERSION or old['data_sha256'] != fingerprint:
                raise ValueError(f'Incompatible cache {path}')
            continue
        cal_idx, test_idx = make_split(len(rows), next(iter(raw.values()))['correct'].values, SEED+split)
        cal = slice_data(raw, costs, list(raw), cal_idx)
        low = min(cal, key=lambda m: (cal[m]['costs'].mean(), m))
        high = min(cal, key=lambda m: (-cal[m]['correct'].mean(), cal[m]['costs'].mean(), m))
        c0, c1 = cal[low]['costs'].mean(), cal[high]['costs'].mean()
        budgets = c0 + FRACTIONS*(c1-c0)
        frozen, counts = exact_select(cal, budgets)
        manifest = dict(version=VERSION, dataset=dataset, split=split, seed=SEED+split,
                        data_sha256=fingerprint, scorer=SCORER, pool=list(raw),
                        pool_rule='full available pool with existing dataset exclusions',
                        cal_idx=rows[cal_idx].tolist(), test_idx=rows[test_idx].tolist(),
                        low=low, high=high, fractions=FRACTIONS.tolist(), budgets=budgets.tolist(),
                        candidate_counts=counts, frozen=frozen)
        manifest_text = json.dumps(encode(manifest), indent=2, allow_nan=False)
        manifest_path.write_text(manifest_text+'\n')
        # No test data passed to search. Save the entire selection before slicing test.
        test = slice_data(raw, costs, list(raw), test_idx)
        records = []
        memo = {}
        for j, budget in enumerate(budgets):
            record = dict(dataset=dataset, split=split, budget_index=j,
                          fraction=FRACTIONS[j], budget=budget)
            for depth in (1, 2, 3):
                p = frozen[str(depth)][j]
                key = None if p is None else (tuple(p['models']), tuple(p['thresholds']))
                if key not in memo:
                    memo[key] = evaluate_policy(p, test)
                c, q, calls = memo[key]
                for metric, value in dict(cost=c, accuracy=q, mean_calls=calls,
                        selected_depth=len(p['models']) if p else np.nan,
                        cal_cost=p['cal_cost'] if p else np.nan,
                        cal_accuracy=p['cal_quality'] if p else np.nan,
                        overshoot=c-budget, feasible=p is not None).items():
                    record[f'd{depth}_{metric}'] = value
            for a, b in ((2, 1), (3, 2), (3, 1)):
                for metric in ('accuracy', 'cost'):
                    record[f'd{a}_minus_d{b}_{metric}'] = record[f'd{a}_{metric}']-record[f'd{b}_{metric}']
            records.append(record)
        pd.DataFrame(records).to_csv(path, index=False)
        print(f'{dataset}: split {split+1}/{splits}, candidates {sum(counts.values()):,}', flush=True)


def report(out, datasets, splits):
    frame = pd.concat([pd.read_csv(out/d/f'split_{s:02d}.csv') for d in datasets for s in range(splits)])
    frame.to_csv(out/'split_metrics.csv', index=False)
    metrics = [c for c in frame if c.startswith('d') and c != 'dataset']
    frame[metrics] = frame[metrics].astype(float)
    # Repeated splits overlap. Quantiles describe split variability, not confidence intervals.
    grouped = frame.groupby(['dataset', 'budget_index', 'fraction'])[metrics]
    def p10(x):
        return x.quantile(.1)
    def p90(x):
        return x.quantile(.9)
    grouped.agg(['mean', 'median', p10, p90]).to_csv(out/'by_budget.csv')
    split_means = frame.groupby(['dataset', 'split'])[metrics].mean()
    split_means.to_csv(out/'split_budget_means.csv')
    summary = split_means.groupby('dataset').mean()
    for depth in (1, 2, 3):
        for selected in range(1, depth+1):
            summary[f'd{depth}_depth{selected}_fraction'] = frame.assign(hit=frame[f'd{depth}_selected_depth']==selected).groupby('dataset').hit.mean()
    summary.to_csv(out/'summary.csv')
    lines = ['# Exact held-out depth results', '',
             f'{splits} splits per dataset, 500 prespecified relative budget positions. '
             'All differences compare depth at most 3 against depth at most 2.', '',
             '| Dataset | Mean accuracy difference (pp) | Mean cost difference ($/1,000 queries) | Depth 3 selected (%) |',
             '|---|---:|---:|---:|']
    for dataset, row in summary.iterrows():
        lines.append(f"| {dataset} | {100*row['d3_minus_d2_accuracy']:.4f} | "
                     f"{1000*row['d3_minus_d2_cost']:.6f} | {100*row['d3_depth3_fraction']:.2f} |")
    lines += ['', 'These are equal-weight averages over splits and budget positions. '
              'Realized test costs can differ and overshooting policies are retained. '
              'Selected depth is chain length, not mean calls. '
              'The by-budget CSV reports split medians and 10th and 90th percentiles. '
              'Overlapping splits do not provide independent confidence intervals.', '',
              'The full available pool is used with existing dataset exclusions. '
              'Exactness covers calibration routing partitions under the documented threshold convention. '
              'See EXACT_HELDOUT_DEPTH.md for the protocol. No manuscript files were updated.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', default=DATASETS)
    parser.add_argument('--splits', type=int, default=50)
    parser.add_argument('--out', type=Path, default=Path('results/exact_heldout_depth'))
    args = parser.parse_args()
    if any(d not in DATASETS for d in args.datasets):
        parser.error(f'datasets must be drawn from {DATASETS}')
    if not 1 <= args.splits <= 50:
        parser.error('--splits must be between 1 and 50')
    for dataset in args.datasets:
        run_dataset(dataset, args.out, args.splits)
    report(args.out, args.datasets, args.splits)
