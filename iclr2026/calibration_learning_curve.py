"""Fixed-test calibration learning curve using the exact B.8 policy classes."""
from __future__ import annotations
from exact_heldout_depth import (exact_select, evaluate_policy, encode, load_full,
    make_split, slice_data, SCORER, SEED, FRACTIONS, DATASETS)
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

VERSION = 'fixed-test-calibration-v1'
SIZES = (.1, .25, .5, .75, 1.)


def stratified_order(indices, labels, seed):
    """Randomize within strata, interleave to keep every prefix proportional.

    At each step choose the stratum with the largest deficit against its
    target cumulative count. Ties use label order. No other outcomes or costs
    enter this nested-sampling operation.
    """
    rng = np.random.default_rng(seed)
    groups = [rng.permutation(indices[labels[indices] == label]) for label in (0, 1)]
    sizes = np.array([len(g) for g in groups])
    used = np.zeros(2, dtype=int)
    order = []
    for step in range(len(indices)):
        deficit = (step + 1) * sizes / len(indices) - used
        deficit[used == sizes] = -np.inf
        k = int(np.argmax(deficit))
        order.append(groups[k][used[k]])
        used[k] += 1
    return np.asarray(order)


def load_clean(dataset):
    raw, costs = load_full(dataset)
    prompts = next(iter(raw.values()))['prompt'].values
    valid = np.ones(len(prompts), dtype=bool)
    for m, df in raw.items():
        if not np.array_equal(prompts, df['prompt'].values):
            raise ValueError('Misaligned queries')
        valid &= np.isfinite(df[SCORER]) & np.isfinite(df['correct']) & np.isfinite(costs[m])
        if (costs[m] < 0).any():
            raise ValueError('Negative cost')
    rows = np.flatnonzero(valid)
    raw = {m: df.iloc[rows].reset_index(drop=True) for m, df in raw.items()}
    costs = {m: np.asarray(c)[rows] for m, c in costs.items()}
    digest = hashlib.sha256()
    for m, df in raw.items():
        digest.update(m.encode())
        digest.update(pd.util.hash_pandas_object(df[['prompt', SCORER, 'correct']], index=False).values.tobytes())
        digest.update(costs[m].tobytes())
    return raw, costs, rows, digest.hexdigest()


def run_dataset(dataset, out, splits):
    raw, costs, rows, fingerprint = load_clean(dataset)
    dest = out / dataset
    dest.mkdir(parents=True, exist_ok=True)
    labels = next(iter(raw.values()))['correct'].values
    for split in range(splits):
        path = dest / f'split_{split:02d}.csv'
        manifest_path = path.with_suffix('.json')
        if path.exists():
            old = json.loads(manifest_path.read_text())
            assert old['version'] == VERSION and old['data_sha256'] == fingerprint
            continue
        reservoir, test_idx = make_split(len(rows), labels, SEED + split)
        full = slice_data(raw, costs, list(raw), reservoir)
        low = min(full, key=lambda m: (full[m]['costs'].mean(), m))
        high = min(full, key=lambda m: (-full[m]['correct'].mean(), full[m]['costs'].mean(), m))
        c0, c1 = full[low]['costs'].mean(), full[high]['costs'].mean()
        budgets = c0 + FRACTIONS * (c1 - c0)
        del full
        order = stratified_order(reservoir, labels, SEED + split)
        selections = []
        for fraction in SIZES:
            n = max(1, int(np.floor(fraction * len(reservoir))))
            idx = order[:n]
            cal = slice_data(raw, costs, list(raw), idx)
            frozen, counts = exact_select(cal, budgets)
            frozen = {k: frozen[k] for k in ('2', '3')}
            for policies in frozen.values():
                for p in policies:
                    if p is not None:
                        identity = json.dumps(encode([p['models'], p['thresholds']]), separators=(',', ':'))
                        p['candidate_id'] = hashlib.sha256(identity.encode()).hexdigest()
            for p2, p3 in zip(frozen['2'], frozen['3']):
                if p2 is not None:
                    assert p3 is not None and p3['cal_quality'] >= p2['cal_quality']
            selections.append(dict(fraction=fraction, n_cal=n, cal_idx=rows[idx].tolist(),
                model_order=sorted(cal, key=lambda m: (cal[m]['costs'].mean(), m)),
                candidate_counts=counts, frozen=frozen))
        manifest = dict(version=VERSION, data_sha256=fingerprint, dataset=dataset,
            seed=SEED+split, scorer=SCORER, pool=list(raw),
            reservoir_idx=rows[reservoir].tolist(), test_idx=rows[test_idx].tolist(),
            budget_low_model=low, budget_high_model=high, budgets=budgets.tolist(), selections=selections)
        manifest_path.write_text(json.dumps(encode(manifest), allow_nan=False)+'\n')
        # All sizes and budgets have been saved before any test policy evaluation.
        test = slice_data(raw, costs, list(raw), test_idx)
        records, memo = [], {}
        for selection in selections:
            for j, budget in enumerate(budgets):
                record = dict(dataset=dataset, split=split, seed=SEED+split,
                    fraction=selection['fraction'], n_cal=selection['n_cal'], n_test=len(test_idx),
                    n_reservoir=len(reservoir), budget_index=j, budget=budget)
                for depth in (2, 3):
                    p = selection['frozen'][str(depth)][j]
                    key = p['candidate_id'] if p else None
                    if key not in memo:
                        memo[key] = evaluate_policy(p, test)
                    c, q, calls = memo[key]
                    for metric, value in dict(feasible=p is not None, cost=c, accuracy=q,
                        calls=calls, nominal_depth=len(p['models']) if p else np.nan,
                        cal_accuracy=p['cal_quality'] if p else np.nan,
                        cal_cost=p['cal_cost'] if p else np.nan,
                        overshoot=c-budget, relative_overshoot=(c-budget)/budget,
                        candidate_id=key).items():
                        record[f'd{depth}_{metric}'] = value
                records.append(record)
        pd.DataFrame(records).to_csv(path, index=False)
        print(f'{dataset}: {split+1}/{splits}', flush=True)


def aggregate(frame):
    """Pair first, then average the same original budget indices at all sizes."""
    frame = frame.copy()
    frame['both'] = frame.d2_feasible & frame.d3_feasible
    frame['common'] = frame.groupby(['dataset', 'split', 'budget_index']).both.transform('all')
    records = []
    for (dataset, split, fraction), group in frame.groupby(['dataset', 'split', 'fraction']):
        common = group[group.common]
        row = dict(dataset=dataset, split=split, fraction=fraction,
            n_cal=int(group.n_cal.iloc[0]), n_test=int(group.n_test.iloc[0]),
            coverage=group.common.mean(), common_budget_count=len(common))
        row['test_gain_pp'] = 100*(common.d3_accuracy-common.d2_accuracy).mean()
        row['cal_gain_pp'] = 100*(common.d3_cal_accuracy-common.d2_cal_accuracy).mean()
        row['cost_gain_pct'] = (100*(common.d3_cost-common.d2_cost)/common.budget).mean()
        for d in (2, 3):
            row[f'd{d}_infeasible_pct'] = 100*(~group[f'd{d}_feasible']).mean()
            row[f'd{d}_overshoot_pct'] = 100*(common[f'd{d}_overshoot'] > 1e-12).mean() if len(common) else np.nan
            row[f'd{d}_positive_overshoot_pct'] = 100*common[f'd{d}_relative_overshoot'].clip(lower=0).mean()
            row[f'd{d}_calls'] = common[f'd{d}_calls'].mean()
        records.append(row)
    return pd.DataFrame(records)


def report(out, datasets, splits):
    frame = pd.concat([pd.read_csv(out/d/f'split_{s:02d}.csv') for d in datasets for s in range(splits)])
    means = aggregate(frame)
    means.to_csv(out/'split_budget_means.csv', index=False)
    metrics = [c for c in means if c not in ('dataset', 'split', 'fraction', 'n_cal', 'n_test')]
    summary = means.groupby(['dataset', 'fraction', 'n_cal', 'n_test'])[metrics].agg(
        ['mean', lambda x: x.quantile(.1), lambda x: x.quantile(.9)])
    summary.columns = [f'{a}_{dict(zip(["mean", "<lambda_0>", "<lambda_1>"], ["mean", "p10", "p90"]))[b]}' for a,b in summary.columns]
    summary.reset_index().to_csv(out/'summary.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', default=DATASETS)
    parser.add_argument('--splits', type=int, default=50)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--out', type=Path, default=Path('results/calibration_learning_curve'))
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_dataset, d, args.out, args.splits) for d in args.datasets]
        for f in futures:
            f.result()
    report(args.out, args.datasets, args.splits)
