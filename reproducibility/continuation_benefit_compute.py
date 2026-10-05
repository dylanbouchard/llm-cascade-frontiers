"""Focused continuation-aware depth ablation using stored model generations.

Run from the repository root. No model calls or changes to existing caches.
Exact first-stage sweeps, with a prespecified grid of second-stage cutoffs.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/continuation-benefit-mpl')
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from scorer_depth_compute import prepare, BASE_SCORERS
from fig2_compute import make_split, SEED
from exact_heldout_depth import FRACTIONS, encode

ROOT = Path(__file__).resolve().parent
VERSION = 'continuation-benefit-v1'
ARMS = ('correctness_ridge', 'continuation_ridge')
SOURCES = ('continuation_benefit_compute.py', 'scorer_depth_compute.py',
           'exact_heldout_depth.py', 'fig2_compute.py', 'voi_compute.py',
           'optuna_frontier.py', 'oracle_depth_analysis.py')


def fit_linear(x, y, alpha=1.):
    """Standardized ridge, unpenalized intercept, coefficients in raw units."""
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.)
    z = (x-mean)/scale
    center = y.mean(axis=0)
    w = np.linalg.solve(z.T@z + alpha*np.eye(x.shape[1]), z.T@(y-center))
    coef = w/scale if y.ndim == 1 else w/scale[:, None]
    return coef, center-mean@coef


def predictor(coef, intercept, sign=1.):
    return dict(coef=(sign*coef).tolist(), intercept=float(sign*intercept))


def predict(x, model):
    return x@np.asarray(model['coef']) + model['intercept']


def oof_prediction(x, y, seed):
    out = np.empty(len(y))
    for train, valid in KFold(n_splits=5, shuffle=True, random_state=seed).split(x):
        coef, intercept = fit_linear(x[train], y[train])
        out[valid] = x[valid]@coef+intercept
    return out


def cutoffs(score):
    u = np.unique(score)
    return np.r_[-np.inf, u[1:], np.inf]


def suffix_grid(score, count):
    cuts = cutoffs(score)
    if count == 0 or len(cuts) <= count:
        return cuts
    ranks = np.unique(np.rint(np.linspace(0, len(cuts)-1, count)).astype(int))
    return cuts[ranks]


def sweep(score, base_cost, base_quality, suffix_cost, suffix_quality, current_quality):
    """Every strict score < threshold routing partition, with ties together."""
    order = np.argsort(score, kind='stable')
    sorted_score = score[order]
    ends = np.r_[0, np.flatnonzero(np.diff(sorted_score) != 0)+1, len(score)]
    cs = np.r_[0., np.cumsum(suffix_cost[order])]
    qs = np.r_[0., np.cumsum((suffix_quality-current_quality)[order])]
    return (base_cost+cs[ends])/len(score), np.rint(base_quality+qs[ends]).astype(int), cutoffs(score)


def evaluate(policy, data):
    active = np.ones(len(next(iter(data.values()))['y']), bool)
    total_cost = np.zeros(len(active))
    answer = np.zeros(len(active))
    calls = np.zeros(len(active))
    for i, name in enumerate(policy['models']):
        total_cost[active] += data[name]['c'][active]
        calls[active] += 1
        if i == len(policy['models'])-1:
            stop = active.copy()
        else:
            stage = policy['stages'][i]
            threshold = stage['threshold']
            if isinstance(threshold, str):
                threshold = -np.inf if threshold == 'never' else np.inf
            stop = active & (predict(data[name]['x'], stage['predictor']) >= threshold)
        answer[stop] = data[name]['y'][stop]
        active &= ~stop
    return float(total_cost.mean()), float(answer.mean()), float(calls.mean())


def select(cal, budgets, arm, seed, grid_count=21):
    """Fit on calibration only, then select the nested pair/triple libraries.

    Continuation B predicts U_C-U_B from B's five scores. For each frozen
    B cutoff t, A predicts U_suffix(t)-U_A from A's five scores. The training
    suffix outcome uses out-of-fold B predictions. Enumeration evaluates the
    deployed, full-calibration B fit. This is a fitted policy library, not an
    optimum over every possible scoring function or unrestricted cascade.
    """
    if arm not in ARMS:
        raise ValueError(arm)
    models = sorted(cal, key=lambda m: (cal[m]['c'].mean(), m))
    n = len(cal[models[0]]['y'])
    own, edge, edge_oof = {}, {}, {}
    for m in models:
        if not np.isin(cal[m]['y'], [0, 1]).all():
            raise ValueError('Binary correctness required')
        own[m] = predictor(*fit_linear(cal[m]['x'], cal[m]['y']))
    if arm == 'continuation_ridge':
        for a, b in combinations(models, 2):
            target = cal[b]['y']-cal[a]['y']
            edge[a, b] = predictor(*fit_linear(cal[a]['x'], target), sign=-1.)
            edge_oof[a, b] = -oof_prediction(cal[a]['x'], target, seed)
    best = np.full(n+1, np.inf)
    policies = [None]*(n+1)
    counts = {'1': 0, '2': 0, '3': 0}
    frozen = {}

    def add(seq, costs, quality, cuts=None, first=None, second=None):
        counts[str(len(seq))] += len(costs)
        local = np.full(n+1, np.inf)
        np.minimum.at(local, quality, costs)
        changed = np.flatnonzero(local < best)
        witness = np.full(n+1, len(costs), int)
        eligible = np.flatnonzero(costs == local[quality])
        np.minimum.at(witness, quality[eligible], eligible)
        for q in changed:
            index = witness[q]
            stages = [] if len(seq) == 1 else [dict(predictor=first, threshold=float(cuts[index]))]
            if second is not None:
                stages.append(second)
            policies[q] = dict(models=list(seq), stages=stages,
                               cal_cost=float(local[q]), cal_quality=float(q/n))
        best[changed] = local[changed]

    for depth in (1, 2, 3):
        for seq in combinations(models, depth):
            a = seq[0]
            base_c, base_q = cal[a]['c'].sum(), cal[a]['y'].sum()
            if depth == 1:
                add(seq, np.array([base_c/n]), np.array([int(base_q)]))
                continue
            b = seq[1]
            if depth == 2:
                first = own[a] if arm == 'correctness_ridge' else edge[a, b]
                c, q, cuts = sweep(predict(cal[a]['x'], first), base_c, base_q,
                                   cal[b]['c'], cal[b]['y'], cal[a]['y'])
                add(seq, c, q, cuts, first)
                continue
            z = seq[2]
            second = own[b] if arm == 'correctness_ridge' else edge[b, z]
            score_b = predict(cal[b]['x'], second)
            grid = suffix_grid(score_b, grid_count)
            if arm == 'continuation_ridge':
                target = cal[b]['y'][:, None] + (cal[z]['y']-cal[b]['y'])[:, None]*(edge_oof[b, z][:, None] < grid)
                target -= cal[a]['y'][:, None]
                coef, intercept = fit_linear(cal[a]['x'], target)
            for j, threshold_b in enumerate(grid):
                first = own[a] if arm == 'correctness_ridge' else predictor(coef[:, j], intercept[j], sign=-1.)
                continuation = score_b < threshold_b
                suffix_c = cal[b]['c'] + continuation*cal[z]['c']
                suffix_q = np.where(continuation, cal[z]['y'], cal[b]['y'])
                c, q, cuts = sweep(predict(cal[a]['x'], first), base_c, base_q,
                                   suffix_c, suffix_q, cal[a]['y'])
                add(seq, c, q, cuts, first, dict(predictor=second, threshold=float(threshold_b)))
        selected = []
        for budget in budgets:
            feasible = np.flatnonzero(best <= budget+1e-12)
            if not len(feasible):
                raise ValueError('Infeasible budget')
            selected.append(policies[feasible[-1]])
        frozen[str(depth)] = selected
    return frozen, counts


def run(dataset, out, splits, grid_count):
    raw, costs, rows, _, fingerprint = prepare(dataset, response=False)
    source_hash = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in SOURCES)).hexdigest()
    print(f'{dataset}: {len(rows)} rows, {len(raw)} models', flush=True)
    for split in range(splits):
        cal_idx, test_idx = make_split(len(rows), next(iter(raw.values())).correct.values, SEED+split)
        def sliced(idx):
            return {m: dict(x=df[BASE_SCORERS].values[idx].astype(float),
                            y=df.correct.values[idx].astype(float), c=costs[m][idx]) for m, df in raw.items()}
        cal = sliced(cal_idx)
        low = min(cal, key=lambda m: (cal[m]['c'].mean(), m))
        high = min(cal, key=lambda m: (-cal[m]['y'].mean(), cal[m]['c'].mean(), m))
        budgets = cal[low]['c'].mean()+FRACTIONS*(cal[high]['c'].mean()-cal[low]['c'].mean())
        # Verify exact alignment with the published scorer ablation.
        reference = ROOT/'results/scorer_depth'/dataset/'mean_token_negentropy'/f'split_{split:02d}.json'
        old = json.loads(reference.read_text())
        assert old['cal_idx'] == rows[cal_idx].tolist()
        assert old['test_idx'] == rows[test_idx].tolist()
        np.testing.assert_allclose(old['budgets'], budgets, rtol=0, atol=1e-15)
        for arm in ARMS:
            dest = out/dataset/arm/f'split_{split:02d}.csv'
            manifest_path = dest.with_suffix('.json')
            if dest.exists():
                previous = json.loads(manifest_path.read_text())
                if (previous['version'], previous['source_sha256'], previous['data_sha256'], previous['grid_count']) != (VERSION, source_hash, fingerprint, grid_count):
                    raise ValueError(f'Incompatible cache {dest}')
                continue
            start = time.perf_counter()
            frozen, counts = select(cal, budgets, arm, SEED+split, grid_count)
            for depth in ('2', '3'):
                for policy in {json.dumps(encode(p), sort_keys=True): p for p in frozen[depth]}.values():
                    c, q, _ = evaluate(policy, cal)
                    np.testing.assert_allclose([c, q], [policy['cal_cost'], policy['cal_quality']], rtol=0, atol=1e-12)
            manifest = dict(version=VERSION, source_sha256=source_hash, data_sha256=fingerprint,
                dataset=dataset, arm=arm, split=split, seed=SEED+split, grid_count=grid_count,
                features=BASE_SCORERS, ridge_alpha=1., crossfit_folds=5,
                cal_idx=rows[cal_idx].tolist(), test_idx=rows[test_idx].tolist(),
                budgets=budgets.tolist(), candidate_counts=counts, frozen=frozen)
            dest.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(encode(manifest), allow_nan=False)+'\n')
            test = sliced(test_idx)
            records, memo = [], {}
            for j, budget in enumerate(budgets):
                record = dict(dataset=dataset, scorer=arm, split=split, fraction=FRACTIONS[j], budget=budget)
                for depth in ('2', '3'):
                    policy = frozen[depth][j]
                    key = json.dumps(encode(policy), sort_keys=True)
                    if key not in memo:
                        memo[key] = evaluate(policy, test)
                    c, q, calls = memo[key]
                    record.update({f'd{depth}_{k}': v for k, v in dict(cost=c, accuracy=q,
                        mean_calls=calls, selected_depth=len(policy['models']), cal_cost=policy['cal_cost'],
                        cal_accuracy=policy['cal_quality'], overshoot=c-budget).items()})
                assert record['d3_cal_accuracy'] >= record['d2_cal_accuracy']-1e-12
                records.append(record)
            pd.DataFrame(records).to_csv(dest.with_suffix('.tmp'), index=False)
            dest.with_suffix('.tmp').replace(dest)
            print(f'{dataset}/{arm} split {split+1}/{splits}: {time.perf_counter()-start:.2f}s', flush=True)


def report(out, datasets, splits):
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    records = []
    for dataset in datasets:
        for arm in (*ARMS, 'mean_token_negentropy', 'logreg_ensemble'):
            root = out if arm in ARMS else ROOT/'results/scorer_depth'
            for split in range(splits):
                f = pd.read_csv(root/dataset/arm/f'split_{split:02d}.csv')
                record = dict(dataset=dataset, scorer=arm, split=split)
                metrics = dict(accuracy_pp=100*(f.d3_accuracy-f.d2_accuracy),
                    cal_accuracy_pp=100*(f.d3_cal_accuracy-f.d2_cal_accuracy),
                    cost_per_1000=1000*(f.d3_cost-f.d2_cost),
                    cost_budget_pct=100*(f.d3_cost-f.d2_cost)/f.budget,
                    triple_pct=100*(f.d3_selected_depth==3),
                    over_half_pp_pct=100*((f.d3_accuracy-f.d2_accuracy)>.005),
                    calls_difference=f.d3_mean_calls-f.d2_mean_calls)
                for depth in ('2', '3'):
                    metrics.update({f'd{depth}_accuracy_pp':100*f[f'd{depth}_accuracy'],
                                    f'd{depth}_cost_per_1000':1000*f[f'd{depth}_cost'],
                                    f'd{depth}_overshoot_pct':100*(f[f'd{depth}_overshoot']>1e-12)})
                record.update({k:float(trap(v, f.fraction)) for k,v in metrics.items()})
                records.append(record)
    frame = pd.DataFrame(records)
    frame.to_csv(out/'split_integrals.csv', index=False)
    summary = frame.groupby(['dataset','scorer']).mean(numeric_only=True).drop(columns='split')
    summary['p10_pp'] = frame.groupby(['dataset','scorer']).accuracy_pp.quantile(.1)
    summary['p90_pp'] = frame.groupby(['dataset','scorer']).accuracy_pp.quantile(.9)
    summary.to_csv(out/'summary.csv')
    print(summary[['accuracy_pp','p10_pp','p90_pp','cost_per_1000','cost_budget_pct']].to_string(), flush=True)
    return frame, summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='*', default=['mmlu','triviaqa'])
    parser.add_argument('--splits', type=int, default=50)
    parser.add_argument('--grid-count', type=int, default=21)
    parser.add_argument('--out', type=Path, default=ROOT/'results/continuation_benefit')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.splits <= 50 or args.grid_count not in (0,) and args.grid_count < 2:
        parser.error('Require 1..50 splits and grid-count 0 (all) or >=2')
    args.out.mkdir(parents=True, exist_ok=True)
    snapshot = args.out/'executed_source'
    snapshot.mkdir(exist_ok=True)
    for name in SOURCES:
        target = snapshot/name
        content = (ROOT/name).read_bytes()
        if target.exists() and target.read_bytes() != content:
            raise ValueError(f'Source changed after snapshot: {target}')
        target.write_bytes(content)
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run, d, args.out, args.splits, args.grid_count) for d in args.datasets]
            for job in jobs:
                job.result()
    report(args.out, args.datasets, args.splits)
