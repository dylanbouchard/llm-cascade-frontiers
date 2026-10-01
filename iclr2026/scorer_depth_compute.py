"""Scorer robustness: exact S2 versus S3, with frozen held-out evaluation.

Run .venv/bin/python scorer_depth_compute.py [datasets] --splits 50
No model calls. Existing AUROC and main-depth caches are read-only.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/scorer-depth-mpl')
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from exact_heldout_depth import exact_select, evaluate_policy, encode, FRACTIONS
from fig2_compute import load_full, make_split, SEED
from voi_compute import BASE_SCORERS, ALL_SCORERS, RESPONSE_SCORER, DATASETS, load_response_embeddings

ROOT = Path(__file__).resolve().parent
VERSION = 'scorer-depth-v1'


def prepare(dataset, response=True):
    raw, costs = load_full(dataset)
    if not raw:
        raise ValueError(f'No data for {dataset}')
    prompts = next(iter(raw.values())).prompt.values
    valid = np.ones(len(prompts), bool)
    correct = valid.copy()
    for m, df in raw.items():
        if not np.array_equal(prompts, df.prompt.values):
            raise ValueError(f'Misaligned prompts: {m}')
        correct &= df.correct.notna().values
        valid &= np.isfinite(df[BASE_SCORERS + ['correct']]).all(axis=1).values
        valid &= np.isfinite(costs[m])
        if (costs[m] < 0).any():
            raise ValueError('Negative costs')
    rows = np.flatnonzero(valid)
    embeddings = load_response_embeddings(dataset, list(raw), rows,
                    np.flatnonzero(correct), len(prompts)) if response else {}
    raw = {m: df.iloc[rows].reset_index(drop=True) for m, df in raw.items()}
    costs = {m: np.asarray(c)[rows] for m, c in costs.items()}
    digest = hashlib.sha256()
    for m, df in raw.items():
        digest.update(m.encode())
        digest.update(pd.util.hash_pandas_object(df[['prompt', 'correct'] + BASE_SCORERS], index=False).values.tobytes())
        digest.update(costs[m].tobytes())
        if m in embeddings:
            if not np.isfinite(embeddings[m]).all():
                raise ValueError(f'Nonfinite embeddings: {m}')
            digest.update(embeddings[m].tobytes())
    return raw, costs, rows, embeddings, digest.hexdigest()


def score_models(raw, embeddings, cal_idx, scorer):
    """Fit one correctness scorer per model on calibration labels only."""
    scores, fitted = {}, {}
    for m, df in raw.items():
        if scorer in BASE_SCORERS:
            scores[m] = df[scorer].values.astype(float)
            continue
        y = df.correct.values[cal_idx].astype(int)
        if len(np.unique(y)) < 2:
            scores[m] = df.mean_token_negentropy.values.astype(float)
            fitted[m] = {'fallback': 'mean_token_negentropy'}
            continue
        x = embeddings[m] if scorer == RESPONSE_SCORER else df[BASE_SCORERS].values.astype(float)
        lr = LogisticRegression(C=1., max_iter=1000, solver='lbfgs', random_state=SEED)
        lr.fit(x[cal_idx], y)
        if lr.n_iter_.max() >= lr.max_iter:
            raise RuntimeError(f'Unconverged scorer: {m}/{scorer}')
        scores[m] = lr.predict_proba(x)[:, 1]
        fitted[m] = dict(coef=lr.coef_.tolist(), intercept=lr.intercept_.tolist(), classes=lr.classes_.tolist())
    return scores, fitted


def run(dataset, out, splits, scorers):
    raw, costs, rows, embeddings, fingerprint = prepare(dataset, RESPONSE_SCORER in scorers)
    sources = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in (
        'scorer_depth_compute.py', 'exact_heldout_depth.py', 'oracle_depth_analysis.py',
        'optuna_frontier.py', 'fig2_compute.py', 'voi_compute.py'))).hexdigest()
    print(f'{dataset}: {len(rows)} rows, {len(raw)} models', flush=True)
    for split in range(splits):
        cal_idx, test_idx = make_split(len(rows), next(iter(raw.values())).correct.values, SEED+split)
        low = min(raw, key=lambda m: (costs[m][cal_idx].mean(), m))
        high = min(raw, key=lambda m: (-raw[m].correct.values[cal_idx].mean(), costs[m][cal_idx].mean(), m))
        c0, c1 = costs[low][cal_idx].mean(), costs[high][cal_idx].mean()
        budgets = c0 + FRACTIONS*(c1-c0)
        for scorer in scorers:
            path = out/dataset/scorer/f'split_{split:02d}.csv'
            manifest_path = path.with_suffix('.json')
            if path.exists():
                old = json.loads(manifest_path.read_text())
                if (old['version'], old['data_sha256']) != (VERSION, fingerprint) or sources not in [old['source_sha256'], *old.get('compatible_source_sha256', [])]:
                    raise ValueError(f'Incompatible cache: {path}')
                continue
            start = time.perf_counter()
            scores, fitted = score_models(raw, embeddings, cal_idx, scorer)
            def sliced(idx):
                return {m: dict(scores=scores[m][idx], correct=raw[m].correct.values[idx].astype(float),
                                costs=costs[m][idx]) for m in raw}
            cal = sliced(cal_idx)
            frozen, counts = exact_select(cal, budgets)
            # Replay selected witnesses on calibration before saving or testing.
            for depth in ('2', '3'):
                for p in {json.dumps(encode(p), sort_keys=True): p for p in frozen[depth]}.values():
                    c, q, _ = evaluate_policy(p, cal)
                    np.testing.assert_allclose([c, q], [p['cal_cost'], p['cal_quality']], atol=1e-12)
            manifest = dict(version=VERSION, source_sha256=sources, data_sha256=fingerprint,
                dataset=dataset, scorer=scorer, split=split, seed=SEED+split, pool=list(raw),
                cal_idx=rows[cal_idx].tolist(), test_idx=rows[test_idx].tolist(),
                fractions=FRACTIONS.tolist(), budgets=budgets.tolist(),
                candidate_counts=counts, fitted_scorers=fitted, frozen=frozen)
            path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(encode(manifest), allow_nan=False)+'\n')
            test, memo, records = sliced(test_idx), {}, []
            for j, budget in enumerate(budgets):
                record = dict(dataset=dataset, scorer=scorer, split=split, fraction=FRACTIONS[j], budget=budget)
                for depth in ('2', '3'):
                    p = frozen[depth][j]
                    key = json.dumps(encode(p), sort_keys=True)
                    if key not in memo:
                        memo[key] = evaluate_policy(p, test)
                    c, q, calls = memo[key]
                    record.update({f'd{depth}_{k}': v for k, v in dict(cost=c, accuracy=q, mean_calls=calls,
                        selected_depth=len(p['models']), cal_cost=p['cal_cost'], cal_accuracy=p['cal_quality'],
                        overshoot=c-budget).items()})
                if record['d3_cal_accuracy'] < record['d2_cal_accuracy']-1e-12:
                    raise AssertionError('Nested calibration accuracy violated')
                records.append(record)
            temp = path.with_suffix('.tmp')
            pd.DataFrame(records).to_csv(temp, index=False)
            temp.replace(path)
            print(f'{dataset}/{scorer}: {split+1}/{splits}, {time.perf_counter()-start:.1f}s', flush=True)


def report(out, datasets, splits, scorers):
    records = []
    trap = getattr(np, 'trapezoid', np.trapz)
    for dataset in datasets:
        for scorer in scorers:
            for split in range(splits):
                f = pd.read_csv(out/dataset/scorer/f'split_{split:02d}.csv')
                records.append(dict(dataset=dataset, scorer=scorer, split=split,
                    accuracy_pp=100*trap(f.d3_accuracy-f.d2_accuracy, f.fraction),
                    cost_per_1000=1000*trap(f.d3_cost-f.d2_cost, f.fraction),
                    depth3_fraction=trap((f.d3_selected_depth==3).astype(float), f.fraction),
                    s2_overshoot_fraction=(f.d2_overshoot>1e-12).mean(),
                    s3_overshoot_fraction=(f.d3_overshoot>1e-12).mean()))
    frame = pd.DataFrame(records)
    frame.to_csv(out/'split_integrals.csv', index=False)
    summary = frame.groupby(['dataset', 'scorer']).agg(
        splits=('split', 'count'), accuracy_pp=('accuracy_pp', 'mean'),
        p10_pp=('accuracy_pp', lambda x: x.quantile(.1)), p90_pp=('accuracy_pp', lambda x: x.quantile(.9)),
        cost_per_1000=('cost_per_1000', 'mean'), depth3_fraction=('depth3_fraction', 'mean'),
        s2_overshoot_fraction=('s2_overshoot_fraction', 'mean'), s3_overshoot_fraction=('s3_overshoot_fraction', 'mean'))
    summary.to_csv(out/'summary.csv')
    print(summary.to_string(), flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets', nargs='*', default=DATASETS)
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--scorers', nargs='+', default=ALL_SCORERS, choices=ALL_SCORERS)
    p.add_argument('--out', type=Path, default=Path('results/scorer_depth'))
    p.add_argument('--report-only', action='store_true')
    p.add_argument('--workers', type=int, default=1)
    args = p.parse_args()
    if any(d not in DATASETS for d in args.datasets):
        p.error(f'datasets must be drawn from {DATASETS}')
    if not 1 <= args.splits <= 50:
        p.error('--splits must be in [1, 50]')
    if args.workers < 1:
        p.error('--workers must be positive')
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run, dataset, args.out, args.splits, args.scorers) for dataset in args.datasets]
            for job in jobs:
                job.result()
    report(args.out, args.datasets, args.splits, args.scorers)
