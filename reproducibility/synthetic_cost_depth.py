"""Exact S1/S2/S3 comparisons with dataset-specific synthetic model prices.

Run --prepare-only to write all price and invariant manifests without selection.
Beta prices take precedence. Budgets and calibration cost order are recomputed
per setting. No model generation or source-cache writes.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/synthetic-cost-mpl')

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import beta

from cascade_core import PRICE_PER_TOKEN, DATA_DIR, _enc
from exact_heldout_depth import FRACTIONS, encode, evaluate_policy, exact_select
from fig2_compute import SEED, make_split
from manuscript_sources import DATASETS, EXPECTED_MODELS
from scorer_depth_compute import prepare

ROOT = Path(__file__).resolve().parent
VERSION = 'synthetic-cost-depth-v1'
SETTINGS = {'OBS': None, 'S--': (.25, .25), 'S-': (.5, .5),
            'U': (1., 1.), 'S+': (2., 2.), 'S++': (4., 4.),
            'L+': (1., 2.), 'L++': (1., 4.), 'R+': (2., 1.), 'R++': (4., 1.)}
CONFIRMATION = ('OBS', 'S--', 'S++')
SOURCES = ('synthetic_cost_depth.py', 'cascade_core.py', 'scorer_depth_compute.py',
           'fig2_compute.py', 'exact_heldout_depth.py', 'oracle_depth_analysis.py',
           'optuna_frontier.py', 'voi_compute.py', 'manuscript_sources.py')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(encode(value), indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def cost_structure(observed, setting):
    """Return target means, quantile positions and uniform price multipliers."""
    observed = np.asarray(observed, dtype=float)
    if not np.isfinite(observed).all() or observed[0] <= 0 or np.any(np.diff(observed) <= 0):
        raise ValueError('Observed mean costs must be finite, positive and strictly increasing')
    ratio = observed[-1] / observed[0]
    t = np.linspace(0., 1., len(observed))
    if setting == 'OBS':
        target = observed.copy()
        u = np.log(observed / observed[0]) / np.log(ratio)
    else:
        u = beta.ppf(t, *SETTINGS[setting])
        u[[0, -1]] = (0., 1.)
        target = observed[0] * ratio ** u
        # Exact endpoint assignment also keeps endpoint prices byte-identical.
        target[[0, -1]] = observed[[0, -1]]
    np.testing.assert_allclose(target[[0, -1]], observed[[0, -1]], rtol=1e-14, atol=0)
    if not np.all(np.diff(target) > 0):
        raise ValueError('Synthetic costs are not strictly increasing')
    return target, u, target / observed


def fit_observed(observed):
    """Least squares over all K log-cost positions, in positive log parameters."""
    _, u, _ = cost_structure(observed, 'OBS')
    t = np.linspace(0., 1., len(u))
    def residual(log_ab):
        return beta.ppf(t, *np.exp(log_ab)) - u
    fits = [least_squares(residual, np.log(start), bounds=(-9., 9.),
                          ftol=1e-12, xtol=1e-12, gtol=1e-12)
            for start in ((1., 1.), (.25, .25), (4., 4.), (1., 4.), (4., 1.))]
    best = min((f for f in fits if f.success), key=lambda f: np.dot(f.fun, f.fun))
    a, b = np.exp(best.x)
    return dict(a=float(a), b=float(b), residuals=best.fun.tolist(),
                sse=float(np.dot(best.fun, best.fun)), rmse=float(np.sqrt(np.mean(best.fun**2))),
                fitted_u=beta.ppf(t, a, b).tolist(), log_parameter_bounds=[-9., 9.])


def recompute_costs(tokens_in, tokens_out, prices):
    """Prices are dollars per token, matching the existing cost computation."""
    return tokens_in * prices[:, 0] + tokens_out * prices[:, 1]


def schedule(models, correct, costs, idx):
    means = {m: costs[idx, j].mean() for j, m in enumerate(models)}
    quality = {m: correct[idx, j].mean() for j, m in enumerate(models)}
    order = sorted(models, key=lambda m: (means[m], m))
    low = order[0]
    high = min(models, key=lambda m: (-quality[m], means[m], m))
    anchors = [float(means[low]), float(means[high])]
    return dict(order=order, low=low, high=high, anchors=anchors), anchors[0] + FRACTIONS * (anchors[1] - anchors[0])


def prepare_dataset(dataset, out, splits):
    raw, observed_costs, rows, _, fingerprint = prepare(dataset, response=False)
    models = list(raw)  # Preserve the reference model used for stratification.
    if set(models) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: full eight-model pool required')
    correct = np.column_stack([raw[m].correct.to_numpy(float) for m in models])
    scores = np.column_stack([raw[m].mean_token_negentropy.to_numpy(float) for m in models])
    tokens_in = np.column_stack([raw[m].prompt.apply(lambda p: len(_enc.encode(p))).to_numpy() for m in models])
    tokens_out = np.column_stack([raw[m].logprob.apply(len).to_numpy() for m in models])
    prices = np.array([PRICE_PER_TOKEN[m] for m in models])
    costs = recompute_costs(tokens_in, tokens_out, prices)
    for j, m in enumerate(models):
        assert costs[:, j].tobytes() == observed_costs[m].tobytes(), f'OBS reconstruction differs: {m}'
    order = sorted(range(len(models)), key=lambda j: (costs[:, j].mean(), models[j]))
    means = np.array([costs[:, j].mean() for j in order])
    arrays = dict(rows=rows, models=np.array(models), correct=correct, scores=scores,
                  tokens_in=tokens_in, tokens_out=tokens_out, costs_OBS=costs, fractions=FRACTIONS)
    selections = [('oracle', np.arange(len(rows)), np.arange(len(rows)), None)]
    for split in range(splits):
        cal, test = make_split(len(rows), correct[:, 0], SEED + split)
        selections.append((f'split_{split:02d}', cal, test, SEED + split))
    audit = {}
    for name, cal, test, seed in selections:
        arrays[f'{name}_cal'] = cal
        arrays[f'{name}_test'] = test
        obs, budget = schedule(models, correct, costs, cal)
        arrays[f'{name}_OBS_budgets'] = budget
        audit[name] = dict(seed=seed, cal_rows_sha256=array_sha(rows[cal]),
                           test_rows_sha256=array_sha(rows[test]), observed=obs, settings={})
    settings = {}
    failures = []
    for setting in SETTINGS:
        target, u, ordered_mult = cost_structure(means, setting)
        multiplier = np.empty(len(models))
        multiplier[order] = ordered_mult
        new_prices = prices * multiplier[:, None]
        new_costs = recompute_costs(tokens_in, tokens_out, new_prices)
        np.testing.assert_allclose([new_costs[:, j].mean() for j in order], target, rtol=2e-14, atol=0)
        assert np.all(np.diff([new_costs[:, j].mean() for j in order]) > 0)
        arrays[f'costs_{setting}'] = new_costs
        settings[setting] = dict(ab=SETTINGS[setting], u=u.tolist(), target_means=target.tolist(),
            multipliers=multiplier.tolist(), prices_per_token=new_prices.tolist(),
            prices_per_million=(new_prices * 1e6).tolist(), costs_sha256=array_sha(new_costs))
        for name, cal, test, seed in selections:
            current, budget = schedule(models, correct, new_costs, cal)
            obs = audit[name]['observed']
            same_budget = budget.tobytes() == arrays[f'{name}_OBS_budgets'].tobytes()
            checks = dict(budgets_byte_identical=same_budget, order_identical=current['order'] == obs['order'],
                          low_identical=current['low'] == obs['low'], high_identical=current['high'] == obs['high'])
            audit[name]['settings'][setting] = dict(**current, **checks, budgets_sha256=array_sha(budget))
            if name == 'oracle' or setting in CONFIRMATION:
                failures.extend(f'{name}/{setting}: {key}' for key, ok in checks.items() if not ok)
            arrays[f'{name}_{setting}_budgets'] = budget
    metadata = dict(version=VERSION, dataset=dataset, models=models, cost_order=[models[j] for j in order],
        observed_means=means.tolist(), observed_u=cost_structure(means, 'OBS')[1].tolist(),
        observed_prices_per_token=prices.tolist(), ratio=float(means[-1] / means[0]),
        fitted_beta=fit_observed(means), settings=settings, splits=splits, seed=SEED,
        scorer='mean_token_negentropy', depths=[1, 2, 3],
        array_layout='prices and multipliers: models order; u and target_means: cost_order',
        token_convention='existing compute_costs: cl100k_base prompt length and recorded logprob length',
        source_fingerprint=fingerprint, sources={p: sha(ROOT / p) for p in SOURCES},
        input_files={str(DATA_DIR / f'{dataset}-{m}.parquet'): sha(DATA_DIR / f'{dataset}-{m}.parquet') for m in models},
        invariant_fingerprints={k: array_sha(arrays[k]) for k in ('rows', 'correct', 'scores', 'tokens_in', 'tokens_out', 'fractions')},
        audit=audit, strict_failures=failures)
    dest = out / dataset
    dest.mkdir(parents=True, exist_ok=True)
    # Reject stale preparation rather than silently replacing an evaluated run.
    manifest = dest / 'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) == json.loads(json.dumps(metadata)):
            with np.load(dest / 'inputs.npz') as cached:
                for key, value in arrays.items():
                    assert array_sha(cached[key]) == array_sha(value), f'Corrupt input: {key}'
            return metadata
        if any(dest.glob('*/*.json')) or any(dest.glob('*/*.csv')):
            raise ValueError(f'Incompatible preparation: {manifest}. Use a fresh output directory.')
    np.savez_compressed(dest / 'inputs.npz', **arrays)
    write_json(manifest, metadata)
    print(f'{dataset}: prepared {len(rows)} rows, {len(failures)} strict invariant failures', flush=True)
    return metadata


def run_cell(dataset, setting, name, out, protocol):
    dest = out / dataset
    metadata = json.loads((dest / 'manifest.json').read_text())
    if metadata['sources'] != {p: sha(ROOT / p) for p in SOURCES}:
        raise ValueError('Source files changed after preparation')
    if protocol == 'strict' and metadata['strict_failures']:
        raise ValueError(f'{dataset}: incompatible requested invariants, inspect manifest.json')
    path = dest / setting / f'{name}.csv'
    selection_path = path.with_suffix('.json')
    identity = dict(version=VERSION, preparation_sha256=sha(dest / 'manifest.json'),
                    inputs_sha256=sha(dest / 'inputs.npz'), protocol=protocol)
    if path.exists():
        old = json.loads(selection_path.read_text())
        if old['identity'] != identity or old['results_sha256'] != sha(path):
            raise ValueError(f'Incompatible result: {path}')
        return
    with np.load(dest / 'inputs.npz') as z:
        models = z['models'].tolist()
        rows, correct, scores, costs = z['rows'], z['correct'], z['scores'], z[f'costs_{setting}']
        cal_idx, test_idx = z[f'{name}_cal'], z[f'{name}_test']
        budgets = z[f'{name}_{setting}_budgets']
        for key in metadata['invariant_fingerprints']:
            assert array_sha(z[key]) == metadata['invariant_fingerprints'][key]
        assert array_sha(costs) == metadata['settings'][setting]['costs_sha256']
    def sliced(idx):
        return {m: dict(scores=scores[idx, j], correct=correct[idx, j], costs=costs[idx, j])
                for j, m in enumerate(models)}
    cal = sliced(cal_idx)
    frozen, counts = exact_select(cal, budgets)
    for policies in frozen.values():
        for policy in {json.dumps(encode(p), sort_keys=True): p for p in policies}.values():
            if policy is None:
                raise AssertionError('No feasible policy at a budget anchor')
            c, q, _ = evaluate_policy(policy, cal)
            np.testing.assert_allclose([c, q], [policy['cal_cost'], policy['cal_quality']], atol=1e-12, rtol=0)
    manifest = dict(identity=identity, dataset=dataset, setting=setting, mode=name,
        seed=metadata['audit'][name]['seed'], cal_idx=rows[cal_idx].tolist(), test_idx=rows[test_idx].tolist(),
        fractions=FRACTIONS.tolist(), budgets=budgets.tolist(), candidate_counts=counts, frozen=frozen,
        prices=metadata['settings'][setting], fitted_beta=metadata['fitted_beta'],
        sources=metadata['sources'], invariant_audit=metadata['audit'][name]['settings'][setting])
    # Selection and complete price provenance are durable before test replay.
    write_json(selection_path, manifest)
    test = sliced(test_idx)
    records, memo = [], {}
    for j, budget in enumerate(budgets):
        record = dict(dataset=dataset, setting=setting, mode=name, fraction=FRACTIONS[j], budget=budget)
        for depth in (1, 2, 3):
            p = frozen[str(depth)][j]
            key = json.dumps(encode(p), sort_keys=True)
            if key not in memo:
                memo[key] = evaluate_policy(p, test)
            c, q, calls = memo[key]
            record.update({f'd{depth}_{k}': v for k, v in dict(cost=c, accuracy=q, mean_calls=calls,
                selected_depth=len(p['models']), cal_cost=p['cal_cost'], cal_accuracy=p['cal_quality'],
                overshoot=c-budget).items()})
        assert record['d3_cal_accuracy'] + 1e-12 >= record['d2_cal_accuracy'] >= record['d1_cal_accuracy'] - 1e-12
        records.append(record)
    temp = path.with_suffix('.tmp')
    pd.DataFrame(records).to_csv(temp, index=False)
    temp.replace(path)
    manifest['results_sha256'] = sha(path)
    write_json(selection_path, manifest)
    print(f'{dataset}/{setting}/{name}: complete', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--out', type=Path, default=Path('results/synthetic_cost_depth'))
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--mode', choices=('oracle', 'confirmation', 'both'), default='both')
    p.add_argument('--protocol', choices=('strict', 'setting-specific'), default='setting-specific',
                   help='setting-specific explicitly permits changed budgets and calibration order')
    args = p.parse_args()
    if not 1 <= args.splits <= 50 or args.workers < 1:
        p.error('splits must be in [1, 50] and workers must be positive')
    prepared = [prepare_dataset(d, args.out, args.splits) for d in args.datasets]
    if args.prepare_only:
        return
    if args.protocol == 'strict' and any(m['strict_failures'] for m in prepared):
        p.error('Requested budget/order invariants fail. Inspect dataset manifest.json files before choosing a revised protocol.')
    jobs = []
    for d in args.datasets:
        if args.mode in ('oracle', 'both'):
            jobs.extend((d, s, 'oracle') for s in SETTINGS)
        if args.mode in ('confirmation', 'both'):
            jobs.extend((d, s, f'split_{i:02d}') for s in CONFIRMATION for i in range(args.splits))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_cell, d, s, name, args.out, args.protocol) for d, s, name in jobs]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
