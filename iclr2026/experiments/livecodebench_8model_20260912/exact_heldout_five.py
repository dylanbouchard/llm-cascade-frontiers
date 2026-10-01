"""Exact held-out depth five with four independent split workers and checkpoints."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import exact_heldout_depth as baseline
from exact_four_model import ROOT
from exact_five_bounded import search_chain
from exact_heldout_four import VERSION as BASE_VERSION

VERSION = 'exact-heldout-five-v1'
COST_TIE = 1e-15
INPUT_CACHE = {}


def source_hash():
    return hashlib.sha256(b''.join((ROOT/name).read_bytes() for name in (
        'exact_heldout_five.py', 'exact_five_bounded.py', 'exact_five_bounded_kernel.cpp',
        'exact_heldout_depth.py', 'optuna_frontier.py'))).hexdigest()


def select_five(cal, budgets, depth_four, checkpoint=None, fingerprint=''):
    """Select the union of depth four and all exact five-model chains."""
    n = len(next(iter(cal.values()))['correct'])
    models = sorted(cal, key=lambda m: (cal[m]['costs'].mean(), m))
    cuts = {m: np.r_[-np.inf, np.unique(cal[m]['scores'])[1:], np.inf] for m in models}
    selected = list(depth_four)
    evaluated = 0
    kernel_seconds = 0.
    pruned_b = pruned_c = 0
    seqs = list(combinations(models, 5))
    if checkpoint is not None:
        checkpoint.mkdir(parents=True, exist_ok=True)
    for index, seq in enumerate(seqs):
        path = checkpoint/f'chain_{index:02d}.npz' if checkpoint is not None else None
        incumbent_hash = hashlib.sha256(json.dumps(baseline.encode(selected), sort_keys=True).encode()).hexdigest()
        if path is not None and path.exists():
            with np.load(path) as z:
                if str(z['fingerprint']) != fingerprint or str(z['incumbent_hash']) != incumbent_hash:
                    raise ValueError(f'Incompatible chain checkpoint {path}')
                costs, ranks, stats = z['costs'], z['ranks'], json.loads(str(z['stats']))
        else:
            costs, ranks, stats = search_chain(cal, seq, max(budgets), 0., selected)
            if not stats['complete']:
                raise AssertionError('Full run requires complete exact chains')
            if path is not None:
                temp = path.with_suffix('.tmp.npz')
                np.savez_compressed(temp, fingerprint=fingerprint, incumbent_hash=incumbent_hash,
                                    costs=costs, ranks=ranks, stats=json.dumps(stats), models=seq)
                temp.replace(path)
        evaluated += stats['evaluated']
        kernel_seconds += stats['seconds']
        pruned_b += stats['pruned_b']
        pruned_c += stats['pruned_c']
        for j, budget in enumerate(budgets):
            feasible = np.flatnonzero(costs <= budget + 1e-12)
            if not len(feasible):
                continue
            q = int(feasible[-1])
            old = selected[j]
            if old is None or q/n > old['cal_quality'] or (q/n == old['cal_quality'] and
                                                        costs[q] < old['cal_cost'] - COST_TIE):
                selected[j] = dict(models=list(seq),
                                   thresholds=[float(cuts[m][r]) for m,r in zip(seq,ranks[q])],
                                   cal_cost=float(costs[q]), cal_quality=float(q/n))
        if checkpoint is not None:
            (checkpoint/'progress.json').write_text(json.dumps(dict(
                completed_chains=index+1, total_chains=len(seqs), kernel_seconds=kernel_seconds,
                updated_unix=time.time(), complete=index+1 == len(seqs)))+'\n')
    return selected, dict(evaluated_candidates=evaluated, kernel_seconds=kernel_seconds,
                          quintuples=len(seqs), pruned_b=pruned_b, pruned_c=pruned_c)


def policy_key(p):
    return None if p is None else (tuple(p['models']), tuple(p['thresholds']))


def prepare_dataset(dataset, out):
    raw, costs = baseline.load_full(dataset)
    prompts = next(iter(raw.values()))['prompt'].values
    valid = np.ones(len(prompts), dtype=bool)
    for m, df in raw.items():
        if not np.array_equal(prompts, df['prompt'].values):
            raise ValueError('Misaligned prompts')
        valid &= np.isfinite(df[baseline.SCORER]) & np.isfinite(df['correct']) & np.isfinite(costs[m])
        if (costs[m] < 0).any():
            raise ValueError('Negative costs')
    rows = np.flatnonzero(valid)
    digest = hashlib.sha256()
    for m, df in raw.items():
        filtered = df.iloc[rows].reset_index(drop=True)
        digest.update(m.encode())
        digest.update(pd.util.hash_pandas_object(
            filtered[['prompt', baseline.SCORER, 'correct']], index=False).values.tobytes())
        digest.update(np.asarray(costs[m])[rows].tobytes())
    fingerprint = digest.hexdigest()
    prepared = out/'inputs'
    prepared.mkdir(parents=True, exist_ok=True)
    temp = prepared/f'{dataset}.tmp.npz'
    np.savez_compressed(temp, models=list(raw), fingerprint=fingerprint,
                        scores=np.array([df[baseline.SCORER].to_numpy(float) for df in raw.values()]),
                        correct=np.array([df['correct'].to_numpy(float) for df in raw.values()]),
                        costs=np.array([costs[m] for m in raw]))
    temp.replace(prepared/f'{dataset}.npz')
    print(f'{dataset}: prepared {len(rows)} jointly valid rows', flush=True)


def run_split(dataset, split, out, base):
    key = str(out/'inputs'/f'{dataset}.npz')
    if key not in INPUT_CACHE:
        with np.load(key) as z:
            INPUT_CACHE[key] = {k: z[k] for k in z.files}
    arrays = INPUT_CACHE[key]
    models = list(arrays['models'])
    fingerprint = str(arrays['fingerprint'])
    code_hash = source_hash()
    dest = out/dataset
    dest.mkdir(parents=True, exist_ok=True)
    def slice_arrays(indices):
        return {m: {k: arrays[k][i, indices] for k in ('scores', 'correct', 'costs')}
                for i, m in enumerate(models)}
    started = time.perf_counter()
    original_path = base/dataset/f'split_{split:02d}.json'
    original_bytes = original_path.read_bytes()
    original = json.loads(original_bytes)
    base_hash = hashlib.sha256(original_bytes).hexdigest()
    if original['version'] != BASE_VERSION or original['data_sha256'] != fingerprint:
        raise ValueError(f'Incompatible baseline {original_path}')
    if set(original['cal_idx']) & set(original['test_idx']):
        raise ValueError('Overlapping calibration and test rows')
    path = dest/f'split_{split:02d}.json'
    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        cached = json.loads(path.read_text())
        if (cached['version'], cached['code_sha256'], cached['baseline_sha256']) != (
                VERSION, code_hash, base_hash):
            raise ValueError(f'Incompatible output {path}')
        return dict(dataset=dataset, split=split, cached=True)
    cal = slice_arrays(np.array(original['cal_idx']))
    checkpoint = dest/f'split_{split:02d}_chains'
    selected, stats = select_five(cal, original['budgets'], original['frozen']['4'],
                                  checkpoint, hashlib.sha256((code_hash+base_hash).encode()).hexdigest())
    for budget, p, old in zip(original['budgets'], selected, original['frozen']['4']):
        if p is not None and p['cal_cost'] > budget + 1e-12:
            raise AssertionError('Infeasible selected calibration policy')
        if old is not None and (p is None or p['cal_quality'] < old['cal_quality']):
            raise AssertionError('Calibration nesting failed before freezing')
    # Replay unique selected calibration policies before freezing.
    seen = set()
    for p in selected:
        key = policy_key(p)
        if key in seen or p is None:
            continue
        seen.add(key)
        c, q, _ = baseline.evaluate_policy(p, cal)
        if abs(c-p['cal_cost']) > 1e-12 or abs(q-p['cal_quality']) > 1e-12:
            raise AssertionError('Calibration witness replay mismatch')
    manifest = dict(original)
    manifest.update(version=VERSION, code_sha256=code_hash, baseline_sha256=base_hash,
                    search_stats=stats, cost_tie_tolerance=COST_TIE)
    manifest['frozen'] = dict(original['frozen'], **{'5': selected})
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(baseline.encode(manifest), indent=2, allow_nan=False)+'\n')
    temp.replace(path)
    # Only after saving the frozen manifest may held-out data enter evaluation.
    test = slice_arrays(np.array(original['test_idx']))
    frame = pd.read_csv(original_path.with_suffix('.csv'))
    if len(frame) != len(selected) or not np.allclose(frame.budget, original['budgets'], rtol=0, atol=1e-15):
        raise ValueError('Baseline CSV/manifest mismatch')
    memo = {}
    records = []
    for budget, p in zip(original['budgets'], selected):
        key = policy_key(p)
        if key not in memo:
            memo[key] = baseline.evaluate_policy(p, test)
        c, q, calls = memo[key]
        records.append(dict(d5_cost=c, d5_accuracy=q, d5_mean_calls=calls,
                            d5_selected_depth=len(p['models']) if p else np.nan,
                            d5_cal_cost=p['cal_cost'] if p else np.nan,
                            d5_cal_accuracy=p['cal_quality'] if p else np.nan,
                            d5_overshoot=c-budget, d5_feasible=p is not None))
    frame = pd.concat([frame, pd.DataFrame(records)], axis=1)
    for depth in (1, 2, 3, 4):
        for metric in ('accuracy', 'cost'):
            frame[f'd5_minus_d{depth}_{metric}'] = frame[f'd5_{metric}'] - frame[f'd{depth}_{metric}']
    if (frame.d5_cal_accuracy + 1e-12 < frame.d4_cal_accuracy).any():
        raise AssertionError('Calibration nesting failed')
    temp_csv = csv_path.with_suffix('.csv.tmp')
    frame.to_csv(temp_csv, index=False)
    temp_csv.replace(csv_path)
    return dict(dataset=dataset, split=split, cached=False, seconds=time.perf_counter()-started,
                kernel_seconds=stats['kernel_seconds'],
                d5_minus_d4_pp=100*frame.d5_minus_d4_accuracy.mean())


def report(out, datasets, splits):
    frame = pd.concat([pd.read_csv(out/d/f'split_{s:02d}.csv')
                       for d in datasets for s in range(splits)], ignore_index=True)
    frame.to_csv(out/'split_metrics.csv', index=False)
    metrics = [c for c in frame if c.startswith('d') and c != 'dataset']
    frame[metrics] = frame[metrics].astype(float)
    means = frame.groupby(['dataset', 'split'])[metrics].mean()
    means.to_csv(out/'split_budget_means.csv')
    summary = means.groupby('dataset').mean()
    for depth in range(1, 6):
        summary[f'd5_depth{depth}_fraction'] = frame.assign(
            hit=frame.d5_selected_depth == depth).groupby('dataset').hit.mean()
    summary.to_csv(out/'summary.csv')
    grouped = frame.groupby(['dataset', 'budget_index', 'fraction'])[metrics]
    pd.concat({'mean': grouped.mean(), 'median': grouped.median(),
               'p10': grouped.quantile(.1), 'p90': grouped.quantile(.9)},
              axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1).to_csv(out/'by_budget.csv')
    lines = ['# Exact held-out five-model results', '',
             f'{splits} overlapping calibration-test splits per dataset and 500 assigned budget positions. '
             'Accuracy changes are percentage points. Cost changes are dollars per 1,000 queries.', '',
             '| Dataset | 5 minus 4 accuracy | 5 minus 4 cost | 5 minus 2 accuracy | 5 minus 2 cost | Depth 5 selected (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for dataset in datasets:
        r = summary.loc[dataset]
        lines.append(f'| {dataset} | {100*r.d5_minus_d4_accuracy:+.4f} | '
                     f'{1000*r.d5_minus_d4_cost:+.6f} | {100*r.d5_minus_d2_accuracy:+.4f} | '
                     f'{1000*r.d5_minus_d2_cost:+.6f} | {100*r.d5_depth5_fraction:.2f} |')
    lines += ['', 'Classes are nested and include standalone models and all cost-ordered pairs, triples, '
              'quadruples and five-model chains from the full available pool with the existing dataset exclusions. '
              'Every policy is selected on calibration data and saved before test evaluation. '
              'Calibration accuracy cannot fall when depth five is allowed. Held-out accuracy can.', '',
              'Averages weight splits and assigned budget positions equally. Test costs are realized '
              'costs and need not match between methods. Overshoots and dominated test points are retained. '
              'Split percentiles describe variability, not independent-sample confidence intervals. '
              'Depth five selected refers to policy chain length, not realized model calls.', '',
              'Existing depth 1/2/3/4 manifests are data-fingerprint verified and their policy selections '
              'are reused unchanged. The numerical cost tie tolerance for new comparisons is 1e-15 '
              'dollars per query, favoring the existing shallower policy. Budget tolerance is 1e-12 dollars.', '',
              'See EXACT_HELDOUT_FIVE.md for methods, validation, and reproduction.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets', nargs='*', default=baseline.DATASETS)
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--out', type=Path, default=ROOT/'results/exact_heldout_five')
    p.add_argument('--baseline', type=Path, default=ROOT/'results/exact_heldout_four')
    args = p.parse_args()
    if not 1 <= args.splits <= 50 or not 1 <= args.workers <= 8 or any(d not in baseline.DATASETS for d in args.datasets):
        p.error('Use 1 to 50 splits, 1 to 8 workers, and supported datasets')
    for dataset in args.datasets:
        prepare_dataset(dataset, args.out)
    started = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_split, dataset, split, args.out, args.baseline)
                   for split in range(args.splits) for dataset in args.datasets]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f'{len(results)}/{len(futures)} complete: {json.dumps(result)}', flush=True)
            temp = args.out/'run_status.tmp.json'
            temp.write_text(json.dumps(dict(completed=len(results), total=len(futures),
                                           workers=args.workers, elapsed_seconds=time.time()-started,
                                           updated_unix=time.time(), last_result=result))+'\n')
            temp.replace(args.out/'run_status.json')
    (args.out/'run_timings.json').write_text(json.dumps(results, indent=2)+'\n')
    report(args.out, args.datasets, args.splits)
