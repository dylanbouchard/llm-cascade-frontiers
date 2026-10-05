"""Exact calibration-selected depth four over the existing 50 held-out splits.

Reuses fingerprint-verified depth 1/2/3 selections without changing their caches.
"""
import argparse
import hashlib
import json
import time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import exact_heldout_depth as baseline
from exact_four_model import search_chain, ROOT

VERSION = 'exact-heldout-four-v1'
COST_TIE = 1e-15


def select_four(cal, budgets, depth_three):
    """Select the union of frozen depth-three optima and every quadruple."""
    n = len(next(iter(cal.values()))['correct'])
    models = sorted(cal, key=lambda m: (cal[m]['costs'].mean(), m))
    cuts = {m: np.r_[-np.inf, np.unique(cal[m]['scores'])[1:], np.inf] for m in models}
    best = np.full(n + 1, np.inf)
    policies = [None] * (n + 1)
    evaluated = 0
    kernel_seconds = 0.
    for seq in combinations(models, 4):
        costs, ranks, count, seconds = search_chain(cal, seq, max(budgets))
        evaluated += count
        kernel_seconds += seconds
        for q in np.flatnonzero(costs < best - COST_TIE):
            best[q] = costs[q]
            policies[q] = dict(models=list(seq),
                               thresholds=[float(cuts[m][r]) for m, r in zip(seq, ranks[q])],
                               cal_cost=float(costs[q]), cal_quality=float(q/n))
    selected = []
    for budget, old in zip(budgets, depth_three):
        feasible = np.flatnonzero(best <= budget + 1e-12)
        new = policies[int(feasible[-1])] if len(feasible) else None
        if new is not None and (old is None or new['cal_quality'] > old['cal_quality'] or
                (new['cal_quality'] == old['cal_quality'] and
                 new['cal_cost'] < old['cal_cost'] - COST_TIE)):
            selected.append(new)
        else:
            selected.append(old)
    return selected, dict(evaluated_candidates=evaluated, kernel_seconds=kernel_seconds,
                          quadruples=len(list(combinations(models, 4))))


def policy_key(p):
    return None if p is None else (tuple(p['models']), tuple(p['thresholds']))


def run_dataset(dataset, out, base, splits):
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
    code_hash = hashlib.sha256((ROOT/'exact_four_kernel.cpp').read_bytes() +
                               (ROOT/'exact_four_model.py').read_bytes() +
                               Path(__file__).read_bytes()).hexdigest()
    dest = out/dataset
    dest.mkdir(parents=True, exist_ok=True)
    for split in range(splits):
        started = time.perf_counter()
        original_path = base/dataset/f'split_{split:02d}.json'
        original_bytes = original_path.read_bytes()
        original = json.loads(original_bytes)
        base_hash = hashlib.sha256(original_bytes).hexdigest()
        if original['version'] != baseline.VERSION or original['data_sha256'] != fingerprint:
            raise ValueError(f'Incompatible baseline {original_path}')
        path = dest/f'split_{split:02d}.json'
        csv_path = path.with_suffix('.csv')
        if csv_path.exists():
            cached = json.loads(path.read_text())
            if (cached['version'], cached['code_sha256'], cached['baseline_sha256']) != (
                    VERSION, code_hash, base_hash):
                raise ValueError(f'Incompatible output {path}')
            continue
        cal = baseline.slice_data(raw, costs, list(raw), np.array(original['cal_idx']))
        selected, stats = select_four(cal, original['budgets'], original['frozen']['3'])
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
        manifest['frozen'] = dict(original['frozen'], **{'4': selected})
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(baseline.encode(manifest), indent=2, allow_nan=False)+'\n')
        temp.replace(path)
        # Only after saving the frozen manifest may held-out data enter evaluation.
        test = baseline.slice_data(raw, costs, list(raw), np.array(original['test_idx']))
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
            records.append(dict(d4_cost=c, d4_accuracy=q, d4_mean_calls=calls,
                                d4_selected_depth=len(p['models']) if p else np.nan,
                                d4_cal_cost=p['cal_cost'] if p else np.nan,
                                d4_cal_accuracy=p['cal_quality'] if p else np.nan,
                                d4_overshoot=c-budget, d4_feasible=p is not None))
        frame = pd.concat([frame, pd.DataFrame(records)], axis=1)
        for depth in (1, 2, 3):
            for metric in ('accuracy', 'cost'):
                frame[f'd4_minus_d{depth}_{metric}'] = frame[f'd4_{metric}'] - frame[f'd{depth}_{metric}']
        if (frame.d4_cal_accuracy + 1e-12 < frame.d3_cal_accuracy).any():
            raise AssertionError('Calibration nesting failed')
        temp_csv = csv_path.with_suffix('.csv.tmp')
        frame.to_csv(temp_csv, index=False)
        temp_csv.replace(csv_path)
        print(f'{dataset} {split+1}/{splits}: {time.perf_counter()-started:.2f}s total, '
              f'{stats["kernel_seconds"]:.2f}s kernel, '
              f'd4-d3 test {100*frame.d4_minus_d3_accuracy.mean():+.4f} pp', flush=True)


def report(out, datasets, splits):
    frame = pd.concat([pd.read_csv(out/d/f'split_{s:02d}.csv')
                       for d in datasets for s in range(splits)], ignore_index=True)
    frame.to_csv(out/'split_metrics.csv', index=False)
    metrics = [c for c in frame if c.startswith('d') and c != 'dataset']
    frame[metrics] = frame[metrics].astype(float)
    means = frame.groupby(['dataset', 'split'])[metrics].mean()
    means.to_csv(out/'split_budget_means.csv')
    summary = means.groupby('dataset').mean()
    for depth in range(1, 5):
        summary[f'd4_depth{depth}_fraction'] = frame.assign(
            hit=frame.d4_selected_depth == depth).groupby('dataset').hit.mean()
    summary.to_csv(out/'summary.csv')
    grouped = frame.groupby(['dataset', 'budget_index', 'fraction'])[metrics]
    def p10(x):
        return x.quantile(.1)
    def p90(x):
        return x.quantile(.9)
    grouped.agg(['mean', 'median', p10, p90]).to_csv(out/'by_budget.csv')
    lines = ['# Exact held-out four-model results', '',
             f'{splits} overlapping calibration-test splits per dataset and 500 assigned budget positions. '
             'Accuracy changes are percentage points. Cost changes are dollars per 1,000 queries.', '',
             '| Dataset | 4 minus 3 accuracy | 4 minus 3 cost | 4 minus 2 accuracy | 4 minus 2 cost | Depth 4 selected (%) |',
             '|---|---:|---:|---:|---:|---:|']
    for dataset in datasets:
        r = summary.loc[dataset]
        lines.append(f'| {dataset} | {100*r.d4_minus_d3_accuracy:+.4f} | '
                     f'{1000*r.d4_minus_d3_cost:+.6f} | {100*r.d4_minus_d2_accuracy:+.4f} | '
                     f'{1000*r.d4_minus_d2_cost:+.6f} | {100*r.d4_depth4_fraction:.2f} |')
    lines += ['', 'Classes are nested and include standalone models and all cost-ordered pairs, triples, '
              'and quadruples from the full available pool with the existing dataset exclusions. '
              'Every policy is selected on calibration data and saved before test evaluation. '
              'Calibration accuracy cannot fall when depth four is allowed. Held-out accuracy can.', '',
              'Averages weight splits and assigned budget positions equally. Test costs are realized '
              'costs and need not match between methods. Overshoots and dominated test points are retained. '
              'Split percentiles describe variability, not independent-sample confidence intervals. '
              'Depth four selected refers to policy chain length, not realized model calls.', '',
              'Existing depth 1/2/3 manifests are data-fingerprint verified and their policy selections '
              'are reused unchanged. The numerical cost tie tolerance for new comparisons is 1e-15 '
              'dollars per query, favoring the existing shallower policy. Budget tolerance is 1e-12 dollars.', '',
              'See EXACT_HELDOUT_FOUR.md for methods, validation, and reproduction.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:12]), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('datasets', nargs='*', default=baseline.DATASETS)
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--out', type=Path, default=ROOT/'results/exact_heldout_four')
    p.add_argument('--baseline', type=Path, default=ROOT/'results/exact_heldout_depth')
    args = p.parse_args()
    if not 1 <= args.splits <= 50 or any(d not in baseline.DATASETS for d in args.datasets):
        p.error('Use 1 to 50 splits and supported datasets')
    for dataset in args.datasets:
        run_dataset(dataset, args.out, args.baseline, args.splits)
    report(args.out, args.datasets, args.splits)
