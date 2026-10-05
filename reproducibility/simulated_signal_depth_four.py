"""Extend frozen AUROC counterfactuals to S4 without altering S1/S2/S3 caches."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from exact_heldout_depth import encode, evaluate_policy
from exact_heldout_four import select_four, policy_key, COST_TIE
from exact_four_model import kernel
from manuscript_sources import EXPECTED_MODELS
from simulated_signal_depth import OUT as BASE, DATASETS, TARGETS

OUT = Path('results/simulated_signal_depth_four')
VERSION = 'synthetic-depth-four-v1'


def source_hash():
    return hashlib.sha256(b''.join(Path(p).read_bytes() for p in (
        'simulated_signal_depth_four.py', 'exact_heldout_four.py',
        'exact_four_model.py', 'exact_four_kernel.cpp', 'exact_heldout_depth.py',
        'optuna_frontier.py'))).hexdigest()


def run(dataset, target, split, out=OUT):
    started = time.monotonic()
    inp = BASE / dataset / 'inputs.npz'
    base = BASE / dataset / f'auroc_{target:.1f}' / f'split_{split:02d}.json'
    original = json.loads(base.read_text())
    fingerprint = hashlib.sha256(inp.read_bytes()).hexdigest()
    hashes = dict(input_sha256=fingerprint, source_sha256=source_hash(),
                  baseline_sha256=hashlib.sha256(base.read_bytes()).hexdigest(),
                  baseline_csv_sha256=hashlib.sha256(base.with_suffix('.csv').read_bytes()).hexdigest())
    if original['input_sha256'] != fingerprint or set(original['pool']) != EXPECTED_MODELS:
        raise ValueError(f'Incompatible eight-model baseline: {base}')
    dest = out / dataset / f'auroc_{target:.1f}' / base.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.with_suffix('.csv').exists():
        prior = json.loads(dest.read_text())
        if prior['version'] != VERSION or any(prior[k] != v for k, v in hashes.items()):
            raise ValueError(f'Incompatible output: {dest}')
        return dataset, target, split, 'cached'
    with np.load(inp) as z:
        rows = z['rows']; models = z['models'].tolist()
        correct = z['correct']; costs = z['costs']; scores = z[f'scores_{target:.1f}']
    assert models == original['pool']
    def sliced(indices):
        idx = np.searchsorted(rows, indices)
        np.testing.assert_array_equal(rows[idx], indices)
        return {m: dict(scores=scores[idx,j], correct=correct[idx,j], costs=costs[idx,j])
                for j,m in enumerate(models)}
    cal = sliced(original['cal_idx'])
    for policies in original['frozen'].values():
        for p in {policy_key(p): p for p in policies}.values():
            c,q,_ = evaluate_policy(p, cal)
            np.testing.assert_allclose([c,q], [p['cal_cost'],p['cal_quality']], rtol=0, atol=1e-12)
    selected, stats = select_four(cal, original['budgets'], original['frozen']['3'])
    for p in {policy_key(p): p for p in selected}.values():
        c,q,_ = evaluate_policy(p, cal)
        np.testing.assert_allclose([c,q], [p['cal_cost'],p['cal_quality']], rtol=0, atol=1e-12)
    manifest = dict(original, version=VERSION, **{k:v for k,v in hashes.items() if k not in original})
    manifest.update(hashes)
    manifest.update(baseline_path=str(base), search_stats=stats, cost_tie_tolerance=COST_TIE,
                    frozen=dict(original['frozen'], **{'4': selected}))
    temp = dest.with_suffix('.json.tmp')
    temp.write_text(json.dumps(encode(manifest), allow_nan=False)+'\n')
    temp.replace(dest)
    test = sliced(original['test_idx'])
    frame = pd.read_csv(base.with_suffix('.csv'))
    np.testing.assert_allclose(frame.budget, original['budgets'], rtol=0, atol=1e-15)
    memo = {}
    def replay(p):
        key = policy_key(p)
        if key not in memo:
            memo[key] = evaluate_policy(p, test)
        return memo[key]
    for depth in (1,2,3):
        vals = np.array([replay(p) for p in original['frozen'][str(depth)]])
        np.testing.assert_allclose(vals, frame[[f'd{depth}_cost',f'd{depth}_accuracy',f'd{depth}_mean_calls']], rtol=0, atol=1e-12)
    vals = np.array([replay(p) for p in selected])
    for j,metric in enumerate(('cost','accuracy','mean_calls')):
        frame[f'd4_{metric}'] = vals[:,j]
    frame['d4_selected_depth'] = [len(p['models']) for p in selected]
    frame['d4_cal_cost'] = [p['cal_cost'] for p in selected]
    frame['d4_cal_accuracy'] = [p['cal_quality'] for p in selected]
    frame['d4_overshoot'] = frame.d4_cost - frame.budget
    assert (frame.d4_cal_accuracy >= frame.d3_cal_accuracy-1e-12).all()
    assert (frame.d4_cal_cost <= frame.budget+1e-12).all()
    temp = dest.with_suffix('.csv.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(dest.with_suffix('.csv'))
    seconds = time.monotonic()-started
    print(f'{dataset} AUROC {target:.1f} split {split:02d}: {seconds:.1f}s, '
          f'{stats["quadruples"]} quadruples, '
          f'S4-S3 {100*np.trapz(frame.d4_accuracy-frame.d3_accuracy,frame.fraction):+.4f} pp', flush=True)
    return dataset, target, split, seconds


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    p.add_argument('--targets', nargs='+', type=float, choices=TARGETS, default=list(TARGETS))
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()
    if not 1 <= args.splits <= 50 or args.workers < 1:
        p.error('Use 1 to 50 splits and positive workers')
    kernel()  # Compile once before workers start.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run,d,t,s) for s in range(args.splits)
                for t in reversed(args.targets) for d in args.datasets]
        for job in as_completed(jobs):
            job.result()
