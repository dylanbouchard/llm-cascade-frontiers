"""Repeat the complete price sweep with stored synthetic AUROC 0.9 scores.

Only the confidence array changes from the prepared observed-score experiment.
The existing exact selector, frozen replay, and price reconstruction are reused.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from synthetic_cost_depth import (
    DATASETS, SETTINGS, SOURCES, ROOT, EXPECTED_MODELS, sha, array_sha,
    recompute_costs, run_cell, write_json,
)

TARGET = .9


def prepare(dataset, base, out, score_root):
    source = base/dataset
    parent = json.loads((source/'manifest.json').read_text())
    if parent['sources'] != {p: sha(ROOT/p) for p in SOURCES}:
        raise ValueError(f'{dataset}: parent sources changed')
    with np.load(source/'inputs.npz') as z:
        arrays = {k: z[k] for k in z.files}
    models, rows = arrays['models'].tolist(), arrays['rows']
    if set(models) != EXPECTED_MODELS or parent['splits'] != 50:
        raise ValueError('Requires the full eight-model pool and 50 matched splits')
    for k, fingerprint in parent['invariant_fingerprints'].items():
        assert array_sha(arrays[k]) == fingerprint, f'Parent array mismatch: {k}'
    score_path = score_root/dataset/'auroc_0.9.parquet'
    frame = pd.read_parquet(score_path).set_index('row_index', verify_integrity=True)
    scores = frame.loc[rows, models].to_numpy(float)
    if scores.shape != arrays['scores'].shape or not np.isfinite(scores).all():
        raise ValueError('Missing or nonfinite synthetic scores')
    # Check alignment against the same synthetic scores used in the AUROC experiment.
    signal_path = ROOT/'results/simulated_signal_depth'/dataset/'inputs.npz'
    with np.load(signal_path) as z:
        for key in ('rows', 'models', 'correct'):
            assert z[key].tobytes() == arrays[key].tobytes(), key
        assert z['scores_0.9'].tobytes() == scores.tobytes()
    arrays['scores'] = scores
    for setting, prices in parent['settings'].items():
        reconstructed = recompute_costs(arrays['tokens_in'], arrays['tokens_out'],
                                       np.asarray(prices['prices_per_token']))
        assert reconstructed.tobytes() == arrays[f'costs_{setting}'].tobytes()
        assert array_sha(reconstructed) == prices['costs_sha256']
    diagnostics = [dict(dataset=dataset, model=m, target_auroc=TARGET,
                        achieved_auroc=float(roc_auc_score(arrays['correct'][:,j], scores[:,j])),
                        n_rows=len(rows)) for j,m in enumerate(models)]
    generator_meta_path = score_root/dataset/'metadata.json'
    generator_meta = json.loads(generator_meta_path.read_text())
    meta = dict(parent)
    meta['scorer'] = 'synthetic_auroc_0.9'
    meta['target_auroc'] = TARGET
    meta['invariant_fingerprints'] = dict(parent['invariant_fingerprints'], scores=array_sha(scores))
    meta['score_provenance'] = dict(
        runner_sha256=sha(Path(__file__)), score_file=str(score_path), score_file_sha256=sha(score_path),
        metadata_file=str(generator_meta_path), metadata_sha256=sha(generator_meta_path),
        generation_metadata=generator_meta, matched_signal_inputs_sha256=sha(signal_path),
        parent_manifest_sha256=sha(source/'manifest.json'), parent_inputs_sha256=sha(source/'inputs.npz'),
        parent_source_fingerprint=parent['source_fingerprint'], achieved_aurocs=diagnostics,
        unchanged_arrays={k: array_sha(v) for k,v in arrays.items() if k != 'scores'})
    meta['source_fingerprint'] = hashlib.sha256(json.dumps(meta['score_provenance'], sort_keys=True).encode()).hexdigest()
    dest = out/dataset
    dest.mkdir(parents=True, exist_ok=True)
    manifest = dest/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != meta:
            raise ValueError(f'Incompatible synthetic-score cache: {dest}')
        with np.load(dest/'inputs.npz') as z:
            assert set(z.files) == set(arrays)
            for k,v in arrays.items():
                assert array_sha(z[k]) == array_sha(v), k
    else:
        np.savez_compressed(dest/'inputs.npz', **arrays)
        write_json(manifest, meta)
    pd.DataFrame(diagnostics).to_csv(dest/'input_aurocs.csv', index=False)
    print(f'{dataset}: synthetic AUROC 0.9 aligned, all non-score arrays byte-identical', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=Path('results/synthetic_cost_depth'))
    parser.add_argument('--out', type=Path, default=Path('results/synthetic_cost_auroc09_depth'))
    parser.add_argument('--scores', type=Path, default=Path('results/simulated_confidence'))
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.out.resolve() == args.base.resolve() or args.workers < 1:
        parser.error('Use a separate output directory and positive workers')
    for dataset in DATASETS:
        prepare(dataset, args.base, args.out, args.scores)
    write_json(args.out/'run_manifest.json', dict(target_auroc=TARGET, datasets=list(DATASETS),
        settings=list(SETTINGS), splits=50, oracle=True, runner_sha256=sha(Path(__file__)),
        protocol='setting-specific',
        preparations={d: sha(args.out/d/'manifest.json') for d in DATASETS}))
    if args.prepare_only:
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_cell, d, s, name, args.out, 'setting-specific')
                for d in DATASETS for s in SETTINGS
                for name in ['oracle']+[f'split_{i:02d}' for i in range(50)]]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()
