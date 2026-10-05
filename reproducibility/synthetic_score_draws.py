"""Independent synthetic-score draws with the existing matched depth protocol.

Run from the repository root. Existing single-draw results are read-only.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import simulated_signal_depth as depth
import simulate_correlated_confidence as synthetic
from manuscript_sources import EXPECTED_MODELS

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'results/simulated_signal_depth'
OUT = ROOT / 'results/synthetic_score_draws'
TARGETS = (.8, .9)
VERSION = 'synthetic-score-draws-v1'
SOURCES = ('synthetic_score_draws.py', 'simulate_correlated_confidence.py',
           'simulated_signal_depth.py', 'exact_heldout_depth.py',
           'oracle_depth_analysis.py', 'optuna_frontier.py', 'fig2_compute.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def draw_seed(dataset, draw, seed):
    # Stable streams when adding datasets or draws, with no reuse of legacy seeds.
    return [int(seed), depth.DATASETS.index(dataset), int(draw)]


def prepare_draw(dataset, draw, seed, out):
    dest = out / f'draw_{draw:02d}' / dataset
    dest.mkdir(parents=True, exist_ok=True)
    source = BASE / dataset / 'inputs.npz'
    row_index, models, corr, outcomes = synthetic.load_dataset(dataset)
    if set(models) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: expected the full eight-model pool')
    with np.load(source) as z:
        base = {key: z[key] for key in ('rows', 'models', 'correct', 'costs')}
    order = [models.index(m) for m in base['models'].tolist()]
    if set(base['models'].tolist()) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: legacy inputs omit manuscript models')
    np.testing.assert_array_equal(outcomes[base['rows']][:, order], base['correct'])
    identity = dict(version=VERSION, dataset=dataset, draw=draw,
                    seed_entropy=draw_seed(dataset, draw, seed), targets=list(TARGETS),
                    baseline_input_sha256=sha(source),
                    outcomes_sha256=hashlib.sha256(outcomes.tobytes()).hexdigest(),
                    correlation_sha256=hashlib.sha256(corr.tobytes()).hexdigest(),
                    source_sha256={p: sha(ROOT / p) for p in SOURCES})
    meta = dest / 'draw_metadata.json'
    if meta.exists():
        saved = json.loads(meta.read_text())
        if saved['identity'] != identity:
            raise ValueError(f'Incompatible draw cache: {dest}')
        for filename, digest in saved['artifacts'].items():
            if sha(dest / filename) != digest:
                raise ValueError(f'Corrupt draw artifact: {dest / filename}')
        return
    rng = np.random.default_rng(np.random.SeedSequence(identity['seed_entropy']))
    y, noise, label_corr = synthetic.orthogonal_noise(outcomes, rng)
    diagnostics = []
    artifacts = []
    for target in TARGETS:
        scores, aucs, achieved_corr = synthetic.calibrate(
            target, y, noise, label_corr, corr, outcomes)
        filename = f'auroc_{target:.1f}.parquet'
        frame = pd.DataFrame(scores, columns=models)
        frame.insert(0, 'row_index', row_index)
        frame.to_parquet(dest / filename, index=False)
        artifacts.append(filename)
        aligned = scores[base['rows']][:, order]
        base[f'scores_{target:.1f}'] = aligned
        filtered_corr_error = np.max(np.abs(np.corrcoef(aligned, rowvar=False)
                                            - corr[np.ix_(order, order)]))
        for j, model in enumerate(base['models'].tolist()):
            diagnostics.append(dict(dataset=dataset, draw=draw, target_auroc=target,
                model=model, full_auroc=aucs[models.index(model)],
                retained_auroc=synthetic.empirical_auc(aligned[:, j], base['correct'][:, j]),
                full_correlation_max_error=np.max(np.abs(achieved_corr-corr)),
                retained_correlation_max_error=filtered_corr_error))
    np.savez_compressed(dest / 'inputs.npz', **base)
    pd.DataFrame(diagnostics).to_csv(dest / 'diagnostics.csv', index=False)
    artifacts += ['inputs.npz', 'diagnostics.csv']
    write_json(meta, dict(identity=identity, artifacts={p: sha(dest / p) for p in artifacts}))
    print(f'Prepared {dataset}, draw {draw}', flush=True)


def validate_match(manifest, reference):
    """Exact row, budget, and pool equality to the existing experiment."""
    for key in ('dataset', 'target_auroc', 'split', 'seed', 'pool',
                'cal_idx', 'test_idx', 'budgets', 'low', 'high'):
        if manifest[key] != reference[key]:
            raise ValueError(f'Unmatched {key}: {manifest.get("dataset")}/{manifest.get("split")}')


def run_cell(dataset, draw, target, splits, out):
    # Each worker invokes the unchanged, audited S1/S2/S3 selection and replay.
    depth.OUT = out / f'draw_{draw:02d}'
    depth.run(dataset, target, splits)
    for split in range(splits):
        relative = Path(dataset) / f'auroc_{target:.1f}' / f'split_{split:02d}.json'
        validate_match(json.loads((depth.OUT / relative).read_text()),
                       json.loads((BASE / relative).read_text()))
    return dataset, draw, target


def summarize(integrals, draws, splits):
    """Keep draw means and conditional split dispersion as separate estimands."""
    keys = ['dataset', 'target_auroc', 'draw']
    if integrals.duplicated(keys + ['split']).any():
        raise ValueError('Duplicate draw/split observations')
    records = []
    for identity, cell in integrals.groupby(keys, sort=True):
        if set(cell.split) != set(range(splits)):
            raise ValueError(f'Incomplete splits: {identity}')
        gain = cell.s3_minus_s2_pp
        records.append(dict(zip(keys, identity), n_splits=len(cell),
            mean_gain_pp=gain.mean(), split_sd_pp=gain.std(ddof=1),
            split_p10_pp=gain.quantile(.1), split_p90_pp=gain.quantile(.9)))
    per_draw = pd.DataFrame(records)
    records = []
    for identity, cell in per_draw.groupby(['dataset', 'target_auroc'], sort=True):
        if set(cell.draw) != set(range(draws)):
            raise ValueError(f'Incomplete draws: {identity}')
        means = cell.mean_gain_pp
        records.append(dict(dataset=identity[0], target_auroc=identity[1], n_draws=len(cell),
            mean_gain_pp=means.mean(), across_draw_sd_pp=means.std(ddof=1),
            across_draw_min_pp=means.min(), across_draw_max_pp=means.max(),
            across_draw_p10_pp=means.quantile(.1), across_draw_p90_pp=means.quantile(.9)))
    return per_draw, pd.DataFrame(records)


def report(out, datasets, draws, splits):
    records = []
    for dataset in datasets:
        for draw in range(draws):
            for target in TARGETS:
                dest = out / f'draw_{draw:02d}' / dataset / f'auroc_{target:.1f}'
                for split in range(splits):
                    path = dest / f'split_{split:02d}.csv'
                    manifest = json.loads(path.with_suffix('.json').read_text())
                    reference = json.loads((BASE / dataset / dest.name / path.with_suffix('.json').name).read_text())
                    validate_match(manifest, reference)
                    if manifest['input_sha256'] != sha(dest.parent / 'inputs.npz'):
                        raise ValueError(f'Input fingerprint mismatch: {path}')
                    f = pd.read_csv(path)
                    np.testing.assert_allclose(f.fraction, depth.FRACTIONS, atol=1e-15)
                    np.testing.assert_allclose(f.budget, manifest['budgets'], atol=1e-15)
                    gain = 100 * np.trapz(f.d3_accuracy - f.d2_accuracy, f.fraction)
                    if not np.isfinite(gain):
                        raise ValueError(f'Nonfinite gain: {path}')
                    records.append(dict(dataset=dataset, target_auroc=target, draw=draw,
                                        split=split, s3_minus_s2_pp=gain))
    integrals = pd.DataFrame(records)
    per_draw, across = summarize(integrals, draws, splits)
    integrals.to_csv(out / 'split_integrals.csv', index=False)
    per_draw.to_csv(out / 'per_draw_summary.csv', index=False)
    across.to_csv(out / 'across_draw_summary.csv', index=False)
    lines = ['# Independent synthetic confidence draws', '',
        f'{draws} independent draws per dataset, {splits} matched calibration/test splits per draw, '
        'and 500 assigned budget positions. All gains are held-out S3 minus S2 accuracy '
        'in percentage points, integrated over normalized assigned budgets. '
        'Each draw mean averages equally over splits. Across-draw dispersion is calculated '
        'from those means, without pooling draw/split observations. Shared noise pairs '
        'AUROC 0.8 and 0.9 within each draw.', '',
        '## Variation across draws', '',
        '| Dataset | AUROC | Mean gain | SD across draw means | Min draw mean | Max draw mean |',
        '|---|---:|---:|---:|---:|---:|']
    for r in across.itertuples():
        lines.append(f'| {r.dataset} | {r.target_auroc:.1f} | {r.mean_gain_pp:+.4f} | '
            f'{r.across_draw_sd_pp:.4f} | {r.across_draw_min_pp:+.4f} | {r.across_draw_max_pp:+.4f} |')
    lines += ['', '## Each draw and its split variability', '',
        '| Dataset | AUROC | Draw | Mean gain | Split SD | Split p10 | Split p90 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in per_draw.itertuples():
        lines.append(f'| {r.dataset} | {r.target_auroc:.1f} | {r.draw} | {r.mean_gain_pp:+.4f} | '
            f'{r.split_sd_pp:.4f} | {r.split_p10_pp:+.4f} | {r.split_p90_pp:+.4f} |')
    lines += ['', 'SDs use the sample denominator. Split percentiles describe overlapping '
        'splits conditional on one draw and are not confidence intervals. Across-draw '
        'variation conditions on the recorded queries, labels, costs, correlation target, '
        'and fixed split schedule. It is not dataset sampling uncertainty. Synthetic scores '
        'use full-sample correctness labels. Held-out evaluation applies to policy selection, '
        'not learning a scorer. Comparisons match assigned budgets, not realized test costs.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(across.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=depth.DATASETS, default=list(depth.DATASETS))
    parser.add_argument('--draws', type=int, default=5)
    parser.add_argument('--splits', type=int, default=50)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.draws < 2 or not 2 <= args.splits <= 50 or args.workers < 1:
        parser.error('Require at least two draws, 2 to 50 splits, and positive workers')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    for dataset in args.datasets:
        for draw in range(args.draws):
            if args.report_only and not (args.out / f'draw_{draw:02d}' / dataset / 'draw_metadata.json').exists():
                raise ValueError('Missing prepared draw')
            prepare_draw(dataset, draw, args.seed, args.out)
    if args.prepare_only:
        return
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs = [pool.submit(run_cell, d, r, t, args.splits, args.out)
                    for d in args.datasets for r in range(args.draws) for t in TARGETS]
            for job in as_completed(jobs):
                print('Completed', job.result(), flush=True)
    report(args.out, args.datasets, args.draws, args.splits)


if __name__ == '__main__':
    main()
