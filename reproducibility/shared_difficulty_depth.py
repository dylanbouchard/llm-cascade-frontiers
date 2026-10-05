"""Shared-difficulty synthetic scores and matched exact S2/S3 evaluation.

No model calls or manuscript edits. Run from the repository root.
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
from scipy.stats import rankdata

import simulate_correlated_confidence as synthetic
import simulated_signal_depth as depth
from manuscript_sources import EXPECTED_MODELS
from synthetic_score_draws import sha, validate_match, write_json

BASE = Path('results/simulated_signal_depth')
OUT = Path('results/shared_difficulty_depth')
TARGETS = (.8, .9)
GRID = np.linspace(0, 1, 101)
VERSION = 'shared-difficulty-v1'


def standardize(x):
    return (x - x.mean(axis=0)) / x.std(axis=0, ddof=1)


def signal_columns(outcomes):
    """Mean-impute missing labels, then form leave-one-model-out difficulty."""
    filled = np.where(np.isfinite(outcomes), outcomes, np.nanmean(outcomes, axis=0))
    errors = 1 - filled
    difficulty = (errors.sum(axis=1, keepdims=True) - errors) / (errors.shape[1] - 1)
    return standardize(filled), standardize(difficulty)


def auc_or_nan(score, label):
    if len(label) == 0 or len(np.unique(label)) < 2:
        return float('nan')
    return synthetic.empirical_auc(score, label)


def calibrate_column(signal, noise, labels, target):
    """Find the first sampled AUROC crossing, then bisect that bracket.

    t*signal+(1-t)*noise is rank-equivalent to a*signal+noise,
    a=t/(1-t). A grid avoids assuming global finite-sample monotonicity.
    Pure-signal feasibility is a conservative, explicit operational rule.
    """
    ceiling = synthetic.empirical_auc(signal, labels)
    if ceiling < target - synthetic.AUROC_TOL:
        raise ValueError('Pure signal does not reach target')
    previous = 0.
    previous_auc = synthetic.empirical_auc(noise, labels)
    if previous_auc >= target:
        raise ValueError('Noise already exceeds target')
    for upper in np.linspace(0, 1 - 1e-10, 65)[1:]:
        current_auc = synthetic.empirical_auc(upper*signal+(1-upper)*noise, labels)
        if abs(current_auc-target) <= synthetic.AUROC_TOL:
            return upper*signal+(1-upper)*noise, float(upper), ceiling
        if current_auc > target:
            lower = previous
            for _ in range(50):
                middle = (lower+upper)/2
                score = middle*signal+(1-middle)*noise
                auc = synthetic.empirical_auc(score, labels)
                if abs(auc-target) <= synthetic.AUROC_TOL:
                    return score, float(middle), ceiling
                if auc < target:
                    lower = middle
                else:
                    upper = middle
            break
        previous = upper
    raise ValueError('No calibrated finite-noise crossing found')


def construct(w, y, difficulty, residual, outcomes, target):
    signal = (1-w)*y-w*difficulty
    fits = [calibrate_column(signal[:, j], residual[:, j], outcomes[:, j], target)
            for j in range(y.shape[1])]
    scores = np.column_stack([fit[0] for fit in fits])
    scores = (scores-scores.min(axis=0))/np.ptp(scores, axis=0)
    return scores, [fit[1] for fit in fits], [fit[2] for fit in fits]


def prepare(dataset, out):
    row_index, models, corr, outcomes = synthetic.load_dataset(dataset)
    assert set(models) == EXPECTED_MODELS
    with np.load(BASE/dataset/'inputs.npz') as z:
        base = {k: z[k] for k in ('rows', 'models', 'correct', 'costs')}
    order = [models.index(m) for m in base['models'].tolist()]
    assert set(base['models'].tolist()) == EXPECTED_MODELS
    np.testing.assert_array_equal(outcomes[base['rows']][:, order], base['correct'])
    y, difficulty = signal_columns(outcomes)
    jobs, diagnostics, feasibility = [], [], []
    for target in TARGETS:
        legacy_path = Path('results/simulated_confidence')/dataset/f'auroc_{target:.1f}.parquet'
        legacy = pd.read_parquet(legacy_path).set_index('row_index').loc[row_index, models].to_numpy()
        scaled = standardize(legacy)
        # Projection coefficient recovers the original own-label signal because
        # its noise is orthogonal to all label columns.
        a0 = (y*scaled).sum(axis=0)/(len(y)-1)
        residual = scaled-y*a0
        ceilings = []
        for w in GRID:
            signal = (1-w)*y-w*difficulty
            aucs = [synthetic.empirical_auc(signal[:, j], outcomes[:, j]) for j in range(len(models))]
            ceilings.append(aucs)
            feasibility.extend(dict(dataset=dataset, target=target, w=w, model=m,
                signal_only_auroc=aucs[j], signal_feasible=aucs[j] >= target-synthetic.AUROC_TOL)
                for j, m in enumerate(models))
        candidates = np.flatnonzero(np.min(ceilings, axis=1) >= target-synthetic.AUROC_TOL)
        maximum = None
        for idx in candidates[::-1]:
            try:
                fitted = construct(GRID[idx], y, difficulty, residual, outcomes, target)
                maximum = (float(GRID[idx]), fitted)
                break
            except ValueError:
                continue
        if maximum is None:
            raise ValueError(f'No feasible mixture for {dataset}/{target}')
        for condition in ('own', 'half', 'max'):
            w = dict(own=0., half=.5, max=maximum[0])[condition]
            if condition == 'own':
                scores, mixing, ceiling = legacy, (a0/(1+a0)).tolist(), [1.]*len(models)
            else:
                try:
                    scores, mixing, ceiling = maximum[1] if condition == 'max' else construct(
                        w, y, difficulty, residual, outcomes, target)
                except ValueError as error:
                    diagnostics.append(dict(dataset=dataset, target=target, condition=condition,
                        w=w, model='ALL', feasible=False, reason=str(error)))
                    continue
            dest = out/condition/f'auroc_{target:.1f}'/dataset
            dest.mkdir(parents=True, exist_ok=True)
            aligned = scores[base['rows']][:, order]
            if condition == 'own':
                with np.load(BASE/dataset/'inputs.npz') as z:
                    np.testing.assert_array_equal(aligned, z[f'scores_{target:.1f}'])
            identity = dict(version=VERSION, dataset=dataset, target=target, condition=condition, w=w,
                base_input_sha256=sha(BASE/dataset/'inputs.npz'), legacy_score_sha256=sha(legacy_path),
                source_sha256={p: sha(Path(p)) for p in ('shared_difficulty_depth.py',
                    'simulate_correlated_confidence.py', 'synthetic_score_draws.py',
                    'simulated_signal_depth.py', 'exact_heldout_depth.py',
                    'oracle_depth_analysis.py', 'optuna_frontier.py', 'fig2_compute.py')},
                correlation_sha256=hashlib.sha256(corr.tobytes()).hexdigest(),
                outcomes_sha256=hashlib.sha256(outcomes.tobytes()).hexdigest())
            meta = dest/'construction.json'
            if meta.exists():
                saved = json.loads(meta.read_text())
                if saved['identity'] != identity or saved['input_sha256'] != sha(dest/'inputs.npz'):
                    raise ValueError(f'Incompatible construction cache {dest}')
            else:
                np.savez_compressed(dest/'inputs.npz', **base, **{f'scores_{target:.1f}': aligned})
                np.savez_compressed(dest/'full_scores.npz', scores=scores, models=models, rows=row_index)
                write_json(meta, dict(identity=identity, input_sha256=sha(dest/'inputs.npz')))
            for j, m in enumerate(models):
                diagnostics.append(dict(dataset=dataset, target=target, condition=condition, w=w,
                    model=m, feasible=True, t=mixing[j], signal_only_auroc=ceiling[j],
                    difficulty_only_auroc=synthetic.empirical_auc(-difficulty[:, j], outcomes[:, j]),
                    full_auroc=synthetic.empirical_auc(scores[:, j], outcomes[:, j]),
                    retained_auroc=synthetic.empirical_auc(aligned[:, order.index(j)], base['correct'][:, order.index(j)]),
                    correlation_max_error=float(np.abs(np.corrcoef(scores, rowvar=False)-corr).max())))
            jobs.append((dataset, target, condition, w))
            print(f'Prepared {dataset} A={target} {condition} w={w:.2f}', flush=True)
    return jobs, diagnostics, feasibility


def stage_diagnostics(policy, scores, correct, models):
    """Reached-population AUROCs and equation 10, for frozen continuation.

    The independent LHS integrates the empirical escalation curve. Midranks
    implement random ordering within ties. Empty reaches are retained explicitly.
    """
    seq = [models.index(m) for m in policy['models']]
    taus = [(-np.inf if t == 'never' else np.inf if t == 'always' else float(t))
            for t in policy['thresholds']]
    reached = np.ones(len(scores), dtype=bool)
    records = []
    for i, j in enumerate(seq[:-1]):
        continuation = correct[:, seq[-1]].copy()
        for h in range(len(seq)-2, i, -1):
            continuation = np.where(scores[:, seq[h]] >= taus[h], correct[:, seq[h]], continuation)
        s, own, downstream = scores[reached, j], correct[reached, j], continuation[reached]
        n = len(s)
        p = n/len(scores)
        ai, ad = auc_or_nan(s, own), auc_or_nan(s, downstream)
        pi, v = (float(own.mean()), float(downstream.mean())) if n else (np.nan, np.nan)
        own_term = pi*(1-pi)*(ai-.5) if np.isfinite(ai) else 0.
        downstream_term = v*(1-v)*(ad-.5) if np.isfinite(ad) else 0.
        rhs = p*(own_term-downstream_term)
        lhs = p*np.mean((downstream-own)*(.5-(rankdata(s)-.5)/n)) if n else 0.
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)
        records.append(dict(stage=i, model=models[j], reached=n, reach_probability=p,
            pi=pi, v=v, A_i=ai, A_downstream=ad, own_term=own_term,
            downstream_term=downstream_term, eq10_rhs=rhs, eq10_lhs=lhs))
        reached &= scores[:, j] < taus[i]
    return records


def run_cell(job, splits, out):
    dataset, target, condition, w = job
    depth.OUT = out/condition/f'auroc_{target:.1f}'
    depth.run(dataset, target, splits)
    dest = depth.OUT/dataset/f'auroc_{target:.1f}'
    with np.load(dest.parent/'inputs.npz') as z:
        rows, models, correct, scores = z['rows'], z['models'].tolist(), z['correct'], z[f'scores_{target:.1f}']
    lookup = {int(r): j for j, r in enumerate(rows)}
    records, integrals = [], []
    for split in range(splits):
        manifest = json.loads((dest/f'split_{split:02d}.json').read_text())
        validate_match(manifest, json.loads((BASE/dataset/dest.name/f'split_{split:02d}.json').read_text()))
        curves = pd.read_csv(dest/f'split_{split:02d}.csv')
        integrals.append(dict(dataset=dataset, target=target, condition=condition, w=w, split=split,
            gain_pp=100*np.trapz(curves.d3_accuracy-curves.d2_accuracy, curves.fraction),
            cost_gap_per_1000=1000*np.trapz(curves.d3_cost-curves.d2_cost, curves.fraction)))
        # Uniform budget weights, trapezoidal at the two endpoints.
        weights = np.ones(len(depth.FRACTIONS))/(len(depth.FRACTIONS)-1)
        weights[[0, -1]] *= .5
        for population, field in (('calibration', 'cal_idx'), ('test', 'test_idx')):
            idx = np.array([lookup[r] for r in manifest[field]])
            for limit in (2, 3):
                policies = {}
                for b, policy in enumerate(manifest['frozen'][str(limit)]):
                    key = json.dumps(policy, sort_keys=True)
                    if key not in policies:
                        policies[key] = [policy, 0.]
                    policies[key][1] += weights[b]
                for policy_id, (policy, weight) in enumerate(policies.values()):
                    for diagnostic in stage_diagnostics(policy, scores[idx], correct[idx], models):
                        records.append(dict(dataset=dataset, target=target, condition=condition, w=w,
                            split=split, population=population, depth_limit=limit, policy_id=policy_id,
                            chain='|'.join(policy['models']), selected_depth=len(policy['models']),
                            budget_weight=weight, **diagnostic))
    pd.DataFrame(records).to_csv(dest/'stage_diagnostics.csv', index=False)
    pd.DataFrame(integrals).to_csv(dest/'integrals.csv', index=False)
    return job


def report(jobs, out, splits):
    gains, stages = [], []
    for dataset, target, condition, w in jobs:
        dest = out/condition/f'auroc_{target:.1f}'/dataset/f'auroc_{target:.1f}'
        gains.append(pd.read_csv(dest/'integrals.csv'))
        stages.append(pd.read_csv(dest/'stage_diagnostics.csv'))
    gains, stages = pd.concat(gains, ignore_index=True), pd.concat(stages, ignore_index=True)
    gains.to_csv(out/'split_integrals.csv', index=False)
    stages.to_parquet(out/'stage_diagnostics.parquet', index=False)
    keys = ['dataset', 'target', 'condition', 'w']
    summary = gains.groupby(keys).agg(gain_pp=('gain_pp', 'mean'),
        split_sd_pp=('gain_pp', 'std'), cost_gap_per_1000=('cost_gap_per_1000', 'mean')).reset_index()
    diagnostics = []
    for identity, cell in stages.groupby(keys+['population', 'depth_limit', 'stage']):
        record = dict(zip(keys+['population', 'depth_limit', 'stage'], identity))
        for metric in ('A_i', 'A_downstream', 'eq10_rhs', 'reach_probability'):
            valid = np.isfinite(cell[metric])
            record[metric] = np.average(cell.loc[valid, metric], weights=cell.loc[valid, 'budget_weight']) if valid.any() else np.nan
        record['policy_budget_mass'] = cell.budget_weight.sum()/splits
        diagnostics.append(record)
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(out/'selected_chain_summary.csv', index=False)
    summary.to_csv(out/'summary.csv', index=False)
    lines = ['# Shared-difficulty experiment', '',
        f'One paired score draw, {splits} matched splits, 500 assigned budget positions, all eight models.', '',
        'Original scores are retained exactly at w=0. For each target, residual noise from the original '
        'construction is held fixed across w. Pearson correlations may change. Maximum feasible w is '
        'the largest point on the 0.01 grid where all eight pure signals reach the target within 0.001 '
        'and finite-noise calibration succeeds. This is an operational grid maximum, not a proven '
        'global feasibility limit. Missing labels are mean-imputed only for construction.', '',
        'Full-sample labels construct the scores. Held-out evaluation concerns policy selection, not '
        'scorer learning. Split SDs describe overlapping splits, not independent uncertainty. '
        'The same assigned budgets can produce different realized test costs.', '',
        '| Dataset | AUROC | Mixture | w | S3 minus S2 (pp) | Split SD |',
        '|---|---:|---|---:|---:|---:|']
    for r in summary.itertuples():
        lines.append(f'| {r.dataset} | {r.target:.1f} | {r.condition} | {r.w:.2f} | {r.gain_pp:+.4f} | {r.split_sd_pp:.4f} |')
    lines += ['', '## Selected-chain diagnostics', '',
        'The CSV reports calibration and held-out reached-population A_i, A_downstream, and '
        'equation 10 RHS separately for S2/S3 and each nonterminal stage. Means weight each '
        'selected policy by its normalized budget interval. Single-model selections have no '
        'stage diagnostic. The policy_budget_mass column reports coverage. Undefined AUROCs '
        'are excluded from AUROC means, and degenerate-label terms in equation 10 are zero. '
        'The independent empirical rank-integral LHS is checked against the RHS for every policy. '
        'Equation 10 measures gain over random escalation with continuation frozen, not the '
        'optimized S3 minus S2 frontier gap. Changes in selected chains and reached populations '
        'can therefore affect the relationship.', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(summary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=depth.DATASETS, default=list(depth.DATASETS))
    parser.add_argument('--splits', type=int, default=50)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.splits <= 50 or args.workers < 1:
        parser.error('Require 1 to 50 splits and positive workers')
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs, diagnostics, feasibility = [], [], []
    for dataset in args.datasets:
        j, d, f = prepare(dataset, args.out)
        jobs.extend(j)
        diagnostics.extend(d)
        feasibility.extend(f)
    pd.DataFrame(diagnostics).to_csv(args.out/'construction_diagnostics.csv', index=False)
    pd.DataFrame(feasibility).to_csv(args.out/'feasibility_grid.csv', index=False)
    write_json(args.out/'run.json', dict(splits=args.splits, jobs=jobs, version=VERSION))
    if args.prepare_only:
        return
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_cell, j, args.splits, args.out) for j in jobs]
            for future in as_completed(futures):
                print('Completed', future.result(), flush=True)
    report(jobs, args.out, args.splits)


if __name__ == '__main__':
    main()
