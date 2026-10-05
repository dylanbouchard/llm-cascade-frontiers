"""Replay frozen observed-score S2/S3 policies without search or scorer fitting."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from scipy.special import expit
from scorer_depth_compute import prepare, ALL_SCORERS, BASE_SCORERS, RESPONSE_SCORER
from manuscript_sources import DATASETS, EXPECTED_MODELS, result_path, file_record, source_root
from shared_difficulty_depth import stage_diagnostics
from exact_heldout_depth import evaluate_policy

OUT = Path('results/shared_difficulty_depth/observed')


def run(dataset):
    import response_scorer_compute
    response_scorer_compute.EMB_DIR = source_root(dataset)/'data/response_embeddings'
    raw, costs, rows, embeddings, fingerprint = prepare(dataset, response=True)
    models = list(raw)
    assert set(models) == EXPECTED_MODELS
    lookup = {int(row): j for j, row in enumerate(rows)}
    records, sources = [], []
    for scorer in ALL_SCORERS:
        for split in range(50):
            path = result_path(dataset, 'scorer_depth')/dataset/scorer/f'split_{split:02d}.json'
            manifest = json.loads(path.read_text())
            assert manifest['data_sha256'] == fingerprint
            assert manifest['pool'] == models and manifest['seed'] == 42+split
            assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
            idx = np.array([lookup[r] for r in manifest['test_idx']])
            sources.append(file_record(path))
            scores = []
            for model in models:
                if scorer in BASE_SCORERS:
                    s = raw[model][scorer].to_numpy(float)
                else:
                    fitted = manifest['fitted_scorers'][model]
                    if 'fallback' in fitted:
                        s = raw[model][fitted['fallback']].to_numpy(float)
                    else:
                        assert fitted['classes'] == [0, 1]
                        x = embeddings[model] if scorer == RESPONSE_SCORER else raw[model][BASE_SCORERS].to_numpy(float)
                        s = expit((x @ np.asarray(fitted['coef']).T + np.asarray(fitted['intercept'])).ravel())
                scores.append(s[idx])
            scores = np.column_stack(scores)
            correct = np.column_stack([raw[m].correct.to_numpy(float)[idx] for m in models])
            data = {m: dict(scores=scores[:, j], correct=correct[:, j], costs=costs[m][idx]) for j,m in enumerate(models)}
            curves = pd.read_csv(path.with_suffix('.csv'))
            np.testing.assert_allclose(curves.budget, manifest['budgets'], atol=1e-15)
            weights = np.ones(500)/499
            weights[[0,-1]] *= .5
            memo = {}
            for limit in (2, 3):
                policies = {}
                replay = []
                for b, policy in enumerate(manifest['frozen'][str(limit)]):
                    key = json.dumps(policy, sort_keys=True)
                    if key not in memo:
                        memo[key] = (evaluate_policy(policy, data), stage_diagnostics(policy, scores, correct, models))
                    replay.append(memo[key][0])
                    if key not in policies:
                        policies[key] = [policy, 0.]
                    policies[key][1] += weights[b]
                for j, metric in enumerate(('cost', 'accuracy', 'mean_calls')):
                    np.testing.assert_allclose(np.asarray(replay)[:,j], curves[f'd{limit}_{metric}'], atol=1e-12, rtol=0)
                for policy_id, (key, (policy, weight)) in enumerate(policies.items()):
                    for diagnostic in memo[key][1]:
                        records.append(dict(dataset=dataset, scorer=scorer, split=split,
                            depth_limit=limit, policy_id=policy_id, chain='|'.join(policy['models']),
                            budget_weight=weight, **diagnostic))
        print(f'{dataset}/{scorer} replayed 50 frozen splits', flush=True)
    frame = pd.DataFrame(records)
    frame.to_parquet(OUT/f'{dataset}.parquet', index=False)
    (OUT/f'{dataset}_provenance.json').write_text(json.dumps(dict(data_sha256=fingerprint, sources=sources),indent=2))
    return dataset


def summarize():
    frame = pd.concat([pd.read_parquet(OUT/f'{d}.parquet') for d in DATASETS],ignore_index=True)
    records = []
    for key, cell in frame.groupby(['dataset','scorer','depth_limit','stage']):
        record = dict(zip(['dataset','scorer','depth_limit','stage'],key))
        for metric in ('A_i','A_downstream','eq10_rhs'):
            valid = np.isfinite(cell[metric])
            record[metric] = np.average(cell.loc[valid,metric],weights=cell.loc[valid,'budget_weight']) if valid.any() else np.nan
        record['coverage'] = cell.budget_weight.sum()/50
        records.append(record)
    pd.DataFrame(records).to_csv(OUT/'summary.csv',index=False)
    (OUT/'validation.json').write_text(json.dumps(dict(frozen_split_replays=1750,
        policy_classes=2,stage_identity_checks=len(frame),new_searches=0,new_scorer_fits=0),indent=2))


if __name__ == '__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(run, DATASETS))
    summarize()
