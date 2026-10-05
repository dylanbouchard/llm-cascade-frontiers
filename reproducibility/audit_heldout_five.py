"""Audit all saved depth-five splits against original baseline artifacts."""
import hashlib
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from exact_heldout_five import ROOT, VERSION, source_hash
from exact_heldout_depth import DATASETS


def audit(out=ROOT/'results/exact_heldout_five', base=ROOT/'results/exact_heldout_four',
          datasets=DATASETS, available=False):
    code_hash = source_hash()
    totals = []
    for dataset in datasets:
        kernel_seconds = 0.
        count = 0
        indices = (sorted(int(p.stem.split('_')[1]) for p in (out/dataset).glob('split_[0-9][0-9].csv'))
                   if available else list(range(50)))
        for split in indices:
            path = out/dataset/f'split_{split:02d}.json'
            manifest = json.loads(path.read_text())
            original_bytes = (base/dataset/path.name).read_bytes()
            original = json.loads(original_bytes)
            assert manifest['version'] == VERSION
            assert manifest['code_sha256'] == code_hash
            assert manifest['baseline_sha256'] == hashlib.sha256(original_bytes).hexdigest()
            assert manifest['seed'] == 42+split
            assert manifest['cal_idx'] == original['cal_idx']
            assert manifest['test_idx'] == original['test_idx']
            assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
            assert len(set(manifest['cal_idx'])) == len(manifest['cal_idx'])
            assert len(set(manifest['test_idx'])) == len(manifest['test_idx'])
            assert manifest['budgets'] == original['budgets']
            chain_dir=out/dataset/f'split_{split:02d}_chains'
            chain_files=sorted(chain_dir.glob('chain_*.npz'))
            assert len(chain_files)==manifest['search_stats']['quintuples']
            assert len(chain_files)==len(list(combinations(original['pool'],5)))
            chain_fingerprint=hashlib.sha256((code_hash+manifest['baseline_sha256']).encode()).hexdigest()
            chain_seconds=0.
            chain_evaluated=0
            for chain in chain_files:
                with np.load(chain) as z:
                    assert str(z['fingerprint'])==chain_fingerprint
                    stats=json.loads(str(z['stats']))
                    assert stats['complete']
                    chain_seconds+=stats['seconds']
                    chain_evaluated+=stats['evaluated']
            assert abs(chain_seconds-manifest['search_stats']['kernel_seconds'])<1e-9
            assert chain_evaluated==manifest['search_stats']['evaluated_candidates']
            for depth in ('1', '2', '3', '4'):
                assert manifest['frozen'][depth] == original['frozen'][depth]
            frame = pd.read_csv(path.with_suffix('.csv'))
            old = pd.read_csv(base/dataset/path.with_suffix('.csv').name)
            assert len(frame) == 500
            pd.testing.assert_frame_equal(frame[old.columns], old, check_exact=False, atol=1e-15, rtol=1e-12)
            assert (frame.d5_cal_accuracy + 1e-12 >= frame.d4_cal_accuracy).all()
            assert (frame.d5_cal_cost <= frame.budget + 1e-12).all()
            assert (np.diff(frame.d5_cal_accuracy) >= -1e-12).all()
            assert frame.d5_accuracy.between(0, 1).all()
            assert (frame.d5_mean_calls <= frame.d5_selected_depth).all()
            assert (frame.d5_mean_calls >= 1).all()
            for j, policy in enumerate(manifest['frozen']['5']):
                assert len(policy['models']) <= 5
                assert len(policy['thresholds']) == len(policy['models'])-1
                assert abs(policy['cal_cost']-frame.d5_cal_cost.iloc[j]) < 1e-15
                assert abs(policy['cal_quality']-frame.d5_cal_accuracy.iloc[j]) < 1e-12
            for depth in (2, 3, 4):
                for metric in ('cost', 'accuracy'):
                    np.testing.assert_allclose(frame[f'd5_minus_d{depth}_{metric}'],
                                               frame[f'd5_{metric}']-frame[f'd{depth}_{metric}'],
                                               atol=1e-15, rtol=1e-10)
            kernel_seconds += manifest['search_stats']['kernel_seconds']
            count += manifest['search_stats']['evaluated_candidates']
        totals.append(dict(dataset=dataset, splits=len(indices), assigned_budget_positions=500*len(indices),
                           kernel_seconds=kernel_seconds, evaluated_candidates=count))
    result = dict(status='passed', total_splits=sum(t['splits'] for t in totals),
                  total_budget_positions=sum(t['assigned_budget_positions'] for t in totals),
                  checks=['source and baseline hashes', 'split identity and disjointness',
                          'unchanged baseline selections and metrics', 'calibration nesting',
                          'calibration budget feasibility and monotonicity',
                          'CSV agreement with frozen selections', 'paired differences',
                          'test accuracy and calls ranges', 'all exact chain checkpoints complete'], datasets=totals)
    name = 'audit.json' if list(datasets) == list(DATASETS) and not available else 'audit-partial.json'
    (out/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
