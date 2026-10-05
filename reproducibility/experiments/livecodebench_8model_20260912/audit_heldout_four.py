"""Audit all saved depth-four splits against original baseline artifacts."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exact_heldout_four import ROOT, VERSION
from exact_heldout_depth import DATASETS


def audit(out=ROOT/'results/exact_heldout_four', base=ROOT/'results/exact_heldout_depth',
          datasets=DATASETS):
    code_hash = hashlib.sha256((ROOT/'exact_four_kernel.cpp').read_bytes() +
                               (ROOT/'exact_four_model.py').read_bytes() +
                               (ROOT/'exact_heldout_four.py').read_bytes()).hexdigest()
    totals = []
    for dataset in datasets:
        kernel_seconds = 0.
        count = 0
        for split in range(50):
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
            for depth in ('1', '2', '3'):
                assert manifest['frozen'][depth] == original['frozen'][depth]
            frame = pd.read_csv(path.with_suffix('.csv'))
            old = pd.read_csv(base/dataset/path.with_suffix('.csv').name)
            assert len(frame) == 500
            pd.testing.assert_frame_equal(frame[old.columns], old, check_exact=False, atol=1e-15, rtol=1e-12)
            assert (frame.d4_cal_accuracy + 1e-12 >= frame.d3_cal_accuracy).all()
            assert (frame.d4_cal_cost <= frame.budget + 1e-12).all()
            assert (np.diff(frame.d4_cal_accuracy) >= -1e-12).all()
            assert frame.d4_accuracy.between(0, 1).all()
            assert (frame.d4_mean_calls <= frame.d4_selected_depth).all()
            assert (frame.d4_mean_calls >= 1).all()
            for j, policy in enumerate(manifest['frozen']['4']):
                assert len(policy['models']) <= 4
                assert len(policy['thresholds']) == len(policy['models'])-1
                assert abs(policy['cal_cost']-frame.d4_cal_cost.iloc[j]) < 1e-15
                assert abs(policy['cal_quality']-frame.d4_cal_accuracy.iloc[j]) < 1e-12
            for depth in (2, 3):
                for metric in ('cost', 'accuracy'):
                    np.testing.assert_allclose(frame[f'd4_minus_d{depth}_{metric}'],
                                               frame[f'd4_{metric}']-frame[f'd{depth}_{metric}'],
                                               atol=1e-15, rtol=1e-10)
            kernel_seconds += manifest['search_stats']['kernel_seconds']
            count += manifest['search_stats']['evaluated_candidates']
        totals.append(dict(dataset=dataset, splits=50, assigned_budget_positions=25000,
                           kernel_seconds=kernel_seconds, evaluated_candidates=count))
    result = dict(status='passed', total_splits=50*len(datasets),
                  total_budget_positions=25000*len(datasets),
                  checks=['source and baseline hashes', 'split identity and disjointness',
                          'unchanged baseline selections and metrics', 'calibration nesting',
                          'calibration budget feasibility and monotonicity',
                          'CSV agreement with frozen selections', 'paired differences',
                          'test accuracy and calls ranges'], datasets=totals)
    name = 'audit.json' if list(datasets) == list(DATASETS) else 'audit-partial.json'
    (out/name).write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    audit()
