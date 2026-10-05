"""Audit matched synthetic scenarios and replay held-out witnesses."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from exact_heldout_depth import evaluate_policy
from simulated_signal_depth import DATASETS, TARGETS, OUT
from manuscript_sources import EXPECTED_MODELS


def audit(splits=50):
    n=0
    integrals=pd.read_csv(OUT/'split_integrals.csv').set_index(['dataset','target_auroc','split'])
    for dataset in DATASETS:
        input_hash=hashlib.sha256((OUT/dataset/'inputs.npz').read_bytes()).hexdigest()
        with np.load(OUT/dataset/'inputs.npz') as z:
            data={k:z[k] for k in z.files}
        rows=data['rows']; models=data['models'].tolist()
        assert set(models) == EXPECTED_MODELS, (dataset, models)
        for split in range(splits):
            reference=None; baseline=None
            for target in TARGETS:
                path=OUT/dataset/f'auroc_{target:.1f}'/f'split_{split:02d}.csv'
                frame=pd.read_csv(path)
                manifest=json.loads(path.with_suffix('.json').read_text())
                assert manifest['pool'] == models
                assert manifest['input_sha256'] == input_hash
                np.testing.assert_allclose(frame.budget,manifest['budgets'],rtol=0,atol=1e-15)
                paired_gain=100*np.trapz(frame.d3_accuracy-frame.d2_accuracy,frame.fraction)
                np.testing.assert_allclose(paired_gain,integrals.loc[(dataset,target,split),'s3_minus_s2_pp'],rtol=0,atol=1e-10)
                signature=[manifest[k] for k in ('pool','cal_idx','test_idx','budgets')]
                if reference is None:
                    reference=signature
                    baseline=frame[['d1_accuracy','d1_cost','d1_cal_accuracy','d1_cal_cost']].to_numpy()
                assert signature==reference
                np.testing.assert_allclose(frame[['d1_accuracy','d1_cost','d1_cal_accuracy','d1_cal_cost']],baseline,atol=1e-12)
                assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
                assert sorted(manifest['cal_idx']+manifest['test_idx'])==rows.tolist()
                assert len(frame)==500
                for depth in (1,2,3):
                    assert (frame[f'd{depth}_cal_cost']<=frame.budget+1e-12).all()
                    assert frame[f'd{depth}_accuracy'].between(0,1).all()
                assert (frame.d3_cal_accuracy>=frame.d2_cal_accuracy-1e-12).all()
                assert (frame.d2_cal_accuracy>=frame.d1_cal_accuracy-1e-12).all()
                if split==0:
                    idx=np.searchsorted(rows,manifest['test_idx'])
                    test={m:dict(scores=data[f'scores_{target:.1f}'][idx,j],
                        correct=data['correct'][idx,j],costs=data['costs'][idx,j]) for j,m in enumerate(models)}
                    for depth in (1,2,3):
                        memo={}
                        for j,policy in enumerate(manifest['frozen'][str(depth)]):
                            key=json.dumps(policy,sort_keys=True)
                            if key not in memo:
                                memo[key]=evaluate_policy(policy,test)
                            np.testing.assert_allclose(memo[key],frame.iloc[j][[f'd{depth}_cost',f'd{depth}_accuracy',f'd{depth}_mean_calls']].to_numpy(float),atol=1e-12)
                n+=1
    summary=pd.read_csv(OUT/'summary.csv').set_index(['dataset','target_auroc'])
    np.testing.assert_allclose(summary.s3_minus_s2_pp,
        integrals.groupby(level=['dataset','target_auroc']).s3_minus_s2_pp.mean().reindex(summary.index),rtol=0,atol=1e-10)
    result=dict(scenario_splits=n,full_eight_model_pool=True,input_hashes_verified=True,
        integrated_gains_verified=True,matched_splits_and_budgets=True,baseline_invariant=True,
        calibration_feasible_and_nested=True,heldout_replayed_scenarios=25,
        heldout_replayed_budget_depth_cells=25*500*3)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    audit()
