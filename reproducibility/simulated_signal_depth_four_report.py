"""Audit and tabulate S1-S4 AUROC counterfactuals without manuscript or figure writes."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exact_heldout_depth import evaluate_policy, FRACTIONS
from exact_heldout_four import policy_key
from manuscript_sources import EXPECTED_MODELS
from simulated_signal_depth import DATASETS, TARGETS
from simulated_signal_depth_four import BASE, OUT, VERSION, source_hash

DEPTHS = (1,2,3,4)
COMPARISONS = ((2,1),(3,1),(4,1),(3,2),(4,2),(4,3))
NAMES = dict(mmlu='MMLU', triviaqa='TriviaQA', math_hard='MATH',
             simpleqa='SimpleQA', livecodebench='LiveCodeBench')


def integrate(frame):
    row = {}
    for depth in DEPTHS:
        for metric,scale in (('accuracy',100),('cost',1000),('cal_accuracy',100),('mean_calls',1)):
            suffix = 'pp' if metric.endswith('accuracy') else 'per_1000' if metric=='cost' else 'mean'
            row[f's{depth}_{metric}_{suffix}'] = scale*np.trapz(frame[f'd{depth}_{metric}'],frame.fraction)
        row[f's{depth}_overshoot_fraction'] = (frame[f'd{depth}_overshoot']>1e-12).mean()
    for high,low in COMPARISONS:
        row[f's{high}_minus_s{low}_pp'] = 100*np.trapz(frame[f'd{high}_accuracy']-frame[f'd{low}_accuracy'],frame.fraction)
        row[f's{high}_minus_s{low}_cost_per_1000'] = 1000*np.trapz(frame[f'd{high}_cost']-frame[f'd{low}_cost'],frame.fraction)
    for depth in DEPTHS:
        row[f's4_depth{depth}_fraction'] = np.trapz((frame.d4_selected_depth==depth).astype(float),frame.fraction)
    return row


def report(out=OUT, splits=50):
    records=[]; witnesses=0; code_hash=source_hash(); cells=0
    for dataset in DATASETS:
        inp=BASE/dataset/'inputs.npz'
        input_hash=hashlib.sha256(inp.read_bytes()).hexdigest()
        with np.load(inp) as z:
            data={k:z[k] for k in z.files}
        rows=data['rows']; models=data['models'].tolist()
        assert set(models)==EXPECTED_MODELS
        for split in range(splits):
            signature=None; baseline=None
            for target in TARGETS:
                path=out/dataset/f'auroc_{target:.1f}'/f'split_{split:02d}.json'
                m=json.loads(path.read_text())
                base=BASE/dataset/f'auroc_{target:.1f}'/path.name
                old=json.loads(base.read_text())
                assert m['version']==VERSION and m['source_sha256']==code_hash
                assert m['input_sha256']==input_hash
                assert m['baseline_sha256']==hashlib.sha256(base.read_bytes()).hexdigest()
                assert m['baseline_csv_sha256']==hashlib.sha256(base.with_suffix('.csv').read_bytes()).hexdigest()
                assert m['pool']==models
                assert set(m['frozen'])==set(map(str,DEPTHS))
                assert m['search_stats']['quadruples']==70
                for depth in (1,2,3):
                    assert m['frozen'][str(depth)]==old['frozen'][str(depth)]
                assert not set(m['cal_idx']) & set(m['test_idx'])
                assert sorted(m['cal_idx']+m['test_idx'])==rows.tolist()
                current=[m[k] for k in ('pool','cal_idx','test_idx','budgets')]
                if signature is None: signature=current
                assert current==signature
                frame=pd.read_csv(path.with_suffix('.csv'))
                oldframe=pd.read_csv(base.with_suffix('.csv'))
                np.testing.assert_allclose(frame[oldframe.columns].select_dtypes('number'),oldframe.select_dtypes('number'),rtol=0,atol=1e-12)
                assert len(frame)==500
                np.testing.assert_allclose(frame.fraction,FRACTIONS,rtol=0,atol=1e-15)
                np.testing.assert_allclose(frame.budget,m['budgets'],rtol=0,atol=1e-15)
                basevalues=frame[['d1_accuracy','d1_cost','d1_cal_accuracy','d1_cal_cost']].to_numpy()
                if baseline is None: baseline=basevalues
                np.testing.assert_allclose(basevalues,baseline,rtol=0,atol=1e-12)
                for depth in DEPTHS:
                    assert (frame[f'd{depth}_cal_cost']<=frame.budget+1e-12).all()
                    assert frame[f'd{depth}_accuracy'].between(0,1).all()
                    assert frame[f'd{depth}_selected_depth'].between(1,depth).all()
                    if depth>1:
                        assert (frame[f'd{depth}_cal_accuracy']>=frame[f'd{depth-1}_cal_accuracy']-1e-12).all()
                # Independently replay every unique S4 witness on both partitions.
                for partition in ('cal','test'):
                    idx=np.searchsorted(rows,m[f'{partition}_idx'])
                    np.testing.assert_array_equal(rows[idx],m[f'{partition}_idx'])
                    subset={model:dict(scores=data[f'scores_{target:.1f}'][idx,j],
                        correct=data['correct'][idx,j],costs=data['costs'][idx,j]) for j,model in enumerate(models)}
                    memo={policy_key(p):evaluate_policy(p,subset)
                          for p in {policy_key(p):p for p in m['frozen']['4']}.values()}
                    witnesses+=len(memo)
                    values=np.array([memo[policy_key(p)] for p in m['frozen']['4']])
                    if partition=='cal':
                        np.testing.assert_allclose(values[:,:2],frame[['d4_cal_cost','d4_cal_accuracy']],rtol=0,atol=1e-12)
                    else:
                        np.testing.assert_allclose(values,frame[['d4_cost','d4_accuracy','d4_mean_calls']],rtol=0,atol=1e-12)
                records.append(dict(dataset=dataset,target_auroc=target,split=split,**integrate(frame)))
                cells+=1
        print(f'Audited {dataset}: all scenarios and splits',flush=True)
    f=pd.DataFrame(records)
    baseline=f.loc[f.target_auroc==.5,['dataset','split']+[f's{d}_accuracy_pp' for d in DEPTHS]]
    f=f.merge(baseline,on=['dataset','split'],suffixes=('','_at_05'),validate='many_to_one')
    for depth in DEPTHS:
        f[f's{depth}_signal_gain_pp']=f[f's{depth}_accuracy_pp']-f[f's{depth}_accuracy_pp_at_05']
    metrics=[c for c in f if c not in ('dataset','target_auroc','split')]
    grouped=f.groupby(['dataset','target_auroc'])[metrics]
    summary=grouped.mean()
    for q in (.1,.9):
        percentiles=grouped.quantile(q).add_suffix(f'_p{int(100*q)}')
        summary=summary.join(percentiles)
    f.to_csv(out/'split_integrals.csv',index=False)
    summary.to_csv(out/'summary.csv')
    audit=dict(scenario_splits=cells,depths=DEPTHS,full_eight_model_pool=True,
        all_70_quadruples_searched=True,input_and_source_hashes_verified=True,
        baseline_cache_hashes_verified=True,original_s1_s2_s3_unchanged=True,
        matched_splits_and_budgets=True,s1_invariant_across_auroc=True,
        calibration_feasible_and_nested=True,s4_unique_partition_witnesses_replayed=witnesses)
    (out/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    lines=['# AUROC counterfactuals with S1 through S4','',
        f'{splits} matched calibration-test splits, 500 assigned budgets, eight models, and five AUROC targets per benchmark. '
        'All accuracy values are budget-integrated test accuracy in percentage points. '
        'S1 through S4 allow at most one through four models.','',
        '| Dataset | AUROC | S1 accuracy | S2 accuracy | S3 accuracy | S4 accuracy | S4 minus S2 | S4 minus S3 [p10, p90] |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for dataset in DATASETS:
        for target in TARGETS:
            r=summary.loc[(dataset,target)]
            lines.append(f'| {NAMES[dataset]} | {target:.1f} | {r.s1_accuracy_pp:.3f} | {r.s2_accuracy_pp:.3f} | '
                f'{r.s3_accuracy_pp:.3f} | {r.s4_accuracy_pp:.3f} | {r.s4_minus_s2_pp:+.3f} | '
                f'{r.s4_minus_s3_pp:+.3f} [{r.s4_minus_s3_pp_p10:+.3f}, {r.s4_minus_s3_pp_p90:+.3f}] |')
    lines+=['','S1 is constant across AUROC because standalone selection does not use confidence scores. '
        'The 10th and 90th split percentiles describe variation across overlapping splits. '
        'Realized test costs and cost differences are retained in summary.csv.','',
        'The original S1-S3 selections and score draws are reused unchanged. Each S4 search covers all 70 '
        'cost-ordered quadruples and retains the frozen S3 optimum as a candidate at each assigned budget. '
        'Policies are frozen before held-out evaluation. All selected S4 policies are replayed on calibration '
        'and test data in the audit.','',
        'The paper figure and manuscript text have not been changed.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(summary.loc[(slice(None),.9),['s1_accuracy_pp','s2_accuracy_pp','s3_accuracy_pp','s4_accuracy_pp','s4_minus_s2_pp','s4_minus_s3_pp']].to_string())
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--splits',type=int,default=50)
    args=p.parse_args()
    report(splits=args.splits)
