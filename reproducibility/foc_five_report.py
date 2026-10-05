"""Report FOC-guided S5 and the completed S2-S5 comparison."""
import json
import numpy as np
import pandas as pd
from foc_pairwise_compare import ROOT, DATASETS
OUT=ROOT/'results/foc_five'
NAMES=dict(zip(DATASETS,['MMLU','TriviaQA','MATH','SimpleQA','LiveCodeBench']))
trap=getattr(np,'trapezoid',np.trapz)
rows=[];times=[]
for dataset in DATASETS:
    for split in range(50):
        path=OUT/dataset/f'split_{split:02d}.csv'
        frame=pd.read_csv(path);meta=json.loads(path.with_suffix('.json').read_text())
        parent=json.loads((ROOT/meta['parent_source']).read_text())
        assert len(frame)==500 and meta['budget_indices']==list(range(500))
        assert len(meta['chain_timings'])==56 and len(meta['sequences'])==218
        np.testing.assert_allclose(frame.fraction,np.linspace(0,1,500),atol=1e-14,rtol=0)
        f=frame.fraction
        for d in [2,3,4,5]:
            assert (frame[f'foc{d}_cal_cost']<=frame.budget+1e-12).all()
            assert (frame[f'foc{d}_cal_accuracy']<=frame[f'exact{d}_cal_accuracy']+1e-12).all()
            if d>2:assert (frame[f'foc{d}_cal_accuracy']>=frame[f'foc{d-1}_cal_accuracy']-1e-12).all()
            row=dict(dataset=dataset,split=split,depth=d)
            for stat,scale in [('accuracy',100),('cost',1000),('cal_accuracy',100),('cal_cost',1000)]:
                row[stat+'_difference']=float(trap(scale*(frame[f'foc{d}_{stat}']-frame[f'exact{d}_{stat}']),f))
            row['cal_gap_pp']=-row['cal_accuracy_difference']
            row['cost_pct_budget']=float(trap(100*(frame[f'foc{d}_cost']-frame[f'exact{d}_cost'])/frame.budget,f))
            row['cal_optimal_fraction']=float(trap(np.isclose(frame[f'foc{d}_cal_accuracy'],frame[f'exact{d}_cal_accuracy'],atol=1e-12,rtol=0).astype(float),f))
            row['max_cal_gap_pp']=float(100*(frame[f'exact{d}_cal_accuracy']-frame[f'foc{d}_cal_accuracy']).max())
            row['gain_vs_foc2_pp']=float(trap(100*(frame[f'foc{d}_accuracy']-frame.foc2_accuracy),f))
            row['gain_vs_exact2_pp']=float(trap(100*(frame[f'foc{d}_accuracy']-frame.exact2_accuracy),f))
            row['gain_vs_foc4_pp']=float(trap(100*(frame[f'foc{d}_accuracy']-frame.foc4_accuracy),f))
            row['cal_gain_vs_foc4_pp']=float(trap(100*(frame[f'foc{d}_cal_accuracy']-frame.foc4_cal_accuracy),f))
            row['full_depth_fraction']=float(trap((frame[f'foc{d}_depth']==d).astype(float),f))
            rows.append(row)
        stats=meta['chain_timings']
        extension=sum(t['seconds'] for t in stats)
        times.append(dict(dataset=dataset,split=split,extension_kernel_seconds=extension,total_selection_kernel_seconds=extension+meta['parent_selection_seconds'],extension_with_replay_seconds=meta['extension_search_seconds'],accepted_transfers=sum(t['accepted_transfers'] for t in stats),starts=sum(t['starts'] for t in stats),coordinate_slices=sum(t['coordinate_slices'] for t in stats)))
summary=pd.DataFrame(rows);summary.to_csv(OUT/'split_summary.csv',index=False)
group=summary.groupby(['dataset','depth']);means=group.mean(numeric_only=True)
for name,quant in [('p10',.1),('p90',.9)]:means['accuracy_'+name]=group.accuracy_difference.quantile(quant)
means.to_csv(OUT/'summary.csv')
timing=pd.DataFrame(times);timing.to_csv(OUT/'timings.csv',index=False)
timing.groupby('dataset').mean(numeric_only=True).to_csv(OUT/'timing_summary.csv')
text='''# FOC-guided S5 versus exhaustive S5

## Completed run

The extension covers all five datasets, all 50 calibration-test splits, and all 500 assigned budgets. It evaluates every cost-ordered five-model subsequence from the eight-model pool, 56 per split and 14,000 subsequence searches in total. The comparison contains 125,000 matched split-budget positions. LiveCodeBench uses the current eight-model inputs including GPT-oss-20B.

Each S5 selection is the best calibration-feasible policy among the existing FOC-selected S4 policies and the newly optimized five-model sequences. No exhaustive policy initializes or guides the search. The earlier S2-S4 results and all exact-search caches remain unchanged. No manuscript files were edited.

## Method

This run uses the same local FOC-guided optimizer as the S3/S4 experiment. Its settings are unchanged:

- Boundary benefit-cost ratios use up to 100 score-ordered reached calibration observations, with at least 50 required for estimation.
- Individual threshold updates consider the current threshold, feasible endpoints, the largest budget-feasible threshold, and estimated positive-to-negative benefit-ratio crossings.
- Budget transfers reduce escalation at a lower-return boundary and increase it at a higher-return boundary. Proposed donor changes cover 1%, 5%, or 20% of score groups.
- Candidate policies are accepted only when calibration accuracy improves, or when accuracy is preserved and cost falls. All accepted policies satisfy the calibration budget.
- Each start receives at most six sweeps. Budgets use warm starts from the preceding selected policy. Four additional common-quantile starts are tried at the first feasible budget and every 25th budget index.
- Thresholds preserve tied scores and explicitly include always-stop and always-continue rules. Sparse and unreachable stages retain endpoint handling even when boundary ratios cannot be estimated.

The optimizer remains local. Boundary estimation, empirical threshold steps, finite restarts, and the sweep limit can leave gaps to the exhaustive optimum. The detailed algorithm is documented in the preceding report at `results/foc_subsequence/REPORT.md`.

## S5 results

Calibration gaps are exhaustive minus FOC, in accuracy percentage points. Test differences are FOC minus exhaustive. Costs are dollars per 1,000 incoming queries. Within each split, differences are integrated over normalized assigned budgets, then averaged equally across splits. Percentiles describe variability across overlapping splits, not confidence intervals.

| Dataset | Mean calibration gap (pp) | Mean test accuracy difference (pp) | Test difference p10 to p90 | Mean test cost difference ($/1,000) | Cost difference (% assigned budget) | Calibration optimum attained (% budget weight) |
|---|---:|---:|---:|---:|---:|---:|
'''
for dataset in DATASETS:
    r=means.loc[(dataset,5)]
    text+=f'| {NAMES[dataset]} | {r.cal_gap_pp:.4f} | {r.accuracy_difference:+.4f} | {r.accuracy_p10:+.4f} to {r.accuracy_p90:+.4f} | {r.cost_difference:+.6f} | {r.cost_pct_budget:+.3f} | {100*r.cal_optimal_fraction:.2f} |\n'
text+='''
These signed test differences compare two calibration-selected policies at the same assigned budgets. They do not measure an advantage at identical realized test costs.

## Complete comparison through five models

Mean calibration accuracy gaps below exhaustive search, in percentage points:

| Dataset | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|
'''
for dataset in DATASETS:
    text+='| '+NAMES[dataset]+' | '+' | '.join(f'{means.loc[(dataset,d)].cal_gap_pp:.4f}' for d in [2,3,4,5])+' |\n'
text+='''
Mean held-out accuracy differences from the corresponding exhaustive class, in percentage points:

| Dataset | S2 | S3 | S4 | S5 |
|---|---:|---:|---:|---:|
'''
for dataset in DATASETS:
    text+='| '+NAMES[dataset]+' | '+' | '.join(f'{means.loc[(dataset,d)].accuracy_difference:+.4f}' for d in [2,3,4,5])+' |\n'
text+='''
## What allowing a fifth model changes

| Dataset | Calibration gain over FOC S4 (pp) | Test gain over FOC S4 (pp) | Test gain over FOC S2 (pp) | Test gain over exhaustive S2 (pp) | Nominal five-model selection (% budget weight) |
|---|---:|---:|---:|---:|---:|
'''
for dataset in DATASETS:
    r=means.loc[(dataset,5)]
    text+=f'| {NAMES[dataset]} | {r.cal_gain_vs_foc4_pp:+.4f} | {r.gain_vs_foc4_pp:+.4f} | {r.gain_vs_foc2_pp:+.4f} | {r.gain_vs_exact2_pp:+.4f} | {100*r.full_depth_fraction:.2f} |\n'
text+='''
Nominal depth counts models in the selected sequence, not calls made on every query. Calibration nestedness is checked exactly, but held-out performance need not be ordered by depth.

## Runtime

The table reports mean measured selection-kernel time per split. The total adds the stored lower-depth selection time to the new five-model kernel time. Dataset processes run concurrently, and the lower-depth runs occurred separately, so these are descriptive workload-dependent timings. Loading, persistence, replay validation, and one-time compilation are excluded from kernel times. No fresh exhaustive S5 timing benchmark was run, and these results do not establish a new S5 speedup factor.

| Dataset | Five-model extension (s/split) | Total through S5 (s/split) |
|---|---:|---:|
'''
for dataset in DATASETS:
    t=timing[timing.dataset==dataset].mean(numeric_only=True)
    text+=f'| {NAMES[dataset]} | {t.extension_kernel_seconds:.3f} | {t.total_selection_kernel_seconds:.3f} |\n'
text+='''
## Validation

- Unit tests now independently enumerate small problems through five models, checking feasibility and that the FOC objective does not exceed exhaustive search.
- Every returned five-model subsequence policy is independently replayed on calibration data. Every selected S5 policy is also replayed before testing.
- All 125,000 selected policies satisfy their calibration budgets and have calibration accuracy at least as high as the corresponding FOC S4 selection and no higher than exhaustive S5.
- Exact S5 calibration and test policies are independently replayed from the saved inputs. Costs and accuracies match the cached comparator within numerical tolerance.
- Input fingerprints, source hashes, code hashes, parent-policy hashes, split indices, and hyperparameters are recorded for each split. Selected policies are persisted before test evaluation.
- The report verifies the complete 500-budget grid and nested FOC calibration performance through S5 for every split.

## Reproduction and files

```sh
.venv/bin/python -m unittest test_foc_subsequence_compare -v
.venv/bin/python foc_five_compare.py
.venv/bin/python foc_five_report.py
```

The extension requires the existing FOC S4 policies under `results/foc_subsequence`. Per-budget comparisons and frozen S5 policies are stored by dataset. `split_summary.csv` and `summary.csv` contain the combined S2-S5 comparison. `timings.csv` records per-split computational statistics. The first-split pilot is retained separately under `results/foc_five_pilot`.
'''
(OUT/'REPORT.md').write_text(text)
print(means.xs(5,level='depth')[['cal_gap_pp','accuracy_difference','cost_difference','cal_optimal_fraction','gain_vs_foc4_pp']].to_string())
