"""Aggregate the completed FOC-guided S3/S4 experiment."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from foc_pairwise_compare import ROOT, DATASETS
OUT=ROOT/'results/foc_subsequence'
NAMES=dict(zip(DATASETS,['MMLU','TriviaQA','MATH','SimpleQA','LiveCodeBench']))
trap=getattr(np,'trapezoid',np.trapz)
rows=[];timings=[]
for dataset in DATASETS:
    for split in range(50):
        path=OUT/dataset/f'split_{split:02d}.csv'
        frame=pd.read_csv(path);m=json.loads(path.with_suffix('.json').read_text())
        assert len(frame)==500 and m['budget_indices']==list(range(500))
        f=frame.fraction
        for d in [3,4]:
            row=dict(dataset=dataset,split=split,depth=d)
            for stat,scale in [('accuracy',100),('cost',1000),('cal_accuracy',100),('cal_cost',1000)]:
                row[stat+'_difference']=float(trap(scale*(frame[f'foc{d}_{stat}']-frame[f'exact{d}_{stat}']),f))
            row['cal_gap_pp']=-row['cal_accuracy_difference']
            row['cost_pct_budget']=float(trap(100*(frame[f'foc{d}_cost']-frame[f'exact{d}_cost'])/frame.budget,f))
            row['cal_optimal_fraction']=float(trap(np.isclose(frame[f'foc{d}_cal_accuracy'],frame[f'exact{d}_cal_accuracy'],atol=1e-12,rtol=0).astype(float),f))
            row['max_cal_gap_pp']=float(100*(frame[f'exact{d}_cal_accuracy']-frame[f'foc{d}_cal_accuracy']).max())
            row['gain_vs_foc2_pp']=float(trap(100*(frame[f'foc{d}_accuracy']-frame.foc2_accuracy),f))
            row['gain_vs_exact2_pp']=float(trap(100*(frame[f'foc{d}_accuracy']-frame.exact2_accuracy),f))
            row['nominal_full_depth_fraction']=float(trap((frame[f'foc{d}_depth']==d).astype(float),f))
            rows.append(row)
            cs=[t for t in m['chain_timings'] if t['depth']<=d]
            timings.append(dict(dataset=dataset,split=split,depth=d,seconds=m['pair_stats']['seconds']+sum(t['seconds'] for t in cs),accepted_transfers=sum(t['accepted_transfers'] for t in cs),starts=sum(t['starts'] for t in cs),coordinate_slices=sum(t['coordinate_slices'] for t in cs)))
        for d in [3,4]:assert (frame[f'foc{d}_cal_accuracy']>=frame[f'foc{d-1}_cal_accuracy']-1e-12).all()
summary=pd.DataFrame(rows);summary.to_csv(OUT/'split_summary.csv',index=False)
group=summary.groupby(['dataset','depth']);means=group.mean(numeric_only=True)
for name,quant in [('p10',.1),('p90',.9)]:means['accuracy_'+name]=group.accuracy_difference.quantile(quant)
means.to_csv(OUT/'summary.csv')
pd.DataFrame(timings).to_csv(OUT/'timings.csv',index=False)
text='''# FOC-guided S3 and S4 versus exhaustive cascade selection

## Completed experiment

All five datasets, 50 saved calibration-test splits per dataset, and all 500 assigned budgets per split. Each depth comparison contains 125,000 matched split-budget positions. All eight models are available, including GPT-oss-20B on LiveCodeBench. The input sources and budget ranges are the same as the current paper and the preceding pairwise experiment.

Every cost-ordered triple (56 per split) and quadruple (70 per split) is optimized. The selected S3 policy is the best calibration-feasible result among triples and the FOC-selected S2 class. S4 additionally includes all quadruples and the selected S3 policies. Exhaustive S3 and S4 are the saved exact comparators. No exact policy or test observation initializes or guides the FOC search.

No manuscript files or existing experiment caches were changed.

## Threshold optimizer

The method is a local FOC-guided candidate search, not a global solver. It extends the k=2 method in two ways.

First, for each threshold with the others fixed, it constructs expected cost and quality contributions from queries reaching that stage and the current downstream policy. It considers the current threshold, always-stop and always-continue endpoints, the largest budget-feasible threshold, and thresholds bracketing positive-to-negative crossings of the estimated boundary benefit-cost ratio. This is the coordinate version of the KKT candidate calculation used for pairs.

Second, it uses the FOC allocation implication. When one active boundary has a larger estimated marginal quality per dollar, it proposes decreasing escalation at a lower-return boundary and increasing it at the higher-return boundary. Donor moves are 1%, 5%, or 20% of that model's score groups. The receiving threshold is selected from its FOC candidates under the resulting budget. The full proposed policy must improve calibration accuracy, or preserve accuracy while reducing cost, to be accepted. All candidates obey the calibration budget.

Ratios use up to 100 score-ordered calibration queries that reach the boundary, with at least 50 required for estimation. They condition on the upstream thresholds and include the actual current suffix quality and cost. No score-cost independence assumption is used. Sparse or unreachable boundaries do not support estimated-ratio transfers, but feasible endpoint candidates remain available. Routing preserves tied scores, and endpoint thresholds explicitly implement never and always escalation.

The optimizer performs at most six sweeps per start. It stops earlier if a sweep produces no accepted improvement. Budgets are processed in increasing order, using the preceding budget's chosen thresholds as a warm start. At the first feasible budget and every 25th budget index, four additional starts set thresholds at common 25%, 50%, 75%, and 100% score-group ranks. Each start is made feasible through the first threshold before further optimization. Thus multiple starts occur at scheduled budget anchors, not at every budget. Hyperparameters were fixed before the full run and were not tuned on held-out results.

## Performance against exhaustive search

Accuracy differences are FOC minus exhaustive, in percentage points. Calibration gaps are exhaustive minus FOC. Costs are dollars per 1,000 incoming queries. Each result integrates over normalized assigned budgets within a split and averages the 50 splits equally. The p10 and p90 values describe overlapping-split variability, not confidence intervals.

| Dataset | Class | Mean calibration gap (pp) | Mean test accuracy difference (pp) | Test difference p10 to p90 | Mean test cost difference ($/1,000) | Cost difference (% assigned budget) | Calibration optimum attained (% budget weight) |
|---|---|---:|---:|---:|---:|---:|---:|
'''
for dataset in DATASETS:
    for d in [3,4]:
        r=means.loc[(dataset,d)]
        text+=f'| {NAMES[dataset]} | S{d} | {r.cal_gap_pp:.4f} | {r.accuracy_difference:+.4f} | {r.accuracy_p10:+.4f} to {r.accuracy_p90:+.4f} | {r.cost_difference:+.6f} | {r.cost_pct_budget:+.3f} | {100*r.cal_optimal_fraction:.2f} |\n'
text+='''
The calibration gap measures failure to recover the exhaustive empirical optimum. Signed test differences compare two calibration-selected policies, not a test-optimized frontier. Realized cost can differ at the same assigned budget, so a positive accuracy difference is not by itself an equal-cost advantage.

## Does the selected depth improve on pairs?

| Dataset | Class | Test gain over FOC S2 (pp) | Test gain over exhaustive S2 (pp) | Full nominal depth selected (% budget weight) |
|---|---|---:|---:|---:|
'''
for dataset in DATASETS:
    for d in [3,4]:
        r=means.loc[(dataset,d)]
        text+=f'| {NAMES[dataset]} | S{d} | {r.gain_vs_foc2_pp:+.4f} | {r.gain_vs_exact2_pp:+.4f} | {100*r.nominal_full_depth_fraction:.2f} |\n'
text+='''
## Timing

The main run records selection-kernel times for all subsequences. Dataset runs execute concurrently, so these are workload-dependent elapsed times. A separate fresh, sequential benchmark on split 0 recomputes both exhaustive and FOC S3/S4 from the same inputs. It includes lower-depth selection in each class and excludes file I/O, compilation, and held-out evaluation. This one-split benchmark is illustrative, not a general runtime guarantee.

'''
bench=OUT/'fresh_timing_benchmark.csv'
if bench.exists():
    b=pd.read_csv(bench)
    text+='| Dataset | Class | Exhaustive (s) | FOC (s) | Exhaustive / FOC |\n|---|---|---:|---:|---:|\n'
    for _,r in b.iterrows():text+=f'| {NAMES[r.dataset]} | S{int(r.depth)} | {r.exact_seconds:.3f} | {r.foc_seconds:.3f} | {r.exact_over_foc:.2f} |\n'
else:text+='The fresh sequential benchmark has not yet been completed.\n'
text+='''
## Validation and limits

- Unit tests independently enumerate small three- and four-model problems and check that FOC policies are feasible and cannot exceed the exhaustive objective. Tests also cover tied scores, unreachable suffixes, infeasible budgets, and terminal endpoint solutions.
- Every returned subsequence policy is replayed on calibration data during the full run, and every selected class-level policy is checked for calibration feasibility and correctness before testing.
- Selected FOC calibration accuracy never exceeds the corresponding exhaustive optimum. The FOC-selected classes preserve calibration nestedness from S2 through S4.
- Every exhaustive comparator is independently replayed on the current test inputs and reproduces the stored cost and accuracy at all 125,000 budget positions for each class.
- Policies, code/input/source hashes, split indices, and hyperparameters are saved before test evaluation.
- Estimation, finite threshold steps, local optima, scheduled restarts, and the six-sweep limit can leave calibration gaps. These runs do not establish convergence to population FOC solutions or global optimality.
- Actual calibration quality validates candidate steps. This is an FOC-guided search with acceptance checks, not a procedure using derivatives alone. It is not yet compared with a matched generic local-search optimizer.

## Reproduction

```sh
.venv/bin/python -m unittest test_foc_subsequence_compare -v
.venv/bin/python foc_subsequence_compare.py --verify-chains
.venv/bin/python foc_subsequence_benchmark.py
.venv/bin/python foc_subsequence_report.py
```

Dataset subdirectories contain selected policies, per-budget metrics, and per-split provenance. `split_summary.csv` contains paired aggregates, `summary.csv` summarizes datasets, and `timings.csv` retains per-split optimization statistics. Pilot runs are retained in separate `results/foc_subsequence_pilot` and `results/foc_subsequence_full_pilot` directories.
'''
(OUT/'REPORT.md').write_text(text)
print(means[['cal_gap_pp','accuracy_difference','cost_difference','cal_optimal_fraction']].to_string())
