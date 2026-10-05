"""Write a report from the FOC pairwise experiment, without editing the paper."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/foc_pairwise'
NAMES={'mmlu':'MMLU','triviaqa':'TriviaQA','math_hard':'MATH','simpleqa':'SimpleQA','livecodebench':'LiveCodeBench'}
summary=pd.read_csv(OUT/'summary.csv').set_index(['dataset','method'])
time=pd.read_csv(OUT/'timing_summary.csv').set_index(['dataset','method'])
trap=getattr(np,'trapezoid',np.trapz)
text='''# FOC-guided pairwise selection versus exhaustive S2

## Run and inputs

Completed 50 calibration-test splits and 500 assigned budgets per split on each of five datasets, for 125,000 matched split-budget comparisons per method. Every dataset uses all eight models and all 28 calibration-cost-ordered pairs, together with standalone policies. Mean token negentropy is the confidence signal.

The experiment uses the exact input arrays, split indices, and budget vectors behind the current paper. The first four datasets use `results/exact_heldout_five`. LiveCodeBench uses `experiments/livecodebench_8model_20260912/results/exact_heldout_five`, which includes GPT-oss-20B. It does not use the older seven-model LiveCodeBench cache.

Input fingerprints, source hashes, code hashes, calibration/test indices, and model sequence lists are recorded in per-split JSON files. Selected sequence indices and thresholds are saved in NPZ files before test evaluation. No manuscript files or existing result caches were changed.

## What the FOC optimizer does

For a fixed pair, let g(s) be expected quality gain from escalation and gamma(s) its expected additional cost. The derivatives are Q'(tau)=f(tau)g(tau) and C'(tau)=f(tau)gamma(tau). Since cost increases with the threshold, a budget restricts the feasible threshold interval.

There is only one threshold, so there is no allocation across multiple boundaries. An optimum is either at an endpoint of the feasible interval or at an interior quality maximum. An interior optimum with a slack budget has lambda=0 and hence g(tau)=0. An active-budget solution can instead have g(tau)=lambda gamma(tau)>0. Searching only for zero benefit would miss these active-budget solutions.

The FOC method checks:

1. Never escalate and always escalate when feasible.
2. The largest empirical threshold partition satisfying the calibration budget.
3. Thresholds bracketing estimated positive-to-negative crossings of g/gamma, including the ends of zero plateaus.

The estimate is the ratio of summed escalation gain to summed escalation cost in a fixed window of up to 100 score-ordered calibration observations around each boundary. Endpoint windows are one-sided. Windows use score order, rather than equal distances in raw score. Threshold partitions preserve tied scores. The window was fixed before the pilot and was not tuned on test results.

Within each pair, candidates are selected by actual calibration accuracy, with lower calibration cost breaking ties. The outer optimization applies the same rule across all pairs and standalone models. This is a smoothed FOC-candidate method, not a guarantee of global optimality on the empirical step function.

The exhaustive comparator considers every distinct calibration threshold partition, using cumulative sums. A boundary-only control uses the same outer optimization and feasible endpoints but omits stationary-point candidates. It measures what the estimated FOC roots add beyond saturating the budget.

## Results against exhaustive S2

All differences below are FOC minus exhaustive, except the calibration gap, which is exhaustive minus FOC. Accuracy units are percentage points. Cost units are dollars per 1,000 incoming queries. Values integrate over the 500 normalized budget positions within each split, then average splits equally. Percentiles describe variation across overlapping splits, not confidence intervals.

| Dataset | Mean calibration gap (pp) | Mean test accuracy difference (pp) | Test difference p10 to p90 | Mean test cost difference ($/1,000) | Cost difference (% assigned budget) | Calibration optimum attained (% budget weight) |
|---|---:|---:|---:|---:|---:|---:|
'''
for d,name in NAMES.items():
    r=summary.loc[(d,'foc')]
    relative=[]
    for path in sorted((OUT/d).glob('split_*.csv')):
        f=pd.read_csv(path)
        relative.append(trap(100*(f.foc_test_cost-f.exact_test_cost)/f.budget,f.fraction))
    text+=f"| {name} | {-r.cal_accuracy_difference:.4f} | {r.test_accuracy_difference:+.4f} | {r.test_accuracy_p10:+.4f} to {r.test_accuracy_p90:+.4f} | {r.test_cost_difference:+.6f} | {np.mean(relative):+.3f} | {100*r.cal_optimal_fraction:.2f} |\n"
text+='''
The method is close to the exhaustive calibration optimum on average, but does not recover it at every budget. Its higher average held-out accuracy comes with higher average realized cost on all five datasets. This is not evidence of an improvement at equal realized cost. FOC selection also spends more calibration cost on average, so some of the difference is already present at selection time.

## What stationary-point candidates add

The calibration gap is reduced on four datasets and unchanged on LiveCodeBench. Test accuracy does not consistently improve over the boundary-only control.

| Dataset | Boundary-only calibration gap (pp) | FOC calibration gap (pp) | FOC minus boundary-only test accuracy (pp) |
|---|---:|---:|---:|
'''
for d,name in NAMES.items():
    f,b=summary.loc[(d,'foc')],summary.loc[(d,'boundary')]
    text+=f'| {name} | {-b.cal_accuracy_difference:.4f} | {-f.cal_accuracy_difference:.4f} | {f.test_accuracy_difference-b.test_accuracy_difference:+.4f} |\n'
text+='''
## Runtime

Times are mean selection time per split for all pairs, all standalone models, and all 500 budgets. They include sorting, cumulative-sum construction, smoothing where applicable, and policy selection. Loading inputs, persistence, and held-out evaluation are excluded. Method order rotates across splits. These millisecond timings describe this implementation and machine, not a rigorous hardware benchmark.

| Dataset | Exhaustive (ms/split) | FOC (ms/split) | FOC / exhaustive |
|---|---:|---:|---:|
'''
for d,name in NAMES.items():
    e,f=time.loc[(d,'exact'),'seconds'],time.loc[(d,'foc'),'seconds']
    text+=f'| {name} | {1000*e:.3f} | {1000*f:.3f} | {f/e:.2f} |\n'
text+='''
The FOC implementation does not provide a runtime advantage here. Both methods construct score-sorted cumulative sums. Exhaustive S2 then obtains all budget optima with an efficient prefix maximum, while FOC adds boundary estimation. The reduced number of admissible candidates does not imply a corresponding reduction in preprocessing or runtime. The `candidate_budget_memberships` diagnostic counts conceptual feasible candidate-budget combinations, not literal objective evaluations or loop iterations.

For k=2, using raw empirical benefit increments instead of smoothing would bring stationary-point search close to another implementation of the exact threshold sweep. A potential computational advantage in higher-dimensional threshold allocation is not established by this run.

## Validation

- All four unit tests passed, covering tied scores, variable costs, stationary crossings, active-budget solutions, and an independent brute-force comparison.
- Every selected policy was replayed on calibration data and verified budget-feasible before test evaluation.
- The independently computed exhaustive S2 calibration quality, calibration cost, held-out quality, and held-out cost reproduce the saved paper comparator at all 125,000 split-budget positions within numerical tolerance.
- FOC calibration accuracy never exceeds exhaustive calibration accuracy, and never falls below the boundary-only control.
- All policies were persisted before their held-out evaluation.

## Reproduction

```sh
.venv/bin/python -m unittest test_foc_pairwise_compare -v
.venv/bin/python foc_pairwise_compare.py
.venv/bin/python foc_pairwise_report.py
```

Per-budget results are in dataset subdirectories. `split_summary.csv` retains paired split-level aggregates, `summary.csv` contains dataset summaries, and `timings.csv` records each measured selection run. The one-split pilot is retained separately in `results/foc_pairwise_pilot`.
'''
(OUT/'REPORT.md').write_text(text)
print(text)
