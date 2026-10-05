"""Fresh sequential split-0 timing comparison of FOC and exact S3/S4."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/foc-subsequence-mpl')
import json
from itertools import combinations
from time import perf_counter
import numpy as np
import pandas as pd
from foc_subsequence_compare import chain_search, kernel
from foc_pairwise_compare import ROOT, DATASETS, source, sliced, select
from exact_heldout_depth import exact_select
from exact_heldout_four import select_four
from exact_four_model import kernel as exact_kernel

out=ROOT/'results/foc_subsequence'
kernel();exact_kernel()
rows=[]
for dataset in DATASETS:
    src=source(dataset)
    with np.load(src/'inputs'/f'{dataset}.npz') as z: arrays={k:z[k] for k in z.files}
    manifest=json.loads((src/dataset/'split_00.json').read_text())
    cal=sliced(arrays,manifest['cal_idx']);budgets=np.array(manifest['budgets'])
    start=perf_counter();frozen,_=exact_select(cal,budgets);e3=perf_counter()-start
    start=perf_counter();four,st=select_four(cal,budgets,frozen['3']);e4=e3+perf_counter()-start
    for d,policies in [(3,frozen['3']),(4,four)]:
        np.testing.assert_allclose([p['cal_quality'] for p in policies],[p['cal_quality'] for p in manifest['frozen'][str(d)]],atol=1e-12,rtol=0)
        np.testing.assert_allclose([p['cal_cost'] for p in policies],[p['cal_cost'] for p in manifest['frozen'][str(d)]],atol=1e-12,rtol=0)
    start=perf_counter();p,_,_=select(cal,budgets,'foc',100)
    q,c=p['cal_accuracy'].copy(),p['cal_cost'].copy()
    models=sorted(cal,key=lambda m:(cal[m]['costs'].mean(),m))
    times={}
    for d in [3,4]:
        for seq in combinations(models,d):
            _,cc,qq,_,_=chain_search(cal,seq,budgets,100,6,25)
            replace=(cc<=budgets+1e-12)&((qq>q+1e-12)|((abs(qq-q)<1e-12)&(cc<c-1e-15)))
            q[replace],c[replace]=qq[replace],cc[replace]
        times[d]=perf_counter()-start
        with np.load(out/dataset/'split_00_policies.npz') as saved:
            np.testing.assert_allclose(q,saved[f'd{d}_cal_accuracy'],atol=1e-12,rtol=0)
            np.testing.assert_allclose(c,saved[f'd{d}_cal_cost'],atol=1e-12,rtol=0)
    for d,e in [(3,e3),(4,e4)]: rows.append(dict(dataset=dataset,split=0,depth=d,exact_seconds=e,foc_seconds=times[d],exact_over_foc=e/times[d]))
    pd.DataFrame(rows).to_csv(out/'fresh_timing_benchmark.csv',index=False)
    print(dataset,rows[-2:],flush=True)
