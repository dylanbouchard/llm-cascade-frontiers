from pathlib import Path
import json,re,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
WORK=HERE.parent
OUT=HERE
ROOT=Path(__file__).resolve().parents[2]
DATASETS=['mmlu','triviaqa','math_hard','simpleqa','livecodebench']
SCORERS=['mean_token_negentropy','min_token_negentropy','probability_margin','min_probability','sequence_probability','logreg_ensemble','response_logreg']
trap=np.trapz
records=[]
for d in DATASETS:
 for s in SCORERS:
  for i in range(50):
   path=HERE/'results'/d/s/f'split_{i:02d}.csv'
   f=pd.read_csv(path);m=json.loads(path.with_suffix('.json').read_text())
   assert len(f)==500 and len(m['pool'])==8
   assert m['dataset']==d and m['scorer']==s and m['seed']==42+i
   assert not set(m['cal_idx'])&set(m['test_idx'])
   assert m['search_stats']['quadruples']==70
   assert hashlib.sha256(Path(m['baseline_path']).read_bytes()).hexdigest()==m['baseline_sha256']
   np.testing.assert_allclose(f.budget,m['budgets'],rtol=0,atol=1e-15)
   np.testing.assert_allclose(f.fraction,np.linspace(0,1,500),rtol=0,atol=1e-14)
   assert (f.d4_cal_accuracy>=f.d3_cal_accuracy-1e-12).all()
   assert (f.d4_cal_cost<=f.budget+1e-12).all()
   np.testing.assert_allclose(f.d4_cal_accuracy,[p['cal_quality'] for p in m['frozen']['4']],rtol=0,atol=1e-12)
   np.testing.assert_allclose(f.d4_cal_cost,[p['cal_cost'] for p in m['frozen']['4']],rtol=0,atol=1e-12)
   np.testing.assert_array_equal(f.d4_selected_depth,[len(p['models']) for p in m['frozen']['4']])
   r=dict(dataset=d,scorer=s,split=i,s3_accuracy_pp=100*trap(f.d3_accuracy-f.d2_accuracy,f.fraction),s4_accuracy_pp=100*trap(f.d4_accuracy-f.d2_accuracy,f.fraction),s4_cost_per_1000=1000*trap(f.d4_cost-f.d2_cost,f.fraction),kernel_seconds=m['search_stats']['kernel_seconds'])
   records.append(r)
f=pd.DataFrame(records)
f.to_csv(HERE/'split_integrals.csv',index=False)
s=f.groupby(['dataset','scorer']).agg(splits=('split','count'),s3_accuracy_pp=('s3_accuracy_pp','mean'),s4_accuracy_pp=('s4_accuracy_pp','mean'),s4_p10_pp=('s4_accuracy_pp',lambda v:v.quantile(.1)),s4_p90_pp=('s4_accuracy_pp',lambda v:v.quantile(.9)),s4_cost_per_1000=('s4_cost_per_1000','mean'),kernel_seconds=('kernel_seconds','sum'))
s.to_csv(OUT/'scorer_s4_summary.csv')
print(s.to_string())
