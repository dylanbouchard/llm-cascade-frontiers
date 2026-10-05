from pathlib import Path
import json,hashlib,re
import numpy as np
import pandas as pd
from run import ROOT,HERE,DATASETS,SCORERS
records=[]
for d in DATASETS:
 for s in SCORERS:
  for i in range(50):
   p=HERE/'results'/d/s/f'split_{i:02d}.csv'
   f=pd.read_csv(p);m=json.loads(p.with_suffix('.json').read_text())
   assert len(f)==500 and len(m['pool'])==8
   assert (m['dataset'],m['scorer'],m['seed'])==(d,s,42+i)
   assert not set(m['cal_idx'])&set(m['test_idx'])
   assert m['search_stats']['quintuples']==56
   assert hashlib.sha256(Path(m['baseline_path']).read_bytes()).hexdigest()==m['baseline_sha256']
   np.testing.assert_allclose(f.budget,m['budgets'],atol=1e-15,rtol=0)
   np.testing.assert_allclose(f.fraction,np.linspace(0,1,500),atol=1e-14,rtol=0)
   assert (f.d5_cal_accuracy>=f.d4_cal_accuracy-1e-12).all()
   assert (f.d5_cal_cost<=f.budget+1e-12).all()
   for metric,key in [('accuracy','cal_quality'),('cost','cal_cost')]:
    np.testing.assert_allclose(f[f'd5_cal_{metric}'],[p[key] for p in m['frozen']['5']],atol=1e-12,rtol=0)
   np.testing.assert_array_equal(f.d5_selected_depth,[len(p['models']) for p in m['frozen']['5']])
   base=pd.read_csv(Path(m['baseline_path']).with_suffix('.csv'))
   pd.testing.assert_frame_equal(f[base.columns],base)
   records.append(dict(dataset=d,scorer=s,split=i,s5_accuracy_pp=100*np.trapz(f.d5_accuracy-f.d2_accuracy,f.fraction),s5_minus_s4_pp=100*np.trapz(f.d5_accuracy-f.d4_accuracy,f.fraction),kernel_seconds=m['search_stats']['kernel_seconds']))
f=pd.DataFrame(records);f.to_csv(HERE/'split_integrals.csv',index=False)
s=f.groupby(['dataset','scorer']).agg(splits=('split','count'),s5_accuracy_pp=('s5_accuracy_pp','mean'),s5_minus_s4_pp=('s5_minus_s4_pp','mean'),kernel_seconds=('kernel_seconds','sum'))
s.to_csv(HERE/'scorer_s5_summary.csv')
p=ROOT/'paper/tables/table_exact_depth.tex'
lines=p.read_text().splitlines();d=-1
for i,line in enumerate(lines):
 if '$S_3-S_2$' in line:d+=1
 if '$S_5-S_2$' in line:
  cells=line.split(' & ')
  # Mean negentropy S5 was computed in the prior exact run. Preserve that
  # manuscript value and fill the six newly evaluated scorer columns.
  lines[i]=' & '.join(cells[:3]+[f'${s.loc[(DATASETS[d],scorer),"s5_accuracy_pp"]:+.3f}$' for scorer in SCORERS])+r' \\'
(HERE/'table_exact_depth.tex').write_text('\n'.join(lines)+'\n')
print(s.to_string())
print('Validated 1,500 split results and 750,000 budget positions.')
