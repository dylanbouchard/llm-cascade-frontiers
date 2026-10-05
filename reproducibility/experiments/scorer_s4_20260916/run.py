import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']='/tmp/scorer-s4-mpl'
import sys, json, hashlib, time, argparse, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
SCORERS=['mean_token_negentropy','min_token_negentropy','probability_margin','min_probability','sequence_probability','logreg_ensemble','response_logreg']
DATASETS=['mmlu','triviaqa','math_hard','simpleqa','livecodebench']
def worker(dataset,scorer,limit):
 source=ROOT/'experiments/livecodebench_8model_20260912' if dataset=='livecodebench' else ROOT
 os.chdir(source);sys.path.insert(0,str(source))
 import numpy as np
 import pandas as pd
 from scipy.special import expit
 import scorer_depth_compute as sd
 from exact_heldout_four import select_four, policy_key
 from exact_heldout_depth import evaluate_policy, encode
 raw,costs,rows,embeddings,fingerprint=sd.prepare(dataset,True)
 assert len(raw)==8
 rowmap={int(r):i for i,r in enumerate(rows)}
 codehash=hashlib.sha256(Path(__file__).read_bytes()+(source/'exact_four_kernel.cpp').read_bytes()+(source/'exact_heldout_four.py').read_bytes()).hexdigest()
 for split in range(limit):
  start=time.monotonic()
  base=source/'results/scorer_depth'/dataset/scorer/f'split_{split:02d}.json'
  m=json.loads(base.read_text())
  assert m['data_sha256']==fingerprint,(dataset,scorer,'data mismatch')
  assert set(m['pool'])==set(raw)
  ci=np.array([rowmap[r] for r in m['cal_idx']]);ti=np.array([rowmap[r] for r in m['test_idx']])
  assert not set(ci)&set(ti)
  dest=HERE/'results'/dataset/scorer/base.name;dest.parent.mkdir(parents=True,exist_ok=True)
  basehash=hashlib.sha256(base.read_bytes()).hexdigest()
  if dest.with_suffix('.csv').exists():
   prior=json.loads(dest.read_text());assert prior['code_sha256']==codehash and prior['baseline_sha256']==basehash
   continue
  scores={}
  for model,df in raw.items():
   if scorer in sd.BASE_SCORERS:scores[model]=df[scorer].to_numpy(float)
   else:
    fit=m['fitted_scorers'][model]
    if 'fallback' in fit:scores[model]=df[fit['fallback']].to_numpy(float)
    else:
     assert fit['classes']==[0,1]
     x=embeddings[model] if scorer==sd.RESPONSE_SCORER else df[sd.BASE_SCORERS].to_numpy(float)
     scores[model]=expit((x@np.asarray(fit['coef']).T+np.asarray(fit['intercept'])).ravel())
  def sliced(idx):return {model:dict(scores=scores[model][idx],correct=df.correct.to_numpy(float)[idx],costs=costs[model][idx]) for model,df in raw.items()}
  cal=sliced(ci)
  # Validate stored scorer reconstruction against every unique frozen baseline witness.
  for depth in ('2','3'):
   for p in {policy_key(p):p for p in m['frozen'][depth]}.values():
    c,q,_=evaluate_policy(p,cal);np.testing.assert_allclose([c,q],[p['cal_cost'],p['cal_quality']],rtol=0,atol=1e-12)
  selected,stats=select_four(cal,m['budgets'],m['frozen']['3'])
  for p in {policy_key(p):p for p in selected}.values():
   c,q,_=evaluate_policy(p,cal);np.testing.assert_allclose([c,q],[p['cal_cost'],p['cal_quality']],rtol=0,atol=1e-12)
  manifest=dict(m,code_sha256=codehash,baseline_sha256=basehash,baseline_path=str(base),search_stats=stats)
  manifest['frozen']=dict(m['frozen'],**{'4':selected})
  dest.write_text(json.dumps(encode(manifest),allow_nan=False)+'\n')
  test=sliced(ti);frame=pd.read_csv(base.with_suffix('.csv'))
  np.testing.assert_allclose(frame.budget,m['budgets'],rtol=0,atol=1e-15)
  memo={}
  def replay(p):
   key=policy_key(p)
   if key not in memo:memo[key]=evaluate_policy(p,test)
   return memo[key]
  for depth in ('2','3'):
   values=np.array([replay(p) for p in m['frozen'][depth]])
   np.testing.assert_allclose(values[:,0],frame[f'd{depth}_cost'],rtol=0,atol=1e-12)
   np.testing.assert_allclose(values[:,1],frame[f'd{depth}_accuracy'],rtol=0,atol=1e-12)
  vals=np.array([replay(p) for p in selected])
  frame['d4_cost']=vals[:,0];frame['d4_accuracy']=vals[:,1];frame['d4_mean_calls']=vals[:,2]
  frame['d4_cal_cost']=[p['cal_cost'] for p in selected];frame['d4_cal_accuracy']=[p['cal_quality'] for p in selected]
  frame['d4_selected_depth']=[len(p['models']) for p in selected]
  assert (frame.d4_cal_accuracy>=frame.d3_cal_accuracy-1e-12).all()
  assert (frame.d4_cal_cost<=frame.budget+1e-12).all()
  temp=dest.with_suffix('.csv.tmp');frame.to_csv(temp,index=False);temp.replace(dest.with_suffix('.csv'))
  print(f'{dataset}/{scorer} {split+1}/{limit} {time.monotonic()-start:.1f}s',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset');p.add_argument('--scorer');p.add_argument('--splits',type=int,default=50);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 if a.dataset:worker(a.dataset,a.scorer,a.splits)
 else:
  from concurrent.futures import ThreadPoolExecutor,as_completed
  def dispatch(d,s):
   log=HERE/f'{d}-{s}.log'
   with log.open('w') as f:
    r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--dataset',d,'--scorer',s,'--splits',str(a.splits)],stdout=f,stderr=subprocess.STDOUT)
   if r.returncode:raise RuntimeError(f'{d}/{s}: see {log}')
   print(f'COMPLETE {d}/{s}',flush=True)
  with ThreadPoolExecutor(max_workers=a.workers) as pool:
   jobs=[pool.submit(dispatch,d,s) for d in DATASETS for s in SCORERS]
   for job in as_completed(jobs):job.result()
