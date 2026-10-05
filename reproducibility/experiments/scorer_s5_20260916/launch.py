import sys,subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from run import DATASETS,SCORERS
H=Path(__file__).resolve().parent

def job(d,s):
 with (H/f'{d}-{s}.log').open('a') as f:
  p=subprocess.run([sys.executable,'-B',str(H/'fast_worker.py'),d,s],stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:raise RuntimeError(f'{d}/{s} failed; see log')
 print('COMPLETE',d,s,flush=True)

if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=8) as pool:
  jobs=[pool.submit(job,d,s) for d in DATASETS for s in SCORERS]
  for j in as_completed(jobs):j.result()
