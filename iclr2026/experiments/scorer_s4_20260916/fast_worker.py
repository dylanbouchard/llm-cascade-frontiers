import os,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ['TIKTOKEN_CACHE_DIR']=str(HERE/'tokenizer-cache')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['MPLCONFIGDIR']='/tmp/scorer-s4-mpl'
ROOT=Path(__file__).resolve().parents[2]
dataset,scorer=sys.argv[1:3]
import fcntl
lockdir=HERE/'locks';lockdir.mkdir(exist_ok=True)
lockfile=(lockdir/f'{dataset}-{scorer}.lock').open('w')
fcntl.flock(lockfile,fcntl.LOCK_EX)
source=ROOT/'experiments/livecodebench_8model_20260912' if dataset=='livecodebench' else ROOT
os.chdir(source);sys.path.insert(0,str(source))
import numpy as np,pandas as pd
import scorer_depth_compute as sd
from fig2_compute import MODEL_COST_ORDER
asset=source/'results/exact_heldout_five/inputs'/f'{dataset}.npz'
def slim_load(d):
 assert d==dataset
 with np.load(asset) as z:arrays={k:z[k] for k in z.files}
 ix={str(m):i for i,m in enumerate(arrays['models'])}
 raw={};costs={}
 for model in MODEL_COST_ORDER:
  if model not in ix:continue
  frame=pd.read_parquet(source/'data/output_data'/f'{d}-{model}.parquet',columns=['prompt','correct']+sd.BASE_SCORERS).iloc[:2000].reset_index(drop=True)
  i=ix[model]
  np.testing.assert_array_equal(frame['correct'].to_numpy(float),arrays['correct'][i])
  np.testing.assert_array_equal(frame['mean_token_negentropy'].to_numpy(float),arrays['scores'][i])
  raw[model]=frame;costs[model]=arrays['costs'][i]
 assert len(raw)==8
 return raw,costs
sd.load_full=slim_load
metadata=HERE/'loader_provenance';metadata.mkdir(exist_ok=True)
(metadata/f'{dataset}-{scorer}.json').write_text(json.dumps({'loader_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cost_asset':str(asset),'cost_asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'validation':'Exact scorer-depth data fingerprint checked by run.worker, including prompts, correctness, all base scores, costs, and embeddings.'},indent=2)+'\n')
sys.path.insert(0,str(HERE))
import run
run.worker(dataset,scorer,50)
