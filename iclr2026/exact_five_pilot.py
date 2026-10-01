"""Benchmark a single calibration split with exact, checkpointed five-model search."""
import argparse
import ctypes
import hashlib
import json
import subprocess
import time
from functools import lru_cache
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import exact_heldout_depth as baseline
from oracle_depth_analysis import score_ranks
from exact_four_model import ROOT


@lru_cache(maxsize=1)
def kernel():
    source=ROOT/'exact_five_kernel.cpp'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library=Path('/tmp')/f'cascade_exact_five_{digest}.so'
    if not library.exists():
        subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    lib=ctypes.CDLL(str(library))
    f=lib.search_five
    ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    up=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int,ip,ip,dp,ctypes.c_double,ctypes.c_double,dp,ip,up]
    f.restype=ctypes.c_int
    return f


def search_chain(data, seq, budget=np.inf, seconds_limit=0.):
    n=len(data[seq[0]]['correct'])
    if len(seq)!=5 or n==0:
        raise ValueError('Requires five models and nonempty aligned data')
    for m in seq:
        if any(len(data[m][k])!=n or not np.isfinite(data[m][k]).all()
               for k in ('scores','costs','correct')):
            raise ValueError('Requires finite aligned arrays')
    rawq=np.array([data[m]['correct'] for m in seq])
    costs=np.ascontiguousarray([data[m]['costs'] for m in seq],dtype=np.float64)
    if not np.isin(rawq,[0,1]).all() or (costs<0).any():
        raise ValueError('Requires binary correctness and nonnegative costs')
    ranks=np.ascontiguousarray([score_ranks(data[m]['scores'])[0] for m in seq[:4]],dtype=np.int32)
    q=np.ascontiguousarray(rawq,dtype=np.int32)
    best=np.full(n+1,np.inf)
    witness=np.full((n+1,4),-1,dtype=np.int32)
    stats=np.zeros(2,dtype=np.uint64)
    f=kernel()
    start=time.perf_counter()
    complete=f(n,ranks,q,costs,(budget+1e-12)*n,seconds_limit,best,witness,stats)
    return best/n,witness,dict(complete=bool(complete),seconds=time.perf_counter()-start,
                              evaluated=int(stats[0]),outer_slices=int(stats[1]))


def run(args):
    original_path=ROOT/'results/exact_heldout_four'/args.dataset/f'split_{args.split:02d}.json'
    original_bytes=original_path.read_bytes()
    original=json.loads(original_bytes)
    raw,costs=baseline.load_full(args.dataset)
    prompts=next(iter(raw.values()))['prompt'].values
    valid=np.ones(len(prompts),dtype=bool)
    for m,df in raw.items():
        if not np.array_equal(prompts,df['prompt'].values):
            raise ValueError('Misaligned prompts')
        valid &= np.isfinite(df[baseline.SCORER]) & np.isfinite(df['correct']) & np.isfinite(costs[m])
    rows=np.flatnonzero(valid)
    digest=hashlib.sha256()
    for m,df in raw.items():
        filtered=df.iloc[rows].reset_index(drop=True)
        digest.update(m.encode())
        digest.update(pd.util.hash_pandas_object(filtered[['prompt',baseline.SCORER,'correct']],index=False).values.tobytes())
        digest.update(np.asarray(costs[m])[rows].tobytes())
    if digest.hexdigest()!=original['data_sha256']:
        raise ValueError('Baseline data fingerprint mismatch')
    fingerprint=hashlib.sha256(original_bytes+(ROOT/'exact_five_kernel.cpp').read_bytes()+Path(__file__).read_bytes()).hexdigest()
    cal=baseline.slice_data(raw,costs,list(raw),np.array(original['cal_idx']))
    models=sorted(cal,key=lambda m:(cal[m]['costs'].mean(),m))
    seqs=list(combinations(models,5))
    n=len(original['cal_idx'])
    cuts={m:np.r_[-np.inf,np.unique(cal[m]['scores'])[1:],np.inf] for m in models}
    selected=list(original['frozen']['4'])
    budgets=original['budgets']
    dest=args.out/args.dataset/f'split_{args.split:02d}'
    dest.mkdir(parents=True,exist_ok=True)
    timings=[]
    completed=0
    summary={}
    new_search_seconds=0.
    for idx,seq in enumerate(seqs):
        path=dest/f'chain_{idx:02d}.npz'
        if path.exists():
            with np.load(path) as z:
                if str(z['fingerprint'])!=fingerprint:
                    raise ValueError('Incompatible chain cache')
                best,witness=z['best'],z['witness']
                stats=json.loads(str(z['stats']))
        else:
            print(f'{args.dataset}: starting chain {idx+1}/{len(seqs)} {seq}',flush=True)
            best,witness,stats=search_chain(cal,seq,max(budgets),args.chain_seconds)
            new_search_seconds+=stats['seconds']
            for q in np.flatnonzero(np.isfinite(best)):
                p=dict(models=list(seq),thresholds=[float(cuts[m][r]) for m,r in zip(seq,witness[q])])
                c,quality,_=baseline.evaluate_policy(p,cal)
                if abs(c-best[q])>1e-12 or round(quality*n)!=q:
                    raise AssertionError('Witness replay mismatch')
            if stats['complete']:
                temp=path.with_suffix('.tmp.npz')
                np.savez_compressed(temp,fingerprint=fingerprint,models=seq,best=best,witness=witness,stats=json.dumps(stats))
                temp.replace(path)
        timings.append(stats['seconds'])
        completed+=int(stats['complete'])
        for j,budget in enumerate(budgets):
            feasible=np.flatnonzero(best<=budget+1e-12)
            if not len(feasible):
                continue
            q=int(feasible[-1]); old=selected[j]
            if old is None or q/n>old['cal_quality'] or (q/n==old['cal_quality'] and best[q]<old['cal_cost']-1e-15):
                selected[j]=dict(models=list(seq),thresholds=[float(cuts[m][r]) for m,r in zip(seq,witness[q])],
                                 cal_cost=float(best[q]),cal_quality=q/n)
        gain=np.array([p['cal_quality']-b['cal_quality'] for p,b in zip(selected,original['frozen']['4'])])
        summary=dict(dataset=args.dataset,split=args.split,seed=original['seed'],n_cal=n,
                     complete=completed==len(seqs),completed_chains=completed,total_chains=len(seqs),
                     attempted_chains=idx+1,search_seconds=sum(timings),last_chain=stats,
                     mean_cal_five_minus_four_pp=float(100*gain.mean()),max_cal_five_minus_four_pp=float(100*gain.max()),
                     four_model_search_seconds=original['search_stats']['kernel_seconds'],
                     fingerprint=fingerprint,interpretation='Calibration-only pilot. No held-out evaluation. Partial runs are lower bounds.')
        (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (dest/'selected.json').write_text(json.dumps(baseline.encode(selected),indent=2,allow_nan=False)+'\n')
        print(f'{args.dataset}: chain {idx+1}/{len(seqs)}, {stats["seconds"]:.2f}s, '
              f'{stats["evaluated"]:,} candidates, complete={stats["complete"]}, '
              f'total {sum(timings):.1f}s, mean gain {100*gain.mean():.4f} pp',flush=True)
        if not stats['complete'] or (args.pilot_seconds>0 and new_search_seconds>=args.pilot_seconds):
            break
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',choices=baseline.DATASETS)
    p.add_argument('--split',type=int,default=0)
    p.add_argument('--chain-seconds',type=float,default=120.)
    p.add_argument('--pilot-seconds',type=float,default=300.,help='New kernel work per invocation, checked between chains. Zero disables.')
    p.add_argument('--out',type=Path,default=ROOT/'results/exact_five_pilot')
    args=p.parse_args()
    if not 0<=args.split<50:
        p.error('split must be between 0 and 49')
    run(args)
