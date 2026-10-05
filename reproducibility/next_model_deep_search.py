"""Exact S4/S5 transition-specific scoring adapter and optimistic-bound kernel."""
import ctypes
import hashlib
import subprocess
from functools import lru_cache
from pathlib import Path
import numpy as np
from next_model_diff01 import fit_edges, predict, cutoffs, score_ranks

@lru_cache(None)
def kernel():
    source=Path(__file__).with_name('next_model_depth_kernel.cpp')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    path=Path('/tmp')/f'next_model_depth_{digest}.so'
    if not path.exists():
        subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(source),'-o',str(path)],check=True)
    lib=ctypes.CDLL(str(path))
    f=lib.search_depth
    ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    up=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,dp,ctypes.c_double,ctypes.c_double,dp,dp,ip,up]
    f.restype=ctypes.c_int
    return f


def search(cal, seq, edges, budgets, incumbent, seconds_limit=0):
    n=len(cal[seq[0]]['y']);k=len(seq)
    scores=[predict(cal[a]['x'],edges[a,b]) for a,b in zip(seq[:-1],seq[1:])]
    ranks=np.ascontiguousarray([score_ranks(s)[0] for s in scores],dtype=np.int32)
    y=np.ascontiguousarray([cal[m]['y'] for m in seq],dtype=np.int32)
    cost=np.ascontiguousarray([cal[m]['c'] for m in seq],dtype=np.float64)
    assert np.isin(y,[0,1]).all() and np.isfinite(cost).all() and (cost>=0).all()
    bound=np.full(n+1,np.inf)
    for p in incumbent:
        q=round(p['cal_quality']*n)
        bound[q]=min(bound[q],p['cal_cost']*n)
    bound=np.minimum.accumulate(bound[::-1])[::-1].copy()
    best=np.full(n+1,np.inf)
    witness=np.full((n+1,k-1),-1,dtype=np.int32)
    stats=np.zeros(2,dtype=np.uint64)
    complete=kernel()(n,k,ranks,y,cost,(max(budgets)+1e-12)*n,seconds_limit,bound,best,witness,stats)
    cuts=[cutoffs(s) for s in scores]
    return best/n,witness,cuts,dict(complete=bool(complete),nodes=int(stats[0]),pruned=int(stats[1]))


def merge(selected,budgets,seq,edges,best,witness,cuts):
    n=len(best)-1
    for j,b in enumerate(budgets):
        feasible=np.flatnonzero(best<=b+1e-12)
        if not len(feasible):continue
        q=int(feasible[-1]);old=selected[j]
        if q/n>old['cal_quality'] or (q/n==old['cal_quality'] and best[q]<old['cal_cost']-1e-15):
            selected[j]=dict(models=list(seq),stages=[dict(predictor=edges[a,b],threshold=float(cuts[s][witness[q,s]]))
                for s,(a,b) in enumerate(zip(seq[:-1],seq[1:]))],cal_cost=float(best[q]),cal_quality=float(q/n))
    return selected
