"""FOC-guided S3/S4 threshold search, compared with frozen exhaustive results."""
import argparse
import ctypes
import hashlib
import json
import subprocess
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from foc_pairwise_compare import source, sliced, select, DATASETS, ROOT


def kernel():
    src=ROOT/'foc_subsequence_kernel.cpp'
    digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    lib=Path('/tmp')/f'foc_subsequence_{digest}.so'
    if not lib.exists():
        subprocess.run(['clang++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(lib)],check=True)
    f=ctypes.CDLL(str(lib)).search_foc_chain
    ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    up=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    f.argtypes=[ctypes.c_int,ctypes.c_int,ip,ip,ip,dp,ctypes.c_int,dp,ctypes.c_int,ctypes.c_int,ctypes.c_int,ip,dp,ip,up]
    f.restype=None
    return f


def chain_search(cal,seq,budgets,window=100,iters=6,stride=25):
    n=len(cal[seq[0]]['scores']);d=len(seq)
    values=[np.unique(cal[m]['scores'],return_inverse=True) for m in seq[:-1]]
    ranks=np.ascontiguousarray([v[1] for v in values],dtype=np.int32)
    groups=np.ascontiguousarray([len(v[0]) for v in values],dtype=np.int32)
    q=np.ascontiguousarray([cal[m]['correct'] for m in seq],dtype=np.int32)
    costs=np.ascontiguousarray([cal[m]['costs'] for m in seq],dtype=np.float64)
    budgets=np.ascontiguousarray(budgets,dtype=np.float64)
    ts=np.zeros((len(budgets),d-1),dtype=np.int32);cs=np.zeros(len(budgets));qs=np.zeros(len(budgets),dtype=np.int32);stats=np.zeros(5,dtype=np.uint64)
    begin=perf_counter()
    kernel()(n,d,ranks,groups,q,costs,len(budgets),budgets,window,iters,stride,ts,cs,qs,stats)
    seconds=perf_counter()-begin
    cuts=[np.r_[-np.inf,v[0][1:],np.inf] for v in values]
    taus=np.column_stack([cuts[j][ts[:,j]] for j in range(d-1)])
    return taus,cs,qs/n,dict(zip(['coordinate_slices','candidate_evaluations','iterations','starts','accepted_transfers'],map(int,stats))),seconds


def replay(models,taus,seqs,data):
    memo={};out=[]
    for si,tt in zip(models,taus):
        seq=seqs[int(si)];t=tuple(tt[:len(seq)-1]);key=(int(si),t)
        if key not in memo:
            n=len(data[seq[0]]['scores']);active=np.ones(n,bool);cost=np.zeros(n);q=np.zeros(n)
            for j,m in enumerate(seq):
                cost[active]+=data[m]['costs'][active]
                stop=active.copy() if j==len(seq)-1 else active & (data[m]['scores']>=t[j])
                q[stop]=data[m]['correct'][stop];active &= ~stop
            assert not active.any()
            memo[key]=(cost.mean(),q.mean())
        out.append(memo[key])
    return np.array(out).T


def run(args):
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    code_hash=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in ['foc_subsequence_compare.py','foc_subsequence_kernel.cpp','foc_pairwise_compare.py'])).hexdigest()
    kernel() # Exclude one-time compilation from timings.
    for dataset in args.datasets:
        src=source(dataset)
        input_path=src/'inputs'/f'{dataset}.npz'
        with np.load(input_path) as z: arrays={k:z[k] for k in z.files}
        input_hash=hashlib.sha256(input_path.read_bytes()).hexdigest()
        for split in range(args.start_split,args.start_split+args.splits):
            dest=out/dataset;dest.mkdir(exist_ok=True)
            jsonpath=dest/f'split_{split:02d}.json';csvpath=jsonpath.with_suffix('.csv')
            if csvpath.exists():
                old=json.loads(jsonpath.read_text())
                assert old['code_sha256']==code_hash and old['window']==args.window and old['iterations']==args.iterations and old['restart_stride']==args.restart_stride and old['input_sha256']==input_hash
                print(dataset,split,'cached',flush=True);continue
            begin=perf_counter()
            path=src/dataset/f'split_{split:02d}.json';manifest=json.loads(path.read_text())
            assert str(arrays['fingerprint'])==manifest['data_sha256']
            assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
            cal=sliced(arrays,manifest['cal_idx']);budgets=np.array(manifest['budgets']);fractions=np.array(manifest['fractions'])
            # Budget subsampling is available only for a separately named pilot.
            idx=np.arange(len(budgets))[::args.budget_stride]
            if idx[-1]!=len(budgets)-1: idx=np.r_[idx,len(budgets)-1]
            budgets, fractions=budgets[idx],fractions[idx]
            selected,base_seqs,pair_stats=select(cal,budgets,'foc',args.window)
            models=sorted(cal,key=lambda m:(cal[m]['costs'].mean(),m))
            seqs=[(m,) for m in models]+list(combinations(models,2))+list(combinations(models,3))+list(combinations(models,4))
            assert seqs[:len(base_seqs)]==base_seqs
            chosen=selected['seq'].copy();taus=np.full((len(budgets),3),-np.inf);taus[:,0]=selected['tau']
            quality=selected['cal_accuracy'].copy();cost=selected['cal_cost'].copy()
            frozen={2:dict(seq=chosen.copy(),tau=taus.copy(),cal_accuracy=quality.copy(),cal_cost=cost.copy())};timings=[]
            start_search=perf_counter()
            for d in [3,4]:
                for si,seq in enumerate(seqs):
                    if len(seq)!=d:continue
                    tt,cc,qq,st,seconds=chain_search(cal,seq,budgets,args.window,args.iterations,args.restart_stride)
                    if args.verify_chains:
                        vc,vq=replay(np.zeros(len(budgets),int),tt,[seq],cal)
                        feasible=qq>=0
                        np.testing.assert_allclose(vc[feasible],cc[feasible],atol=1e-12,rtol=0)
                        np.testing.assert_allclose(vq[feasible],qq[feasible],atol=1e-12,rtol=0)
                    replace=(cc<=budgets+1e-12) & ((qq>quality+1e-12) | ((abs(qq-quality)<1e-12)&(cc<cost-1e-15)))
                    chosen[replace]=si;taus[replace,:d-1]=tt[replace];taus[replace,d-1:]=-np.inf;quality[replace]=qq[replace];cost[replace]=cc[replace]
                    timings.append(dict(depth=d,sequence=seq,seconds=seconds,**st))
                frozen[d]=dict(seq=chosen.copy(),tau=taus.copy(),cal_accuracy=quality.copy(),cal_cost=cost.copy())
                print(dataset,split,'depth',d,'seconds',round(perf_counter()-begin,2),flush=True)
            search_seconds=perf_counter()-start_search
            # Validate chosen policies and exact calibration upper bounds before testing.
            for d,p in frozen.items():
                c,q=replay(p['seq'],p['tau'],seqs,cal)
                np.testing.assert_allclose(c,p['cal_cost'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(q,p['cal_accuracy'],atol=1e-12,rtol=0)
                assert (c<=budgets+1e-12).all()
                exact=[manifest['frozen'][str(d)][int(i)] for i in idx]
                assert (q<=np.array([p['cal_quality'] for p in exact])+1e-12).all()
            np.savez_compressed(dest/f'split_{split:02d}_policies.npz',budgets=budgets,fractions=fractions,**{f'd{d}_{k}':v for d,p in frozen.items() for k,v in p.items()})
            meta=dict(dataset=dataset,split=split,code_sha256=code_hash,input_sha256=input_hash,data_sha256=manifest['data_sha256'],source=str(path.relative_to(ROOT)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cal_idx=manifest['cal_idx'],test_idx=manifest['test_idx'],sequences=seqs,window=args.window,iterations=args.iterations,restart_stride=args.restart_stride,budget_indices=idx.tolist(),chain_timings=timings,pair_stats=pair_stats,search_seconds=search_seconds)
            jsonpath.write_text(json.dumps(meta,indent=2)+'\n')
            test=sliced(arrays,manifest['test_idx']);frame=pd.DataFrame(dict(fraction=fractions,budget=budgets))
            saved=pd.read_csv(path.with_suffix('.csv')).iloc[idx]
            for d,p in frozen.items():
                tc,tq=replay(p['seq'],p['tau'],seqs,test)
                frame[f'foc{d}_cost'],frame[f'foc{d}_accuracy']=tc,tq
                frame[f'foc{d}_cal_cost'],frame[f'foc{d}_cal_accuracy']=p['cal_cost'],p['cal_accuracy']
                frame[f'foc{d}_depth']=[len(seqs[i]) for i in p['seq']]
                # Replay the exact comparator independently using the current inputs.
                ep=[manifest['frozen'][str(d)][int(i)] for i in idx]
                esi=np.array([seqs.index(tuple(p['models'])) for p in ep]);ett=np.full((len(idx),3),-np.inf)
                for j,p in enumerate(ep):
                    for k,t in enumerate(p['thresholds']):ett[j,k]=-np.inf if t=='never' else np.inf if t=='always' else float(t)
                ec,eq=replay(esi,ett,seqs,test)
                np.testing.assert_allclose(ec,saved[f'd{d}_cost'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(eq,saved[f'd{d}_accuracy'],atol=1e-12,rtol=0)
                for col in ['cost','accuracy','cal_cost','cal_accuracy']:frame[f'exact{d}_{col}']=saved[f'd{d}_{col}'].to_numpy()
            frame.to_csv(csvpath,index=False)
            print(dataset,split,'complete',round(perf_counter()-begin,2),'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--splits',type=int,default=50);p.add_argument('--start-split',type=int,default=0)
    p.add_argument('--window',type=int,default=100);p.add_argument('--iterations',type=int,default=6)
    p.add_argument('--restart-stride',type=int,default=25);p.add_argument('--budget-stride',type=int,default=1)
    p.add_argument('--verify-chains',action='store_true');p.add_argument('--out',default='results/foc_subsequence')
    a=p.parse_args()
    if a.splits<1 or a.start_split<0 or a.start_split+a.splits>50 or min(a.window,a.iterations,a.restart_stride,a.budget_stride)<1:p.error('invalid positive run controls')
    run(a)
