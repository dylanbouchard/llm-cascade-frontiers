"""Full-sample exact up-to-four diagnostic, with resumable per-chain caches.

No model calls. No held-out interpretation. Run --max-chains 1 to benchmark.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import ctypes
import hashlib
import json
import subprocess
import time
from itertools import combinations
from pathlib import Path
import numpy as np
from oracle_depth_analysis import clean_data, score_ranks, SCORER
from exact_heldout_depth import exact_select, evaluate_policy

ROOT = Path(__file__).resolve().parent
VERSION = 'exact-four-v1'


def kernel():
    source = ROOT/'exact_four_kernel.cpp'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    library = Path('/tmp')/f'cascade_exact_four_{digest}.so'
    if not library.exists():
        subprocess.run(['clang++', '-O3', '-std=c++17', '-shared', '-fPIC',
                        str(source), '-o', str(library)], check=True)
    lib = ctypes.CDLL(str(library))
    f = lib.search_four
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    f.argtypes = [ctypes.c_int, ip, ip, dp, ctypes.c_double, dp, ip]
    f.restype = ctypes.c_uint64
    return f


def search_chain(data, seq, budget=np.inf):
    n = len(data[seq[0]]['correct'])
    ranks = np.ascontiguousarray([score_ranks(data[m]['scores'])[0] for m in seq[:3]], dtype=np.int32)
    rawq = np.asarray([data[m]['correct'] for m in seq])
    costs = np.ascontiguousarray([data[m]['costs'] for m in seq], dtype=np.float64)
    if not np.isin(rawq, [0, 1]).all() or not np.isfinite(costs).all() or (costs<0).any():
        raise ValueError('Requires binary correctness and finite nonnegative costs')
    if not all(np.isfinite(data[m]['scores']).all() for m in seq):
        raise ValueError('Requires finite scores')
    q = np.ascontiguousarray(rawq, dtype=np.int32)
    best = np.full(n+1, np.inf)
    witness = np.full((n+1,3), -1, dtype=np.int32)
    start = time.perf_counter()
    count = kernel()(n, ranks, q, costs, (budget+1e-12)*n, best, witness)
    return best/n, witness, count, time.perf_counter()-start


def frontier(best, budgets):
    return np.array([np.flatnonzero(best<=b+1e-12)[-1]/(len(best)-1)
                     if np.any(best<=b+1e-12) else np.nan for b in budgets])


def run(args):
    raw, costs, n, _ = clean_data(args.dataset)
    prompts = next(iter(raw.values()))['prompt'].values
    if not all(np.array_equal(prompts, d['prompt'].values) for d in raw.values()):
        raise ValueError('Misaligned prompts')
    data = {m:dict(scores=d[SCORER].to_numpy(float), correct=d.correct.to_numpy(float),
                   costs=np.asarray(costs[m],dtype=float)) for m,d in raw.items()}
    models = sorted(data, key=lambda m:(data[m]['costs'].mean(),m))
    low = models[0]
    high = min(models,key=lambda m:(-data[m]['correct'].mean(),data[m]['costs'].mean(),m))
    budgets = np.linspace(data[low]['costs'].mean(),data[high]['costs'].mean(),10001)
    digest = hashlib.sha256(VERSION.encode()+ (ROOT/'exact_four_kernel.cpp').read_bytes())
    for m in models:
        digest.update(m.encode())
        for k in ('scores','correct','costs'): digest.update(data[m][k].tobytes())
    fingerprint = digest.hexdigest()
    dest = args.out/args.dataset
    dest.mkdir(parents=True,exist_ok=True)
    basepath = dest/'baseline.npz'
    if basepath.exists():
        with np.load(basepath) as z:
            if str(z['fingerprint'])!=fingerprint: raise ValueError('Incompatible baseline cache')
            pair, triple = z['pair'],z['triple']
    else:
        print(f'{args.dataset}: computing exact depth 1/2/3 baseline on {n} examples',flush=True)
        frozen,_ = exact_select(data,budgets)
        pair = np.array([p['cal_quality'] for p in frozen['2']])
        triple = np.array([p['cal_quality'] for p in frozen['3']])
        np.savez_compressed(basepath,fingerprint=fingerprint,budgets=budgets,pair=pair,triple=triple)
    seqs = list(combinations(models,4))
    combined = np.full(n+1,np.inf)
    timings=[]
    for idx,seq in enumerate(seqs[:args.max_chains]):
        path=dest/f'chain_{idx:02d}.npz'
        if path.exists():
            with np.load(path) as z:
                if str(z['fingerprint'])!=fingerprint or list(z['models'])!=list(seq):
                    raise ValueError(f'Incompatible chain cache {path}')
                best,witness,count,seconds=z['best'],z['witness'],int(z['count']),float(z['seconds'])
        else:
            best,witness,count,seconds=search_chain(data,seq,budgets[-1])
            # Independently replay every retained correctness witness.
            cuts=[np.r_[-np.inf,np.unique(data[m]['scores'])[1:],np.inf] for m in seq[:3]]
            for correct in np.flatnonzero(np.isfinite(best)):
                p=dict(models=seq,thresholds=[cuts[j][r] for j,r in enumerate(witness[correct])])
                c,q,_=evaluate_policy(p,data)
                if abs(c-best[correct])>1e-12 or round(q*n)!=correct:
                    raise AssertionError('Witness replay mismatch')
            np.savez_compressed(path,fingerprint=fingerprint,models=seq,best=best,
                                witness=witness,count=count,seconds=seconds)
        combined=np.minimum(combined,best)
        timings.append(seconds)
        four=np.fmax(triple,frontier(combined,budgets))
        summary=dict(dataset=args.dataset,n_examples=n,n_models=len(models),
                     completed_chains=idx+1,total_chains=len(seqs),complete=idx+1==len(seqs),
                     search_seconds=sum(timings),estimated_total_seconds=float(np.mean(timings)*len(seqs)),
                     mean_four_minus_three_pp=float(100*np.mean(four-triple)),
                     max_four_minus_three_pp=float(100*np.max(four-triple)),
                     mean_four_minus_pair_pp=float(100*np.mean(four-pair)),
                     max_four_minus_pair_pp=float(100*np.max(four-pair)),
                     interpretation='Full-sample descriptive oracle. Partial runs are lower bounds for depth four.')
        (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        np.savez_compressed(dest/'frontiers.npz',budgets=budgets,pair=pair,triple=triple,four=four)
        print(f'{idx+1}/{len(seqs)} {seq}: {seconds:.2f}s, {count:,} evaluated, '
              f'mean/max increment {summary["mean_four_minus_three_pp"]:.4f}/'
              f'{summary["max_four_minus_three_pp"]:.4f} pp, '
              f'estimated total {summary["estimated_total_seconds"]/60:.1f} min',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('dataset',nargs='?',default='triviaqa')
    p.add_argument('--max-chains',type=int,default=70)
    p.add_argument('--out',type=Path,default=ROOT/'results/exact_four_model')
    run(p.parse_args())
