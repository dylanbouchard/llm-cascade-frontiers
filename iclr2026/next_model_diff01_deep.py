"""Extend frozen Diff-01 experiments to exact S4 or S5, with checkpoints."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse
import hashlib
import json
import time
import shutil
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from next_model_diff01 import prepare,BASE_SCORERS,ARMS,fit_edges,evaluate,encode,DATASETS
from next_model_deep_search import search,merge,kernel

BASE=Path('results/next_model_diff01_full')
INPUT_CACHE={}
SOURCES=['next_model_diff01_deep.py','next_model_deep_search.py','next_model_depth_kernel.cpp','test_next_model_deep_search.py','next_model_diff01.py','continuation_benefit_compute.py','scorer_depth_compute.py','fig2_compute.py','optuna_frontier.py','voi_compute.py']


def prepare_input(dataset,out):
    raw,costs,rows,_,fingerprint=prepare(dataset,response=False)
    assert len(raw)==8
    path=out/'inputs'/f'{dataset}.npz'
    with path.open('xb') as handle:
        np.savez_compressed(handle,models=list(raw),rows=rows,fingerprint=fingerprint,
            x=np.array([df[BASE_SCORERS].to_numpy(float) for df in raw.values()]),
            y=np.array([df.correct.to_numpy(float) for df in raw.values()]),c=np.array([costs[m] for m in raw]))
    print(f'{dataset}: prepared {len(rows)} rows, eight models',flush=True)


def run(dataset,out,splits,arms,limit,max_depth,split_ids=None):
    key=str(out/'inputs'/f'{dataset}.npz')
    if key not in INPUT_CACHE:
        with np.load(key) as z:INPUT_CACHE[key]={k:z[k] for k in z.files}
    data=INPUT_CACHE[key];rows=data['rows'];fingerprint=str(data['fingerprint'])
    for split in (range(splits) if split_ids is None else split_ids):
        for arm in arms:
            dest=out/dataset/arm
            if (dest/f'split_{split:02d}.csv').exists():continue
            started=time.perf_counter()
            basepath=BASE/dataset/arm/f'split_{split:02d}.json'
            original=basepath.read_bytes();m=json.loads(original)
            if m['data_sha256']!=fingerprint:raise ValueError('Baseline fingerprint mismatch')
            def sliced(original_idx):
                idx=np.searchsorted(rows,np.asarray(original_idx))
                np.testing.assert_array_equal(rows[idx],original_idx)
                return {a:{k:data[k][i,idx] for k in ('x','y','c')} for i,a in enumerate(data['models'])}
            cal=sliced(m['cal_idx']);models,edges=fit_edges(cal,arm);budgets=np.array(m['budgets'])
            dest=out/dataset/arm;dest.mkdir(parents=True,exist_ok=True)
            selected=list(m['frozen']['3']);stats_all={};frozen=dict(m['frozen'])
            for depth in range(4,max_depth+1):
                chain_dir=dest/f'split_{split:02d}_s{depth}_chains';chain_dir.mkdir(exist_ok=True)
                for i,seq in enumerate(combinations(models,depth)):
                    begin=time.perf_counter()
                    path=chain_dir/f'chain_{i:02d}.npz'
                    incumbent_hash=hashlib.sha256(json.dumps(encode(selected),sort_keys=True).encode()).hexdigest()
                    if path.exists():
                        from next_model_diff01 import cutoffs,predict
                        with np.load(path) as z:
                            assert str(z['incumbent_hash'])==incumbent_hash
                            best,witness,stats=z['best'],z['witness'],json.loads(str(z['stats']))
                        cuts=[cutoffs(predict(cal[a]['x'],edges[a,b])) for a,b in zip(seq[:-1],seq[1:])]
                    else:
                        best,witness,cuts,stats=search(cal,seq,edges,budgets,selected,limit)
                        stats['seconds']=time.perf_counter()-begin
                        if stats['complete']:
                            temp=path.with_suffix('.partial.npz')
                            np.savez_compressed(temp,best=best,witness=witness,models=seq,stats=json.dumps(stats),incumbent_hash=incumbent_hash)
                            temp.rename(path)
                    if not stats['complete']:raise RuntimeError(f'Incomplete exact chain {dataset}/{arm}/{split}/S{depth}/{i}: {stats}')
                    selected=merge(selected,budgets,seq,edges,best,witness,cuts)
                    stats_all[f'{depth}/{i}']=stats
                unique={json.dumps(encode(p),sort_keys=True):p for p in selected}
                for p in unique.values():
                    np.testing.assert_allclose(evaluate(p,cal)[:2],[p['cal_cost'],p['cal_quality']],atol=1e-12)
                frozen[str(depth)]=list(selected)
                selected=list(selected)
                depth_path=dest/f'split_{split:02d}_s{depth}.json'
                if not depth_path.exists():
                    with depth_path.open('x') as handle:json.dump(encode(dict(selected=selected,stats=stats_all)),handle,allow_nan=False)
                print(f'{dataset} {arm} {split+1}/{splits} S{depth}: {time.perf_counter()-started:.1f}s elapsed',flush=True)
            manifest=dict(m,frozen=frozen,deep_search_stats=stats_all,baseline_sha256=hashlib.sha256(original).hexdigest())
            final_manifest=dest/f'split_{split:02d}.json'
            if not final_manifest.exists():
                with final_manifest.open('x') as handle:json.dump(encode(manifest),handle,allow_nan=False)
            test=sliced(m['test_idx']);f=pd.read_csv(basepath.with_suffix('.csv'))
            for depth in range(4,max_depth+1):
                memo={};records=[]
                for b,p in zip(budgets,frozen[str(depth)]):
                    key=json.dumps(encode(p),sort_keys=True)
                    if key not in memo:memo[key]=evaluate(p,test)
                    c,q,calls=memo[key]
                    records.append({f's{depth}_{k}':v for k,v in dict(cost=c,accuracy=q,mean_calls=calls,depth=len(p['models']),cal_cost=p['cal_cost'],cal_accuracy=p['cal_quality'],overshoot=c-b).items()})
                f=pd.concat([f,pd.DataFrame(records)],axis=1)
                assert (f[f's{depth}_cal_accuracy']>=f[f's{depth-1}_cal_accuracy']-1e-12).all()
            f.to_csv(dest/f'split_{split:02d}.csv',index=False,mode='x')


def report(out,datasets,splits,arms,max_depth):
    records=[]
    for d in datasets:
        for arm in arms:
            for split in range(splits):
                f=pd.read_csv(out/d/arm/f'split_{split:02d}.csv')
                r=dict(dataset=d,arm=arm,split=split)
                for depth in range(2,max_depth+1):
                    r[f's{depth}_accuracy']=f[f's{depth}_accuracy'].mean()
                    r[f's{depth}_cost']=f[f's{depth}_cost'].mean()
                    r[f's{depth}_minus_s2_pp']=100*(f[f's{depth}_accuracy']-f.s2_accuracy).mean()
                    r[f's{depth}_gain_over_s1_pp']=100*(f[f's{depth}_accuracy']-f.s1_accuracy).mean()
                    r[f's{depth}_overshoot_fraction']=(f[f's{depth}_overshoot']>1e-12).mean()
                r['s4_minus_s3_pp']=100*(f.s4_accuracy-f.s3_accuracy).mean()
                if max_depth >= 5:
                    r['s5_minus_s4_pp']=100*(f.s5_accuracy-f.s4_accuracy).mean()
                records.append(r)
    f=pd.DataFrame(records);f.to_csv(out/'split_summary.csv',index=False,mode='x')
    summary=f.groupby(['dataset','arm']).mean(numeric_only=True)
    summary.to_csv(out/'summary.csv',mode='x')
    columns=['s3_minus_s2_pp','s4_minus_s2_pp','s4_minus_s3_pp']
    if max_depth >= 5:
        columns += ['s5_minus_s2_pp','s5_minus_s4_pp']
    print(summary[columns].to_string(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',default=DATASETS,choices=DATASETS)
    p.add_argument('--splits',type=int,default=50)
    p.add_argument('--arms',nargs='+',default=list(ARMS),choices=ARMS)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--limit',type=float,default=0.)
    p.add_argument('--max-depth',type=int,choices=(4,5),default=5)
    p.add_argument('--out',type=Path,default=Path('results/next_model_diff01_deep'))
    args=p.parse_args()
    provenance=dict(datasets=args.datasets,splits=args.splits,arms=args.arms,max_depth=args.max_depth,
        source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in SOURCES})
    if args.resume:
        assert json.loads((args.out/'provenance.json').read_text())==provenance
    else:
        args.out.mkdir(parents=True,exist_ok=False)
        snap=args.out/'executed_source';snap.mkdir()
        for s in SOURCES:shutil.copy2(s,snap/s)
        (args.out/'provenance.json').write_text(json.dumps(provenance,indent=2))
        (args.out/'inputs').mkdir()
    kernel()
    with ProcessPoolExecutor(max_workers=min(3,args.workers)) as pool:
        jobs=[pool.submit(prepare_input,d,args.out) for d in args.datasets if not (args.out/'inputs'/f'{d}.npz').exists()]
        for job in jobs:job.result()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[pool.submit(run,d,args.out,args.splits,[arm],args.limit,args.max_depth,[split]) for split in range(args.splits) for d in args.datasets for arm in args.arms]
        for job in jobs:job.result()
    report(args.out,args.datasets,args.splits,args.arms,args.max_depth)
