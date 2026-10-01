"""Extend frozen FOC S4 selections through every five-model subsequence.

Keeps the earlier FOC results and all exact caches read-only.
"""
import argparse
import hashlib
import json
from itertools import combinations
from time import perf_counter
import numpy as np
import pandas as pd
from foc_pairwise_compare import ROOT, DATASETS, source, sliced
from foc_subsequence_compare import chain_search, kernel, replay


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    out=ROOT/args.out
    code_hash=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in [
        'foc_five_compare.py','foc_subsequence_compare.py',
        'foc_subsequence_kernel.cpp','foc_pairwise_compare.py'])).hexdigest()
    parent_code=hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in [
        'foc_subsequence_compare.py','foc_subsequence_kernel.cpp','foc_pairwise_compare.py'])).hexdigest()
    kernel()
    for dataset in args.datasets:
        src=source(dataset)
        input_path=src/'inputs'/f'{dataset}.npz'
        with np.load(input_path) as z: arrays={k:z[k] for k in z.files}
        input_hash=sha(input_path)
        assert len(arrays['models'])==8
        for split in range(args.start_split,args.start_split+args.splits):
            start=perf_counter()
            dest=out/dataset;dest.mkdir(parents=True,exist_ok=True)
            path=src/dataset/f'split_{split:02d}.json'
            baseline=ROOT/'results/foc_subsequence'/dataset/f'split_{split:02d}.json'
            policy_path=baseline.with_name(f'split_{split:02d}_policies.npz')
            manifest=json.loads(path.read_text());old=json.loads(baseline.read_text())
            assert old['code_sha256']==parent_code
            assert old['input_sha256']==input_hash and old['source_sha256']==sha(path)
            assert str(arrays['fingerprint'])==manifest['data_sha256']==old['data_sha256']
            assert old['cal_idx']==manifest['cal_idx'] and old['test_idx']==manifest['test_idx']
            assert not set(old['cal_idx']) & set(old['test_idx'])
            assert old['budget_indices']==list(range(500))
            assert (old['window'],old['iterations'],old['restart_stride'])==(100,6,25)
            provenance=dict(code_sha256=code_hash,input_sha256=input_hash,parent_json_sha256=sha(baseline),parent_policies_sha256=sha(policy_path),exact_json_sha256=sha(path))
            target=dest/f'split_{split:02d}.json';csv=target.with_suffix('.csv')
            if csv.exists():
                completed=json.loads(target.read_text())
                assert all(completed[k]==v for k,v in provenance.items())
                print(dataset,split,'cached',flush=True);continue
            budgets=np.array(manifest['budgets']);fractions=np.array(manifest['fractions'])
            cal=sliced(arrays,manifest['cal_idx'])
            models=sorted(cal,key=lambda m:(cal[m]['costs'].mean(),m))
            seqs=[tuple(s) for s in old['sequences']]+list(combinations(models,5))
            assert len(old['sequences'])==162 and len(seqs)==218
            with np.load(policy_path) as z:
                np.testing.assert_array_equal(z['budgets'],budgets)
                np.testing.assert_array_equal(z['fractions'],fractions)
                chosen=z['d4_seq'].copy();taus=np.full((500,4),-np.inf);taus[:,:3]=z['d4_tau']
                quality=z['d4_cal_accuracy'].copy();cost=z['d4_cal_cost'].copy()
                old_quality=quality.copy()
            c,q=replay(chosen,taus,seqs,cal)
            np.testing.assert_allclose(c,cost,atol=1e-12,rtol=0)
            np.testing.assert_allclose(q,quality,atol=1e-12,rtol=0)
            timings=[];search_start=perf_counter()
            for si in range(162,218):
                seq=seqs[si]
                tt,cc,qq,st,seconds=chain_search(cal,seq,budgets,100,6,25)
                # Independently replay every returned five-model policy.
                vc,vq=replay(np.zeros(500,int),tt,[seq],cal)
                feasible=qq>=0
                np.testing.assert_allclose(vc[feasible],cc[feasible],atol=1e-12,rtol=0)
                np.testing.assert_allclose(vq[feasible],qq[feasible],atol=1e-12,rtol=0)
                assert (cc[feasible]<=budgets[feasible]+1e-12).all()
                replace=(cc<=budgets+1e-12)&((qq>quality+1e-12)|((abs(qq-quality)<1e-12)&(cc<cost-1e-15)))
                chosen[replace]=si;taus[replace]=tt[replace];quality[replace]=qq[replace];cost[replace]=cc[replace]
                timings.append(dict(depth=5,sequence=seq,seconds=seconds,**st))
            search_seconds=perf_counter()-search_start
            c,q=replay(chosen,taus,seqs,cal)
            np.testing.assert_allclose(c,cost,atol=1e-12,rtol=0)
            np.testing.assert_allclose(q,quality,atol=1e-12,rtol=0)
            assert (c<=budgets+1e-12).all() and (q>=old_quality-1e-12).all()
            ep=manifest['frozen']['5']
            assert (q<=np.array([p['cal_quality'] for p in ep])+1e-12).all()
            esi=np.array([seqs.index(tuple(p['models'])) for p in ep]);ett=np.full((500,4),-np.inf)
            for j,p in enumerate(ep):
                for k,t in enumerate(p['thresholds']):ett[j,k]=-np.inf if t=='never' else np.inf if t=='always' else float(t)
            ec,eq=replay(esi,ett,seqs,cal)
            np.testing.assert_allclose(ec,[p['cal_cost'] for p in ep],atol=1e-12,rtol=0)
            np.testing.assert_allclose(eq,[p['cal_quality'] for p in ep],atol=1e-12,rtol=0)
            np.savez_compressed(dest/f'split_{split:02d}_policies.npz',budgets=budgets,fractions=fractions,d5_seq=chosen,d5_tau=taus,d5_cal_cost=cost,d5_cal_accuracy=quality)
            meta=dict(dataset=dataset,split=split,**provenance,data_sha256=manifest['data_sha256'],parent_source=str(baseline.relative_to(ROOT)),exact_source=str(path.relative_to(ROOT)),cal_idx=old['cal_idx'],test_idx=old['test_idx'],sequences=seqs,window=100,iterations=6,restart_stride=25,budget_indices=list(range(500)),chain_timings=timings,extension_search_seconds=search_seconds,parent_selection_seconds=old['pair_stats']['seconds']+sum(t['seconds'] for t in old['chain_timings']))
            # Freeze all selections before exposing test arrays to evaluation.
            target.write_text(json.dumps(meta,indent=2)+'\n')
            test=sliced(arrays,manifest['test_idx'])
            tc,tq=replay(chosen,taus,seqs,test);ec,eq=replay(esi,ett,seqs,test)
            saved=pd.read_csv(path.with_suffix('.csv'))
            np.testing.assert_allclose(ec,saved.d5_cost,atol=1e-12,rtol=0)
            np.testing.assert_allclose(eq,saved.d5_accuracy,atol=1e-12,rtol=0)
            frame=pd.read_csv(baseline.with_suffix('.csv'))
            np.testing.assert_allclose(frame.budget,budgets,atol=1e-15,rtol=0)
            frame['foc5_cost'],frame['foc5_accuracy']=tc,tq
            frame['foc5_cal_cost'],frame['foc5_cal_accuracy']=cost,quality
            frame['foc5_depth']=[len(seqs[i]) for i in chosen]
            for col in ['cost','accuracy','cal_cost','cal_accuracy']:frame[f'exact5_{col}']=saved[f'd5_{col}']
            frame.to_csv(csv,index=False)
            print(dataset,split,'complete',round(perf_counter()-start,2),'seconds',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=DATASETS,default=DATASETS)
    p.add_argument('--splits',type=int,default=50);p.add_argument('--start-split',type=int,default=0)
    p.add_argument('--out',default='results/foc_five')
    args=p.parse_args()
    if args.splits<1 or args.start_split<0 or args.start_split+args.splits>50:p.error('invalid split range')
    run(args)
