"""Exact S1/S2/S3 evaluation of stored synthetic confidence, with matched splits."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/simulated-signal-mpl')
import argparse
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scorer_depth_compute import prepare
from exact_heldout_depth import exact_select, evaluate_policy, encode, FRACTIONS
from fig2_compute import make_split, SEED
from manuscript_sources import EXPECTED_MODELS

OUT = Path('results/simulated_signal_depth')
SCORES = Path('results/simulated_confidence')
DATASETS = ('mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench')
TARGETS = (.5, .6, .7, .8, .9)
VERSION = 'synthetic-depth-v1'


def prepare_inputs(dataset):
    dest = OUT / dataset
    dest.mkdir(parents=True, exist_ok=True)
    raw, costs, rows, _, fingerprint = prepare(dataset, response=False)
    models = list(raw)
    if set(models) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: synthetic depth requires the full eight-model pool')
    arrays = dict(rows=rows, models=np.array(models),
                  correct=np.column_stack([raw[m].correct.to_numpy(float) for m in models]),
                  costs=np.column_stack([costs[m] for m in models]))
    diagnostics = []
    for target in TARGETS:
        frame = pd.read_parquet(SCORES/dataset/f'auroc_{target:.1f}.parquet').set_index('row_index')
        scores = frame.loc[rows, models].to_numpy(float)
        if not np.isfinite(scores).all():
            raise ValueError('Missing synthetic scores')
        arrays[f'scores_{target:.1f}'] = scores
        for j, model in enumerate(models):
            diagnostics.append(dict(dataset=dataset, target_auroc=target, model=model,
                                    n_rows=len(rows), achieved_auroc=roc_auc_score(arrays['correct'][:, j], scores[:, j])))
    np.savez_compressed(dest/'inputs.npz', **arrays)
    pd.DataFrame(diagnostics).to_csv(dest/'input_aurocs.csv', index=False)
    (dest/'input_metadata.json').write_text(json.dumps(dict(source_fingerprint=fingerprint,
        n_rows=len(rows), models=models, row_filter='same complete cases as scorer_depth_compute.prepare',
        excluded_rows=np.setdiff1d(np.arange(1055 if dataset=='livecodebench' else 2000),rows).tolist()), indent=2))
    print(f'Prepared {dataset}: {len(rows)} queries, {len(models)} models', flush=True)


def run(dataset, target, splits):
    inp=OUT/dataset/'inputs.npz'
    fingerprint=hashlib.sha256(inp.read_bytes()).hexdigest()
    source_hash=hashlib.sha256(b''.join(Path(p).read_bytes() for p in
        ('simulated_signal_depth.py','exact_heldout_depth.py','oracle_depth_analysis.py','optuna_frontier.py'))).hexdigest()
    with np.load(inp) as z:
        rows=z['rows']; models=z['models'].tolist(); correct=z['correct']; costs=z['costs']; scores=z[f'scores_{target:.1f}']
    if set(models) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: cached inputs omit manuscript models, rebuild inputs')
    dest=OUT/dataset/f'auroc_{target:.1f}'
    dest.mkdir(parents=True, exist_ok=True)
    started=time.monotonic()
    for split in range(splits):
        path=dest/f'split_{split:02d}.csv'
        manifest_path=path.with_suffix('.json')
        if path.exists():
            old=json.loads(manifest_path.read_text())
            if (old['version'],old['input_sha256'],old['source_sha256']) != (VERSION,fingerprint,source_hash):
                raise ValueError(f'Incompatible cache: {path}')
            continue
        cal_idx,test_idx=make_split(len(rows),correct[:,0],SEED+split)
        def sliced(idx):
            return {m:dict(scores=scores[idx,j],correct=correct[idx,j],costs=costs[idx,j]) for j,m in enumerate(models)}
        cal=sliced(cal_idx)
        low=min(cal,key=lambda m:(cal[m]['costs'].mean(),m))
        high=min(cal,key=lambda m:(-cal[m]['correct'].mean(),cal[m]['costs'].mean(),m))
        c0,c1=cal[low]['costs'].mean(),cal[high]['costs'].mean()
        budgets=c0+FRACTIONS*(c1-c0)
        frozen,counts=exact_select(cal,budgets)
        for policies in frozen.values():
            for policy in {json.dumps(encode(p),sort_keys=True):p for p in policies}.values():
                c,q,_=evaluate_policy(policy,cal)
                np.testing.assert_allclose([c,q],[policy['cal_cost'],policy['cal_quality']],atol=1e-12)
        manifest=dict(version=VERSION,input_sha256=fingerprint,source_sha256=source_hash,
            dataset=dataset,target_auroc=target,split=split,seed=SEED+split,pool=models,
            cal_idx=rows[cal_idx].tolist(),test_idx=rows[test_idx].tolist(),
            budgets=budgets.tolist(),low=low,high=high,candidate_counts=counts,frozen=frozen)
        manifest_path.write_text(json.dumps(encode(manifest),allow_nan=False))
        test=sliced(test_idx)
        records=[]; memo={}
        for j,budget in enumerate(budgets):
            rec=dict(dataset=dataset,target_auroc=target,split=split,fraction=FRACTIONS[j],budget=budget)
            for depth in (1,2,3):
                policy=frozen[str(depth)][j]
                key=json.dumps(encode(policy),sort_keys=True)
                if key not in memo:
                    memo[key]=evaluate_policy(policy,test)
                c,q,calls=memo[key]
                rec.update({f'd{depth}_{k}':v for k,v in dict(cost=c,accuracy=q,mean_calls=calls,
                    selected_depth=len(policy['models']),cal_cost=policy['cal_cost'],
                    cal_accuracy=policy['cal_quality'],overshoot=c-budget).items()})
            assert rec['d3_cal_accuracy']+1e-12>=rec['d2_cal_accuracy']>=rec['d1_cal_accuracy']-1e-12
            records.append(rec)
        temp=path.with_suffix('.tmp')
        pd.DataFrame(records).to_csv(temp,index=False); temp.replace(path)
        if (split+1)%10==0 or split==0:
            print(f'{dataset} X={target:.1f}: {split+1}/{splits}, elapsed {time.monotonic()-started:.0f}s',flush=True)
    return dataset,target


def report(splits):
    integrals=[]; curves=[]
    for dataset in DATASETS:
        for target in TARGETS:
            for split in range(splits):
                f=pd.read_csv(OUT/dataset/f'auroc_{target:.1f}'/f'split_{split:02d}.csv')
                row=dict(dataset=dataset,target_auroc=target,split=split)
                for depth in (1,2,3):
                    row[f's{depth}_accuracy_pp']=100*np.trapz(f[f'd{depth}_accuracy'],f.fraction)
                    row[f's{depth}_cost_per_1000']=1000*np.trapz(f[f'd{depth}_cost'],f.fraction)
                    row[f's{depth}_overshoot_fraction']=(f[f'd{depth}_overshoot']>1e-12).mean()
                row['s2_gain_pp']=row['s2_accuracy_pp']-row['s1_accuracy_pp']
                row['s3_gain_pp']=row['s3_accuracy_pp']-row['s1_accuracy_pp']
                row['s3_minus_s2_pp']=row['s3_accuracy_pp']-row['s2_accuracy_pp']
                row['s3_minus_s2_cost_per_1000']=row['s3_cost_per_1000']-row['s2_cost_per_1000']
                row['depth3_fraction']=np.trapz((f.d3_selected_depth==3).astype(float),f.fraction)
                integrals.append(row); curves.append(f)
    f=pd.DataFrame(integrals)
    f.to_csv(OUT/'split_integrals.csv',index=False)
    metrics=[c for c in f if c not in ('dataset','target_auroc','split')]
    summary=f.groupby(['dataset','target_auroc'])[metrics].mean()
    for metric in ('s2_gain_pp','s3_gain_pp','s3_minus_s2_pp'):
        for q in (.1,.9):
            summary[f'{metric}_p{int(q*100)}']=f.groupby(['dataset','target_auroc'])[metric].quantile(q)
    summary.to_csv(OUT/'summary.csv')
    pd.concat(curves,ignore_index=True).to_parquet(OUT/'budget_results.parquet',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,5,figsize=(17,6),sharex=True)
    for j,dataset in enumerate(DATASETS):
        cell=summary.loc[dataset]
        for depth,color in ((2,'#2166ac'),(3,'#d6604d')):
            key=f's{depth}_gain_pp'
            axes[0,j].plot(cell.index,cell[key],marker='o',label=f'S{depth}',color=color)
            axes[0,j].fill_between(cell.index,cell[key+'_p10'],cell[key+'_p90'],alpha=.12,color=color)
        key='s3_minus_s2_pp'
        axes[1,j].plot(cell.index,cell[key],marker='o',color='#444444')
        axes[1,j].fill_between(cell.index,cell[key+'_p10'],cell[key+'_p90'],alpha=.15,color='#444444')
        axes[1,j].axhline(0,color='gray',lw=.8)
        axes[0,j].set_title(dataset)
        axes[1,j].set_xlabel('Target AUROC')
        for ax in axes[:,j]:
            ax.grid(alpha=.2); ax.set_xticks(TARGETS)
    axes[0,0].set_ylabel('Gain over S1 (accuracy pp)')
    axes[1,0].set_ylabel('S3 minus S2 (accuracy pp)')
    axes[0,0].legend()
    fig.tight_layout()
    fig.savefig(OUT/'signal_depth.png',dpi=180)
    fig.savefig(OUT/'signal_depth.pdf')
    plt.close(fig)
    print(summary[['s2_gain_pp','s3_gain_pp','s3_minus_s2_pp','s3_minus_s2_cost_per_1000']].to_string())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--splits',type=int,default=50)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if not args.report_only:
        for dataset in args.datasets:
            if not (OUT/dataset/'inputs.npz').exists():
                prepare_inputs(dataset)
        if args.prepare_only:
            raise SystemExit()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(run,d,x,args.splits) for d in args.datasets for x in TARGETS]
            for job in as_completed(jobs):
                job.result()
    report(args.splits)
