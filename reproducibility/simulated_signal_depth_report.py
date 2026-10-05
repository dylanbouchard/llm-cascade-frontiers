"""Summarize paired changes relative to the AUROC 0.5 scenario."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/simulated-signal-mpl')
from pathlib import Path
import json
from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from manuscript_sources import EXPECTED_MODELS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT=Path('results/simulated_signal_depth')
DATASETS=('mmlu','triviaqa','math_hard','simpleqa','livecodebench')
NAMES=('MMLU','TriviaQA','MATH 3–5','SimpleQA','LiveCodeBench')
PAPER=Path('paper')


def diagnostics():
    """Regenerate manuscript diagnostics from the exact depth input rows and pool."""
    records=[]
    lines=[r'\begin{tabular}{lrrrcc}', r'\toprule',
        r'Dataset & Queries & Models & \shortstack{Real correctness\\AUROC} & \shortstack{Real benefit-\\AUROC} & \shortstack{Synthetic benefit-\\AUROC} \\',
        r'\midrule']
    for dataset, name in zip(DATASETS, ('MMLU','TriviaQA','MATH','SimpleQA','LiveCodeBench')):
        with np.load(OUT/dataset/'inputs.npz') as z:
            rows=z['rows']; models=z['models'].tolist()
            correct=z['correct']; costs=z['costs']; synthetic=z['scores_0.8']
        assert set(models)==EXPECTED_MODELS, (dataset, models)
        real=[]
        for j, model in enumerate(models):
            frame=pd.read_parquet(Path('data/output_data')/f'{dataset}-{model}.parquet',
                columns=['correct','mean_token_negentropy']).iloc[rows]
            np.testing.assert_array_equal(frame.correct,correct[:,j])
            real.append(frame.mean_token_negentropy.to_numpy(float))
        real=np.column_stack(real)
        auc=np.array([roc_auc_score(correct[:,j],real[:,j]) for j in range(len(models))])
        real_benefit=[]; synthetic_benefit=[]
        for i,j in combinations(np.argsort(costs.mean(axis=0)),2):
            label=(correct[:,j]>correct[:,i]).astype(int)
            real_benefit.append(roc_auc_score(label,-real[:,i]) if len(np.unique(label))>1 else .5)
            synthetic_benefit.append(roc_auc_score(label,-synthetic[:,i]) if len(np.unique(label))>1 else .5)
        record=dict(dataset=dataset,queries=len(rows),models=len(models),pairs=len(real_benefit),
            real_correctness_median=float(np.median(auc)),real_correctness_min=float(auc.min()),
            real_correctness_max=float(auc.max()),real_benefit=float(np.mean(real_benefit)),
            synthetic_benefit=float(np.mean(synthetic_benefit)))
        records.append(record)
        lines.append(f'{name} & {len(rows):,} & {len(models)} & '
            f'${np.median(auc):.3f}\\,[{auc.min():.3f},{auc.max():.3f}]$ & '
            f'${np.mean(real_benefit):.3f}$ & ${np.mean(synthetic_benefit):.3f}$ '+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}'])
    (PAPER/'tables/table_synthetic_diagnostics.tex').write_text('\n'.join(lines)+'\n')
    pd.DataFrame(records).to_csv(OUT/'matched_diagnostics.csv',index=False)


def main():
    # Refuse reports based on historical seven-model inputs or split manifests.
    for dataset in DATASETS:
        with np.load(OUT/dataset/'inputs.npz') as z:
            assert set(z['models'].tolist())==EXPECTED_MODELS
        for target in (.5,.6,.7,.8,.9):
            for split in range(50):
                path=OUT/dataset/f'auroc_{target:.1f}'/f'split_{split:02d}.json'
                assert set(json.loads(path.read_text())['pool'])==EXPECTED_MODELS
    diagnostics()
    f=pd.read_csv(OUT/'split_integrals.csv')
    base=f.loc[f.target_auroc==.5,['dataset','split','s2_accuracy_pp','s3_accuracy_pp']]
    matched=f.merge(base,on=['dataset','split'],suffixes=('','_at_05'),validate='many_to_one')
    for depth in (2,3):
        matched[f's{depth}_signal_gain_pp']=matched[f's{depth}_accuracy_pp']-matched[f's{depth}_accuracy_pp_at_05']
    cols=['s2_signal_gain_pp','s3_signal_gain_pp','s3_minus_s2_pp']
    grouped=matched.groupby(['dataset','target_auroc'])[cols]
    summary=grouped.mean()
    for c in cols:
        summary[c+'_p10']=grouped.quantile(.1)[c]
        summary[c+'_p90']=grouped.quantile(.9)[c]
    summary.to_csv(OUT/'signal_improvement_summary.csv')
    matched.to_csv(OUT/'signal_improvement_splits.csv',index=False)
    fig,axes=plt.subplots(2,5,figsize=(16,6),sharex=True,sharey='row')
    for j,(dataset,name) in enumerate(zip(DATASETS,NAMES)):
        cell=summary.loc[dataset]
        for depth,color in ((2,'#2166ac'),(3,'#d6604d')):
            key=f's{depth}_signal_gain_pp'
            axes[0,j].plot(cell.index,cell[key],marker='o',label=f'$S_{depth}$',color=color)
            axes[0,j].fill_between(cell.index,cell[key+'_p10'],cell[key+'_p90'],alpha=.12,color=color)
        key='s3_minus_s2_pp'
        axes[1,j].plot(cell.index,cell[key],marker='o',color='#444444')
        axes[1,j].fill_between(cell.index,cell[key+'_p10'],cell[key+'_p90'],alpha=.15,color='#444444')
        axes[0,j].set_title(name)
        axes[1,j].set_xlabel('Target AUROC')
        for ax in axes[:,j]:
            ax.axhline(0,color='gray',lw=.7);ax.grid(alpha=.15);ax.set_xticks([.5,.6,.7,.8,.9])
    axes[0,0].set_ylabel('Gain relative to AUROC 0.5\n(percentage points)')
    axes[1,0].set_ylabel('$S_3$ minus $S_2$\n(percentage points)')
    axes[0,0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT/'signal_improvement.png',dpi=180)
    fig.savefig(OUT/'signal_improvement.pdf')
    fig.savefig(PAPER/'figures/fig_signal_quality.pdf')
    plt.close(fig)
    results=pd.read_csv(OUT/'summary.csv').set_index(['dataset','target_auroc'])
    lines=['# Cascade depth and synthetic signal quality','',
        '50 matched splits, 500 assigned budget positions, exact calibration selection. '
        'All accuracy differences are budget-integrated test accuracy in percentage points. '
        'Brackets give 10th to 90th percentiles across overlapping splits, not confidence intervals.','',
        '| Dataset | AUROC | S2 gain over S1 | S3 gain over S1 | S3 minus S2 [p10, p90] | S3 minus S2 cost ($/1,000 queries) |',
        '|---|---:|---:|---:|---:|---:|']
    for dataset,name in zip(DATASETS,NAMES):
        for target in (.5,.6,.7,.8,.9):
            r=results.loc[(dataset,target)]
            lines.append(f'| {name} | {target:.1f} | {r.s2_gain_pp:.3f} | {r.s3_gain_pp:.3f} | {r.s3_minus_s2_pp:+.3f} [{r.s3_minus_s2_pp_p10:+.3f}, {r.s3_minus_s2_pp_p90:+.3f}] | {r.s3_minus_s2_cost_per_1000:+.5f} |')
    lines+=['','The single-model baseline is the calibration-selected best affordable model at each assigned budget. '
        'Cascade gains at AUROC 0.5 can arise from allocating queries among models at intermediate budgets, '
        'even without an informative marginal correctness ranking. The signal-improvement figure therefore '
        'also reports each class relative to its own AUROC 0.5 scenario.','',
        'Realized test costs need not match at the same assigned budget. Synthetic score generation uses '
        'full-sample labels, while cascade thresholds and chains are selected only on calibration data. '
        'These results condition on one synthetic draw per dataset. Existing complete-case filtering '
        'can slightly change AUROC and correlations from the full simulated sample. See input_aurocs.csv '
        'in each dataset directory and README.md for the protocol.','',
        '![Signal quality and depth](signal_improvement.png)','']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(summary.loc[(slice(None),.9),cols].to_string())

if __name__=='__main__':
    main()
