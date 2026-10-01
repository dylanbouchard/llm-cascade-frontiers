"""Assemble current table and signal figure from copied, validated reports.

The original report scripts predate the final seven-scorer table and S4 signal
figure. This packaging helper uses their existing loaders and summaries.
"""
from pathlib import Path
import shutil
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'iclr'))
from frontier_figure import load_scores, X, NAMES, LABELS
from cascade_gain_table import SCORERS
from manuscript_sources import DATASETS
from diff01_report import ARMS, gains, update_depth_table


def main():
    _, data = load_scores()
    lines = [r'\begin{tabular}{llrrrrrrrrr}', r'\toprule',
             'Dataset & Comparison & ' + ' & '.join(['Mean negent.', 'Min negent.', 'Margin', 'Min prob.', 'LNP', 'LogReg', 'Response LR', *ARMS]) + r' \\',
             r'\midrule', r'\multicolumn{11}{l}{\textbf{Cascade gain}} \\']
    for ds, name in zip(DATASETS, NAMES):
        vals = [np.trapz(data[s][ds][:,1]-data[s][ds][:,0], X, axis=1).mean() for s in SCORERS]
        vals += [gains(label, ds, 2, 1).mean() for label in ARMS]
        lines.append(name + r' & $S_2-S_1$ & ' + ' & '.join(f'${v:+.3f}$' for v in vals) + r' \\')
    lines += [r'\midrule', r'\multicolumn{11}{l}{\textbf{Depth gain}} \\']
    for ds, name in zip(DATASETS, NAMES):
        for d in [3,4,5]:
            vals = [np.trapz(data[s][ds][:,d-1]-data[s][ds][:,1], X, axis=1).mean() for s in SCORERS]
            for label in ARMS:
                v = gains(label, ds, d, 2)
                vals.append(None if v is None else v.mean())
            lines.append((name if d==3 else '') + f' & $S_{d}-S_2$ & ' + ' & '.join('---' if v is None else f'${v:+.3f}$' for v in vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (ROOT/'iclr/tables/table_exact_depth.tex').write_text('\n'.join(lines)+'\n')
    update_depth_table(DATASETS)
    summary = pd.read_csv(ROOT/'results/simulated_signal_depth_four/summary.csv').set_index(['dataset','target_auroc'])
    fig, axes = plt.subplots(2,5,figsize=(13,5),sharex=True)
    colors = ['#60636a','#147878','#d99218','#6854a3']
    for j, (ds,name) in enumerate(zip(DATASETS,NAMES)):
        cell = summary.loc[ds].sort_index()
        x = cell.index.to_numpy(float)
        for d in range(1,5):
            key=f's{d}_accuracy_pp'
            axes[0,j].plot(x,cell[key],marker='o',color=colors[d-1],label=f'$S_{d}$')
            axes[0,j].fill_between(x,cell[key+'_p10'],cell[key+'_p90'],color=colors[d-1],alpha=.12)
        for d in [3,4]:
            key=f's{d}_minus_s2_pp'
            axes[1,j].plot(x,cell[key],marker='o',color=colors[d-1],label=f'$S_{d}-S_2$')
            axes[1,j].fill_between(x,cell[key+'_p10'],cell[key+'_p90'],color=colors[d-1],alpha=.12)
        axes[0,j].set_title(name)
        axes[1,j].axhline(0,color='gray',lw=.6)
        axes[1,j].set_xlabel('Target correctness AUROC')
    axes[0,0].set_ylabel('Mean test accuracy (%)')
    axes[1,0].set_ylabel('Mean accuracy gain (pp)')
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT/'iclr/figures/fig_signal_quality.pdf')
    plt.close(fig)
    for src, dest in [('synthetic_cost_depth','cost_depth_observed.pdf'),('synthetic_cost_auroc09_depth','cost_depth_auroc09.pdf')]:
        shutil.copyfile(ROOT/'results'/src/'cost_depth.pdf', ROOT/'iclr/figures'/dest)

if __name__ == '__main__':
    main()
