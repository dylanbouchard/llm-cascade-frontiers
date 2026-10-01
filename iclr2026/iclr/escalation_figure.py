"""Print-sized representative benefit curves from the current manuscript caches."""
from pathlib import Path
import sys,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from manuscript_sources import result_path,file_record
PAIRS={
 'mmlu':('MMLU','gpt-4o-mini','gpt-oss-20b'),
 'triviaqa':('TriviaQA','gpt-4o-mini','llama-3.3-70b'),
 'math_hard':('MATH (levels 3–5)','gpt-oss-20b','deepseek-v3'),
 'simpleqa':('SimpleQA','deepseek-v3','gpt-4o'),
 'livecodebench':('LiveCodeBench','gpt-4o-mini','gpt-oss-20b')}
DISPLAY={'gpt-4o-mini':'GPT-4o mini','gpt-oss-20b':'GPT-oss-20B',
         'llama-3.3-70b':'Llama 3.3-70B','deepseek-v3':'DeepSeek-V3','gpt-4o':'GPT-4o'}

def main():
    plt.rcParams.update({'font.family':'serif','font.size':9,'axes.titlesize':9.5,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(6.3,6.0),layout='constrained')
    blue,red,green='#4878CF','#D65F5F','#6ACC65'
    provenance=[]
    for ax,(ds,(name,cheap,exp)) in zip(axes.flat,PAIRS.items()):
        path=result_path(ds,'escalation_benefit')/ds/f'{ds}_{cheap}_{exp}_mean_token_negentropy.npz'
        with np.load(path) as z:
            x=z['s_grid'];y=z['median_curve'];valid=np.isfinite(y)
            assert valid.any()
            xx,yy=x[valid],y[valid]
            # Preserve the original positive-region shading convention.
            dx=(xx[-1]-xx[0])/len(xx)/2 if len(xx)>1 else 0
            start=None
            for xi,yi in zip(xx,yy):
                if yi>0 and start is None:start=xi-dx
                elif yi<=0 and start is not None:
                    ax.axvspan(start,xi+dx,color=green,alpha=.08,lw=0);start=None
            if start is not None:ax.axvspan(start,xx[-1]+dx,color=green,alpha=.08,lw=0)
            ax.fill_between(xx,z['band_lo'][valid],z['band_hi'][valid],color=blue,alpha=.25)
            ax.plot(xx,yy,color=blue,lw=1.4)
            ax.plot(xx,z['isotonic_curve'][valid],color=red,lw=1.3,ls='--')
            ax.axhline(0,color='gray',ls=':',lw=.7)
            ax.set_title(f'{name}\n{DISPLAY[cheap]} → {DISPLAY[exp]}',pad=6)
            ax.set_xlabel(r'Confidence score $s_L$');ax.set_ylabel(r'$m_H(s)-m_L(s)$')
            ax.text(.97,.97,f"dom={float(z['dominance_frac']):.0%}  dec={float(z['decreasing_frac']):.0%}",
                transform=ax.transAxes,ha='right',va='top',fontsize=8,
                bbox=dict(facecolor='white',edgecolor='.7',alpha=.85,pad=2))
            ax.grid(alpha=.18,lw=.5)
        provenance.append(dict(dataset=ds,input=file_record(path)))
    legend=axes[2,1];legend.axis('off')
    legend.legend(handles=[Patch(facecolor=green,alpha=.15,label=r'$m_H>m_L$ region'),
        Patch(facecolor=blue,alpha=.25,label='10th–90th split percentiles'),
        Line2D([0],[0],color=blue,lw=1.4,label='Median spline'),
        Line2D([0],[0],color=red,ls='--',label='Isotonic non-increasing fit'),
        Line2D([0],[0],color='gray',ls=':',label='Zero')],loc='center',frameon=False,fontsize=9)
    path=ROOT/'iclr/figures/fig3_escalation_benefit.pdf'
    fig.savefig(path);plt.close(fig)
    out=ROOT/'iclr/results/escalation_figure';out.mkdir(parents=True,exist_ok=True)
    (out/'provenance.json').write_text(json.dumps(dict(generator=file_record(Path(__file__)),inputs=provenance,output=file_record(path)),indent=2)+'\n')

if __name__=='__main__':main()
