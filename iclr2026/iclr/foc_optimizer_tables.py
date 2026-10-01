"""Generate FOC optimizer tables from completed, frozen comparisons."""
import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
NAMES={'mmlu':'MMLU','triviaqa':'TriviaQA','math_hard':'MATH','simpleqa':'SimpleQA','livecodebench':'LiveCodeBench'}
with (ROOT/'results/foc_five/summary.csv').open() as f:
    rows={(r['dataset'],int(r['depth'])):r for r in csv.DictReader(f)}
out=ROOT/'iclr/tables'
def table(name,columns,body):
    text='\\begin{tabular}{l'+'r'*(len(columns)-1)+'}\n\\toprule\n'+' & '.join(columns)+r' \\'+'\n\\midrule\n'
    text+='\n'.join(' & '.join(row)+r' \\' for row in body)
    text+='\n\\bottomrule\n\\end{tabular}\n'
    (out/name).write_text(text)
gap_lines = [r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrr@{}}',
             r'\toprule',
             r'& \multicolumn{4}{c}{Calibration set} & \multicolumn{4}{c}{Test set} \\',
             r'\cmidrule(lr){2-5}\cmidrule(l){6-9}',
             'Dataset & '+ ' & '.join([f'$S_{k}$' for k in range(2,6)]*2)+r' \\',
             r'\midrule']
for d,name in NAMES.items():
    # One sign convention for both evaluation samples: FOC minus exhaustive.
    values = [-float(rows[d,k]['cal_gap_pp']) for k in range(2,6)]
    values += [float(rows[d,k]['accuracy_difference']) for k in range(2,6)]
    gap_lines.append(' & '.join([name]+[f'{v:+.3f}' for v in values])+r' \\')
gap_lines += [r'\bottomrule', r'\end{tabular*}']
(out/'table_foc_optimizer_gap.tex').write_text('\n'.join(gap_lines)+'\n')
body=[]
for d,name in NAMES.items():
    for k in range(2,6):
        r=rows[d,k]
        body.append([name if k==2 else '',f'$S_{k}$',f"{-float(r['cal_gap_pp']):+.3f}",f"{float(r['accuracy_difference']):+.3f}",
             f"[{float(r['accuracy_p10']):+.3f}, {float(r['accuracy_p90']):+.3f}]",
             f"{float(r['cost_difference']):+.6f}",f"{float(r['cost_pct_budget']):+.3f}",
             f"{100*float(r['cal_optimal_fraction']):.1f}"])
table('table_foc_optimizer_heldout.tex',['Dataset','Class',r'Cal. $\Delta$ acc. (pp)',r'Test $\Delta$ acc. (pp)','10th--90th',r'$\Delta$ cost (\$/1,000)',r'$\Delta$ cost (\% budget)',r'Cal. optimum (\%)'],body)
with (ROOT/'results/foc_subsequence/fresh_timing_benchmark.csv').open() as f:
    timings={(r['dataset'],int(r['depth'])):r for r in csv.DictReader(f)}
body=[]
for d,name in NAMES.items():
    for k in [3,4]:
        r=timings[d,k]
        body.append([name if k==3 else '',f'$S_{k}$',f"{float(r['exact_seconds']):.3f}",f"{float(r['foc_seconds']):.3f}",f"{float(r['exact_over_foc']):.2f}"])
table('table_foc_optimizer_timing.tex',['Dataset','Class','Exhaustive (s)','FOC-guided (s)','Exhaustive / FOC'],body)
