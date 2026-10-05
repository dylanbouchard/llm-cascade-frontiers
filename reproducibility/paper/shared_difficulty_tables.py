"""Export only the requested shared-difficulty manuscript tables from frozen results."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'results/shared_difficulty_depth'
OUT=ROOT/'paper/tables'
NAMES={'mmlu':'MMLU','triviaqa':'TriviaQA','math_hard':'MATH','simpleqa':'SimpleQA','livecodebench':'LiveCodeBench'}
CONDITIONS=('own','half','max')
SCORERS={'mean_token_negentropy':'Mean negentropy','min_token_negentropy':'Min negentropy','probability_margin':'Probability margin','min_probability':'Min probability','sequence_probability':'Length-normalized probability','logreg_ensemble':'Correctness-trained token-prob.','response_logreg':'Correctness-trained embedding'}

def save(name,lines):
 (OUT/name).write_text('\n'.join(lines)+'\n')
def row(cells):return ' & '.join(cells)+r' \\'
def long_start(columns,caption,label,header):
 return [r'\begingroup',r'\scriptsize',r'\renewcommand{\arraystretch}{0.85}',r'\setlength{\tabcolsep}{4pt}',r'\begin{longtable}{'+columns+'}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',header,r'\midrule',r'\endfirsthead',r'\multicolumn{'+str(len(columns))+r'}{c}{\tablename\ \thetable\ continued} \\',r'\toprule',header,r'\midrule',r'\endhead',r'\midrule',r'\multicolumn{'+str(len(columns))+r'}{r}{Continued on next page} \\',r'\endfoot',r'\bottomrule',r'\endlastfoot']
def long_end():return [r'\end{longtable}',r'\endgroup']

def main():
 s=pd.read_csv(SRC/'summary.csv').set_index(['dataset','target','condition'])
 d=pd.read_csv(SRC/'selected_chain_summary.csv')
 d=d[d.population=='test'].set_index(['dataset','target','condition','depth_limit','stage'])
 lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrrrrrr@{}}',r'\toprule',r'& \multicolumn{3}{c}{$S_3-S_2$ (pp)} & \multicolumn{3}{c}{Downstream AUROC} & \multicolumn{3}{c}{\shortstack{Gain over random\\escalation (pp)}} \\',r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}',row([r'Dataset \quad $w$']+['$0$','$0.5$','max']*3),r'\midrule']
 for ds,name in NAMES.items():
  vals=[f'${s.loc[ds,.9,c].gain_pp:+.2f}$' for c in CONDITIONS]
  vals += [f'${d.loc[ds,.9,c,3,0].A_downstream:.2f}$' for c in CONDITIONS]
  vals += [f'${100*d.loc[ds,.9,c,3,0].eq10_rhs:.1f}$' for c in CONDITIONS]
  lines.append(row([name]+vals))
 save('table_shared_difficulty_main.tex',lines+[r'\bottomrule',r'\end{tabular*}'])
 lines=[r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llrrrrrrr@{}}',r'\toprule',r'& & \multicolumn{3}{c}{Gain (split SD), pp} & \multicolumn{3}{c}{$\Delta$ test cost (\$/1,000)} & \\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',row(['Dataset','Target']+['$0$','$0.5$','max']*2+['Max $w$']),r'\midrule']
 for ds,name in NAMES.items():
  for t in (.8,.9):
   vals=[f'${s.loc[ds,t,c].gain_pp:+.3f}$ ({s.loc[ds,t,c].split_sd_pp:.3f})' for c in CONDITIONS]
   vals += [f'${s.loc[ds,t,c].cost_gap_per_1000:+.4f}$' for c in CONDITIONS]
   lines.append(row([name if t==.8 else '',f'{t:.1f}']+vals+[f'{s.loc[ds,t,"max"].w:.2f}']))
 save('table_shared_difficulty_gains.tex',lines+[r'\bottomrule',r'\end{tabular*}'])
 lines=long_start('llrrrrr',r'Test-set diagnostics for selected synthetic-score policies. Blocks give $S_2$ stage 1, $S_3$ stage 1, and $S_3$ stage 2. AUROCs and the right side of Equation~\ref{eq:auc-gain} are averaged with assigned-budget weights over selected policies and 50 splits. The RHS is in percentage points and includes reach probability. Coverage is the fraction of budget weight with that nonterminal stage. Undefined AUROCs are omitted from their means.','tab:shared-stage',row(['Dataset','Target','$w$',r'$\operatorname{AUROC}(s_i,U_i)$',r'$\operatorname{AUROC}(s_i,U_{>i})$','RHS (pp)','Coverage']))
 for limit,stage in ((2,0),(3,0),(3,1)):
  lines += [r'\multicolumn{7}{l}{\textbf{'+f'$S_{limit}$, stage {stage+1}'+r'}} \\*',r'\midrule']
  for ds,name in NAMES.items():
   for t in (.8,.9):
    for c in CONDITIONS:
     r=d.loc[ds,t,c,limit,stage]
     w=s.loc[ds,t,c].w
     lines.append(row([name if t==.8 and c=='own' else '',f'{t:.1f}' if c=='own' else '',f'{w:.2f}',f'{r.A_i:.3f}',f'{r.A_downstream:.3f}',f'${100*r.eq10_rhs:+.3f}$',f'{r.policy_budget_mass:.3f}']))
 save('table_shared_difficulty_stages.tex',lines+long_end())
 path=SRC/'observed/summary.csv'
 if not path.exists():return
 observed=pd.read_csv(path).set_index(['dataset','scorer','depth_limit','stage'])
 lines=long_start('llrrrr',r'Test-set diagnostics for frozen policies under seven observed scorers. No threshold search or scorer fitting is repeated. Saved logistic-regression coefficients reconstruct learned scores. Blocks give $S_2$ stage 1, $S_3$ stage 1, and $S_3$ stage 2, with the same weighting and conventions as Table~\ref{tab:shared-stage}. The RHS of Equation~\ref{eq:auc-gain} is in percentage points.','tab:observed-stage',row(['Dataset','Scorer',r'$\operatorname{AUROC}(s_i,U_i)$',r'$\operatorname{AUROC}(s_i,U_{>i})$','RHS (pp)','Coverage']))
 for limit,stage in ((2,0),(3,0),(3,1)):
  lines += [r'\multicolumn{6}{l}{\textbf{'+f'$S_{limit}$, stage {stage+1}'+r'}} \\*',r'\midrule']
  for ds,name in NAMES.items():
   for scorer,label in SCORERS.items():
    r=observed.loc[ds,scorer,limit,stage]
    lines.append(row([name if scorer=='mean_token_negentropy' else '',label,f'{r.A_i:.3f}',f'{r.A_downstream:.3f}',f'${100*r.eq10_rhs:+.3f}$',f'{r.coverage:.3f}']))
 save('table_shared_difficulty_observed.tex',lines+long_end())

if __name__=='__main__':main()
