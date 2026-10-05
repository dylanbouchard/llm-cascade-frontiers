"""One reporting convention for frozen exact test-set comparisons.

Run from the repository root with .venv/bin/python paper/exact_depth_report.py.
This never changes search results. Primary summaries integrate piecewise-linear
paired differences on the original 500 normalized budget positions. Split means
receive equal weight. Arithmetic means are retained as a cross-check.
"""
import os
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/exact-replacement-mpl')
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from manuscript_sources import result_path, validate_exact, file_record
PAPER = ROOT/'paper'
OUT = PAPER/'results/exact_depth'
DATASETS = ['mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench']
NAMES = dict(zip(DATASETS, ['MMLU', 'TriviaQA', 'MATH', 'SimpleQA', 'LiveCodeBench']))
COLORS = ['#60636a', '#147878', '#d99218', '#6854a3', '#be5369']
trap = getattr(np, 'trapezoid', np.trapz)


def integrate(y, x, lo=0., hi=1.):
    """Insert exact interval endpoints by linear interpolation, then normalize."""
    inner = (x > lo) & (x < hi)
    xx = np.r_[lo, x[inner], hi]
    yy = np.r_[np.interp(lo, x, y), y[inner], np.interp(hi, x, y)]
    return float(trap(yy, xx)/(hi-lo))


def load():
    pieces, provenance = [], []
    fixed_paths = [result_path(d, 'exact_replacement')/'fixed'/d/f'split_{s:02d}.csv'
                   for d in DATASETS for s in range(50)]
    fixed_complete = all(p.exists() for p in fixed_paths)
    for dataset in DATASETS:
        input_path = result_path(dataset, 'exact_heldout_five')/'inputs'/f'{dataset}.npz'
        with np.load(input_path) as z:
            arrays = {k: z[k] for k in z.files}
        provenance.append(dict(dataset=dataset, input=file_record(input_path)))
        for split in range(50):
            path = result_path(dataset, 'exact_heldout_five')/dataset/f'split_{split:02d}.csv'
            manifest_path = path.with_suffix('.json')
            m = json.loads(manifest_path.read_text())
            validate_exact(m, arrays, dataset, split)
            frame = pd.read_csv(path)
            assert m['seed'] == split+42 and not set(m['cal_idx']) & set(m['test_idx'])
            np.testing.assert_allclose(frame.fraction, np.linspace(0, 1, 500), atol=1e-14)
            np.testing.assert_allclose(frame.budget, m['budgets'], rtol=0, atol=1e-15)
            for d in range(1, 6):
                assert frame[f'd{d}_feasible'].all()
                assert (frame[f'd{d}_cal_cost'] <= frame.budget+1e-12).all()
                if d > 1:
                    assert (frame[f'd{d}_cal_accuracy'] >= frame[f'd{d-1}_cal_accuracy']-1e-12).all()
            assert str(arrays['fingerprint']) == m['data_sha256']
            index = {str(model): i for i, model in enumerate(arrays['models'])}
            scores = arrays['scores'][:, m['test_idx']]
            memo = {}
            for d in range(1, 6):
                records = []
                for p in m['frozen'][str(d)]:
                    key = (tuple(p['models']), tuple(p['thresholds']))
                    if key not in memo:
                        active = np.ones(scores.shape[1], bool)
                        reach, stop = np.zeros(5), np.zeros(5)
                        for stage, model in enumerate(p['models']):
                            reach[stage] = active.mean()
                            if stage == len(p['models'])-1:
                                stop[stage] = active.mean()
                            else:
                                t = p['thresholds'][stage]
                                t = -np.inf if t=='never' else np.inf if t=='always' else float(t)
                                nxt = active & (scores[index[model]] < t)
                                stop[stage] = (active & ~nxt).mean()
                                active = nxt
                        memo[key] = np.r_[reach, stop]
                    records.append(memo[key])
                values = np.array(records)
                np.testing.assert_allclose(values[:, :5].sum(axis=1), frame[f'd{d}_mean_calls'], atol=1e-12)
                for i, name in enumerate([f'reach{j}' for j in range(1,6)]+[f'stop{j}' for j in range(1,6)]):
                    frame[f'd{d}_{name}'] = values[:, i]
            if fixed_complete:
                fp = result_path(d, 'exact_replacement')/'fixed'/dataset/path.name
                ff = pd.read_csv(fp)
                fm = json.loads(fp.with_suffix('.json').read_text())
                assert fm['complete'] and fm['data_sha256'] == m['data_sha256']
                assert fm['cal_idx'] == m['cal_idx'] and fm['test_idx'] == m['test_idx']
                np.testing.assert_allclose(ff.budget, frame.budget, rtol=0, atol=1e-15)
                for d in range(1, 6):
                    for metric in ['accuracy', 'cost', 'cal_accuracy', 'cal_cost', 'mean_calls', 'selected_depth']:
                        np.testing.assert_allclose(ff[f'd{d}_{metric}'], frame[f'd{d}_{metric}'], rtol=0, atol=1e-12)
                frame = frame.join(ff[[c for c in ff if c not in frame]])
                provenance.append(dict(path=str(fp.relative_to(ROOT)), sha256=hashlib.sha256(fp.read_bytes()).hexdigest()))
            pieces.append(frame)
            provenance.append(dict(path=str(manifest_path.relative_to(ROOT)),
                sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(), data_sha256=m['data_sha256'],
                csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return pd.concat(pieces, ignore_index=True), provenance, fixed_complete


def summarize(frame, fixed):
    pairs = [(f'd{d}', 'd2') for d in [1, 3, 4, 5]]+[(f'd{d}', f'd{d-1}') for d in [3, 4, 5]]
    pairs = list(dict.fromkeys(pairs))
    if fixed:
        pairs += [(f'{p}{d}', b) for p in ['f', 'a'] for d in [3, 4, 5] for b in ['d2', f'd{d}']]
    records, diagnostic = [], []
    methods = [f'd{d}' for d in range(1, 6)] + ([f'{p}{d}' for p in ['f', 'a'] for d in [3, 4, 5]] if fixed else [])
    for (dataset, split), f in frame.groupby(['dataset', 'split'], sort=False):
        f = f.sort_values('budget_index')
        x = f.fraction.to_numpy()
        for method, reference in pairs:
            values = dict(accuracy_pp=100*(f[f'{method}_accuracy']-f[f'{reference}_accuracy']),
                cal_accuracy_pp=100*(f[f'{method}_cal_accuracy']-f[f'{reference}_cal_accuracy']),
                cost_per_1000=1000*(f[f'{method}_cost']-f[f'{reference}_cost']),
                cost_budget_pct=100*(f[f'{method}_cost']-f[f'{reference}_cost'])/f.budget)
            for band, lo, hi in [('all', 0, 1), ('low', 0, 1/3), ('middle', 1/3, 2/3), ('high', 2/3, 1)]:
                row = dict(dataset=dataset, split=split, method=method, reference=reference, band=band)
                for metric, v in values.items():
                    row[metric] = integrate(v.to_numpy(), x, lo, hi)
                    if band == 'all':
                        row[metric+'_arithmetic'] = float(v.mean())
                records.append(row)
        for method in methods:
            excess = (f[f'{method}_cost']-f.budget).clip(lower=0)/f.budget
            row = dict(dataset=dataset, split=split, method=method,
                overshoot_pct=integrate(100*(f[f'{method}_cost'].to_numpy()>f.budget.to_numpy()+1e-12), x),
                mean_positive_excess_pct=integrate(100*excess.to_numpy(), x),
                calls=integrate(f[f'{method}_mean_calls'].to_numpy(), x))
            for d in range(1, 6):
                row[f'nominal_{d}_pct'] = integrate(100*(f[f'{method}_selected_depth'].to_numpy()==d), x)
                for metric in ['reach', 'stop']:
                    col = f'{method}_{metric}{d}'
                    if col in f:
                        row[f'{metric}{d}_pct'] = integrate(100*f[col].to_numpy(), x)
            diagnostic.append(row)
    paired, usage = pd.DataFrame(records), pd.DataFrame(diagnostic)
    paired.to_csv(OUT/'paired_split_summaries.csv', index=False)
    usage.to_csv(OUT/'usage_split_summaries.csv', index=False)
    rows = []
    for keys, group in paired.groupby(['dataset', 'method', 'reference', 'band']):
        row = dict(zip(['dataset', 'method', 'reference', 'band'], keys))
        for col in group.select_dtypes('number'):
            if col != 'split':
                row[col] = group[col].mean()
                row[col+'_p10'], row[col+'_p90'] = np.quantile(group[col].dropna(), [.1, .9]) if group[col].notna().any() else (np.nan, np.nan)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT/'summary.csv', index=False)
    return paired, usage, summary


def signed(value, places=3):
    if abs(value) < .5*10**(-places):
        return f'{0:.{places}f}'
    return f'{value:+.{places}f}'


def table(path, header, rows, spec=None):
    path.write_text('\\begin{tabular}{'+(spec or 'l'*len(header))+'}\n\\toprule\n'+' & '.join(header)+r' \\'+'\n\\midrule\n'+
                    '\n'.join(' & '.join(row)+r' \\' for row in rows)+'\n\\bottomrule\n\\end{tabular}\n')


def make_tables(frame, summary, usage, fixed):
    rows = []
    for dataset in DATASETS:
        for d in [3, 4, 5]:
            r = summary[(summary.dataset==dataset)&(summary.method==f'd{d}')&(summary.reference=='d2')&(summary.band=='all')].iloc[0]
            rows.append([NAMES[dataset] if d == 3 else '', f'$S_{d}-S_2$',
                f"${r.accuracy_pp:+.3f}$ [$ {signed(r.accuracy_pp_p10)}, {signed(r.accuracy_pp_p90)} $]",
                f"${r.cost_per_1000:+.5f}$", f"${r.cost_budget_pct:+.2f}$"])
    third = summary[(summary.method=='d3')&(summary.reference=='d2')&(summary.band=='all')]
    fifth = summary[(summary.method=='d5')&(summary.reference=='d2')&(summary.band=='all')]
    (PAPER/'tables/exact_depth_findings.tex').write_text(
        f"Allowing a third model changes mean test-set accuracy by ${third.accuracy_pp.min():+.3f}$ to "
        f"${third.accuracy_pp.max():+.3f}$ percentage points across datasets. TriviaQA has the only "
        "positive mean change at this depth. Allowing five models changes mean accuracy by "
        f"${fifth.accuracy_pp.min():+.3f}$ to ${fifth.accuracy_pp.max():+.3f}$ percentage points relative to $S_2$, "
        "and raises mean realized test cost on every dataset.\n")
    selected = usage[(usage.dataset=='mmlu')&(usage.method=='d5')].mean(numeric_only=True)
    (PAPER/'tables/exact_usage_findings.tex').write_text(
        "Exact selections can still be nominally deep. On MMLU, $S_5$ selects a five-model sequence "
        f"for {selected.nominal_5_pct:.1f}\\% of budget weight, while an average of "
        f"{selected.reach5_pct:.2f}\\% of test-set queries reach stage five. "
        "Nominal depth and realized model use therefore describe different aspects of the selected policy "
        "(Tables~\\ref{tab:exact-usage} and~\\ref{tab:exact-reach}).\n")
    table(PAPER/'tables/table_exact_depth.tex', ['Dataset', 'Comparison', r'$\Delta$ accuracy (pp) [10th, 90th]', r'$\Delta$ cost (\$/1,000)', r'$\Delta$ cost (\% budget)'], rows)
    rows = []
    for dataset in DATASETS:
        f = frame[frame.dataset==dataset]
        for d in range(1, 6):
            u = usage[(usage.dataset==dataset)&(usage.method==f'd{d}')].mean(numeric_only=True)
            excess = 100*((f[f'd{d}_cost']-f.budget)/f.budget).clip(lower=0)
            rows.append([NAMES[dataset] if d==1 else '', f'$S_{d}$', f'{u.overshoot_pct:.1f}',
                f'{u.mean_positive_excess_pct:.2f}', f'{np.quantile(excess, .9):.2f}', f'{u.calls:.3f}'])
    table(PAPER/'tables/table_exact_feasibility.tex', ['Dataset', 'Class', r'Overshoot (\%)', r'Mean excess (\%)', r'90th excess (\%)', 'Calls/query'], rows, 'llrrrr')
    rows = []
    for dataset in DATASETS:
        for d in [2, 3, 4, 5]:
            u = usage[(usage.dataset==dataset)&(usage.method==f'd{d}')].mean(numeric_only=True)
            rows.append([NAMES[dataset] if d==2 else '', f'$S_{d}$']+[f'{u[f"nominal_{i}_pct"]:.1f}' for i in range(1, 6)])
    table(PAPER/'tables/table_exact_usage.tex', ['Dataset', 'Class']+[f'{i} model'+('s' if i>1 else '') for i in range(1, 6)], rows)
    rows = []
    timings = pd.concat([pd.DataFrame(json.loads((result_path(d, 'exact_heldout_five')/'run_timings.json').read_text())).query('dataset == @d') for d in DATASETS], ignore_index=True)
    for dataset in DATASETS:
        t = timings[timings.dataset==dataset]
        rows.append([NAMES[dataset], '50/50', '56',
            f'{t.seconds.median():.1f}', f'{t.seconds.quantile(.9):.1f}',
            f'{t.kernel_seconds.median():.1f}', f'{t.kernel_seconds.quantile(.9):.1f}'])
    table(PAPER/'tables/table_exact_search.tex', ['Dataset', 'Complete', '5-model sequences', 'Wall median (s)', 'Wall 90th', 'Kernel median (s)', 'Kernel 90th'], rows)
    if fixed:
        primary = summary[summary.method.isin(['f3', 'f4', 'f5'])&(summary.reference=='d2')&(summary.band=='all')]
        math_rows = summary[(summary.dataset=='math_hard')&(summary.reference=='d2')&(summary.band=='all')].set_index('method')
        (PAPER/'tables/exact_fixed_findings.tex').write_text(
            f"The primary fixed model sequences have mean test-set accuracy differences of ${primary.accuracy_pp.min():+.2f}$ "
            f"to ${primary.accuracy_pp.max():+.2f}$ percentage points relative to $S_2$. "
            "Their performance is sensitive to the prespecified sequence. For MATH at length four, "
            f"the coverage and spacing rules yield differences of ${math_rows.loc['f4', 'accuracy_pp']:+.2f}$ "
            f"and ${math_rows.loc['a4', 'accuracy_pp']:+.2f}$ points, respectively. "
            "Both use exact threshold optimization. These results describe the specified sequence "
            "restrictions and do not establish a universal penalty from increasing cascade depth.\n")
        rows = []
        for dataset in DATASETS:
            for d in [3, 4, 5]:
                for ref in ['d2', f'd{d}']:
                    r = summary[(summary.dataset==dataset)&(summary.method==f'f{d}')&(summary.reference==ref)&(summary.band=='all')].iloc[0]
                    rows.append([NAMES[dataset] if d==3 and ref=='d2' else '', f'$F_{d}-S_{ref[1:]}$',
                        f'{r.accuracy_pp:+.3f}', f'[{signed(r.accuracy_pp_p10)}, {signed(r.accuracy_pp_p90)}]',
                        f'{r.cost_per_1000:+.5f}', f'{r.cost_budget_pct:+.2f}'])
        table(PAPER/'tables/table_exact_fixed.tex', ['Dataset', 'Comparison', r'$\Delta$ acc. (pp)', '[10th, 90th]', r'$\Delta$ \$/1,000', r'$\Delta$ \% budget'], rows)


def supplementary_outputs(frame, paired, summary, usage, fixed):
    # Reverse the paired S1-S2 differences before computing quantiles.
    rows = []
    selected = paired[(paired.method=='d1') & (paired.reference=='d2') & (paired.band=='all')]
    for dataset in DATASETS:
        part = selected[selected.dataset==dataset]
        gains = -part.accuracy_pp
        rows.append([NAMES[dataset], f"${gains.mean():+.3f}$ [$ {gains.quantile(.1):+.3f}, {gains.quantile(.9):+.3f} $]",
            f"${-part.cost_per_1000.mean():+.5f}$", f"${-part.cost_budget_pct.mean():+.2f}$"])
    table(PAPER/'tables/table_pair_standalone.tex',
        ['Dataset', r'$\Delta$ accuracy (pp) [10th, 90th]', r'$\Delta$ cost (\$/1,000)', r'$\Delta$ cost (\% budget)'], rows)
    """Retain budget-resolved, adjacent-depth, and margin diagnostics."""
    pairs = [(f'd{d}', 'd2') for d in [1, 3, 4, 5]]+[(f'd{d}', f'd{d-1}') for d in [4, 5]]
    if fixed:
        pairs += [(f'{p}{d}', ref) for p in ['f', 'a'] for d in [3, 4, 5] for ref in ['d2', f'd{d}']]
    resolved, margins = [], []
    for dataset in DATASETS:
        f = frame[frame.dataset==dataset]
        for method, reference in pairs:
            for metric, scale in [('accuracy', 100), ('cal_accuracy', 100), ('cost', 1000)]:
                values = np.stack([(g[f'{method}_{metric}']-g[f'{reference}_{metric}']).to_numpy()*scale
                                   for _, g in f.groupby('split')])
                lo, hi = np.quantile(values, [.1, .9], axis=0)
                for j in range(500):
                    resolved.append(dict(dataset=dataset, method=method, reference=reference,
                        metric=metric, budget_index=j, fraction=j/499, mean=values[:,j].mean(),
                        p10=lo[j], p90=hi[j]))
                if metric == 'accuracy':
                    margins.append(dict(dataset=dataset, method=method, reference=reference,
                        cells_gt_half_pp_pct=100*np.mean(values>.5), cells_gt_one_pp_pct=100*np.mean(values>1)))
    pd.DataFrame(resolved).to_csv(OUT/'budget_resolved.csv', index=False)
    pd.DataFrame(margins).to_csv(OUT/'heldout_margin_frequencies.csv', index=False)
    rows = []
    for dataset in DATASETS:
        for d in [3, 4, 5]:
            g = summary[(summary.dataset==dataset)&(summary.method==f'd{d}')&(summary.reference=='d2')].set_index('band')
            row = [NAMES[dataset] if d==3 else '', f'$S_{d}-S_2$']
            row += [f"${g.loc[b, 'accuracy_pp']:+.3f}$" for b in ['low', 'middle', 'high']]
            m = next(m for m in margins if m['dataset']==dataset and m['method']==f'd{d}' and m['reference']=='d2')
            row += [f"{m['cells_gt_half_pp_pct']:.1f}", f"{m['cells_gt_one_pp_pct']:.1f}"]
            rows.append(row)
    table(PAPER/'tables/table_exact_budget_thirds.tex',
          ['Dataset', 'Comparison', r'Low (pp)', r'Middle (pp)', r'High (pp)', r'Cells $>0.5$ pp (\%)', r'Cells $>1$ pp (\%)'], rows, 'llrrrrr')
    rows = []
    for dataset in DATASETS:
        for d in [3, 4, 5]:
            u = usage[(usage.dataset==dataset)&(usage.method==f'd{d}')].mean(numeric_only=True)
            rows.append([NAMES[dataset] if d==3 else '', f'$S_{d}$']+
                        [f"{u[f'reach{i}_pct']:.2f}" for i in [3, 4, 5]])
    table(PAPER/'tables/table_exact_reach.tex', ['Dataset', 'Class', r'Reach 3 (\%)', r'Reach 4 (\%)', r'Reach 5 (\%)'], rows)

    if fixed:
        identities, timings = [], []
        for dataset in DATASETS:
            for split in range(50):
                path = result_path(d, 'exact_replacement')/'fixed'/dataset/f'split_{split:02d}.json'
                m = json.loads(path.read_text())
                for method, seq in m['chains'].items():
                    identities.append(dict(dataset=dataset, split=split, method=method,
                        chain=' -> '.join(seq), first=seq[0], terminal=seq[-1]))
                    timings.append(dict(dataset=dataset, split=split, method=method, **m['stats'][method],
                        split_wall_seconds=m['seconds'], worker_peak_rss_bytes=m['peak_rss_bytes']))
        pd.DataFrame(identities).to_csv(OUT/'fixed_chain_identities.csv', index=False)
        aliases = {'llama-3.1-8b':'L8', 'qwen2.5-7b':'Q7', 'deepseek-v3':'D',
                   'llama-3.3-70b':'L70', 'gpt-4o-mini':'Gm', 'gpt-oss-20b':'O20',
                   'gpt-4o':'G4', 'MiniMax-M2.7':'M27'}
        identity_frame = pd.DataFrame(identities)
        rows = []
        for dataset in DATASETS:
            df = identity_frame[(identity_frame.dataset==dataset)&identity_frame.method.str.startswith('f')]
            for (method, chain), group in df.groupby(['method', 'chain']):
                sequence = r' $\to$ '.join(aliases[m] for m in chain.split(' -> '))
                rows.append([NAMES[dataset], f'$F_{method[1:]}$', sequence, str(len(group))])
        table(PAPER/'tables/table_exact_chain_identities.tex', ['Dataset', 'Class', 'Sequence', 'Splits'], rows)
        pd.DataFrame(timings).to_csv(OUT/'fixed_search_timings.csv', index=False)
        rows = []
        for dataset in DATASETS:
            for d in [3, 4, 5]:
                a = summary[(summary.dataset==dataset)&(summary.method==f'a{d}')&(summary.reference=='d2')&(summary.band=='all')].iloc[0]
                f = summary[(summary.dataset==dataset)&(summary.method==f'f{d}')&(summary.reference=='d2')&(summary.band=='all')].iloc[0]
                rows.append([NAMES[dataset] if d==3 else '', str(d), f'{f.accuracy_pp:+.3f}',
                             f'{a.accuracy_pp:+.3f}', f'{f.cost_per_1000:+.5f}', f'{a.cost_per_1000:+.5f}'])
        table(PAPER/'tables/table_exact_chain_sensitivity.tex', ['Dataset', 'Length', r'Coverage (\%)', r'Spacing (\%)',
              r'Coverage \$/1,000', r'Spacing \$/1,000'], rows)


def figures(frame, fixed):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6), layout='constrained')
    for col, dataset in enumerate(DATASETS):
        f = frame[frame.dataset==dataset]
        for row, metric in enumerate(['cal_accuracy', 'accuracy']):
            values = []
            for d in [2, 3, 4, 5]:
                values.append([integrate(100*(g[f'd{d}_{metric}']-g[f'd2_{metric}']).to_numpy(), g.fraction.to_numpy()) for _, g in f.groupby('split')])
            values = np.array(values)
            ax = axes[row, col]
            ax.axhline(0, color='#999999', lw=.7)
            ax.plot([2, 3, 4, 5], values.mean(axis=1), '-o', color='#147878', ms=3)
            lo, hi = np.quantile(values, [.1, .9], axis=1)
            ax.fill_between([2, 3, 4, 5], lo, hi, color='#147878', alpha=.18)
            ax.set_xticks([2, 3, 4, 5])
            if row==0: ax.set_title(NAMES[dataset])
            if row==1: ax.set_xlabel('Maximum allowed\nmodels')
            if col==0: ax.set_ylabel(('Calibration' if row==0 else 'Test')+' accuracy gain (pp)')
    fig.savefig(PAPER/'figures/fig_exact_depth.pdf')
    fig.savefig(PAPER/'figures/fig_exact_depth.png', dpi=180)
    plt.close(fig)
    # The main observed/synthetic comparison and scorer appendix share one renderer.
    from frontier_figure import main as render_frontiers
    render_frontiers()
    x = np.linspace(0, 1, 500)
    if fixed:
        fig, axes = plt.subplots(2, 5, figsize=(10, 4.8), layout='constrained')
        for col, dataset in enumerate(DATASETS):
            f = frame[frame.dataset==dataset]
            for row, metric in enumerate(['accuracy', 'cost']):
                for d in [3, 4, 5]:
                    a = np.stack([(g[f'f{d}_{metric}']-g[f'd2_{metric}']).to_numpy()*(100 if row==0 else 1000) for _, g in f.groupby('split')])
                    ax = axes[row, col]
                    ax.plot(x, a.mean(axis=0), color=COLORS[d-1], label=f'$F_{d}-S_2$', lw=1)
                    lo, hi = np.quantile(a, [.1, .9], axis=0)
                    ax.fill_between(x, lo, hi, color=COLORS[d-1], alpha=.12)
                axes[row, col].axhline(0, color='#999999', lw=.6)
            axes[0, col].set_title(NAMES[dataset])
            axes[1, col].set_xlabel('Assigned budget\nfraction')
        axes[0, 0].set_ylabel('Test-set accuracy\ndifference (percentage points)')
        axes[1, 0].set_ylabel('Realized cost difference\n($/1,000 queries)')
        axes[0, 0].legend(fontsize=9)
        fig.savefig(PAPER/'figures/fig_exact_fixed.pdf')
        plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frame, provenance, fixed = load()
    paired, usage, summary = summarize(frame, fixed)
    make_tables(frame, summary, usage, fixed)
    supplementary_outputs(frame, paired, summary, usage, fixed)
    figures(frame, fixed)
    coverage = dict(report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), splits=250, budget_positions=125000, fixed_complete=fixed,
        weighting='Trapezoidal normalized budget integration, then equal split weights',
        thirds=[0, 1/3, 2/3, 1], source_map=file_record(ROOT/'manuscript_sources.py'), provenance=provenance)
    (OUT/'provenance.json').write_text(json.dumps(coverage, indent=2)+'\n')
    lines = ['# Exact test-set depth report', '', coverage['weighting'], '',
             '| Dataset | S3 minus S2 pp | S4 minus S2 pp | S5 minus S2 pp |', '|---|---:|---:|---:|']
    for dataset in DATASETS:
        r = summary[(summary.dataset==dataset)&(summary.reference=='d2')&(summary.band=='all')].set_index('method')
        lines.append('| '+NAMES[dataset]+' | '+' | '.join(f'{r.loc[f"d{d}", "accuracy_pp"]:+.4f}' for d in [3, 4, 5])+' |')
    lines += ['', f'Fixed-chain completion: {fixed}.', 'Split percentiles measure sensitivity, not confidence intervals.',
              'Quality-target metrics are omitted because the saved budget-capped selections do not certify target-cost optima.']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
