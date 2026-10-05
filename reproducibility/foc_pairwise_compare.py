"""FOC candidates versus exhaustive S2 on the current paper's saved splits.

All selection is calibration-only. No manuscript writes or model calls.
"""
import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATASETS = ['mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench']
METHODS = ['exact', 'foc', 'boundary']


def source(dataset):
    base = ROOT
    if dataset == 'livecodebench':
        base /= 'experiments/livecodebench_8model_20260912'
    return base / 'results/exact_heldout_five'


def stationary_candidates(ratio):
    """Bracket positive-to-negative crossings, including flat zero maxima.

    Each index is a tie-preserving empirical threshold partition. Both sides
    of a crossing are retained because the empirical objective is discrete.
    """
    r = np.asarray(ratio)
    cross = ((r[:-1] > 0) & (r[1:] <= 0)) | ((r[:-1] >= 0) & (r[1:] < 0))
    ix = np.flatnonzero(cross)
    return np.unique(np.r_[0, len(r)-1, ix, ix+1]).astype(int)


def pair_surface(scores, cost_a, cost_b, correct_a, correct_b, window):
    order = np.argsort(scores, kind='stable')
    ss = scores[order]
    unique, counts = np.unique(ss, return_counts=True)
    ends = np.r_[0, counts.cumsum()]
    gain = (correct_b-correct_a)[order]
    extra = cost_b[order]
    g = np.r_[0., np.cumsum(gain)]
    c = np.r_[0., np.cumsum(extra)]
    n = len(scores)
    costs = (cost_a.sum()+c[ends])/n
    correct = np.rint(correct_a.sum()+g[ends]).astype(int)
    cuts = np.r_[-np.inf, unique[1:], np.inf]
    ratio = None
    if window:
        width = min(window, n)
        lo = np.clip(ends-width//2, 0, n-width)
        hi = lo+width
        # One-sided windows at endpoints. Ratio uses local benefit / local
        # escalation cost, without assuming cost-score independence.
        den = c[hi]-c[lo]
        ratio = np.divide(g[hi]-g[lo], den, out=np.zeros_like(den), where=den>0)
    return costs, correct, cuts, ratio


def best_static(cost, correct, candidates, budgets):
    ix = np.sort(np.unique(candidates))
    q = correct[ix]
    improved = q > np.r_[-np.inf, np.maximum.accumulate(q)[:-1]]
    winners = np.maximum.accumulate(np.where(improved, np.arange(len(ix)), 0))
    feasible = np.searchsorted(cost[ix], budgets+1e-12, side='right')-1
    return ix[winners[np.maximum(feasible, 0)]], feasible >= 0


def select(data, budgets, method, window=100):
    t0 = perf_counter()
    models = sorted(data, key=lambda m: (data[m]['costs'].mean(), m))
    seqs = [(m,) for m in models]+list(combinations(models, 2))
    n = len(data[models[0]]['scores'])
    best_q = np.full(len(budgets), -1, int)
    best_c = np.full(len(budgets), np.inf)
    best_seq = np.full(len(budgets), -1, int)
    best_tau = np.full(len(budgets), -np.inf)
    stats = dict(partitions=0, stationary_candidates=0, candidate_budget_memberships=0)
    for si, seq in enumerate(seqs):
        a = data[seq[0]]
        if len(seq)==1:
            cost = np.array([a['costs'].mean()])
            q = np.array([int(a['correct'].sum())])
            cuts = np.array([-np.inf])
            ranks = np.zeros(len(budgets), int)
            valid = cost[0] <= budgets+1e-12
            stats['candidate_budget_memberships'] += int(valid.sum())
        else:
            b = data[seq[1]]
            cost, q, cuts, ratio = pair_surface(a['scores'], a['costs'], b['costs'], a['correct'], b['correct'], window if method=='foc' else 0)
            stats['partitions'] += len(cost)
            if method=='exact':
                candidates = np.arange(len(cost))
            elif method=='foc':
                candidates = stationary_candidates(ratio)
                stats['stationary_candidates'] += len(candidates)
            else:
                candidates = np.array([0, len(cost)-1])
            ranks, valid = best_static(cost, q, candidates, budgets)
            stats['candidate_budget_memberships'] += int(np.searchsorted(cost[candidates], budgets+1e-12, side='right').sum())
            if method!='exact':
                upper = np.clip(np.searchsorted(cost, budgets+1e-12, side='right')-1, 0, len(cost)-1)
                # The budget boundary is the active-constraint KKT candidate.
                better = (q[upper]>q[ranks]) | ((q[upper]==q[ranks]) & (cost[upper]<cost[ranks]))
                ranks = np.where(better, upper, ranks)
                stats['candidate_budget_memberships'] += int(valid.sum())
        qc, cc = q[ranks], cost[ranks]
        replace = valid & ((qc>best_q) | ((qc==best_q) & (cc<best_c)))
        best_q[replace], best_c[replace] = qc[replace], cc[replace]
        best_seq[replace], best_tau[replace] = si, cuts[ranks[replace]]
    assert (best_seq>=0).all()
    stats['seconds'] = perf_counter()-t0
    return dict(seq=best_seq, tau=best_tau, cal_cost=best_c, cal_accuracy=best_q/n), seqs, stats


def evaluate(selected, seqs, data):
    memo = {}
    values = []
    for si, tau in zip(selected['seq'], selected['tau']):
        key = (int(si), float(tau))
        if key not in memo:
            seq = seqs[si]
            a = data[seq[0]]
            c, q = a['costs'], a['correct']
            if len(seq)==2:
                b = data[seq[1]]
                esc = a['scores'] < tau
                c, q = c+esc*b['costs'], np.where(esc, b['correct'], q)
            memo[key] = (c.mean(), q.mean())
        values.append(memo[key])
    return np.array(values).T


def sliced(arrays, idx):
    return {str(m): {key: arrays[key][j, idx] for key in ['scores','costs','correct']}
            for j,m in enumerate(arrays['models'])}


def run(args):
    out = ROOT/args.out
    out.mkdir(parents=True, exist_ok=True)
    summaries, timings = [], []
    for dataset in args.datasets:
        src = source(dataset)
        input_path = src/'inputs'/f'{dataset}.npz'
        with np.load(input_path) as z:
            arrays = {k:z[k] for k in z.files}
        assert len(arrays['models'])==8
        input_hash = hashlib.sha256(input_path.read_bytes()).hexdigest()
        for split in range(args.splits):
            path = src/dataset/f'split_{split:02d}.json'
            manifest = json.loads(path.read_text())
            assert str(arrays['fingerprint'])==manifest['data_sha256']
            assert not set(manifest['cal_idx']) & set(manifest['test_idx'])
            budgets, fractions = np.array(manifest['budgets']), np.array(manifest['fractions'])
            cal = sliced(arrays, manifest['cal_idx'])
            selected, stats = {}, {}
            # Rotate run order to reduce systematic warm-cache timing bias.
            for method in METHODS[split%3:]+METHODS[:split%3]:
                selected[method], seqs, stats[method] = select(cal, budgets, method, args.window)
                check_c, check_q = evaluate(selected[method], seqs, cal)
                np.testing.assert_allclose(check_c, selected[method]['cal_cost'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(check_q, selected[method]['cal_accuracy'], atol=1e-12, rtol=0)
                assert (check_c<=budgets+1e-12).all()
            baseline = manifest['frozen']['2']
            np.testing.assert_allclose(selected['exact']['cal_accuracy'], [p['cal_quality'] for p in baseline], atol=1e-12, rtol=0)
            np.testing.assert_allclose(selected['exact']['cal_cost'], [p['cal_cost'] for p in baseline], atol=1e-12, rtol=0)
            for method in ['foc','boundary']:
                assert (selected[method]['cal_accuracy']<=selected['exact']['cal_accuracy']+1e-12).all()
            assert (selected['foc']['cal_accuracy']>=selected['boundary']['cal_accuracy']-1e-12).all()
            dest = out/dataset
            dest.mkdir(exist_ok=True)
            # Persist chosen policies before test data are passed to evaluation.
            np.savez_compressed(dest/f'split_{split:02d}_policies.npz', **{f'{m}_{k}':v for m in METHODS for k,v in selected[m].items()})
            meta = dict(dataset=dataset, split=split, source=str(path.relative_to(ROOT)), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), input_sha256=input_hash, data_sha256=manifest['data_sha256'], cal_idx=manifest['cal_idx'], test_idx=manifest['test_idx'], window=args.window, sequences=seqs, stats=stats, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
            (dest/f'split_{split:02d}.json').write_text(json.dumps(meta, indent=2)+'\n')
            test = sliced(arrays, manifest['test_idx'])
            frame = pd.DataFrame(dict(fraction=fractions, budget=budgets))
            for method in METHODS:
                tc,tq = evaluate(selected[method], seqs, test)
                frame[f'{method}_test_cost'],frame[f'{method}_test_accuracy'] = tc,tq
                for k in ['cal_cost','cal_accuracy']: frame[f'{method}_{k}'] = selected[method][k]
                timings.append(dict(dataset=dataset, split=split, method=method, **stats[method]))
            # Confirm the rerun also reproduces the cached exhaustive test result.
            saved = pd.read_csv(path.with_suffix('.csv'))
            np.testing.assert_allclose(frame.exact_test_accuracy, saved.d2_accuracy, atol=1e-12, rtol=0)
            np.testing.assert_allclose(frame.exact_test_cost, saved.d2_cost, atol=1e-12, rtol=0)
            frame.to_csv(dest/f'split_{split:02d}.csv', index=False)
            trap = getattr(np,'trapezoid',np.trapz)
            for method in ['foc','boundary']:
                row = dict(dataset=dataset, split=split, method=method)
                for suffix,scale in [('test_accuracy',100),('test_cost',1000),('cal_accuracy',100),('cal_cost',1000)]:
                    row[suffix+'_difference'] = float(trap((frame[f'{method}_{suffix}']-frame[f'exact_{suffix}'])*scale, fractions))
                row['cal_optimal_fraction'] = float(trap(np.isclose(frame[f'{method}_cal_accuracy'],frame.exact_cal_accuracy,rtol=0,atol=1e-12).astype(float), fractions))
                row['max_cal_gap_pp'] = float(100*(frame.exact_cal_accuracy-frame[f'{method}_cal_accuracy']).max())
                summaries.append(row)
            if split%10==0 or split==args.splits-1: print(dataset, split+1, '/', args.splits, flush=True)
    df=pd.DataFrame(summaries);df.to_csv(out/'split_summary.csv',index=False)
    summary=df.groupby(['dataset','method']).agg({c:'mean' for c in df.columns if c not in ['dataset','method','split']})
    for label,quant in [('p10',.1),('p90',.9)]: summary['test_accuracy_'+label]=df.groupby(['dataset','method']).test_accuracy_difference.quantile(quant)
    summary.to_csv(out/'summary.csv')
    timing=pd.DataFrame(timings);timing.to_csv(out/'timings.csv',index=False)
    timing.groupby(['dataset','method']).mean(numeric_only=True).to_csv(out/'timing_summary.csv')
    print(summary.to_string(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',default=DATASETS,choices=DATASETS)
    p.add_argument('--splits',type=int,default=50)
    p.add_argument('--window',type=int,default=100)
    p.add_argument('--out',default='results/foc_pairwise')
    args=p.parse_args()
    if args.window<2 or not 1<=args.splits<=50: p.error('window >= 2 and 1 <= splits <= 50 required')
    run(args)
