"""Load complete Diff-01 split outputs for manuscript table reproduction."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARMS = {
    'Diff-01 UQ ridge': ('next_model_diff01_s4_full', 'diff01_ridge'),
    'Diff-01 emb. ridge': ('next_model_embedding_diff01_s4_full', 'embedding_diff01_ridge'),
}


def frames(label, dataset, depth=4):
    folder, arm = ARMS[label]
    if depth == 5:
        folder = ('next_model_diff01_s5_full' if arm == 'diff01_ridge'
                  else 'next_model_embedding_diff01_s5_full')
    paths = [ROOT/'results'/folder/dataset/arm/f'split_{s:02d}.csv' for s in range(50)]
    output = []
    for split, path in enumerate(paths):
        f = pd.read_csv(path)
        assert len(f) == 500 and (f.split == split).all()
        np.testing.assert_allclose(f.fraction, np.linspace(0, 1, 500))
        assert np.isfinite(f[[f's{d}_accuracy' for d in range(1, depth+1)]]).all().all()
        output.append(f)
    return output


def gains(label, dataset, depth, baseline):
    fs = frames(label, dataset, max(4, depth))
    return np.array([100*np.trapz(f[f's{depth}_accuracy']-f[f's{baseline}_accuracy'], f.fraction) for f in fs])


def update_depth_table(datasets):
    path = ROOT/'iclr/tables/table_scorer_depth.tex'
    lines = [line for line in path.read_text().splitlines() if not line.startswith('Diff-01')]
    additions = [[], []]
    for label in ARMS:
        accuracy, cost = [], []
        for ds in datasets:
            v = gains(label, ds, 3, 2)
            lo, hi = np.quantile(v, [.1, .9])
            accuracy.append(r'\shortstack{$' + f'{v.mean():+.3f}' + r'$ \\ $[' + f'{lo:+.3f}, {hi:+.3f}' + r']$}')
            fs = frames(label, ds)
            c = np.mean([1000*np.trapz(f.s3_cost-f.s2_cost, f.fraction) for f in fs])
            cost.append(f'${c:+.5f}$')
        additions[0].append(label + ' & ' + ' & '.join(accuracy) + r' \\')
        additions[1].append(label + ' & ' + ' & '.join(cost) + r' \\')
    index = 0
    output = []
    for line in lines:
        output.append(line)
        if line.startswith(r'Resp.\ LR'):
            output.extend(additions[index])
            index += 1
    assert index == 2
    path.write_text('\n'.join(output)+'\n')
