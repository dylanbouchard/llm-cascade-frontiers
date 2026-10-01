"""Verify saved exact-search input arrays against current response data."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/exact-replacement-mpl')
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from exact_heldout_depth import load_full, SCORER, DATASETS

ROOT = Path(__file__).resolve().parent


def audit():
    records = []
    for dataset in DATASETS:
        raw, costs = load_full(dataset)
        prompts = next(iter(raw.values()))['prompt'].values
        valid = np.ones(len(prompts), dtype=bool)
        for model, frame in raw.items():
            assert np.array_equal(prompts, frame['prompt'].values)
            valid &= np.isfinite(frame[SCORER]) & np.isfinite(frame['correct']) & np.isfinite(costs[model])
        rows = np.flatnonzero(valid)
        digest = hashlib.sha256()
        for model, frame in raw.items():
            filtered = frame.iloc[rows].reset_index(drop=True)
            digest.update(model.encode())
            digest.update(pd.util.hash_pandas_object(filtered[['prompt', SCORER, 'correct']], index=False).values.tobytes())
            digest.update(np.asarray(costs[model])[rows].tobytes())
        with np.load(ROOT/'results/exact_heldout_five/inputs'/f'{dataset}.npz') as cached:
            assert str(cached['fingerprint']) == digest.hexdigest()
            assert list(cached['models']) == list(raw)
            for i, model in enumerate(raw):
                values = dict(scores=raw[model][SCORER].to_numpy(), correct=raw[model]['correct'].to_numpy(),
                              costs=np.asarray(costs[model]))
                for key, value in values.items():
                    np.testing.assert_equal(cached[key][i], value)
        record = dict(dataset=dataset, jointly_valid_rows=len(rows), data_sha256=digest.hexdigest(), status='passed')
        records.append(record)
        print(record, flush=True)
    dest = ROOT/'iclr/results/exact_depth/input_audit.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    audit()
