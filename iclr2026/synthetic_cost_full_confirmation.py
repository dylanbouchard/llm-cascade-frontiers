"""Extend the prepared cost experiment to 50 test splits for every price setting.

Uses the original run_cell unchanged and preserves all existing source identities.
Run after synthetic_cost_depth.py has prepared the inputs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from synthetic_cost_depth import DATASETS, SETTINGS, SOURCES, ROOT, sha, run_cell, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('results/synthetic_cost_depth'))
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    sources = {p: sha(ROOT/p) for p in SOURCES}
    for dataset in DATASETS:
        meta = json.loads((args.out/dataset/'manifest.json').read_text())
        if meta['sources'] != sources or meta['splits'] != 50:
            raise ValueError(f'{dataset}: expected unchanged sources and 50 prepared splits')
    write_json(args.out/'full_confirmation_manifest.json', dict(
        datasets=list(DATASETS), settings=list(SETTINGS), splits=50,
        runner_sha256=sha(Path(__file__)), sources=sources,
        protocol='setting-specific',
        preparation_sha256={d: sha(args.out/d/'manifest.json') for d in DATASETS}))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(run_cell, d, s, f'split_{i:02d}', args.out, 'setting-specific')
                for d in DATASETS for s in SETTINGS for i in range(50)]
        for job in jobs:
            job.result()


if __name__ == '__main__':
    main()
