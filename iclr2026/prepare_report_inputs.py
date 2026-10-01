"""Stage eight-model LiveCodeBench outputs for the older scorer-table reporter.

The copied scorer_depth_report predates manuscript_sources and expects every
dataset under one results directory. Copy its two required dataset directories
from the canonical eight-model run. Reject conflicting files on repeated runs.
"""
from pathlib import Path
import hashlib
import shutil

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'experiments/livecodebench_8model_20260912/results'


def digest(path):
    return hashlib.sha256(path.read_bytes()).digest()


def main():
    for analysis in ['scorer_depth', 'exact_heldout_depth']:
        src = SOURCE/analysis/'livecodebench'
        if not src.is_dir():
            raise FileNotFoundError(src)
        for path in sorted(src.rglob('*')):
            if not path.is_file() or path.suffix not in ('.csv', '.json'):
                continue
            dest = ROOT/'results'/analysis/'livecodebench'/path.relative_to(src)
            if dest.exists():
                if digest(path) != digest(dest):
                    raise ValueError(f'Conflicting report input: {dest}')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)

if __name__ == '__main__':
    main()
