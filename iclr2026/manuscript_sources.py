"""Explicit result sources for the current eight-model ICLR manuscript."""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
DATASETS = ('mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench')
EXPECTED_MODELS = frozenset(('llama-3.1-8b', 'gpt-oss-20b', 'qwen2.5-7b',
    'gpt-4o-mini', 'MiniMax-M2.7', 'llama-3.3-70b', 'deepseek-v3', 'gpt-4o'))


def source_root(dataset):
    if dataset not in DATASETS:
        raise ValueError(f'Unknown manuscript dataset: {dataset}')
    return ROOT / 'experiments/livecodebench_8model_20260912' if dataset == 'livecodebench' else ROOT


def result_path(dataset, analysis):
    return source_root(dataset) / 'results' / analysis


def file_record(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def validate_exact(manifest, arrays, dataset, split):
    """Reject stale pools or mismatched manifests before reporting."""
    if set(manifest['pool']) != EXPECTED_MODELS or set(map(str, arrays['models'])) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: manuscript requires the eight-model pool')
    if manifest['dataset'] != dataset or manifest['seed'] != 42 + split:
        raise ValueError(f'{dataset}: wrong split identity')
    if str(arrays['fingerprint']) != manifest['data_sha256']:
        raise ValueError(f'{dataset}: input fingerprint mismatch')
    if set(manifest['cal_idx']) & set(manifest['test_idx']):
        raise ValueError(f'{dataset}: overlapping calibration and test rows')
    np.testing.assert_allclose(manifest['fractions'], np.linspace(0, 1, 500), rtol=0, atol=1e-14)


def validate_learning(manifest, exact, dataset, split):
    if set(manifest['pool']) != EXPECTED_MODELS or set(exact['pool']) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: learning curve requires the eight-model pool')
    if manifest['dataset'] != dataset or manifest['seed'] != 42 + split:
        raise ValueError(f'{dataset}: wrong learning-curve split')
    if manifest['data_sha256'] != exact['data_sha256']:
        raise ValueError(f'{dataset}: learning/exact fingerprint mismatch')
    if set(manifest['reservoir_idx']) != set(exact['cal_idx']) or manifest['test_idx'] != exact['test_idx']:
        raise ValueError(f'{dataset}: learning/exact query identities differ')
    np.testing.assert_allclose(manifest['budgets'], exact['budgets'], rtol=0, atol=1e-15)
