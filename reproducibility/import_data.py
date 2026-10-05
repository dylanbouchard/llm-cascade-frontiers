"""Copy required recorded inputs from an existing experiment repository."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
DATASETS = ['mmlu', 'triviaqa', 'math_hard', 'simpleqa', 'livecodebench']
MODELS = ['llama-3.1-8b', 'qwen2.5-7b', 'gpt-4o-mini', 'gpt-oss-20b',
          'deepseek-v3', 'llama-3.3-70b', 'gpt-4o', 'MiniMax-M2.7']
LCB = Path('experiments/livecodebench_8model_20260912')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path)
    p.add_argument('--copy', action='store_true', help='Without this flag, only check inputs.')
    a = p.parse_args()
    source = a.source.resolve()
    if source == ROOT:
        p.error('source must be a separate data repository')
    jobs = []
    for dataset in DATASETS:
        for model in MODELS:
            rel = Path('data/output_data') / f'{dataset}-{model}.parquet'
            jobs.append((source / rel, ROOT / rel, True))
            emb = Path('data/response_embeddings') / rel.name
            esrc = source / emb
            if dataset == 'livecodebench' and (source / LCB / emb).exists():
                esrc = source / LCB / emb
            jobs.append((esrc, ROOT / emb, False))
            if dataset == 'livecodebench':
                jobs.append((source / rel, ROOT / LCB / rel, True))
                jobs.append((esrc, ROOT / LCB / emb, False))
    missing = sorted({str(s.relative_to(source)) for s, _, required in jobs if required and not s.is_file()})
    if missing:
        p.error('Missing response files:\n' + '\n'.join(missing))
    print('All 40 required response files found. Cached embeddings are optional.')
    if not a.copy:
        return
    for src, dst, required in jobs:
        if not src.is_file():
            continue
        if dst.exists():
            raise FileExistsError(f'Refusing to overwrite {dst}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print('Copied response files and available embeddings. No result caches were imported.')

if __name__ == '__main__':
    main()
