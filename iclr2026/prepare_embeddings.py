"""Build missing response embeddings using the copied source implementation."""
from pathlib import Path
import shutil
from fig2_compute import load_full
from response_scorer_compute import ensure_response_embeddings, embedding_path
from manuscript_sources import DATASETS, EXPECTED_MODELS, source_root

for dataset in DATASETS:
    raw, _ = load_full(dataset)
    if set(raw) != EXPECTED_MODELS:
        raise ValueError(f'{dataset}: all eight response files are required')
    ensure_response_embeddings(dataset, raw)
    if dataset == 'livecodebench':
        for model in raw:
            dest = source_root(dataset) / embedding_path(dataset, model)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copyfile(embedding_path(dataset, model), dest)
