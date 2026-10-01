# Input data and provenance limits

The numerical analysis starts from recorded, graded model responses. This folder
contains code only. To reproduce the reported numbers, supply the original query
rows, generations, log probabilities, correctness labels, and response embeddings.
New generations can produce different results even with identical prompts.

## Required response files

Provide `data/output_data/{dataset}-{model}.parquet` for all 40 combinations.

- Datasets: `mmlu`, `triviaqa`, `math_hard`, `simpleqa`, `livecodebench`.
- Models: `llama-3.1-8b`, `qwen2.5-7b`, `gpt-4o-mini`, `gpt-oss-20b`,
  `deepseek-v3`, `llama-3.3-70b`, `gpt-4o`, `MiniMax-M2.7`.
- Required columns include `prompt`, `response`, `correct`, `logprob`,
  `sequence_probability`, `min_probability`, `min_token_negentropy`,
  `mean_token_negentropy`, and `probability_margin`.
- `logprob` must retain the sequence representation expected by the copied cost
  code. Output token count is its length. Input tokens use `cl100k_base`.
- Preserve row order and prompt alignment across models. Do not shuffle or
  independently drop rows before loading. Missing values are filtered jointly by
  the experiment-specific loader.
- Load the first 2,000 rows for the four non-code benchmarks and all 1,055
  LiveCodeBench rows. The main complete-case sizes are 1,999, 2,000, 1,971,
  1,891, and 1,055 respectively.

The LiveCodeBench source tree also expects its eight response files under
`experiments/livecodebench_8model_20260912/data/output_data/`.
`import_data.py` creates that layout with regular copies rather than links to
an author's machine. It checks file presence before copying, not scientific
contents. The original loaders and run fingerprints validate contents.

## Response embeddings

Seven-scorer analyses require
`data/response_embeddings/{dataset}-{model}.parquet`, with an `embedding` column
holding aligned vectors. Original files are preferable for exact reproduction.
Use the eight-model rerun's LiveCodeBench embeddings, including GPT-oss-20B,
and place copies under that rerun's `data/response_embeddings/` as well.

If missing, `prepare_embeddings.py` calls the copied
`response_scorer_compute.ensure_response_embeddings` implementation using
`sentence-transformers/all-MiniLM-L6-v2`, then copies LiveCodeBench embeddings to
the second source tree. Install `requirements-generation.txt` for this step.
It may download public encoder weights. Hardware or library differences can
change embedding values and fitted classifiers. Existing scorer manifests must
not be mixed with newly generated embeddings.

## Generation and grading source

The copied loaders, prompt formatter, model configuration, and per-dataset
graders document the original pipeline. The generation notebook is retained
under `generation_reference/` with outputs and execution metadata removed.
It is reference material, not a clean unattended entry point: it imports
`utils.get_datasets` and `utils.get_models`, and that module is absent from the
available repository. Do not treat it as a complete fresh-generation pipeline.
The prompt formatter also contains datasets outside this paper.

`grade_livecodebench.py` was available only in the older
`llm-cascade-frontiers/src` source bundle. It executes **public tests only**.
Its inclusion documents the available grader, but does not establish that
rerunning it recreates the paper's saved LiveCodeBench labels or a full official
benchmark evaluation. Use the recorded labels for reproducing cascade results.
Run untrusted generated solutions only in an appropriate isolated environment.

SimpleQA grading and model generation require provider credentials and incur
API charges. The reproduction wrapper performs neither. No `.env` file,
credential, notebook output, or paid-generation command is included in its run
plan. Prompt files and original benchmark metadata are additionally required
when regenerating or regrading answers.

## Preserved protocol details and discrepancies

The main analyses use all eight models, seeds 42 through 91, 50 stratified
calibration/test splits, and 500 assigned budgets. Thresholds and model sequences
are frozen before test evaluation. Compare assigned budgets, retaining realized
cost overshoots. Split percentiles are sensitivity summaries, not confidence
intervals. Synthetic-score construction uses full-sample labels.

The copied synthetic-price runner explicitly uses the `setting-specific`
protocol. It recomputes calibration cost order and budget anchors after applying
price multipliers. Comparisons match normalized budget positions, with setting-specific dollar
budgets and calibration ordering, as described in the synthetic-cost appendix.
Unchanged full-sample endpoint costs do not guarantee identical calibration
dollar budgets. The script writes invariant diagnostics
and offers a `strict` mode that rejects violations. The wrapper preserves the
executed setting-specific protocol rather than silently changing the experiment.

The older concavity, benefit-AUROC, and benefit-cost diagnostics retain their
original pair-validity and representative-pair conventions. These are distinct
from the unrestricted eight-model exact-depth search. The copied legacy helper
modules may refer to earlier figure numbering in their docstrings.

Exact source-hash matching is intended for fresh, internally consistent runs.
Do not copy historical results into the new output tree and bypass a hash
mismatch. `test_manuscript_sources_with_historical_data.py` is an archival check
requiring both current eight-model and explicitly historical seven-model caches.
It is excluded from the data-free test command.
