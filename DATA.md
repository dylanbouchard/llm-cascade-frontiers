# Data Notes

This repository includes formatted prompts/source records and cached numerical
results, but not the large raw model-response cache.

## Included Data

- `data/prompts/*.parquet`: formatted prompts or source records for MMLU,
  TriviaQA, MATH levels 3-5, SimpleQA, and LiveCodeBench reconstruction.
- `data/prompts/livecodebench_indices.json`: the 1055 LiveCodeBench
  `question_id`s used in the paper.
- `data/*_embeddings.parquet`: frozen sentence-transformer embeddings used by
  the diagnostic router.
- `results/`: cached numerical outputs used to regenerate figures and tables.

## Omitted Data

The raw generated model-response files are omitted:

```text
data/output_data/{dataset}-{model}.parquet
```

These files are large and API-dependent. They can be regenerated with
`generate_whitebox.py` if provider credentials are available.

## Dataset Sources

| Dataset | Source | Notes |
|---|---|---|
| MATH levels 3-5 | Hendrycks et al. (2021) | Formatted prompt parquet included |
| MMLU | Hendrycks et al. (2021) | Formatted prompt parquet included |
| TriviaQA | Joshi et al. (2017) | Formatted prompt parquet with answer aliases included |
| SimpleQA | Wei et al. (2024) / OpenAI | Formatted prompt parquet included |
| LiveCodeBench | Jain et al. (2024) | Reconstructed from HuggingFace using included question ids |

Please cite the original datasets when using or extending this repository.

## LiveCodeBench Reconstruction

LiveCodeBench is reconstructed from HuggingFace and filtered to the included
`question_id` manifest. To regenerate all formatted prompt files, including
`data/prompts/livecodebench.parquet`, run:

```bash
python3 format_prompts.py
```

Depending on upstream dataset settings, LiveCodeBench reconstruction may
require HuggingFace authentication.

## Provider Costs

Per-query costs are computed from realized input and output token counts using
the token prices encoded in the experiment scripts. The released results use the
prices in effect at the time the experiments were run.
