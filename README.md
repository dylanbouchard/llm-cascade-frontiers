# Is Escalation Worth It? Reproducibility Code

This repository contains the public reproducibility code and cached numerical
artifacts for the paper:

**Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades**

The project studies deterministic threshold cascades for large language models.
It compares the pairwise envelope of two-model cascades against full fixed
chains, optimized cost-ordered subsequence cascades, single-model endpoints, and
a diagnostic frozen-embedding router.

## Repository Contents

- `cascade_core.py`, `optuna_frontier.py`: core cascade simulation, cost
  computation, threshold sweeps, and frontier utilities.
- `fig2_compute.py`, `router_compute.py`, `voi_compute.py`,
  `cal_sensitivity.py`, `grid_sensitivity.py`, `opt_sensitivity.py`,
  `foc_verify.py`, `cost_variability.py`, `escalation_benefit.py`: scripts used
  to compute cached results.
- `figures.py`: regenerates the paper figures from cached results.
- `generate_whitebox.py`: regenerates model responses and UQ scores using UQLM
  `WhiteBoxUQ`.
- `grade_outputs.py` and `grade_*.py`: dataset-specific grading scripts.
- `data/prompts/`: formatted public benchmark prompts/source records.
- `data/*_embeddings.parquet`: frozen embeddings used by the diagnostic router.
- `results/`: cached numerical outputs used to regenerate figures and tables.
- `figures/`: rendered figure assets.

Large raw model-response parquets are not included. See
[`DATA.md`](DATA.md) and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for details.

## Quick Start

Create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Regenerate the bundled paper figures from cached results:

```bash
python3 figures.py
```

Individual figure targets are also available:

```bash
python3 figures.py fig2
python3 figures.py fig3
python3 figures.py figA5
```

The cached result files are sufficient for figure regeneration. Recomputing the
frontiers from raw generations requires `data/output_data/*.parquet`, which is
not included because the files are large and API-dependent.

## Full Pipeline

The full response-generation path is:

```bash
python3 format_prompts.py
python3 generate_whitebox.py --dataset mmlu --model gpt-4o-mini
python3 grade_outputs.py --dataset mmlu
```

Use `--dataset all --model all` to regenerate the full response cache. This
requires API credentials for the model providers and can be expensive. SimpleQA
grading uses an LLM judge and also requires provider credentials.

After graded response parquets exist in `data/output_data/`, the cached results
can be recomputed with the corresponding scripts, for example:

```bash
python3 fig2_compute.py
python3 router_compute.py
python3 voi_compute.py
python3 cal_sensitivity.py
python3 grid_sensitivity.py
python3 opt_sensitivity.py
python3 foc_verify.py
python3 cost_variability.py
python3 escalation_benefit.py
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for a more detailed map from
scripts to paper artifacts.

## Data Availability

Formatted prompts and router embeddings are included. Raw model generations are
omitted from the repository. The original benchmark datasets are public, with
the exception that LiveCodeBench is reconstructed from HuggingFace using the
included `question_id` manifest.

See [`DATA.md`](DATA.md) for dataset sources, redistribution notes, and
LiveCodeBench reconstruction instructions.

## License

The code in this repository is released under the Apache License 2.0. Dataset
records remain subject to the licenses and terms of their original sources.
