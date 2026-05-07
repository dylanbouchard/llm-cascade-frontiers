# Reproducibility Guide

This document describes how to reproduce the figures and cached numerical
artifacts for the paper.

## Environment

Install dependencies from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts assume they are run from `src/`. Cached results and figures
use relative paths such as `results/`, `figures/`, and `data/prompts/`
inside that directory.

## Cached-Result Reproduction

The fastest reproduction path uses the cached numerical arrays in `src/results/`.
This does not require model API access.

```bash
cd src
python3 figures.py
```

This regenerates the main held-out frontier comparison, the representative
escalation-benefit figure, and the appendix escalation-benefit panels. The
rendered figures are written to `src/figures/`.

Individual figure targets:

```bash
cd src
python3 figures.py fig2
python3 figures.py fig3
python3 figures.py figA5
```

## Mapping from Scripts to Artifacts

| Script | Purpose | Requires raw responses? |
|---|---|---|
| `src/figures.py` | Render main and appendix figures from cached results | No |
| `src/fig2_compute.py` | Held-out pairwise envelope, full fixed chain, optimized subsequence, and single-model baseline curves | Yes |
| `src/router_compute.py` | Diagnostic frozen-embedding router and same-signal comparisons | Yes |
| `src/voi_compute.py` | Scorer-choice ablation and benefit-AUROC diagnostics | Yes |
| `src/cal_sensitivity.py` | Calibration-size sensitivity | Yes |
| `src/grid_sensitivity.py` | Threshold-grid sensitivity | Yes |
| `src/opt_sensitivity.py` | NSGA-II versus random-search sensitivity | Yes |
| `src/foc_verify.py` | Two-model sweep verification | Yes |
| `src/cost_variability.py` | Cost-score correlation diagnostics | Yes |
| `src/escalation_benefit.py` | Escalation-benefit curves for all pairs | Yes |

Scripts that require raw responses expect graded parquet files under
`src/data/output_data/{dataset}-{model}.parquet`.

## Raw Response Generation

Model generation uses UQLM `WhiteBoxUQ` and writes one parquet per
dataset-model pair:

```bash
cd src
python3 generate_whitebox.py --dataset mmlu --model gpt-4o-mini
python3 generate_whitebox.py --dataset all --model all
```

The generation script records the white-box log-probability scorers used in the
paper:

- `sequence_probability`
- `min_probability`
- `min_token_negentropy`
- `mean_token_negentropy`
- `probability_margin`

API-backed generation requires provider credentials for the selected models.
No credentials are included in this repository.

## Grading

After generation, grade response parquets in place:

```bash
cd src
python3 grade_outputs.py --dataset all
```

Dataset-specific graders are included for MMLU, TriviaQA, MATH, SimpleQA, and
LiveCodeBench. SimpleQA grading uses an LLM judge and therefore requires the
configured Gemini credentials. LiveCodeBench grading executes generated code
against test cases and should be run in an appropriately isolated environment.

## Notes on Randomness

The held-out experiments use 50 random 50/50 calibration-test splits with fixed
script-level seeds. Model pools, admissible pairs, thresholds, subsequences, and
router classifiers are selected on calibration data only and evaluated on the
held-out half.

The optimized subsequence search is capped at four models in the main
comparison scripts. The full fixed-chain baseline is uncapped and uses the full
calibration-selected non-dominated pool.

## Expected Limitations

Full end-to-end regeneration can differ slightly if upstream model providers
change model implementations, tokenization behavior, rate limits, or API
availability. The cached numerical results in `src/results/` are the artifacts
used to render the released figures.
